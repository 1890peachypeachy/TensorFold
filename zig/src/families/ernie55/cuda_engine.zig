//! Multi-slot ERNIE 5.5 Flash Preview CUDA engine for tensorfold-native (TP=1/2/4, RoCE, chunked prefill, CUDA graphs, bundled 3-layer DSpark).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const kv_mod = @import("cuda_kv.zig");
const kernels = @import("cuda_kernels.zig");
const fw = @import("cuda_forward.zig");
const dspark = @import("cuda_dspark.zig");
const sampler = @import("cuda_sampler.zig");

pub const prompt_rows: usize = 4096;
/// Kept prompt states stand on this grid (chunks end early at a mark the cache asked for).
pub const snap_grid: usize = 128;
pub const window_rows: usize = 64;
pub const max_slots: usize = 16;

pub const Seq = struct {
    kv: kv_mod.KvCache,
    draft: ?dspark.Caches = null,
    stash: ?cuda.DeviceBuffer = null, // BF16 [window_rows, 12288] target aux taps from last verify
    first_logits: ?cuda.DeviceBuffer = null, // F32 [vocab_width] last prompt row logits for firstMasked
    mark_ring: ?cuda.DeviceBuffer = null, // draft ring copied at a late mark inside a chunk
    mark_at: usize = 0, // the position `mark_ring` stands at (0: none)
    sampling: ?sampler.Sampling = null,
    verify_graphs: [8]?cuda.graph.Exec = @splat(null),
    verify_warmed: [8]bool = @splat(false),
    sparse_ceil: usize = 0,
    sparse_graphs: [8]?cuda.graph.Exec = @splat(null),
    sparse_warmed: [8]bool = @splat(false),
    draft_graphs: [dspark.block_size]?cuda.graph.Exec = @splat(null),
    draft_warmed: [dspark.block_size]bool = @splat(false),

    fn clearSparseGraphs(self: *Seq) void {
        for (&self.sparse_graphs) |*g| if (g.*) |*ge| {
            ge.deinit();
            g.* = null;
        };
        self.sparse_warmed = @splat(false);
        self.sparse_ceil = 0;
    }

    pub fn deinit(self: *Seq) void {
        self.clearSparseGraphs();
        for (&self.draft_graphs) |*g| if (g.*) |*ge| {
            ge.deinit();
            g.* = null;
        };
        for (&self.verify_graphs) |*g| if (g.*) |*ge| {
            ge.deinit();
            g.* = null;
        };
        if (self.first_logits) |*fl| fl.free();
        if (self.mark_ring) |*mr| mr.free();
        if (self.stash) |*st| st.free();
        if (self.draft) |*dr| dr.deinit();
        self.kv.deinit();
    }
};

/// A kept prompt state: its id (the same on every rank) and the position it stands at.
pub const Kept = struct { id: u64, at: u32 };
/// Where a prompt pass resumes: a kept state's id and position.
pub const Resume = struct { id: u64, at: usize };
const Snap = struct { buf: cuda.DeviceBuffer, at: usize };
/// Kept states stand on the prompt grid; the HCA rings keep this many positions below one.
pub const snap_ring: usize = 128;

/// One stream's window in a shared verify: its slot, its ids (pending token, then drafts), its draws.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };
pub const BlockPart = struct { slot: u32, kept_n: usize, anchor: u32, out: []u32, advance: usize = 0 };
pub const max_parts: usize = 8;

const SharedVerifyKey = struct {
    bucket_rows: u16 = 0,
    any_sparse: bool = false,
    older_ceil: u32 = 0,
};

fn bucketFor(rows: usize) usize {
    const buckets = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64 };
    for (buckets) |b| if (rows <= b) return b;
    return window_rows;
}

const SharedVerifyEntry = struct {
    key: SharedVerifyKey = .{},
    graph: ?cuda.graph.Exec = null,
    warmed: bool = false,
    tick: u64 = 0,

    fn clear(self: *SharedVerifyEntry) void {
        if (self.graph) |*ge| {
            ge.deinit();
            self.graph = null;
        }
        self.warmed = false;
        self.key = .{};
    }
};

const SharedDraftKey = struct {
    np: u8 = 0,
    max_out: u8 = 0,
    slots: [max_parts]u8 = @splat(0),
};

const SharedDraftEntry = struct {
    key: SharedDraftKey = .{},
    graph: ?cuda.graph.Exec = null,
    warmed: bool = false,
    tick: u64 = 0,

    fn clear(self: *SharedDraftEntry) void {
        if (self.graph) |*ge| {
            ge.deinit();
            self.graph = null;
        }
        self.warmed = false;
        self.key.np = 0;
    }
};

/// TF_ERNIE55_GRAPH_STATS=1: every 256 shared verifies, each rank logs how its shared verify and shared draft
/// rounds ran (graph launch, eager, capture) by stream count, with the mean host-timed ms of each (verify and draft
/// both end in a host sync, so the time is the round's GPU time plus its launch overhead). Off by default.
const GraphStats = struct {
    on: ?bool = null,
    n: [2][max_parts + 1][3]u64 = @splat(@splat(@splat(0))), // [verify, draft][streams][launch, eager, capture]
    ns: [2][max_parts + 1][3]u64 = @splat(@splat(@splat(0))),
    verifies: u64 = 0,

    fn enabled(g: *GraphStats) bool {
        if (g.on) |o| return o;
        const o = if (std.c.getenv("TF_ERNIE55_GRAPH_STATS")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        g.on = o;
        return o;
    }

    fn add(g: *GraphStats, io: std.Io, which: usize, kind: usize, t0: std.Io.Timestamp, streams: usize) void {
        const d = t0.durationTo(std.Io.Timestamp.now(io, .awake));
        const s = @min(streams, max_parts);
        g.n[which][s][kind] += 1;
        g.ns[which][s][kind] += if (d.nanoseconds > 0) @intCast(d.nanoseconds) else 0;
        if (which == 0) {
            g.verifies += 1;
            if (g.verifies % 256 == 0) g.print();
        }
    }

    fn print(g: *const GraphStats) void {
        const names = [_][]const u8{ "verify", "draft" };
        for (0..2) |w| for (1..max_parts + 1) |s| {
            const c = g.n[w][s];
            if (c[0] + c[1] + c[2] == 0) continue;
            std.log.info("ernie55 graph stats: {s} streams={d} launch={d} ({d:.2} ms) eager={d} ({d:.2} ms) capture={d} ({d:.2} ms)", .{
                names[w], s, c[0], meanMs(g.ns[w][s][0], c[0]), c[1], meanMs(g.ns[w][s][1], c[1]), c[2], meanMs(g.ns[w][s][2], c[2]),
            });
        };
    }

    fn meanMs(ns: u64, n: u64) f64 {
        return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
    }
};

pub const Options = struct {
    capacity: usize,
    load_dspark: bool = true,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    comm_stream: cuda.Stream,
    ev_shared: cuda.Event,
    ev_comm: cuda.Event,
    lane: cuda.qmmf.Lane,
    k: kernels.Kernels,
    w: weights.Weights,
    group: ?*const cuda.tp_group.Group = null,
    sc_pf: kv_mod.Scratch,
    sc_dec: kv_mod.Scratch,
    smp: sampler.Sampler,
    drafter: ?dspark.Drafter = null,
    seqs: std.ArrayList(?*Seq) = .empty,
    cur: ?*Seq = null,
    slot: u32 = 0,
    capacity: usize,
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty,
    next_snap: u64 = 1,
    dropped: std.ArrayList(u32) = .empty, // ids rank 0 freed, sent to the other ranks with the next step
    shared_verify_graphs: [32]SharedVerifyEntry = @splat(.{}),
    shared_draft_graphs: [16]SharedDraftEntry = @splat(.{}), // 8 in use unless TF_ERNIE55_DRAFT_STAGE=1
    graph_tick: u64 = 0,
    gstats: GraphStats = .{},
    draft_stage_on: ?bool = null,
    draft_stage: ?cuda.DeviceBuffer = null, // max_parts staged DSpark ring sets (TF_ERNIE55_DRAFT_STAGE=1)

    fn draftStageOn(e: *Engine) bool {
        if (e.draft_stage_on) |o| return o;
        const o = if (std.c.getenv("TF_ERNIE55_DRAFT_STAGE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (o) std.log.info("ernie55: shared drafts read staged rings, graphs keyed without slots (TF_ERNIE55_DRAFT_STAGE=1)", .{});
        e.draft_stage_on = o;
        return o;
    }

    fn draftStage(e: *Engine) !*const cuda.DeviceBuffer {
        if (e.draft_stage == null) e.draft_stage = try cuda.DeviceBuffer.alloc(e.d, max_parts * dspark.num_layers * dspark.ring_cap * 512 * 2);
        return &e.draft_stage.?;
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, rank: usize, world: usize, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);

        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        var comm_stream = try cuda.Stream.init(d, true);
        errdefer comm_stream.deinit();
        var ev_shared = try cuda.Event.init(d, false);
        errdefer ev_shared.deinit();
        var ev_comm = try cuda.Event.init(d, false);
        errdefer ev_comm.deinit();

        var lane = try cuda.qmmf.Lane.load(d, 12);
        errdefer lane.unload();
        lane.pdl = true;

        var k = try kernels.Kernels.load(d);
        errdefer k.unload();

        var w = try weights.load(gpa, io, d, dir, rank, world, .{ .load_dspark = o.load_dspark });
        errdefer w.deinit();

        var sc_pf = try kv_mod.Scratch.alloc(d, &w.config, &w.share, prompt_rows, o.capacity);
        errdefer sc_pf.deinit();

        var sc_dec = try kv_mod.Scratch.alloc(d, &w.config, &w.share, window_rows, o.capacity);
        errdefer sc_dec.deinit();

        var smp = try sampler.Sampler.init(d, w.config.vocab, w.share.vocab_width, window_rows);
        errdefer smp.deinit();

        e.* = .{
            .gpa = gpa,
            .io = io,
            .d = d,
            .stream = stream,
            .comm_stream = comm_stream,
            .ev_shared = ev_shared,
            .ev_comm = ev_comm,
            .lane = lane,
            .k = k,
            .w = w,
            .sc_pf = sc_pf,
            .sc_dec = sc_dec,
            .smp = smp,
            .drafter = null,
            .capacity = o.capacity,
        };
        if (e.w.dspark) |*dw| {
            e.drafter = try dspark.Drafter.init(d, dw, e.w.share.vocab_width);
        }
        return e;
    }

    pub fn deinit(e: *Engine) void {
        for (e.seqs.items) |slot_opt| if (slot_opt) |sq| {
            sq.deinit();
            e.gpa.destroy(sq);
        };
        e.seqs.deinit(e.gpa);
        var it = e.snaps.valueIterator();
        while (it.next()) |x| x.buf.free();
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        if (e.drafter) |*dr| dr.deinit();
        if (e.draft_stage) |*b| b.free();
        e.smp.deinit();
        e.sc_dec.deinit();
        e.sc_pf.deinit();
        e.w.deinit();
        e.k.unload();
        e.lane.unload();
        e.ev_comm.deinit();
        e.ev_shared.deinit();
        e.comm_stream.deinit();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    pub fn runner(e: *const Engine) fw.Runner {
        return .{
            .k = &e.k,
            .lane = &e.lane,
            .w = &e.w,
            .group = e.group,
            .comm_stream = e.comm_stream,
            .ev_shared = e.ev_shared,
            .ev_comm = e.ev_comm,
        };
    }

    pub fn streamBytes(e: *const Engine) usize {
        const cap = e.capacity;
        const c = &e.w.config;
        const comp_cap = (cap + c.compress_ratio - 1) / c.compress_ratio + 2;
        var total: usize = 0;
        for (0..e.w.loaded_layers) |li| {
            if (c.isDsa(@intCast(li))) {
                total += cap * (576 + 128) * 2;
            } else {
                total += kv_mod.swa_cap * (512 * 2 + 2 * 512 * 4) + comp_cap * 512 * 2;
            }
        }
        if (e.drafter != null) {
            total += dspark.num_layers * dspark.ring_cap * 512 * 2;
            total += window_rows * 3 * c.hidden * 2;
        }
        total += e.w.share.vocab_width * 4;
        return total;
    }

    pub fn bind(e: *Engine, slot: u32) !void {
        while (e.seqs.items.len <= slot) try e.seqs.append(e.gpa, null);
        if (e.seqs.items[slot] == null) {
            const sq = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(sq);
            var kvc = try kv_mod.KvCache.alloc(e.gpa, e.d, &e.w.config, e.w.loaded_layers, e.capacity);
            errdefer kvc.deinit();
            var dc: ?dspark.Caches = null;
            var stash: ?cuda.DeviceBuffer = null;
            if (e.drafter != null) {
                dc = try dspark.Caches.init(e.d);
                errdefer if (dc) |*x| x.deinit();
                stash = try cuda.DeviceBuffer.alloc(e.d, window_rows * 3 * e.w.config.hidden * 2);
            }
            errdefer if (stash) |*x| x.free();
            const fl = try cuda.DeviceBuffer.alloc(e.d, e.w.share.vocab_width * 4);
            sq.* = .{ .kv = kvc, .draft = dc, .stash = stash, .first_logits = fl };
            e.seqs.items[slot] = sq;
        }
        e.cur = e.seqs.items[slot];
        e.slot = slot;
    }

    fn invalidateSharedSlot(e: *Engine, slot: u32) void {
        for (&e.shared_draft_graphs) |*ent| {
            if (ent.key.np == 0) continue;
            for (0..ent.key.np) |i| if (ent.key.slots[i] == slot) {
                ent.clear();
                break;
            };
        }
    }

    /// Keep up to 2 warm sequence slots pooled across requests so sequential/c2 turns skip
    /// ~150 ms of cuMemAlloc + 4.3 GiB zero-fill and keep their captured CUDA graphs hot.
    const pooled_slots: u32 = 4;

    pub fn release(e: *Engine, slot: u32) void {
        if (slot < e.seqs.items.len) {
            if (e.seqs.items[slot]) |sq| {
                if (slot < pooled_slots) {
                    sq.kv.reset();
                    sq.mark_at = 0;
                    sq.sampling = null;
                } else {
                    e.invalidateSharedSlot(slot);
                    sq.deinit();
                    e.gpa.destroy(sq);
                    e.seqs.items[slot] = null;
                }
            }
            if (e.slot == slot) e.cur = null;
        }
    }

    pub fn pos(e: *const Engine) usize {
        return if (e.cur) |sq| sq.kv.pos else 0;
    }

    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        const sq = e.cur orelse return error.NoSequence;
        sq.kv.reset();
        sq.sampling = sampler.check(sampling);
    }

    /// Run one prompt chunk (`ids.len <= prompt_rows`) starting at `pos0` of a `total`-token prompt.
    /// If `is_last`, saves last row's logits for `firstMasked` and returns the drawn first token.
    pub fn chunk(e: *Engine, ids: []const u32, pos0: usize, total: usize, is_last: bool, resume_id: u64, late_mark: usize) !u32 {
        const sq = e.cur orelse return error.NoSequence;
        const n = ids.len;
        if (n == 0 or n > prompt_rows or pos0 + n > e.capacity) return error.BadChunk;
        if (resume_id != 0) {
            const at = try e.restoreState(resume_id);
            if (at != pos0) return error.SnapshotOutOfStep;
        }
        if (sq.kv.pos != pos0) return error.ChunkOutOfStep;
        sq.mark_at = 0;

        var ids_i32: [prompt_rows]i32 = undefined;
        var pos_i32: [prompt_rows]i32 = undefined;
        for (0..n) |i| {
            ids_i32[i] = @intCast(ids[i]);
            pos_i32[i] = @intCast(pos0 + i);
        }
        try e.sc_pf.buf.upload(e.sc_pf.ids - e.sc_pf.buf.ptr, std.mem.sliceAsBytes(ids_i32[0..n]));
        try e.sc_pf.buf.upload(e.sc_pf.positions - e.sc_pf.buf.ptr, std.mem.sliceAsBytes(pos_i32[0..n]));

        // Every chunk's trailing rows go into the draft ring, so a state kept at any chunk start carries it.
        _ = total;
        const need_taps = e.drafter != null;
        const taps_ptr: u64 = if (need_taps) e.sc_pf.dspark_taps else 0;

        const r = e.runner();
        try r.forwardWithTapsLastOnly(e.stream, &sq.kv, &e.sc_pf, n, pos0 + n - 1, is_last, taps_ptr);

        if (late_mark > pos0 and late_mark < pos0 + n) sq.mark_at = late_mark;
        if (need_taps) {
            if (sq.mark_at != 0) {
                // A state kept at the mark inside this chunk: absorb up to it, stash the draft ring, absorb the rest.
                const kk = late_mark - pos0;
                const dc = &sq.draft.?;
                try e.drafter.?.absorb(&r, e.stream, dc, &e.sc_pf, e.sc_pf.dspark_taps, kk, pos0);
                const rb = dspark.num_layers * dc.layer_bytes;
                if (sq.mark_ring == null) sq.mark_ring = try cuda.DeviceBuffer.alloc(e.d, rb);
                try e.copy(sq.mark_ring.?.ptr, dc.buf.ptr, rb);
                try e.drafter.?.absorb(&r, e.stream, dc, &e.sc_pf, e.sc_pf.dspark_taps + kk * (3 * e.w.config.hidden) * 2, n - kk, late_mark);
            } else try e.drafter.?.absorb(&r, e.stream, &sq.draft.?, &e.sc_pf, e.sc_pf.dspark_taps, n, pos0);
        }

        if (is_last) {
            // Save last row F32 logits (written to slot 0 by forwardWithTapsLastOnly) for optional firstMasked redraw
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(sq.first_logits.?.ptr, e.sc_pf.logits, e.w.share.vocab_width * 4, e.stream.handle), "cuMemcpyDtoDAsync");
        }

        sq.kv.pos = pos0 + n;
        if (!is_last) {
            try e.stream.synchronize();
            if (e.group) |g| try g.check();
            return 0;
        }

        if (sq.sampling) |smp_cfg| {
            try e.smp.gatherF32(&e.k, e.stream, e.group, &e.w.share, e.sc_pf.logits, 1);
            var out: [1]u32 = undefined;
            try e.smp.drawFull(e.stream, 1, pos0 + n, sampler.rule(smp_cfg), &out);
            if (e.group) |g| try g.check();
            return out[0];
        } else {
            try e.stream.synchronize();
            if (e.group) |g| try g.check();
            var out: [1]u32 = undefined;
            try e.pickGreedy(&e.sc_pf, 0, 1, &out);
            return out[0];
        }
    }

    /// Redraw the prompt's first token under a grammar mask (`bits` device ptr to bitmask words).
    pub fn firstMasked(e: *Engine, bits: u64) !u32 {
        const sq = e.cur orelse return error.NoSequence;
        const words = (e.w.config.vocab + 31) / 32;
        try e.smp.gatherF32(&e.k, e.stream, e.group, &e.w.share, sq.first_logits.?.ptr, 1);
        if (bits != 0) try e.smp.mask(e.stream, 1, bits, words);
        const r = if (sq.sampling) |scfg| sampler.rule(scfg) else sampler.greedyRule();
        var out: [1]u32 = undefined;
        try e.smp.drawFull(e.stream, 1, sq.kv.pos, r, &out);
        if (e.group) |g| try g.check();
        return out[0];
    }

    /// Verify `ids` (`1 <= ids.len <= window_rows`) at `sq.kv.pos .. sq.kv.pos + rows - 1`.
    /// Does NOT advance `sq.kv.pos`; caller calls `keep(kept_n)` afterwards!
    pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
        const sq = e.cur orelse return error.NoSequence;
        const rows = ids.len;
        if (rows == 0 or rows > window_rows or sq.kv.pos + rows > e.capacity) return error.BadWindow;

        var ids_i32: [window_rows]i32 = undefined;
        var pos_i32: [window_rows]i32 = undefined;
        for (0..rows) |i| {
            ids_i32[i] = @intCast(ids[i]);
            pos_i32[i] = @intCast(sq.kv.pos + i);
        }
        try e.sc_dec.buf.upload(e.sc_dec.ids - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(ids_i32[0..rows]));
        try e.sc_dec.buf.upload(e.sc_dec.positions - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(pos_i32[0..rows]));

        const taps_out: u64 = if (e.drafter != null) e.sc_dec.dspark_taps else 0;
        const r = e.runner();
        const can_graph = (e.group == null or e.group.?.fast != null) and rows <= 8;
        const max_pos = sq.kv.pos + rows - 1;

        if (can_graph and max_pos + 1 <= 2176) {
            const gi = rows - 1;
            if (sq.verify_graphs[gi]) |ge| {
                try ge.launchOn(e.stream);
            } else if (!sq.verify_warmed[gi]) {
                try r.forwardWithTaps(e.stream, &sq.kv, &e.sc_dec, rows, max_pos, true, taps_out);
                sq.verify_warmed[gi] = true;
            } else {
                try cuda.graph.beginCapture(e.stream, .thread_local);
                try r.forwardWithTaps(e.stream, &sq.kv, &e.sc_dec, rows, max_pos, true, taps_out);
                var g = try cuda.graph.endCapture(e.stream);
                defer g.deinit();
                sq.verify_graphs[gi] = try g.instantiate();
                try sq.verify_graphs[gi].?.launchOn(e.stream);
            }
        } else if (can_graph) {
            // Bucketed sparse CUDA graph covering pos = 2176 .. capacity!
            const older_need = max_pos + 1 - e.w.config.sliding_window;
            var bucket: usize = 4096;
            while (bucket < older_need and bucket < e.capacity) bucket *= 2;
            bucket = @min(bucket, e.capacity);
            if (sq.sparse_ceil != bucket) {
                sq.clearSparseGraphs();
                sq.sparse_ceil = bucket;
            }
            const gi = rows - 1;
            if (sq.sparse_graphs[gi]) |ge| {
                try ge.launchOn(e.stream);
            } else if (!sq.sparse_warmed[gi]) {
                try r.forwardWithTapsBucket(e.stream, &sq.kv, &e.sc_dec, rows, max_pos, true, taps_out, bucket);
                sq.sparse_warmed[gi] = true;
            } else {
                try cuda.graph.beginCapture(e.stream, .thread_local);
                try r.forwardWithTapsBucket(e.stream, &sq.kv, &e.sc_dec, rows, max_pos, true, taps_out, bucket);
                var g = try cuda.graph.endCapture(e.stream);
                defer g.deinit();
                sq.sparse_graphs[gi] = try g.instantiate();
                try sq.sparse_graphs[gi].?.launchOn(e.stream);
            }
        } else {
            try r.forwardWithTaps(e.stream, &sq.kv, &e.sc_dec, rows, max_pos, true, taps_out);
        }

        if (sq.stash) |st| {
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(st.ptr, e.sc_dec.dspark_taps, rows * 3 * e.w.config.hidden * 2, e.stream.handle), "cuMemcpyDtoDAsync");
        }

        if (sq.sampling != null or bits != 0) {
            try e.smp.gatherF32(&e.k, e.stream, e.group, &e.w.share, e.sc_dec.logits, rows);
            if (bits != 0) {
                const words = (e.w.config.vocab + 31) / 32;
                try e.smp.mask(e.stream, rows, bits, words);
            }
            const rl = if (sq.sampling) |scfg| sampler.rule(scfg) else sampler.greedyRule();
            try e.smp.drawFull(e.stream, rows, sq.kv.pos + 1, rl, out);
            if (e.group) |g| try g.check();
        } else {
            try e.stream.synchronize();
            if (e.group) |g| try g.check();
            try e.pickGreedy(&e.sc_dec, 0, rows, out);
        }
    }

    /// Several streams' windows (no grammar masks) through one forward: weights read once for all of them.
    /// Like verify(), positions do not advance; each stream's keep() follows.
    pub fn verifyMany(e: *Engine, parts: []const Part) !void {
        if (parts.len == 0 or parts.len > max_parts) return error.BadParts;
        const stats_t0: ?std.Io.Timestamp = if (e.gstats.enabled()) std.Io.Timestamp.now(e.io, .awake) else null;
        var stats_kind: usize = 1;
        defer if (stats_t0) |t0| e.gstats.add(e.io, 0, stats_kind, t0, parts.len);
        var ids_i32: [window_rows]i32 = @splat(0);
        var pos_i32: [window_rows]i32 = @splat(0);
        var row_seg_i32: [window_rows]i32 = @splat(-1);
        var segs: [max_parts]fw.Seg = undefined;
        var seg_tab_arr: [max_parts]kv_mod.SegEntry = undefined;
        var kv_ptrs_arr: [max_parts * 64 * 3]u64 = undefined;
        var seqs: [max_parts]*Seq = undefined;
        var rows: usize = 0;
        var any_sparse = false;
        var max_older: usize = 0;
        var greedy = true;
        const n_layers = e.w.loaded_layers;
        for (parts, 0..) |pt, i| {
            if (pt.slot >= e.seqs.items.len) return error.NoSequence;
            const sq = e.seqs.items[pt.slot] orelse return error.NoSequence;
            const n = pt.ids.len;
            if (n == 0 or rows + n > window_rows or sq.kv.pos + n > e.capacity) return error.BadWindow;
            for (0..n) |j| {
                ids_i32[rows + j] = @intCast(pt.ids[j]);
                pos_i32[rows + j] = @intCast(sq.kv.pos + j);
                row_seg_i32[rows + j] = @intCast(i);
            }
            const mp = sq.kv.pos + n - 1;
            var bucket: usize = 0;
            const sparse = (mp + 1 > 2176);
            if (sparse) {
                const need = mp + 1 - e.w.config.sliding_window;
                bucket = 4096;
                while (bucket < need and bucket < e.capacity) bucket *= 2;
                bucket = @min(bucket, e.capacity);
                any_sparse = true;
                max_older = @max(max_older, bucket);
            }
            segs[i] = .{ .kv = &sq.kv, .row0 = rows, .n = n, .max_pos = mp, .older_ceil = bucket };
            seg_tab_arr[i] = .{
                .stash_ptr = if (sq.stash) |st| st.ptr else 0,
                .row0 = @intCast(rows),
                .n = @intCast(n),
                .max_pos = @intCast(mp),
                .older_ceil = @intCast(bucket),
                .sparse = if (sparse) 1 else 0,
            };
            const base = i * n_layers * 3;
            for (0..n_layers) |li| {
                switch (sq.kv.layers[li]) {
                    .hca => |h| {
                        kv_ptrs_arr[base + li * 3 + 0] = h.swa_kv;
                        kv_ptrs_arr[base + li * 3 + 1] = h.comp_state;
                        kv_ptrs_arr[base + li * 3 + 2] = h.comp_kv;
                    },
                    .dsa => |d| {
                        kv_ptrs_arr[base + li * 3 + 0] = d.mla_kv;
                        kv_ptrs_arr[base + li * 3 + 1] = d.idx_k;
                        kv_ptrs_arr[base + li * 3 + 2] = d.idx_k_sc;
                    },
                }
            }
            seqs[i] = sq;
            if (sq.sampling != null) greedy = false;
            rows += n;
        }
        const bucket_rows = bucketFor(rows);
        try e.sc_dec.buf.upload(e.sc_dec.ids - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(ids_i32[0..bucket_rows]));
        try e.sc_dec.buf.upload(e.sc_dec.positions - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(pos_i32[0..bucket_rows]));
        try e.sc_dec.buf.upload(e.sc_dec.row_seg - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(row_seg_i32[0..bucket_rows]));
        try e.sc_dec.buf.upload(e.sc_dec.seg_tab - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(seg_tab_arr[0..parts.len]));
        try e.sc_dec.buf.upload(e.sc_dec.kv_ptrs - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(kv_ptrs_arr[0 .. parts.len * n_layers * 3]));

        const key: SharedVerifyKey = .{ .bucket_rows = @intCast(bucket_rows), .any_sparse = any_sparse, .older_ceil = @intCast(max_older) };
        const taps_out: u64 = if (e.drafter != null) e.sc_dec.dspark_taps else 0;
        const tap_elems = 3 * e.w.config.hidden;
        const r = e.runner();
        const can_graph = (e.group == null or e.group.?.fast != null) and e.w.share.heads == 16;

        if (can_graph) {
            e.graph_tick += 1;
            var slot_idx: ?usize = null;
            var lru_idx: usize = 0;
            var lru_tick: u64 = std.math.maxInt(u64);
            for (&e.shared_verify_graphs, 0..) |*ent, ei| {
                if (std.meta.eql(ent.key, key)) {
                    slot_idx = ei;
                    break;
                }
                if (ent.key.bucket_rows == 0) {
                    lru_idx = ei;
                    lru_tick = 0;
                } else if (ent.tick < lru_tick) {
                    lru_tick = ent.tick;
                    lru_idx = ei;
                }
            }
            const ent = if (slot_idx) |si| &e.shared_verify_graphs[si] else blk: {
                e.shared_verify_graphs[lru_idx].clear();
                e.shared_verify_graphs[lru_idx].key = key;
                break :blk &e.shared_verify_graphs[lru_idx];
            };
            ent.tick = e.graph_tick;
            if (ent.graph) |ge| {
                stats_kind = 0;
                try ge.launchOn(e.stream);
            } else if (!ent.warmed) {
                try r.forwardSegsTab(e.stream, any_sparse, max_older, &e.sc_dec, bucket_rows, taps_out);
                if (taps_out != 0) try e.k.stashTapsTab(e.stream, taps_out, e.sc_dec.row_seg, e.sc_dec.seg_tab, bucket_rows, tap_elems);
                ent.warmed = true;
            } else {
                stats_kind = 2;
                try cuda.graph.beginCapture(e.stream, .thread_local);
                try r.forwardSegsTab(e.stream, any_sparse, max_older, &e.sc_dec, bucket_rows, taps_out);
                if (taps_out != 0) try e.k.stashTapsTab(e.stream, taps_out, e.sc_dec.row_seg, e.sc_dec.seg_tab, bucket_rows, tap_elems);
                var g = try cuda.graph.endCapture(e.stream);
                defer g.deinit();
                ent.graph = try g.instantiate();
                try ent.graph.?.launchOn(e.stream);
            }
        } else {
            try r.forwardSegs(e.stream, segs[0..parts.len], &e.sc_dec, rows, true, taps_out);
            if (taps_out != 0) try e.k.stashTapsTab(e.stream, taps_out, e.sc_dec.row_seg, e.sc_dec.seg_tab, bucket_rows, tap_elems);
        }

        if (greedy) {
            try e.stream.synchronize();
            if (e.group) |grp| try grp.check();
            var all: [window_rows]u32 = undefined;
            try e.pickGreedy(&e.sc_dec, 0, rows, all[0..rows]);
            for (parts, 0..) |pt, i| @memcpy(pt.out, all[segs[i].row0..][0..segs[i].n]);
            return;
        }
        // A round with a sampled stream: one RoCE allGather + one drawBatch for the sampled streams. Greedy streams
        // in the same round take the F32 argmax an all-greedy round takes (pickGreedy), so a greedy stream's tokens
        // never depend on whether a sampled stream shares its round.
        try e.smp.gatherF32(&e.k, e.stream, e.group, &e.w.share, e.sc_dec.logits, rows);
        var draw_items: [max_parts]sampler.Sampler.DrawItem = undefined;
        var n_draw: usize = 0;
        var any_greedy = false;
        for (parts, 0..) |pt, i| {
            const scfg = seqs[i].sampling orelse {
                any_greedy = true;
                continue;
            };
            const g = segs[i];
            draw_items[n_draw] = .{ .row0 = g.row0, .rows = g.n, .first_pos = seqs[i].kv.pos + 1, .rule = sampler.rule(scfg), .out = pt.out };
            n_draw += 1;
        }
        try e.smp.drawBatch(e.stream, draw_items[0..n_draw], rows);
        if (e.group) |grp| try grp.check();
        if (any_greedy) {
            var all: [window_rows]u32 = undefined;
            try e.pickGreedy(&e.sc_dec, 0, rows, all[0..rows]);
            for (parts, 0..) |pt, i| {
                if (seqs[i].sampling == null) @memcpy(pt.out, all[segs[i].row0..][0..segs[i].n]);
            }
        }
    }

    pub fn keep(e: *Engine, n: usize) !void {
        const sq = e.cur orelse return error.NoSequence;
        sq.kv.pos += n;
    }

    /// Absorb the `kept_n` committed rows from `sq.stash` (if > 0) and draft `out.len` speculative tokens from `anchor`.
    pub fn block(e: *Engine, kept_n: usize, anchor: u32, out: []u32) !void {
        const sq = e.cur orelse return error.NoSequence;
        const dr = &(e.drafter orelse return error.NoDraftHead);
        const r = e.runner();
        if (kept_n > 0) {
            const p0 = sq.kv.pos -| kept_n;
            try dr.absorb(&r, e.stream, &sq.draft.?, &e.sc_dec, sq.stash.?.ptr, kept_n, p0);
        }
        if (out.len == 0) return;
        var out_i32: [dspark.block_size]i32 = undefined;
        const can_graph = (e.group == null or e.group.?.fast != null) and e.w.share.heads == 16 and out.len <= dspark.block_size;
        if (can_graph) {
            try dr.prepareDraft(&e.sc_dec, @intCast(anchor), sq.kv.pos);
            const gi = out.len - 1;
            if (sq.draft_graphs[gi]) |ge| {
                try ge.launchOn(e.stream);
            } else if (!sq.draft_warmed[gi]) {
                try dr.launchDraftBody(&r, e.stream, &sq.draft.?, &e.sc_dec, sq.kv.pos, out.len);
                sq.draft_warmed[gi] = true;
            } else {
                try cuda.graph.beginCapture(e.stream, .thread_local);
                try dr.launchDraftBody(&r, e.stream, &sq.draft.?, &e.sc_dec, sq.kv.pos, out.len);
                var g = try cuda.graph.endCapture(e.stream);
                defer g.deinit();
                sq.draft_graphs[gi] = try g.instantiate();
                try sq.draft_graphs[gi].?.launchOn(e.stream);
            }
            try dr.finishDraft(&r, e.stream, out_i32[0..out.len]);
        } else {
            try dr.draft(&r, e.stream, &sq.draft.?, &e.sc_dec, @intCast(anchor), sq.kv.pos, out_i32[0..out.len]);
        }
        for (out, out_i32[0..out.len]) |*dst, v| dst.* = @intCast(v);
    }

    /// Batched absorb + DSpark drafting across multiple concurrent streams (reads draft weights ONCE!).
    pub fn blockMany(e: *Engine, parts: []const BlockPart) !void {
        if (parts.len == 0 or parts.len > max_parts) return error.BadParts;
        const stats_t0: ?std.Io.Timestamp = if (e.gstats.enabled()) std.Io.Timestamp.now(e.io, .awake) else null;
        var stats_kind: usize = 3; // 3: no draft body ran this round (absorb only), not counted
        defer if (stats_t0) |t0| if (stats_kind < 3) e.gstats.add(e.io, 1, stats_kind, t0, parts.len);
        const dr = &(e.drafter orelse return error.NoDraftHead);
        const r = e.runner();

        var abs_items: [max_parts]dspark.AbsorbItem = undefined;
        var n_abs: usize = 0;
        var draft_items: [max_parts]dspark.DraftItem = undefined;
        var draft_part_idx: [max_parts]usize = undefined;
        var n_draft: usize = 0;
        var out_i32: [max_parts][dspark.block_size]i32 = undefined;

        for (parts, 0..) |pt, i| {
            if (pt.slot >= e.seqs.items.len) return error.NoSequence;
            const sq = e.seqs.items[pt.slot] orelse return error.NoSequence;
            if (pt.advance > 0) sq.kv.pos += pt.advance;
            if (pt.kept_n > 0) {
                abs_items[n_abs] = .{
                    .caches = &sq.draft.?,
                    .taps_dev = sq.stash.?.ptr,
                    .rows = pt.kept_n,
                    .p0 = sq.kv.pos -| pt.kept_n,
                };
                n_abs += 1;
            }
            if (pt.out.len > 0) {
                draft_items[n_draft] = .{
                    .caches = &sq.draft.?,
                    .anchor = @intCast(pt.anchor),
                    .p = sq.kv.pos,
                    .out = out_i32[n_draft][0..pt.out.len],
                };
                draft_part_idx[n_draft] = i;
                n_draft += 1;
            }
        }
        if (n_abs > 0) try dr.absorbMany(&r, e.stream, &e.sc_pf, abs_items[0..n_abs]);
        if (n_draft > 0) {
            // TF_ERNIE55_DRAFT_STAGE=1: copy each drafting stream's DSpark rings into a fixed staging ring per batch
            // position, so the shared draft graph reads the same addresses whatever slots draft (its key drops the
            // slots). The draft reads exactly the values it read in place; what it writes into the staged rings
            // (its own block) absorb rewrites from the target's taps before any later draft reads it.
            const stage_on = e.draftStageOn();
            var stage_caches: [max_parts]dspark.Caches = undefined;
            if (stage_on) {
                const stage = try e.draftStage();
                const bytes = dspark.num_layers * dspark.ring_cap * 512 * 2;
                for (0..n_draft) |di| {
                    const src = draft_items[di].caches;
                    std.debug.assert(src.buf.len >= bytes);
                    stage_caches[di] = .{ .buf = .{ .d = e.d, .ptr = stage.ptr + di * bytes, .len = bytes, .borrowed = true }, .layer_bytes = src.layer_bytes };
                    try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(stage_caches[di].buf.ptr, src.buf.ptr, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
                    draft_items[di].caches = &stage_caches[di];
                }
            }
            const max_out = try dr.prepareDraftMany(&e.sc_pf, draft_items[0..n_draft]);
            if (max_out > 0) {
                const can_graph = (e.group == null or e.group.?.fast != null) and e.w.share.heads == 16;
                if (can_graph) {
                    var dkey: SharedDraftKey = .{ .np = @intCast(n_draft), .max_out = @intCast(max_out) };
                    if (!stage_on) {
                        for (0..n_draft) |di| dkey.slots[di] = @intCast(parts[draft_part_idx[di]].slot);
                    }
                    const draft_cap: usize = if (stage_on) e.shared_draft_graphs.len else 8;
                    e.graph_tick += 1;
                    var slot_idx: ?usize = null;
                    var lru_idx: usize = 0;
                    var lru_tick: u64 = std.math.maxInt(u64);
                    for (e.shared_draft_graphs[0..draft_cap], 0..) |*ent, ei| {
                        if (std.meta.eql(ent.key, dkey)) {
                            slot_idx = ei;
                            break;
                        }
                        if (ent.key.np == 0) {
                            lru_idx = ei;
                            lru_tick = 0;
                        } else if (ent.tick < lru_tick) {
                            lru_tick = ent.tick;
                            lru_idx = ei;
                        }
                    }
                    const ent = if (slot_idx) |si| &e.shared_draft_graphs[si] else blk: {
                        e.shared_draft_graphs[lru_idx].clear();
                        e.shared_draft_graphs[lru_idx].key = dkey;
                        break :blk &e.shared_draft_graphs[lru_idx];
                    };
                    ent.tick = e.graph_tick;
                    if (ent.graph) |ge| {
                        stats_kind = 0;
                        try ge.launchOn(e.stream);
                    } else if (!ent.warmed) {
                        stats_kind = 1;
                        try dr.launchDraftManyBody(&r, e.stream, &e.sc_pf, draft_items[0..n_draft], max_out);
                        ent.warmed = true;
                    } else {
                        stats_kind = 2;
                        try cuda.graph.beginCapture(e.stream, .thread_local);
                        try dr.launchDraftManyBody(&r, e.stream, &e.sc_pf, draft_items[0..n_draft], max_out);
                        var g = try cuda.graph.endCapture(e.stream);
                        defer g.deinit();
                        ent.graph = try g.instantiate();
                        try ent.graph.?.launchOn(e.stream);
                    }
                } else {
                    stats_kind = 1;
                    try dr.launchDraftManyBody(&r, e.stream, &e.sc_pf, draft_items[0..n_draft], max_out);
                }
                try dr.finishDraftMany(&r, e.stream, draft_items[0..n_draft]);
            }
            for (0..n_draft) |di| {
                const pt = parts[draft_part_idx[di]];
                for (pt.out, out_i32[di][0..pt.out.len]) |*dst, v| dst.* = @intCast(v);
            }
        }
    }

    /// Bytes a state kept at `at` takes on this rank: DSA rows below it, the HCA rings' last positions and
    /// compressed blocks, the draft ring.
    pub fn stateBytes(e: *const Engine, at: usize) u64 {
        var total: u64 = 0;
        for (0..e.w.loaded_layers) |li| {
            if (e.w.config.isDsa(@intCast(li))) {
                total += at * (576 * 2 + 128 + 4);
            } else {
                total += @as(u64, @min(at, snap_ring)) * 512 * 2 + (at / e.w.config.compress_ratio) * 512 * 2;
            }
        }
        if (e.drafter != null) total += dspark.num_layers * dspark.ring_cap * 512 * 2;
        return @max(total, 256);
    }

    fn copy(e: *const Engine, dst: u64, src: u64, n: usize) !void {
        if (n == 0) return;
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, e.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// The bound sequence's state at `at` and `buf` (out: to buf), part after part in stateBytes' order.
    fn copyState(e: *Engine, sq: *Seq, buf: u64, at: usize, out: bool) !void {
        var off: u64 = 0;
        const c = &e.w.config;
        for (sq.kv.layers[0..e.w.loaded_layers]) |ly| {
            switch (ly) {
                .dsa => |d| {
                    for ([_]u64{ d.mla_kv, d.idx_k, d.idx_k_sc }, [_]usize{ at * 576 * 2, at * 128, at * 4 }) |src, n| {
                        try e.copy(if (out) buf + off else src, if (out) src else buf + off, n);
                        off += n;
                    }
                },
                .hca => |h| {
                    const lo = at -| snap_ring; // at sits on the grid: these slots stand side by side in the ring
                    const ring_at = h.swa_kv + (lo & kv_mod.swa_mask) * 512 * 2;
                    const ring_n = (at - lo) * 512 * 2;
                    try e.copy(if (out) buf + off else ring_at, if (out) ring_at else buf + off, ring_n);
                    off += ring_n;
                    const comp_n = (at / c.compress_ratio) * 512 * 2;
                    try e.copy(if (out) buf + off else h.comp_kv, if (out) h.comp_kv else buf + off, comp_n);
                    off += comp_n;
                },
            }
        }
        if (sq.draft) |*dc| {
            const n = dspark.num_layers * dc.layer_bytes;
            const draft_src = if (out and sq.mark_at == at and sq.mark_ring != null) sq.mark_ring.?.ptr else dc.buf.ptr;
            try e.copy(if (out) buf + off else dc.buf.ptr, if (out) draft_src else buf + off, n);
        }
    }

    /// `at` is a state the bound sequence can keep now: its position, or a late mark just below it.
    pub fn keepable(e: *const Engine, at: usize) bool {
        const sq = e.cur orelse return false;
        if (at == 0 or at > sq.kv.pos) return false;
        return sq.kv.pos - at < snap_ring or (sq.mark_at == at and sq.kv.pos - at <= prompt_rows);
    }

    /// The bound sequence's state at its position `at` (a chunk start) copied under `id` (every rank keeps one id).
    pub fn keepAs(e: *Engine, id: u64, at: usize) !void {
        const sq = e.cur orelse return error.NoSequence;
        if (!e.keepable(at) or at % snap_grid != 0 or e.snaps.contains(id)) return error.BadKeep;
        var buf = try cuda.DeviceBuffer.alloc(e.d, e.stateBytes(at));
        errdefer buf.free();
        try e.copyState(sq, buf.ptr, at, true);
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .at = at });
    }

    /// The bound sequence restored to state `id`: its caches below the state's position, its length there.
    pub fn restoreState(e: *Engine, id: u64) !usize {
        const sq = e.cur orelse return error.NoSequence;
        const x = e.snaps.get(id) orelse return error.SnapshotStale;
        try e.copyState(sq, x.buf.ptr, x.at, false);
        sq.kv.pos = x.at;
        return x.at;
    }

    pub fn holds(e: *const Engine, id: u64) bool {
        return e.snaps.contains(id);
    }

    pub fn dropState(e: *Engine, id: u64) void {
        const kv = e.snaps.fetchRemove(id) orelse return;
        e.stream.synchronize() catch {};
        var b = kv.value.buf;
        b.free();
    }

    fn pickGreedy(e: *const Engine, sc: *const kv_mod.Scratch, row0: usize, n_rows: usize, out: []u32) !void {
        const Pair16 = extern struct { val: f32, id: i32, pad0: u32 = 0, pad1: u32 = 0 };
        var ids_loc: [window_rows]i32 = undefined;
        var vals_loc: [window_rows]f32 = undefined;
        try cuda.memory.download(e.d, sc.argmax_id + row0 * 4, std.mem.sliceAsBytes(ids_loc[0..n_rows]));
        try cuda.memory.download(e.d, sc.argmax_val + row0 * 4, std.mem.sliceAsBytes(vals_loc[0..n_rows]));
        if (e.group) |g| {
            if (g.world > 1) {
                var loc: [window_rows]Pair16 = undefined;
                for (0..n_rows) |r| loc[r] = .{ .val = vals_loc[r], .id = ids_loc[r] };
                const bytes = n_rows * @sizeOf(Pair16);
                try sc.buf.upload(sc.moe_pack - sc.buf.ptr, std.mem.sliceAsBytes(loc[0..n_rows]));
                try g.allGather(e.stream, sc.moe_pack, sc.tp_gather, bytes);
                try e.stream.synchronize();
                try g.check();
                var all: [8 * window_rows]Pair16 = undefined;
                try cuda.memory.download(e.d, sc.tp_gather, std.mem.sliceAsBytes(all[0 .. g.world * n_rows]));
                for (0..n_rows) |r| {
                    var best = all[r];
                    for (1..g.world) |wk| {
                        const p = all[wk * n_rows + r];
                        if (p.val > best.val or (p.val == best.val and p.id < best.id)) best = p;
                    }
                    out[r] = @intCast(best.id);
                }
                return;
            }
        }
        for (0..n_rows) |r| out[r] = @intCast(ids_loc[r]);
    }
};
