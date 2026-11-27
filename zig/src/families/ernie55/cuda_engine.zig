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

/// Shared-window graph caches: room for `shared_graph_cap` keys, of which TF_ERNIE55_SHARED_GRAPHS (default 8, the
/// original size) are used. A shared verify's key holds every part's row count, so mixed draft depths across streams
/// make many keys; a miss runs the forward eagerly (first sight) or captures it (second sight).
/// TF_ERNIE55_GRAPH_STATS=1 logs the hit/eager/capture counts every 500 shared verifies.
/// A capture costs far more than an eager forward (measured: a 64-entry cache whose keys are captured on their second
/// sight and then rarely reused ran c8/c16 23% slower than the 8-entry one), so two knobs bound it:
/// TF_ERNIE55_CAPTURE_AFTER=k captures a key on its k-th sight (default 2, the original; at least 2), and
/// TF_ERNIE55_GRAPH_MAX_PARTS=m graphs only shared verifies of at most m streams (default 8, all), the wider ones eager.
pub const shared_graph_cap: usize = 64;
const GraphCfg = struct { active: usize = 8, stats: bool = false, capture_after: u8 = 2, max_graph_parts: usize = max_parts, read: bool = false };
var graph_cfg: GraphCfg = .{};

fn graphCfg() *const GraphCfg {
    if (!graph_cfg.read) {
        graph_cfg.read = true;
        if (std.c.getenv("TF_ERNIE55_SHARED_GRAPHS")) |v| {
            const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
            if (n >= 1) graph_cfg.active = @min(n, shared_graph_cap);
        }
        if (std.c.getenv("TF_ERNIE55_GRAPH_STATS")) |v| graph_cfg.stats = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_ERNIE55_CAPTURE_AFTER")) |v| {
            const k = std.fmt.parseInt(u8, std.mem.span(v), 10) catch 2;
            graph_cfg.capture_after = @max(k, 2);
        }
        if (std.c.getenv("TF_ERNIE55_GRAPH_MAX_PARTS")) |v| {
            const m = std.fmt.parseInt(usize, std.mem.span(v), 10) catch max_parts;
            graph_cfg.max_graph_parts = @min(m, max_parts);
        }
        std.log.info("ernie55: shared graph cache {d} of {d} entries, capture after {d} sights, graphs up to {d} streams, stats {s}", .{ graph_cfg.active, shared_graph_cap, graph_cfg.capture_after, graph_cfg.max_graph_parts, if (graph_cfg.stats) "on" else "off" });
    }
    return &graph_cfg;
}

const SharedVerifyKey = struct {
    np: u8 = 0,
    slots: [max_parts]u8 = @splat(0),
    ns: [max_parts]u8 = @splat(0),
    buckets: [max_parts]u32 = @splat(0),
};

const SharedVerifyEntry = struct {
    key: SharedVerifyKey = .{},
    graph: ?cuda.graph.Exec = null,
    seen: u8 = 0, // eager forwards run for this key so far (it is captured on sight `capture_after`)
    tick: u64 = 0,

    fn clear(self: *SharedVerifyEntry) void {
        if (self.graph) |*ge| {
            ge.deinit();
            self.graph = null;
        }
        self.seen = 0;
        self.key.np = 0;
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
    shared_verify_graphs: [shared_graph_cap]SharedVerifyEntry = @splat(.{}),
    shared_draft_graphs: [shared_graph_cap]SharedDraftEntry = @splat(.{}),
    graph_tick: u64 = 0,
    sv_stats: [3]u64 = @splat(0), // shared verify: graph launches, eager first sights, captures
    sd_stats: [3]u64 = @splat(0), // shared draft: the same

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
        for (&e.shared_verify_graphs) |*ent| {
            if (ent.key.np == 0) continue;
            for (0..ent.key.np) |i| if (ent.key.slots[i] == slot) {
                ent.clear();
                break;
            };
        }
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
        var ids_i32: [window_rows]i32 = undefined;
        var pos_i32: [window_rows]i32 = undefined;
        var segs: [max_parts]fw.Seg = undefined;
        var seqs: [max_parts]*Seq = undefined;
        var key: SharedVerifyKey = .{ .np = @intCast(parts.len) };
        var rows: usize = 0;
        var greedy = true;
        for (parts, 0..) |pt, i| {
            if (pt.slot >= e.seqs.items.len) return error.NoSequence;
            const sq = e.seqs.items[pt.slot] orelse return error.NoSequence;
            const n = pt.ids.len;
            if (n == 0 or rows + n > window_rows or sq.kv.pos + n > e.capacity) return error.BadWindow;
            for (0..n) |j| {
                ids_i32[rows + j] = @intCast(pt.ids[j]);
                pos_i32[rows + j] = @intCast(sq.kv.pos + j);
            }
            const mp = sq.kv.pos + n - 1;
            var bucket: usize = 0;
            if (mp + 1 > 2176) {
                const need = mp + 1 - e.w.config.sliding_window;
                bucket = 4096;
                while (bucket < need and bucket < e.capacity) bucket *= 2;
                bucket = @min(bucket, e.capacity);
            }
            segs[i] = .{ .kv = &sq.kv, .row0 = rows, .n = n, .max_pos = mp, .older_ceil = bucket };
            seqs[i] = sq;
            key.slots[i] = @intCast(pt.slot);
            key.ns[i] = @intCast(n);
            key.buckets[i] = @intCast(bucket);
            if (sq.sampling != null) greedy = false;
            rows += n;
        }
        try e.sc_dec.buf.upload(e.sc_dec.ids - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(ids_i32[0..rows]));
        try e.sc_dec.buf.upload(e.sc_dec.positions - e.sc_dec.buf.ptr, std.mem.sliceAsBytes(pos_i32[0..rows]));

        const taps_out: u64 = if (e.drafter != null) e.sc_dec.dspark_taps else 0;
        const r = e.runner();
        const tap_row = 3 * e.w.config.hidden * 2;
        const can_graph = (e.group == null or e.group.?.fast != null) and parts.len <= graphCfg().max_graph_parts;

        if (can_graph) {
            const gc = graphCfg();
            e.graph_tick += 1;
            var slot_idx: ?usize = null;
            var lru_idx: usize = 0;
            var lru_tick: u64 = std.math.maxInt(u64);
            for (e.shared_verify_graphs[0..gc.active], 0..) |*ent, ei| {
                if (std.meta.eql(ent.key, key)) {
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
            const ent = if (slot_idx) |si| &e.shared_verify_graphs[si] else blk: {
                e.shared_verify_graphs[lru_idx].clear();
                e.shared_verify_graphs[lru_idx].key = key;
                break :blk &e.shared_verify_graphs[lru_idx];
            };
            ent.tick = e.graph_tick;
            const eager_sight = ent.graph == null and ent.seen + 1 < gc.capture_after;
            e.sv_stats[if (ent.graph != null) 0 else if (eager_sight) 1 else 2] += 1;
            if (gc.stats) {
                const total = e.sv_stats[0] + e.sv_stats[1] + e.sv_stats[2];
                if (total % 500 == 0) std.log.info("ernie55 graphs: shared verify launches={d} eager={d} captures={d} | shared draft launches={d} eager={d} captures={d} (cache {d})", .{ e.sv_stats[0], e.sv_stats[1], e.sv_stats[2], e.sd_stats[0], e.sd_stats[1], e.sd_stats[2], gc.active });
            }
            if (ent.graph) |ge| {
                try ge.launchOn(e.stream);
            } else if (eager_sight) {
                try r.forwardSegs(e.stream, segs[0..parts.len], &e.sc_dec, rows, true, taps_out);
                for (0..parts.len) |i| {
                    if (seqs[i].stash) |st| try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(st.ptr, e.sc_dec.dspark_taps + segs[i].row0 * tap_row, segs[i].n * tap_row, e.stream.handle), "cuMemcpyDtoDAsync");
                }
                ent.seen += 1;
            } else {
                try cuda.graph.beginCapture(e.stream, .thread_local);
                try r.forwardSegs(e.stream, segs[0..parts.len], &e.sc_dec, rows, true, taps_out);
                for (0..parts.len) |i| {
                    if (seqs[i].stash) |st| try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(st.ptr, e.sc_dec.dspark_taps + segs[i].row0 * tap_row, segs[i].n * tap_row, e.stream.handle), "cuMemcpyDtoDAsync");
                }
                var g = try cuda.graph.endCapture(e.stream);
                defer g.deinit();
                ent.graph = try g.instantiate();
                try ent.graph.?.launchOn(e.stream);
            }
        } else {
            try r.forwardSegs(e.stream, segs[0..parts.len], &e.sc_dec, rows, true, taps_out);
            for (0..parts.len) |i| {
                if (seqs[i].stash) |st| try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(st.ptr, e.sc_dec.dspark_taps + segs[i].row0 * tap_row, segs[i].n * tap_row, e.stream.handle), "cuMemcpyDtoDAsync");
            }
        }

        if (greedy) {
            try e.stream.synchronize();
            if (e.group) |grp| try grp.check();
            var all: [window_rows]u32 = undefined;
            try e.pickGreedy(&e.sc_dec, 0, rows, all[0..rows]);
            for (parts, 0..) |pt, i| @memcpy(pt.out, all[segs[i].row0..][0..segs[i].n]);
            return;
        }
        // Single RoCE allGather + single-sync drawBatch for ALL concurrent streams!
        try e.smp.gatherF32(&e.k, e.stream, e.group, &e.w.share, e.sc_dec.logits, rows);
        var draw_items: [max_parts]sampler.Sampler.DrawItem = undefined;
        for (parts, 0..) |pt, i| {
            const g = segs[i];
            const rl = if (seqs[i].sampling) |scfg| sampler.rule(scfg) else sampler.greedyRule();
            draw_items[i] = .{ .row0 = g.row0, .rows = g.n, .first_pos = seqs[i].kv.pos + 1, .rule = rl, .out = pt.out };
        }
        try e.smp.drawBatch(e.stream, draw_items[0..parts.len], rows);
        if (e.group) |grp| try grp.check();
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
            const max_out = try dr.prepareDraftMany(&e.sc_pf, draft_items[0..n_draft]);
            if (max_out > 0) {
                const can_graph = (e.group == null or e.group.?.fast != null) and e.w.share.heads == 16;
                if (can_graph) {
                    var dkey: SharedDraftKey = .{ .np = @intCast(n_draft), .max_out = @intCast(max_out) };
                    for (0..n_draft) |di| dkey.slots[di] = @intCast(parts[draft_part_idx[di]].slot);
                    e.graph_tick += 1;
                    var slot_idx: ?usize = null;
                    var lru_idx: usize = 0;
                    var lru_tick: u64 = std.math.maxInt(u64);
                    for (e.shared_draft_graphs[0..graphCfg().active], 0..) |*ent, ei| {
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
                    e.sd_stats[if (ent.graph != null) 0 else if (!ent.warmed) 1 else 2] += 1;
                    if (ent.graph) |ge| {
                        try ge.launchOn(e.stream);
                    } else if (!ent.warmed) {
                        try dr.launchDraftManyBody(&r, e.stream, &e.sc_pf, draft_items[0..n_draft], max_out);
                        ent.warmed = true;
                    } else {
                        try cuda.graph.beginCapture(e.stream, .thread_local);
                        try dr.launchDraftManyBody(&r, e.stream, &e.sc_pf, draft_items[0..n_draft], max_out);
                        var g = try cuda.graph.endCapture(e.stream);
                        defer g.deinit();
                        ent.graph = try g.instantiate();
                        try ent.graph.?.launchOn(e.stream);
                    }
                } else {
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
