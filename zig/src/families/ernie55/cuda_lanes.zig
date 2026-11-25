//! ERNIE 5.5 Flash Preview CUDA lanes backend (`lanes.backend.Backend` over `cuda_follow.Leader`).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const dspark = @import("cuda_dspark.zig");
const be = lanes.backend;

const ring: usize = 64;
const max_rows: usize = 8; // 1 pending + up to 7 DSpark drafts
const max_held: usize = dspark.block_size; // 7

/// Measured 4x DGX Spark (TP=4 RoCE, CUDA-graphed) ERNIE 5.5 single-stream verify costs (ms) for 1..8 rows.
const gb10_window_costs = [_]lanes.allocate.Cost{
    .{ .width = 1, .ms = 36.0 }, .{ .width = 2, .ms = 38.0 }, .{ .width = 3, .ms = 40.0 }, .{ .width = 4, .ms = 42.0 },
    .{ .width = 5, .ms = 44.0 }, .{ .width = 6, .ms = 46.0 }, .{ .width = 7, .ms = 48.0 }, .{ .width = 8, .ms = 50.0 },
};

/// Measured 4x DGX Spark (TP=4 RoCE, CUDA-graphed) shared verify costs by total rows (dense once, experts by rows).
const gb10_shared_costs = [_]lanes.allocate.Cost{
    .{ .width = 1, .ms = 36.0 },  .{ .width = 2, .ms = 40.0 },  .{ .width = 4, .ms = 45.0 },  .{ .width = 6, .ms = 49.0 },
    .{ .width = 8, .ms = 53.0 },  .{ .width = 10, .ms = 57.0 }, .{ .width = 12, .ms = 61.0 }, .{ .width = 14, .ms = 65.0 },
    .{ .width = 16, .ms = 69.0 }, .{ .width = 20, .ms = 77.0 }, .{ .width = 24, .ms = 85.0 }, .{ .width = 28, .ms = 93.0 },
    .{ .width = 32, .ms = 101.0 },
};

const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [max_held]u32 = undefined,
    held_n: usize = 0,
    first: ?u64 = null,
    fill: ?usize = null,
};

pub const Lanes = struct {
    x: follow.Leader,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    bits: ?cuda.HostBuffer = null,
    bits_dev: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        if (self.bits) |*b| b.free();
        self.lanes.deinit(self.x.e.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &vt };
    }

    const vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_step = prefillStepFn,
        .cancel_prefill = cancelPrefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .held = heldFn,
        .first_masked = firstMaskedFn,
        .release = releaseFn,
    };

    pub fn facts(self: *const Lanes) lanes.Model {
        const head = self.x.e.drafter != null;
        return .{
            .root_copies = false,
            .context_copies = false,
            .copy_match = 8,
            .exact_width = max_rows,
            .gpu_tokens = false,
            .mtp = head,
            .speculate = head,
            .speculate_early = false,
            .fixed_draft_block = head,
            .drafts = if (head) max_held else 0,
            .draft_streams = head,
            .mtp_step_ms = if (head) 4.5 / @as(f64, max_held) else 0,
            .shared_step_ms = if (head) 0.12 else null,
            .window_costs = if (head) &gb10_window_costs else &.{},
            .hidden_rows = true,
            .batch_rows = engine.window_rows, // shared verifies: every stream's window in one forward (verifyShared)
            .shared_costs = if (head) &gb10_shared_costs else &.{},
            .learn_row_costs = false,
            .max_streams = engine.max_parts,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn freeSlot(self: *const Lanes) u32 {
        var slot: u32 = 0;
        while (true) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
    }

    fn claim(self: *Lanes, s: *const lanes.Stream) !*Lane {
        if (self.lanes.getPtr(s)) |existing| return existing;
        const slot = self.freeSlot();
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        gop.value_ptr.* = .{ .slot = slot };
        return gop.value_ptr;
    }

    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn putBits(self: *Lanes, rows: usize, masks: []const u32) !u64 {
        if (masks.len == 0) return 0;
        const words = (self.x.e.w.config.vocab + 31) / 32;
        if (masks.len != rows * words or rows > engine.window_rows) return error.InvalidGrammarMasks;
        if (self.bits == null) {
            var buf = try cuda.HostBuffer.allocMapped(self.x.e.d, engine.window_rows * words * 4);
            errdefer buf.free();
            self.bits_dev = try buf.device();
            self.bits = buf;
        }
        @memcpy(self.bits.?.slice(u32)[0..masks.len], masks);
        return self.bits_dev;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        while (!(try prefillStepFn(ptr, s, 0))) {}
    }

    /// The stream's kept state, where its prompt pass starts (null: none, off the chunk grid, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % engine.snap_grid != 0 or !self.x.e.holds(k.id)) {
            s.reuse_failed = true;
            s.reuse_error = if (k.at == 0 or k.at >= len) error.SnapshotNotPrefix else if (k.at != s.reuse.at or k.at % engine.snap_grid != 0) error.SnapshotOutOfStep else error.SnapshotStale;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at `at` (a chunk start): the store keeps the state there when it marked it.
    fn mark(s: *lanes.Stream, at: usize) void {
        if (at == 0) return;
        const hook = s.reuse.hook orelse return;
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) == null) return;
        hook.at(hook.ptr, s, @intCast(at));
    }

    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream, max_step: u32) anyerror!bool {
        _ = max_step; // chunks stay on the prompt grid (kept states stand on it)
        const self = of(ptr);
        const l = try self.claim(s);
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + max_rows >= self.x.e.capacity) return error.BadPrompt;

        try self.x.bind(l.slot);
        var resume_id: u64 = 0;
        if (l.fill == null) {
            try self.x.begin(s.sampling);
            l.fill = 0;
            l.pending_rows = null;
            l.held_n = 0;
            l.first = null;
            s.cached = 0;
            if (s.prompt_logprobs == null) if (self.resumeFrom(s, prompt.len)) |r| {
                l.fill = r.at;
                resume_id = r.id;
            };
        }

        const pos0 = l.fill.?;
        if (resume_id == 0) mark(s, pos0);
        var take_n: usize = @min(prompt.len - pos0, engine.prompt_rows);
        for (s.reuse.marks) |m| { // end the chunk where the cache wants a state kept
            if (m > pos0 and m < pos0 + take_n) take_n = m - pos0;
        }
        const is_last = (pos0 + take_n == prompt.len);
        const tok = self.x.chunk(prompt[pos0 .. pos0 + take_n], pos0, prompt.len, is_last, resume_id) catch |err| {
            l.fill = null;
            return err;
        };
        l.fill = pos0 + take_n;
        if (is_last) {
            l.first = self.take(tok);
            l.fill = null;
            return true;
        }
        return false;
    }

    fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        if (self.lanes.getPtr(s)) |l| {
            l.fill = null;
        }
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        _ = position;
        const self = of(ptr);
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        return l.first orelse return error.NoFirstToken;
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        _ = position;
        const self = of(ptr);
        const l = try self.bind(s, null);
        const bits = try self.putBits(1, mask);
        const tok = try self.x.firstMasked(bits, mask);
        const h = self.take(tok);
        l.first = h;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = try self.bind(s, null);
        if (position != self.x.e.pos() + 1) return error.PositionMismatch;
        const tok_in = [1]u32{self.value(feed)};
        var out: [1]u32 = undefined;
        try self.x.verify(&tok_in, &out, 0, &.{});
        l.pending_rows = 1;
        return self.take(out[0]);
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return of(ptr).drawn[handle % ring];
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1 and windows.len <= engine.max_parts) {
            var rows: usize = 0;
            var masked = false;
            for (windows) |w| {
                rows += w.rows();
                masked = masked or w.masks.len > 0 or w.parents != null;
            }
            if (!masked and rows <= engine.window_rows) return self.verifyShared(windows, out);
        }
        var ids: [max_rows]u32 = undefined;
        var sampled: [max_rows]u32 = undefined;
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            const p = self.x.e.pos();
            for (w.positions, 0..) |pos_w, r| if (pos_w != p + 1 + r) return error.PositionMismatch;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            const bits = try self.putBits(rows, w.masks);
            try self.x.verify(ids[0..rows], sampled[0..rows], bits, w.masks);
            @memcpy(o.sampled[0..rows], sampled[0..rows]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    /// Every stream's window in one forward (Engine.verifyMany): the weights are read once for the round.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        var ids: [engine.window_rows]u32 = undefined;
        var draws: [engine.window_rows]u32 = undefined;
        var parts: [engine.max_parts]engine.Part = undefined;
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = try self.bind(w.stream, null); // settles the stream's last keep
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const rows = w.rows();
            const p = self.x.e.pos();
            for (w.positions, 0..) |pos_w, r| if (pos_w != p + 1 + r) return error.PositionMismatch;
            ids[at] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[at + 1 ..][0..drafts.len], drafts);
            parts[i] = .{ .slot = l.slot, .ids = ids[at..][0..rows], .out = draws[at..][0..rows] };
            at += rows;
        }
        try self.x.verifyMany(parts[0..windows.len]);
        for (windows, out, 0..) |w, *o, i| {
            const l = self.lanes.getPtr(w.stream).?;
            const rows = w.rows();
            @memcpy(o.sampled[0..rows], parts[i].out);
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            _ = try self.bind(w.stream, path.len);
        }
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (self.x.e.drafter == null) return error.NoDraftHead;
        if (requests.len > 1 and requests.len <= engine.max_parts) {
            var parts: [engine.max_parts]engine.BlockPart = undefined;
            var ls: [engine.max_parts]*Lane = undefined;
            for (requests, 0..) |r, i| {
                const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
                const kept: ?usize = if (r.rows) |rows| rows.len else null;
                const advance: usize = if (l.pending_rows) |pending| (kept orelse pending) else 0;
                const sq = if (l.slot < self.x.e.seqs.items.len) self.x.e.seqs.items[l.slot] orelse return error.NoSequence else return error.NoSequence;
                if (r.position != sq.kv.pos + advance + 1) return error.PositionMismatch;
                if (r.lanes != null) return error.TreesNotBuilt;
                var n: usize = 0;
                const anchor = if (r.rows) |rows| a: {
                    for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
                    n = rows.len;
                    break :a r.follow[rows.len - 1];
                } else self.value(r.first orelse return error.NoFirstToken);
                const depth = @min(r.depth, max_held);
                ls[i] = l;
                parts[i] = .{ .slot = l.slot, .kept_n = n, .anchor = anchor, .out = l.held[0..depth], .advance = advance };
            }
            try self.x.blockMany(parts[0..requests.len]);
            for (requests, 0..) |r, i| {
                ls[i].pending_rows = null;
                ls[i].held_n = @min(r.depth, max_held);
            }
            return;
        }
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            if (r.lanes != null) return error.TreesNotBuilt;
            var n: usize = 0;
            const anchor = if (r.rows) |rows| a: {
                for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
                n = rows.len;
                break :a r.follow[rows.len - 1];
            } else self.value(r.first orelse return error.NoFirstToken);
            const depth = @min(r.depth, max_held);
            try self.x.block(n, anchor, l.held[0..depth]);
            l.held_n = depth;
        }
    }

    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const l = of(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("ernie55 releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
        if (self.lanes.count() == 0) self.x.idle() catch |err| std.log.err("ernie55 idle step: {s}", .{@errorName(err)});
    }
};
