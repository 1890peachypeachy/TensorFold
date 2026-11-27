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
    .{ .width = 1, .ms = 36.0 },  .{ .width = 2, .ms = 40.0 },  .{ .width = 4, .ms = 44.0 },  .{ .width = 6, .ms = 48.0 },
    .{ .width = 8, .ms = 51.0 },  .{ .width = 12, .ms = 56.0 }, .{ .width = 16, .ms = 60.0 }, .{ .width = 20, .ms = 64.0 },
    .{ .width = 24, .ms = 67.0 }, .{ .width = 28, .ms = 71.0 }, .{ .width = 32, .ms = 75.0 }, .{ .width = 40, .ms = 83.0 },
    .{ .width = 48, .ms = 91.0 }, .{ .width = 56, .ms = 99.0 }, .{ .width = 64, .ms = 107.0 },
};

/// Cost tables a launch may set in place of the built-in ones (rank 0 prices rounds; every rank reads the same env):
/// TF_ERNIE55_WINDOW_MS / TF_ERNIE55_SHARED_MS = "width:ms,width:ms,..." with widths ascending, the way the
/// DeepSeek-V4.1 TP4 recipe sets its measured TF_DS_ROUND_MS. Anything malformed keeps the built-in table.
/// TF_ERNIE55_LEARN_ROWS=1 lets the lane core learn what a row costs here past the table (learn_row_costs).
const Overrides = struct {
    window: [24]lanes.allocate.Cost = undefined,
    window_n: usize = 0,
    shared: [48]lanes.allocate.Cost = undefined,
    shared_n: usize = 0,
    learn_rows: bool = false,
    block_ms: f64 = 4.5, // TF_ERNIE55_BLOCK_MS: one DSpark block pass (all 7 drafts), the lane's mtp_step_ms x 7
    read: bool = false,
};
var overrides: Overrides = .{};

fn readOverrides() *const Overrides {
    if (!overrides.read) {
        overrides.read = true;
        overrides.window_n = parseCosts(std.c.getenv("TF_ERNIE55_WINDOW_MS"), &overrides.window);
        overrides.shared_n = parseCosts(std.c.getenv("TF_ERNIE55_SHARED_MS"), &overrides.shared);
        if (std.c.getenv("TF_ERNIE55_LEARN_ROWS")) |v| overrides.learn_rows = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_ERNIE55_BLOCK_MS")) |v| {
            const ms = std.fmt.parseFloat(f64, std.mem.span(v)) catch 0;
            if (ms > 0 and ms < 200) {
                overrides.block_ms = ms;
                std.log.info("ernie55: DSpark block pass priced at {d:.2} ms (TF_ERNIE55_BLOCK_MS)", .{ms});
            }
        }
        if (overrides.window_n > 0) std.log.info("ernie55: window costs from TF_ERNIE55_WINDOW_MS ({d} widths)", .{overrides.window_n});
        if (overrides.shared_n > 0) std.log.info("ernie55: shared costs from TF_ERNIE55_SHARED_MS ({d} widths)", .{overrides.shared_n});
        if (overrides.learn_rows) std.log.info("ernie55: learn_row_costs on (TF_ERNIE55_LEARN_ROWS=1)", .{});
    }
    return &overrides;
}

/// "w:ms,w:ms,..." into `out`; 0 (no override) on any malformed pair, a width out of order, or too many pairs.
fn parseCosts(text: ?[*:0]const u8, out: []lanes.allocate.Cost) usize {
    const t = std.mem.span(text orelse return 0);
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, t, ',');
    while (it.next()) |pair| {
        if (n == out.len) return 0;
        const colon = std.mem.indexOfScalar(u8, pair, ':') orelse return 0;
        const w = std.fmt.parseInt(u32, std.mem.trim(u8, pair[0..colon], " "), 10) catch return 0;
        const ms = std.fmt.parseFloat(f64, std.mem.trim(u8, pair[colon + 1 ..], " ")) catch return 0;
        if (w == 0 or !(ms > 0)) return 0;
        if (n > 0 and w <= out[n - 1].width) return 0;
        out[n] = .{ .width = w, .ms = ms };
        n += 1;
    }
    return n;
}

test "parseCosts reads ascending width:ms pairs and refuses anything else" {
    var buf: [4]lanes.allocate.Cost = undefined;
    try std.testing.expectEqual(@as(usize, 3), parseCosts("1:36,8:51.5, 16:80", &buf));
    try std.testing.expectEqual(@as(u32, 16), buf[2].width);
    try std.testing.expectEqual(@as(f64, 51.5), buf[1].ms);
    try std.testing.expectEqual(@as(usize, 0), parseCosts("8:51,1:36", &buf)); // out of order
    try std.testing.expectEqual(@as(usize, 0), parseCosts("1:36,x:5", &buf));
    try std.testing.expectEqual(@as(usize, 0), parseCosts("1:0", &buf));
    try std.testing.expectEqual(@as(usize, 0), parseCosts("1:1,2:2,3:3,4:4,5:5", &buf)); // too many
    try std.testing.expectEqual(@as(usize, 0), parseCosts(null, &buf));
}

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
        const o = readOverrides();
        const window_costs: []const lanes.allocate.Cost = if (o.window_n > 0) o.window[0..o.window_n] else &gb10_window_costs;
        const shared_costs: []const lanes.allocate.Cost = if (o.shared_n > 0) o.shared[0..o.shared_n] else &gb10_shared_costs;
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
            .mtp_step_ms = if (head) o.block_ms / @as(f64, max_held) else 0,
            .shared_step_ms = if (head) 0.12 else null,
            .window_costs = if (head) window_costs else &.{},
            .hidden_rows = true,
            .batch_rows = engine.window_rows, // shared verifies: every stream's window in one forward (verifyShared)
            .shared_costs = if (head) shared_costs else &.{},
            .learn_row_costs = o.learn_rows,
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
        var late_mark: usize = 0;
        if (prompt.len - pos0 <= engine.prompt_rows) {
            // Entire remaining prompt fits in one chunk: never split it for intermediate marks
            // (splitting costs a whole extra ~250-550ms weight-streaming forward pass + redundant alloc).
            // Run the full chunk in one pass and keep the final mark (if any) at the end.
            var last_mark: usize = 0;
            for (s.reuse.marks) |m| {
                if (m > pos0 and m < prompt.len) last_mark = @max(last_mark, m);
            }
            late_mark = last_mark;
        } else {
            for (s.reuse.marks) |m| { // end the chunk where the cache wants a state kept
                if (m > pos0 and m < pos0 + take_n) take_n = m - pos0;
            }
        }
        const is_last = (pos0 + take_n == prompt.len);
        const tok = self.x.chunk(prompt[pos0 .. pos0 + take_n], pos0, prompt.len, is_last, resume_id, late_mark) catch |err| {
            l.fill = null;
            return err;
        };
        if (late_mark != 0) mark(s, late_mark);
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
