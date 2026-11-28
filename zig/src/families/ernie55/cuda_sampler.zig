//! TP-aware sampler over sample.cu (BF16 whole-row gather + grammar mask + top-k/top-p/temperature draw).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const shard = @import("cuda_shard.zig");
const kernels = @import("cuda_kernels.zig");

pub const Sampling = lanes.Sampling;
pub const Rule = cuda.sample.Rule;

pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

pub fn rule(s: Sampling) Rule {
    return cuda.sample.rule(s);
}

/// The argmax as a draw, for the paths that must draw a greedy row from `full` (a grammar-masked row). sample.cu
/// ranks candidates by logit * inv_t, so inv_t = 1 with top_k = 1 keeps exactly the largest logit. (inv_t = 0
/// scaled every logit to 0, and the tie went to token 0, <|ERNIE_PAD|>.) Unmasked greedy rows take
/// Engine.pickGreedy, the F32 argmax, instead.
pub fn greedyRule() Rule {
    return .{
        .seed = 0,
        .inv_t = 1,
        .top_p = 1,
        .near = 0,
        .min_log = -std.math.inf(f32),
        .top_k = 1,
    };
}

test "greedyRule keeps the logits as they are and one candidate" {
    const r = greedyRule();
    try std.testing.expectEqual(@as(f32, 1), r.inv_t);
    try std.testing.expectEqual(@as(u32, 1), r.top_k);
}

pub const Sampler = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    tf_draw: cuda.Function,
    tf_grammar_mask: cuda.Function,
    vocab: usize,
    max_rows: usize,
    local_bf16: cuda.DeviceBuffer, // bf16 [max_rows, vocab_width]
    full: cuda.DeviceBuffer, // bf16 [max_rows, vocab]
    gather_buf: cuda.DeviceBuffer, // bf16 [max_rows, vocab]
    rule_buf: cuda.DeviceBuffer,
    meta_buf: cuda.DeviceBuffer,
    tok_buf: cuda.DeviceBuffer,
    prob_buf: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, vocab: usize, vocab_width: usize, max_rows: usize) !Sampler {
        var m = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer m.unload();
        const f = try m.function("tf_draw");
        const gm = try m.function("tf_grammar_mask");
        var local_bf16 = try cuda.DeviceBuffer.alloc(d, max_rows * vocab_width * 2);
        errdefer local_bf16.free();
        var full = try cuda.DeviceBuffer.alloc(d, max_rows * vocab * 2);
        errdefer full.free();
        var gather_buf = try cuda.DeviceBuffer.alloc(d, max_rows * vocab * 2);
        errdefer gather_buf.free();
        var rb = try cuda.DeviceBuffer.alloc(d, 8 * @sizeOf(Rule));
        errdefer rb.free();
        var mb = try cuda.DeviceBuffer.alloc(d, 8 * 16);
        errdefer mb.free();
        var tb = try cuda.DeviceBuffer.alloc(d, max_rows * 4);
        errdefer tb.free();
        const pb = try cuda.DeviceBuffer.alloc(d, max_rows * 4);
        return .{
            .d = d,
            .module = m,
            .tf_draw = f,
            .tf_grammar_mask = gm,
            .vocab = vocab,
            .max_rows = max_rows,
            .local_bf16 = local_bf16,
            .full = full,
            .gather_buf = gather_buf,
            .rule_buf = rb,
            .meta_buf = mb,
            .tok_buf = tb,
            .prob_buf = pb,
        };
    }

    pub fn deinit(s: *Sampler) void {
        s.prob_buf.free();
        s.tok_buf.free();
        s.meta_buf.free();
        s.rule_buf.free();
        s.gather_buf.free();
        s.full.free();
        s.local_bf16.free();
        s.module.unload();
    }

    pub fn gatherF32(
        s: *const Sampler,
        k: *const kernels.Kernels,
        stream: cuda.Stream,
        group: ?*const cuda.tp_group.Group,
        sh: *const shard.Share,
        logits_f32: u64,
        rows: usize,
    ) !void {
        const vw = sh.vocab_width;
        try k.f32ToBf16(stream, logits_f32, s.local_bf16.ptr, rows * vw);
        if (group) |g| {
            if (g.world > 1) {
                // Single RoCE allGather (replaces rows * 4 ncclBroadcasts: ~2.5-4 ms -> ~80 us!)
                try g.allGather(stream, s.local_bf16.ptr, s.gather_buf.ptr, rows * vw * 2);
                try k.unpackVocabBf16(stream, s.gather_buf.ptr, s.full.ptr, g.world, rows, vw);
                return;
            }
        }
        try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(s.full.ptr, s.local_bf16.ptr, rows * vw * 2, stream.handle), "cuMemcpyDtoDAsync");
    }

    pub fn mask(s: *const Sampler, stream: cuda.Stream, rows: usize, bits: u64, words: usize) !void {
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(bits);
        args.add(@as(u32, @intCast(words)));
        try cuda.launch.launch(s.tf_grammar_mask, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }

    pub fn drawFull(s: *Sampler, stream: cuda.Stream, rows: usize, first_pos: u64, r: Rule, out: []u32) !void {
        return s.drawFullAt(stream, 0, rows, first_pos, r, out);
    }

    pub fn drawFullAt(s: *Sampler, stream: cuda.Stream, row0: usize, rows: usize, first_pos: u64, r: Rule, out: []u32) !void {
        std.debug.assert(row0 + rows <= s.max_rows and out.len >= rows);
        try s.rule_buf.uploadAsync(0, std.mem.asBytes(&r), stream.handle);
        const key_base: i32 = @intCast(if (first_pos > 0) first_pos - 1 else 0);
        const meta = [4]i32{ key_base, 0, 0, 0 };
        try s.meta_buf.uploadAsync(0, std.mem.asBytes(&meta), stream.handle);
        var args: cuda.Args = .{};
        args.add(s.full.ptr + row0 * s.vocab * 2);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(s.rule_buf.ptr);
        args.add(s.meta_buf.ptr);
        args.add(@as(i32, 0));
        args.add(s.tok_buf.ptr);
        args.add(s.prob_buf.ptr);
        try cuda.launch.launch(s.tf_draw, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &args);
        try stream.synchronize();
        try s.tok_buf.download(0, std.mem.sliceAsBytes(out[0..rows]));
    }

    pub const DrawItem = struct {
        row0: usize,
        rows: usize,
        first_pos: u64,
        rule: Rule,
        out: []u32,
    };

    /// Draw multiple concurrent streams from `s.full` with ONE single stream.synchronize() + download!
    pub fn drawBatch(s: *Sampler, stream: cuda.Stream, items: []const DrawItem, total_rows: usize) !void {
        std.debug.assert(items.len <= 8 and total_rows <= s.max_rows);
        var rules: [8]Rule = undefined;
        var metas: [8][4]i32 = undefined;
        for (items, 0..) |it, i| {
            rules[i] = it.rule;
            const key_base: i32 = @intCast(if (it.first_pos > 0) it.first_pos - 1 else 0);
            metas[i] = .{ key_base, 0, 0, 0 };
        }
        try s.rule_buf.uploadAsync(0, std.mem.sliceAsBytes(rules[0..items.len]), stream.handle);
        try s.meta_buf.uploadAsync(0, std.mem.sliceAsBytes(metas[0..items.len]), stream.handle);
        for (items, 0..) |it, i| {
            var args: cuda.Args = .{};
            args.add(s.full.ptr + it.row0 * s.vocab * 2);
            args.add(@as(u32, @intCast(s.vocab)));
            args.add(s.rule_buf.ptr + i * @sizeOf(Rule));
            args.add(s.meta_buf.ptr + i * 16);
            args.add(@as(i32, 0));
            args.add(s.tok_buf.ptr + it.row0 * 4);
            args.add(s.prob_buf.ptr + it.row0 * 4);
            try cuda.launch.launch(s.tf_draw, .{ .grid = .{ .x = @intCast(it.rows) }, .block = .{ .x = 1024 } }, stream, &args);
        }
        try stream.synchronize();
        var toks: [64]u32 = undefined;
        try s.tok_buf.download(0, std.mem.sliceAsBytes(toks[0..total_rows]));
        for (items) |it| @memcpy(it.out, toks[it.row0..][0..it.rows]);
    }
};
