//! Bundled 3-layer DSpark speculative drafter for ERNIE 5.5 Flash Preview (SWA-128 non-causal HCA + Latent MXFP4 MoE + Markov head + MultiMax).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const kv_mod = @import("cuda_kv.zig");
const fw = @import("cuda_forward.zig");

const Driver = cuda.Driver;
const Stream = cuda.Stream;
const DeviceBuffer = cuda.DeviceBuffer;
const Scratch = kv_mod.Scratch;

pub const block_size: usize = 7;
pub const noise_token: i32 = 1024;
pub const num_layers: usize = 3;
/// Power-of-2 ring capacity > sliding_window (128) + block_size (7).
pub const ring_cap: usize = 256;
pub const ring_mask: usize = ring_cap - 1;
/// Target layers tapped (0-indexed; checkpoint's dspark_target_layer_ids = [40, 41, 42] are 1-indexed).
pub const tap_layers = [3]usize{ 39, 40, 41 };

/// Per-sequence DSpark SWA KV ring cache: 3 layers x 256 slots x 512 BF16 = 768 KiB total!
pub const Caches = struct {
    buf: DeviceBuffer,
    layer_bytes: usize,

    pub fn init(d: *const Driver) !Caches {
        const layer_bytes = ring_cap * 512 * 2;
        var buf = try DeviceBuffer.alloc(d, num_layers * layer_bytes);
        errdefer buf.free();
        try buf.fill8(0, null);
        return .{ .buf = buf, .layer_bytes = layer_bytes };
    }

    pub fn deinit(self: *Caches) void {
        self.buf.free();
        self.* = undefined;
    }

    pub fn ring(self: *const Caches, l: usize) u64 {
        return self.buf.ptr + l * self.layer_bytes;
    }
};

pub const max_batch: usize = 8;
const pick_stride: usize = 16; // 16-byte aligned {float val, i32 id, pad[8]} for RoCE allGather
const max_world: usize = 8;

pub const AbsorbItem = struct {
    caches: *const Caches,
    taps_dev: u64,
    rows: usize,
    p0: usize,
};

pub const DraftItem = struct {
    caches: *const Caches,
    anchor: i32,
    p: usize,
    out: []i32,
};

/// Reusable DSpark runner state (on-device Markov walk + RoCE 16B pick gather, single & batched).
pub const Drafter = struct {
    d: *const Driver,
    w: *const weights.DsparkWeights,
    scores: DeviceBuffer, // F32 [max_batch, vocab_width]
    picks: DeviceBuffer, // [max_world * max_batch * 16] + walk [max_batch * 8 * 4]
    walk_off: usize,

    pub fn init(d: *const Driver, w: *const weights.DsparkWeights, vocab_width: usize) !Drafter {
        var scores = try DeviceBuffer.alloc(d, max_batch * vocab_width * 4);
        errdefer scores.free();
        const walk_off = max_world * max_batch * pick_stride;
        var picks = try DeviceBuffer.alloc(d, walk_off + max_batch * 8 * 4);
        errdefer picks.free();
        try picks.fill8(0, null);
        return .{
            .d = d,
            .w = w,
            .scores = scores,
            .picks = picks,
            .walk_off = walk_off,
        };
    }

    pub fn deinit(self: *Drafter) void {
        self.picks.free();
        self.scores.free();
        self.* = undefined;
    }

    /// Absorb `rows` target context hidden taps (`taps_dev` = BF16 [rows, 12288]) at positions `p0 .. p0 + rows - 1`.
    /// Only the trailing min(rows, 128) tokens can ever fall inside the SWA-128 window, so long prefills stay O(128).
    pub fn absorb(
        self: *const Drafter,
        runner: *const fw.Runner,
        s: Stream,
        caches: *const Caches,
        sc: *const Scratch,
        taps_dev: u64,
        rows: usize,
        p0: usize,
    ) !void {
        if (rows == 0) return;
        const c = &runner.w.config;
        const take = @min(rows, c.sliding_window);
        const skip = rows - take;
        const start_pos = p0 + skip;
        const taps_ptr = taps_dev + skip * (3 * c.hidden) * 2;

        var pos_host: [128]i32 = undefined;
        for (0..take) |i| pos_host[i] = @intCast(start_pos + i);
        try cuda.copy_source.enqueue(self.d, sc.positions, std.mem.sliceAsBytes(pos_host[0..take]), s);

        const full_k = 3 * c.hidden;
        const k_w = self.w.feature_proj.k;
        const k0: usize = if (k_w < full_k) runner.w.share.rank * k_w else 0;
        try runner.fp8(s, taps_ptr + k0 * 2, full_k, take, self.w.feature_proj, sc.layer_in, sc.qmmf_part);
        if (k_w < full_k) try runner.allReduceBf16(s, sc.layer_in, sc.tp_gather, take * c.hidden);
        try runner.k.rmsNorm(s, sc.layer_in, self.w.feature_norm, sc.layer_in, take, c.hidden, c.eps);

        for (0..num_layers) |li| {
            const dl = &self.w.layers[li];
            try runner.fp8(s, sc.layer_in, c.hidden, take, dl.hca.kv_proj, sc.kv_a, sc.qmmf_part);
            try runner.k.hcaKvStoreRing(s, sc.kv_a, dl.hca.kv_norm, sc.positions, runner.w.mla_inv_freq, caches.ring(li), take, ring_mask, c.eps);
        }
    }

    /// Batched absorb across multiple concurrent streams (reads feature_proj & 3x kv_proj ONCE!).
    pub fn absorbMany(
        self: *const Drafter,
        runner: *const fw.Runner,
        s: Stream,
        sc: *const Scratch,
        items: []const AbsorbItem,
    ) !void {
        const c = &runner.w.config;
        const tap_row = 3 * c.hidden * 2;
        var pos_host: [max_batch * 8]i32 = undefined;
        var row0s: [max_batch]usize = undefined;
        var ns: [max_batch]usize = undefined;
        var total: usize = 0;
        for (items, 0..) |it, i| {
            const take = @min(it.rows, 8);
            row0s[i] = total;
            ns[i] = take;
            if (take == 0) continue;
            const skip = it.rows - take;
            const src_ptr = it.taps_dev + skip * tap_row;
            const dst_ptr = sc.dspark_taps + total * tap_row;
            try self.d.check(self.d.api.cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, take * tap_row, s.handle), "cuMemcpyDtoDAsync");
            for (0..take) |j| pos_host[total + j] = @intCast(it.p0 + skip + j);
            total += take;
        }
        if (total == 0) return;

        try sc.buf.upload(sc.positions - sc.buf.ptr, std.mem.sliceAsBytes(pos_host[0..total]));
        const full_k = 3 * c.hidden;
        const k_w = self.w.feature_proj.k;
        const k0: usize = if (k_w < full_k) runner.w.share.rank * k_w else 0;
        try runner.fp8(s, sc.dspark_taps + k0 * 2, full_k, total, self.w.feature_proj, sc.layer_in, sc.qmmf_part);
        if (k_w < full_k) try runner.allReduceBf16(s, sc.layer_in, sc.tp_gather, total * c.hidden);
        try runner.k.rmsNorm(s, sc.layer_in, self.w.feature_norm, sc.layer_in, total, c.hidden, c.eps);

        for (0..num_layers) |li| {
            const dl = &self.w.layers[li];
            try runner.fp8(s, sc.layer_in, c.hidden, total, dl.hca.kv_proj, sc.kv_a, sc.qmmf_part);
            for (items, 0..) |it, i| {
                if (ns[i] == 0) continue;
                try runner.k.hcaKvStoreRing(s, sc.kv_a + row0s[i] * 512 * 2, dl.hca.kv_norm, sc.positions + row0s[i] * 4, runner.w.mla_inv_freq, it.caches.ring(li), ns[i], ring_mask, c.eps);
            }
        }
    }

    /// Stage host inputs (ids, positions, walk[0]) before launching or capturing `launchDraftBody`.
    pub fn prepareDraft(self: *const Drafter, sc: *const Scratch, anchor: i32, p: usize) !void {
        var ids: [block_size]i32 = .{ anchor, noise_token, noise_token, noise_token, noise_token, noise_token, noise_token };
        var pos: [block_size]i32 = undefined;
        for (0..block_size) |j| pos[j] = @intCast(p + j);
        try sc.buf.upload(sc.ids - sc.buf.ptr, std.mem.sliceAsBytes(&ids));
        try sc.buf.upload(sc.positions - sc.buf.ptr, std.mem.sliceAsBytes(&pos));
        try self.picks.upload(self.walk_off, std.mem.asBytes(&anchor));
    }

    /// Pure device-side draft body (CUDA-graph capturable when H_loc == 16 and RoCE is active!).
    pub fn launchDraftBody(
        self: *const Drafter,
        runner: *const fw.Runner,
        s: Stream,
        caches: *const Caches,
        sc: *const Scratch,
        p: usize,
        n_out: usize,
    ) !void {
        const c = &runner.w.config;
        const sh = &runner.w.share;
        const k = runner.k;

        try k.embedBroadcast(s, sc.ids, runner.w.embed, sc.res_a, block_size, c.hidden, c.hc_mult);
        var res_cur = sc.res_a;
        var res_nxt = sc.res_b;

        const lo: usize = (p + 1) -| c.sliding_window;
        const hi: usize = p + block_size - 1;

        for (0..num_layers) |li| {
            const dl = &self.w.layers[li];
            try k.mhcPreFast(
                s, res_cur, dl.hc_attn.fn_w, sc.mhc_mix_raw, sc.mhc_sqrsum,
                dl.hc_attn.scale, dl.hc_attn.base, dl.attn_norm,
                sc.post_mix, sc.comb_mix, sc.layer_in,
                block_size, c.hidden, c.eps, c.hc_eps, c.hc_post_alpha, c.hc_sinkhorn_iters, c.eps,
            );

            try runner.forwardHcaDspark(s, &dl.hca, caches.ring(li), sc, block_size, lo, hi, ring_mask);

            try k.mhcPost(s, sc.sub_out, res_cur, sc.post_mix, sc.comb_mix, res_nxt, 0, block_size, c.hidden);
            std.mem.swap(u64, &res_cur, &res_nxt);

            try k.mhcPreFast(
                s, res_cur, dl.hc_ffn.fn_w, sc.mhc_mix_raw, sc.mhc_sqrsum,
                dl.hc_ffn.scale, dl.hc_ffn.base, dl.ffn_norm,
                sc.post_mix, sc.comb_mix, sc.layer_in,
                block_size, c.hidden, c.eps, c.hc_eps, c.hc_post_alpha, c.hc_sinkhorn_iters, c.eps,
            );

            try runner.forwardMoe(s, &dl.moe, sc, block_size);

            try k.mhcPost(s, sc.sub_out, res_cur, sc.post_mix, sc.comb_mix, res_nxt, 0, block_size, c.hidden);
            std.mem.swap(u64, &res_cur, &res_nxt);
        }

        try k.mhcContractNorm(
            s, res_cur, self.w.hc_head_fn, self.w.hc_head_scale, self.w.hc_head_base, self.w.norm,
            sc.layer_in, n_out, c.hidden, c.eps, c.hc_eps, c.eps,
        );
        try k.lmHeadBf16(s, sc.layer_in, runner.w.lm_head_raw, sc.logits, n_out, sh.vocab_width, c.hidden);

        const walk_ptr = self.picks.ptr + self.walk_off;
        const mine = self.picks.ptr + sh.rank * pick_stride;
        for (0..n_out) |j| {
            const row_logits = sc.logits + j * sh.vocab_width * 4;
            try k.dsparkMarkovMultimax(
                s, row_logits, self.w.markov_w2, self.w.markov_w1,
                walk_ptr + j * 4, self.scores.ptr,
                runner.w.multimax_ranges, runner.w.multimax_ts,
                sh.vocab_width,
            );
            try k.argmaxF32(s, self.scores.ptr, mine + 4, mine, 1, sh.vocab_width, sh.vocab_lo);
            if (runner.group) |g| {
                if (g.world > 1) {
                    try g.allGather(s, mine, self.picks.ptr, pick_stride);
                }
            }
            try k.pickMerge(s, self.picks.ptr, sh.world, pick_stride, walk_ptr, j);
        }
    }

    pub fn finishDraft(self: *const Drafter, runner: *const fw.Runner, s: Stream, out: []i32) !void {
        try s.synchronize();
        if (runner.group) |g| try g.check();
        const walk_ptr = self.picks.ptr + self.walk_off;
        var walk_host: [block_size + 1]i32 = undefined;
        try cuda.memory.download(self.d, walk_ptr, std.mem.sliceAsBytes(walk_host[0 .. out.len + 1]));
        @memcpy(out, walk_host[1 .. out.len + 1]);
    }

    /// Draft `out.len` speculative tokens (1 <= out.len <= 7) starting from `anchor` at position `p`.
    pub fn draft(
        self: *const Drafter,
        runner: *const fw.Runner,
        s: Stream,
        caches: *const Caches,
        sc: *const Scratch,
        anchor: i32,
        p: usize,
        out: []i32,
    ) !void {
        const n_out = out.len;
        if (n_out == 0 or n_out > block_size) return error.InvalidDraftCount;
        try self.prepareDraft(sc, anchor, p);
        try self.launchDraftBody(runner, s, caches, sc, p, n_out);
        try self.finishDraft(runner, s, out);
    }

    pub fn prepareDraftMany(self: *const Drafter, sc: *const Scratch, items: []const DraftItem) !usize {
        const m_streams = items.len;
        if (m_streams == 0 or m_streams > max_batch) return error.InvalidDraftCount;
        const total_rows = m_streams * block_size;
        std.debug.assert(total_rows <= sc.max_rows);

        var ids: [max_batch * block_size]i32 = undefined;
        var pos: [max_batch * block_size]i32 = undefined;
        var walk_init: [max_batch * 8]i32 = @splat(0);
        var max_out: usize = 0;

        for (items, 0..) |it, mi| {
            if (it.out.len > block_size) return error.InvalidDraftCount;
            max_out = @max(max_out, it.out.len);
            const r0 = mi * block_size;
            ids[r0] = it.anchor;
            for (1..block_size) |j| ids[r0 + j] = noise_token;
            for (0..block_size) |j| pos[r0 + j] = @intCast(it.p + j);
            walk_init[mi * 8] = it.anchor;
        }
        if (max_out == 0) return 0;

        try sc.buf.upload(sc.ids - sc.buf.ptr, std.mem.sliceAsBytes(ids[0..total_rows]));
        try sc.buf.upload(sc.positions - sc.buf.ptr, std.mem.sliceAsBytes(pos[0..total_rows]));
        try self.picks.upload(self.walk_off, std.mem.sliceAsBytes(walk_init[0 .. m_streams * 8]));
        return max_out;
    }

    /// Pure device-side batched draft body (CUDA-graph capturable when H_loc == 16 and RoCE is active!).
    pub fn launchDraftManyBody(
        self: *const Drafter,
        runner: *const fw.Runner,
        s: Stream,
        sc: *const Scratch,
        items: []const DraftItem,
        max_out: usize,
    ) !void {
        const m_streams = items.len;
        const c = &runner.w.config;
        const sh = &runner.w.share;
        const k = runner.k;
        const total_rows = m_streams * block_size;

        var los: [max_batch]usize = undefined;
        var his: [max_batch]usize = undefined;
        for (items, 0..) |it, mi| {
            los[mi] = (it.p + 1) -| c.sliding_window;
            his[mi] = it.p + block_size - 1;
        }

        try k.embedBroadcast(s, sc.ids, runner.w.embed, sc.res_a, total_rows, c.hidden, c.hc_mult);
        var res_cur = sc.res_a;
        var res_nxt = sc.res_b;

        var rings: [max_batch]u64 = undefined;
        for (0..num_layers) |li| {
            const dl = &self.w.layers[li];
            for (items, 0..) |it, mi| rings[mi] = it.caches.ring(li);

            try k.mhcPreFast(
                s, res_cur, dl.hc_attn.fn_w, sc.mhc_mix_raw, sc.mhc_sqrsum,
                dl.hc_attn.scale, dl.hc_attn.base, dl.attn_norm,
                sc.post_mix, sc.comb_mix, sc.layer_in,
                total_rows, c.hidden, c.eps, c.hc_eps, c.hc_post_alpha, c.hc_sinkhorn_iters, c.eps,
            );
            try runner.forwardHcaDsparkSegs(s, &dl.hca, rings[0..m_streams], los[0..m_streams], his[0..m_streams], sc, block_size, ring_mask);
            try k.mhcPost(s, sc.sub_out, res_cur, sc.post_mix, sc.comb_mix, res_nxt, 0, total_rows, c.hidden);
            std.mem.swap(u64, &res_cur, &res_nxt);

            try k.mhcPreFast(
                s, res_cur, dl.hc_ffn.fn_w, sc.mhc_mix_raw, sc.mhc_sqrsum,
                dl.hc_ffn.scale, dl.hc_ffn.base, dl.ffn_norm,
                sc.post_mix, sc.comb_mix, sc.layer_in,
                total_rows, c.hidden, c.eps, c.hc_eps, c.hc_post_alpha, c.hc_sinkhorn_iters, c.eps,
            );
            try runner.forwardMoe(s, &dl.moe, sc, total_rows);
            try k.mhcPost(s, sc.sub_out, res_cur, sc.post_mix, sc.comb_mix, res_nxt, 0, total_rows, c.hidden);
            std.mem.swap(u64, &res_cur, &res_nxt);
        }

        try k.mhcContractNorm(
            s, res_cur, self.w.hc_head_fn, self.w.hc_head_scale, self.w.hc_head_base, self.w.norm,
            sc.layer_in, total_rows, c.hidden, c.eps, c.hc_eps, c.eps,
        );
        if (total_rows <= 16) {
            try k.lmHeadBf16(s, sc.layer_in, runner.w.lm_head_raw, sc.logits, total_rows, sh.vocab_width, c.hidden);
        } else {
            try runner.bf16F32(s, sc.layer_in, c.hidden, total_rows, runner.w.lm_head, sc.logits, sc.qmmf_part);
        }

        const walk_ptr = self.picks.ptr + self.walk_off;
        const chunk_bytes = m_streams * pick_stride;
        const mine = self.picks.ptr + sh.rank * chunk_bytes;
        for (0..max_out) |step| {
            try k.dsparkMarkovMultimaxBatch(
                s, sc.logits, self.w.markov_w2, self.w.markov_w1,
                walk_ptr, self.scores.ptr,
                runner.w.multimax_ranges, runner.w.multimax_ts,
                m_streams, step, sh.vocab_width,
            );
            try k.argmaxPair16(s, self.scores.ptr, mine, m_streams, sh.vocab_width, sh.vocab_lo);
            if (runner.group) |g| {
                if (g.world > 1) {
                    try g.allGather(s, mine, self.picks.ptr, chunk_bytes);
                }
            }
            try k.pickMergeBatch(s, self.picks.ptr, sh.world, m_streams, walk_ptr, step);
        }
    }

    pub fn finishDraftMany(self: *const Drafter, runner: *const fw.Runner, s: Stream, items: []const DraftItem) !void {
        const m_streams = items.len;
        try s.synchronize();
        if (runner.group) |g| try g.check();
        const walk_ptr = self.picks.ptr + self.walk_off;
        var walk_host: [max_batch * 8]i32 = undefined;
        try cuda.memory.download(self.d, walk_ptr, std.mem.sliceAsBytes(walk_host[0 .. m_streams * 8]));
        for (items, 0..) |it, mi| {
            @memcpy(it.out, walk_host[mi * 8 + 1 ..][0..it.out.len]);
        }
    }

    pub fn draftMany(
        self: *const Drafter,
        runner: *const fw.Runner,
        s: Stream,
        sc: *const Scratch,
        items: []const DraftItem,
    ) !void {
        const max_out = try self.prepareDraftMany(sc, items);
        if (max_out == 0) return;
        try self.launchDraftManyBody(runner, s, sc, items, max_out);
        try self.finishDraftMany(runner, s, items);
    }
};
