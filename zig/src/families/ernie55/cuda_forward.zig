//! ERNIE 5.5 Flash Preview full CUDA forward pass (TP=1/2/4, HCA + C128 + MLA-DSA + Latent MXFP4 SITU MoE + mHC + MultiMax).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const kv_mod = @import("cuda_kv.zig");
const kernels = @import("cuda_kernels.zig");

const Stream = cuda.Stream;
const Lane = cuda.qmmf.Lane;
const Group = cuda.tp_group.Group;
const Kernels = kernels.Kernels;
const Weights = weights.Weights;
const KvCache = kv_mod.KvCache;
const Scratch = kv_mod.Scratch;

/// Rows from which the routed experts run expert-major on tensor cores (below: the per-pick kernels).
pub const grouped_min_rows: usize = 24;
/// Bytes from which a TP all-reduce is NCCL's ring (1.5x the shard on the wire) rather than the RoCE all-gather
/// plus rank sum (3x at TP4, but one hop's latency): prompt chunks take the ring, decode windows the gather.
pub const ring_min_bytes: usize = 1 << 20;

/// Decode-sized calls (verify windows, drafts) take their own K slices (qmmf_bench.cu on GB10, FP8G, M <= 8):
/// TensorFold's shape rule slices up to 8 in a cluster, which runs 15-50% slower here than 1-4 slices.
pub const decode_rows: usize = 16;

/// The fewest K slices that still give a call about 64 blocks: 1, 2 or 4, each slice at least 8 groups of 64.
pub fn decodeSlices(n: usize, k: usize) u32 {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    const max_sk: usize = if (tiles <= 8) 4 else 2;
    var sk: usize = 1;
    while (sk < max_sk and tiles * sk * 2 <= 64 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return @intCast(sk);
}

/// One sequence's rows in a shared forward: its caches, its first row, its row count, its last position.
pub const Seg = struct { kv: *KvCache, row0: usize, n: usize, max_pos: usize, older_ceil: usize = 0 };

pub const Runner = struct {
    k: *const Kernels,
    lane: *const Lane,
    w: *const Weights,
    group: ?*const Group = null,
    comm_stream: ?Stream = null,
    ev_shared: ?cuda.Event = null,
    ev_comm: ?cuda.Event = null,

    pub fn allReduceBf16(self: *const Runner, s: Stream, buf: u64, gather_buf: u64, count: usize) !void {
        const g = self.group orelse return;
        if (g.world <= 1) return;
        if (count * 2 >= ring_min_bytes) {
            try g.lib.check(g.lib.api.ncclAllReduce(buf, buf, count, .bf16, .sum, g.comm, s.handle), "ncclAllReduce");
            return;
        }
        try g.allGather(s, buf, gather_buf, count * 2);
        try self.k.rankSumBf16(s, gather_buf, buf, count, g.world);
    }

    pub fn allReduceF32(self: *const Runner, s: Stream, buf: u64, gather_buf: u64, count: usize) !void {
        const g = self.group orelse return;
        if (g.world <= 1) return;
        if (count * 4 >= ring_min_bytes) {
            try g.lib.check(g.lib.api.ncclAllReduce(buf, buf, count, .f32, .sum, g.comm, s.handle), "ncclAllReduce");
            return;
        }
        try g.allGather(s, buf, gather_buf, count * 4);
        try self.k.rankSumF32(s, gather_buf, buf, count, g.world);
    }

    pub fn fp8(self: *const Runner, s: Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64, part: u64) !void {
        if (rows >= 32 and w.mode == .bf16 and ldx == w.k and w.k % 32 == 0) {
            if (self.w.bf16_raw.get(w.codes)) |raw_w| {
                try self.k.bf16GemmBf16Mma(s, x, raw_w, y, rows, w.n, w.k);
                return;
            }
        }
        if (rows >= 24 and w.k % 128 == 0) {
            if (self.w.fp8_raw.get(w.codes)) |raw| {
                const q_bytes = ((rows * @as(usize, w.k)) + 255) & ~@as(usize, 255);
                const xq = part;
                const sx = part + q_bytes;
                try self.k.quantBf16ToFp8B128(s, x, ldx, xq, sx, rows, w.k);
                try self.k.fp8GemmB128Mma(s, xq, sx, raw.codes, raw.scales, y, rows, w.n, w.k);
                return;
            }
        }
        if (rows <= decode_rows and w.sk == 0) {
            var wd = w;
            wd.sk = decodeSlices(w.n, w.k);
            return self.lane.matmul(s, x, ldx, rows, wd, y, part);
        }
        try self.lane.matmul(s, x, ldx, rows, w, y, part);
    }

    pub fn bf16F32(self: *const Runner, s: Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64, part: u64) !void {
        if (rows >= 32 and ldx == w.k and w.k % 32 == 0) {
            if (self.w.bf16_raw.get(w.codes)) |raw_w| {
                try self.k.bf16GemmF32Mma(s, x, raw_w, y, rows, w.n, w.k);
                return;
            }
        }
        if (rows <= decode_rows and w.sk == 0) {
            var wd = w;
            wd.sk = decodeSlices(w.n, w.k);
            return self.lane.matmulF32(s, x, ldx, rows, wd, y, part);
        }
        try self.lane.matmulF32(s, x, ldx, rows, w, y, part);
    }

    fn attnSplits(rows: usize, max_sp: usize) usize {
        if (rows > 16) return 1;
        if (rows <= 4) return max_sp;
        if (rows <= 8) return 6;
        return 4;
    }

    /// Run `rows` tokens (`ids_dev [rows] i32`, `pos_dev [rows] i32`) through all loaded layers.
    /// `max_pos` is the maximum position in this batch (used to skip DSA indexer when max_pos + 1 <= 2176).
    /// Writes final contracted+normed hidden states to `sc.layer_in [rows, 4096]`, and logits to `sc.logits [rows, vocab_width]`.
    pub fn forward(
        self: *const Runner,
        s: Stream,
        kv: *KvCache,
        sc: *const Scratch,
        rows: usize,
        max_pos: usize,
        compute_logits: bool,
    ) !void {
        return self.forwardWithTaps(s, kv, sc, rows, max_pos, compute_logits, 0);
    }

    pub fn forwardWithTaps(
        self: *const Runner,
        s: Stream,
        kv: *KvCache,
        sc: *const Scratch,
        rows: usize,
        max_pos: usize,
        compute_logits: bool,
        taps_out: u64,
    ) !void {
        return self.forwardWithTapsBucket(s, kv, sc, rows, max_pos, compute_logits, taps_out, 0);
    }

    pub fn forwardWithTapsBucket(
        self: *const Runner,
        s: Stream,
        kv: *KvCache,
        sc: *const Scratch,
        rows: usize,
        max_pos: usize,
        compute_logits: bool,
        taps_out: u64,
        older_ceil: usize,
    ) !void {
        const one = [1]Seg{.{ .kv = kv, .row0 = 0, .n = rows, .max_pos = max_pos, .older_ceil = older_ceil }};
        return self.forwardSegsFull(s, &one, sc, rows, compute_logits, false, taps_out);
    }

    /// Prompt chunk forward: when `compute_logits` is true, computes contract + lm_head + multimax + argmax ONLY on the
    /// last row (`rows - 1`), writing to `sc.logits` slot 0 (saves ~35 ms on 1024-row prefill!).
    pub fn forwardWithTapsLastOnly(
        self: *const Runner,
        s: Stream,
        kv: *KvCache,
        sc: *const Scratch,
        rows: usize,
        max_pos: usize,
        compute_logits: bool,
        taps_out: u64,
    ) !void {
        const one = [1]Seg{.{ .kv = kv, .row0 = 0, .n = rows, .max_pos = max_pos }};
        return self.forwardSegsFull(s, &one, sc, rows, compute_logits, true, taps_out);
    }

    /// `rows` tokens of one or more sequences (`segs`, rows side by side) through every layer: the linears, experts,
    /// mHC and all-reduces run once over all rows; the cache stores and attention run per sequence.
    const TabMode = struct { any_sparse: bool, max_older: usize };

    pub fn forwardSegs(
        self: *const Runner,
        s: Stream,
        segs: []const Seg,
        sc: *const Scratch,
        rows: usize,
        compute_logits: bool,
        taps_out: u64,
    ) !void {
        return self.forwardSegsImpl(s, segs, null, sc, rows, compute_logits, false, taps_out);
    }

    /// Bucketed table-mode forward over all segments described in `sc.row_seg`, `sc.seg_tab`, and `sc.kv_ptrs`.
    pub fn forwardSegsTab(
        self: *const Runner,
        s: Stream,
        any_sparse: bool,
        max_older: usize,
        sc: *const Scratch,
        rows: usize,
        taps_out: u64,
    ) !void {
        return self.forwardSegsImpl(s, &.{}, .{ .any_sparse = any_sparse, .max_older = max_older }, sc, rows, true, false, taps_out);
    }

    pub fn forwardSegsFull(
        self: *const Runner,
        s: Stream,
        segs: []const Seg,
        sc: *const Scratch,
        rows: usize,
        compute_logits: bool,
        logits_last_only: bool,
        taps_out: u64,
    ) !void {
        return self.forwardSegsImpl(s, segs, null, sc, rows, compute_logits, logits_last_only, taps_out);
    }

    fn forwardSegsImpl(
        self: *const Runner,
        s: Stream,
        segs: []const Seg,
        tab: ?TabMode,
        sc: *const Scratch,
        rows: usize,
        compute_logits: bool,
        logits_last_only: bool,
        taps_out: u64,
    ) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;

        // 1. Embed lookup + broadcast to 4-stream mHC residual
        try k.embedBroadcast(s, sc.ids, self.w.embed, sc.res_a, rows, c.hidden, c.hc_mult);

        var res_cur = sc.res_a;
        var res_nxt = sc.res_b;

        // 2. Decoder layers
        for (self.w.layers, 0..) |*ly, li| {
            // --- 2a. Attention mHC pre (collapse + attn_norm) ---
            try k.mhcPreFast(
                s,
                res_cur,
                ly.hc_attn.fn_w,
                sc.mhc_mix_raw,
                sc.mhc_sqrsum,
                ly.hc_attn.scale,
                ly.hc_attn.base,
                ly.attn_norm,
                sc.post_mix,
                sc.comb_mix,
                sc.layer_in,
                rows,
                c.hidden,
                c.eps,
                c.hc_eps,
                c.hc_post_alpha,
                c.hc_sinkhorn_iters,
                c.eps,
            );

            // --- 2b. Attention (HCA or MLA-DSA) -> sc.sub_out [rows, 4096] ---
            switch (ly.attn) {
                .hca => |*hca| if (tab != null) try self.forwardHcaTab(s, hca, li, sc, rows) else try self.forwardHca(s, hca, segs, li, sc, rows),
                .dsa => |*dsa| if (tab) |tm| try self.forwardDsaTab(s, dsa, tm, li, sc, rows) else try self.forwardDsa(s, dsa, segs, li, sc, rows),
            }

            // --- 2c. Attention mHC post ---
            try k.mhcPost(s, sc.sub_out, res_cur, sc.post_mix, sc.comb_mix, res_nxt, 0, rows, c.hidden);
            std.mem.swap(u64, &res_cur, &res_nxt);

            // --- 2d. FFN mHC pre (collapse + ffn_norm) ---
            try k.mhcPreFast(
                s,
                res_cur,
                ly.hc_ffn.fn_w,
                sc.mhc_mix_raw,
                sc.mhc_sqrsum,
                ly.hc_ffn.scale,
                ly.hc_ffn.base,
                ly.ffn_norm,
                sc.post_mix,
                sc.comb_mix,
                sc.layer_in,
                rows,
                c.hidden,
                c.eps,
                c.hc_eps,
                c.hc_post_alpha,
                c.hc_sinkhorn_iters,
                c.eps,
            );

            // --- 2e. FFN (Dense SITU MLP or Latent MXFP4 SITU MoE + Shared) -> sc.sub_out ---
            switch (ly.ffn) {
                .dense => |*mlp| try self.forwardDenseMlp(s, mlp, sc, rows),
                .moe => |*moe| try self.forwardMoe(s, moe, sc, rows),
            }

            // --- 2f. FFN mHC post (with optional DSpark aux_mean tap at layers 39, 40, 41) ---
            const tap_ptr: u64 = if (taps_out != 0 and li >= 39 and li <= 41)
                sc.dspark_taps_tmp + (li - 39) * rows * c.hidden * 2
            else
                0;
            try k.mhcPost(s, sc.sub_out, res_cur, sc.post_mix, sc.comb_mix, res_nxt, tap_ptr, rows, c.hidden);
            std.mem.swap(u64, &res_cur, &res_nxt);
        }

        if (taps_out != 0 and self.w.layers.len >= 42) {
            const slot_bytes = rows * c.hidden * 2;
            try k.concat3Bf16(
                s,
                sc.dspark_taps_tmp,
                sc.dspark_taps_tmp + slot_bytes,
                sc.dspark_taps_tmp + 2 * slot_bytes,
                taps_out,
                rows,
                c.hidden,
            );
        }

        // 3+4. Final mHC contract + norm + sharded LM head + MultiMax + argmax
        if (compute_logits) {
            const l_rows: usize = if (logits_last_only) 1 else rows;
            const res_ptr: u64 = if (logits_last_only) res_cur + (rows - 1) * c.hcDim() * 2 else res_cur;
            try k.mhcContractNorm(
                s,
                res_ptr,
                self.w.hc_head_fn,
                self.w.hc_head_scale,
                self.w.hc_head_base,
                self.w.norm,
                sc.layer_in,
                l_rows,
                c.hidden,
                c.eps,
                c.hc_eps,
                c.eps,
            );
            if (l_rows <= 16) {
                try k.lmHeadBf16(s, sc.layer_in, self.w.lm_head_raw, sc.logits, l_rows, sh.vocab_width, c.hidden);
            } else {
                try self.bf16F32(s, sc.layer_in, c.hidden, l_rows, self.w.lm_head, sc.logits, sc.qmmf_part);
            }
            try k.applyMultimax(s, sc.logits, self.w.multimax_ranges, self.w.multimax_ts, l_rows * sh.vocab_width);
            try k.argmaxF32(s, sc.logits, sc.argmax_id, sc.argmax_val, l_rows, sh.vocab_width, sh.vocab_lo);
        }
    }

    fn forwardHca(self: *const Runner, s: Stream, hca: *const weights.HcaWeights, segs: []const Seg, li: usize, sc: *const Scratch, rows: usize) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;

        // 1+2. Fused q_down + kv_proj -> RMSNorm(q_down) + RMSNorm/RoPE/store(swa_kv)
        try self.fp8(s, sc.layer_in, c.hidden, rows, hca.qdown_kv, sc.proj_tmp, sc.qmmf_part);
        for (segs) |g| {
            const lkv = &g.kv.layers[li].hca;
            try k.hcaQkvPrep(s, sc.proj_tmp + g.row0 * (c.q_lora + c.head_dim) * 2, hca.q_norm, hca.kv_norm, sc.positions + g.row0 * 4, self.w.hca_inv_freq, sc.q_down + g.row0 * c.q_lora * 2, lkv.swa_kv, g.n, kv_mod.swa_mask, c.eps);
        }

        // 3. Fused C128 Compressor (wkv+wgate in 1 BF16 GEMV + packed save + boundary flush)
        if (hca.comp_wkv_wgate) |cw| {
            try self.fp8(s, sc.layer_in, c.hidden, rows, cw, sc.proj_tmp, sc.qmmf_part);
            for (segs) |g| {
                const lkv = &g.kv.layers[li].hca;
                const pos_at = sc.positions + g.row0 * 4;
                try k.hcaCompSavePacked(s, sc.proj_tmp + g.row0 * 1024 * 2, hca.comp_ape, pos_at, lkv.comp_state, g.n, kv_mod.swa_mask);
                try k.hcaCompFlush(s, lkv.comp_state, hca.comp_norm, pos_at, self.w.hca_inv_freq, lkv.comp_kv, g.n, kv_mod.swa_mask, c.eps);
            }
        }

        // 4+5. Fused q_up + o_gate -> weightless per-head Q-RMSNorm + GPT-J RoPE + o_gate split
        try self.fp8(s, sc.q_down, c.q_lora, rows, hca.qup_ogate, sc.proj_tmp, sc.qmmf_part);
        try k.hcaQnormRopeSplit(s, sc.proj_tmp, sc.positions, self.w.hca_inv_freq, sc.q_up, sc.o_gate, rows, sh.heads, sh.o_width, c.eps);

        // 6. HCA Attention (SM121a Tensor-Core MMA + Split-K when H_loc==16!) -> sc.attn_o [rows, heads, 512]
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
        for (segs) |g| {
            const lkv = &g.kv.layers[li].hca;
            const comp_ptr = if (hca.comp_wkv_wgate != null) lkv.comp_kv else 0;
            const head_rows = g.row0 * sh.heads * 512 * 2;
            if (sh.heads == 16 and g.n >= 16) {
                try k.hcaAttnPrefillMma(s, sc.q_up + head_rows, lkv.swa_kv, comp_ptr, hca.attn_sink, sc.positions + g.row0 * 4, sc.attn_o + head_rows, g.n, kv_mod.swa_mask, c.sliding_window, scale);
            } else if (sh.heads == 16) {
                try k.hcaAttnMma(s, sc.q_up + head_rows, lkv.swa_kv, comp_ptr, hca.attn_sink, sc.positions + g.row0 * 4, sc.attn_o + head_rows, sc.attn_split_o, sc.attn_split_m, sc.attn_split_l, g.n, kv_mod.swa_mask, c.sliding_window, attnSplits(g.n, 8), scale);
            } else {
                try k.hcaAttn(s, sc.q_up + head_rows, lkv.swa_kv, comp_ptr, hca.attn_sink, sc.positions + g.row0 * 4, sc.attn_o + head_rows, g.n, sh.heads, kv_mod.swa_mask, c.sliding_window, scale);
            }
        }

        // 7. VHA post-mix (U^T @ o -> TP AllReduce -> o += V @ z)
        try k.vhaUProjBf16(s, sc.attn_o, hca.vha_u, sc.vha_z, rows, sh.heads, c.head_dim);
        try self.allReduceBf16(s, sc.vha_z, sc.vha_z_all, rows * c.vha_rank * c.head_dim);
        try k.vhaVAddBf16(s, sc.attn_o, hca.vha_v, sc.vha_z, rows, sh.heads, c.head_dim);

        // 8. Inverse GPT-J RoPE on dims [448..512) of sc.attn_o [rows, heads, 512]
        try k.ropeGptj(s, sc.attn_o, sc.positions, self.w.hca_inv_freq, rows, sh.heads, c.head_dim, c.qk_nope, -1);

        // 9. Grouped wo_a (sh.groups local groups, each [1024, 4096]) -> sc.wo_a_out [rows, o_width]
        const group_k: usize = (c.heads / c.o_groups) * c.head_dim; // 4096
        const ldx: usize = sh.groups * group_k;
        if (rows == 1 or sh.groups == 1) {
            for (0..sh.groups) |g| {
                const x_ptr = sc.attn_o + g * group_k * 2;
                const y_ptr = sc.wo_a_out + g * c.o_lora * 2;
                try self.fp8(s, x_ptr, ldx, rows, hca.wo_a[g], y_ptr, sc.qmmf_part);
            }
        } else {
            // Multi-row prefill: write contiguous [groups, rows, o_lora] into sc.q_up (temp) then interleave to [rows, groups, o_lora]
            for (0..sh.groups) |g| {
                const x_ptr = sc.attn_o + g * group_k * 2;
                const y_ptr = sc.q_up + g * rows * c.o_lora * 2;
                try self.fp8(s, x_ptr, ldx, rows, hca.wo_a[g], y_ptr, sc.qmmf_part);
            }
            try k.interleaveGroups(s, sc.q_up, sc.wo_a_out, sh.groups, rows, c.o_lora);
        }

        // 10. Gate * wo_a_out -> wo_b -> TP AllReduce -> sc.sub_out [rows, 4096]
        try k.sigmoidMulBf16(s, sc.wo_a_out, sc.o_gate, sc.wo_a_out, rows * sh.o_width);
        try self.fp8(s, sc.wo_a_out, sh.o_width, rows, hca.wo_b, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    fn forwardHcaTab(self: *const Runner, s: Stream, hca: *const weights.HcaWeights, li: usize, sc: *const Scratch, rows: usize) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;
        const layer_stride = self.w.loaded_layers * 3;
        const kv_li = sc.kv_ptrs + li * 3 * 8;

        try self.fp8(s, sc.layer_in, c.hidden, rows, hca.qdown_kv, sc.proj_tmp, sc.qmmf_part);
        try k.hcaQkvPrepTab(s, sc.proj_tmp, hca.q_norm, hca.kv_norm, sc.positions, self.w.hca_inv_freq, sc.q_down, sc.row_seg, kv_li, layer_stride, rows, kv_mod.swa_mask, c.eps);

        const has_comp = (hca.comp_wkv_wgate != null);
        if (hca.comp_wkv_wgate) |cw| {
            try self.fp8(s, sc.layer_in, c.hidden, rows, cw, sc.proj_tmp, sc.qmmf_part);
            try k.hcaCompSavePackedTab(s, sc.proj_tmp, hca.comp_ape, sc.positions, sc.row_seg, kv_li, layer_stride, rows, kv_mod.swa_mask);
            try k.hcaCompFlushTab(s, hca.comp_norm, sc.positions, self.w.hca_inv_freq, sc.row_seg, kv_li, layer_stride, rows, kv_mod.swa_mask, c.eps);
        }

        try self.fp8(s, sc.q_down, c.q_lora, rows, hca.qup_ogate, sc.proj_tmp, sc.qmmf_part);
        try k.hcaQnormRopeSplit(s, sc.proj_tmp, sc.positions, self.w.hca_inv_freq, sc.q_up, sc.o_gate, rows, sh.heads, sh.o_width, c.eps);

        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
        try k.hcaAttnMmaTab(s, sc.q_up, hca.attn_sink, sc.positions, sc.attn_o, sc.attn_split_o, sc.attn_split_m, sc.attn_split_l, sc.row_seg, sc.seg_tab, kv_li, layer_stride, has_comp, rows, kv_mod.swa_mask, c.sliding_window, 8, scale);

        try k.vhaUProjBf16(s, sc.attn_o, hca.vha_u, sc.vha_z, rows, sh.heads, c.head_dim);
        try self.allReduceBf16(s, sc.vha_z, sc.vha_z_all, rows * c.vha_rank * c.head_dim);
        try k.vhaVAddBf16(s, sc.attn_o, hca.vha_v, sc.vha_z, rows, sh.heads, c.head_dim);
        try k.ropeGptj(s, sc.attn_o, sc.positions, self.w.hca_inv_freq, rows, sh.heads, c.head_dim, c.qk_nope, -1);

        const group_k: usize = (c.heads / c.o_groups) * c.head_dim;
        const ldx: usize = sh.groups * group_k;
        for (0..sh.groups) |g| {
            const x_ptr = sc.attn_o + g * group_k * 2;
            const y_ptr = sc.q_up + g * rows * c.o_lora * 2;
            try self.fp8(s, x_ptr, ldx, rows, hca.wo_a[g], y_ptr, sc.qmmf_part);
        }
        try k.interleaveGroups(s, sc.q_up, sc.wo_a_out, sh.groups, rows, c.o_lora);

        try k.sigmoidMulBf16(s, sc.wo_a_out, sc.o_gate, sc.wo_a_out, rows * sh.o_width);
        try self.fp8(s, sc.wo_a_out, sh.o_width, rows, hca.wo_b, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    fn forwardDsaTab(self: *const Runner, s: Stream, dsa: *const weights.DsaWeights, tm: TabMode, li: usize, sc: *const Scratch, rows: usize) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;
        const layer_stride = self.w.loaded_layers * 3;
        const kv_li = sc.kv_ptrs + li * 3 * 8;

        try self.fp8(s, sc.layer_in, c.hidden, rows, dsa.qa_kva, sc.proj_tmp, sc.qmmf_part);
        try k.dsaQkvPrepTab(s, sc.proj_tmp, dsa.q_a_norm, dsa.kv_a_norm, sc.positions, self.w.mla_inv_freq, sc.q_down, sc.row_seg, kv_li, layer_stride, rows, c.eps);

        try self.fp8(s, sc.layer_in, c.hidden, rows, dsa.idx_wk, sc.idx_k, sc.qmmf_part);
        try k.layerNorm(s, sc.idx_k, dsa.idx_k_norm_w, dsa.idx_k_norm_b, sc.idx_k, rows, c.index_dim, c.eps);
        try k.ropeNeox(s, sc.idx_k, sc.positions, self.w.mla_inv_freq, rows, 1, c.index_dim, 0);
        try k.hadamard128(s, sc.idx_k, rows);
        try k.dsaIdxKStoreTab(s, sc.idx_k, sc.positions, sc.row_seg, kv_li, layer_stride, rows);

        const dense_width = c.index_topk + c.sliding_window;
        if (tm.any_sparse) {
            try self.fp8(s, sc.q_down, c.mla_q_lora, rows, dsa.idx_wq_b, sc.idx_q, sc.qmmf_part);
            try k.ropeNeox(s, sc.idx_q, sc.positions, self.w.mla_inv_freq, rows, c.index_heads, c.index_dim, 0);
            try k.hadamard128(s, sc.idx_q, rows * c.index_heads);
            try self.fp8(s, sc.layer_in, c.hidden, rows, dsa.idx_weights_proj, sc.idx_w, sc.qmmf_part);

            const w_scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(c.index_dim))) * @sqrt(@as(f64, @floatFromInt(c.index_heads)))));
            if (rows * tm.max_older > @min(sc.max_rows, 64) * sc.capacity) return error.IndexerScoresTooSmall;
            try k.dsaIndexerScoresMmaTab(s, sc.idx_q, sc.idx_w, sc.positions, sc.idx_scores, sc.row_seg, sc.seg_tab, kv_li, layer_stride, rows, tm.max_older, c.sliding_window, w_scale);
            try k.dsaSelectTopkTab(s, sc.idx_scores, sc.positions, sc.idx_sel, sc.idx_cnt, sc.row_seg, sc.seg_tab, rows, tm.max_older, c.index_topk, c.sliding_window);
        }

        const q_head = c.mla_qk_nope + c.mla_qk_rope;
        try self.fp8(s, sc.q_down, c.mla_q_lora, rows, dsa.qb_ogate, sc.proj_tmp, sc.qmmf_part);
        try k.dsaQbRopeSplit(s, sc.proj_tmp, sc.positions, self.w.mla_inv_freq, sc.q_up, sc.o_gate, rows, sh.heads, sh.dsa_o_width);
        try k.dsaAbsorbQ(s, sc.q_up, dsa.wk, sc.q_abs, rows, sh.heads);

        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q_head))));
        try k.dsaAttnMmaTab(s, sc.q_abs, sc.idx_sel, sc.idx_cnt, dsa.sinks, sc.positions, sc.attn_o, sc.attn_split_o, sc.attn_split_m, sc.attn_split_l, sc.row_seg, sc.seg_tab, kv_li, layer_stride, rows, dense_width, 12, scale);
        try k.dsaExpandV(s, sc.attn_o, dsa.wv, sc.attn_o_v, rows, sh.heads);

        try k.vhaUProjBf16(s, sc.attn_o_v, dsa.vha_u, sc.vha_z, rows, sh.heads, c.mla_v_dim);
        try self.allReduceBf16(s, sc.vha_z, sc.vha_z_all, rows * c.vha_rank * c.mla_v_dim);
        try k.vhaVAddBf16(s, sc.attn_o_v, dsa.vha_v, sc.vha_z, rows, sh.heads, c.mla_v_dim);

        try k.sigmoidMulBf16(s, sc.attn_o_v, sc.o_gate, sc.attn_o_v, rows * sh.dsa_o_width);
        try self.fp8(s, sc.attn_o_v, sh.dsa_o_width, rows, dsa.o_proj, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    /// Non-causal SWA-128 HCA attention for DSpark draft layers (plain RoPE at mla_inv_freq, no compressor).
    pub fn forwardHcaDspark(
        self: *const Runner,
        s: Stream,
        hca: *const weights.HcaWeights,
        kv_ring: u64,
        sc: *const Scratch,
        rows: usize,
        lo: usize,
        hi: usize,
        ring_mask: usize,
    ) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;

        // 1+2. Fused q_down + kv_proj -> RMSNorm(q_down) + RMSNorm/plain-RoPE/store(kv_ring)
        try self.fp8(s, sc.layer_in, c.hidden, rows, hca.qdown_kv, sc.proj_tmp, sc.qmmf_part);
        try k.hcaQkvPrep(s, sc.proj_tmp, hca.q_norm, hca.kv_norm, sc.positions, self.w.mla_inv_freq, sc.q_down, kv_ring, rows, ring_mask, c.eps);

        // 3+4. Fused q_up + o_gate -> weightless per-head Q-RMSNorm + plain GPT-J RoPE + o_gate split
        try self.fp8(s, sc.q_down, c.q_lora, rows, hca.qup_ogate, sc.proj_tmp, sc.qmmf_part);
        try k.hcaQnormRopeSplit(s, sc.proj_tmp, sc.positions, self.w.mla_inv_freq, sc.q_up, sc.o_gate, rows, sh.heads, sh.o_width, c.eps);

        // 5. Non-causal SWA-128 attention over [lo .. hi] + attn_sink (SM121a MMA + Split-K when H_loc==16!)
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
        if (sh.heads == 16 and rows <= 16) {
            try k.hcaAttnDsparkMma(s, sc.q_up, kv_ring, hca.attn_sink, sc.positions, sc.attn_o, sc.attn_split_o, sc.attn_split_m, sc.attn_split_l, rows, c.sliding_window, ring_mask, attnSplits(rows, 8), scale);
        } else {
            try k.hcaAttnDspark(s, sc.q_up, kv_ring, hca.attn_sink, sc.attn_o, rows, sh.heads, lo, hi, ring_mask, scale);
        }

        // 6. VHA post-mix (U^T @ o -> TP AllReduce -> o += V @ z)
        try k.vhaUProjBf16(s, sc.attn_o, hca.vha_u, sc.vha_z, rows, sh.heads, c.head_dim);
        try self.allReduceBf16(s, sc.vha_z, sc.vha_z_all, rows * c.vha_rank * c.head_dim);
        try k.vhaVAddBf16(s, sc.attn_o, hca.vha_v, sc.vha_z, rows, sh.heads, c.head_dim);

        // 7. Inverse plain GPT-J RoPE on dims [448..512)
        try k.ropeGptj(s, sc.attn_o, sc.positions, self.w.mla_inv_freq, rows, sh.heads, c.head_dim, c.qk_nope, -1);

        // 8. Grouped wo_a -> sigmoid gate -> wo_b -> TP AllReduce
        const group_k: usize = (c.heads / c.o_groups) * c.head_dim;
        const ldx: usize = sh.groups * group_k;
        if (rows == 1 or sh.groups == 1) {
            for (0..sh.groups) |g| {
                const x_ptr = sc.attn_o + g * group_k * 2;
                const y_ptr = sc.wo_a_out + g * c.o_lora * 2;
                try self.fp8(s, x_ptr, ldx, rows, hca.wo_a[g], y_ptr, sc.qmmf_part);
            }
        } else {
            for (0..sh.groups) |g| {
                const x_ptr = sc.attn_o + g * group_k * 2;
                const y_ptr = sc.q_up + g * rows * c.o_lora * 2;
                try self.fp8(s, x_ptr, ldx, rows, hca.wo_a[g], y_ptr, sc.qmmf_part);
            }
            try k.interleaveGroups(s, sc.q_up, sc.wo_a_out, sh.groups, rows, c.o_lora);
        }
        try k.sigmoidMulBf16(s, sc.wo_a_out, sc.o_gate, sc.wo_a_out, rows * sh.o_width);
        try self.fp8(s, sc.wo_a_out, sh.o_width, rows, hca.wo_b, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    /// Batched multi-stream non-causal SWA-128 HCA for DSpark (`M` streams x `B=7` rows = `7*M` rows):
    /// reads all draft HCA linear weights ONCE across all `M` concurrent streams!
    pub fn forwardHcaDsparkSegs(
        self: *const Runner,
        s: Stream,
        hca: *const weights.HcaWeights,
        rings: []const u64,
        los: []const usize,
        his: []const usize,
        sc: *const Scratch,
        block_rows: usize,
        ring_mask: usize,
    ) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;
        const m_streams = rings.len;
        const rows = m_streams * block_rows;

        // 1+2. Fused q_down + kv_proj ONCE over all 7*M rows -> per-stream ring store
        try self.fp8(s, sc.layer_in, c.hidden, rows, hca.qdown_kv, sc.proj_tmp, sc.qmmf_part);
        for (0..m_streams) |mi| {
            const row0 = mi * block_rows;
            try k.hcaQkvPrep(s, sc.proj_tmp + row0 * (c.q_lora + c.head_dim) * 2, hca.q_norm, hca.kv_norm, sc.positions + row0 * 4, self.w.mla_inv_freq, sc.q_down + row0 * c.q_lora * 2, rings[mi], block_rows, ring_mask, c.eps);
        }

        // 3+4. Fused q_up + o_gate ONCE over all 7*M rows -> weightless Q-RMSNorm + RoPE + o_gate split
        try self.fp8(s, sc.q_down, c.q_lora, rows, hca.qup_ogate, sc.proj_tmp, sc.qmmf_part);
        try k.hcaQnormRopeSplit(s, sc.proj_tmp, sc.positions, self.w.mla_inv_freq, sc.q_up, sc.o_gate, rows, sh.heads, sh.o_width, c.eps);

        // 5. Per-stream non-causal SWA-128 attention over each stream's ring buffer
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))));
        for (0..m_streams) |mi| {
            const row0 = mi * block_rows;
            const head_off = row0 * sh.heads * 512 * 2;
            if (sh.heads == 16 and block_rows <= 16) {
                try k.hcaAttnDsparkMma(s, sc.q_up + head_off, rings[mi], hca.attn_sink, sc.positions + row0 * 4, sc.attn_o + head_off, sc.attn_split_o, sc.attn_split_m, sc.attn_split_l, block_rows, c.sliding_window, ring_mask, attnSplits(block_rows, 8), scale);
            } else {
                try k.hcaAttnDspark(s, sc.q_up + head_off, rings[mi], hca.attn_sink, sc.attn_o + head_off, block_rows, sh.heads, los[mi], his[mi], ring_mask, scale);
            }
        }

        // 6. VHA post-mix ONCE over all 7*M rows (single F32 AllReduce!)
        try k.vhaUProjBf16(s, sc.attn_o, hca.vha_u, sc.vha_z, rows, sh.heads, c.head_dim);
        try self.allReduceBf16(s, sc.vha_z, sc.vha_z_all, rows * c.vha_rank * c.head_dim);
        try k.vhaVAddBf16(s, sc.attn_o, hca.vha_v, sc.vha_z, rows, sh.heads, c.head_dim);

        // 7. Inverse GPT-J RoPE ONCE over all 7*M rows
        try k.ropeGptj(s, sc.attn_o, sc.positions, self.w.mla_inv_freq, rows, sh.heads, c.head_dim, c.qk_nope, -1);

        // 8. Grouped wo_a ONCE over all 7*M rows
        const group_k: usize = (c.heads / c.o_groups) * c.head_dim;
        const ldx: usize = sh.groups * group_k;
        if (sh.groups == 1) {
            try self.fp8(s, sc.attn_o, ldx, rows, hca.wo_a[0], sc.wo_a_out, sc.qmmf_part);
        } else {
            for (0..sh.groups) |g| {
                const x_ptr = sc.attn_o + g * group_k * 2;
                const y_ptr = sc.q_up + g * rows * c.o_lora * 2;
                try self.fp8(s, x_ptr, ldx, rows, hca.wo_a[g], y_ptr, sc.qmmf_part);
            }
            try k.interleaveGroups(s, sc.q_up, sc.wo_a_out, sh.groups, rows, c.o_lora);
        }

        // 9. Gate * wo_a_out -> wo_b -> single BF16 AllReduce
        try k.sigmoidMulBf16(s, sc.wo_a_out, sc.o_gate, sc.wo_a_out, rows * sh.o_width);
        try self.fp8(s, sc.wo_a_out, sh.o_width, rows, hca.wo_b, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    fn forwardDsa(self: *const Runner, s: Stream, dsa: *const weights.DsaWeights, segs: []const Seg, li: usize, sc: *const Scratch, rows: usize) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;

        // 1+2. Fused q_a + kv_a [rows, 1600] -> q_down + mla_kv store
        try self.fp8(s, sc.layer_in, c.hidden, rows, dsa.qa_kva, sc.proj_tmp, sc.qmmf_part);
        for (segs) |g| try k.dsaQkvPrep(s, sc.proj_tmp + g.row0 * 1600 * 2, dsa.q_a_norm, dsa.kv_a_norm, sc.positions + g.row0 * 4, self.w.mla_inv_freq, sc.q_down + g.row0 * c.mla_q_lora * 2, g.kv.layers[li].dsa.mla_kv, g.n, c.eps);

        // 3. Lightning Indexer: always update idx_k cache! (wk -> LayerNorm -> NeoX RoPE[0..64) -> Hadamard-128 -> store)
        try self.fp8(s, sc.layer_in, c.hidden, rows, dsa.idx_wk, sc.idx_k, sc.qmmf_part);
        try k.layerNorm(s, sc.idx_k, dsa.idx_k_norm_w, dsa.idx_k_norm_b, sc.idx_k, rows, c.index_dim, c.eps);
        try k.ropeNeox(s, sc.idx_k, sc.positions, self.w.mla_inv_freq, rows, 1, c.index_dim, 0);
        try k.hadamard128(s, sc.idx_k, rows);
        for (segs) |g| try k.dsaIdxKStore(s, sc.idx_k + g.row0 * 128 * 2, sc.positions + g.row0 * 4, g.kv.layers[li].dsa.idx_k, g.kv.layers[li].dsa.idx_k_sc, g.n);

        // 4. Score & select top-2048 + SWA-128 ONLY when max_pos + 1 > 2176!
        const dense_width = c.index_topk + c.sliding_window; // 2176
        var any_sparse = false;
        for (segs) |g| any_sparse = any_sparse or g.max_pos + 1 > dense_width;
        if (any_sparse) {
            try self.fp8(s, sc.q_down, c.mla_q_lora, rows, dsa.idx_wq_b, sc.idx_q, sc.qmmf_part);
            try k.ropeNeox(s, sc.idx_q, sc.positions, self.w.mla_inv_freq, rows, c.index_heads, c.index_dim, 0);
            try k.hadamard128(s, sc.idx_q, rows * c.index_heads);
            try self.fp8(s, sc.layer_in, c.hidden, rows, dsa.idx_weights_proj, sc.idx_w, sc.qmmf_part);

            const w_scale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(c.index_dim))) * @sqrt(@as(f64, @floatFromInt(c.index_heads)))));
            const use_fi = (rows >= 16 and dsa.fi_gate != 0 and dsa.fi_bias != 0);
            const fi_gates = sc.qmmf_part; // F32 [rows, 64]
            const fi_q_sel = fi_gates + ((rows * 64 * 4 + 255) & ~@as(usize, 255)); // BF16 [rows, 8, 128]
            const fi_w_sel = fi_q_sel + ((rows * 8 * 128 * 2 + 255) & ~@as(usize, 255)); // F32 [rows, 8]
            if (use_fi) {
                try k.bf16GemmF32Mma(s, sc.layer_in, dsa.fi_gate, fi_gates, rows, 64, c.hidden);
                try k.dsaFlashindexerSelect(s, fi_gates, dsa.fi_bias, sc.idx_q, sc.idx_w, fi_q_sel, fi_w_sel, rows, w_scale);
            }
            // sc.idx_scores holds [score_rows, capacity] fp32: score the rows in slices that fit it (rows sit at
            // consecutive positions ending at max_pos, so a slice's own last position bounds its older tokens).
            const score_rows = @min(sc.max_rows, 64);
            for (segs) |g| {
            if (g.max_pos + 1 <= dense_width) continue;
            var r0: usize = g.row0;
            while (r0 < g.row0 + g.n) : (r0 += score_rows) {
                const b = @min(score_rows, g.row0 + g.n - r0);
                const slice_max = g.max_pos - (g.row0 + g.n - (r0 + b));
                if (slice_max + 1 <= dense_width) continue; // dense rows read 0..pos directly
                const older = if (g.older_ceil > 0) g.older_ceil else (slice_max + 1 - c.sliding_window);
                if (b * older > score_rows * sc.capacity) return error.IndexerScoresTooSmall;
                const pos_at = sc.positions + r0 * 4;
                if (use_fi) {
                    try k.dsaFlashindexerScoresFp8(s, fi_q_sel + r0 * 8 * 128 * 2, g.kv.layers[li].dsa.idx_k, g.kv.layers[li].dsa.idx_k_sc, fi_w_sel + r0 * 8 * 4, pos_at, sc.idx_scores, b, older, c.sliding_window);
                } else {
                    try k.dsaIndexerScoresMma(s, sc.idx_q + r0 * c.index_heads * c.index_dim * 2, g.kv.layers[li].dsa.idx_k, g.kv.layers[li].dsa.idx_k_sc, sc.idx_w + r0 * c.index_heads * 2, pos_at, sc.idx_scores, b, older, c.sliding_window, w_scale);
                }
                try k.dsaSelectTopk(s, sc.idx_scores, pos_at, sc.idx_sel + r0 * dense_width * 4, sc.idx_cnt + r0 * 4, b, older, c.index_topk, c.sliding_window);
            }
            }
        }

        // 5+6. Fused q_b + o_gate -> GPT-J RoPE [192..256) + o_gate split + absorb_q -> sc.q_abs [rows, heads, 576]
        const q_head = c.mla_qk_nope + c.mla_qk_rope;
        try self.fp8(s, sc.q_down, c.mla_q_lora, rows, dsa.qb_ogate, sc.proj_tmp, sc.qmmf_part);
        try k.dsaQbRopeSplit(s, sc.proj_tmp, sc.positions, self.w.mla_inv_freq, sc.q_up, sc.o_gate, rows, sh.heads, sh.dsa_o_width);
        try k.dsaAbsorbQ(s, sc.q_up, dsa.wk, sc.q_abs, rows, sh.heads);

        // 7. MLA-DSA Attention (SM121a Tensor-Core MMA + Split-K when H_loc==16!) -> sc.attn_o -> expand_v
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q_head))));
        for (segs) |g| {
            const sparse = g.max_pos + 1 > dense_width;
            const idx_ptr: u64 = if (sparse) sc.idx_sel + g.row0 * dense_width * 4 else 0;
            const cnt_ptr: u64 = if (sparse) sc.idx_cnt + g.row0 * 4 else 0;
            if (sh.heads == 16) {
                try k.dsaAttnMma(s, sc.q_abs + g.row0 * sh.heads * 576 * 2, g.kv.layers[li].dsa.mla_kv, idx_ptr, cnt_ptr, dsa.sinks, sc.positions + g.row0 * 4, sc.attn_o + g.row0 * sh.heads * 512 * 2, sc.attn_split_o, sc.attn_split_m, sc.attn_split_l, g.n, dense_width, attnSplits(g.n, 12), scale);
            } else {
                try k.dsaAttn(s, sc.q_abs + g.row0 * sh.heads * 576 * 2, g.kv.layers[li].dsa.mla_kv, idx_ptr, cnt_ptr, dsa.sinks, sc.positions + g.row0 * 4, sc.attn_o + g.row0 * sh.heads * 512 * 2, g.n, sh.heads, dense_width, scale);
            }
        }
        try k.dsaExpandV(s, sc.attn_o, dsa.wv, sc.attn_o_v, rows, sh.heads);

        // 8. VHA post-mix (U^T @ o_v -> TP AllReduce -> o_v += V @ z)
        try k.vhaUProjBf16(s, sc.attn_o_v, dsa.vha_u, sc.vha_z, rows, sh.heads, c.mla_v_dim);
        try self.allReduceBf16(s, sc.vha_z, sc.vha_z_all, rows * c.vha_rank * c.mla_v_dim);
        try k.vhaVAddBf16(s, sc.attn_o_v, dsa.vha_v, sc.vha_z, rows, sh.heads, c.mla_v_dim);

        // 9. Gate * o_v -> o_proj -> TP AllReduce -> sc.sub_out [rows, 4096]
        try k.sigmoidMulBf16(s, sc.attn_o_v, sc.o_gate, sc.attn_o_v, rows * sh.dsa_o_width);
        try self.fp8(s, sc.attn_o_v, sh.dsa_o_width, rows, dsa.o_proj, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    fn forwardDenseMlp(self: *const Runner, s: Stream, mlp: *const weights.DenseMlpWeights, sc: *const Scratch, rows: usize) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;

        try self.fp8(s, sc.layer_in, c.hidden, rows, mlp.gate_up, sc.proj_tmp, sc.qmmf_part);
        try k.situAndMulPacked(s, sc.proj_tmp, sc.dense_gate, rows, sh.dense_width, c.situ_beta, c.situ_linear_beta);
        try self.fp8(s, sc.dense_gate, sh.dense_width, rows, mlp.down, sc.sub_out, sc.qmmf_part);
        try self.allReduceBf16(s, sc.sub_out, sc.tp_gather, rows * c.hidden);
    }

    pub fn forwardMoe(self: *const Runner, s: Stream, moe: *const weights.MoeWeights, sc: *const Scratch, rows: usize) !void {
        const c = &self.w.config;
        const sh = &self.w.share;
        const k = self.k;

        // 1. Split-feature router: fused 213 GB/s BF16 GEMV for rows <= 8, qmmf tensor-core BF16 for rows > 8
        if (rows <= 8) {
            try k.routerFusedBf16(s, sc.layer_in, moe.gate_raw0, moe.gate_raw1, sc.router_scores, rows, c.experts, c.hidden);
        } else {
            try self.bf16F32(s, sc.layer_in, c.hidden, rows, moe.gate_packed, sc.router_raw, sc.qmmf_part);
            try k.routerSplitSum(s, sc.router_raw, sc.router_scores, rows, c.experts);
        }
        try k.routerTopk(s, sc.router_scores, moe.bias, moe.routed_scale, sc.topk_ids, sc.topk_w, rows, c.experts, c.topk, c.norm_topk);

        // 2+3. Fused fc1_latent + shared_w1 + shared_w3 -> split latent_in + SITU(shared_w1)*shared_w3
        try self.fp8(s, sc.layer_in, c.hidden, rows, moe.fc1_shared_w13, sc.proj_tmp, sc.qmmf_part);
        try k.moeSplitSitu(s, sc.proj_tmp, sc.latent_in, sc.dense_gate, rows, c.moe_latent, sh.moe_width, c.situ_beta, c.situ_linear_beta);

        const overlap_shared = rows >= 256 and self.group != null and self.group.?.world > 1 and
            self.comm_stream != null and self.ev_shared != null and self.ev_comm != null;
        if (overlap_shared) {
            const cs = self.comm_stream.?;
            // Shared expert down projection first -> overlap its ncclAllReduce on comm_stream with moeGrouped on s!
            try self.fp8(s, sc.dense_gate, sh.moe_width, rows, moe.shared_w2, sc.shared_out, sc.qmmf_part);
            try self.ev_shared.?.record(s);
            try cs.wait(self.ev_shared.?);
            try self.allReduceBf16(cs, sc.shared_out, sc.tp_gather, rows * c.hidden);
            try self.ev_comm.?.record(cs);

            try k.moeGrouped(s, sc.latent_in, moe.w1_q, moe.w1_s, moe.w3_q, moe.w3_s, moe.w2_q, moe.w2_s, sc.topk_ids, sc.topk_w, sc.route_counts, sc.route_offsets, sc.route_cursor, sc.route_list, sc.moe_mid, sc.moe_part, sc.routed_out, rows, c.topk, c.experts, sh.moe_width, c.moe_latent, c.moe_latent, sh.moe_width, c.situ_beta, c.situ_linear_beta);

            try s.wait(self.ev_comm.?);
            try self.allReduceBf16(s, sc.routed_out, sc.tp_gather, rows * c.moe_latent);
        } else {
            // Routed MXFP4 experts (TP partial [rows, 2048])
            if (rows >= grouped_min_rows) {
                try k.moeGrouped(s, sc.latent_in, moe.w1_q, moe.w1_s, moe.w3_q, moe.w3_s, moe.w2_q, moe.w2_s, sc.topk_ids, sc.topk_w, sc.route_counts, sc.route_offsets, sc.route_cursor, sc.route_list, sc.moe_mid, sc.moe_part, sc.routed_out, rows, c.topk, c.experts, sh.moe_width, c.moe_latent, c.moe_latent, sh.moe_width, c.situ_beta, c.situ_linear_beta);
            } else {
                try k.mxfp4W13Situ(s, sc.latent_in, moe.w1_q, moe.w1_s, moe.w3_q, moe.w3_s, sc.topk_ids, sc.moe_mid, rows, c.topk, sh.moe_width, c.moe_latent, c.situ_beta, c.situ_linear_beta);
                try k.mxfp4W2Combine(s, sc.moe_mid, moe.w2_q, moe.w2_s, sc.topk_ids, sc.topk_w, sc.routed_out, rows, c.topk, c.moe_latent, sh.moe_width);
            }

            // Shared expert down projection (TP partial [rows, 4096])
            try self.fp8(s, sc.dense_gate, sh.moe_width, rows, moe.shared_w2, sc.shared_out, sc.qmmf_part);

            // 4. Single packed AllReduce for BOTH [routed_out (2048), shared_out (4096)] = 6144 elements!
            if (self.group) |g| {
                if (g.world > 1) {
                    try k.packPairBf16(s, sc.routed_out, sc.shared_out, sc.moe_pack, rows, c.moe_latent, c.hidden);
                    try self.allReduceBf16(s, sc.moe_pack, sc.tp_gather, rows * (c.moe_latent + c.hidden));
                    try k.unpackPairBf16(s, sc.moe_pack, sc.routed_out, sc.shared_out, rows, c.moe_latent, c.hidden);
                }
            }
        }

        // 5. Post-combine latent_norm -> fc2_latent_proj + shared_out -> sc.sub_out [rows, 4096]
        try k.rmsNorm(s, sc.routed_out, moe.latent_norm, sc.routed_out, rows, c.moe_latent, c.eps);
        try self.fp8(s, sc.routed_out, c.moe_latent, rows, moe.fc2_latent, sc.sub_out, sc.qmmf_part);
        try k.addBf16(s, sc.sub_out, sc.shared_out, sc.sub_out, rows * c.hidden);
    }
};
