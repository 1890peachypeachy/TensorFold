//! ERNIE 5.5 Flash Preview CUDA kernel module loader and typed launchers.
const std = @import("std");
const cuda = @import("cuda");

const Driver = cuda.Driver;
const Stream = cuda.Stream;
const Function = cuda.Function;

fn ci(x: usize) c_int {
    return @intCast(x);
}

fn cdiv(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    ops_mod: cuda.Module,
    mxfp4_mod: cuda.Module,
    hca_mod: cuda.Module,
    dsa_mod: cuda.Module,

    // ops.cu
    embed_broadcast: Function,
    mhc_pre: Function,
    mhc_pre_gemv: Function,
    mhc_pre_finish: Function,
    bf16_linear_v4: Function,
    mhc_post: Function,
    mhc_contract_norm: Function,
    router_logits: Function,
    router_split_sum: Function,
    router_topk: Function,
    concat3_bf16: Function,
    dspark_markov_multimax: Function,
    pick_merge: Function,
    f32_to_bf16: Function,
    put_words: Function,
    situ_and_mul: Function,
    rms_norm: Function,
    layer_norm: Function,
    hca_qnorm_rope: Function,
    rope_gptj: Function,
    rope_neox: Function,
    vha_u_proj: Function,
    vha_v_add: Function,
    hadamard_128: Function,
    sigmoid_mul_bf16: Function,
    add_bf16: Function,
    multimax: Function,
    bf16_linear: Function,
    pack_pair_bf16: Function,
    unpack_pair_bf16: Function,
    rank_sum_bf16: Function,
    rank_sum_f32: Function,
    argmax_f32: Function,
    interleave_groups: Function,
    router_fused_bf16: Function,
    moe_split_situ: Function,
    situ_and_mul_packed: Function,
    hca_qnorm_rope_split: Function,
    dsa_qb_rope_split: Function,
    unpack_vocab_bf16: Function,
    dspark_markov_multimax_batch: Function,
    argmax_pair16: Function,
    pick_merge_batch: Function,
    lm_head_bf16: Function,
    quant_bf16_to_fp8_b128: Function,
    fp8_gemm_b128_mma: Function,
    bf16_gemm_bf16_mma: Function,
    bf16_gemm_f32_mma: Function,
    vha_u_proj_bf16: Function,
    vha_v_add_bf16: Function,
    mhc_pre_mma: Function,

    // mxfp4_situ.cu
    mxfp4_w13_situ: Function,
    mxfp4_w2_combine: Function,
    // expert-major prefill path (rows >= grouped_min_rows)
    mxfp4_route_zero: Function,
    mxfp4_route_count: Function,
    mxfp4_route_scan: Function,
    mxfp4_route_fill: Function,
    mxfp4_w13_mma: Function,
    mxfp4_w2_mma: Function,
    mxfp4_moe_combine: Function,
    dsa_indexer_scores_mma: Function,
    dsa_indexer_scores_fp8_mma: Function,
    dsa_flashindexer_select: Function,
    dsa_flashindexer_scores_fp8_8h_mma: Function,
    dsa_absorb_q_mma: Function,
    dsa_rope_copy: Function,
    dsa_expand_v_mma: Function,
    mhc_pre_tiled: Function,

    // hca_attn.cu
    hca_kv_rope_store: Function,
    hca_comp_save: Function,
    hca_comp_flush: Function,
    hca_attn: Function,
    hca_kv_store_ring: Function,
    hca_attn_dspark: Function,
    hca_qkv_prep: Function,
    hca_comp_save_packed: Function,
    hca_attn_mma: Function,
    hca_attn_prefill_mma: Function,
    hca_attn_dspark_mma: Function,
    attn_splitk_combine: Function,

    // dsa_attn.cu
    dsa_kv_store: Function,
    dsa_kv_norm_rope_store: Function,
    dsa_idx_k_store: Function,
    dsa_absorb_q: Function,
    dsa_expand_v: Function,
    dsa_indexer_scores: Function,
    dsa_select_topk: Function,
    dsa_attn: Function,
    dsa_qkv_prep: Function,
    dsa_attn_mma: Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var ops_mod = try cuda.Module.load(d, cuda.kernels.ernie55_ops);
        errdefer ops_mod.unload();
        var mxfp4_mod = try cuda.Module.load(d, cuda.kernels.ernie55_mxfp4);
        errdefer mxfp4_mod.unload();
        var hca_mod = try cuda.Module.load(d, cuda.kernels.ernie55_hca);
        errdefer hca_mod.unload();
        var dsa_mod = try cuda.Module.load(d, cuda.kernels.ernie55_dsa);
        errdefer dsa_mod.unload();

        var k: Kernels = undefined;
        k.ops_mod = ops_mod;
        k.mxfp4_mod = mxfp4_mod;
        k.hca_mod = hca_mod;
        k.dsa_mod = dsa_mod;

        inline for (@typeInfo(Kernels).@"struct".field_names, @typeInfo(Kernels).@"struct".field_types) |name, T| {
            if (T == Function) {
                const sym = "tf_ernie55_" ++ name;
                const mod = if (comptime std.mem.startsWith(u8, name, "mxfp4_"))
                    mxfp4_mod
                else if (comptime (std.mem.startsWith(u8, name, "hca_") and !std.mem.eql(u8, name, "hca_qnorm_rope") and !std.mem.eql(u8, name, "hca_qnorm_rope_split")) or std.mem.eql(u8, name, "attn_splitk_combine"))
                    hca_mod
                else if (comptime std.mem.startsWith(u8, name, "dsa_") and !std.mem.eql(u8, name, "dsa_qb_rope_split"))
                    dsa_mod
                else
                    ops_mod;
                @field(k, name) = try mod.function(sym);
            }
        }
        try k.mhc_pre.allowDynamicShared(48 * 1024);
        try k.mhc_contract_norm.allowDynamicShared(48 * 1024);
        try k.mxfp4_w2_combine.allowDynamicShared(48 * 1024);
        try k.hca_attn_mma.allowDynamicShared(54 * 1024);
        try k.hca_attn_prefill_mma.allowDynamicShared(92 * 1024);
        try k.hca_attn_dspark_mma.allowDynamicShared(54 * 1024);
        try k.dsa_attn_mma.allowDynamicShared(60 * 1024);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.dsa_mod.unload();
        k.hca_mod.unload();
        k.mxfp4_mod.unload();
        k.ops_mod.unload();
    }

    pub fn embedBroadcast(k: *const Kernels, s: Stream, ids: u64, table: u64, out: u64, rows: usize, hidden: usize, hc_mult: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ ids, table, out, ci(rows), ci(hidden), ci(hc_mult) }) |v| a.add(v);
        try cuda.launch.launch(k.embed_broadcast, .{ .grid = .{ .x = cdiv(hidden, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn mhcPreFast(k: *const Kernels, s: Stream, res: u64, fn_w: u64, mix_raw: u64, sqrsum: u64, scale: u64, base: u64, norm_w: u64, post_mix: u64, comb_mix: u64, layer_in: u64, rows: usize, hidden: usize, rms_eps: f32, hc_eps: f32, post_alpha: f32, iters: u32, norm_eps: f32) !void {
        var a1: cuda.Args = .{};
        inline for (.{ res, fn_w, mix_raw, sqrsum, ci(rows), ci(4 * hidden) }) |v| a1.add(v);
        if (rows >= 64) { // prompt rows: 16 rows x 4-warp K-split BF16 Tensor-Core MMA (ops.cu tf_ernie55_mhc_pre_mma)
            try cuda.launch.launch(k.mhc_pre_mma, .{ .grid = .{ .x = cdiv(rows, 16) }, .block = .{ .x = 128 } }, s, &a1);
        } else {
            try cuda.launch.launch(k.mhc_pre_gemv, .{ .grid = .{ .x = 24, .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a1);
        }
        var a2: cuda.Args = .{};
        inline for (.{ res, mix_raw, sqrsum, scale, base, norm_w, post_mix, comb_mix, layer_in, ci(rows), ci(hidden), rms_eps, hc_eps, post_alpha, @as(c_int, @intCast(iters)), norm_eps }) |v| a2.add(v);
        try cuda.launch.launch(k.mhc_pre_finish, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .shared = @intCast(hidden * 2 + 256) }, s, &a2);
    }

    pub fn mhcPre(k: *const Kernels, s: Stream, res: u64, fn_w: u64, scale: u64, base: u64, norm_w: u64, post_mix: u64, comb_mix: u64, layer_in: u64, rows: usize, hidden: usize, rms_eps: f32, hc_eps: f32, post_alpha: f32, iters: u32, norm_eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ res, fn_w, scale, base, norm_w, post_mix, comb_mix, layer_in, ci(rows), ci(hidden), rms_eps, hc_eps, post_alpha, @as(c_int, @intCast(iters)), norm_eps }) |v| a.add(v);
        try cuda.launch.launch(k.mhc_pre, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .shared = @intCast(4 * hidden * 2 + 256) }, s, &a);
    }

    pub fn mhcPost(k: *const Kernels, s: Stream, x: u64, res: u64, post_mix: u64, comb_mix: u64, out: u64, aux_mean: u64, rows: usize, hidden: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, res, post_mix, comb_mix, out, aux_mean, ci(rows), ci(hidden) }) |v| a.add(v);
        try cuda.launch.launch(k.mhc_post, .{ .grid = .{ .x = cdiv(hidden, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn mhcContractNorm(k: *const Kernels, s: Stream, res: u64, fn_w: u64, scale: u64, base: u64, norm_w: u64, out: u64, rows: usize, hidden: usize, rms_eps: f32, hc_eps: f32, norm_eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ res, fn_w, scale, base, norm_w, out, ci(rows), ci(hidden), rms_eps, hc_eps, norm_eps }) |v| a.add(v);
        try cuda.launch.launch(k.mhc_contract_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .shared = @intCast(4 * hidden * 2 + 256) }, s, &a);
    }

    pub fn routerLogits(k: *const Kernels, s: Stream, x: u64, w0: u64, w1: u64, scores: u64, rows: usize, experts: usize, hidden: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w0, w1, scores, ci(rows), ci(experts), ci(hidden) }) |v| a.add(v);
        try cuda.launch.launch(k.router_logits, .{ .grid = .{ .x = cdiv(experts, 4), .y = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn routerTopk(k: *const Kernels, s: Stream, scores: u64, bias: u64, scale: u64, out_ids: u64, out_w: u64, rows: usize, experts: usize, topk: usize, norm_topk: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ scores, bias, scale, out_ids, out_w, ci(rows), ci(experts), ci(topk), @as(c_int, @intFromBool(norm_topk)) }) |v| a.add(v);
        try cuda.launch.launch(k.router_topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, s, &a);
    }

    pub fn situAndMul(k: *const Kernels, s: Stream, gate: u64, up: u64, out: u64, count: usize, beta: f32, linear_beta: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ gate, up, out, @as(i64, @intCast(count)), beta, linear_beta }) |v| a.add(v);
        try cuda.launch.launch(k.situ_and_mul, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn rmsNorm(k: *const Kernels, s: Stream, x: u64, w: u64, out: u64, rows: usize, dim: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w, out, ci(rows), ci(dim), eps }) |v| a.add(v);
        try cuda.launch.launch(k.rms_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn layerNorm(k: *const Kernels, s: Stream, x: u64, w: u64, b: u64, out: u64, rows: usize, dim: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w, b, out, ci(rows), ci(dim), eps }) |v| a.add(v);
        try cuda.launch.launch(k.layer_norm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn hcaQnormRope(k: *const Kernels, s: Stream, q: u64, positions: u64, inv_freq: u64, rows: usize, heads: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, positions, inv_freq, ci(rows), ci(heads), eps }) |v| a.add(v);
        try cuda.launch.launch(k.hca_qnorm_rope, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn ropeGptj(k: *const Kernels, s: Stream, x: u64, positions: u64, inv_freq: u64, rows: usize, heads: usize, head_dim: usize, rope_offset: usize, dir: i32) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, positions, inv_freq, ci(rows), ci(heads), ci(head_dim), ci(rope_offset), @as(c_int, dir) }) |v| a.add(v);
        try cuda.launch.launch(k.rope_gptj, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(rows) }, .block = .{ .x = 32 } }, s, &a);
    }

    pub fn ropeNeox(k: *const Kernels, s: Stream, x: u64, positions: u64, inv_freq: u64, rows: usize, heads: usize, head_dim: usize, rope_offset: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, positions, inv_freq, ci(rows), ci(heads), ci(head_dim), ci(rope_offset) }) |v| a.add(v);
        try cuda.launch.launch(k.rope_neox, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(rows) }, .block = .{ .x = 32 } }, s, &a);
    }

    pub fn vhaUProj(k: *const Kernels, s: Stream, o: u64, u: u64, z: u64, rows: usize, h_loc: usize, head_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ o, u, z, ci(rows), ci(h_loc), ci(head_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.vha_u_proj, .{ .grid = .{ .x = cdiv(head_dim, 32), .y = @intCast(rows) }, .block = .{ .x = 32, .y = 16 } }, s, &a);
    }

    pub fn vhaVAdd(k: *const Kernels, s: Stream, o: u64, v: u64, z: u64, rows: usize, h_loc: usize, head_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ o, v, z, ci(rows), ci(h_loc), ci(head_dim) }) |val| a.add(val);
        try cuda.launch.launch(k.vha_v_add, .{ .grid = .{ .x = cdiv(head_dim, 32), .y = @intCast(rows) }, .block = .{ .x = 32, .y = 16 } }, s, &a);
    }

    pub fn hadamard128(k: *const Kernels, s: Stream, x: u64, num_rows: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, ci(num_rows) }) |v| a.add(v);
        try cuda.launch.launch(k.hadamard_128, .{ .grid = .{ .x = cdiv(num_rows, 4) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn sigmoidMulBf16(k: *const Kernels, s: Stream, x: u64, gate: u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, gate, out, @as(i64, @intCast(count)) }) |v| a.add(v);
        try cuda.launch.launch(k.sigmoid_mul_bf16, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn addBf16(k: *const Kernels, s: Stream, a_ptr: u64, b_ptr: u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ a_ptr, b_ptr, out, @as(i64, @intCast(count)) }) |v| a.add(v);
        try cuda.launch.launch(k.add_bf16, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn applyMultimax(k: *const Kernels, s: Stream, logits: u64, ranges: u64, ts: u64, count: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ logits, ranges, ts, @as(i64, @intCast(count)) }) |v| a.add(v);
        try cuda.launch.launch(k.multimax, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn bf16Linear(k: *const Kernels, s: Stream, x: u64, w: u64, out_bf16: u64, out_f32: u64, rows: usize, n: usize, k_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w, out_bf16, out_f32, ci(rows), ci(n), ci(k_dim) }) |v| a.add(v);
        try cuda.launch.launch(if (k_dim % 8 == 0) k.bf16_linear_v4 else k.bf16_linear, .{ .grid = .{ .x = cdiv(n, 4), .y = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn packPairBf16(k: *const Kernels, s: Stream, a_ptr: u64, b_ptr: u64, dst: u64, rows: usize, wa: usize, wb: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ a_ptr, b_ptr, dst, ci(rows), ci(wa), ci(wb) }) |v| a.add(v);
        try cuda.launch.launch(k.pack_pair_bf16, .{ .grid = .{ .x = cdiv(wa + wb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn unpackPairBf16(k: *const Kernels, s: Stream, src: u64, a_ptr: u64, b_ptr: u64, rows: usize, wa: usize, wb: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ src, a_ptr, b_ptr, ci(rows), ci(wa), ci(wb) }) |v| a.add(v);
        try cuda.launch.launch(k.unpack_pair_bf16, .{ .grid = .{ .x = cdiv(wa + wb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn rankSumBf16(k: *const Kernels, s: Stream, gathered: u64, out: u64, count: usize, world: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ gathered, out, @as(i64, @intCast(count)), ci(world) }) |v| a.add(v);
        try cuda.launch.launch(k.rank_sum_bf16, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn rankSumF32(k: *const Kernels, s: Stream, gathered: u64, out: u64, count: usize, world: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ gathered, out, @as(i64, @intCast(count)), ci(world) }) |v| a.add(v);
        try cuda.launch.launch(k.rank_sum_f32, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn argmaxF32(k: *const Kernels, s: Stream, logits: u64, out_id: u64, out_val: u64, rows: usize, vocab_loc: usize, vocab_lo: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ logits, out_id, out_val, ci(rows), ci(vocab_loc), ci(vocab_lo) }) |v| a.add(v);
        try cuda.launch.launch(k.argmax_f32, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn interleaveGroups(k: *const Kernels, s: Stream, src: u64, dst: u64, groups: usize, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ src, dst, ci(groups), ci(rows), ci(width) }) |v| a.add(v);
        try cuda.launch.launch(k.interleave_groups, .{ .grid = .{ .x = cdiv(width, 256), .y = @intCast(rows), .z = @intCast(groups) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn mxfp4W13Situ(k: *const Kernels, s: Stream, x: u64, w1_q: u64, w1_s: u64, w3_q: u64, w3_s: u64, ids: u64, out: u64, rows: usize, topk: usize, n_loc: usize, k_dim: usize, beta: f32, linear_beta: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w1_q, w1_s, w3_q, w3_s, ids, out, ci(rows), ci(topk), ci(n_loc), ci(k_dim), beta, linear_beta }) |v| a.add(v);
        try cuda.launch.launch(k.mxfp4_w13_situ, .{ .grid = .{ .x = cdiv(n_loc, 4), .y = @intCast(topk), .z = @intCast(rows) }, .block = .{ .x = 128 }, .shared = @intCast(k_dim * 2) }, s, &a);
    }

    /// Routed MoE as per-expert tensor-core GEMMs (mxfp4_situ.cu's expert-major path): picks sorted by expert,
    /// w13 + SITU into `mid`, w2 into `part` (fp32 per pick), then the in-order top-k combine into `out`.
    pub fn moeGrouped(k: *const Kernels, s: Stream, x: u64, w1_q: u64, w1_s: u64, w3_q: u64, w3_s: u64, w2_q: u64, w2_s: u64, ids: u64, weights: u64, counts: u64, offsets: u64, cursor: u64, list: u64, mid: u64, part: u64, out: u64, rows: usize, topk: usize, experts: usize, n_loc: usize, k_dim: usize, n_out: usize, k_loc: usize, beta: f32, linear_beta: f32) !void {
        const pairs = rows * topk;
        {
            var a: cuda.Args = .{};
            inline for (.{ counts, ci(experts) }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_route_zero, .{ .grid = .{ .x = cdiv(experts, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            inline for (.{ ids, ci(pairs), counts }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_route_count, .{ .grid = .{ .x = cdiv(pairs, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            inline for (.{ counts, offsets, cursor, ci(experts) }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_route_scan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            inline for (.{ ids, ci(pairs), cursor, list }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_route_fill, .{ .grid = .{ .x = cdiv(pairs, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        const gz: u32 = if (rows >= 256) @min(@as(u32, 8), @as(u32, @intCast((pairs + experts * 64 - 1) / (experts * 64) + 3))) else 1;
        {
            var a: cuda.Args = .{};
            inline for (.{ x, w1_q, w1_s, w3_q, w3_s, offsets, list, mid, ci(topk), ci(n_loc), ci(k_dim), beta, linear_beta }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_w13_mma, .{ .grid = .{ .x = cdiv(n_loc, 64), .y = @intCast(experts), .z = gz }, .block = .{ .x = 128 }, .shared = 0 }, s, &a);
        }
        try s.d.check(s.d.api.cuMemsetD32Async(part, 0, rows * n_out, s.handle), "cuMemsetD32Async");
        {
            var a: cuda.Args = .{};
            inline for (.{ mid, w2_q, w2_s, offsets, list, weights, part, ci(topk), ci(n_out), ci(k_loc) }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_w2_mma, .{ .grid = .{ .x = cdiv(n_out, 64), .y = @intCast(experts), .z = gz }, .block = .{ .x = 128 }, .shared = 0 }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            inline for (.{ part, weights, out, ci(topk), ci(n_out) }) |v| a.add(v);
            try cuda.launch.launch(k.mxfp4_moe_combine, .{ .grid = .{ .x = cdiv(n_out, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
        }
    }

    pub fn mxfp4W2Combine(k: *const Kernels, s: Stream, mid: u64, w2_q: u64, w2_s: u64, ids: u64, weights: u64, out: u64, rows: usize, topk: usize, n_out: usize, k_loc: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ mid, w2_q, w2_s, ids, weights, out, ci(rows), ci(topk), ci(n_out), ci(k_loc) }) |v| a.add(v);
        try cuda.launch.launch(k.mxfp4_w2_combine, .{ .grid = .{ .x = cdiv(n_out, 4), .y = @intCast(rows) }, .block = .{ .x = 128 }, .shared = @intCast(topk * k_loc * 2) }, s, &a);
    }

    pub fn hcaKvRopeStore(k: *const Kernels, s: Stream, kv: u64, positions: u64, inv_freq: u64, swa_kv: u64, rows: usize, swa_mask: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ kv, positions, inv_freq, swa_kv, ci(rows), ci(swa_mask) }) |v| a.add(v);
        try cuda.launch.launch(k.hca_kv_rope_store, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaCompSave(k: *const Kernels, s: Stream, wkv: u64, wgate: u64, ape: u64, positions: u64, comp_state: u64, rows: usize, state_mask: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ wkv, wgate, ape, positions, comp_state, ci(rows), ci(state_mask) }) |v| a.add(v);
        try cuda.launch.launch(k.hca_comp_save, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaCompFlush(k: *const Kernels, s: Stream, comp_state: u64, norm_w: u64, positions: u64, inv_freq: u64, comp_kv: u64, rows: usize, state_mask: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ comp_state, norm_w, positions, inv_freq, comp_kv, ci(rows), ci(state_mask), eps }) |v| a.add(v);
        try cuda.launch.launch(k.hca_comp_flush, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaAttn(k: *const Kernels, s: Stream, q: u64, swa_kv: u64, comp_kv: u64, sink: u64, positions: u64, out: u64, rows: usize, h_loc: usize, swa_mask: usize, window: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, swa_kv, comp_kv, sink, positions, out, ci(rows), ci(h_loc), ci(swa_mask), ci(window), scale }) |v| a.add(v);
        try cuda.launch.launch(k.hca_attn, .{ .grid = .{ .x = cdiv(h_loc, 16), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn dsaKvNormRopeStore(k: *const Kernels, s: Stream, kv_a: u64, norm_w: u64, positions: u64, inv_freq: u64, mla_kv: u64, rows: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ kv_a, norm_w, positions, inv_freq, mla_kv, ci(rows), eps }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_kv_norm_rope_store, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn dsaIdxKStore(k: *const Kernels, s: Stream, k_idx: u64, positions: u64, idx_k: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ k_idx, positions, idx_k, ci(rows) }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_idx_k_store, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn dsaAbsorbQ(k: *const Kernels, s: Stream, q: u64, wk: u64, q_abs: u64, rows: usize, h_loc: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, wk, q_abs, ci(rows), ci(h_loc) }) |v| a.add(v);
        // per-head tensor-core GEMM (dsa_attn.cu tf_ernie55_dsa_absorb_q_mma: bf16 outputs equal the per-warp kernel's)
        try cuda.launch.launch(k.dsa_absorb_q_mma, .{ .grid = .{ .x = 512 / 64, .y = @intCast(h_loc), .z = cdiv(rows, 64) }, .block = .{ .x = 128 } }, s, &a);
        var rc: cuda.Args = .{};
        inline for (.{ q, q_abs, ci(rows), ci(h_loc) }) |v| rc.add(v);
        try cuda.launch.launch(k.dsa_rope_copy, .{ .grid = .{ .x = @intCast(h_loc), .y = @intCast(rows) }, .block = .{ .x = 64 } }, s, &rc);
    }

    pub fn dsaExpandV(k: *const Kernels, s: Stream, o_lat: u64, wv: u64, o_v: u64, rows: usize, h_loc: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ o_lat, wv, o_v, ci(rows), ci(h_loc) }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_expand_v_mma, .{ .grid = .{ .x = 256 / 64, .y = @intCast(h_loc), .z = cdiv(rows, 64) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn dsaIndexerScores(k: *const Kernels, s: Stream, q_idx: u64, idx_k: u64, weights: u64, positions: u64, scores: u64, rows: usize, max_older: usize, window: usize, w_scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q_idx, idx_k, weights, positions, scores, ci(rows), ci(max_older), ci(window), w_scale }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_indexer_scores, .{ .grid = .{ .x = cdiv(max_older, 4), .y = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// dsaIndexerScores on tensor cores (dsa_attn.cu): one block a query row x 64 key tokens.
    pub fn dsaFlashindexerSelect(k: *const Kernels, s: Stream, gates: u64, bias: u64, idx_q: u64, idx_w: u64, q_sel: u64, w_sel: u64, rows: usize, weights_scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ gates, bias, idx_q, idx_w, q_sel, w_sel, ci(rows), weights_scale }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_flashindexer_select, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn dsaFlashindexerScoresFp8(k: *const Kernels, s: Stream, q_sel: u64, idx_k: u64, w_sel: u64, positions: u64, scores: u64, rows: usize, max_older: usize, window_size: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ q_sel, idx_k, w_sel, positions, scores, ci(rows), ci(max_older), ci(window_size) }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_flashindexer_scores_fp8_8h_mma, .{ .grid = .{ .x = cdiv(max_older, 64), .y = cdiv(rows, 8) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn dsaIndexerScoresMma(k: *const Kernels, s: Stream, q_idx: u64, idx_k: u64, weights: u64, positions: u64, scores: u64, rows: usize, max_older: usize, window: usize, w_scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q_idx, idx_k, weights, positions, scores, ci(rows), ci(max_older), ci(window), w_scale }) |v| a.add(v);
        if (rows >= 4) {
            try cuda.launch.launch(k.dsa_indexer_scores_fp8_mma, .{ .grid = .{ .x = cdiv(max_older, 64), .y = cdiv(rows, 4) }, .block = .{ .x = 128 } }, s, &a);
            return;
        }
        try cuda.launch.launch(k.dsa_indexer_scores_mma, .{ .grid = .{ .x = cdiv(max_older, 64), .y = @intCast(rows) }, .block = .{ .x = 128 }, .shared = 128 * 136 * 2 + 4 * 64 * 4 }, s, &a);
    }

    pub fn dsaSelectTopk(k: *const Kernels, s: Stream, scores: u64, positions: u64, out_idx: u64, out_cnt: u64, rows: usize, max_older: usize, topk: usize, window: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ scores, positions, out_idx, out_cnt, ci(rows), ci(max_older), ci(topk), ci(window) }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_select_topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn dsaAttn(k: *const Kernels, s: Stream, q_abs: u64, mla_kv: u64, indices: u64, counts: u64, sinks: u64, positions: u64, out_lat: u64, rows: usize, h_loc: usize, idx_stride: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q_abs, mla_kv, indices, counts, sinks, positions, out_lat, ci(rows), ci(h_loc), ci(idx_stride), scale }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_attn, .{ .grid = .{ .x = cdiv(h_loc, 16), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn routerSplitSum(k: *const Kernels, s: Stream, raw: u64, scores: u64, rows: usize, experts: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ raw, scores, ci(rows), ci(experts) }) |v| a.add(v);
        try cuda.launch.launch(k.router_split_sum, .{ .grid = .{ .x = cdiv(rows * experts, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn concat3Bf16(k: *const Kernels, s: Stream, a0: u64, b0: u64, c0: u64, out: u64, rows: usize, dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ a0, b0, c0, out, ci(rows), ci(dim) }) |v| a.add(v);
        try cuda.launch.launch(k.concat3_bf16, .{ .grid = .{ .x = cdiv(rows * dim, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn dsparkMarkovMultimax(k: *const Kernels, s: Stream, logits: u64, w2: u64, w1: u64, walk_ptr: u64, scores: u64, ranges: u64, ts: u64, nv: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ logits, w2, w1, walk_ptr, scores, ranges, ts, ci(nv) }) |v| a.add(v);
        try cuda.launch.launch(k.dspark_markov_multimax, .{ .grid = .{ .x = cdiv(nv, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn pickMerge(k: *const Kernels, s: Stream, picks: u64, world: usize, stride_bytes: usize, walk: u64, step: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ picks, ci(world), ci(stride_bytes), walk, ci(step) }) |v| a.add(v);
        try cuda.launch.launch(k.pick_merge, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, s, &a);
    }

    pub fn hcaKvStoreRing(k: *const Kernels, s: Stream, kv_in: u64, kv_norm: u64, positions: u64, inv_freq: u64, kv_ring: u64, rows: usize, ring_mask: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ kv_in, kv_norm, positions, inv_freq, kv_ring, ci(rows), ci(ring_mask), eps }) |v| a.add(v);
        try cuda.launch.launch(k.hca_kv_store_ring, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaAttnDspark(k: *const Kernels, s: Stream, q: u64, kv_ring: u64, sink: u64, out: u64, rows: usize, h_loc: usize, lo: usize, hi: usize, ring_mask: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, kv_ring, sink, out, ci(rows), ci(h_loc), ci(lo), ci(hi), ci(ring_mask), scale }) |v| a.add(v);
        try cuda.launch.launch(k.hca_attn_dspark, .{ .grid = .{ .x = cdiv(h_loc, 4), .y = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn f32ToBf16(k: *const Kernels, s: Stream, src: u64, dst: u64, count: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ src, dst, @as(i64, @intCast(count)) }) |v| a.add(v);
        try cuda.launch.launch(k.f32_to_bf16, .{ .grid = .{ .x = cdiv(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn routerFusedBf16(k: *const Kernels, s: Stream, x: u64, w0: u64, w1: u64, scores: u64, rows: usize, experts: usize, hidden: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w0, w1, scores, ci(rows), ci(experts), ci(hidden) }) |v| a.add(v);
        try cuda.launch.launch(k.router_fused_bf16, .{ .grid = .{ .x = cdiv(experts, 8) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn moeSplitSitu(k: *const Kernels, s: Stream, packed_in: u64, latent_out: u64, shared_mid: u64, rows: usize, latent: usize, w: usize, beta: f32, linear_beta: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, latent_out, shared_mid, ci(rows), ci(latent), ci(w), beta, linear_beta }) |v| a.add(v);
        try cuda.launch.launch(k.moe_split_situ, .{ .grid = .{ .x = cdiv(latent + w, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn situAndMulPacked(k: *const Kernels, s: Stream, packed_in: u64, out: u64, rows: usize, w: usize, beta: f32, linear_beta: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, out, ci(rows), ci(w), beta, linear_beta }) |v| a.add(v);
        try cuda.launch.launch(k.situ_and_mul_packed, .{ .grid = .{ .x = cdiv(w, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaQnormRopeSplit(k: *const Kernels, s: Stream, packed_in: u64, positions: u64, inv_freq: u64, q_out: u64, o_gate: u64, rows: usize, heads: usize, o_width: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, positions, inv_freq, q_out, o_gate, ci(rows), ci(heads), ci(o_width), eps }) |v| a.add(v);
        try cuda.launch.launch(k.hca_qnorm_rope_split, .{ .grid = .{ .x = @intCast(heads + (o_width + 511) / 512), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn dsaQbRopeSplit(k: *const Kernels, s: Stream, packed_in: u64, positions: u64, inv_freq: u64, q_out: u64, o_gate: u64, rows: usize, heads: usize, o_width: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, positions, inv_freq, q_out, o_gate, ci(rows), ci(heads), ci(o_width) }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_qb_rope_split, .{ .grid = .{ .x = @intCast(heads + (o_width + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn unpackVocabBf16(k: *const Kernels, s: Stream, gathered: u64, full: u64, world: usize, rows: usize, vw: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ gathered, full, ci(world), ci(rows), ci(vw) }) |v| a.add(v);
        try cuda.launch.launch(k.unpack_vocab_bf16, .{ .grid = .{ .x = cdiv(world * vw, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaQkvPrep(k: *const Kernels, s: Stream, packed_in: u64, q_norm: u64, kv_norm: u64, positions: u64, inv_freq: u64, q_down: u64, swa_kv: u64, rows: usize, swa_mask: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, q_norm, kv_norm, positions, inv_freq, q_down, swa_kv, ci(rows), ci(swa_mask), eps }) |v| a.add(v);
        try cuda.launch.launch(k.hca_qkv_prep, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaCompSavePacked(k: *const Kernels, s: Stream, packed_in: u64, ape: u64, positions: u64, comp_state: u64, rows: usize, state_mask: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, ape, positions, comp_state, ci(rows), ci(state_mask) }) |v| a.add(v);
        try cuda.launch.launch(k.hca_comp_save_packed, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn dsaQkvPrep(k: *const Kernels, s: Stream, packed_in: u64, q_a_norm: u64, kv_a_norm: u64, positions: u64, inv_freq: u64, q_down: u64, mla_kv: u64, rows: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ packed_in, q_a_norm, kv_a_norm, positions, inv_freq, q_down, mla_kv, ci(rows), eps }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_qkv_prep, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn hcaAttnPrefillMma(k: *const Kernels, s: Stream, q: u64, swa_kv: u64, comp_kv: u64, sink: u64, positions: u64, out: u64, rows: usize, swa_mask: usize, window_size: usize, scale: f32) !void {
        const smem: u32 = (4 * 16 * 520 + 16 * 520) * 2 + 4 * 16 * 16 * 6 + 4 * 16 * 12;
        var a: cuda.Args = .{};
        inline for (.{ q, swa_kv, comp_kv, sink, positions, out, ci(rows), ci(swa_mask), ci(window_size), scale }) |v| a.add(v);
        try cuda.launch.launch(k.hca_attn_prefill_mma, .{ .grid = .{ .x = cdiv(rows, 4) }, .block = .{ .x = 512 }, .shared = smem }, s, &a);
    }

    pub fn hcaAttnMma(k: *const Kernels, s: Stream, q: u64, swa_kv: u64, comp_kv: u64, sink: u64, positions: u64, out: u64, opart: u64, mpart: u64, lpart: u64, rows: usize, swa_mask: usize, window_size: usize, splits: usize, scale: f32) !void {
        const smem: u32 = (16 * 520 + 32 * 520) * 2 + 16 * 32 * 4 + 16 * 32 * 2 + 192;
        var a: cuda.Args = .{};
        inline for (.{ q, swa_kv, comp_kv, sink, positions, out, opart, mpart, lpart, ci(rows), ci(swa_mask), ci(window_size), ci(splits), scale }) |v| a.add(v);
        try cuda.launch.launch(k.hca_attn_mma, .{ .grid = .{ .x = @intCast(splits), .y = @intCast(rows) }, .block = .{ .x = 128 }, .shared = smem }, s, &a);
        if (splits > 1) {
            var ac: cuda.Args = .{};
            inline for (.{ opart, mpart, lpart, out, ci(rows), ci(splits) }) |v| ac.add(v);
            try cuda.launch.launch(k.attn_splitk_combine, .{ .grid = .{ .x = 16, .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &ac);
        }
    }

    pub fn hcaAttnDsparkMma(k: *const Kernels, s: Stream, q: u64, kv_ring: u64, sink: u64, positions: u64, out: u64, opart: u64, mpart: u64, lpart: u64, rows: usize, window_size: usize, ring_mask: usize, splits: usize, scale: f32) !void {
        const smem: u32 = (16 * 520 + 32 * 520) * 2 + 16 * 32 * 4 + 16 * 32 * 2 + 192;
        var a: cuda.Args = .{};
        inline for (.{ q, kv_ring, sink, positions, out, opart, mpart, lpart, ci(rows), ci(window_size), ci(ring_mask), ci(splits), scale }) |v| a.add(v);
        try cuda.launch.launch(k.hca_attn_dspark_mma, .{ .grid = .{ .x = @intCast(splits), .y = @intCast(rows) }, .block = .{ .x = 128 }, .shared = smem }, s, &a);
        if (splits > 1) {
            var ac: cuda.Args = .{};
            inline for (.{ opart, mpart, lpart, out, ci(rows), ci(splits) }) |v| ac.add(v);
            try cuda.launch.launch(k.attn_splitk_combine, .{ .grid = .{ .x = 16, .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &ac);
        }
    }

    pub fn dsaAttnMma(k: *const Kernels, s: Stream, q_abs: u64, mla_kv: u64, indices: u64, counts: u64, sinks: u64, positions: u64, out_lat: u64, opart: u64, mpart: u64, lpart: u64, rows: usize, idx_stride: usize, splits: usize, scale: f32) !void {
        const smem: u32 = (16 * 584 + 32 * 584) * 2 + 16 * 32 * 4 + 16 * 32 * 2 + 192;
        var a: cuda.Args = .{};
        inline for (.{ q_abs, mla_kv, indices, counts, sinks, positions, out_lat, opart, mpart, lpart, ci(rows), ci(idx_stride), ci(splits), scale }) |v| a.add(v);
        try cuda.launch.launch(k.dsa_attn_mma, .{ .grid = .{ .x = @intCast(splits), .y = @intCast(rows) }, .block = .{ .x = 128 }, .shared = smem }, s, &a);
        if (splits > 1) {
            var ac: cuda.Args = .{};
            inline for (.{ opart, mpart, lpart, out_lat, ci(rows), ci(splits) }) |v| ac.add(v);
            try cuda.launch.launch(k.attn_splitk_combine, .{ .grid = .{ .x = 16, .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &ac);
        }
    }

    pub fn dsparkMarkovMultimaxBatch(k: *const Kernels, s: Stream, logits: u64, w2: u64, w1: u64, walk_ptr: u64, scores: u64, ranges: u64, ts: u64, m: usize, step: usize, nv: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ logits, w2, w1, walk_ptr, scores, ranges, ts, ci(m), ci(step), ci(nv) }) |v| a.add(v);
        try cuda.launch.launch(k.dspark_markov_multimax_batch, .{ .grid = .{ .x = cdiv(nv, 256), .y = @intCast(m) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn argmaxPair16(k: *const Kernels, s: Stream, scores: u64, out_pairs: u64, rows: usize, n: usize, id_offset: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ scores, out_pairs, ci(rows), ci(n), ci(id_offset) }) |v| a.add(v);
        try cuda.launch.launch(k.argmax_pair16, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn pickMergeBatch(k: *const Kernels, s: Stream, picks: u64, world: usize, m: usize, walk: u64, step: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ picks, ci(world), ci(m), walk, ci(step) }) |v| a.add(v);
        try cuda.launch.launch(k.pick_merge_batch, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1 } }, s, &a);
    }

    pub fn lmHeadBf16(k: *const Kernels, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, hidden: usize) !void {
        var r0: usize = 0;
        while (r0 < rows) {
            const b = @min(@as(usize, 8), rows - r0);
            var a: cuda.Args = .{};
            inline for (.{ x + r0 * hidden * 2, w, out + r0 * n * 4, ci(b), ci(n), ci(hidden) }) |v| a.add(v);
            try cuda.launch.launch(k.lm_head_bf16, .{ .grid = .{ .x = cdiv(n, 8) }, .block = .{ .x = 256 } }, s, &a);
            r0 += b;
        }
    }

    pub fn vhaUProjBf16(k: *const Kernels, s: Stream, o: u64, u: u64, z: u64, rows: usize, h_loc: usize, head_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ o, u, z, ci(rows), ci(h_loc), ci(head_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.vha_u_proj_bf16, .{ .grid = .{ .x = cdiv(head_dim, 32), .y = @intCast(rows) }, .block = .{ .x = 32, .y = 16 } }, s, &a);
    }

    pub fn vhaVAddBf16(k: *const Kernels, s: Stream, o: u64, v_mat: u64, z: u64, rows: usize, h_loc: usize, head_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ o, v_mat, z, ci(rows), ci(h_loc), ci(head_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.vha_v_add_bf16, .{ .grid = .{ .x = cdiv(head_dim, 32), .y = @intCast(rows) }, .block = .{ .x = 32, .y = @intCast(@min(h_loc, 16)) } }, s, &a);
    }

    pub fn quantBf16ToFp8B128(k: *const Kernels, s: Stream, x: u64, ldx: usize, out_q: u64, out_s: u64, m: usize, k_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, ci(ldx), out_q, out_s, ci(m), ci(k_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.quant_bf16_to_fp8_b128, .{ .grid = .{ .x = @intCast(k_dim / 128), .y = @intCast(m) }, .block = .{ .x = 32 } }, s, &a);
    }

    pub fn fp8GemmB128Mma(k: *const Kernels, s: Stream, xq: u64, sx: u64, wq: u64, sw: u64, y: u64, m: usize, n: usize, k_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ xq, sx, wq, sw, y, ci(m), ci(n), ci(k_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.fp8_gemm_b128_mma, .{ .grid = .{ .x = cdiv(n, 128), .y = cdiv(m, 64) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn bf16GemmBf16Mma(k: *const Kernels, s: Stream, x: u64, w: u64, y: u64, m: usize, n: usize, k_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w, y, ci(m), ci(n), ci(k_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.bf16_gemm_bf16_mma, .{ .grid = .{ .x = cdiv(n, 64), .y = cdiv(m, 64) }, .block = .{ .x = 128 } }, s, &a);
    }

    pub fn bf16GemmF32Mma(k: *const Kernels, s: Stream, x: u64, w: u64, y: u64, m: usize, n: usize, k_dim: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, w, y, ci(m), ci(n), ci(k_dim) }) |v| a.add(v);
        try cuda.launch.launch(k.bf16_gemm_f32_mma, .{ .grid = .{ .x = cdiv(n, 64), .y = cdiv(m, 64) }, .block = .{ .x = 128 } }, s, &a);
    }
};
