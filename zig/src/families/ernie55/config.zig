//! ERNIE 5.5 Flash Preview (`model_type: "ernie5_5_moe"`) architecture configuration.
const std = @import("std");

pub const max_layers = 64;
pub const max_dspark_targets = 8;

pub const AttnKind = enum(u8) { hca, dsa };

pub const RopeScaling = struct {
    factor: f64 = 16.0,
    original_max_positions: u32 = 65536,
    beta_fast: f64 = 32.0,
    beta_slow: f64 = 1.0,
    yarn: bool = true,
};

pub const Config = struct {
    vocab: u32 = 201216,
    hidden: u32 = 4096,
    layers: u32 = 43,
    first_k_dense: u32 = 1,
    eps: f32 = 1e-6,

    // mHC hyper-connections
    hc_mult: u32 = 4,
    hc_sinkhorn_iters: u32 = 20,
    hc_eps: f32 = 1e-6,
    hc_post_alpha: f32 = 2.0,

    // HCA attention (compress_ratios[i] > 0, e.g. 128)
    heads: u32 = 64,
    kv_heads: u32 = 1,
    head_dim: u32 = 512,
    q_lora: u32 = 1024,
    kv_lora: u32 = 512,
    qk_nope: u32 = 448,
    qk_rope: u32 = 64,
    o_groups: u32 = 8,
    o_lora: u32 = 1024,
    sliding_window: u32 = 128,
    compress_ratio: u32 = 128,
    vha_rank: u32 = 16,

    // Hybrid MLA-DSA attention (compress_ratios[i] <= 0, e.g. -2)
    mla_heads: u32 = 64,
    mla_q_lora: u32 = 1024,
    mla_kv_lora: u32 = 512,
    mla_qk_nope: u32 = 192,
    mla_qk_rope: u32 = 64,
    mla_v_dim: u32 = 256,
    index_heads: u32 = 64,
    index_dim: u32 = 128,
    index_topk: u32 = 2048,

    // RoPE
    max_positions: u32 = 1048576,
    rope_theta: f64 = 8000000.0,
    compress_rope_theta: f64 = 160000.0,
    rope_scaling: RopeScaling = .{},

    // Dense MLP & Latent MXFP4 MoE
    intermediate: u32 = 14336,
    moe_intermediate: u32 = 2048,
    moe_latent: u32 = 2048,
    experts: u32 = 512,
    shared_experts: u32 = 1,
    topk: u32 = 10,
    norm_topk: bool = true,
    situ_beta: f32 = 4.0,
    situ_linear_beta: f32 = 25.0,

    // Tokens
    bos_id: u32 = 1,
    eos_id: u32 = 2,
    pad_id: u32 = 0,

    // Per-layer schedule
    compress_ratios: [max_layers]i32 = @splat(128),

    // Bundled DSpark draft head (model.mtp.*)
    dspark_block_size: u32 = 7,
    dspark_markov_rank: u32 = 256,
    dspark_noise_token: u32 = 1024,
    dspark_targets: [max_dspark_targets]u32 = @splat(0),
    dspark_target_count: u32 = 0,

    pub fn attnKind(c: *const Config, layer: u32) AttnKind {
        return if (c.compress_ratios[layer] <= 0) .dsa else .hca;
    }

    pub fn isDsa(c: *const Config, layer: u32) bool {
        return c.attnKind(layer) == .dsa;
    }

    pub fn isHca(c: *const Config, layer: u32) bool {
        return c.attnKind(layer) == .hca;
    }

    pub fn isMoe(c: *const Config, layer: u32) bool {
        return layer >= c.first_k_dense;
    }

    pub fn dsaCount(c: *const Config) u32 {
        var n: u32 = 0;
        for (0..c.layers) |i| n += @intFromBool(c.isDsa(@intCast(i)));
        return n;
    }

    pub fn hcaCount(c: *const Config) u32 {
        return c.layers - c.dsaCount();
    }

    pub fn hasBundledDspark(c: *const Config) bool {
        return c.dspark_target_count > 0 and c.layers > c.dspark_target_count;
    }

    pub fn hcDim(c: *const Config) u32 {
        return c.hc_mult * c.hidden;
    }

    pub fn hcMixOut(c: *const Config) u32 {
        return c.hc_mult * (2 + c.hc_mult);
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadErnie55Config;
        const o = doc.value.object;

        try expectStr(o, "model_type", "ernie5_5_moe");
        try expectStr(o, "hidden_act", "situ");

        var c: Config = .{};
        c.vocab = try optU32(o, "vocab_size", c.vocab);
        c.hidden = try optU32(o, "hidden_size", c.hidden);
        c.layers = try optU32(o, "num_hidden_layers", c.layers);
        if (c.layers == 0 or c.layers > max_layers) return error.BadErnie55Config;
        c.first_k_dense = try optU32(o, "first_k_dense_replace", c.first_k_dense);
        c.eps = @floatCast(try optF64(o, "rms_norm_eps", c.eps));

        c.hc_mult = try optU32(o, "hc_mult", c.hc_mult);
        c.hc_sinkhorn_iters = try optU32(o, "hc_sinkhorn_iters", c.hc_sinkhorn_iters);
        c.hc_eps = @floatCast(try optF64(o, "hc_eps", c.hc_eps));

        c.heads = try optU32(o, "num_attention_heads", c.heads);
        c.kv_heads = try optU32(o, "num_key_value_heads", c.kv_heads);
        c.head_dim = try optU32(o, "head_dim", c.head_dim);
        c.q_lora = try optU32(o, "q_lora_rank", c.q_lora);
        c.kv_lora = try optU32(o, "kv_lora_rank", c.kv_lora);
        c.qk_nope = try optU32(o, "qk_nope_head_dim", c.qk_nope);
        c.qk_rope = try optU32(o, "qk_rope_head_dim", c.qk_rope);
        c.o_groups = try optU32(o, "o_groups", c.o_groups);
        c.o_lora = try optU32(o, "o_lora_rank", c.o_lora);
        c.sliding_window = try optU32(o, "sliding_window", c.sliding_window);

        c.mla_heads = try optU32(o, "hybrid_mla_num_attention_heads", c.mla_heads);
        c.mla_q_lora = try optU32(o, "hybrid_mla_q_lora_rank", c.mla_q_lora);
        c.mla_kv_lora = try optU32(o, "hybrid_mla_kv_lora_rank", c.mla_kv_lora);
        c.mla_qk_nope = try optU32(o, "hybrid_mla_qk_nope_head_dim", c.mla_qk_nope);
        c.mla_qk_rope = try optU32(o, "hybrid_mla_qk_rope_head_dim", c.mla_qk_rope);
        c.mla_v_dim = try optU32(o, "hybrid_mla_v_head_dim", c.mla_v_dim);
        c.index_heads = try optU32(o, "index_n_heads", c.index_heads);
        c.index_dim = try optU32(o, "index_head_dim", c.index_dim);
        c.index_topk = try optU32(o, "index_topk", c.index_topk);

        c.max_positions = try optU32(o, "max_position_embeddings", c.max_positions);
        c.rope_theta = try optF64(o, "rope_theta", c.rope_theta);
        c.compress_rope_theta = try optF64(o, "compress_rope_theta", c.compress_rope_theta);
        if (o.get("rope_scaling")) |rs| {
            if (rs != .object) return error.BadErnie55Config;
            const ro = rs.object;
            c.rope_scaling.factor = try optF64(ro, "factor", c.rope_scaling.factor);
            c.rope_scaling.original_max_positions = try optU32(ro, "original_max_position_embeddings", c.rope_scaling.original_max_positions);
            c.rope_scaling.beta_fast = try optF64(ro, "beta_fast", c.rope_scaling.beta_fast);
            c.rope_scaling.beta_slow = try optF64(ro, "beta_slow", c.rope_scaling.beta_slow);
            if (ro.get("type")) |tv| {
                c.rope_scaling.yarn = (tv == .string and std.mem.eql(u8, tv.string, "yarn"));
            }
        }

        c.intermediate = try optU32(o, "intermediate_size", c.intermediate);
        c.moe_intermediate = try optU32(o, "moe_intermediate_size", c.moe_intermediate);
        c.moe_latent = try optU32(o, "moe_latent_size", c.moe_latent);
        c.experts = try optU32(o, "n_routed_experts", c.experts);
        c.shared_experts = try optU32(o, "n_shared_experts", c.shared_experts);
        c.topk = try optU32(o, "num_experts_per_tok", c.topk);
        if (o.get("norm_topk_prob")) |v| {
            if (v != .bool) return error.BadErnie55Config;
            c.norm_topk = v.bool;
        }
        c.situ_beta = @floatCast(try optF64(o, "activation_situ_beta", c.situ_beta));
        c.situ_linear_beta = @floatCast(try optF64(o, "activation_situ_linear_beta", c.situ_linear_beta));

        c.bos_id = try optU32(o, "bos_token_id", c.bos_id);
        c.eos_id = try optU32(o, "eos_token_id", c.eos_id);
        c.pad_id = try optU32(o, "pad_token_id", c.pad_id);

        if (o.get("compress_ratios")) |cr| {
            if (cr != .array or cr.array.items.len < c.layers) return error.BadErnie55Config;
            var max_cr: u32 = 1;
            for (0..c.layers) |i| {
                const v = cr.array.items[i];
                if (v != .integer) return error.BadErnie55Config;
                c.compress_ratios[i] = @intCast(v.integer);
                if (v.integer > 1) max_cr = @max(max_cr, @as(u32, @intCast(v.integer)));
            }
            c.compress_ratio = max_cr;
        }

        c.dspark_block_size = try optU32(o, "dspark_block_size", c.dspark_block_size);
        c.dspark_markov_rank = try optU32(o, "dspark_markov_rank", c.dspark_markov_rank);
        c.dspark_noise_token = try optU32(o, "dspark_noise_token_id", c.dspark_noise_token);
        if (o.get("dspark_target_layer_ids")) |dt| {
            if (dt != .array or dt.array.items.len > max_dspark_targets) return error.BadErnie55Config;
            c.dspark_target_count = @intCast(dt.array.items.len);
            for (dt.array.items, 0..) |v, i| {
                if (v != .integer or v.integer < 0) return error.BadErnie55Config;
                c.dspark_targets[i] = @intCast(v.integer);
            }
        }

        // Validate quantization_config: must be block-128 FP8 + MXFP4 experts
        if (o.get("quantization_config")) |qc| {
            if (qc != .object) return error.UnsupportedErnie55Quant;
            const qo = qc.object;
            try expectStr(qo, "quant_method", "fp8");
            try expectStr(qo, "expert_dtype", "mxfp4");
        } else return error.UnsupportedErnie55Quant;

        // Sanity checks on fixed kernel invariants
        if (c.hc_mult != 4 or c.head_dim != 512 or c.qk_rope != 64 or c.qk_nope != 448) return error.UnsupportedErnie55Shape;
        if (c.mla_kv_lora != 512 or c.mla_qk_nope != 192 or c.mla_qk_rope != 64 or c.mla_v_dim != 256) return error.UnsupportedErnie55Shape;
        if (c.index_dim != 128 or c.o_groups != 8) return error.UnsupportedErnie55Shape;
        return c;
    }
};

fn expectStr(o: std.json.ObjectMap, key: []const u8, want: []const u8) !void {
    const v = o.get(key) orelse return error.BadErnie55Config;
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedErnie55Config;
}

fn optU32(o: std.json.ObjectMap, key: []const u8, fallback: u32) !u32 {
    const v = o.get(key) orelse return fallback;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadErnie55Config;
    return @intCast(v.integer);
}

fn optF64(o: std.json.ObjectMap, key: []const u8, fallback: f64) !f64 {
    const v = o.get(key) orelse return fallback;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadErnie55Config,
    };
}

test "ernie55 config parses the release shape and layer schedule" {
    const sample =
        \\{
        \\  "model_type": "ernie5_5_moe",
        \\  "hidden_act": "situ",
        \\  "vocab_size": 201216,
        \\  "hidden_size": 4096,
        \\  "num_hidden_layers": 43,
        \\  "first_k_dense_replace": 1,
        \\  "compress_ratios": [128,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2],
        \\  "dspark_target_layer_ids": [40, 41, 42],
        \\  "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4", "weight_block_size": [128, 128]}
        \\}
    ;
    const c = try Config.parse(std.testing.allocator, sample);
    try std.testing.expectEqual(@as(u32, 43), c.layers);
    try std.testing.expectEqual(@as(u32, 6), c.dsaCount());
    try std.testing.expectEqual(@as(u32, 37), c.hcaCount());
    try std.testing.expect(c.isHca(0) and !c.isMoe(0));
    try std.testing.expect(c.isHca(1) and c.isMoe(1));
    try std.testing.expect(c.isDsa(7) and c.isDsa(14) and c.isDsa(42));
    try std.testing.expect(c.hasBundledDspark());
}
