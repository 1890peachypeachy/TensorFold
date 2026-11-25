//! ERNIE 5.5 RoPE inverse frequencies:
//! - HCA layers: DeepSeek YaRN (compress_rope_theta = 160000, factor = 16, orig = 65536, beta_fast = 32, beta_slow = 1)
//! - DSA layers, Lightning Indexer, DSpark MTP: Plain RoPE (rope_theta = 8000000, no YaRN)
const std = @import("std");
const cfg = @import("config.zig");

pub const half_dim = 32; // qk_rope_head_dim / 2 = 64 / 2

fn findCorrectionDim(num_rotations: f64, dim: f64, base: f64, max_pos: f64) f64 {
    return (dim * std.math.log(f64, std.math.e, max_pos / (num_rotations * 2.0 * std.math.pi))) /
        (2.0 * std.math.log(f64, std.math.e, base));
}

/// DeepSeek YaRN inverse frequencies matching vLLM `DeepseekScalingRotaryEmbedding` (rope_dim = 64).
pub fn hcaInvFreq(c: *const cfg.Config) [half_dim]f32 {
    const dim: f64 = @floatFromInt(c.qk_rope);
    const base = c.compress_rope_theta;
    const factor = c.rope_scaling.factor;
    const orig: f64 = @floatFromInt(c.rope_scaling.original_max_positions);

    var out: [half_dim]f32 = undefined;
    if (!c.rope_scaling.yarn or factor <= 1.0) {
        for (0..half_dim) |i| {
            const p = @as(f64, @floatFromInt(2 * i)) / dim;
            out[i] = @floatCast(1.0 / std.math.pow(f64, base, p));
        }
        return out;
    }

    const low = @max(0.0, @floor(findCorrectionDim(c.rope_scaling.beta_fast, dim, base, orig)));
    const high = @min(dim - 1.0, @ceil(findCorrectionDim(c.rope_scaling.beta_slow, dim, base, orig)));
    const span = if (low == high) 0.001 else (high - low);

    for (0..half_dim) |i| {
        const p = @as(f64, @floatFromInt(2 * i)) / dim;
        const pos_freq = std.math.pow(f64, base, p);
        const inv_extrap = 1.0 / pos_freq;
        const inv_interp = 1.0 / (factor * pos_freq);
        const linear = (@as(f64, @floatFromInt(i)) - low) / span;
        const ramp = std.math.clamp(linear, 0.0, 1.0);
        const mask = 1.0 - ramp; // 1 = keep high-freq extrapolation, 0 = scale by factor
        out[i] = @floatCast(inv_interp * (1.0 - mask) + inv_extrap * mask);
    }
    return out;
}

/// Plain RoPE inverse frequencies for DSA attention, Lightning Indexer, and DSpark MTP (theta = 8e6).
pub fn mlaInvFreq(c: *const cfg.Config) [half_dim]f32 {
    const dim: f64 = @floatFromInt(c.mla_qk_rope);
    const base = c.rope_theta;
    var out: [half_dim]f32 = undefined;
    for (0..half_dim) |i| {
        const p = @as(f64, @floatFromInt(2 * i)) / dim;
        out[i] = @floatCast(1.0 / std.math.pow(f64, base, p));
    }
    return out;
}

test "ernie55 rope frequencies are monotone decreasing and positive" {
    const c: cfg.Config = .{};
    const h = hcaInvFreq(&c);
    const m = mlaInvFreq(&c);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), m[0], 1e-6);
    for (1..half_dim) |i| {
        try std.testing.expect(h[i] > 0.0 and h[i] < h[i - 1]);
        try std.testing.expect(m[i] > 0.0 and m[i] < m[i - 1]);
    }
}
