//! TP sharding geometry for ERNIE 5.5 Flash Preview (world = 1, 2, or 4).
const std = @import("std");
const cfg = @import("config.zig");

pub const Share = struct {
    rank: usize,
    world: usize,

    // Attention heads & groups (HCA & DSA both have 64 heads, HCA has 8 o_groups)
    head_lo: usize,
    heads: usize,
    group_lo: usize,
    groups: usize,
    o_lo: usize, // HCA o_lora slice: group_lo * o_lora .. (group_lo + groups) * o_lora
    o_width: usize,
    dsa_o_lo: usize, // DSA v slice: head_lo * mla_v_dim .. (head_lo + heads) * mla_v_dim
    dsa_o_width: usize,

    // Dense layer 0 MLP (intermediate = 14336 -> 3584/rank at TP=4; exact multiple of 128!)
    dense_lo: usize,
    dense_width: usize,

    // Shared expert & routed MXFP4 experts (moe_intermediate = 2048 -> 512/rank at TP=4)
    moe_lo: usize,
    moe_width: usize,

    // Vocabulary LM head (vocab = 201216 -> 50304/rank at TP=4)
    vocab_lo: usize,
    vocab_width: usize,

    pub fn of(c: *const cfg.Config, rank: usize, world: usize) !Share {
        if (world != 1 and world != 2 and world != 4) return error.UnsupportedErnie55Tp;
        if (rank >= world) return error.BadRank;
        if (c.heads % world != 0 or c.o_groups % world != 0 or c.mla_heads % world != 0) return error.BadErnie55Shard;
        if (c.intermediate % world != 0 or (c.intermediate / world) % 128 != 0) return error.BadErnie55Shard;
        if (c.moe_intermediate % world != 0 or (c.moe_intermediate / world) % 128 != 0) return error.BadErnie55Shard;
        if (c.vocab % world != 0) return error.BadErnie55Shard;

        const heads = c.heads / world;
        const groups = c.o_groups / world;
        const dense_w = c.intermediate / world;
        const moe_w = c.moe_intermediate / world;
        const vocab_w = c.vocab / world;

        return .{
            .rank = rank,
            .world = world,
            .head_lo = rank * heads,
            .heads = heads,
            .group_lo = rank * groups,
            .groups = groups,
            .o_lo = rank * groups * c.o_lora,
            .o_width = groups * c.o_lora,
            .dsa_o_lo = rank * heads * c.mla_v_dim,
            .dsa_o_width = heads * c.mla_v_dim,
            .dense_lo = rank * dense_w,
            .dense_width = dense_w,
            .moe_lo = rank * moe_w,
            .moe_width = moe_w,
            .vocab_lo = rank * vocab_w,
            .vocab_width = vocab_w,
        };
    }
};

test "TP=4 shard divides every ERNIE 5.5 axis on 128-element FP8 block boundaries" {
    const c: cfg.Config = .{};
    for (0..4) |r| {
        const s = try Share.of(&c, r, 4);
        try std.testing.expectEqual(@as(usize, 16), s.heads);
        try std.testing.expectEqual(@as(usize, 2), s.groups);
        try std.testing.expectEqual(@as(usize, 2048), s.o_width);
        try std.testing.expectEqual(@as(usize, 4096), s.dsa_o_width);
        try std.testing.expectEqual(@as(usize, 3584), s.dense_width);
        try std.testing.expectEqual(@as(usize, 512), s.moe_width);
        try std.testing.expectEqual(@as(usize, 50304), s.vocab_width);
        try std.testing.expectEqual(@as(usize, 0), s.dense_lo % 128);
        try std.testing.expectEqual(@as(usize, 0), s.moe_lo % 128);
    }
}
