//! Per-sequence KV cache and reusable forward scratch buffers for ERNIE 5.5 Flash Preview.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");

const Driver = cuda.Driver;
const DeviceBuffer = cuda.DeviceBuffer;

/// Ring capacity for HCA SWA-128 KV cache and C128 compressor state (power of 2 >= max_chunk + 128).
pub const swa_cap: usize = 2048; // >= the 1024-row prompt chunk + 128
pub const swa_mask: usize = swa_cap - 1;

pub const HcaKv = struct {
    swa_kv: u64, // BF16 [swa_cap, 512]
    comp_state: u64, // F32 [swa_cap, 2, 512]
    comp_kv: u64, // BF16 [comp_cap, 512]
};

pub const DsaKv = struct {
    mla_kv: u64, // BF16 [capacity, 576]
    idx_k: u64, // BF16 [capacity, 128]
};

pub const LayerKv = union(enum) {
    hca: HcaKv,
    dsa: DsaKv,
};

pub const KvCache = struct {
    gpa: std.mem.Allocator,
    capacity: usize,
    comp_cap: usize,
    pos: usize = 0,
    buf: DeviceBuffer,
    layers: []LayerKv,

    pub fn alloc(gpa: std.mem.Allocator, d: *const Driver, c: *const cfg.Config, n_layers: u32, capacity: usize) !KvCache {
        const comp_cap = (capacity + c.compress_ratio - 1) / c.compress_ratio + 2;
        var layers = try gpa.alloc(LayerKv, n_layers);
        errdefer gpa.free(layers);

        var total: usize = 0;
        for (0..n_layers) |li| {
            if (c.isDsa(@intCast(li))) {
                total = alignUp(total, 256);
                total += capacity * 576 * 2;
                total = alignUp(total, 256);
                total += capacity * 128 * 2;
            } else {
                total = alignUp(total, 256);
                total += swa_cap * 512 * 2;
                total = alignUp(total, 256);
                total += swa_cap * 2 * 512 * 4;
                total = alignUp(total, 256);
                total += comp_cap * 512 * 2;
            }
        }
        const buf = try DeviceBuffer.alloc(d, @max(total, 256));
        errdefer {
            var owned = buf;
            owned.free();
        }
        try buf.fill8(0, null);

        var off: usize = 0;
        for (0..n_layers) |li| {
            if (c.isDsa(@intCast(li))) {
                off = alignUp(off, 256);
                const mla_kv = buf.ptr + off;
                off += capacity * 576 * 2;
                off = alignUp(off, 256);
                const idx_k = buf.ptr + off;
                off += capacity * 128 * 2;
                layers[li] = .{ .dsa = .{ .mla_kv = mla_kv, .idx_k = idx_k } };
            } else {
                off = alignUp(off, 256);
                const swa_kv = buf.ptr + off;
                off += swa_cap * 512 * 2;
                off = alignUp(off, 256);
                const comp_state = buf.ptr + off;
                off += swa_cap * 2 * 512 * 4;
                off = alignUp(off, 256);
                const comp_kv = buf.ptr + off;
                off += comp_cap * 512 * 2;
                layers[li] = .{ .hca = .{ .swa_kv = swa_kv, .comp_state = comp_state, .comp_kv = comp_kv } };
            }
        }

        return .{
            .gpa = gpa,
            .capacity = capacity,
            .comp_cap = comp_cap,
            .pos = 0,
            .buf = buf,
            .layers = layers,
        };
    }

    pub fn reset(self: *KvCache) void {
        self.pos = 0;
    }

    pub fn deinit(self: *KvCache) void {
        self.buf.free();
        self.gpa.free(self.layers);
        self.* = undefined;
    }
};

/// Reusable device scratch for up to `max_rows` tokens in one forward pass.
pub const Scratch = struct {
    max_rows: usize,
    capacity: usize,
    buf: DeviceBuffer,

    ids: u64, // i32 [max_rows]
    positions: u64, // i32 [max_rows]
    res_a: u64, // BF16 [max_rows, 4, 4096]
    res_b: u64, // BF16 [max_rows, 4, 4096]
    post_mix: u64, // F32 [max_rows, 4]
    mhc_mix_raw: u64, // F32 [max_rows, 24]
    mhc_sqrsum: u64, // F32 [max_rows]
    comb_mix: u64, // F32 [max_rows, 16]
    layer_in: u64, // BF16 [max_rows, 4096]
    sub_out: u64, // BF16 [max_rows, 4096]

    // Attention scratch (shared between HCA and DSA)
    q_down: u64, // BF16 [max_rows, 1024]
    kv_a: u64, // BF16 [max_rows, 576] (512 for HCA, 576 for DSA)
    kv_norm: u64, // BF16 [max_rows, 512]
    q_up: u64, // BF16 [max_rows, max(heads*512, heads*256)]
    q_abs: u64, // BF16 [max_rows, heads, 576]
    attn_o: u64, // BF16 [max_rows, heads * 512]
    attn_o_v: u64, // BF16 [max_rows, heads * 256]
    o_gate: u64, // BF16 [max_rows, max(o_width, dsa_o_width)]
    wo_a_out: u64, // BF16 [max_rows, o_width]
    vha_z: u64, // F32 [max_rows, 16, 512]
    vha_z_all: u64, // F32 [world, max_rows, 16, 512]
    comp_wkv: u64, // BF16 [max_rows, 512]
    comp_wgate: u64, // BF16 [max_rows, 512]

    // DSA indexer scratch
    idx_q: u64, // BF16 [max_rows, 64, 128]
    idx_k: u64, // BF16 [max_rows, 128]
    idx_w: u64, // BF16 [max_rows, 64]
    idx_scores: u64, // F32 [min(max_rows, 64), capacity]
    idx_sel: u64, // i32 [max_rows, 2176]
    idx_cnt: u64, // i32 [max_rows]

    // FFN / MoE scratch
    dense_gate: u64, // BF16 [max_rows, dense_width]
    dense_up: u64, // BF16 [max_rows, dense_width]
    router_raw: u64, // F32 [max_rows, 2 * experts]
    router_scores: u64, // F32 [max_rows, experts]
    topk_ids: u64, // i32 [max_rows, topk]
    topk_w: u64, // F32 [max_rows, topk]
    latent_in: u64, // BF16 [max_rows, 2048]
    moe_mid: u64, // BF16 [max_rows, topk, moe_width]
    routed_out: u64, // BF16 [max_rows, 2048]
    shared_out: u64, // BF16 [max_rows, 4096]
    moe_pack: u64, // BF16 [max_rows, 6144]
    route_counts: u64, // i32 [experts]
    route_offsets: u64, // i32 [experts + 1]
    route_cursor: u64, // i32 [experts]
    route_list: u64, // i32 [max_rows, topk]
    moe_part: u64, // F32 [max_rows, topk, 2048] the expert-major w2's per-pick sums
    tp_gather: u64, // BF16 [world, max_rows, 6144]

    // DSpark target aux taps: 3 slots of [max_rows, 4096] + concatenated [max_rows, 12288]
    dspark_taps_tmp: u64, // BF16 [3, max_rows, 4096]
    dspark_taps: u64, // BF16 [max_rows, 12288]

    // qmmf split-K partial accumulator + logits
    qmmf_part: u64, // F32 [8, max_rows, max_n]
    logits: u64, // F32 [max_rows, vocab_width]
    argmax_id: u64, // i32 [max_rows]
    argmax_val: u64, // F32 [max_rows]
    proj_tmp: u64, // BF16 [max_rows, max_proj_n]
    attn_split_o: u64, // F32 [16, 12, 16, 512]
    attn_split_m: u64, // F32 [16, 12, 16]
    attn_split_l: u64, // F32 [16, 12, 16]

    pub fn alloc(d: *const Driver, c: *const cfg.Config, sh: *const shard.Share, max_rows: usize, capacity: usize) !Scratch {
        const R = max_rows;
        const max_proj_n = @max(sh.heads * 512 + sh.o_width, @max(sh.heads * 256 + sh.dsa_o_width, @max(c.moe_latent + 2 * sh.moe_width, 2 * sh.dense_width)));
        const max_n = @max(sh.vocab_width, max_proj_n);
        const idx_score_rows = @min(R, 64);

        var off: usize = 0;
        const reserve = struct {
            fn f(cur: *usize, bytes: usize) usize {
                cur.* = alignUp(cur.*, 256);
                const p = cur.*;
                cur.* += bytes;
                return p;
            }
        }.f;

        var s: Scratch = undefined;
        s.max_rows = R;
        s.capacity = capacity;

        const o_ids = reserve(&off, R * 4);
        const o_pos = reserve(&off, R * 4);
        const o_res_a = reserve(&off, R * 4 * c.hidden * 2);
        const o_res_b = reserve(&off, R * 4 * c.hidden * 2);
        const o_post_mix = reserve(&off, R * 4 * 4);
        const o_mhc_mix = reserve(&off, R * 24 * 4);
        const o_mhc_sq = reserve(&off, R * 4);
        const o_comb_mix = reserve(&off, R * 16 * 4);
        const o_layer_in = reserve(&off, R * c.hidden * 2);
        const o_sub_out = reserve(&off, R * c.hidden * 2);

        const o_q_down = reserve(&off, R * c.q_lora * 2);
        const o_kv_a = reserve(&off, R * 576 * 2);
        const o_kv_norm = reserve(&off, R * 512 * 2);
        const o_q_up = reserve(&off, R * sh.heads * 512 * 2);
        const o_q_abs = reserve(&off, R * sh.heads * 576 * 2);
        const o_attn_o = reserve(&off, R * sh.heads * 512 * 2);
        const o_attn_o_v = reserve(&off, R * sh.heads * 256 * 2);
        const o_o_gate = reserve(&off, R * @max(sh.o_width, sh.dsa_o_width) * 2);
        const o_wo_a = reserve(&off, R * sh.o_width * 2);
        const o_vha_z = reserve(&off, R * 16 * 512 * 4);
        const o_vha_z_all = reserve(&off, sh.world * R * 16 * 512 * 4);
        const o_comp_wkv = reserve(&off, R * 512 * 2);
        const o_comp_wgate = reserve(&off, R * 512 * 2);

        const o_idx_q = reserve(&off, R * 64 * 128 * 2);
        const o_idx_k = reserve(&off, R * 128 * 2);
        const o_idx_w = reserve(&off, R * 64 * 2);
        const o_idx_scores = reserve(&off, idx_score_rows * capacity * 4);
        const o_idx_sel = reserve(&off, R * (c.index_topk + c.sliding_window) * 4);
        const o_idx_cnt = reserve(&off, R * 4);

        const o_dense_gate = reserve(&off, R * sh.dense_width * 2);
        const o_dense_up = reserve(&off, R * sh.dense_width * 2);
        const o_router_raw = reserve(&off, R * 2 * c.experts * 4);
        const o_router_sc = reserve(&off, R * c.experts * 4);
        const o_topk_ids = reserve(&off, R * c.topk * 4);
        const o_topk_w = reserve(&off, R * c.topk * 4);
        const o_latent_in = reserve(&off, R * c.moe_latent * 2);
        const o_moe_mid = reserve(&off, R * c.topk * sh.moe_width * 2);
        const o_routed_out = reserve(&off, R * c.moe_latent * 2);
        const o_shared_out = reserve(&off, R * c.hidden * 2);
        const o_moe_pack = reserve(&off, R * (c.moe_latent + c.hidden) * 2);
        const o_route_counts = reserve(&off, c.experts * 4);
        const o_route_offsets = reserve(&off, (c.experts + 1) * 4);
        const o_route_cursor = reserve(&off, c.experts * 4);
        const o_route_list = reserve(&off, R * c.topk * 4);
        const o_moe_part = reserve(&off, R * c.topk * c.moe_latent * 4);
        const o_tp_gather = reserve(&off, sh.world * R * (c.moe_latent + c.hidden) * 2);

        const o_taps_tmp = reserve(&off, 3 * R * c.hidden * 2);
        const o_taps = reserve(&off, R * 3 * c.hidden * 2);

        const o_qmmf_part = reserve(&off, 8 * R * max_n * 4);
        const o_logits = reserve(&off, R * sh.vocab_width * 4);
        const o_argmax_id = reserve(&off, R * 4);
        const o_argmax_val = reserve(&off, R * 4);
        const o_proj_tmp = reserve(&off, R * max_proj_n * 2);
        const o_attn_split_o = reserve(&off, 16 * 12 * 16 * 512 * 4);
        const o_attn_split_m = reserve(&off, 16 * 12 * 16 * 4);
        const o_attn_split_l = reserve(&off, 16 * 12 * 16 * 4);

        s.buf = try DeviceBuffer.alloc(d, off);
        const b = s.buf.ptr;

        s.ids = b + o_ids;
        s.positions = b + o_pos;
        s.res_a = b + o_res_a;
        s.res_b = b + o_res_b;
        s.post_mix = b + o_post_mix;
        s.mhc_mix_raw = b + o_mhc_mix;
        s.mhc_sqrsum = b + o_mhc_sq;
        s.comb_mix = b + o_comb_mix;
        s.layer_in = b + o_layer_in;
        s.sub_out = b + o_sub_out;
        s.q_down = b + o_q_down;
        s.kv_a = b + o_kv_a;
        s.kv_norm = b + o_kv_norm;
        s.q_up = b + o_q_up;
        s.q_abs = b + o_q_abs;
        s.attn_o = b + o_attn_o;
        s.attn_o_v = b + o_attn_o_v;
        s.o_gate = b + o_o_gate;
        s.wo_a_out = b + o_wo_a;
        s.vha_z = b + o_vha_z;
        s.vha_z_all = b + o_vha_z_all;
        s.comp_wkv = b + o_comp_wkv;
        s.comp_wgate = b + o_comp_wgate;
        s.idx_q = b + o_idx_q;
        s.idx_k = b + o_idx_k;
        s.idx_w = b + o_idx_w;
        s.idx_scores = b + o_idx_scores;
        s.idx_sel = b + o_idx_sel;
        s.idx_cnt = b + o_idx_cnt;
        s.dense_gate = b + o_dense_gate;
        s.dense_up = b + o_dense_up;
        s.router_raw = b + o_router_raw;
        s.router_scores = b + o_router_sc;
        s.topk_ids = b + o_topk_ids;
        s.topk_w = b + o_topk_w;
        s.latent_in = b + o_latent_in;
        s.moe_mid = b + o_moe_mid;
        s.routed_out = b + o_routed_out;
        s.shared_out = b + o_shared_out;
        s.moe_pack = b + o_moe_pack;
        s.route_counts = b + o_route_counts;
        s.route_offsets = b + o_route_offsets;
        s.route_cursor = b + o_route_cursor;
        s.route_list = b + o_route_list;
        s.moe_part = b + o_moe_part;
        s.tp_gather = b + o_tp_gather;
        s.dspark_taps_tmp = b + o_taps_tmp;
        s.dspark_taps = b + o_taps;
        s.qmmf_part = b + o_qmmf_part;
        s.logits = b + o_logits;
        s.argmax_id = b + o_argmax_id;
        s.argmax_val = b + o_argmax_val;
        s.proj_tmp = b + o_proj_tmp;
        s.attn_split_o = b + o_attn_split_o;
        s.attn_split_m = b + o_attn_split_m;
        s.attn_split_l = b + o_attn_split_l;
        return s;
    }

    pub fn deinit(self: *Scratch) void {
        self.buf.free();
        self.* = undefined;
    }
};

fn alignUp(v: usize, a: usize) usize {
    return (v + a - 1) / a * a;
}
