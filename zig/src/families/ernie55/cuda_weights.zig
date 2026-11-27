//! ERNIE 5.5 Flash Preview CUDA weight loader (block-128 FP8 packed for qmmf + bulk MXFP4 routed experts + packed BF16 router/lm_head + optional bundled 3-layer DSpark).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const rope = @import("rope.zig");

const Driver = cuda.Driver;
const DeviceBuffer = cuda.DeviceBuffer;
const QWeight = cuda.qmmf.Weight;

pub const MhcWeights = struct {
    fn_w: u64, // F32 [24, 16384]
    scale: u64, // F32 [3] = {alpha_pre, alpha_post, alpha_res}
    base: u64, // F32 [24]
};

pub const HcaWeights = struct {
    qdown_kv: QWeight, // FP8 [1536, 4096] = concat(q_down [1024], kv_proj [512])
    q_norm: u64, // BF16 [1024]
    qup_ogate: QWeight, // FP8 [heads * 512 + o_width, 1024] = concat(q_up, o_gate)
    kv_proj: QWeight, // Zero-copy FP8 [512, 4096] view inside qdown_kv (used by DSpark absorb)
    kv_norm: u64, // BF16 [512]
    wo_a: [8]QWeight, // FP8 [1024, 4096] per local group (only 0..share.groups used)
    wo_b: QWeight, // FP8 [4096, groups * 1024] (row-parallel)
    vha_u: u64, // BF16 [heads, 16]
    vha_v: u64, // BF16 [heads, 16]
    attn_sink: u64, // F32 [heads]
    comp_ape: u64 = 0, // F32 [128, 512]
    comp_wkv_wgate: ?QWeight = null, // BF16 [1024, 4096] = concat(wkv [512], wgate [512]) packed for qmmf
    comp_norm: u64 = 0, // BF16 [512]
};

pub const DsaWeights = struct {
    qa_kva: QWeight, // FP8 [1600, 4096] = concat(q_a [1024], kv_a [576])
    q_a_norm: u64, // BF16 [1024]
    qb_ogate: QWeight, // FP8 [heads * 256 + dsa_o_width, 1024] = concat(q_b, o_gate)
    kv_a_norm: u64, // BF16 [512]
    wk: u64, // BF16 [heads, 192, 512]
    wv: u64, // BF16 [heads, 256, 512]
    o_proj: QWeight, // FP8 [4096, heads * 256] (row-parallel)
    vha_u: u64, // BF16 [heads, 16]
    vha_v: u64, // BF16 [heads, 16]
    sinks: u64, // F32 [heads]
    idx_wq_b: QWeight, // FP8 [8192, 1024]
    idx_wk: QWeight, // BF16 [128, 4096] packed for qmmf
    idx_k_norm_w: u64, // BF16 [128]
    idx_k_norm_b: u64, // BF16 [128]
    idx_weights_proj: QWeight, // BF16 [64, 4096] packed for qmmf
    fi_gate: u64 = 0, // optional FlashIndexer BF16 [64, 4096]
    fi_bias: u64 = 0, // optional FlashIndexer BF16 [64]
};

pub const DenseMlpWeights = struct {
    gate_up: QWeight, // FP8 [2 * dense_width, 4096] = concat(gate, up)
    down: QWeight, // FP8 [4096, dense_width] (row-parallel)
};

pub const MoeWeights = struct {
    gate_packed: QWeight, // BF16 [2 * experts, 4096] packed for qmmf ([w0; w1]) (rows > 8)
    gate_raw0: u64, // BF16 [experts, 4096] raw row-major w0 for fused decode router (rows <= 8)
    gate_raw1: u64, // BF16 [experts, 4096] raw row-major w1 for fused decode router (rows <= 8)
    bias: u64, // F32 [512]
    routed_scale: u64, // F32 [512]
    fc1_shared_w13: QWeight, // FP8 [2048 + 2 * moe_width, 4096] = concat(fc1_latent, shared_w1, shared_w3)
    latent_norm: u64, // BF16 [2048]
    fc2_latent: QWeight, // FP8 [4096, 2048]
    shared_w2: QWeight, // FP8 [4096, moe_width] (row-parallel)
    w1_q: u64, // uint8 [512, moe_width, 1024]
    w1_s: u64, // uint8 [512, moe_width, 64]
    w3_q: u64, // uint8 [512, moe_width, 1024]
    w3_s: u64, // uint8 [512, moe_width, 64]
    w2_q: u64, // uint8 [512, 2048, moe_width / 2]
    w2_s: u64, // uint8 [512, 2048, moe_width / 32]
};

pub const LayerWeights = struct {
    attn_norm: u64, // BF16 [4096]
    ffn_norm: u64, // BF16 [4096]
    hc_attn: MhcWeights,
    hc_ffn: MhcWeights,
    attn: union(enum) { hca: HcaWeights, dsa: DsaWeights },
    ffn: union(enum) { dense: DenseMlpWeights, moe: MoeWeights },
};

pub const DsparkLayerWeights = struct {
    attn_norm: u64, // BF16 [4096]
    ffn_norm: u64, // BF16 [4096]
    hc_attn: MhcWeights,
    hc_ffn: MhcWeights,
    hca: HcaWeights,
    moe: MoeWeights,
};

pub const DsparkWeights = struct {
    feature_proj: QWeight, // FP8 [4096, 12288]
    feature_norm: u64, // BF16 [4096]
    hc_head_fn: u64, // F32 [4, 16384]
    hc_head_base: u64, // F32 [4]
    hc_head_scale: u64, // F32 [1]
    norm: u64, // BF16 [4096]
    markov_w1: u64, // BF16 [vocab, 256] (transposed from checkpoint [256, vocab])
    markov_w2: u64, // BF16 [vocab_width, 256] (vocab-sharded)
    layers: [3]DsparkLayerWeights,
};

pub const Fp8Raw = struct { codes: u64, scales: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    loaded_layers: u32,
    embed: u64 = 0, // BF16 [vocab, 4096]
    norm: u64 = 0, // BF16 [4096]
    hc_head_fn: u64 = 0, // F32 [4, 16384]
    hc_head_base: u64 = 0, // F32 [4]
    hc_head_scale: u64 = 0, // F32 [1]
    lm_head: QWeight = undefined, // BF16 [vocab_width, 4096] packed for qmmf
    lm_head_raw: u64 = 0, // BF16 [vocab_width, 4096] raw row-major for 250 GB/s decode GEMV
    multimax_ranges: u64 = 0, // F32 [4]
    multimax_ts: u64 = 0, // F32 [4]
    hca_inv_freq: u64 = 0, // F32 [32]
    mla_inv_freq: u64 = 0, // F32 [32]
    layers: []LayerWeights = &.{},
    dspark: ?DsparkWeights = null,
    fp8_raw: std.AutoHashMapUnmanaged(u64, Fp8Raw) = .empty,
    bf16_raw: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.fp8_raw.deinit(w.gpa);
        w.bf16_raw.deinit(w.gpa);
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

pub fn e4m3ToF32(b: u8) f32 {
    const sign: f32 = if ((b & 0x80) != 0) -1.0 else 1.0;
    const exp: i32 = @intCast((b >> 3) & 0xF);
    const mant: f32 = @floatFromInt(b & 0x7);
    if (exp == 0) {
        return sign * std.math.scalbn(mant * 0.125, -6);
    }
    if (exp == 15 and (b & 0x7) == 7) return std.math.nan(f32);
    return sign * std.math.scalbn(1.0 + mant * 0.125, exp - 7);
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn f32ToBf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    const lsb = (u >> 16) & 1;
    const rounded = u +% (0x7FFF + lsb);
    return @truncate(rounded >> 16);
}

pub const LoadOptions = struct {
    max_layers: ?u32 = null,
    load_dspark: bool = false,
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, rank: usize, world: usize, opts: LoadOptions) !Weights {
    const c = try cfg.Config.read(gpa, io, dir);
    const sh = try shard.Share.of(&c, rank, world);
    const n_layers: u32 = if (opts.max_layers) |ml| @min(ml, c.layers) else c.layers;

    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);

    var w: Weights = .{
        .gpa = gpa,
        .config = c,
        .share = sh,
        .loaded_layers = n_layers,
        .layers = try gpa.alloc(LayerWeights, n_layers),
    };
    errdefer w.deinit();

    var pstream = try cuda.Stream.init(d, false);
    defer pstream.deinit();
    var packer = try cuda.modelopt.Packer.load(d);
    defer packer.unload();
    var L: Loader = .{ .gpa = gpa, .d = d, .s = pstream, .packer = &packer, .w = &w, .ck = &ck };
    defer L.deinit();

    const hf = rope.hcaInvFreq(&c);
    const mf = rope.mlaInvFreq(&c);
    w.hca_inv_freq = try L.put(std.mem.sliceAsBytes(&hf));
    w.mla_inv_freq = try L.put(std.mem.sliceAsBytes(&mf));

    w.embed = try L.putRaw("model.embed_tokens.weight");
    w.norm = try L.putRaw("model.norm.weight");
    w.hc_head_fn = try L.putTransposedF32("model.mhc_contract.hc_head_fn", c.hcDim(), c.hc_mult);
    w.hc_head_base = try L.putRaw("model.mhc_contract.hc_head_base");
    w.hc_head_scale = try L.putRaw("model.mhc_contract.hc_head_scale");
    w.lm_head = try L.loadBf16Lane("lm_head.weight", sh.vocab_lo, sh.vocab_width, c.hidden);
    w.lm_head_raw = try L.putRowSliceBf16("lm_head.weight", sh.vocab_lo, sh.vocab_width, c.hidden);
    w.multimax_ranges = try L.putBf16AsF32("model.lm_head.multimax_ranges", 0, 4);
    w.multimax_ts = try L.putBf16AsF32("model.lm_head.multimax_ts", 0, 4);

    for (0..n_layers) |li| {
        const l: u32 = @intCast(li);
        w.layers[li] = try L.loadLayer(l);
        if (rank == 0 and (li == 0 or li == 1 or li == 7 or li + 1 == n_layers or (li + 1) % 10 == 0)) {
            std.log.info("[ernie55] loaded layer {d}/{d} ({d:.2} GiB)", .{ li + 1, n_layers, @as(f64, @floatFromInt(w.bytes)) / (1024.0 * 1024.0 * 1024.0) });
        }
    }


    // Optional Baidu FlashIndexer 8-of-64 head router (flashindexer/flashindexer.safetensors, 3.1 MB)
    {
        var fi_path_buf: [512]u8 = undefined;
        const fi_path = std.fmt.bufPrint(&fi_path_buf, "{s}/flashindexer/flashindexer.safetensors", .{dir}) catch "";
        if (fi_path.len > 0) {
            if (std.Io.Dir.cwd().readFileAlloc(io, fi_path, gpa, .unlimited)) |raw| {
                defer gpa.free(raw);
                if (raw.len >= 8) {
                    const hdr_len: usize = @intCast(std.mem.readInt(u64, raw[0..8], .little));
                    if (8 + hdr_len <= raw.len) {
                        const data_base = raw[8 + hdr_len ..];
                        if (std.json.parseFromSlice(std.json.Value, gpa, raw[8 .. 8 + hdr_len], .{})) |parsed| {
                            defer parsed.deinit();
                            if (parsed.value == .object) {
                                const obj = parsed.value.object;
                                var fi_loaded: usize = 0;
                                for (0..n_layers) |li| {
                                    switch (w.layers[li].attn) {
                                        .dsa => |*dsa| {
                                            var nb1: [256]u8 = undefined;
                                            var nb2: [256]u8 = undefined;
                                            const gn = std.fmt.bufPrint(&nb1, "model.layers.{d}.self_attn.core_attention.indexer.linear_gate.weight", .{li}) catch continue;
                                            const bn = std.fmt.bufPrint(&nb2, "model.layers.{d}.self_attn.core_attention.indexer.flashindexer_bias", .{li}) catch continue;
                                            const gv = obj.get(gn) orelse continue;
                                            const bv = obj.get(bn) orelse continue;
                                            const goff = gv.object.get("data_offsets") orelse continue;
                                            const boff = bv.object.get("data_offsets") orelse continue;
                                            const g0: usize = @intCast(goff.array.items[0].integer);
                                            const g1: usize = @intCast(goff.array.items[1].integer);
                                            const b0: usize = @intCast(boff.array.items[0].integer);
                                            const b1: usize = @intCast(boff.array.items[1].integer);
                                            if (g1 <= data_base.len and b1 <= data_base.len) {
                                                dsa.fi_gate = try L.put(data_base[g0..g1]);
                                                dsa.fi_bias = try L.put(data_base[b0..b1]);
                                                fi_loaded += 1;
                                            }
                                        },
                                        .hca => {},
                                    }
                                }
                                if (rank == 0 and fi_loaded > 0) {
                                    std.log.info("[ernie55] loaded FlashIndexer 8-of-64 prefill head router ({d} DSA layers)", .{fi_loaded});
                                }
                            }
                        } else |_| {}
                    }
                }
            } else |_| {}
        }
    }

    if (opts.load_dspark) {
        w.dspark = try L.loadDspark();
        if (rank == 0) {
            std.log.info("[ernie55] loaded bundled 3-layer DSpark (total {d:.2} GiB)", .{@as(f64, @floatFromInt(w.bytes)) / (1024.0 * 1024.0 * 1024.0)});
        }
    }

    return w;
}

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    s: cuda.Stream,
    packer: *const cuda.modelopt.Packer,
    w: *Weights,
    ck: *core.Checkpoint,
    nbuf: [256]u8 = undefined,
    nbuf2: [256]u8 = undefined,
    fp8_raw_scratch: []u8 = &.{},
    fp8_pack_scratch: []u8 = &.{},
    fp8_scale_scratch: []f32 = &.{},
    // Reusable bulk staging buffers for 512 MXFP4 experts per layer
    moe_w1_q: []u8 = &.{},
    moe_w1_s: []u8 = &.{},
    moe_w3_q: []u8 = &.{},
    moe_w3_s: []u8 = &.{},
    moe_w2_q: []u8 = &.{},
    moe_w2_s: []u8 = &.{},

    fn deinit(L: *Loader) void {
        if (L.fp8_raw_scratch.len > 0) L.gpa.free(L.fp8_raw_scratch);
        if (L.fp8_pack_scratch.len > 0) L.gpa.free(L.fp8_pack_scratch);
        if (L.fp8_scale_scratch.len > 0) L.gpa.free(L.fp8_scale_scratch);
        if (L.moe_w1_q.len > 0) L.gpa.free(L.moe_w1_q);
        if (L.moe_w1_s.len > 0) L.gpa.free(L.moe_w1_s);
        if (L.moe_w3_q.len > 0) L.gpa.free(L.moe_w3_q);
        if (L.moe_w3_s.len > 0) L.gpa.free(L.moe_w3_s);
        if (L.moe_w2_q.len > 0) L.gpa.free(L.moe_w2_q);
        if (L.moe_w2_s.len > 0) L.gpa.free(L.moe_w2_s);
    }

    fn fmt1(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&L.nbuf, fmt, args);
    }

    fn fmt2(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&L.nbuf2, fmt, args);
    }

    fn put(L: *Loader, bytes: []const u8) !u64 {
        const b = try DeviceBuffer.fromHost(L.d, bytes);
        errdefer {
            var owned = b;
            owned.free();
        }
        try L.w.buffers.append(L.gpa, b);
        L.w.bytes += b.len;
        return b.ptr;
    }

    fn allocDev(L: *Loader, len: usize) !DeviceBuffer {
        const b = try DeviceBuffer.alloc(L.d, len);
        errdefer {
            var owned = b;
            owned.free();
        }
        try L.w.buffers.append(L.gpa, b);
        L.w.bytes += b.len;
        return b;
    }

    fn putRaw(L: *Loader, name: []const u8) !u64 {
        const t = try L.ck.get(name);
        return L.put(t.bytes);
    }

    fn loadBf16Lane(L: *Loader, name: []const u8, row0: usize, n_rows: usize, cols: usize) !QWeight {
        const t = try L.ck.get(name);
        const row_bytes = cols * 2;
        const src_bytes = t.bytes[row0 * row_bytes .. (row0 + n_rows) * row_bytes];
        const raw_ptr = try L.put(src_bytes);
        const npad = cuda.modelopt.padded(n_rows);
        const dst = try L.allocDev(npad * cols * 2);
        try L.packer.bf16Lane(L.s, raw_ptr, dst.ptr, n_rows, cols);
        try L.s.synchronize();
        if (cols % 32 == 0) try L.w.bf16_raw.put(L.gpa, dst.ptr, raw_ptr);
        return cuda.modelopt.bf16Weight(dst.ptr, n_rows, cols);
    }

    fn loadBf16LanePair(L: *Loader, name0: []const u8, name1: []const u8, rows_each: usize, cols: usize) !QWeight {
        const t0 = try L.ck.get(name0);
        const t1 = try L.ck.get(name1);
        const part_bytes = rows_each * cols * 2;
        const raw_buf = try L.allocDev(2 * part_bytes);
        try raw_buf.upload(0, t0.bytes[0..part_bytes]);
        try raw_buf.upload(part_bytes, t1.bytes[0..part_bytes]);
        const n_rows = 2 * rows_each;
        const npad = cuda.modelopt.padded(n_rows);
        const dst = try L.allocDev(npad * cols * 2);
        try L.packer.bf16Lane(L.s, raw_buf.ptr, dst.ptr, n_rows, cols);
        try L.s.synchronize();
        if (cols % 32 == 0) try L.w.bf16_raw.put(L.gpa, dst.ptr, raw_buf.ptr);
        return cuda.modelopt.bf16Weight(dst.ptr, n_rows, cols);
    }

    fn putRowSliceBf16(L: *Loader, name: []const u8, row0: usize, n_rows: usize, cols: usize) !u64 {
        const t = try L.ck.get(name);
        const row_bytes = cols * 2;
        return L.put(t.bytes[row0 * row_bytes .. (row0 + n_rows) * row_bytes]);
    }

    fn putRowSliceF32(L: *Loader, name: []const u8, row0: usize, n_rows: usize, cols: usize) !u64 {
        const t = try L.ck.get(name);
        const row_bytes = cols * 4;
        return L.put(t.bytes[row0 * row_bytes .. (row0 + n_rows) * row_bytes]);
    }

    fn putBf16AsF32(L: *Loader, name: []const u8, lo: usize, count: usize) !u64 {
        const t = try L.ck.get(name);
        const src = std.mem.bytesAsSlice(u16, t.bytes);
        const tmp = try L.gpa.alloc(f32, count);
        defer L.gpa.free(tmp);
        for (0..count) |i| tmp[i] = bf16ToF32(src[lo + i]);
        return L.put(std.mem.sliceAsBytes(tmp));
    }

    fn putTransposedF32(L: *Loader, name: []const u8, in_rows: usize, in_cols: usize) !u64 {
        const t = try L.ck.get(name);
        const src = std.mem.bytesAsSlice(f32, t.bytes);
        const dst = try L.gpa.alloc(f32, in_rows * in_cols);
        defer L.gpa.free(dst);
        for (0..in_rows) |r| {
            for (0..in_cols) |c| {
                dst[c * in_rows + r] = src[r * in_cols + c];
            }
        }
        return L.put(std.mem.sliceAsBytes(dst));
    }

    fn putTransposedBf16(L: *Loader, name: []const u8, in_rows: usize, in_cols: usize) !u64 {
        const t = try L.ck.get(name);
        const src = std.mem.bytesAsSlice(u16, t.bytes);
        const dst = try L.gpa.alloc(u16, in_rows * in_cols);
        defer L.gpa.free(dst);
        for (0..in_rows) |r| {
            for (0..in_cols) |c| {
                dst[c * in_rows + r] = src[r * in_cols + c];
            }
        }
        return L.put(std.mem.sliceAsBytes(dst));
    }

    fn ensureScratch(L: *Loader, raw_n: usize, pack_n: usize, scale_n: usize) !void {
        if (L.fp8_raw_scratch.len < raw_n) {
            if (L.fp8_raw_scratch.len > 0) L.gpa.free(L.fp8_raw_scratch);
            L.fp8_raw_scratch = try L.gpa.alloc(u8, raw_n);
        }
        if (L.fp8_pack_scratch.len < pack_n) {
            if (L.fp8_pack_scratch.len > 0) L.gpa.free(L.fp8_pack_scratch);
            L.fp8_pack_scratch = try L.gpa.alloc(u8, pack_n);
        }
        if (L.fp8_scale_scratch.len < scale_n) {
            if (L.fp8_scale_scratch.len > 0) L.gpa.free(L.fp8_scale_scratch);
            L.fp8_scale_scratch = try L.gpa.alloc(f32, scale_n);
        }
    }

    fn loadFp8Slice(L: *Loader, w_name: []const u8, s_name: []const u8, row0: usize, n: usize, k0: usize, k: usize) !QWeight {
        const wt = try L.ck.get(w_name);
        const st = try L.ck.get(s_name);
        const full_rows = wt.shape[0];
        const full_cols = wt.shape[1];
        const npad = cuda.fp8.padded(n);
        const kg = k / cuda.fp8.group;
        try L.ensureScratch(n * k, npad * k, npad * kg);

        const raw_slice: []const u8 = if (k0 == 0 and k == full_cols)
            wt.bytes[row0 * full_cols .. (row0 + n) * full_cols]
        else blk: {
            const dst = L.fp8_raw_scratch[0 .. n * k];
            for (0..n) |r| {
                const s_off = (row0 + r) * full_cols + k0;
                @memcpy(dst[r * k .. (r + 1) * k], wt.bytes[s_off .. s_off + k]);
            }
            break :blk dst;
        };

        const packed_codes = L.fp8_pack_scratch[0 .. npad * k];
        const packed_scales = L.fp8_scale_scratch[0 .. npad * kg];
        cuda.fp8.packCodes(packed_codes, raw_slice, n, k);
        const scale_inv = try L.gpa.alloc(f32, st.bytes.len / 4);
        defer L.gpa.free(scale_inv);
        @memcpy(std.mem.sliceAsBytes(scale_inv), st.bytes);
        cuda.fp8.packScalesAt(packed_scales, scale_inv, full_rows, full_cols, row0, k0, n, k);

        const d_codes = try L.put(packed_codes);
        const d_scales = try L.put(std.mem.sliceAsBytes(packed_scales));
        if (k % 128 == 0 and k0 % 128 == 0 and row0 % 128 == 0) {
            const nb = (n + 127) / 128;
            const kb = k / 128;
            const full_kb = (full_cols + 127) / 128;
            const raw_sc = try L.gpa.alloc(f32, nb * kb);
            defer L.gpa.free(raw_sc);
            for (0..nb) |r| {
                const src_r = (row0 / 128) + r;
                @memcpy(raw_sc[r * kb .. (r + 1) * kb], scale_inv[src_r * full_kb + (k0 / 128) ..][0..kb]);
            }
            const d_raw = try L.put(raw_slice);
            const d_raw_s = try L.put(std.mem.sliceAsBytes(raw_sc));
            try L.w.fp8_raw.put(L.gpa, d_codes, .{ .codes = d_raw, .scales = d_raw_s });
        }
        return cuda.fp8.weight(d_codes, d_scales, n, k);
    }

    const Fp8Part = struct { w: []const u8, s: []const u8, row0: usize, n: usize };

    /// Pack 2..3 block-128 FP8 projections with identical K (k0=0) into one contiguous qmmf Weight.
    /// Every part before the last must have `n % 128 == 0` so 64-row code/scale tiles align identically!
    fn loadFp8Concat(L: *Loader, parts: []const Fp8Part, k: usize) !QWeight {
        var total_n: usize = 0;
        var total_npad: usize = 0;
        for (parts, 0..) |pt, pi| {
            if (pi + 1 < parts.len) std.debug.assert(pt.n % cuda.fp8.block == 0);
            total_n += pt.n;
            total_npad += cuda.fp8.padded(pt.n);
        }
        std.debug.assert(total_npad == cuda.fp8.padded(total_n));
        const kg = k / cuda.fp8.group;
        const all_codes = try L.gpa.alloc(u8, total_npad * k);
        defer L.gpa.free(all_codes);
        const all_scales = try L.gpa.alloc(f32, total_npad * kg);
        defer L.gpa.free(all_scales);

        var row_off: usize = 0;
        for (parts) |pt| {
            const wt = try L.ck.get(pt.w);
            const st = try L.ck.get(pt.s);
            const full_rows = wt.shape[0];
            const full_cols = wt.shape[1];
            std.debug.assert(full_cols == k);
            const npad = cuda.fp8.padded(pt.n);
            const raw_slice = wt.bytes[pt.row0 * full_cols .. (pt.row0 + pt.n) * full_cols];
            cuda.fp8.packCodes(all_codes[row_off * k .. (row_off + npad) * k], raw_slice, pt.n, k);
            const scale_inv = try L.gpa.alloc(f32, st.bytes.len / 4);
            defer L.gpa.free(scale_inv);
            @memcpy(std.mem.sliceAsBytes(scale_inv), st.bytes);
            cuda.fp8.packScalesAt(all_scales[row_off * kg .. (row_off + npad) * kg], scale_inv, full_rows, full_cols, pt.row0, 0, pt.n, k);
            row_off += npad;
        }
        const d_codes = try L.put(all_codes);
        const d_scales = try L.put(std.mem.sliceAsBytes(all_scales));
        if (k % 128 == 0) {
            const kb = k / 128;
            const total_nb = (total_n + 127) / 128;
            const raw_codes = try L.gpa.alloc(u8, total_n * k);
            defer L.gpa.free(raw_codes);
            const raw_sc = try L.gpa.alloc(f32, total_nb * kb);
            defer L.gpa.free(raw_sc);
            var r_off: usize = 0;
            var nb_off: usize = 0;
            for (parts) |pt| {
                const wt = try L.ck.get(pt.w);
                const st = try L.ck.get(pt.s);
                const full_cols = wt.shape[1];
                const raw_slice = wt.bytes[pt.row0 * full_cols .. (pt.row0 + pt.n) * full_cols];
                @memcpy(raw_codes[r_off * k .. (r_off + pt.n) * k], raw_slice);
                const scale_inv = try L.gpa.alloc(f32, st.bytes.len / 4);
                defer L.gpa.free(scale_inv);
                @memcpy(std.mem.sliceAsBytes(scale_inv), st.bytes);
                const p_nb = (pt.n + 127) / 128;
                const full_kb = (full_cols + 127) / 128;
                for (0..p_nb) |r| {
                    const src_r = (pt.row0 / 128) + r;
                    @memcpy(raw_sc[(nb_off + r) * kb .. (nb_off + r + 1) * kb], scale_inv[src_r * full_kb ..][0..kb]);
                }
                r_off += pt.n;
                nb_off += p_nb;
            }
            const d_raw = try L.put(raw_codes);
            const d_raw_s = try L.put(std.mem.sliceAsBytes(raw_sc));
            try L.w.fp8_raw.put(L.gpa, d_codes, .{ .codes = d_raw, .scales = d_raw_s });
        }
        return cuda.fp8.weight(d_codes, d_scales, total_n, k);
    }

    fn loadMhcPrefix(L: *Loader, prefix: []const u8, comptime kind: []const u8) !MhcWeights {
        const c = &L.w.config;
        const fn_w = try L.putTransposedF32(try L.fmt1("{s}.{s}_hyper_connection.mapping_proj.weight", .{ prefix, kind }), c.hcDim(), c.hcMixOut());
        const base = try L.putRaw(try L.fmt1("{s}.{s}_hyper_connection.bias", .{ prefix, kind }));

        var sc: [3]f32 = undefined;
        const pre_t = try L.ck.get(try L.fmt1("{s}.{s}_hyper_connection.alpha_pre", .{ prefix, kind }));
        const post_t = try L.ck.get(try L.fmt1("{s}.{s}_hyper_connection.alpha_post", .{ prefix, kind }));
        const res_t = try L.ck.get(try L.fmt1("{s}.{s}_hyper_connection.alpha_res", .{ prefix, kind }));
        sc[0] = std.mem.bytesAsSlice(f32, pre_t.bytes)[0];
        sc[1] = std.mem.bytesAsSlice(f32, post_t.bytes)[0];
        sc[2] = std.mem.bytesAsSlice(f32, res_t.bytes)[0];
        const scale = try L.put(std.mem.sliceAsBytes(&sc));

        return .{ .fn_w = fn_w, .scale = scale, .base = base };
    }

    fn loadHcaPrefix(L: *Loader, prefix: []const u8, with_comp: bool) !HcaWeights {
        const c = &L.w.config;
        const sh = &L.w.share;

        var b0: [256]u8 = undefined;
        var b1: [256]u8 = undefined;
        var b2: [256]u8 = undefined;
        var b3: [256]u8 = undefined;
        const qdown_kv = try L.loadFp8Concat(&.{
            .{
                .w = try std.fmt.bufPrint(&b0, "{s}.self_attn.linear_q_down_proj.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b1, "{s}.self_attn.linear_q_down_proj.weight_scale_inv", .{prefix}),
                .row0 = 0,
                .n = c.q_lora,
            },
            .{
                .w = try std.fmt.bufPrint(&b2, "{s}.self_attn.linear_kv_proj.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b3, "{s}.self_attn.linear_kv_proj.weight_scale_inv", .{prefix}),
                .row0 = 0,
                .n = c.head_dim,
            },
        }, c.hidden);
        const kv_proj = cuda.fp8.weight(
            qdown_kv.codes + c.q_lora * c.hidden,
            qdown_kv.scales + c.q_lora * (c.hidden / cuda.fp8.group) * 4,
            c.head_dim,
            c.hidden,
        );
        const q_norm = try L.putRaw(try L.fmt1("{s}.self_attn.q_layernorm.weight", .{prefix}));
        const kv_norm = try L.putRaw(try L.fmt1("{s}.self_attn.kv_layernorm.weight", .{prefix}));
        const qup_ogate = try L.loadFp8Concat(&.{
            .{
                .w = try std.fmt.bufPrint(&b0, "{s}.self_attn.linear_q_up_proj.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b1, "{s}.self_attn.linear_q_up_proj.weight_scale_inv", .{prefix}),
                .row0 = sh.head_lo * c.head_dim,
                .n = sh.heads * c.head_dim,
            },
            .{
                .w = try std.fmt.bufPrint(&b2, "{s}.self_attn.gate_proj.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b3, "{s}.self_attn.gate_proj.weight_scale_inv", .{prefix}),
                .row0 = sh.o_lo,
                .n = sh.o_width,
            },
        }, c.q_lora);

        var wo_a: [8]QWeight = undefined;
        const group_k = (c.heads / c.o_groups) * c.head_dim;
        for (0..sh.groups) |g| {
            const global_g = sh.group_lo + g;
            wo_a[g] = try L.loadFp8Slice(
                try L.fmt1("{s}.self_attn.linear_o_group_proj", .{prefix}),
                try L.fmt2("{s}.self_attn.linear_o_group_proj_scale_inv", .{prefix}),
                global_g * c.o_lora, c.o_lora, 0, group_k,
            );
        }

        const wo_b = try L.loadFp8Slice(
            try L.fmt1("{s}.self_attn.o_proj.weight", .{prefix}),
            try L.fmt2("{s}.self_attn.o_proj.weight_scale_inv", .{prefix}),
            0, c.hidden, sh.o_lo, sh.o_width,
        );
        const vha_u = try L.putRowSliceBf16(try L.fmt1("{s}.self_attn.vha_postmix_U", .{prefix}), sh.head_lo, sh.heads, c.vha_rank);
        const vha_v = try L.putRowSliceBf16(try L.fmt1("{s}.self_attn.vha_postmix_V", .{prefix}), sh.head_lo, sh.heads, c.vha_rank);
        const attn_sink = try L.putRowSliceF32(try L.fmt1("{s}.self_attn.core_attention.attn_sink", .{prefix}), sh.head_lo, sh.heads, 1);

        var hca: HcaWeights = .{
            .qdown_kv = qdown_kv,
            .q_norm = q_norm,
            .qup_ogate = qup_ogate,
            .kv_proj = kv_proj,
            .kv_norm = kv_norm,
            .wo_a = wo_a,
            .wo_b = wo_b,
            .vha_u = vha_u,
            .vha_v = vha_v,
            .attn_sink = attn_sink,
        };
        if (with_comp) {
            hca.comp_ape = try L.putRaw(try L.fmt1("{s}.self_attn.core_attention.compressor.ape", .{prefix}));
            hca.comp_wkv_wgate = try L.loadBf16LanePair(
                try L.fmt1("{s}.self_attn.core_attention.compressor.linear_wkv.weight", .{prefix}),
                try L.fmt2("{s}.self_attn.core_attention.compressor.linear_wgate.weight", .{prefix}),
                c.head_dim,
                c.hidden,
            );
            hca.comp_norm = try L.putRaw(try L.fmt1("{s}.self_attn.core_attention.compressor.norm.weight", .{prefix}));
        }
        return hca;
    }

    fn loadDsa(L: *Loader, l: u32) !DsaWeights {
        const c = &L.w.config;
        const sh = &L.w.share;

        var b0: [256]u8 = undefined;
        var b1: [256]u8 = undefined;
        var b2: [256]u8 = undefined;
        var b3: [256]u8 = undefined;
        const qa_kva = try L.loadFp8Concat(&.{
            .{
                .w = try std.fmt.bufPrint(&b0, "model.layers.{d}.self_attn.q_a_proj.weight", .{l}),
                .s = try std.fmt.bufPrint(&b1, "model.layers.{d}.self_attn.q_a_proj.weight_scale_inv", .{l}),
                .row0 = 0,
                .n = c.mla_q_lora,
            },
            .{
                .w = try std.fmt.bufPrint(&b2, "model.layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{l}),
                .s = try std.fmt.bufPrint(&b3, "model.layers.{d}.self_attn.kv_a_proj_with_mqa.weight_scale_inv", .{l}),
                .row0 = 0,
                .n = c.mla_kv_lora + c.mla_qk_rope,
            },
        }, c.hidden);
        const q_a_norm = try L.putRaw(try L.fmt1("model.layers.{d}.self_attn.q_a_layernorm.weight", .{l}));
        const q_head = c.mla_qk_nope + c.mla_qk_rope;
        const qb_ogate = try L.loadFp8Concat(&.{
            .{
                .w = try std.fmt.bufPrint(&b0, "model.layers.{d}.self_attn.q_b_proj.weight", .{l}),
                .s = try std.fmt.bufPrint(&b1, "model.layers.{d}.self_attn.q_b_proj.weight_scale_inv", .{l}),
                .row0 = sh.head_lo * q_head,
                .n = sh.heads * q_head,
            },
            .{
                .w = try std.fmt.bufPrint(&b2, "model.layers.{d}.self_attn.gate_proj.weight", .{l}),
                .s = try std.fmt.bufPrint(&b3, "model.layers.{d}.self_attn.gate_proj.weight_scale_inv", .{l}),
                .row0 = sh.dsa_o_lo,
                .n = sh.dsa_o_width,
            },
        }, c.mla_q_lora);
        const kv_a_norm = try L.putRaw(try L.fmt1("model.layers.{d}.self_attn.kv_a_layernorm.weight", .{l}));

        const kvb_w = try L.ck.get(try L.fmt1("model.layers.{d}.self_attn.kv_b_proj.weight", .{l}));
        const kvb_s = try L.ck.get(try L.fmt2("model.layers.{d}.self_attn.kv_b_proj.weight_scale_inv", .{l}));
        const kvb_scales = std.mem.bytesAsSlice(f32, kvb_s.bytes);
        const kvb_kb = (c.mla_kv_lora + 127) / 128;
        const head_stride = c.mla_qk_nope + c.mla_v_dim;
        const wk_host = try L.gpa.alloc(u16, sh.heads * c.mla_qk_nope * c.mla_kv_lora);
        defer L.gpa.free(wk_host);
        const wv_host = try L.gpa.alloc(u16, sh.heads * c.mla_v_dim * c.mla_kv_lora);
        defer L.gpa.free(wv_host);
        for (0..sh.heads) |lh| {
            const gh = sh.head_lo + lh;
            const row_base = gh * head_stride;
            for (0..c.mla_qk_nope) |d| {
                const gr = row_base + d;
                const sr = gr / 128;
                for (0..c.mla_kv_lora) |col| {
                    const qv = kvb_w.bytes[gr * c.mla_kv_lora + col];
                    const sc = kvb_scales[sr * kvb_kb + col / 128];
                    wk_host[(lh * c.mla_qk_nope + d) * c.mla_kv_lora + col] = f32ToBf16(e4m3ToF32(qv) * sc);
                }
            }
            for (0..c.mla_v_dim) |d| {
                const gr = row_base + c.mla_qk_nope + d;
                const sr = gr / 128;
                for (0..c.mla_kv_lora) |col| {
                    const qv = kvb_w.bytes[gr * c.mla_kv_lora + col];
                    const sc = kvb_scales[sr * kvb_kb + col / 128];
                    wv_host[(lh * c.mla_v_dim + d) * c.mla_kv_lora + col] = f32ToBf16(e4m3ToF32(qv) * sc);
                }
            }
        }
        const wk = try L.put(std.mem.sliceAsBytes(wk_host));
        const wv = try L.put(std.mem.sliceAsBytes(wv_host));

        const o_proj = try L.loadFp8Slice(
            try L.fmt1("model.layers.{d}.self_attn.o_proj.weight", .{l}),
            try L.fmt2("model.layers.{d}.self_attn.o_proj.weight_scale_inv", .{l}),
            0, c.hidden, sh.dsa_o_lo, sh.dsa_o_width,
        );
        const vha_u = try L.putRowSliceBf16(try L.fmt1("model.layers.{d}.self_attn.vha_postmix_U", .{l}), sh.head_lo, sh.heads, c.vha_rank);
        const vha_v = try L.putRowSliceBf16(try L.fmt1("model.layers.{d}.self_attn.vha_postmix_V", .{l}), sh.head_lo, sh.heads, c.vha_rank);
        const sinks = try L.putBf16AsF32(try L.fmt1("model.layers.{d}.self_attn.core_attention.softmax_offset", .{l}), sh.head_lo, sh.heads);

        const idx_wq_b = try L.loadFp8Slice(
            try L.fmt1("model.layers.{d}.self_attn.core_attention.indexer.wq_b.weight", .{l}),
            try L.fmt2("model.layers.{d}.self_attn.core_attention.indexer.wq_b.weight_scale_inv", .{l}),
            0, c.index_heads * c.index_dim, 0, c.mla_q_lora,
        );
        const idx_wk = try L.loadBf16Lane(try L.fmt1("model.layers.{d}.self_attn.core_attention.indexer.wk.weight", .{l}), 0, c.index_dim, c.hidden);
        const idx_k_norm_w = try L.putRaw(try L.fmt1("model.layers.{d}.self_attn.core_attention.indexer.k_norm.weight", .{l}));
        const idx_k_norm_b = try L.putRaw(try L.fmt1("model.layers.{d}.self_attn.core_attention.indexer.k_norm.bias", .{l}));
        const idx_weights_proj = try L.loadBf16Lane(try L.fmt1("model.layers.{d}.self_attn.core_attention.indexer.weights_proj.weight", .{l}), 0, c.index_heads, c.hidden);

        return .{
            .qa_kva = qa_kva,
            .q_a_norm = q_a_norm,
            .qb_ogate = qb_ogate,
            .kv_a_norm = kv_a_norm,
            .wk = wk,
            .wv = wv,
            .o_proj = o_proj,
            .vha_u = vha_u,
            .vha_v = vha_v,
            .sinks = sinks,
            .idx_wq_b = idx_wq_b,
            .idx_wk = idx_wk,
            .idx_k_norm_w = idx_k_norm_w,
            .idx_k_norm_b = idx_k_norm_b,
            .idx_weights_proj = idx_weights_proj,
        };
    }

    fn loadDenseMlp(L: *Loader, l: u32) !DenseMlpWeights {
        const c = &L.w.config;
        const sh = &L.w.share;
        var b0: [256]u8 = undefined;
        var b1: [256]u8 = undefined;
        var b2: [256]u8 = undefined;
        var b3: [256]u8 = undefined;
        const gate_up = try L.loadFp8Concat(&.{
            .{
                .w = try std.fmt.bufPrint(&b0, "model.layers.{d}.mlp.gate_proj.weight", .{l}),
                .s = try std.fmt.bufPrint(&b1, "model.layers.{d}.mlp.gate_proj.weight_scale_inv", .{l}),
                .row0 = sh.dense_lo,
                .n = sh.dense_width,
            },
            .{
                .w = try std.fmt.bufPrint(&b2, "model.layers.{d}.mlp.up_proj.weight", .{l}),
                .s = try std.fmt.bufPrint(&b3, "model.layers.{d}.mlp.up_proj.weight_scale_inv", .{l}),
                .row0 = sh.dense_lo,
                .n = sh.dense_width,
            },
        }, c.hidden);
        const down = try L.loadFp8Slice(
            try L.fmt1("model.layers.{d}.mlp.down_proj.weight", .{l}),
            try L.fmt2("model.layers.{d}.mlp.down_proj.weight_scale_inv", .{l}),
            0, c.hidden, sh.dense_lo, sh.dense_width,
        );
        return .{ .gate_up = gate_up, .down = down };
    }

    fn ensureMoeStage(L: *Loader, w13_q_all: usize, w13_s_all: usize, w2_q_all: usize, w2_s_all: usize) !void {
        if (L.moe_w1_q.len < w13_q_all) {
            if (L.moe_w1_q.len > 0) L.gpa.free(L.moe_w1_q);
            L.moe_w1_q = try L.gpa.alloc(u8, w13_q_all);
        }
        if (L.moe_w1_s.len < w13_s_all) {
            if (L.moe_w1_s.len > 0) L.gpa.free(L.moe_w1_s);
            L.moe_w1_s = try L.gpa.alloc(u8, w13_s_all);
        }
        if (L.moe_w3_q.len < w13_q_all) {
            if (L.moe_w3_q.len > 0) L.gpa.free(L.moe_w3_q);
            L.moe_w3_q = try L.gpa.alloc(u8, w13_q_all);
        }
        if (L.moe_w3_s.len < w13_s_all) {
            if (L.moe_w3_s.len > 0) L.gpa.free(L.moe_w3_s);
            L.moe_w3_s = try L.gpa.alloc(u8, w13_s_all);
        }
        if (L.moe_w2_q.len < w2_q_all) {
            if (L.moe_w2_q.len > 0) L.gpa.free(L.moe_w2_q);
            L.moe_w2_q = try L.gpa.alloc(u8, w2_q_all);
        }
        if (L.moe_w2_s.len < w2_s_all) {
            if (L.moe_w2_s.len > 0) L.gpa.free(L.moe_w2_s);
            L.moe_w2_s = try L.gpa.alloc(u8, w2_s_all);
        }
    }

    fn loadMoePrefix(L: *Loader, prefix: []const u8) !MoeWeights {
        const c = &L.w.config;
        const sh = &L.w.share;

        const gate_packed = try L.loadBf16LanePair(
            try L.fmt1("{s}.block_sparse_moe.gate.weight", .{prefix}),
            try L.fmt2("{s}.block_sparse_moe.gate.weight_1", .{prefix}),
            c.experts,
            c.hidden,
        );
        const gate_raw0 = try L.putRaw(try L.fmt1("{s}.block_sparse_moe.gate.weight", .{prefix}));
        const gate_raw1 = try L.putRaw(try L.fmt1("{s}.block_sparse_moe.gate.weight_1", .{prefix}));
        const bias = try L.putRaw(try L.fmt1("{s}.block_sparse_moe.e_score_correction_bias", .{prefix}));
        const routed_scale = try L.putRaw(try L.fmt1("{s}.block_sparse_moe.gate.routed_scaling_factor_param", .{prefix}));

        var b0: [256]u8 = undefined;
        var b1: [256]u8 = undefined;
        var b2: [256]u8 = undefined;
        var b3: [256]u8 = undefined;
        var b4: [256]u8 = undefined;
        var b5: [256]u8 = undefined;
        const fc1_shared_w13 = try L.loadFp8Concat(&.{
            .{
                .w = try std.fmt.bufPrint(&b0, "{s}.block_sparse_moe.fc1_latent_proj.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b1, "{s}.block_sparse_moe.fc1_latent_proj.weight_scale_inv", .{prefix}),
                .row0 = 0,
                .n = c.moe_latent,
            },
            .{
                .w = try std.fmt.bufPrint(&b2, "{s}.block_sparse_moe.shared_experts.w1.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b3, "{s}.block_sparse_moe.shared_experts.w1.weight_scale_inv", .{prefix}),
                .row0 = sh.moe_lo,
                .n = sh.moe_width,
            },
            .{
                .w = try std.fmt.bufPrint(&b4, "{s}.block_sparse_moe.shared_experts.w3.weight", .{prefix}),
                .s = try std.fmt.bufPrint(&b5, "{s}.block_sparse_moe.shared_experts.w3.weight_scale_inv", .{prefix}),
                .row0 = sh.moe_lo,
                .n = sh.moe_width,
            },
        }, c.hidden);
        const latent_norm = try L.putRaw(try L.fmt1("{s}.block_sparse_moe.latent_norm.weight", .{prefix}));
        const fc2 = try L.loadFp8Slice(
            try L.fmt1("{s}.block_sparse_moe.fc2_latent_proj.weight", .{prefix}),
            try L.fmt2("{s}.block_sparse_moe.fc2_latent_proj.weight_scale_inv", .{prefix}),
            0, c.hidden, 0, c.moe_latent,
        );
        const shared_w2 = try L.loadFp8Slice(
            try L.fmt1("{s}.block_sparse_moe.shared_experts.w2.weight", .{prefix}),
            try L.fmt2("{s}.block_sparse_moe.shared_experts.w2.weight_scale_inv", .{prefix}),
            0, c.hidden, sh.moe_lo, sh.moe_width,
        );

        const E = c.experts;
        const w13_q_row = c.moe_latent / 2;
        const w13_s_row = c.moe_latent / 32;
        const w13_q_exp = sh.moe_width * w13_q_row;
        const w13_s_exp = sh.moe_width * w13_s_row;

        const w2_q_full_row = c.moe_intermediate / 2;
        const w2_s_full_row = c.moe_intermediate / 32;
        const w2_q_loc_row = sh.moe_width / 2;
        const w2_s_loc_row = sh.moe_width / 32;
        const w2_q_exp = c.moe_latent * w2_q_loc_row;
        const w2_s_exp = c.moe_latent * w2_s_loc_row;

        try L.ensureMoeStage(E * w13_q_exp, E * w13_s_exp, E * w2_q_exp, E * w2_s_exp);

        // Stage all 512 experts on host, then 6 bulk H2D transfers (vs 3,072 individual uploads).
        for (0..E) |e| {
            const w1q_t = try L.ck.get(try L.fmt1("{s}.block_sparse_moe.experts.{d}.w1.weight", .{ prefix, e }));
            const w1s_t = try L.ck.get(try L.fmt1("{s}.block_sparse_moe.experts.{d}.w1.weight_scale_inv", .{ prefix, e }));
            @memcpy(L.moe_w1_q[e * w13_q_exp .. (e + 1) * w13_q_exp], w1q_t.bytes[sh.moe_lo * w13_q_row .. (sh.moe_lo + sh.moe_width) * w13_q_row]);
            @memcpy(L.moe_w1_s[e * w13_s_exp .. (e + 1) * w13_s_exp], w1s_t.bytes[sh.moe_lo * w13_s_row .. (sh.moe_lo + sh.moe_width) * w13_s_row]);

            const w3q_t = try L.ck.get(try L.fmt1("{s}.block_sparse_moe.experts.{d}.w3.weight", .{ prefix, e }));
            const w3s_t = try L.ck.get(try L.fmt1("{s}.block_sparse_moe.experts.{d}.w3.weight_scale_inv", .{ prefix, e }));
            @memcpy(L.moe_w3_q[e * w13_q_exp .. (e + 1) * w13_q_exp], w3q_t.bytes[sh.moe_lo * w13_q_row .. (sh.moe_lo + sh.moe_width) * w13_q_row]);
            @memcpy(L.moe_w3_s[e * w13_s_exp .. (e + 1) * w13_s_exp], w3s_t.bytes[sh.moe_lo * w13_s_row .. (sh.moe_lo + sh.moe_width) * w13_s_row]);

            const w2q_t = try L.ck.get(try L.fmt1("{s}.block_sparse_moe.experts.{d}.w2.weight", .{ prefix, e }));
            const w2s_t = try L.ck.get(try L.fmt1("{s}.block_sparse_moe.experts.{d}.w2.weight_scale_inv", .{ prefix, e }));
            if (sh.world == 1) {
                @memcpy(L.moe_w2_q[e * w2_q_exp .. (e + 1) * w2_q_exp], w2q_t.bytes[0..w2_q_exp]);
                @memcpy(L.moe_w2_s[e * w2_s_exp .. (e + 1) * w2_s_exp], w2s_t.bytes[0..w2_s_exp]);
            } else {
                const q_col0 = sh.moe_lo / 2;
                const s_col0 = sh.moe_lo / 32;
                const q_dst = L.moe_w2_q[e * w2_q_exp .. (e + 1) * w2_q_exp];
                const s_dst = L.moe_w2_s[e * w2_s_exp .. (e + 1) * w2_s_exp];
                for (0..c.moe_latent) |row| {
                    const qs = row * w2_q_full_row + q_col0;
                    @memcpy(q_dst[row * w2_q_loc_row .. (row + 1) * w2_q_loc_row], w2q_t.bytes[qs .. qs + w2_q_loc_row]);
                    const ss = row * w2_s_full_row + s_col0;
                    @memcpy(s_dst[row * w2_s_loc_row .. (row + 1) * w2_s_loc_row], w2s_t.bytes[ss .. ss + w2_s_loc_row]);
                }
            }
        }

        const w1_q = try L.put(L.moe_w1_q[0 .. E * w13_q_exp]);
        const w1_s = try L.put(L.moe_w1_s[0 .. E * w13_s_exp]);
        const w3_q = try L.put(L.moe_w3_q[0 .. E * w13_q_exp]);
        const w3_s = try L.put(L.moe_w3_s[0 .. E * w13_s_exp]);
        const w2_q = try L.put(L.moe_w2_q[0 .. E * w2_q_exp]);
        const w2_s = try L.put(L.moe_w2_s[0 .. E * w2_s_exp]);

        return .{
            .gate_packed = gate_packed,
            .gate_raw0 = gate_raw0,
            .gate_raw1 = gate_raw1,
            .bias = bias,
            .routed_scale = routed_scale,
            .fc1_shared_w13 = fc1_shared_w13,
            .latent_norm = latent_norm,
            .fc2_latent = fc2,
            .shared_w2 = shared_w2,
            .w1_q = w1_q,
            .w1_s = w1_s,
            .w3_q = w3_q,
            .w3_s = w3_s,
            .w2_q = w2_q,
            .w2_s = w2_s,
        };
    }

    fn loadLayer(L: *Loader, l: u32) !LayerWeights {
        const c = &L.w.config;
        var pbuf: [64]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&pbuf, "model.layers.{d}", .{l});
        const attn_norm = try L.putRaw(try L.fmt1("{s}.input_layernorm.weight", .{prefix}));
        const ffn_norm = try L.putRaw(try L.fmt1("{s}.post_attention_layernorm.weight", .{prefix}));
        const hc_attn = try L.loadMhcPrefix(prefix, "self_attention");
        const hc_ffn = try L.loadMhcPrefix(prefix, "mlp");

        return .{
            .attn_norm = attn_norm,
            .ffn_norm = ffn_norm,
            .hc_attn = hc_attn,
            .hc_ffn = hc_ffn,
            .attn = if (c.isDsa(l)) .{ .dsa = try L.loadDsa(l) } else .{ .hca = try L.loadHcaPrefix(prefix, true) },
            .ffn = if (c.isMoe(l)) .{ .moe = try L.loadMoePrefix(prefix) } else .{ .dense = try L.loadDenseMlp(l) },
        };
    }

    fn loadDspark(L: *Loader) !DsparkWeights {
        const c = &L.w.config;
        const sh = &L.w.share;

        const feature_proj = try L.loadFp8Slice(
            "model.mtp.feature_projection.weight",
            "model.mtp.feature_projection.weight_scale_inv",
            0, c.hidden, 0, 3 * c.hidden,
        );
        const feature_norm = try L.putRaw("model.mtp.feature_norm.weight");
        const hc_head_fn = try L.putTransposedF32("model.mtp.mhc_contract.hc_head_fn", c.hcDim(), c.hc_mult);
        const hc_head_base = try L.putRaw("model.mtp.mhc_contract.hc_head_base");
        const hc_head_scale = try L.putRaw("model.mtp.mhc_contract.hc_head_scale");
        const norm = try L.putRaw("model.mtp.norm.weight");
        const markov_w1 = try L.putTransposedBf16("model.mtp.markov_head.markov_w1.weight", 256, c.vocab);
        const markov_w2 = try L.putRowSliceBf16("model.mtp.markov_head.markov_w2.weight", sh.vocab_lo, sh.vocab_width, 256);

        var layers: [3]DsparkLayerWeights = undefined;
        for (0..3) |i| {
            var pbuf: [64]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&pbuf, "model.mtp.{d}", .{i});
            layers[i] = .{
                .attn_norm = try L.putRaw(try L.fmt1("{s}.input_layernorm.weight", .{prefix})),
                .ffn_norm = try L.putRaw(try L.fmt1("{s}.post_attention_layernorm.weight", .{prefix})),
                .hc_attn = try L.loadMhcPrefix(prefix, "self_attention"),
                .hc_ffn = try L.loadMhcPrefix(prefix, "mlp"),
                .hca = try L.loadHcaPrefix(prefix, false),
                .moe = try L.loadMoePrefix(prefix),
            };
        }

        return .{
            .feature_proj = feature_proj,
            .feature_norm = feature_norm,
            .hc_head_fn = hc_head_fn,
            .hc_head_base = hc_head_base,
            .hc_head_scale = hc_head_scale,
            .norm = norm,
            .markov_w1 = markov_w1,
            .markov_w2 = markov_w2,
            .layers = layers,
        };
    }
};

test "e4m3ToF32 decodes exact OCP FP8 E4M3 values" {
    try std.testing.expectEqual(@as(f32, 0.0), e4m3ToF32(0x00));
    try std.testing.expectEqual(@as(f32, 1.0), e4m3ToF32(0x38));
    try std.testing.expectEqual(@as(f32, -1.5), e4m3ToF32(0xBC));
    try std.testing.expectEqual(@as(f32, 448.0), e4m3ToF32(0x7E));
}
