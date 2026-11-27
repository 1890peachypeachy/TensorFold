// ERNIE 5.5 Flash Preview MLA-DSA (6 layers: 7, 14, 21, 28, 35, 42):
//   - MLA KV cache [capacity, 576] BF16 (512 normed kv_c + 64 roped k_pe)
//   - Lightning Indexer K cache [capacity, 128] BF16 (LayerNorm + NeoX RoPE-64 + Hadamard-128)
//   - Absorbed Q [rows, H_loc, 576] = [q_nope @ wk, q_rope]
//   - Multi-head MQA MLA-DSA attention over SWA-128 + top-2048 indexer tokens + softmax_offset sink
//   - Expanded V [rows, H_loc, 256] = o_lat @ wv^T

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

}  // namespace

// Store normed kv_c [rows, 512] + roped k_pe [rows, 64] into mla_kv[positions[r], 0..575]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_dsa_kv_store(
    const __nv_bfloat16* __restrict__ kv_c,  // [rows, 512]
    const __nv_bfloat16* __restrict__ k_pe,  // [rows, 64] (already GPT-J roped)
    const int32_t* __restrict__ positions,   // [rows]
    __nv_bfloat16* __restrict__ mla_kv,      // [capacity, 576]
    int rows
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int pos = positions[r];
    __nv_bfloat16* dst = mla_kv + (int64_t)pos * 576;
    const __nv_bfloat16* sc = kv_c + (int64_t)r * 512;
    const __nv_bfloat16* sp = k_pe + (int64_t)r * 64;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) dst[i] = sc[i];
    if (threadIdx.x < 64) dst[512 + threadIdx.x] = sp[threadIdx.x];
}

// Store indexer key k_idx [rows, 128] into idx_k[positions[r], 0..127]
// Launch: grid=(rows), block=(128)
extern "C" __global__ void tf_ernie55_dsa_idx_k_store(
    const __nv_bfloat16* __restrict__ k_idx,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ idx_k,
    int rows
) {
    int r = blockIdx.x;
    int d = threadIdx.x;
    if (r >= rows || d >= 128) return;
    idx_k[(int64_t)positions[r] * 128 + d] = k_idx[(int64_t)r * 128 + d];
}

// Absorb Q into latent space:
//   q     : [rows, H_loc, 256] BF16 (192 nope + 64 rope)
//   wk    : [H_loc, 192, 512] BF16 (from kv_b_proj)
//   q_abs : [rows, H_loc, 576] BF16 (512 absorbed + 64 rope)
// Launch: grid=(ceil(512/4), H_loc, rows), block=(128) -> 1 warp per output latent dim l (0..511), plus rope copy
extern "C" __global__ void tf_ernie55_dsa_absorb_q(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ wk,
    __nv_bfloat16* __restrict__ q_abs,
    int rows,
    int H_loc
) {
    int r = blockIdx.z;
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int l = blockIdx.x * 4 + warp;
    if (r >= rows || h >= H_loc || l >= 512) return;

    const __nv_bfloat16* qp = q + ((int64_t)r * H_loc + h) * 256;
    const __nv_bfloat16* wp = wk + (int64_t)h * 192 * 512 + l;
    float acc = 0.0f;
    // 192 / 32 = 6 elems per lane
    for (int d = lane; d < 192; d += 32) {
        acc = fmaf(bf2f(qp[d]), bf2f(wp[(int64_t)d * 512]), acc);
    }
    acc = warp_sum(acc);
    __nv_bfloat16* dst = q_abs + ((int64_t)r * H_loc + h) * 576;
    if (lane == 0) {
        dst[l] = f2bf(acc);
        if (l < 64) {
            dst[512 + l] = qp[192 + l];
        }
    }
}

// Expand V from latent space:
//   o_lat : [rows, H_loc, 512] BF16
//   wv    : [H_loc, 256, 512] BF16 (row-major [h, d, l]!)
//   o_v   : [rows, H_loc, 256] BF16
// Launch: grid=(ceil(256/4), H_loc, rows), block=(128) -> 1 warp per output v_dim d (0..255)
extern "C" __global__ void tf_ernie55_dsa_expand_v(
    const __nv_bfloat16* __restrict__ o_lat,
    const __nv_bfloat16* __restrict__ wv,
    __nv_bfloat16* __restrict__ o_v,
    int rows,
    int H_loc
) {
    int r = blockIdx.z;
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int d = blockIdx.x * 4 + warp;
    if (r >= rows || h >= H_loc || d >= 256) return;

    const __nv_bfloat16* op = o_lat + ((int64_t)r * H_loc + h) * 512;
    const __nv_bfloat16* wp = wv + ((int64_t)h * 256 + d) * 512;
    float acc = 0.0f;
    for (int l = lane * 4; l < 512; l += 128) {
        uint2 ov = *reinterpret_cast<const uint2*>(op + l);
        uint2 wv2 = *reinterpret_cast<const uint2*>(wp + l);
        const __nv_bfloat16* o2 = reinterpret_cast<const __nv_bfloat16*>(&ov);
        const __nv_bfloat16* w2 = reinterpret_cast<const __nv_bfloat16*>(&wv2);
        acc = fmaf(bf2f(o2[0]), bf2f(w2[0]), acc);
        acc = fmaf(bf2f(o2[1]), bf2f(w2[1]), acc);
        acc = fmaf(bf2f(o2[2]), bf2f(w2[2]), acc);
        acc = fmaf(bf2f(o2[3]), bf2f(w2[3]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        o_v[((int64_t)r * H_loc + h) * 256 + d] = f2bf(acc);
    }
}

// Lightning Indexer scores: for query row r at pos, score every older token t in [0 .. pos - window_size]:
//   score[r, t] = sum_{h=0..63} (weights[r, h] * weights_scale) * max(0, dot(q_idx[r, h, 0..127], idx_k[t, 0..127]))
// Launch: grid=(ceil(max_older / 4), rows), block=(128) -> 1 warp per key token t (only launched when max_older > topk)
extern "C" __global__ void tf_ernie55_dsa_indexer_scores(
    const __nv_bfloat16* __restrict__ q_idx,   // [rows, 64, 128]
    const __nv_bfloat16* __restrict__ idx_k,   // [capacity, 128]
    const __nv_bfloat16* __restrict__ weights, // [rows, 64]
    const int32_t* __restrict__ positions,     // [rows]
    float* __restrict__ scores,                // [rows, max_older]
    int rows,
    int max_older,
    int window_size,
    float weights_scale
) {
    int r = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int t = blockIdx.x * 4 + warp;
    if (r >= rows || t >= max_older) return;
    int n_older = positions[r] - window_size + 1;
    if (t >= n_older) {
        if (lane == 0) scores[(int64_t)r * max_older + t] = -1e38f;
        return;
    }

    // Load key k_idx[t, 0..127] into registers: 4 floats per lane!
    const __nv_bfloat16* kp = idx_k + (int64_t)t * 128 + lane * 4;
    uint2 kv = *reinterpret_cast<const uint2*>(kp);
    const __nv_bfloat16* k2 = reinterpret_cast<const __nv_bfloat16*>(&kv);
    float k0 = bf2f(k2[0]), k1 = bf2f(k2[1]), k2f = bf2f(k2[2]), k3 = bf2f(k2[3]);

    const __nv_bfloat16* qr = q_idx + (int64_t)r * 64 * 128;
    const __nv_bfloat16* wr = weights + (int64_t)r * 64;
    float total = 0.0f;
    for (int h = 0; h < 64; ++h) {
        const __nv_bfloat16* qh = qr + (int64_t)h * 128 + lane * 4;
        uint2 qv = *reinterpret_cast<const uint2*>(qh);
        const __nv_bfloat16* q2 = reinterpret_cast<const __nv_bfloat16*>(&qv);
        float d = bf2f(q2[0]) * k0 + bf2f(q2[1]) * k1 + bf2f(q2[2]) * k2f + bf2f(q2[3]) * k3;
        d = warp_sum(d);
        if (lane == 0 && d > 0.0f) {
            total = fmaf(bf2f(wr[h]) * weights_scale, d, total);
        }
    }
    if (lane == 0) {
        scores[(int64_t)r * max_older + t] = total;
    }
}

// Fast radix-threshold top-k selection for Lightning Indexer (selects top `topk` from `n_older`, appends SWA window).
// Uses parallel 256-bin float-key histogram in shared memory (O(n_older) in ~15 us even at 200K context!).
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_dsa_select_topk(
    const float* __restrict__ scores,      // [rows, max_older]
    const int32_t* __restrict__ positions, // [rows]
    int32_t* __restrict__ out_indices,     // [rows, topk + window_size]
    int32_t* __restrict__ out_counts,      // [rows]
    int rows,
    int max_older,
    int topk,
    int window_size
) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    if (r >= rows) return;

    int pos = positions[r];
    int width = topk + window_size;
    int32_t* dst = out_indices + (int64_t)r * width;

    // Fast path: full sequence fits inside topk + window_size -> exact dense attention over 0..pos!
    if (pos + 1 <= width) {
        int n = pos + 1;
        for (int i = tid; i < n; i += blockDim.x) dst[i] = i;
        if (tid == 0) out_counts[r] = n;
        return;
    }

    int n_older = pos - window_size + 1;
    const float* sr = scores + (int64_t)r * max_older;

    // Map IEEE-754 float -> order-preserving uint32 key
    auto f2ord = [](float f) -> uint32_t {
        uint32_t u = __float_as_uint(f);
        uint32_t mask = (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
        return u ^ mask;
    };

    // 4-pass MSB 8-bit radix threshold finder to locate exact k-th largest uint32 key!
    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_rem;
    __shared__ uint32_t s_write_cnt;
    if (tid == 0) {
        s_prefix = 0u;
        s_rem = (uint32_t)topk;
        s_write_cnt = 0u;
    }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - pass * 8;
        uint32_t pmask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        uint32_t pval = s_prefix;
        s_hist[tid] = 0u;
        __syncthreads();

        for (int i = tid; i < n_older; i += blockDim.x) {
            uint32_t k = f2ord(sr[i]);
            if ((k & pmask) == pval) {
                atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t rem = s_rem;
            for (int b = 255; b >= 0; --b) {
                uint32_t c = s_hist[b];
                if (c >= rem) {
                    s_prefix = pval | ((uint32_t)b << shift);
                    s_rem = rem;
                    break;
                }
                rem -= c;
            }
        }
        __syncthreads();
    }

    uint32_t thresh = s_prefix;
    // Pass A: collect all elements strictly greater than thresh
    for (int i = tid; i < n_older; i += blockDim.x) {
        if (f2ord(sr[i]) > thresh) {
            uint32_t slot = atomicAdd(&s_write_cnt, 1u);
            if (slot < (uint32_t)topk) dst[slot] = i;
        }
    }
    __syncthreads();
    // Pass B: fill ties equal to thresh up to exact topk
    for (int i = tid; i < n_older; i += blockDim.x) {
        if (f2ord(sr[i]) == thresh) {
            uint32_t slot = atomicAdd(&s_write_cnt, 1u);
            if (slot < (uint32_t)topk) dst[slot] = i;
        }
    }
    __syncthreads();

    // Append SWA window [pos - window_size + 1 .. pos]
    int swa_start = pos - window_size + 1;
    for (int i = tid; i < window_size; i += blockDim.x) {
        dst[topk + i] = swa_start + i;
    }
    if (tid == 0) out_counts[r] = width;
}

// High-throughput multi-head MQA MLA-DSA absorbed latent attention:
//   q_abs   : [rows, H_loc, 576] BF16 (512 latent + 64 rope)
//   mla_kv  : [capacity, 576] BF16
//   indices : nullable [rows, idx_stride] int32 (when null or pos+1 <= idx_stride, reads 0..pos directly!)
//   counts  : nullable [rows] int32
//   sinks   : [H_loc] F32 (from softmax_offset)
//   out_lat : [rows, H_loc, 512] BF16
//
// Each CTA has 256 threads = 8 warps, 2 heads per warp = 16 heads per CTA!
// Lane 0..31 owns 16 latent dims (d = lane*16..lane*16+15) + 2 rope dims (r = lane*2, lane*2+1)!
// Launch: grid=(ceil(H_loc / 16), rows), block=(256)
extern "C" __global__ void tf_ernie55_dsa_attn(
    const __nv_bfloat16* __restrict__ q_abs,
    const __nv_bfloat16* __restrict__ mla_kv,
    const int32_t* __restrict__ indices,   // nullable
    const int32_t* __restrict__ counts,    // nullable
    const float* __restrict__ sinks,       // [H_loc]
    const int32_t* __restrict__ positions, // [rows]
    __nv_bfloat16* __restrict__ out_lat,   // [rows, H_loc, 512]
    int rows,
    int H_loc,
    int idx_stride,
    float scale
) {
    int r = blockIdx.y;
    if (r >= rows) return;
    int h_base = blockIdx.x * 16;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int h0 = h_base + warp * 2;
    int h1 = h0 + 1;
    bool v0 = (h0 < H_loc);
    bool v1 = (h1 < H_loc);

    int pos = positions[r];
    bool use_idx = (indices != nullptr && counts != nullptr);
    int total_k = use_idx ? counts[r] : (pos + 1);
    const int32_t* idx_row = use_idx ? (indices + (int64_t)r * idx_stride) : nullptr;

    float q0[16], q1[16];
    float qr0[2], qr1[2];
    float acc0[16], acc1[16];
    const __nv_bfloat16* qp0 = q_abs + ((int64_t)r * H_loc + h0) * 576;
    const __nv_bfloat16* qp1 = q_abs + ((int64_t)r * H_loc + h1) * 576;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q0[i] = v0 ? bf2f(qp0[lane * 16 + i]) * scale : 0.0f;
        q1[i] = v1 ? bf2f(qp1[lane * 16 + i]) * scale : 0.0f;
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }
    qr0[0] = v0 ? bf2f(qp0[512 + lane * 2]) * scale : 0.0f;
    qr0[1] = v0 ? bf2f(qp0[512 + lane * 2 + 1]) * scale : 0.0f;
    qr1[0] = v1 ? bf2f(qp1[512 + lane * 2]) * scale : 0.0f;
    qr1[1] = v1 ? bf2f(qp1[512 + lane * 2 + 1]) * scale : 0.0f;

    float m0 = v0 ? sinks[h0] : -1e38f;
    float m1 = v1 ? sinks[h1] : -1e38f;
    float l0 = 1.0f;
    float l1 = 1.0f;

    constexpr int BK = 16;
    __shared__ __nv_bfloat16 s_kv[BK][576];

    for (int k0 = 0; k0 < total_k; k0 += BK) {
        int cnt = (total_k - k0 < BK) ? (total_k - k0) : BK;
        for (int t = 0; t < cnt; ++t) {
            int tok = use_idx ? idx_row[k0 + t] : (k0 + t);
            const __nv_bfloat16* src = mla_kv + (int64_t)tok * 576;
            // 256 threads load 576 BF16: first 256 uint32 (= 512 BF16), then 32 uint32 (= 64 BF16)
            reinterpret_cast<uint32_t*>(s_kv[t])[threadIdx.x] =
                reinterpret_cast<const uint32_t*>(src)[threadIdx.x];
            if (threadIdx.x < 32) {
                reinterpret_cast<uint32_t*>(s_kv[t] + 512)[threadIdx.x] =
                    reinterpret_cast<const uint32_t*>(src + 512)[threadIdx.x];
            }
        }
        __syncthreads();

        for (int t = 0; t < cnt; ++t) {
            const __nv_bfloat16* kp = &s_kv[t][lane * 16];
            const __nv_bfloat16* kr = &s_kv[t][512 + lane * 2];
            float kv[16];
            float kr0 = bf2f(kr[0]), kr1 = bf2f(kr[1]);
            float d0 = qr0[0] * kr0 + qr0[1] * kr1;
            float d1 = qr1[0] * kr0 + qr1[1] * kr1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                kv[i] = bf2f(kp[i]);
                d0 = fmaf(q0[i], kv[i], d0);
                d1 = fmaf(q1[i], kv[i], d1);
            }
            d0 = warp_allreduce_sum(d0);
            d1 = warp_allreduce_sum(d1);

            if (v0) {
                float nm = fmaxf(m0, d0);
                float alpha = expf(m0 - nm);
                float p = expf(d0 - nm);
                l0 = l0 * alpha + p;
#pragma unroll
                for (int i = 0; i < 16; ++i) acc0[i] = fmaf(p, kv[i], acc0[i] * alpha);
                m0 = nm;
            }
            if (v1) {
                float nm = fmaxf(m1, d1);
                float alpha = expf(m1 - nm);
                float p = expf(d1 - nm);
                l1 = l1 * alpha + p;
#pragma unroll
                for (int i = 0; i < 16; ++i) acc1[i] = fmaf(p, kv[i], acc1[i] * alpha);
                m1 = nm;
            }
        }
        __syncthreads();
    }

    if (v0) {
        float inv_l = 1.0f / l0;
        __nv_bfloat16* op = out_lat + ((int64_t)r * H_loc + h0) * 512 + lane * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) op[i] = f2bf(acc0[i] * inv_l);
    }
    if (v1) {
        float inv_l = 1.0f / l1;
        __nv_bfloat16* op = out_lat + ((int64_t)r * H_loc + h1) * 512 + lane * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) op[i] = f2bf(acc1[i] * inv_l);
    }
}

// Fused DSA kv_a [rows, 576] -> kv_a_layernorm(0..511) + GPT-J RoPE(512..575) -> mla_kv[positions[r], 0..575]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_dsa_kv_norm_rope_store(
    const __nv_bfloat16* __restrict__ kv_a,   // [rows, 576]
    const __nv_bfloat16* __restrict__ norm_w, // [512]
    const int32_t* __restrict__ positions,    // [rows]
    const float* __restrict__ inv_freq,       // [32]
    __nv_bfloat16* __restrict__ mla_kv,       // [capacity, 576]
    int rows,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int pos = positions[r];
    const __nv_bfloat16* src = kv_a + (int64_t)r * 576;
    __nv_bfloat16* dst = mla_kv + (int64_t)pos * 576;

    __shared__ float s_red[32];
    float v0 = bf2f(src[threadIdx.x * 2]);
    float v1 = bf2f(src[threadIdx.x * 2 + 1]);
    float sq = v0 * v0 + v1 * v1;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    sq = warp_sum(sq);
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? s_red[lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0) s_red[0] = rsqrtf(tot / 512.0f + eps);
    }
    __syncthreads();
    float rrms = s_red[0];

    dst[threadIdx.x * 2]     = f2bf(v0 * rrms * bf2f(norm_w[threadIdx.x * 2]));
    dst[threadIdx.x * 2 + 1] = f2bf(v1 * rrms * bf2f(norm_w[threadIdx.x * 2 + 1]));

    if (threadIdx.x < 32) {
        int pair = threadIdx.x;
        float e = bf2f(src[512 + pair * 2]);
        float o = bf2f(src[512 + pair * 2 + 1]);
        float theta = (float)pos * inv_freq[pair];
        float c, s;
        __sincosf(theta, &s, &c);
        dst[512 + pair * 2]     = f2bf(e * c - o * s);
        dst[512 + pair * 2 + 1] = f2bf(o * c + e * s);
    }
}
// Lightning Indexer scores on tensor cores (Spark, 2026-11-24; appended to dsa_attn.cu).
// Same contract as tf_ernie55_dsa_indexer_scores: score[r, t] = sum_h (w[r,h] * scale) * relu(q[r,h,:] . k[t,:])
// for t < n_older(r), -1e38 beyond. One block: one query row (its 64 heads) x 64 key tokens, four warps of 16 heads;
// mma.sync m16n8k16 bf16 -> fp32 over the 128 dims, then relu, the weighted head sum (lanes, then warps).
#define TF_IDX_LD 136  // bf16 row stride: 272 bytes, conflict-free fragment loads

__device__ __forceinline__ void idx_mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Launch: grid=(ceil(max_older / 64), rows), block=(128), smem = (64 + 64) * TF_IDX_LD * 2 + 4 * 64 * 4.
extern "C" __global__ void __launch_bounds__(128) tf_ernie55_dsa_indexer_scores_mma(
    const __nv_bfloat16* __restrict__ q_idx,    // [rows, 64, 128]
    const __nv_bfloat16* __restrict__ idx_k,    // [capacity, 128]
    const __nv_bfloat16* __restrict__ weights,  // [rows, 64]
    const int32_t* __restrict__ positions,      // [rows]
    float* __restrict__ scores,                 // [rows, max_older]
    int rows, int max_older, int window_size, float weights_scale) {
    const int r = blockIdx.y;
    const int t0 = blockIdx.x * 64;
    if (r >= rows || t0 >= max_older) return;
    const int n_older = positions[r] - window_size + 1;
    extern __shared__ __align__(16) __nv_bfloat16 sm[];
    __nv_bfloat16* qs = sm;                       // [64 heads][LD]
    __nv_bfloat16* ks = qs + 64 * TF_IDX_LD;      // [64 tokens][LD]
    float* red = reinterpret_cast<float*>(ks + 64 * TF_IDX_LD);  // [4 warps][64 tokens]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;

    if (t0 >= n_older) {  // the whole tile is past this row's older tokens
        for (int i = threadIdx.x; i < 64; i += blockDim.x)
            if (t0 + i < max_older) scores[(int64_t)r * max_older + t0 + i] = -1e38f;
        return;
    }
    // 64 x 128 bf16 each: 16 bytes a thread-step
    const __nv_bfloat16* qr = q_idx + (int64_t)r * 64 * 128;
    for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
        const int row = i >> 4, c = (i & 15) * 8;
        *reinterpret_cast<uint4*>(qs + row * TF_IDX_LD + c) = *reinterpret_cast<const uint4*>(qr + row * 128 + c);
        const int t = t0 + row;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t < n_older) v = *reinterpret_cast<const uint4*>(idx_k + (int64_t)t * 128 + c);
        *reinterpret_cast<uint4*>(ks + row * TF_IDX_LD + c) = v;
    }
    __syncthreads();

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 128; kk += 16) {
        uint32_t a[4];
        const __nv_bfloat16* ap = qs + (warp * 16 + g) * TF_IDX_LD + kk + tq * 2;
        a[0] = *reinterpret_cast<const uint32_t*>(ap);
        a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * TF_IDX_LD);
        a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
        a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * TF_IDX_LD + 8);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t b[2];
            const __nv_bfloat16* bp = ks + (j * 8 + g) * TF_IDX_LD + kk + tq * 2;
            b[0] = *reinterpret_cast<const uint32_t*>(bp);
            b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
            idx_mma16816(acc[j], a, b);
        }
    }
    // c0, c1: head warp*16 + g, tokens j*8 + tq*2 + {0,1}; c2, c3: head + 8
    const float w_lo = __bfloat162float(weights[(int64_t)r * 64 + warp * 16 + g]) * weights_scale;
    const float w_hi = __bfloat162float(weights[(int64_t)r * 64 + warp * 16 + g + 8]) * weights_scale;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float v = w_lo * fmaxf(acc[j][e], 0.0f) + w_hi * fmaxf(acc[j][e + 2], 0.0f);
            v += __shfl_xor_sync(0xffffffffu, v, 4);  // the 8 lanes (g) sharing tq: the warp's 16 heads
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            if (g == 0) red[warp * 64 + j * 8 + tq * 2 + e] = v;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        const int t = t0 + i;
        if (t >= max_older) continue;
        const float total = red[i] + red[64 + i] + red[128 + i] + red[192 + i];
        scores[(int64_t)r * max_older + t] = t < n_older ? total : -1e38f;
    }
}
// Per-head projections of the MLA-DSA layers on tensor cores (Spark, 2026-11-24; appended to dsa_attn.cu).
// C[r, h, n] = sum_k A[r, h, k] * B_h(k, n) for every local head h:
//   absorb_q: A = q [rows, H, 256] (first 192 dims), B_h = wk[h] stored [k = 192][n = 512] (BKN = 1),
//             C = q_abs [rows, H, 576] (first 512); the rope tail is copied by tf_ernie55_dsa_rope_copy.
//   expand_v: A = o_lat [rows, H, 512], B_h = wv[h] stored [n = 256][k = 512] (BKN = 0), C = o_v [rows, H, 256].
// One block: one head, 64 rows x 64 columns, K in steps of 64; four warps of 16 rows; mma.sync m16n8k16, fp32 sums.
#define TF_HG_LD 72

__device__ __forceinline__ void hg_mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int BKN>
__device__ __forceinline__ void head_gemm_body(const __nv_bfloat16* __restrict__ A, int lda, int a_head,
                                               const __nv_bfloat16* __restrict__ B, int b_head, int ldb,
                                               __nv_bfloat16* __restrict__ C, int ldc, int c_head, int rows, int N, int K) {
    const int n0 = blockIdx.x * 64, h = blockIdx.y, r0 = blockIdx.z * 64;
    __shared__ __align__(16) __nv_bfloat16 as[64 * TF_HG_LD];
    __shared__ __align__(16) __nv_bfloat16 bs[64 * TF_HG_LD];  // [n][k]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    const __nv_bfloat16* Bh = B + (int64_t)h * b_head;
    for (int k0 = 0; k0 < K; k0 += 64) {
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * 8; i += blockDim.x) {  // A: 64 rows x 64 k, 8 bf16 a step
            const int rr = i >> 3, c = (i & 7) * 8, r = r0 + rr;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < rows) v = *reinterpret_cast<const uint4*>(A + (int64_t)r * lda + (int64_t)h * a_head + k0 + c);
            *reinterpret_cast<uint4*>(as + rr * TF_HG_LD + c) = v;
        }
        if (BKN == 0) {  // B stored [n][k]: rows of k, straight in
            for (int i = threadIdx.x; i < 64 * 8; i += blockDim.x) {
                const int nn = i >> 3, c = (i & 7) * 8, n = n0 + nn;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (n < N) v = *reinterpret_cast<const uint4*>(Bh + (int64_t)n * ldb + k0 + c);
                *reinterpret_cast<uint4*>(bs + nn * TF_HG_LD + c) = v;
            }
        } else {  // B stored [k][n]: coalesced along n, transposed into bs[n][k]
            for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
                const int kk = i >> 6, nn = i & 63, n = n0 + nn;
                bs[nn * TF_HG_LD + kk] = n < N ? Bh[(int64_t)(k0 + kk) * ldb + n] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
            uint32_t a[4];
            const __nv_bfloat16* ap = as + (warp * 16 + g) * TF_HG_LD + kk + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * TF_HG_LD);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * TF_HG_LD + 8);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t b[2];
                const __nv_bfloat16* bp = bs + (j * 8 + g) * TF_HG_LD + kk + tq * 2;
                b[0] = *reinterpret_cast<const uint32_t*>(bp);
                b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
                hg_mma16816(acc[j], a, b);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + warp * 16 + g + (i >= 2 ? 8 : 0);
            const int n = n0 + j * 8 + tq * 2 + (i & 1);
            if (r < rows && n < N) C[(int64_t)r * ldc + (int64_t)h * c_head + n] = __float2bfloat16_rn(acc[j][i]);
        }
}

// Launch: grid=(ceil(512/64), H_loc, ceil(rows/64)), block=(128).
extern "C" __global__ void __launch_bounds__(128) tf_ernie55_dsa_absorb_q_mma(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ wk, __nv_bfloat16* __restrict__ q_abs, int rows, int H_loc) {
    head_gemm_body<1>(q, H_loc * 256, 256, wk, 192 * 512, 512, q_abs, H_loc * 576, 576, rows, 512, 192);
}

// q_abs[r, h, 512 .. 576) = q[r, h, 192 .. 256) (the rope dims pass through). Launch: grid=(H_loc, rows), block=(64).
extern "C" __global__ void tf_ernie55_dsa_rope_copy(const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ q_abs, int rows, int H_loc) {
    const int h = blockIdx.x, r = blockIdx.y, l = threadIdx.x;
    if (r >= rows || h >= H_loc || l >= 64) return;
    q_abs[((int64_t)r * H_loc + h) * 576 + 512 + l] = q[((int64_t)r * H_loc + h) * 256 + 192 + l];
}

// Launch: grid=(ceil(256/64), H_loc, ceil(rows/64)), block=(128).
extern "C" __global__ void __launch_bounds__(128) tf_ernie55_dsa_expand_v_mma(
    const __nv_bfloat16* __restrict__ o_lat, const __nv_bfloat16* __restrict__ wv, __nv_bfloat16* __restrict__ o_v, int rows, int H_loc) {
    head_gemm_body<0>(o_lat, H_loc * 512, 512, wv, 256 * 512, 512, o_v, H_loc * 256, 256, rows, 256, 512);
}


// --- Batch 1 fused DSA prep & SM121a Tensor-Core MMA FlashAttention + Split-K ---

// Fused DSA q_a + kv_a post-op:
//   packed [rows, 1600] ->
//     q_down [rows, 1024] = RMSNorm(packed[0..1024), q_a_norm)
//     mla_kv[pos, 0..512] = RMSNorm(packed[1024..1536), kv_a_norm)
//     mla_kv[pos, 512..576] = GPT-J RoPE(packed[1536..1600))
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_dsa_qkv_prep(
    const __nv_bfloat16* __restrict__ packed,    // [rows, 1600]
    const __nv_bfloat16* __restrict__ q_a_norm,  // [1024]
    const __nv_bfloat16* __restrict__ kv_a_norm, // [512]
    const int32_t* __restrict__ positions,       // [rows]
    const float* __restrict__ inv_freq,          // [32]
    __nv_bfloat16* __restrict__ q_down,          // [rows, 1024]
    __nv_bfloat16* __restrict__ mla_kv,          // [capacity, 576]
    int rows,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    __shared__ float s_red[8];

    const __nv_bfloat16* src = packed + (int64_t)r * 1600;

    // 1. q_a [0..1024)
    float q0 = bf2f(src[tid]);
    float q1 = bf2f(src[tid + 256]);
    float q2 = bf2f(src[tid + 512]);
    float q3 = bf2f(src[tid + 768]);
    float qsq = warp_sum(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (lane == 0) s_red[warp] = qsq;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? s_red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) s_red[0] = rsqrtf(t / 1024.0f + eps);
    }
    __syncthreads();
    float q_rrms = s_red[0];
    __nv_bfloat16* qdst = q_down + (int64_t)r * 1024;
    qdst[tid]       = f2bf(q0 * q_rrms * bf2f(q_a_norm[tid]));
    qdst[tid + 256] = f2bf(q1 * q_rrms * bf2f(q_a_norm[tid + 256]));
    qdst[tid + 512] = f2bf(q2 * q_rrms * bf2f(q_a_norm[tid + 512]));
    qdst[tid + 768] = f2bf(q3 * q_rrms * bf2f(q_a_norm[tid + 768]));

    // 2. kv_a [1024..1600): norm [0..512), GPT-J RoPE [512..576)
    const __nv_bfloat16* ksrc = src + 1024;
    float k0 = bf2f(ksrc[tid * 2]);
    float k1 = bf2f(ksrc[tid * 2 + 1]);
    float ksq = warp_sum(k0 * k0 + k1 * k1);
    __syncthreads();
    if (lane == 0) s_red[warp] = ksq;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? s_red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) s_red[0] = rsqrtf(t / 512.0f + eps);
    }
    __syncthreads();
    float k_rrms = s_red[0];
    int pos = positions[r];
    __nv_bfloat16* kdst = mla_kv + (int64_t)pos * 576;
    kdst[tid * 2]     = f2bf(k0 * k_rrms * bf2f(kv_a_norm[tid * 2]));
    kdst[tid * 2 + 1] = f2bf(k1 * k_rrms * bf2f(kv_a_norm[tid * 2 + 1]));

    if (tid < 32) {
        float e = bf2f(ksrc[512 + tid * 2]);
        float o = bf2f(ksrc[512 + tid * 2 + 1]);
        float theta = (float)pos * inv_freq[tid];
        float c, s;
        __sincosf(theta, &s, &c);
        kdst[512 + tid * 2]     = f2bf(e * c - o * s);
        kdst[512 + tid * 2 + 1] = f2bf(o * c + e * s);
    }
}

// SM121a Tensor-Core FlashAttention + Split-K for MLA-DSA (H_loc=16, D_QK=576, D_V=512, LD=584)
// Launch: grid=(splits, rows), block=(128), smem=59328
extern "C" __global__ void __launch_bounds__(128) tf_ernie55_dsa_attn_mma(
    const __nv_bfloat16* __restrict__ q_abs,   // [rows, 16, 576]
    const __nv_bfloat16* __restrict__ mla_kv,  // [capacity, 576]
    const int32_t* __restrict__ indices,       // nullable [rows, idx_stride]
    const int32_t* __restrict__ counts,        // nullable [rows]
    const float* __restrict__ sinks,           // [16]
    const int32_t* __restrict__ positions,     // [rows]
    __nv_bfloat16* __restrict__ out_lat,       // [rows, 16, 512] (written when splits == 1)
    float* __restrict__ opart,                 // [rows, splits, 16, 512] (written when splits > 1)
    float* __restrict__ mpart,                 // [rows, splits, 16]
    float* __restrict__ lpart,                 // [rows, splits, 16]
    int rows,
    int idx_stride,
    int splits,
    float scale
) {
    int r = blockIdx.y;
    int sp = blockIdx.x;
    if (r >= rows) return;

    int pos = positions[r];
    bool use_idx = (indices != nullptr && counts != nullptr);
    int total_k = use_idx ? counts[r] : (pos + 1);
    const int32_t* idx_row = use_idx ? (indices + (int64_t)r * idx_stride) : nullptr;

    int chunk = (total_k + splits - 1) / splits;
    int k_start = sp * chunk;
    int k_end = (k_start + chunk < total_k) ? (k_start + chunk) : total_k;

    constexpr int LD = 584;
    extern __shared__ unsigned char raw[];
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(raw);
    __nv_bfloat16* s_k = s_q + 16 * LD;
    float* s_sc = reinterpret_cast<float*>(s_k + 32 * LD);
    __nv_bfloat16* s_p = reinterpret_cast<__nv_bfloat16*>(s_sc + 16 * 32);
    float* s_m = reinterpret_cast<float*>(s_p + 16 * 32);
    float* s_l = s_m + 16;
    float* s_alpha = s_l + 16;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int tq = lane & 3;

    const __nv_bfloat16* qr = q_abs + (int64_t)r * 16 * 576;
    for (int idx = tid; idx < 16 * 72; idx += 128) {
        int h = idx / 72;
        int c = (idx - h * 72) * 8;
        *reinterpret_cast<uint4*>(s_q + h * LD + c) = *reinterpret_cast<const uint4*>(qr + h * 576 + c);
    }
    if (tid < 16) {
        if (sp == 0) {
            s_m[tid] = sinks[tid];
            s_l[tid] = 1.0f;
        } else {
            s_m[tid] = -1e38f;
            s_l[tid] = 0.0f;
        }
    }
    __syncthreads();

    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

    for (int k0 = k_start; k0 < k_end; k0 += 32) {
        int cnt = (k_end - k0 < 32) ? (k_end - k0) : 32;
        for (int idx = tid; idx < 32 * 72; idx += 128) {
            int t = idx / 72;
            int c = (idx - t * 72) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < cnt) {
                int tok = use_idx ? idx_row[k0 + t] : (k0 + t);
                v = *reinterpret_cast<const uint4*>(mla_kv + (int64_t)tok * 576 + c);
            }
            *reinterpret_cast<uint4*>(s_k + t * LD + c) = v;
        }
        __syncthreads();

        float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int n_base = warp * 8;
        for (int d = 0; d < 576; d += 16) {
            uint32_t a[4], b[2];
            const __nv_bfloat16* ap = s_q + g * LD + d + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD + 8);
            const __nv_bfloat16* bp = s_k + (n_base + g) * LD + d + tq * 2;
            b[0] = *reinterpret_cast<const uint32_t*>(bp);
            b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
            hg_mma16816(sc, a, b);
        }
        int c0 = n_base + tq * 2;
        s_sc[g * 32 + c0]       = sc[0] * scale;
        s_sc[g * 32 + c0 + 1]   = sc[1] * scale;
        s_sc[(g + 8) * 32 + c0]     = sc[2] * scale;
        s_sc[(g + 8) * 32 + c0 + 1] = sc[3] * scale;
        __syncthreads();

        if (tid < 16) {
            int h = tid;
            float m_old = s_m[h];
            float m_new = m_old;
            for (int t = 0; t < cnt; ++t) m_new = fmaxf(m_new, s_sc[h * 32 + t]);
            float alpha = (m_old > -1e37f) ? expf(m_old - m_new) : 0.0f;
            float l_new = s_l[h] * alpha;
            for (int t = 0; t < cnt; ++t) {
                float p = expf(s_sc[h * 32 + t] - m_new);
                s_p[h * 32 + t] = f2bf(p);
                l_new += p;
            }
            for (int t = cnt; t < 32; ++t) s_p[h * 32 + t] = f2bf(0.0f);
            s_m[h] = m_new;
            s_l[h] = l_new;
            s_alpha[h] = alpha;
        }
        __syncthreads();

        float a_lo = s_alpha[g];
        float a_hi = s_alpha[g + 8];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j][0] *= a_lo; acc[j][1] *= a_lo;
            acc[j][2] *= a_hi; acc[j][3] *= a_hi;
        }
        int d_warp = warp * 128;
        for (int kt = 0; kt < 32; kt += 16) {
            uint32_t a[4];
            const __nv_bfloat16* ap = s_p + g * 32 + kt + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * 32);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * 32 + 8);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                int dn = d_warp + j * 8 + g;
                uint16_t bv0 = *reinterpret_cast<const uint16_t*>(s_k + (kt + tq * 2) * LD + dn);
                uint16_t bv1 = *reinterpret_cast<const uint16_t*>(s_k + (kt + tq * 2 + 1) * LD + dn);
                uint16_t bv2 = *reinterpret_cast<const uint16_t*>(s_k + (kt + 8 + tq * 2) * LD + dn);
                uint16_t bv3 = *reinterpret_cast<const uint16_t*>(s_k + (kt + 8 + tq * 2 + 1) * LD + dn);
                uint32_t b[2] = { (uint32_t)bv0 | ((uint32_t)bv1 << 16), (uint32_t)bv2 | ((uint32_t)bv3 << 16) };
                hg_mma16816(acc[j], a, b);
            }
        }
        __syncthreads();
    }

    int d_warp = warp * 128;
    if (splits == 1) {
        float il0 = 1.0f / s_l[g];
        float il1 = 1.0f / s_l[g + 8];
        __nv_bfloat16* o0 = out_lat + ((int64_t)r * 16 + g) * 512;
        __nv_bfloat16* o1 = out_lat + ((int64_t)r * 16 + g + 8) * 512;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int dn = d_warp + j * 8 + tq * 2;
            o0[dn]     = f2bf(acc[j][0] * il0);
            o0[dn + 1] = f2bf(acc[j][1] * il0);
            o1[dn]     = f2bf(acc[j][2] * il1);
            o1[dn + 1] = f2bf(acc[j][3] * il1);
        }
    } else {
        if (tid < 16) {
            mpart[((int64_t)r * splits + sp) * 16 + tid] = s_m[tid];
            lpart[((int64_t)r * splits + sp) * 16 + tid] = s_l[tid];
        }
        float* o0 = opart + (((int64_t)r * splits + sp) * 16 + g) * 512;
        float* o1 = opart + (((int64_t)r * splits + sp) * 16 + g + 8) * 512;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int dn = d_warp + j * 8 + tq * 2;
            o0[dn]     = acc[j][0];
            o0[dn + 1] = acc[j][1];
            o1[dn]     = acc[j][2];
            o1[dn + 1] = acc[j][3];
        }
    }
}


// 4-row-tiled SM121a FP8 m16n8k32 DSA Lightning Indexer for prefill (rows >= 4):
// Loads ks[64, 128] ONCE per 4 query rows (4x less DRAM traffic on idx_k at 70K+ context!)
// and runs mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 (2x Tensor Core throughput!).
// Launch: grid=(ceil(max_older / 64), ceil(rows / 4)), block=(128)
#define TF_IDX_FP8_LD 144
__device__ __forceinline__ void idx_mma16832_fp8(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(128) tf_ernie55_dsa_indexer_scores_fp8_mma(
    const __nv_bfloat16* __restrict__ q_idx,    // [rows, 64, 128]
    const __nv_bfloat16* __restrict__ idx_k,    // [capacity, 128]
    const __nv_bfloat16* __restrict__ weights,  // [rows, 64]
    const int32_t* __restrict__ positions,      // [rows]
    float* __restrict__ scores,                 // [rows, max_older]
    int rows, int max_older, int window_size, float weights_scale) {
    const int r0 = blockIdx.y * 4;
    const int t0 = blockIdx.x * 64;
    if (r0 >= rows || t0 >= max_older) return;
    const int nr = (rows - r0 < 4) ? (rows - r0) : 4;

    int max_n_older = 0;
    for (int dr = 0; dr < nr; ++dr) {
        int no = positions[r0 + dr] - window_size + 1;
        if (no > max_n_older) max_n_older = no;
    }
    if (t0 >= max_n_older) {
        for (int dr = 0; dr < nr; ++dr) {
            for (int i = threadIdx.x; i < 64; i += blockDim.x) {
                if (t0 + i < max_older) scores[(int64_t)(r0 + dr) * max_older + t0 + i] = -1e38f;
            }
        }
        return;
    }

    __shared__ __align__(16) uint8_t ks[64 * TF_IDX_FP8_LD];
    __shared__ __align__(16) uint8_t qs[64 * TF_IDX_FP8_LD];
    __shared__ float k_sc[64];
    __shared__ float q_sc[64];
    __shared__ float red[4 * 64];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int tq = lane & 3;

    // 1. Load & quantize ks[64, 128] ONCE (4 warps x 16 rows each, 1 warp = 1 row -> 4 elems/lane!)
    for (int row = warp; row < 64; row += 4) {
        int t = t0 + row;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (t < max_n_older) {
            uint2 raw = *reinterpret_cast<const uint2*>(idx_k + (int64_t)t * 128 + lane * 4);
            const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
            v0 = __bfloat162float(b[0]); v1 = __bfloat162float(b[1]);
            v2 = __bfloat162float(b[2]); v3 = __bfloat162float(b[3]);
        }
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        float sc = fmaxf(amax, 1e-12f) * (1.0f / 448.0f);
        float inv = 1.0f / sc;
        if (lane == 0) k_sc[row] = sc;
        uint8_t q[4];
        q[0] = __nv_fp8_e4m3(v0 * inv).__x;
        q[1] = __nv_fp8_e4m3(v1 * inv).__x;
        q[2] = __nv_fp8_e4m3(v2 * inv).__x;
        q[3] = __nv_fp8_e4m3(v3 * inv).__x;
        *reinterpret_cast<uint32_t*>(ks + row * TF_IDX_FP8_LD + lane * 4) = *reinterpret_cast<uint32_t*>(q);
    }

    // 2. Loop over the 4 query rows sharing ks!
    for (int dr = 0; dr < nr; ++dr) {
        const int r = r0 + dr;
        const int n_older = positions[r] - window_size + 1;
        if (t0 >= n_older) {
            for (int i = tid; i < 64; i += blockDim.x) {
                if (t0 + i < max_older) scores[(int64_t)r * max_older + t0 + i] = -1e38f;
            }
            continue;
        }
        __syncthreads();
        const __nv_bfloat16* qr = q_idx + (int64_t)r * 64 * 128;
        for (int h = warp; h < 64; h += 4) {
            uint2 raw = *reinterpret_cast<const uint2*>(qr + (int64_t)h * 128 + lane * 4);
            const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
            float v0 = __bfloat162float(b[0]), v1 = __bfloat162float(b[1]);
            float v2 = __bfloat162float(b[2]), v3 = __bfloat162float(b[3]);
            float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            float sc = fmaxf(amax, 1e-12f) * (1.0f / 448.0f);
            float inv = 1.0f / sc;
            if (lane == 0) q_sc[h] = sc;
            uint8_t q[4];
            q[0] = __nv_fp8_e4m3(v0 * inv).__x;
            q[1] = __nv_fp8_e4m3(v1 * inv).__x;
            q[2] = __nv_fp8_e4m3(v2 * inv).__x;
            q[3] = __nv_fp8_e4m3(v3 * inv).__x;
            *reinterpret_cast<uint32_t*>(qs + h * TF_IDX_FP8_LD + lane * 4) = *reinterpret_cast<uint32_t*>(q);
        }
        __syncthreads();

        float acc[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

        #pragma unroll
        for (int kk = 0; kk < 128; kk += 32) {
            uint32_t a[4];
            const uint8_t* ap = qs + (warp * 16 + g) * TF_IDX_FP8_LD + kk + tq * 4;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * TF_IDX_FP8_LD);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 16);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * TF_IDX_FP8_LD + 16);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t b[2];
                const uint8_t* bp = ks + (j * 8 + g) * TF_IDX_FP8_LD + kk + tq * 4;
                b[0] = *reinterpret_cast<const uint32_t*>(bp);
                b[1] = *reinterpret_cast<const uint32_t*>(bp + 16);
                idx_mma16832_fp8(acc[j], a, b);
            }
        }

        const int h_lo = warp * 16 + g;
        const int h_hi = h_lo + 8;
        const float w_lo = __bfloat162float(weights[(int64_t)r * 64 + h_lo]) * weights_scale * q_sc[h_lo];
        const float w_hi = __bfloat162float(weights[(int64_t)r * 64 + h_hi]) * weights_scale * q_sc[h_hi];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = j * 8 + tq * 2 + e;
                const float ks_c = k_sc[col];
                float v = (w_lo * fmaxf(acc[j][e], 0.0f) + w_hi * fmaxf(acc[j][e + 2], 0.0f)) * ks_c;
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                v += __shfl_xor_sync(0xffffffffu, v, 16);
                if (g == 0) red[warp * 64 + col] = v;
            }
        }
        __syncthreads();
        for (int i = tid; i < 64; i += blockDim.x) {
            const int t = t0 + i;
            if (t >= max_older) continue;
            const float total = red[i] + red[64 + i] + red[128 + i] + red[192 + i];
            scores[(int64_t)r * max_older + t] = (t < n_older) ? total : -1e38f;
        }
    }
}
