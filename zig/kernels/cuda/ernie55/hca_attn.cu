// ERNIE 5.5 Flash Preview HCA (Heavily Compressed Attention, 37 layers + 3 DSpark layers):
//   - SWA-128 ring cache [swa_cap, 512] BF16 (swa_cap is power of 2, e.g. 1024)
//   - C128 compressor state ring [state_cap, 2, 512] F32 + compressed KV cache [comp_cap, 512] BF16
//   - Shared-memory multi-head MQA attention (loads each KV row ONCE across all H_loc=16 heads!)
//   - Virtual sink logit attn_sink[h] folded cleanly into online softmax init (m = sink, l = 1, acc = 0)

#include <cuda_bf16.h>
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

// Apply GPT-J RoPE on dims [448..512) of normed kv [rows, 512] and store into swa_kv[pos & swa_mask]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_kv_rope_store(
    const __nv_bfloat16* __restrict__ kv,   // [rows, 512] (already kv_layernormed)
    const int32_t* __restrict__ positions,  // [rows]
    const float* __restrict__ inv_freq,     // [32]
    __nv_bfloat16* __restrict__ swa_kv,     // [swa_cap, 512]
    int rows,
    int swa_mask
) {
    int r = blockIdx.x;
    int t = threadIdx.x;  // 0..255 -> pair [2*t, 2*t+1]
    if (r >= rows) return;
    int pos = positions[r];
    const __nv_bfloat16* src = kv + (int64_t)r * 512;
    __nv_bfloat16* dst = swa_kv + (int64_t)(pos & swa_mask) * 512;
    float e = bf2f(src[2 * t]);
    float o = bf2f(src[2 * t + 1]);
    if (t >= 224) {
        int pair = t - 224;
        float theta = (float)pos * inv_freq[pair];
        float c, s;
        __sincosf(theta, &s, &c);
        float ne = e * c - o * s;
        float no = o * c + e * s;
        e = ne;
        o = no;
    }
    dst[2 * t]     = f2bf(e);
    dst[2 * t + 1] = f2bf(o);
}

// Save compressor partial states:
//   comp_state[pos & state_mask, 0, d] = float(wkv[r, d])
//   comp_state[pos & state_mask, 1, d] = float(wgate[r, d]) + ape[pos % 128, d]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_comp_save(
    const __nv_bfloat16* __restrict__ wkv,    // [rows, 512]
    const __nv_bfloat16* __restrict__ wgate,  // [rows, 512]
    const float* __restrict__ ape,            // [128, 512]
    const int32_t* __restrict__ positions,    // [rows]
    float* __restrict__ comp_state,           // [state_cap, 2, 512]
    int rows,
    int state_mask
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int pos = positions[r];
    int slot = pos & state_mask;
    int ape_row = pos & 127;
    const __nv_bfloat16* kvr = wkv + (int64_t)r * 512;
    const __nv_bfloat16* gr = wgate + (int64_t)r * 512;
    const float* ar = ape + (int64_t)ape_row * 512;
    float* dst_kv = comp_state + (int64_t)slot * 1024;
    float* dst_sc = dst_kv + 512;
    for (int d = threadIdx.x; d < 512; d += blockDim.x) {
        dst_kv[d] = bf2f(kvr[d]);
        dst_sc[d] = bf2f(gr[d]) + ar[d];
    }
}

// Flush completed C128 compressed KV blocks for any row where (pos + 1) % 128 == 0:
//   1. Per-dim softmax over t=0..127 of comp_state[(blk*128 + t) & state_mask, 1, d]
//   2. Weighted sum of comp_state[..., 0, d] -> ckv[d]
//   3. RMSNorm(ckv, norm_w, eps) -> round to BF16 -> F32
//   4. GPT-J RoPE on dims [448..512) at position blk * 128 (start of block!)
//   5. Store BF16 to comp_kv[blk, 0..511]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_comp_flush(
    const float* __restrict__ comp_state,     // [state_cap, 2, 512]
    const __nv_bfloat16* __restrict__ norm_w, // [512]
    const int32_t* __restrict__ positions,    // [rows]
    const float* __restrict__ inv_freq,       // [32]
    __nv_bfloat16* __restrict__ comp_kv,      // [comp_cap, 512]
    int rows,
    int state_mask,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int pos = positions[r];
    if (((pos + 1) & 127) != 0) return;

    int blk = pos >> 7;
    int s0 = blk << 7;
    __shared__ float s_ckv[512];
    __shared__ float s_red[32];

    // Each of 256 threads handles 2 dims: d0 = threadIdx.x, d1 = threadIdx.x + 256
    for (int pass = 0; pass < 2; ++pass) {
        int d = threadIdx.x + pass * 256;
        float mx = -1e38f;
        for (int t = 0; t < 128; ++t) {
            int slot = (s0 + t) & state_mask;
            float sc = comp_state[(int64_t)slot * 1024 + 512 + d];
            if (sc > mx) mx = sc;
        }
        float denom = 0.0f;
        float wsum = 0.0f;
        for (int t = 0; t < 128; ++t) {
            int slot = (s0 + t) & state_mask;
            float sc = comp_state[(int64_t)slot * 1024 + 512 + d];
            float kv = comp_state[(int64_t)slot * 1024 + d];
            float p = expf(sc - mx);
            denom += p;
            wsum += p * kv;
        }
        s_ckv[d] = wsum / denom;
    }
    __syncthreads();

    // RMSNorm over 512 dims
    float v0 = s_ckv[2 * threadIdx.x];
    float v1 = s_ckv[2 * threadIdx.x + 1];
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

    // Round to BF16 -> F32 before RoPE (exact match with fused_compress_quant_cache.py)
    float e = bf2f(f2bf(v0 * rrms * bf2f(norm_w[2 * threadIdx.x])));
    float o = bf2f(f2bf(v1 * rrms * bf2f(norm_w[2 * threadIdx.x + 1])));

    if (threadIdx.x >= 224) {
        int pair = threadIdx.x - 224;
        float theta = (float)s0 * inv_freq[pair];
        float c, s;
        __sincosf(theta, &s, &c);
        float ne = e * c - o * s;
        float no = o * c + e * s;
        e = ne;
        o = no;
    }

    __nv_bfloat16* dst = comp_kv + (int64_t)blk * 512;
    dst[2 * threadIdx.x]     = f2bf(e);
    dst[2 * threadIdx.x + 1] = f2bf(o);
}

// High-throughput multi-head MQA HCA attention (16 heads per CTA -> each KV row read ONCE from DRAM!)
// Supports both SWA-128 + C128 compressed KV (or SWA-only when comp_kv == nullptr for DSpark).
// Each CTA has 256 threads = 8 warps; each warp handles 2 Q heads (16 heads/CTA total!).
// Each lane owns 16 dims of head_dim=512 (d = lane*16 .. lane*16+15) in registers!
// Launch: grid=(ceil(H_loc / 16), rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_attn(
    const __nv_bfloat16* __restrict__ q,       // [rows, H_loc, 512]
    const __nv_bfloat16* __restrict__ swa_kv,  // [swa_cap, 512]
    const __nv_bfloat16* __restrict__ comp_kv, // nullable [comp_cap, 512]
    const float* __restrict__ attn_sink,       // [H_loc]
    const int32_t* __restrict__ positions,     // [rows]
    __nv_bfloat16* __restrict__ out,           // [rows, H_loc, 512]
    int rows,
    int H_loc,
    int swa_mask,
    int window_size,
    float scale
) {
    int r = blockIdx.y;
    if (r >= rows) return;
    int h_base = blockIdx.x * 16;
    int warp = threadIdx.x >> 5;   // 0..7
    int lane = threadIdx.x & 31;   // 0..31 -> 16 dims [lane*16 .. lane*16+15]
    int h0 = h_base + warp * 2;
    int h1 = h0 + 1;
    bool v0 = (h0 < H_loc);
    bool v1 = (h1 < H_loc);

    int pos = positions[r];
    int swa_start = (pos >= window_size - 1) ? (pos - (window_size - 1)) : 0;
    int n_swa = pos - swa_start + 1;
    int n_comp = (comp_kv != nullptr) ? ((pos + 1) >> 7) : 0;
    int total_k = n_swa + n_comp;

    // Load Q[h0] and Q[h1] into registers (16 floats each per lane)
    float q0[16], q1[16];
    float acc0[16], acc1[16];
    const __nv_bfloat16* qp0 = q + ((int64_t)r * H_loc + h0) * 512 + lane * 16;
    const __nv_bfloat16* qp1 = q + ((int64_t)r * H_loc + h1) * 512 + lane * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q0[i] = v0 ? bf2f(qp0[i]) * scale : 0.0f;
        q1[i] = v1 ? bf2f(qp1[i]) * scale : 0.0f;
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }
    float m0 = v0 ? attn_sink[h0] : -1e38f;
    float m1 = v1 ? attn_sink[h1] : -1e38f;
    float l0 = 1.0f;
    float l1 = 1.0f;

    // Shared memory tile: 16 KV rows x 512 BF16 = 16 KB
    constexpr int BK = 16;
    __shared__ __nv_bfloat16 s_kv[BK][512];

    for (int k0 = 0; k0 < total_k; k0 += BK) {
        int cnt = (total_k - k0 < BK) ? (total_k - k0) : BK;
        // Cooperatively load cnt rows of 512 BF16 (256 threads x 8 elems = 2048 elems = 4 rows per step)
        for (int t = 0; t < cnt; ++t) {
            int gi = k0 + t;
            const __nv_bfloat16* src_row = (gi < n_swa)
                ? (swa_kv + (int64_t)((swa_start + gi) & swa_mask) * 512)
                : (comp_kv + (int64_t)(gi - n_swa) * 512);
            // 256 threads -> 2 BF16 (uint32_t) per thread covers all 512 BF16 in 1 coalesced load!
            reinterpret_cast<uint32_t*>(s_kv[t])[threadIdx.x] =
                reinterpret_cast<const uint32_t*>(src_row)[threadIdx.x];
        }
        __syncthreads();

        for (int t = 0; t < cnt; ++t) {
            const __nv_bfloat16* kp = &s_kv[t][lane * 16];
            float kv[16];
            float d0 = 0.0f, d1 = 0.0f;
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
        __nv_bfloat16* op = out + ((int64_t)r * H_loc + h0) * 512 + lane * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) op[i] = f2bf(acc0[i] * inv_l);
    }
    if (v1) {
        float inv_l = 1.0f / l1;
        __nv_bfloat16* op = out + ((int64_t)r * H_loc + h1) * 512 + lane * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) op[i] = f2bf(acc1[i] * inv_l);
    }
}

// DSpark ring-buffer KV norm + trailing-64 GPT-J RoPE store into kv_ring[pos & ring_mask].
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_kv_store_ring(
    const __nv_bfloat16* __restrict__ kv_in,     // [rows, 512]
    const __nv_bfloat16* __restrict__ kv_norm,   // [512]
    const int32_t* __restrict__ positions,       // [rows]
    const float* __restrict__ inv_freq,          // [32]
    __nv_bfloat16* __restrict__ kv_ring,         // [ring_size, 512]
    int rows,
    int ring_mask,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    const __nv_bfloat16* src = kv_in + (int64_t)r * 512;
    int pos = positions[r];
    __nv_bfloat16* dst = kv_ring + (int64_t)(pos & ring_mask) * 512;

    float v0 = __bfloat162float(src[tid]);
    float v1 = __bfloat162float(src[tid + 256]);
    float ss = v0 * v0 + v1 * v1;

    __shared__ float smem[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = (lane < 8) ? smem[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
        if (lane == 0) smem[0] = rsqrtf(ss / 512.0f + eps);
    }
    __syncthreads();
    float inv = smem[0];

    float n0 = v0 * inv * __bfloat162float(kv_norm[tid]);
    float n1 = v1 * inv * __bfloat162float(kv_norm[tid + 256]);
    dst[tid] = __float2bfloat16(n0);
    // Keep normed value in shared memory for trailing-64 GPT-J RoPE [448..512)
    __shared__ float rope_buf[64];
    if (tid + 256 < 448) {
        dst[tid + 256] = __float2bfloat16(n1);
    } else {
        rope_buf[tid + 256 - 448] = __bfloat162float(__float2bfloat16(n1));
    }
    __syncthreads();
    if (tid < 32) {
        float ang = (float)pos * inv_freq[tid];
        float c, s;
        __sincosf(ang, &s, &c);
        float a = rope_buf[2 * tid];
        float b = rope_buf[2 * tid + 1];
        dst[448 + 2 * tid]     = __float2bfloat16(a * c - b * s);
        dst[448 + 2 * tid + 1] = __float2bfloat16(a * s + b * c);
    }
}

// DSpark non-causal HCA attention over ring buffer kv_ring[k & ring_mask] for k in [lo .. hi] + attn_sink.
// Launch: grid=(ceil(H/4), B), block=(128) (4 warps = 4 heads per block; shared memory holds KV row)
extern "C" __global__ void tf_ernie55_hca_attn_dspark(
    const __nv_bfloat16* __restrict__ q,         // [B, H, 512]
    const __nv_bfloat16* __restrict__ kv_ring,   // [ring_size, 512]
    const float* __restrict__ attn_sink,         // [H]
    __nv_bfloat16* __restrict__ out,             // [B, H, 512]
    int B,
    int H,
    int lo,
    int hi,
    int ring_mask,
    float scale
) {
    int r = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int h = blockIdx.x * 4 + warp;
    if (r >= B) return;

    float qr[16];
    float acc[16];
    if (h < H) {
        const __nv_bfloat16* qp = q + ((int64_t)r * H + h) * 512;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            qr[i] = __bfloat162float(qp[lane * 16 + i]);
            acc[i] = 0.0f;
        }
    }

    float m = (h < H) ? attn_sink[h] : -INFINITY;
    float l = 1.0f;

    __shared__ __nv_bfloat16 s_kv[512];
    int tid = threadIdx.x;

    for (int k = lo; k <= hi; ++k) {
        const __nv_bfloat16* kp = kv_ring + (int64_t)(k & ring_mask) * 512;
        // Cooperative load of 512 BF16 (1024 bytes = 128 threads x 8 bytes)
        reinterpret_cast<uint2*>(s_kv)[tid] = reinterpret_cast<const uint2*>(kp)[tid];
        __syncthreads();

        if (h < H) {
            float dot = 0.0f;
            float kv_reg[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float v = __bfloat162float(s_kv[lane * 16 + i]);
                kv_reg[i] = v;
                dot += qr[i] * v;
            }
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
            float score = __shfl_sync(0xffffffff, dot, 0) * scale;
            float m_new = fmaxf(m, score);
            float alpha = expf(m - m_new);
            float beta = expf(score - m_new);
            l = l * alpha + beta;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                acc[i] = acc[i] * alpha + beta * kv_reg[i];
            }
            m = m_new;
        }
        __syncthreads();
    }

    if (h < H) {
        float inv_l = 1.0f / l;
        __nv_bfloat16* op = out + ((int64_t)r * H + h) * 512;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            op[lane * 16 + i] = __float2bfloat16(acc[i] * inv_l);
        }
    }
}


// --- Batch 1 fused HCA prep & SM121a Tensor-Core MMA FlashAttention + Split-K ---

__device__ __forceinline__ void hca_mma_m16n8k16_bf16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fused HCA q_down + kv_proj post-op:
//   packed [rows, 1536] ->
//     q_down [rows, 1024] = RMSNorm(packed[0..1024), q_norm)
//     swa_kv[pos & swa_mask, 0..512] = GPT-J RoPE[448..512)(bf16(RMSNorm(packed[1024..1536), kv_norm)))
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_qkv_prep(
    const __nv_bfloat16* __restrict__ packed,   // [rows, 1536]
    const __nv_bfloat16* __restrict__ q_norm,   // [1024]
    const __nv_bfloat16* __restrict__ kv_norm,  // [512]
    const int32_t* __restrict__ positions,      // [rows]
    const float* __restrict__ inv_freq,         // [32]
    __nv_bfloat16* __restrict__ q_down,         // [rows, 1024]
    __nv_bfloat16* __restrict__ swa_kv,         // [swa_cap, 512]
    int rows,
    int swa_mask,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    __shared__ float s_red[8];

    const __nv_bfloat16* src = packed + (int64_t)r * 1536;

    // 1. q_down [0..1024): 4 elements per thread
    float q0 = bf2f(src[tid]);
    float q1 = bf2f(src[tid + 256]);
    float q2 = bf2f(src[tid + 512]);
    float q3 = bf2f(src[tid + 768]);
    float qsq = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    qsq = warp_sum(qsq);
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
    qdst[tid]       = f2bf(q0 * q_rrms * bf2f(q_norm[tid]));
    qdst[tid + 256] = f2bf(q1 * q_rrms * bf2f(q_norm[tid + 256]));
    qdst[tid + 512] = f2bf(q2 * q_rrms * bf2f(q_norm[tid + 512]));
    qdst[tid + 768] = f2bf(q3 * q_rrms * bf2f(q_norm[tid + 768]));

    // 2. kv [1024..1536): 2 elements per thread (pair [2*tid, 2*tid+1])
    const __nv_bfloat16* ksrc = src + 1024;
    float k0 = bf2f(ksrc[2 * tid]);
    float k1 = bf2f(ksrc[2 * tid + 1]);
    float ksq = k0 * k0 + k1 * k1;
    ksq = warp_sum(ksq);
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
    // Round to BF16 before RoPE to match exact 2-kernel precision
    float e = bf2f(f2bf(k0 * k_rrms * bf2f(kv_norm[2 * tid])));
    float o = bf2f(f2bf(k1 * k_rrms * bf2f(kv_norm[2 * tid + 1])));
    int pos = positions[r];
    if (tid >= 224) {
        int pair = tid - 224;
        float theta = (float)pos * inv_freq[pair];
        float c, s;
        __sincosf(theta, &s, &c);
        float ne = e * c - o * s;
        float no = o * c + e * s;
        e = ne;
        o = no;
    }
    __nv_bfloat16* kdst = swa_kv + (int64_t)(pos & swa_mask) * 512;
    kdst[2 * tid]     = f2bf(e);
    kdst[2 * tid + 1] = f2bf(o);
}

// Save compressor partial states from fused [rows, 1024] = [wkv (512), wgate (512)]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_comp_save_packed(
    const __nv_bfloat16* __restrict__ packed, // [rows, 1024]
    const float* __restrict__ ape,            // [128, 512]
    const int32_t* __restrict__ positions,    // [rows]
    float* __restrict__ comp_state,           // [state_cap, 2, 512]
    int rows,
    int state_mask
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int pos = positions[r];
    int slot = pos & state_mask;
    int ape_row = pos & 127;
    const __nv_bfloat16* kvr = packed + (int64_t)r * 1024;
    const __nv_bfloat16* gr = kvr + 512;
    const float* ar = ape + (int64_t)ape_row * 512;
    float* dst_kv = comp_state + (int64_t)slot * 1024;
    float* dst_sc = dst_kv + 512;
    for (int d = threadIdx.x; d < 512; d += blockDim.x) {
        dst_kv[d] = bf2f(kvr[d]);
        dst_sc[d] = bf2f(gr[d]) + ar[d];
    }
}

// SM121a Tensor-Core FlashAttention + Split-K for HCA (H_loc=16, D_QK=512, D_V=512, LD=520)
// Launch: grid=(splits, rows), block=(128), smem=53184
struct TFHcaSegEntry {
    uint64_t stash_ptr;
    int32_t row0;
    int32_t n;
    int32_t max_pos;
    int32_t older_ceil;
    int32_t sparse;
    int32_t pad;
};

__device__ __forceinline__ int tf_hca_attn_splits(int n, int max_sp) {
    return (n > 16) ? 1 : (n <= 4) ? max_sp : (n <= 8) ? 6 : 2;
}

__device__ __forceinline__ void hca_attn_mma_impl(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ swa_kv,
    const __nv_bfloat16* __restrict__ comp_kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ opart,
    float* __restrict__ mpart,
    float* __restrict__ lpart,
    int r,
    int sp,
    int swa_mask,
    int window_size,
    int splits,
    int split_stride,
    float scale
) {

    int pos = positions[r];
    int swa_start = (pos >= window_size - 1) ? (pos - (window_size - 1)) : 0;
    int n_swa = pos - swa_start + 1;
    int n_comp = (comp_kv != nullptr) ? ((pos + 1) >> 7) : 0;
    int total_k = n_swa + n_comp;

    int chunk = (total_k + splits - 1) / splits;
    int k_start = sp * chunk;
    int k_end = (k_start + chunk < total_k) ? (k_start + chunk) : total_k;

    constexpr int LD = 520;
    extern __shared__ unsigned char raw_hca_mma[];
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(raw_hca_mma);
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

    const __nv_bfloat16* qr = q + (int64_t)r * 16 * 512;
    for (int idx = tid; idx < 16 * 64; idx += 128) {
        int h = idx >> 6;
        int c = (idx & 63) * 8;
        *reinterpret_cast<uint4*>(s_q + h * LD + c) = *reinterpret_cast<const uint4*>(qr + h * 512 + c);
    }
    if (tid < 16) {
        if (sp == 0) {
            s_m[tid] = attn_sink[tid];
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
        for (int idx = tid; idx < 32 * 64; idx += 128) {
            int t = idx >> 6;
            int c = (idx & 63) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < cnt) {
                int gi = k0 + t;
                const __nv_bfloat16* src = (gi < n_swa)
                    ? (swa_kv + (int64_t)((swa_start + gi) & swa_mask) * 512)
                    : (comp_kv + (int64_t)(gi - n_swa) * 512);
                v = *reinterpret_cast<const uint4*>(src + c);
            }
            *reinterpret_cast<uint4*>(s_k + t * LD + c) = v;
        }
        __syncthreads();

        float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int n_base = warp * 8;
        for (int d = 0; d < 512; d += 16) {
            uint32_t a[4], b[2];
            const __nv_bfloat16* ap = s_q + g * LD + d + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD + 8);
            const __nv_bfloat16* bp = s_k + (n_base + g) * LD + d + tq * 2;
            b[0] = *reinterpret_cast<const uint32_t*>(bp);
            b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
            hca_mma_m16n8k16_bf16(sc, a, b);
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
                hca_mma_m16n8k16_bf16(acc[j], a, b);
            }
        }
        __syncthreads();
    }

    int d_warp = warp * 128;
    if (splits == 1) {
        float il0 = 1.0f / s_l[g];
        float il1 = 1.0f / s_l[g + 8];
        __nv_bfloat16* o0 = out + ((int64_t)r * 16 + g) * 512;
        __nv_bfloat16* o1 = out + ((int64_t)r * 16 + g + 8) * 512;
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
            mpart[((int64_t)r * split_stride + sp) * 16 + tid] = s_m[tid];
            lpart[((int64_t)r * split_stride + sp) * 16 + tid] = s_l[tid];
        }
        float* o0 = opart + (((int64_t)r * split_stride + sp) * 16 + g) * 512;
        float* o1 = opart + (((int64_t)r * split_stride + sp) * 16 + g + 8) * 512;
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


extern "C" __global__ void __launch_bounds__(128) tf_ernie55_hca_attn_mma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ swa_kv,
    const __nv_bfloat16* __restrict__ comp_kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ opart,
    float* __restrict__ mpart,
    float* __restrict__ lpart,
    int rows,
    int swa_mask,
    int window_size,
    int splits,
    float scale
) {
    int r = blockIdx.y;
    int sp = blockIdx.x;
    if (r >= rows) return;
    hca_attn_mma_impl(q, swa_kv, comp_kv, attn_sink, positions, out, opart, mpart, lpart, r, sp, swa_mask, window_size, splits, splits, scale);
}

extern "C" __global__ void __launch_bounds__(128) tf_ernie55_hca_attn_mma_tab(
    const __nv_bfloat16* __restrict__ q,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ opart,
    float* __restrict__ mpart,
    float* __restrict__ lpart,
    const int32_t* __restrict__ row_seg,
    const TFHcaSegEntry* __restrict__ seg_tab,
    const uint64_t* __restrict__ kv_ptrs,
    int layer_off,
    int has_comp,
    int rows,
    int swa_mask,
    int window_size,
    int max_sp,
    float scale
) {
    int r = blockIdx.y;
    int sp = blockIdx.x;
    if (r >= rows) return;
    int seg = row_seg[r];
    if (seg < 0) {
        if (sp == 0) {
            __nv_bfloat16* orow = out + (int64_t)r * 16 * 512;
            for (int i = threadIdx.x; i < 16 * 512; i += blockDim.x) orow[i] = f2bf(0.0f);
        }
        return;
    }
    int splits = tf_hca_attn_splits(seg_tab[seg].n, max_sp);
    if (sp >= splits) return;
    const uint64_t* p = kv_ptrs + seg * layer_off;
    const __nv_bfloat16* swa_kv = reinterpret_cast<const __nv_bfloat16*>(p[0]);
    const __nv_bfloat16* comp_kv = has_comp ? reinterpret_cast<const __nv_bfloat16*>(p[2]) : nullptr;
    hca_attn_mma_impl(q, swa_kv, comp_kv, attn_sink, positions, out, opart, mpart, lpart, r, sp, swa_mask, window_size, splits, max_sp, scale);
}
// SM121a Tensor-Core non-causal ring-buffer HCA attention for DSpark (H_loc=16, D_QK=512, LD=520)
// Computes lo = max(0, positions[0] + 1 - window_size), hi = positions[0] + B - 1 on device (CUDA-graph safe!).
// Launch: grid=(splits, B), block=(128), smem=53184
extern "C" __global__ void __launch_bounds__(128) tf_ernie55_hca_attn_dspark_mma(
    const __nv_bfloat16* __restrict__ q,         // [B, 16, 512]
    const __nv_bfloat16* __restrict__ kv_ring,   // [ring_size, 512]
    const float* __restrict__ attn_sink,         // [16]
    const int32_t* __restrict__ positions,       // [B] (positions[0] = anchor pos p)
    __nv_bfloat16* __restrict__ out,             // [B, 16, 512]
    float* __restrict__ opart,
    float* __restrict__ mpart,
    float* __restrict__ lpart,
    int B,
    int window_size,
    int ring_mask,
    int splits,
    float scale
) {
    int r = blockIdx.y;
    int sp = blockIdx.x;
    if (r >= B) return;

    int p0 = positions[0];
    int lo = (p0 + 1 > window_size) ? (p0 + 1 - window_size) : 0;
    int hi = p0 + B - 1;
    int total_k = hi - lo + 1;

    int chunk = (total_k + splits - 1) / splits;
    int k_start = sp * chunk;
    int k_end = (k_start + chunk < total_k) ? (k_start + chunk) : total_k;

    constexpr int LD = 520;
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

    const __nv_bfloat16* qr = q + (int64_t)r * 16 * 512;
    for (int idx = tid; idx < 16 * 64; idx += 128) {
        int h = idx >> 6;
        int c = (idx & 63) * 8;
        *reinterpret_cast<uint4*>(s_q + h * LD + c) = *reinterpret_cast<const uint4*>(qr + h * 512 + c);
    }
    if (tid < 16) {
        if (sp == 0) {
            s_m[tid] = attn_sink[tid];
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
        for (int idx = tid; idx < 32 * 64; idx += 128) {
            int t = idx >> 6;
            int c = (idx & 63) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < cnt) {
                const __nv_bfloat16* src = kv_ring + (int64_t)((lo + k0 + t) & ring_mask) * 512;
                v = *reinterpret_cast<const uint4*>(src + c);
            }
            *reinterpret_cast<uint4*>(s_k + t * LD + c) = v;
        }
        __syncthreads();

        float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int n_base = warp * 8;
        for (int d = 0; d < 512; d += 16) {
            uint32_t a[4], b[2];
            const __nv_bfloat16* ap = s_q + g * LD + d + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD + 8);
            const __nv_bfloat16* bp = s_k + (n_base + g) * LD + d + tq * 2;
            b[0] = *reinterpret_cast<const uint32_t*>(bp);
            b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
            hca_mma_m16n8k16_bf16(sc, a, b);
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
                hca_mma_m16n8k16_bf16(acc[j], a, b);
            }
        }
        __syncthreads();
    }

    int d_warp = warp * 128;
    if (splits == 1) {
        float il0 = 1.0f / s_l[g];
        float il1 = 1.0f / s_l[g + 8];
        __nv_bfloat16* o0 = out + ((int64_t)r * 16 + g) * 512;
        __nv_bfloat16* o1 = out + ((int64_t)r * 16 + g + 8) * 512;
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

// Split-K combine for both HCA and DSA MMA attention (H_loc=16, D_V=512)
// Launch: grid=(16, rows), block=(256)
extern "C" __global__ void tf_ernie55_attn_splitk_combine(
    const float* __restrict__ opart,
    const float* __restrict__ mpart,
    const float* __restrict__ lpart,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int splits
) {
    int r = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;  // 0..255 -> dims d and d + 256
    if (r >= rows || h >= 16) return;
    const float* mr = mpart + (int64_t)r * splits * 16;
    const float* lr = lpart + (int64_t)r * splits * 16;
    float mmax = -1e38f;
    for (int s = 0; s < splits; ++s) mmax = fmaxf(mmax, mr[s * 16 + h]);
    float lsum = 0.0f, v0 = 0.0f, v1 = 0.0f;
    for (int s = 0; s < splits; ++s) {
        float ls = lr[s * 16 + h];
        if (ls > 0.0f) {
            float w = expf(mr[s * 16 + h] - mmax);
            lsum += ls * w;
            const float* op = opart + ((int64_t)r * splits + s) * 16 * 512 + h * 512;
            v0 += op[d] * w;
            v1 += op[d + 256] * w;
        }
    }
    float inv = 1.0f / lsum;
    __nv_bfloat16* dst = out + ((int64_t)r * 16 + h) * 512;
    dst[d]       = f2bf(v0 * inv);
    dst[d + 256] = f2bf(v1 * inv);
}


// 4-row Prefill HCA FlashAttention (NR=4, TK=16, 512 threads, 90112 B smem, 0.00000 bit-exact, 2.49x faster at 66K!)
extern "C" __global__ void __launch_bounds__(512) tf_ernie55_hca_attn_prefill_mma(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ swa_kv,
    const __nv_bfloat16* __restrict__ comp_kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ out,
    int rows, int swa_mask, int swa_win, float scale
) {
    int r0 = blockIdx.x * 4;
    if (r0 >= rows) return;
    int nr = min(4, rows - r0);
    constexpr int LD = 520;
    constexpr int TK = 16;
    extern __shared__ unsigned char raw[];
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(raw);
    __nv_bfloat16* s_k = s_q + 4 * 16 * LD;
    float* s_sc        = reinterpret_cast<float*>(s_k + TK * LD);
    __nv_bfloat16* s_p = reinterpret_cast<__nv_bfloat16*>(s_sc + 4 * 16 * TK);
    float* s_m         = reinterpret_cast<float*>(s_p + 4 * 16 * TK);
    float* s_l         = s_m + 64;
    float* s_alpha     = s_l + 64;

    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int qi = warp >> 2, wn = warp & 3, g = lane >> 2, tq = lane & 3;

    int my_pos = (qi < nr) ? positions[r0 + qi] : -1;
    int pos_first = positions[r0], pos_last = positions[r0 + nr - 1];

    for (int idx = tid; idx < nr * 16 * 64; idx += 512) {
        int rq = idx >> 10, rem = idx & 1023, h = rem >> 6, c = (rem & 63) << 3;
        *reinterpret_cast<uint4*>(s_q + (rq * 16 + h) * LD + c) =
            *reinterpret_cast<const uint4*>(q + ((int64_t)(r0 + rq) * 16 + h) * 512 + c);
    }
    if (tid < 64) { s_m[tid] = attn_sink[tid & 15]; s_l[tid] = 1.0f; }
    __syncthreads();

    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

    int swa_lo = (pos_first + 1 > swa_win) ? (pos_first + 1 - swa_win) : 0;
    int n_swa_union = pos_last - swa_lo + 1;
    int n_comp_max = (comp_kv != nullptr) ? ((pos_last + 1) >> 7) : 0;
    int total_k = n_swa_union + n_comp_max;
    int my_swa_lo = (my_pos + 1 > swa_win) ? (my_pos + 1 - swa_win) : 0;
    int my_comp_end = (comp_kv != nullptr && my_pos >= 0) ? ((my_pos + 1) >> 7) : 0;

    for (int k0 = 0; k0 < total_k; k0 += TK) {
        int cnt = min(TK, total_k - k0);
        for (int idx = tid; idx < TK * 64; idx += 512) {
            int t = idx >> 6, c = (idx & 63) << 3;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < cnt) {
                int gi = k0 + t;
                const __nv_bfloat16* src = (gi < n_swa_union)
                    ? (swa_kv + (int64_t)((swa_lo + gi) & swa_mask) * 512)
                    : (comp_kv + (int64_t)(gi - n_swa_union) * 512);
                v = *reinterpret_cast<const uint4*>(src + c);
            }
            *reinterpret_cast<uint4*>(s_k + t * LD + c) = v;
        }
        __syncthreads();

        if (qi < nr) {
            const __nv_bfloat16* sq_row = s_q + qi * 16 * LD;
            float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int n_base = (wn & 1) * 8;
            int d0 = (wn >> 1) * 256;
            for (int d = d0; d < d0 + 256; d += 16) {
                uint32_t a[4], b[2];
                const __nv_bfloat16* ap = sq_row + g * LD + d + tq * 2;
                a[0] = *reinterpret_cast<const uint32_t*>(ap);
                a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD);
                a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
                a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD + 8);
                const __nv_bfloat16* bp = s_k + (n_base + g) * LD + d + tq * 2;
                b[0] = *reinterpret_cast<const uint32_t*>(bp);
                b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
                hca_mma_m16n8k16_bf16(sc, a, b);
            }
            float* sc_row = s_sc + qi * 16 * TK;
            int c0 = n_base + tq * 2;
            if (wn < 2) {
                sc_row[g * TK + c0] = sc[0]; sc_row[g * TK + c0 + 1] = sc[1];
                sc_row[(g + 8) * TK + c0] = sc[2]; sc_row[(g + 8) * TK + c0 + 1] = sc[3];
            }
            __syncthreads();
            if (wn >= 2) {
                auto vk = [&](int ti) -> bool {
                    if (ti >= cnt) return false;
                    int gi = k0 + ti;
                    if (gi < n_swa_union) { int kp = swa_lo + gi; return (kp >= my_swa_lo && kp <= my_pos); }
                    return ((gi - n_swa_union) < my_comp_end);
                };
                sc_row[g * TK + c0]         = vk(c0)     ? (sc_row[g * TK + c0] + sc[0]) * scale : -1e38f;
                sc_row[g * TK + c0 + 1]     = vk(c0 + 1) ? (sc_row[g * TK + c0 + 1] + sc[1]) * scale : -1e38f;
                sc_row[(g + 8) * TK + c0]   = vk(c0)     ? (sc_row[(g + 8) * TK + c0] + sc[2]) * scale : -1e38f;
                sc_row[(g + 8) * TK + c0+1] = vk(c0 + 1) ? (sc_row[(g + 8) * TK + c0 + 1] + sc[3]) * scale : -1e38f;
            }
        } else {
            __syncthreads();
        }
        __syncthreads();

        if (tid < nr * 16) {
            int rq = tid >> 4, h = tid & 15;
            float* sc_h = s_sc + (rq * 16 + h) * TK;
            __nv_bfloat16* sp_h = s_p + (rq * 16 + h) * TK;
            float m_old = s_m[tid], m_new = m_old;
            for (int t = 0; t < cnt; ++t) m_new = fmaxf(m_new, sc_h[t]);
            float alpha = expf(m_old - m_new), l_add = 0.0f;
            for (int t = 0; t < TK; ++t) {
                float p = (t < cnt && sc_h[t] > -1e30f) ? expf(sc_h[t] - m_new) : 0.0f;
                sp_h[t] = f2bf(p); l_add += p;
            }
            s_m[tid] = m_new; s_l[tid] = s_l[tid] * alpha + l_add; s_alpha[tid] = alpha;
        }
        __syncthreads();

        if (qi < nr) {
            float a0 = s_alpha[qi * 16 + g], a1 = s_alpha[qi * 16 + g + 8];
#pragma unroll
            for (int j = 0; j < 16; ++j) { acc[j][0]*=a0; acc[j][1]*=a0; acc[j][2]*=a1; acc[j][3]*=a1; }
            const __nv_bfloat16* sp_row = s_p + qi * 16 * TK;
            int d_base = wn * 128;
            uint32_t a[4];
            const __nv_bfloat16* ap = sp_row + g * TK + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * TK);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * TK + 8);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                int db = d_base + j * 8 + g;
                uint32_t b[2];
                b[0] = ((uint32_t)__bfloat16_as_ushort(s_k[(tq * 2) * LD + db])) |
                       (((uint32_t)__bfloat16_as_ushort(s_k[(tq * 2 + 1) * LD + db])) << 16);
                b[1] = ((uint32_t)__bfloat16_as_ushort(s_k[(tq * 2 + 8) * LD + db])) |
                       (((uint32_t)__bfloat16_as_ushort(s_k[(tq * 2 + 9) * LD + db])) << 16);
                hca_mma_m16n8k16_bf16(acc[j], a, b);
            }
        }
        __syncthreads();
    }
    if (qi < nr) {
        float inv0 = 1.0f / s_l[qi * 16 + g], inv1 = 1.0f / s_l[qi * 16 + g + 8];
        int d_base = wn * 128;
        __nv_bfloat16* orow = out + (int64_t)(r0 + qi) * 16 * 512;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int c0 = d_base + j * 8 + tq * 2;
            orow[g * 512 + c0]         = f2bf(acc[j][0] * inv0);
            orow[g * 512 + c0 + 1]     = f2bf(acc[j][1] * inv0);
            orow[(g + 8) * 512 + c0]   = f2bf(acc[j][2] * inv1);
            orow[(g + 8) * 512 + c0+1] = f2bf(acc[j][3] * inv1);
        }
    }
}

extern "C" __global__ void tf_ernie55_hca_qkv_prep_tab(
    const __nv_bfloat16* __restrict__ packed,
    const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ kv_norm,
    const int32_t* __restrict__ positions,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ q_down,
    const int32_t* __restrict__ row_seg,
    const uint64_t* __restrict__ kv_ptrs,
    int layer_off,
    int rows,
    int swa_mask,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    int seg = row_seg[r];
    if (seg < 0) {
        __nv_bfloat16* qdst = q_down + (int64_t)r * 1024;
        qdst[tid] = f2bf(0.0f); qdst[tid + 256] = f2bf(0.0f); qdst[tid + 512] = f2bf(0.0f); qdst[tid + 768] = f2bf(0.0f);
        return;
    }
    __nv_bfloat16* swa_kv = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[seg * layer_off + 0]);
    int lane = tid & 31, warp = tid >> 5;
    __shared__ float s_red[8];
    const __nv_bfloat16* src = packed + (int64_t)r * 1536;
    float q0 = bf2f(src[tid]), q1 = bf2f(src[tid + 256]), q2 = bf2f(src[tid + 512]), q3 = bf2f(src[tid + 768]);
    float qsq = warp_sum(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (lane == 0) s_red[warp] = qsq;
    __syncthreads();
    if (warp == 0) { float t = (lane < 8) ? s_red[lane] : 0.0f; t = warp_sum(t); if (lane == 0) s_red[0] = rsqrtf(t / 1024.0f + eps); }
    __syncthreads();
    float q_rrms = s_red[0];
    __nv_bfloat16* qdst = q_down + (int64_t)r * 1024;
    qdst[tid] = f2bf(q0 * q_rrms * bf2f(q_norm[tid]));
    qdst[tid + 256] = f2bf(q1 * q_rrms * bf2f(q_norm[tid + 256]));
    qdst[tid + 512] = f2bf(q2 * q_rrms * bf2f(q_norm[tid + 512]));
    qdst[tid + 768] = f2bf(q3 * q_rrms * bf2f(q_norm[tid + 768]));

    const __nv_bfloat16* ksrc = src + 1024;
    float k0 = bf2f(ksrc[2 * tid]), k1 = bf2f(ksrc[2 * tid + 1]);
    float ksq = warp_sum(k0 * k0 + k1 * k1);
    __syncthreads();
    if (lane == 0) s_red[warp] = ksq;
    __syncthreads();
    if (warp == 0) { float t = (lane < 8) ? s_red[lane] : 0.0f; t = warp_sum(t); if (lane == 0) s_red[0] = rsqrtf(t / 512.0f + eps); }
    __syncthreads();
    float k_rrms = s_red[0];
    float e = bf2f(f2bf(k0 * k_rrms * bf2f(kv_norm[2 * tid])));
    float o = bf2f(f2bf(k1 * k_rrms * bf2f(kv_norm[2 * tid + 1])));
    int pos = positions[r];
    if (tid >= 224) {
        int pair = tid - 224; float c, sv; __sincosf((float)pos * inv_freq[pair], &sv, &c);
        float ne = e * c - o * sv, no = o * c + e * sv; e = ne; o = no;
    }
    __nv_bfloat16* kdst = swa_kv + (int64_t)(pos & swa_mask) * 512;
    kdst[2 * tid] = f2bf(e); kdst[2 * tid + 1] = f2bf(o);
}

extern "C" __global__ void tf_ernie55_hca_comp_save_packed_tab(
    const __nv_bfloat16* __restrict__ packed,
    const float* __restrict__ ape,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ row_seg,
    const uint64_t* __restrict__ kv_ptrs,
    int layer_off,
    int rows,
    int state_mask
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int seg = row_seg[r];
    if (seg < 0) return;
    float* comp_state = reinterpret_cast<float*>(kv_ptrs[seg * layer_off + 1]);
    int pos = positions[r], slot = pos & state_mask, ape_row = pos & 127;
    const __nv_bfloat16* kvr = packed + (int64_t)r * 1024;
    const __nv_bfloat16* gr = kvr + 512;
    const float* ar = ape + (int64_t)ape_row * 512;
    float* dst_kv = comp_state + (int64_t)slot * 1024;
    float* dst_sc = dst_kv + 512;
    for (int d = threadIdx.x; d < 512; d += blockDim.x) {
        dst_kv[d] = bf2f(kvr[d]);
        dst_sc[d] = bf2f(gr[d]) + ar[d];
    }
}

extern "C" __global__ void tf_ernie55_hca_comp_flush_tab(
    const __nv_bfloat16* __restrict__ norm_w,
    const int32_t* __restrict__ positions,
    const float* __restrict__ inv_freq,
    const int32_t* __restrict__ row_seg,
    const uint64_t* __restrict__ kv_ptrs,
    int layer_off,
    int rows,
    int state_mask,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int seg = row_seg[r];
    if (seg < 0) return;
    int pos = positions[r];
    if (((pos + 1) & 127) != 0) return;
    const float* comp_state = reinterpret_cast<const float*>(kv_ptrs[seg * layer_off + 1]);
    __nv_bfloat16* comp_kv = reinterpret_cast<__nv_bfloat16*>(kv_ptrs[seg * layer_off + 2]);
    int blk = pos >> 7, s0 = blk << 7;
    __shared__ float s_ckv[512]; __shared__ float s_red[32];
    for (int pass = 0; pass < 2; ++pass) {
        int d = threadIdx.x + pass * 256; float mx = -1e38f;
        for (int t = 0; t < 128; ++t) { int slot = (s0 + t) & state_mask; mx = fmaxf(mx, comp_state[(int64_t)slot * 1024 + 512 + d]); }
        float denom = 0.0f, wsum = 0.0f;
        for (int t = 0; t < 128; ++t) {
            int slot = (s0 + t) & state_mask;
            float p = expf(comp_state[(int64_t)slot * 1024 + 512 + d] - mx);
            denom += p; wsum += p * comp_state[(int64_t)slot * 1024 + d];
        }
        s_ckv[d] = wsum / denom;
    }
    __syncthreads();
    float v0 = s_ckv[2 * threadIdx.x], v1 = s_ckv[2 * threadIdx.x + 1];
    float sq = warp_sum(v0 * v0 + v1 * v1);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();
    if (warp == 0) { float tot = (lane < 8) ? s_red[lane] : 0.0f; tot = warp_sum(tot); if (lane == 0) s_red[0] = rsqrtf(tot / 512.0f + eps); }
    __syncthreads();
    float rrms = s_red[0];
    float e = bf2f(f2bf(v0 * rrms * bf2f(norm_w[2 * threadIdx.x])));
    float o = bf2f(f2bf(v1 * rrms * bf2f(norm_w[2 * threadIdx.x + 1])));
    if (threadIdx.x >= 224) {
        int pair = threadIdx.x - 224; float c, sv; __sincosf((float)s0 * inv_freq[pair], &sv, &c);
        float ne = e * c - o * sv, no = o * c + e * sv; e = ne; o = no;
    }
    __nv_bfloat16* dst = comp_kv + (int64_t)blk * 512;
    dst[2 * threadIdx.x] = f2bf(e); dst[2 * threadIdx.x + 1] = f2bf(o);
}

extern "C" __global__ void tf_ernie55_attn_splitk_combine_tab(
    const float* __restrict__ opart,
    const float* __restrict__ mpart,
    const float* __restrict__ lpart,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ row_seg,
    const TFHcaSegEntry* __restrict__ seg_tab,
    int rows,
    int max_sp,
    int stride_sp
) {
    int r = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    if (r >= rows || h >= 16) return;
    int seg = row_seg[r];
    if (seg < 0) return;
    int splits = tf_hca_attn_splits(seg_tab[seg].n, max_sp);
    if (splits <= 1) return;
    const float* mr = mpart + (int64_t)r * stride_sp * 16;
    const float* lr = lpart + (int64_t)r * stride_sp * 16;
    float mmax = -1e38f;
    for (int s = 0; s < splits; ++s) mmax = fmaxf(mmax, mr[s * 16 + h]);
    float lsum = 0.0f, v0 = 0.0f, v1 = 0.0f;
    for (int s = 0; s < splits; ++s) {
        float ls = lr[s * 16 + h];
        if (ls > 0.0f) {
            float w = expf(mr[s * 16 + h] - mmax);
            lsum += ls * w;
            const float* op = opart + ((int64_t)r * stride_sp + s) * 16 * 512 + h * 512;
            v0 += op[d] * w; v1 += op[d + 256] * w;
        }
    }
    float inv = 1.0f / lsum;
    __nv_bfloat16* dst = out + ((int64_t)r * 16 + h) * 512;
    dst[d] = f2bf(v0 * inv); dst[d + 256] = f2bf(v1 * inv);
}
