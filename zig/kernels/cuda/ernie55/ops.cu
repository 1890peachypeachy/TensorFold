// ERNIE 5.5 Flash Preview core CUDA kernels (SM121 / GB10):
// mHC pre/post/contract, split-feature router, SITU, VHA, GPT-J RoPE/inv-RoPE,
// Walsh-Hadamard 128, RMSNorm/LayerNorm, MultiMax, and BF16 GEMV/GEMM.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float sigmoid_rn(float x) {
    return __fdiv_rn(1.0f, 1.0f + expf(-x));
}

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f2bf(float v) {
    return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float out = 0.0f;
    if (warp == 0) {
        out = (lane < nwarps) ? smem[lane] : 0.0f;
        out = warp_sum(out);
        if (lane == 0) smem[0] = out;
    }
    __syncthreads();
    return smem[0];
}

}  // namespace

// Broadcast token embeddings [rows, H] -> 4-stream mHC residual [rows, 4, H]
extern "C" __global__ void tf_ernie55_embed_broadcast(
    const int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int hidden,
    int hc_mult
) {
    int r = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || h >= hidden) return;
    int32_t tok = ids[r];
    __nv_bfloat16 v = table[(int64_t)tok * hidden + h];
    int64_t base = (int64_t)r * hc_mult * hidden + h;
    for (int c = 0; c < hc_mult; ++c) {
        out[base + (int64_t)c * hidden] = v;
    }
}

// ERNIE 5.5 mHC pre:
//   rms = 1 / (sqrt(mean(res^2)) + rms_eps)
//   mix[0..23] = (res_flat @ fn^T) * rms   (fn is [24, 4*H] F32)
//   pre[0..3]  = sigmoid(mix[0..3] * scale[0] + base[0..3]) + hc_eps
//   post[0..3] = sigmoid(mix[4..7] * scale[1] + base[4..7]) * post_alpha
//   comb[4x4]  = Sinkhorn(mix[8..23] * scale[2] + base[8..23], iters, hc_eps)
//   layer_in   = RMSNorm(bf16(sum_c pre[c] * res[c, :]), norm_w, norm_eps)
// Launch: grid=(rows), block=(256), smem = 4*H*2 + 256 bytes (~33 KB for H=4096)
extern "C" __global__ void tf_ernie55_mhc_pre(
    const __nv_bfloat16* __restrict__ res,      // [rows, 4, H]
    const float* __restrict__ fn,               // [24, 4*H]
    const float* __restrict__ hc_scale,         // [3]
    const float* __restrict__ hc_base,          // [24]
    const __nv_bfloat16* __restrict__ norm_w,   // [H]
    float* __restrict__ post_mix,               // [rows, 4]
    float* __restrict__ comb_mix,               // [rows, 16]
    __nv_bfloat16* __restrict__ layer_in,       // [rows, H]
    int rows,
    int hidden,
    float rms_eps,
    float hc_eps,
    float post_alpha,
    int sinkhorn_iters,
    float norm_eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const int hc_dim = 4 * hidden;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    extern __shared__ unsigned char raw_smem[];
    __nv_bfloat16* s_res = reinterpret_cast<__nv_bfloat16*>(raw_smem);
    float* s_red = reinterpret_cast<float*>(raw_smem + hc_dim * sizeof(__nv_bfloat16));
    float* s_mix = s_red + 32;  // 24 floats
    float* s_pre = s_mix + 24;  // 4 floats

    // Load residual [4*H] into shared memory and compute sqrsum
    const __nv_bfloat16* rptr = res + (int64_t)r * hc_dim;
    float sq = 0.0f;
    for (int i = tid; i < hc_dim; i += nthr) {
        __nv_bfloat16 b = rptr[i];
        s_res[i] = b;
        float f = bf2f(b);
        sq += f * f;
    }
    float total_sq = block_sum(sq, s_red);
    float inv_rms = 1.0f / (sqrtf(total_sq / (float)hc_dim) + rms_eps);

    // 24 dot products: 8 warps, each warp handles 3 output rows of fn [24, hc_dim]
    int warp = tid >> 5;
    int lane = tid & 31;
    int nwarps = nthr >> 5;
    for (int j = warp; j < 24; j += nwarps) {
        const float* frow = fn + (int64_t)j * hc_dim;
        float acc = 0.0f;
        for (int i = lane; i < hc_dim; i += 32) {
            acc += bf2f(s_res[i]) * frow[i];
        }
        acc = warp_sum(acc);
        if (lane == 0) s_mix[j] = acc * inv_rms;
    }
    __syncthreads();

    // Thread 0 computes pre, post, and 20-iter 4x4 Sinkhorn
    if (tid == 0) {
        float sc0 = hc_scale[0], sc1 = hc_scale[1], sc2 = hc_scale[2];
        for (int c = 0; c < 4; ++c) {
            s_pre[c] = sigmoid_rn(s_mix[c] * sc0 + hc_base[c]) + hc_eps;
            post_mix[(int64_t)r * 4 + c] = sigmoid_rn(s_mix[4 + c] * sc1 + hc_base[4 + c]) * post_alpha;
        }
        float cm[4][4];
        for (int i = 0; i < 4; ++i) {
            float rmax = -1e38f;
            for (int j = 0; j < 4; ++j) {
                float v = s_mix[8 + i * 4 + j] * sc2 + hc_base[8 + i * 4 + j];
                cm[i][j] = v;
                if (v > rmax) rmax = v;
            }
            float rsum = 0.0f;
            for (int j = 0; j < 4; ++j) {
                cm[i][j] = expf(cm[i][j] - rmax);
                rsum += cm[i][j];
            }
            for (int j = 0; j < 4; ++j) {
                cm[i][j] = cm[i][j] / rsum + hc_eps;
            }
        }
        // First col-norm
        for (int j = 0; j < 4; ++j) {
            float csum = cm[0][j] + cm[1][j] + cm[2][j] + cm[3][j] + hc_eps;
            for (int i = 0; i < 4; ++i) cm[i][j] /= csum;
        }
        // Remaining sinkhorn_iters - 1 iterations
        for (int it = 1; it < sinkhorn_iters; ++it) {
            for (int i = 0; i < 4; ++i) {
                float rsum = cm[i][0] + cm[i][1] + cm[i][2] + cm[i][3] + hc_eps;
                for (int j = 0; j < 4; ++j) cm[i][j] /= rsum;
            }
            for (int j = 0; j < 4; ++j) {
                float csum = cm[0][j] + cm[1][j] + cm[2][j] + cm[3][j] + hc_eps;
                for (int i = 0; i < 4; ++i) cm[i][j] /= csum;
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                comb_mix[(int64_t)r * 16 + i * 4 + j] = cm[i][j];
            }
        }
    }
    __syncthreads();

    // Collapse residual with pre_mix, round to BF16 (matching mhc_kernels.py), then RMSNorm
    float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float norm_sq = 0.0f;
    for (int h = tid; h < hidden; h += nthr) {
        float v = p0 * bf2f(s_res[h]) +
                  p1 * bf2f(s_res[hidden + h]) +
                  p2 * bf2f(s_res[2 * hidden + h]) +
                  p3 * bf2f(s_res[3 * hidden + h]);
        __nv_bfloat16 vb = f2bf(v);
        s_res[h] = vb;  // stash collapsed BF16 in s_res[0..hidden)
        float vf = bf2f(vb);
        norm_sq += vf * vf;
    }
    float total_norm_sq = block_sum(norm_sq, s_red);
    float rrms = rsqrtf(total_norm_sq / (float)hidden + norm_eps);

    __nv_bfloat16* optr = layer_in + (int64_t)r * hidden;
    for (int h = tid; h < hidden; h += nthr) {
        float v = bf2f(s_res[h]) * rrms * bf2f(norm_w[h]);
        optr[h] = f2bf(v);
    }
}

// ERNIE 5.5 mHC post:
//   out[r, j, h] = post_mix[r, j] * x[r, h] + sum_{k=0..3} comb_mix[r, k*4 + j] * res[r, k, h]
//   optional aux_mean[r, h] = 0.25 * sum_{j=0..3} out[r, j, h]
extern "C" __global__ void tf_ernie55_mhc_post(
    const __nv_bfloat16* __restrict__ x,        // [rows, H]
    const __nv_bfloat16* __restrict__ res,      // [rows, 4, H]
    const float* __restrict__ post_mix,         // [rows, 4]
    const float* __restrict__ comb_mix,         // [rows, 16] (row=in, col=out)
    __nv_bfloat16* __restrict__ out,            // [rows, 4, H]
    __nv_bfloat16* __restrict__ aux_mean,       // nullable [rows, H]
    int rows,
    int hidden
) {
    int r = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || h >= hidden) return;

    float xv = bf2f(x[(int64_t)r * hidden + h]);
    int64_t rbase = (int64_t)r * 4 * hidden + h;
    float b0 = bf2f(res[rbase]);
    float b1 = bf2f(res[rbase + hidden]);
    float b2 = bf2f(res[rbase + 2 * hidden]);
    float b3 = bf2f(res[rbase + 3 * hidden]);

    const float* pm = post_mix + (int64_t)r * 4;
    const float* cm = comb_mix + (int64_t)r * 16;

    float mean_acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float v = pm[j] * xv
                + cm[0 * 4 + j] * b0
                + cm[1 * 4 + j] * b1
                + cm[2 * 4 + j] * b2
                + cm[3 * 4 + j] * b3;
        __nv_bfloat16 vb = f2bf(v);
        out[rbase + (int64_t)j * hidden] = vb;
        mean_acc += bf2f(vb);
    }
    if (aux_mean != nullptr) {
        aux_mean[(int64_t)r * hidden + h] = f2bf(mean_acc * 0.25f);
    }
}

// ERNIE 5.5 final mHC contract + final RMSNorm (model.mhc_contract + model.norm):
//   rsqrt = rsqrt(mean(res^2) + rms_eps)
//   pre[m] = sigmoid((res_flat @ hc_head_fn[m]) * rsqrt * hc_head_scale[0] + hc_head_base[m]) + hc_eps
//   out = RMSNorm(bf16(sum_m pre[m] * res[m, :]), norm_w, norm_eps)
extern "C" __global__ void tf_ernie55_mhc_contract_norm(
    const __nv_bfloat16* __restrict__ res,      // [rows, 4, H]
    const float* __restrict__ fn,               // [4, 4*H]
    const float* __restrict__ hc_scale,         // [1]
    const float* __restrict__ hc_base,          // [4]
    const __nv_bfloat16* __restrict__ norm_w,   // [H]
    __nv_bfloat16* __restrict__ out,            // [rows, H]
    int rows,
    int hidden,
    float rms_eps,
    float hc_eps,
    float norm_eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const int hc_dim = 4 * hidden;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    extern __shared__ unsigned char raw_smem[];
    __nv_bfloat16* s_res = reinterpret_cast<__nv_bfloat16*>(raw_smem);
    float* s_red = reinterpret_cast<float*>(raw_smem + hc_dim * sizeof(__nv_bfloat16));
    float* s_pre = s_red + 32;

    const __nv_bfloat16* rptr = res + (int64_t)r * hc_dim;
    float sq = 0.0f;
    for (int i = tid; i < hc_dim; i += nthr) {
        __nv_bfloat16 b = rptr[i];
        s_res[i] = b;
        float f = bf2f(b);
        sq += f * f;
    }
    float total_sq = block_sum(sq, s_red);
    float rrms_hc = rsqrtf(total_sq / (float)hc_dim + rms_eps);

    int warp = tid >> 5;
    int lane = tid & 31;
    if (warp < 4) {
        const float* frow = fn + (int64_t)warp * hc_dim;
        float acc = 0.0f;
        for (int i = lane; i < hc_dim; i += 32) {
            acc += bf2f(s_res[i]) * frow[i];
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            s_pre[warp] = sigmoid_rn(acc * rrms_hc * hc_scale[0] + hc_base[warp]) + hc_eps;
        }
    }
    __syncthreads();

    float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    float norm_sq = 0.0f;
    for (int h = tid; h < hidden; h += nthr) {
        float v = p0 * bf2f(s_res[h]) +
                  p1 * bf2f(s_res[hidden + h]) +
                  p2 * bf2f(s_res[2 * hidden + h]) +
                  p3 * bf2f(s_res[3 * hidden + h]);
        __nv_bfloat16 vb = f2bf(v);
        s_res[h] = vb;
        float vf = bf2f(vb);
        norm_sq += vf * vf;
    }
    float total_norm_sq = block_sum(norm_sq, s_red);
    float rrms = rsqrtf(total_norm_sq / (float)hidden + norm_eps);

    __nv_bfloat16* optr = out + (int64_t)r * hidden;
    for (int h = tid; h < hidden; h += nthr) {
        optr[h] = f2bf(bf2f(s_res[h]) * rrms * bf2f(norm_w[h]));
    }
}

// Split-feature router logits: scores[r, e] = sigmoid(x[r] . w0[e]) + sigmoid(x[r] . w1[e])
// Launch: grid=(ceil(E/4), rows), block=(128) (1 warp per expert)
extern "C" __global__ void tf_ernie55_router_logits(
    const __nv_bfloat16* __restrict__ x,   // [rows, H]
    const __nv_bfloat16* __restrict__ w0,  // [E, H]
    const __nv_bfloat16* __restrict__ w1,  // [E, H]
    float* __restrict__ scores,            // [rows, E]
    int rows,
    int E,
    int hidden
) {
    int r = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int e = blockIdx.x * 4 + warp;
    if (r >= rows || e >= E) return;

    const __nv_bfloat16* xr = x + (int64_t)r * hidden;
    const __nv_bfloat16* r0 = w0 + (int64_t)e * hidden;
    const __nv_bfloat16* r1 = w1 + (int64_t)e * hidden;
    float a0 = 0.0f, a1 = 0.0f;
    for (int i = lane * 4; i < hidden; i += 128) {
        uint2 xv = *reinterpret_cast<const uint2*>(xr + i);
        uint2 v0 = *reinterpret_cast<const uint2*>(r0 + i);
        uint2 v1 = *reinterpret_cast<const uint2*>(r1 + i);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
        const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float xf = bf2f(xp[j]);
            a0 += xf * bf2f(p0[j]);
            a1 += xf * bf2f(p1[j]);
        }
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    if (lane == 0) {
        scores[(int64_t)r * E + e] = sigmoid_rn(a0) + sigmoid_rn(a1);
    }
}

// Top-K routing with e_score_correction_bias + renorm + learnable routed_scaling_factor_param
// Launch: grid=(rows), block=(32)
extern "C" __global__ void tf_ernie55_router_topk(
    const float* __restrict__ scores,      // [rows, E]
    const float* __restrict__ bias,        // [E]
    const float* __restrict__ scale,       // [E]
    int32_t* __restrict__ out_ids,         // [rows, K]
    float* __restrict__ out_weights,       // [rows, K]
    int rows,
    int E,
    int K,
    int norm_topk
) {
    int r = blockIdx.x;
    int lane = threadIdx.x;
    if (r >= rows) return;

    const float* sr = scores + (int64_t)r * E;
    // E <= 512 => 16 entries per lane
    float my_sel[16];
    float my_sc[16];
    int nloc = (E + 31) >> 5;
    for (int i = 0; i < nloc; ++i) {
        int idx = lane + i * 32;
        if (idx < E) {
            float sc = sr[idx];
            my_sc[i] = sc;
            my_sel[i] = sc + bias[idx];
        } else {
            my_sc[i] = 0.0f;
            my_sel[i] = -1e38f;
        }
    }

    int sel_ids[16];
    float sel_w[16];
    float wsum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float best_v = -1e38f;
        int best_i = E;
        float best_sc = 0.0f;
        for (int i = 0; i < nloc; ++i) {
            int idx = lane + i * 32;
            float v = my_sel[i];
            if (v > best_v || (v == best_v && idx < best_i)) {
                best_v = v;
                best_i = idx;
                best_sc = my_sc[i];
            }
        }
        // Warp reduction for max value with lowest-index tie-break
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
            int oi = __shfl_xor_sync(0xffffffffu, best_i, off);
            float osc = __shfl_xor_sync(0xffffffffu, best_sc, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
                best_sc = osc;
            }
        }
        sel_ids[k] = best_i;
        sel_w[k] = best_sc;
        wsum += best_sc;
        // Mask out chosen expert in owning lane
        if ((best_i & 31) == lane) {
            int slot = best_i >> 5;
            my_sel[slot] = -1e38f;
        }
    }

    if (lane == 0) {
        float inv_sum = (norm_topk && wsum > 0.0f) ? (1.0f / (wsum + 1e-20f)) : 1.0f;
        // Note: match vllm_ernie exact renorm: weights / sum (no 1e-20 if sum > 0)
        if (norm_topk && wsum > 0.0f) inv_sum = 1.0f / wsum;
        for (int k = 0; k < K; ++k) {
            int id = sel_ids[k];
            out_ids[(int64_t)r * K + k] = id;
            out_weights[(int64_t)r * K + k] = sel_w[k] * inv_sum * scale[id];
        }
    }
}

// SITU-GLU activation: (beta * tanh(g / beta) * sigmoid(g)) * (linear_beta * tanh(u / linear_beta))
extern "C" __global__ void tf_ernie55_situ_and_mul(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int64_t count,
    float beta,
    float linear_beta
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g = bf2f(gate[i]);
    float u = bf2f(up[i]);
    float sg = beta * tanhf(g / beta) * sigmoid_rn(g);
    float su = linear_beta * tanhf(u / linear_beta);
    out[i] = f2bf(sg * su);
}

// Standard RMSNorm (optional weight: if norm_w == nullptr, weightless RMSNorm)
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_rms_norm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,  // nullable
    __nv_bfloat16* __restrict__ out,
    int rows,
    int dim,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float s_red[32];
    const __nv_bfloat16* xr = x + (int64_t)r * dim;
    __nv_bfloat16* yr = out + (int64_t)r * dim;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = bf2f(xr[i]);
        sq += v * v;
    }
    float rrms = rsqrtf(block_sum(sq, s_red) / (float)dim + eps);
    if (w != nullptr) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            yr[i] = f2bf(bf2f(xr[i]) * rrms * bf2f(w[i]));
        }
    } else {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            yr[i] = f2bf(bf2f(xr[i]) * rrms);
        }
    }
}

// LayerNorm (for DSA indexer k_norm, dim=128)
// Launch: grid=(rows), block=(128)
extern "C" __global__ void tf_ernie55_layer_norm(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int dim,
    float eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float s_red[32];
    const __nv_bfloat16* xr = x + (int64_t)r * dim;
    __nv_bfloat16* yr = out + (int64_t)r * dim;
    float s1 = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s1 += bf2f(xr[i]);
    float mean = block_sum(s1, s_red) / (float)dim;
    float s2 = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = bf2f(xr[i]) - mean;
        s2 += d * d;
    }
    float inv = rsqrtf(block_sum(s2, s_red) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (bf2f(xr[i]) - mean) * inv * bf2f(w[i]) + bf2f(b[i]);
        yr[i] = f2bf(v);
    }
}

// Fused weightless per-head RMSNorm + GPT-J interleaved RoPE on trailing 64 dims of HCA Q [rows, heads, 512]
// Launch: grid=(heads, rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_qnorm_rope(
    __nv_bfloat16* __restrict__ q,          // [rows, heads, 512] in-place
    const int32_t* __restrict__ positions,  // [rows]
    const float* __restrict__ inv_freq,     // [32]
    int rows,
    int heads,
    float eps
) {
    int h = blockIdx.x;
    int r = blockIdx.y;
    if (h >= heads || r >= rows) return;
    __shared__ float s_red[32];
    __nv_bfloat16* qp = q + ((int64_t)r * heads + h) * 512;

    float v0 = bf2f(qp[threadIdx.x * 2]);
    float v1 = bf2f(qp[threadIdx.x * 2 + 1]);
    float rrms = rsqrtf(block_sum(v0 * v0 + v1 * v1, s_red) / 512.0f + eps);
    v0 *= rrms;
    v1 *= rrms;

    if (threadIdx.x >= 224) {  // pairs 224..255 => rope pairs 0..31 (dims 448..511)
        int pair = threadIdx.x - 224;
        float theta = (float)positions[r] * inv_freq[pair];
        float c, s;
        __sincosf(theta, &s, &c);
        float nv0 = v0 * c - v1 * s;
        float nv1 = v1 * c + v0 * s;
        v0 = nv0;
        v1 = nv1;
    }
    qp[threadIdx.x * 2] = f2bf(v0);
    qp[threadIdx.x * 2 + 1] = f2bf(v1);
}

// Interleaved GPT-J RoPE (forward: dir=+1, inverse: dir=-1) on [rows, heads, head_dim] at [rope_offset .. rope_offset+64)
// Launch: grid=(heads, rows), block=(32)
extern "C" __global__ void tf_ernie55_rope_gptj(
    __nv_bfloat16* __restrict__ x,          // [rows, heads, head_dim] in-place
    const int32_t* __restrict__ positions,  // [rows]
    const float* __restrict__ inv_freq,     // [32]
    int rows,
    int heads,
    int head_dim,
    int rope_offset,
    int dir                                 // +1 forward, -1 inverse
) {
    int h = blockIdx.x;
    int r = blockIdx.y;
    int pair = threadIdx.x;  // 0..31
    if (h >= heads || r >= rows || pair >= 32) return;
    __nv_bfloat16* p = x + ((int64_t)r * heads + h) * head_dim + rope_offset + pair * 2;
    float e = bf2f(p[0]);
    float o = bf2f(p[1]);
    float theta = (float)positions[r] * inv_freq[pair];
    float c, s;
    __sincosf(theta, &s, &c);
    if (dir < 0) s = -s;
    p[0] = f2bf(e * c - o * s);
    p[1] = f2bf(o * c + e * s);
}

// VHA post-mix step 1 (local U^T @ o): z[r, k, d] = sum_{h=0..H_loc-1} U[h, k] * o[r, h, d] (FP32)
// Launch: grid=(ceil(head_dim/32), rows), block=(32, 16) -> threadIdx.x = d_off, threadIdx.y = k (0..15)
extern "C" __global__ void tf_ernie55_vha_u_proj(
    const __nv_bfloat16* __restrict__ o,  // [rows, H_loc, head_dim]
    const __nv_bfloat16* __restrict__ U,  // [H_loc, 16]
    float* __restrict__ z,                // [rows, 16, head_dim]
    int rows,
    int H_loc,
    int head_dim
) {
    int d = blockIdx.x * 32 + threadIdx.x;
    int k = threadIdx.y;  // 0..15
    int r = blockIdx.y;
    if (r >= rows || d >= head_dim || k >= 16) return;
    float acc = 0.0f;
    const __nv_bfloat16* op = o + (int64_t)r * H_loc * head_dim + d;
    for (int h = 0; h < H_loc; ++h) {
        acc += bf2f(U[h * 16 + k]) * bf2f(op[(int64_t)h * head_dim]);
    }
    z[((int64_t)r * 16 + k) * head_dim + d] = acc;
}

// VHA post-mix step 2 (o += V @ z): o[r, h, d] = bf16(o[r, h, d] + sum_{k=0..15} V[h, k] * z[r, k, d])
// Launch: grid=(ceil(head_dim/32), rows), block=(32, H_loc) (H_loc <= 32, e.g. 16 on TP=4)
extern "C" __global__ void tf_ernie55_vha_v_add(
    __nv_bfloat16* __restrict__ o,        // [rows, H_loc, head_dim] in-place
    const __nv_bfloat16* __restrict__ V,  // [H_loc, 16]
    const float* __restrict__ z,          // [rows, 16, head_dim] (after TP AllReduce)
    int rows,
    int H_loc,
    int head_dim
) {
    int d = blockIdx.x * 32 + threadIdx.x;
    int r = blockIdx.y;
    if (r >= rows || d >= head_dim) return;
    const float* zp = z + (int64_t)r * 16 * head_dim + d;
    for (int h = threadIdx.y; h < H_loc; h += blockDim.y) {
        float acc = bf2f(o[((int64_t)r * H_loc + h) * head_dim + d]);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc += bf2f(V[h * 16 + k]) * zp[(int64_t)k * head_dim];
        }
        o[((int64_t)r * H_loc + h) * head_dim + d] = f2bf(acc);
    }
}

// Normalized Fast Walsh-Hadamard Transform of width 128 (in-place, 1 warp per row of 128 elements)
// Each lane holds 4 consecutive elements: [4*lane .. 4*lane + 3]
// Launch: grid=(ceil(num_rows / 4)), block=(128) (4 warps per block)
extern "C" __global__ void tf_ernie55_hadamard_128(
    __nv_bfloat16* __restrict__ x,  // [num_rows, 128]
    int num_rows
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= num_rows) return;

    __nv_bfloat16* p = x + (int64_t)row * 128 + lane * 4;
    float v0 = bf2f(p[0]), v1 = bf2f(p[1]), v2 = bf2f(p[2]), v3 = bf2f(p[3]);

    // Stride 1
    float a0 = v0 + v1, a1 = v0 - v1;
    float a2 = v2 + v3, a3 = v2 - v3;
    // Stride 2
    v0 = a0 + a2; v2 = a0 - a2;
    v1 = a1 + a3; v3 = a1 - a3;

    // Strides 4, 8, 16, 32, 64 across warp lanes (xor 1, 2, 4, 8, 16)
#pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1) {
        float o0 = __shfl_xor_sync(0xffffffffu, v0, mask);
        float o1 = __shfl_xor_sync(0xffffffffu, v1, mask);
        float o2 = __shfl_xor_sync(0xffffffffu, v2, mask);
        float o3 = __shfl_xor_sync(0xffffffffu, v3, mask);
        bool hi = (lane & mask) != 0;
        v0 = hi ? (o0 - v0) : (v0 + o0);
        v1 = hi ? (o1 - v1) : (v1 + o1);
        v2 = hi ? (o2 - v2) : (v2 + o2);
        v3 = hi ? (o3 - v3) : (v3 + o3);
    }

    const float inv_s = 0.08838834764831845f;  // 1 / sqrt(128)
    p[0] = f2bf(v0 * inv_s);
    p[1] = f2bf(v1 * inv_s);
    p[2] = f2bf(v2 * inv_s);
    p[3] = f2bf(v3 * inv_s);
}

// Sigmoid output gate: out[i] = bf16(bf2f(x[i]) * bf2f(f2bf(sigmoid(bf2f(gate[i])))))
extern "C" __global__ void tf_ernie55_sigmoid_mul_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int64_t count
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float s = bf2f(f2bf(sigmoid_rn(bf2f(gate[i]))));
    out[i] = f2bf(bf2f(x[i]) * s);
}

extern "C" __global__ void tf_ernie55_add_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t count
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}

// Exact 4-segment MultiMax transform on FP32 logits (preserves -inf)
extern "C" __global__ void tf_ernie55_multimax(
    float* __restrict__ logits,
    const float* __restrict__ ranges,  // [4]
    const float* __restrict__ ts,      // [4]
    int64_t count
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = logits[i];
    if (isinf(x) && x < 0.0f) return;
    float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
    float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];
    float d0 = fmaxf(__fsub_rn(r0, x), 0.0f);
    float d1 = fmaxf(__fsub_rn(x, r1), 0.0f);
    float d2 = fmaxf(__fsub_rn(r2, x), 0.0f);
    float d3 = fmaxf(__fsub_rn(x, r3), 0.0f);
    float y = __fadd_rn(x, __fmul_rn(t0, d0));
    y = __fadd_rn(y, __fmul_rn(t1, d1));
    y = __fadd_rn(y, __fmul_rn(t2, __fmul_rn(d2, d2)));
    y = __fadd_rn(y, __fmul_rn(t3, __fmul_rn(d3, d3)));
    logits[i] = y;
}

// Fast BF16 matmul: out[r, n] = sum_k x[r, k] * w[n, k] (out is BF16 if out_f32==nullptr, else F32)
// Launch: grid=(ceil(N/4), rows), block=(128) (1 warp per output row n)
extern "C" __global__ void tf_ernie55_bf16_linear(
    const __nv_bfloat16* __restrict__ x,     // [rows, K]
    const __nv_bfloat16* __restrict__ w,     // [N, K]
    __nv_bfloat16* __restrict__ out_bf16,    // nullable [rows, N]
    float* __restrict__ out_f32,             // nullable [rows, N]
    int rows,
    int N,
    int K
) {
    int r = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 4 + warp;
    if (r >= rows || n >= N) return;

    const __nv_bfloat16* xr = x + (int64_t)r * K;
    const __nv_bfloat16* wr = w + (int64_t)n * K;
    float acc = 0.0f;
    int k4 = K & ~3;
    for (int i = lane * 4; i < k4; i += 128) {
        uint2 xv = *reinterpret_cast<const uint2*>(xr + i);
        uint2 wv = *reinterpret_cast<const uint2*>(wr + i);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wv);
        acc += bf2f(xp[0]) * bf2f(wp[0])
             + bf2f(xp[1]) * bf2f(wp[1])
             + bf2f(xp[2]) * bf2f(wp[2])
             + bf2f(xp[3]) * bf2f(wp[3]);
    }
    for (int i = k4 + lane; i < K; i += 32) {
        acc += bf2f(xr[i]) * bf2f(wr[i]);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        if (out_f32 != nullptr) out_f32[(int64_t)r * N + n] = acc;
        if (out_bf16 != nullptr) out_bf16[(int64_t)r * N + n] = f2bf(acc);
    }
}

// NeoX split-half RoPE (for DSA Lightning Indexer leading 64 dims of head_dim=128)
// Launch: grid=(heads, rows), block=(32)
extern "C" __global__ void tf_ernie55_rope_neox(
    __nv_bfloat16* __restrict__ x,          // [rows, heads, head_dim] in-place
    const int32_t* __restrict__ positions,  // [rows]
    const float* __restrict__ inv_freq,     // [32]
    int rows,
    int heads,
    int head_dim,
    int rope_offset
) {
    int h = blockIdx.x;
    int r = blockIdx.y;
    int i = threadIdx.x;  // 0..31
    if (h >= heads || r >= rows || i >= 32) return;
    __nv_bfloat16* p = x + ((int64_t)r * heads + h) * head_dim + rope_offset;
    float x0 = bf2f(p[i]);
    float x1 = bf2f(p[i + 32]);
    float theta = (float)positions[r] * inv_freq[i];
    float c, s;
    __sincosf(theta, &s, &c);
    p[i]      = f2bf(x0 * c - x1 * s);
    p[i + 32] = f2bf(x1 * c + x0 * s);
}

// Pack [a (rows, wa), b (rows, wb)] -> dst (rows, wa+wb) BF16 for single fused AllReduce
extern "C" __global__ void tf_ernie55_pack_pair_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ dst,
    int rows,
    int wa,
    int wb
) {
    int r = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int wt = wa + wb;
    if (r >= rows || col >= wt) return;
    if (col < wa) {
        dst[(int64_t)r * wt + col] = a[(int64_t)r * wa + col];
    } else {
        dst[(int64_t)r * wt + col] = b[(int64_t)r * wb + (col - wa)];
    }
}

extern "C" __global__ void tf_ernie55_unpack_pair_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ a,
    __nv_bfloat16* __restrict__ b,
    int rows,
    int wa,
    int wb
) {
    int r = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int wt = wa + wb;
    if (r >= rows || col >= wt) return;
    __nv_bfloat16 v = src[(int64_t)r * wt + col];
    if (col < wa) {
        a[(int64_t)r * wa + col] = v;
    } else {
        b[(int64_t)r * wb + (col - wa)] = v;
    }
}

// Deterministic rank-order sum over all-gathered [world, count] -> [count] BF16
extern "C" __global__ void tf_ernie55_rank_sum_bf16(
    const __nv_bfloat16* __restrict__ gathered,
    __nv_bfloat16* __restrict__ out,
    int64_t count,
    int world
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float s = 0.0f;
    for (int r = 0; r < world; ++r) {
        s += bf2f(gathered[(int64_t)r * count + i]);
    }
    out[i] = f2bf(s);
}

// Deterministic rank-order sum over all-gathered [world, count] -> [count] F32
extern "C" __global__ void tf_ernie55_rank_sum_f32(
    const float* __restrict__ gathered,
    float* __restrict__ out,
    int64_t count,
    int world
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float s = 0.0f;
    for (int r = 0; r < world; ++r) {
        s += gathered[(int64_t)r * count + i];
    }
    out[i] = s;
}

// Local argmax over [rows, vocab_loc] FP32 -> out_id [rows] (global id = vocab_lo + local_id), out_val [rows]
// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_argmax_f32(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_id,
    float* __restrict__ out_val,
    int rows,
    int vocab_loc,
    int vocab_lo
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* lr = logits + (int64_t)r * vocab_loc;
    float bv = -1e38f;
    int bi = 0;
    for (int i = threadIdx.x; i < vocab_loc; i += blockDim.x) {
        float v = lr[i];
        if (v > bv) { bv = v; bi = i; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        bv = (lane < 8) ? sv[lane] : -1e38f;
        bi = (lane < 8) ? si[lane] : 0;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, bv, off);
            int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            out_id[r] = vocab_lo + bi;
            out_val[r] = bv;
        }
    }
}

// Transpose [groups, rows, width] BF16 -> [rows, groups, width] BF16 (for HCA multi-row wo_a output)
extern "C" __global__ void tf_ernie55_interleave_groups(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int groups,
    int rows,
    int width
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int g = blockIdx.z;
    if (col >= width || r >= rows || g >= groups) return;
    dst[((int64_t)r * groups + g) * width + col] = src[((int64_t)g * rows + r) * width + col];
}

// Fast multi-CTA mHC pre pass 1: 24 CTAs per token compute mix_raw[r, j] = dot(res[r], fn[j]) with float4 loads,
// and CTA 0 also computes sqrsum[r] = sum(res[r]^2). Saturates 24 SMs even at rows=1!
// Launch: grid=(24, rows), block=(256)
extern "C" __global__ void tf_ernie55_mhc_pre_gemv(
    const __nv_bfloat16* __restrict__ res,  // [rows, 4*H]
    const float* __restrict__ fn,           // [24, 4*H]
    float* __restrict__ mix_raw,            // [rows, 24]
    float* __restrict__ sqrsum_out,         // [rows]
    int rows,
    int hc_dim
) {
    int j = blockIdx.x;
    int r = blockIdx.y;
    if (j >= 24 || r >= rows) return;
    __shared__ float s_red[32];
    const __nv_bfloat16* rp = res + (int64_t)r * hc_dim;
    const float* fp = fn + (int64_t)j * hc_dim;

    float acc = 0.0f;
    float sq = 0.0f;
    int n4 = hc_dim >> 2;
    const uint2* rp4 = reinterpret_cast<const uint2*>(rp);
    const float4* fp4 = reinterpret_cast<const float4*>(fp);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        uint2 rv = __ldg(rp4 + i);
        float4 fv = __ldg(fp4 + i);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&rv);
        float x0 = bf2f(b[0]), x1 = bf2f(b[1]), x2 = bf2f(b[2]), x3 = bf2f(b[3]);
        acc = fmaf(x0, fv.x, acc);
        acc = fmaf(x1, fv.y, acc);
        acc = fmaf(x2, fv.z, acc);
        acc = fmaf(x3, fv.w, acc);
        if (j == 0) {
            sq = fmaf(x0, x0, sq);
            sq = fmaf(x1, x1, sq);
            sq = fmaf(x2, x2, sq);
            sq = fmaf(x3, x3, sq);
        }
    }
    float tot = block_sum(acc, s_red);
    if (threadIdx.x == 0) mix_raw[(int64_t)r * 24 + j] = tot;
    if (j == 0) {
        float tot_sq = block_sum(sq, s_red);
        if (threadIdx.x == 0) sqrsum_out[r] = tot_sq;
    }
}

// Fast mHC pre pass 2: Sinkhorn + BF16 collapse + fused RMSNorm (zero fn reads!)
// Launch: grid=(rows), block=(256), smem = hidden * 2 + 256 bytes (~8.5 KB)
extern "C" __global__ void tf_ernie55_mhc_pre_finish(
    const __nv_bfloat16* __restrict__ res,    // [rows, 4, H]
    const float* __restrict__ mix_raw,        // [rows, 24]
    const float* __restrict__ sqrsum_in,      // [rows]
    const float* __restrict__ hc_scale,       // [3]
    const float* __restrict__ hc_base,        // [24]
    const __nv_bfloat16* __restrict__ norm_w, // [H]
    float* __restrict__ post_mix,             // [rows, 4]
    float* __restrict__ comb_mix,             // [rows, 16]
    __nv_bfloat16* __restrict__ layer_in,     // [rows, H]
    int rows,
    int hidden,
    float rms_eps,
    float hc_eps,
    float post_alpha,
    int sinkhorn_iters,
    float norm_eps
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int hc_dim = 4 * hidden;

    extern __shared__ unsigned char raw_smem[];
    __nv_bfloat16* s_col = reinterpret_cast<__nv_bfloat16*>(raw_smem);
    float* s_red = reinterpret_cast<float*>(raw_smem + hidden * sizeof(__nv_bfloat16));
    float* s_pre = s_red + 32;

    if (tid < 16) {
        float inv_rms = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(sqrsum_in[r], (float)hc_dim)) + rms_eps);
        const float* mr = mix_raw + (int64_t)r * 24;
        if (tid < 4) {
            s_pre[tid] = sigmoid_rn((mr[tid] * inv_rms) * hc_scale[0] + hc_base[tid]) + hc_eps;
            post_mix[(int64_t)r * 4 + tid] = sigmoid_rn((mr[4 + tid] * inv_rms) * hc_scale[1] + hc_base[4 + tid]) * post_alpha;
        }
        // Lanes 0..15 execute 4x4 Sinkhorn in parallel (16x fewer serial IEEE divisions!)
        float v = (mr[8 + tid] * inv_rms) * hc_scale[2] + hc_base[8 + tid];
        float rmax = v;
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFu, rmax, 1));
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFu, rmax, 2));
        float ev = expf(v - rmax);
        float rsum = ev;
        rsum += __shfl_xor_sync(0xFFFFu, rsum, 1);
        rsum += __shfl_xor_sync(0xFFFFu, rsum, 2);
        float cm = __fdiv_rn(ev, rsum) + hc_eps;
        float csum = cm;
        csum += __shfl_xor_sync(0xFFFFu, csum, 4);
        csum += __shfl_xor_sync(0xFFFFu, csum, 8);
        cm = __fdiv_rn(cm, csum + hc_eps);
        for (int it = 1; it < sinkhorn_iters; ++it) {
            float rs = cm;
            rs += __shfl_xor_sync(0xFFFFu, rs, 1);
            rs += __shfl_xor_sync(0xFFFFu, rs, 2);
            cm = __fdiv_rn(cm, rs + hc_eps);
            float cs = cm;
            cs += __shfl_xor_sync(0xFFFFu, cs, 4);
            cs += __shfl_xor_sync(0xFFFFu, cs, 8);
            cm = __fdiv_rn(cm, cs + hc_eps);
        }
        comb_mix[(int64_t)r * 16 + tid] = cm;
    }
    __syncthreads();

    float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
    const __nv_bfloat16* rptr = res + (int64_t)r * hc_dim;
    float norm_sq = 0.0f;
    for (int h = tid; h < hidden; h += nthr) {
        float v = p0 * bf2f(rptr[h])
                + p1 * bf2f(rptr[hidden + h])
                + p2 * bf2f(rptr[2 * hidden + h])
                + p3 * bf2f(rptr[3 * hidden + h]);
        __nv_bfloat16 vb = f2bf(v);
        s_col[h] = vb;
        float vf = bf2f(vb);
        norm_sq += vf * vf;
    }
    float total_norm_sq = block_sum(norm_sq, s_red);
    float rrms = rsqrtf(total_norm_sq / (float)hidden + norm_eps);

    __nv_bfloat16* optr = layer_in + (int64_t)r * hidden;
    for (int h = tid; h < hidden; h += nthr) {
        optr[h] = f2bf(bf2f(s_col[h]) * rrms * bf2f(norm_w[h]));
    }
}

// Fast uint4-vectorized BF16 matmul (128-bit loads = 8 BF16/load, 2x memory bandwidth over uint2!)
// Launch: grid=(ceil(N/4), rows), block=(128) (1 warp per output row n)
extern "C" __global__ void tf_ernie55_bf16_linear_v4(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out_bf16,
    float* __restrict__ out_f32,
    int rows,
    int N,
    int K
) {
    int r = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 4 + warp;
    if (r >= rows || n >= N) return;

    const __nv_bfloat16* xr = x + (int64_t)r * K;
    const __nv_bfloat16* wr = w + (int64_t)n * K;
    float acc0 = 0.0f, acc1 = 0.0f;
    int k8 = K >> 3;
    const uint4* x8 = reinterpret_cast<const uint4*>(xr);
    const uint4* w8 = reinterpret_cast<const uint4*>(wr);
    for (int i = lane; i < k8; i += 32) {
        uint4 xv = __ldg(x8 + i);
        uint4 wv = __ldg(w8 + i);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __bfloat1622float2(xp[j]);
            float2 wf = __bfloat1622float2(wp[j]);
            acc0 = fmaf(xf.x, wf.x, acc0);
            acc1 = fmaf(xf.y, wf.y, acc1);
        }
    }
    float acc = warp_sum(acc0 + acc1);
    if (lane == 0) {
        if (out_f32 != nullptr) out_f32[(int64_t)r * N + n] = acc;
        if (out_bf16 != nullptr) out_bf16[(int64_t)r * N + n] = f2bf(acc);
    }
}

// Split-feature router combine: scores[r, e] = sigmoid_rn(raw[r, e]) + sigmoid_rn(raw[r, E + e])
// Launch: grid=(ceil(rows*E/256)), block=(256)
extern "C" __global__ void tf_ernie55_router_split_sum(
    const float* __restrict__ raw,   // [rows, 2*E]
    float* __restrict__ scores,      // [rows, E]
    int rows,
    int E
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * E;
    if (idx >= total) return;
    int r = idx / E;
    int e = idx - r * E;
    const float* row = raw + (int64_t)r * (2 * E);
    scores[idx] = sigmoid_rn(row[e]) + sigmoid_rn(row[E + e]);
}

// Concat three [rows, D] BF16 buffers into [rows, 3*D] BF16 for DSpark taps.
// Launch: grid=(ceil(rows*D/256)), block=(256)
extern "C" __global__ void tf_ernie55_concat3_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ c,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * D;
    if (idx >= total) return;
    int r = (int)(idx / D);
    int d = (int)(idx - (int64_t)r * D);
    __nv_bfloat16* o = out + (int64_t)r * (3 * D);
    o[d] = a[idx];
    o[D + d] = b[idx];
    o[2 * D + d] = c[idx];
}

// DSpark Markov bias + MultiMax:
// scores[v] = multimax(logits[v] + dot(w2[v, 0..256], w1[walk_ptr[0], 0..256]))
// Launch: grid=(ceil(nv / 256)), block=(256)
extern "C" __global__ void tf_ernie55_dspark_markov_multimax(
    const float* __restrict__ logits,       // [nv]
    const __nv_bfloat16* __restrict__ w2,   // [nv, 256]
    const __nv_bfloat16* __restrict__ w1,   // [vocab, 256]
    const int32_t* __restrict__ walk_ptr,   // [1] previous token id
    float* __restrict__ scores,             // [nv]
    const float* __restrict__ ranges,       // [4]
    const float* __restrict__ ts,           // [4]
    int nv
) {
    __shared__ float s_w1[256];
    int tid = threadIdx.x;
    int prev = walk_ptr[0];
    if (tid < 256) {
        s_w1[tid] = bf2f(w1[(int64_t)prev * 256 + tid]);
    }
    __syncthreads();

    int v = blockIdx.x * blockDim.x + tid;
    if (v >= nv) return;

    const uint4* w2v = reinterpret_cast<const uint4*>(w2 + (int64_t)v * 256);
    float bias = 0.0f;
#pragma unroll 4
    for (int i = 0; i < 32; ++i) {
        uint4 pack = w2v[i];
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&pack);
        int base = i * 8;
        bias += bf2f(p[0]) * s_w1[base + 0]
              + bf2f(p[1]) * s_w1[base + 1]
              + bf2f(p[2]) * s_w1[base + 2]
              + bf2f(p[3]) * s_w1[base + 3]
              + bf2f(p[4]) * s_w1[base + 4]
              + bf2f(p[5]) * s_w1[base + 5]
              + bf2f(p[6]) * s_w1[base + 6]
              + bf2f(p[7]) * s_w1[base + 7];
    }

    float x = logits[v] + bias;
    if (!(isinf(x) && x < 0.0f)) {
        float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
        float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];
        float d0 = fmaxf(__fsub_rn(r0, x), 0.0f);
        float d1 = fmaxf(__fsub_rn(x, r1), 0.0f);
        float d2 = fmaxf(__fsub_rn(r2, x), 0.0f);
        float d3 = fmaxf(__fsub_rn(x, r3), 0.0f);
        float y = __fadd_rn(x, __fmul_rn(t0, d0));
        y = __fadd_rn(y, __fmul_rn(t1, d1));
        y = __fadd_rn(y, __fmul_rn(t2, __fmul_rn(d2, d2)));
        y = __fadd_rn(y, __fmul_rn(t3, __fmul_rn(d3, d3)));
        x = y;
    }
    scores[v] = x;
}

// Merge world ranks' 16-byte {float val, int32 id, ...} picks -> walk[step + 1]
// Launch: grid=(1), block=(1)
extern "C" __global__ void tf_ernie55_pick_merge(
    const uint8_t* __restrict__ picks,  // [world * stride_bytes]
    int world,
    int stride_bytes,
    int32_t* __restrict__ walk,         // [max_steps + 1]
    int step
) {
    float best_v = -INFINITY;
    int32_t best_id = 0;
    for (int r = 0; r < world; ++r) {
        const uint8_t* p = picks + r * stride_bytes;
        float v = *reinterpret_cast<const float*>(p);
        int32_t id = *reinterpret_cast<const int32_t*>(p + 4);
        if (r == 0 || v > best_v || (v == best_v && id < best_id)) {
            best_v = v;
            best_id = id;
        }
    }
    walk[step + 1] = best_id;
}

// Convert F32 array to BF16 array (for sample.cu BF16 logits gather).
// Launch: grid=(ceil(count/256)), block=(256)
extern "C" __global__ void tf_ernie55_f32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t count
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2bfloat16(src[i]);
}

// Write up to 96 u32 command words into device buffer without host-to-device sync (for TP leader broadcast).
// Launch: grid=(1), block=(128)
struct TfErnie55Words { uint32_t w[96]; };
extern "C" __global__ void tf_ernie55_put_words(
    uint32_t* __restrict__ dst,
    uint32_t count,
    TfErnie55Words words
) {
    uint32_t i = threadIdx.x;
    if (i < count && i < 96) dst[i] = words.w[i];
}
// mHC pre-mix for prompt rows (Spark, 2026-11-24; appended to ops.cu). Same outputs as tf_ernie55_mhc_pre_gemv:
//   mix_raw[r, j] = sum_i res[r, i] * fn[j, i] (j < 24), sqrsum[r] = sum_i res[r, i]^2, over hc_dim = 4 * hidden.
// tf_ernie55_mhc_pre_gemv runs a block a (j, r) and reads row r 24 times and fn once a row; this one tiles 8 rows a
// block over K chunks of 256 in shared memory: each of the 8 x 24 outputs (and 8 squares) is one thread's fp32 sum.
// Launch: grid=(ceil(rows/8)), block=(256), smem static.
#define TF_MHC_ROWS 8
#define TF_MHC_KC 256
extern "C" __global__ void __launch_bounds__(256) tf_ernie55_mhc_pre_tiled(
    const __nv_bfloat16* __restrict__ res, const float* __restrict__ fn, float* __restrict__ mix_raw,
    float* __restrict__ sqrsum_out, int rows, int hc_dim) {
    __shared__ float xs[TF_MHC_ROWS][TF_MHC_KC + 1];
    __shared__ float fs[24][TF_MHC_KC + 1];
    const int r0 = blockIdx.x * TF_MHC_ROWS;
    const int t = threadIdx.x;
    // outputs: t < 192 -> (row t / 24, j t % 24); 192 <= t < 200 -> square of row t - 192
    const int orow = t < 192 ? t / 24 : t - 192;
    const int oj = t % 24;
    float acc = 0.0f;
    for (int k0 = 0; k0 < hc_dim; k0 += TF_MHC_KC) {
        __syncthreads();
        for (int i = t; i < TF_MHC_ROWS * TF_MHC_KC; i += blockDim.x) {
            const int rr = i / TF_MHC_KC, c = i % TF_MHC_KC, r = r0 + rr;
            xs[rr][c] = r < rows ? __bfloat162float(res[(int64_t)r * hc_dim + k0 + c]) : 0.0f;
        }
        for (int i = t; i < 24 * TF_MHC_KC; i += blockDim.x) {
            const int jj = i / TF_MHC_KC, c = i % TF_MHC_KC;
            fs[jj][c] = fn[(int64_t)jj * hc_dim + k0 + c];
        }
        __syncthreads();
        if (t < 192) {
#pragma unroll 8
            for (int c = 0; c < TF_MHC_KC; ++c) acc = fmaf(xs[orow][c], fs[oj][c], acc);
        } else if (t < 200) {
#pragma unroll 8
            for (int c = 0; c < TF_MHC_KC; ++c) acc = fmaf(xs[orow][c], xs[orow][c], acc);
        }
    }
    const int r = r0 + orow;
    if (r < rows) {
        if (t < 192) mix_raw[(int64_t)r * 24 + oj] = acc;
        else if (t < 200) sqrsum_out[r] = acc;
    }
}


// --- Batch 1 fused decode & sampler kernels ---

// Fused split-feature BF16 router for decode rows <= 8:
//   scores[r, e] = sigmoid(x[r] . w0[e]) + sigmoid(x[r] . w1[e])
// Reads w0[e] and w1[e] once with uint4 across all `rows` (1..8) -> ~213 GB/s on GB10!
// Launch: grid=(ceil(E / 8)), block=(256) (8 warps = 8 experts per block)
template <int MAX_R>
__device__ __forceinline__ void router_fused_bf16_body(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w0,
    const __nv_bfloat16* __restrict__ w1,
    float* __restrict__ scores,
    int rows,
    int E,
    int H
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int e = blockIdx.x * 8 + warp;
    if (e >= E) return;

    float a0[MAX_R], a1[MAX_R];
#pragma unroll
    for (int r = 0; r < MAX_R; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }

    int n8 = H >> 3;
    const uint4* p0 = reinterpret_cast<const uint4*>(w0 + (int64_t)e * H);
    const uint4* p1 = reinterpret_cast<const uint4*>(w1 + (int64_t)e * H);

    for (int i = lane; i < n8; i += 32) {
        uint4 v0 = __ldg(p0 + i);
        uint4 v1 = __ldg(p1 + i);
        const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
        const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
        float f0[8], f1[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { f0[u] = bf2f(b0[u]); f1[u] = bf2f(b1[u]); }
#pragma unroll
        for (int r = 0; r < MAX_R; ++r) {
            if (r < rows) {
                uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)r * H) + i);
                const __nv_bfloat16* bx = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float fx = bf2f(bx[u]);
                    a0[r] = fmaf(fx, f0[u], a0[r]);
                    a1[r] = fmaf(fx, f1[u], a1[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        if (r < rows) {
            float s0 = a0[r], s1 = a1[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s0 += __shfl_down_sync(0xffffffffu, s0, off);
                s1 += __shfl_down_sync(0xffffffffu, s1, off);
            }
            if (lane == 0) {
                scores[(int64_t)r * E + e] = sigmoid_rn(s0) + sigmoid_rn(s1);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_ernie55_router_fused_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w0,
    const __nv_bfloat16* __restrict__ w1,
    float* __restrict__ scores,
    int rows,
    int E,
    int H
) {
    if (rows <= 4) {
        router_fused_bf16_body<4>(x, w0, w1, scores, rows, E, H);
    } else {
        router_fused_bf16_body<8>(x, w0, w1, scores, rows, E, H);
    }
}

// Split fused MoE projection [rows, latent + 2*w] into latent_out [rows, latent]
// and shared_mid [rows, w] = SITU(w1) * w3 in ONE launch!
// Launch: grid=(ceil((latent + w) / 256), rows), block=(256)
extern "C" __global__ void tf_ernie55_moe_split_situ(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ latent_out,
    __nv_bfloat16* __restrict__ shared_mid,
    int rows,
    int latent,
    int w,
    float beta,
    float linear_beta
) {
    int r = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || col >= latent + w) return;
    const __nv_bfloat16* src = packed + (int64_t)r * (latent + 2 * w);
    if (col < latent) {
        latent_out[(int64_t)r * latent + col] = src[col];
    } else {
        int j = col - latent;
        float g = bf2f(src[latent + j]);
        float u = bf2f(src[latent + w + j]);
        float sg = beta * tanhf(g / beta) * sigmoid_rn(g);
        float su = linear_beta * tanhf(u / linear_beta);
        shared_mid[(int64_t)r * w + j] = f2bf(sg * su);
    }
}

// Packed Dense MLP SITU-GLU: reads [rows, 2*w] -> writes [rows, w] = SITU(gate) * up
// Launch: grid=(ceil(w / 256), rows), block=(256)
extern "C" __global__ void tf_ernie55_situ_and_mul_packed(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int w,
    float beta,
    float linear_beta
) {
    int r = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || j >= w) return;
    const __nv_bfloat16* src = packed + (int64_t)r * (2 * w);
    float g = bf2f(src[j]);
    float u = bf2f(src[w + j]);
    float sg = beta * tanhf(g / beta) * sigmoid_rn(g);
    float su = linear_beta * tanhf(u / linear_beta);
    out[(int64_t)r * w + j] = f2bf(sg * su);
}

// Fused HCA q_up + o_gate post-op:
//   packed [rows, heads*512 + o_width] ->
//     q_out [rows, heads, 512] = weightless per-head RMSNorm + GPT-J RoPE [448..512)
//     o_gate [rows, o_width]   = copy of packed[r, heads*512 ..]
// Launch: grid=(heads + ceil(o_width / 512), rows), block=(256)
extern "C" __global__ void tf_ernie55_hca_qnorm_rope_split(
    const __nv_bfloat16* __restrict__ packed,
    const int32_t* __restrict__ positions,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ o_gate,
    int rows,
    int heads,
    int o_width,
    float eps
) {
    int b = blockIdx.x;
    int r = blockIdx.y;
    if (r >= rows) return;
    int q_width = heads * 512;
    const __nv_bfloat16* row_in = packed + (int64_t)r * (q_width + o_width);
    if (b < heads) {
        __shared__ float s_red[32];
        const __nv_bfloat16* src = row_in + (int64_t)b * 512;
        __nv_bfloat16* dst = q_out + ((int64_t)r * heads + b) * 512;
        float v0 = bf2f(src[threadIdx.x * 2]);
        float v1 = bf2f(src[threadIdx.x * 2 + 1]);
        float rrms = rsqrtf(block_sum(v0 * v0 + v1 * v1, s_red) / 512.0f + eps);
        v0 *= rrms;
        v1 *= rrms;
        if (threadIdx.x >= 224) {
            int pair = threadIdx.x - 224;
            float theta = (float)positions[r] * inv_freq[pair];
            float c, s;
            __sincosf(theta, &s, &c);
            float nv0 = v0 * c - v1 * s;
            float nv1 = v1 * c + v0 * s;
            v0 = nv0;
            v1 = nv1;
        }
        dst[threadIdx.x * 2] = f2bf(v0);
        dst[threadIdx.x * 2 + 1] = f2bf(v1);
    } else {
        int g = b - heads;
        int off = g * 512 + threadIdx.x * 2;
        if (off < o_width) {
            const __nv_bfloat16* src = row_in + q_width + off;
            __nv_bfloat16* dst = o_gate + (int64_t)r * o_width + off;
            dst[0] = src[0];
            if (off + 1 < o_width) dst[1] = src[1];
        }
    }
}

// Fused DSA q_b + o_gate post-op:
//   packed [rows, heads*256 + o_width] ->
//     q_out [rows, heads, 256] with GPT-J RoPE on [192..256)
//     o_gate [rows, o_width]   = copy of packed[r, heads*256 ..]
// Launch: grid=(heads + ceil(o_width / 256), rows), block=(128)
extern "C" __global__ void tf_ernie55_dsa_qb_rope_split(
    const __nv_bfloat16* __restrict__ packed,
    const int32_t* __restrict__ positions,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ o_gate,
    int rows,
    int heads,
    int o_width
) {
    int b = blockIdx.x;
    int r = blockIdx.y;
    if (r >= rows) return;
    int q_width = heads * 256;
    const __nv_bfloat16* row_in = packed + (int64_t)r * (q_width + o_width);
    if (b < heads) {
        const __nv_bfloat16* src = row_in + (int64_t)b * 256;
        __nv_bfloat16* dst = q_out + ((int64_t)r * heads + b) * 256;
        int i0 = threadIdx.x * 2;
        if (i0 < 192) {
            dst[i0] = src[i0];
            dst[i0 + 1] = src[i0 + 1];
        } else {
            int pair = (i0 - 192) >> 1;
            float e = bf2f(src[i0]);
            float o = bf2f(src[i0 + 1]);
            float theta = (float)positions[r] * inv_freq[pair];
            float c, s;
            __sincosf(theta, &s, &c);
            dst[i0] = f2bf(e * c - o * s);
            dst[i0 + 1] = f2bf(o * c + e * s);
        }
    } else {
        int g = b - heads;
        int off = g * 256 + threadIdx.x * 2;
        if (off < o_width) {
            const __nv_bfloat16* src = row_in + q_width + off;
            __nv_bfloat16* dst = o_gate + (int64_t)r * o_width + off;
            dst[0] = src[0];
            if (off + 1 < o_width) dst[1] = src[1];
        }
    }
}

// Unpack rank-major RoCE allGather buffer [world, rows, vw] BF16 -> row-major [rows, world * vw] BF16
// Launch: grid=(ceil(world * vw / 256), rows), block=(256)
extern "C" __global__ void tf_ernie55_unpack_vocab_bf16(
    const __nv_bfloat16* __restrict__ gathered,
    __nv_bfloat16* __restrict__ full,
    int world,
    int rows,
    int vw
) {
    int r = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int V = world * vw;
    if (r >= rows || col >= V) return;
    int q = col / vw;
    int off = col - q * vw;
    full[(int64_t)r * V + col] = gathered[((int64_t)q * rows + r) * vw + off];
}


// --- Batch 2 batched multi-stream DSpark Markov walk kernels ---

// Launch: grid=(ceil(nv / 256), M), block=(256)
extern "C" __global__ void tf_ernie55_dspark_markov_multimax_batch(
    const float* __restrict__ logits,          // [M * 7, nv]
    const __nv_bfloat16* __restrict__ w2,      // [nv, 256]
    const __nv_bfloat16* __restrict__ w1,      // [vocab, 256]
    const int32_t* __restrict__ walk,          // [M, 8]
    float* __restrict__ scores,                // [M, nv]
    const float* __restrict__ ranges,          // [4]
    const float* __restrict__ ts,              // [4]
    int M,
    int step,
    int nv
) {
    int m = blockIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    __shared__ float s_w1[256];
    int tok = walk[m * 8 + step];
    const __nv_bfloat16* w1_row = w1 + (int64_t)tok * 256;
    s_w1[threadIdx.x] = bf2f(w1_row[threadIdx.x]);
    __syncthreads();

    if (v >= nv) return;
    const __nv_bfloat16* w2_row = w2 + (int64_t)v * 256;
    float dot = 0.0f;
    const uint4* w2_v4 = reinterpret_cast<const uint4*>(w2_row);
#pragma unroll 4
    for (int i = 0; i < 32; ++i) {
        uint4 raw = __ldg(w2_v4 + i);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
        int k = i * 8;
        dot = fmaf(bf2f(b[0]), s_w1[k + 0], dot);
        dot = fmaf(bf2f(b[1]), s_w1[k + 1], dot);
        dot = fmaf(bf2f(b[2]), s_w1[k + 2], dot);
        dot = fmaf(bf2f(b[3]), s_w1[k + 3], dot);
        dot = fmaf(bf2f(b[4]), s_w1[k + 4], dot);
        dot = fmaf(bf2f(b[5]), s_w1[k + 5], dot);
        dot = fmaf(bf2f(b[6]), s_w1[k + 6], dot);
        dot = fmaf(bf2f(b[7]), s_w1[k + 7], dot);
    }

    float x = logits[((int64_t)m * 7 + step) * nv + v] + dot;
    if (!(isinf(x) && x < 0.0f)) {
        float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
        float t0 = ts[0], t1 = ts[1], t2 = ts[2], t3 = ts[3];
        float d0 = fmaxf(__fsub_rn(r0, x), 0.0f);
        float d1 = fmaxf(__fsub_rn(x, r1), 0.0f);
        float d2 = fmaxf(__fsub_rn(r2, x), 0.0f);
        float d3 = fmaxf(__fsub_rn(x, r3), 0.0f);
        float y = __fadd_rn(x, __fmul_rn(t0, d0));
        y = __fadd_rn(y, __fmul_rn(t1, d1));
        y = __fadd_rn(y, __fmul_rn(t2, __fmul_rn(d2, d2)));
        y = __fadd_rn(y, __fmul_rn(t3, __fmul_rn(d3, d3)));
        x = y;
    }
    scores[(int64_t)m * nv + v] = x;
}

// Launch: grid=(rows), block=(256)
extern "C" __global__ void tf_ernie55_argmax_pair16(
    const float* __restrict__ scores,  // [rows, n]
    uint32_t* __restrict__ out_pairs,  // [rows, 4] (16 bytes/row: [0]=val bits, [1]=id, [2]=0, [3]=0)
    int rows,
    int n,
    int id_offset
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = scores + (int64_t)r * n;
    float best_v = -1e38f;
    int best_i = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best_v, off);
        int oi = __shfl_down_sync(0xffffffffu, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) { s_val[warp] = best_v; s_idx[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_val[lane] : -1e38f;
        int idx = (lane < 8) ? s_idx[lane] : 0;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, off);
            int oi = __shfl_down_sync(0xffffffffu, idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) {
            uint32_t* dst = out_pairs + (int64_t)r * 4;
            dst[0] = __float_as_uint(v);
            dst[1] = (uint32_t)(idx + id_offset);
            dst[2] = 0u;
            dst[3] = 0u;
        }
    }
}

// Launch: grid=(M), block=(1)
extern "C" __global__ void tf_ernie55_pick_merge_batch(
    const uint8_t* __restrict__ picks,  // [world, M, 16]
    int world,
    int M,
    int32_t* __restrict__ walk,         // [M, 8]
    int step
) {
    int m = blockIdx.x;
    if (m >= M || threadIdx.x != 0) return;
    float best_val = -1e38f;
    int32_t best_id = 0;
    for (int q = 0; q < world; ++q) {
        const uint8_t* p = picks + ((int64_t)q * M + m) * 16;
        float v = *reinterpret_cast<const float*>(p);
        int32_t id = *reinterpret_cast<const int32_t*>(p + 4);
        if (q == 0 || v > best_val || (v == best_val && id < best_id)) {
            best_val = v;
            best_id = id;
        }
    }
    walk[m * 8 + step + 1] = best_id;
}


// --- Batch 3 decode & prefill kernels ---

// Dedicated 250 GB/s BF16 lm_head GEMV for decode rows <= 8 (4.3x faster than qmmf wide=1 BF16 on GB10!)
// Launch: grid=(ceil(N / 8)), block=(256)
template <int MAX_R>
__device__ __forceinline__ void lm_head_bf16_body(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out,
    int rows,
    int N,
    int H
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    if (n >= N) return;

    float acc[MAX_R];
#pragma unroll
    for (int r = 0; r < MAX_R; ++r) acc[r] = 0.0f;

    int n8 = H >> 3;
    const uint4* wp = reinterpret_cast<const uint4*>(w + (int64_t)n * H);
    for (int i = lane; i < n8; i += 32) {
        uint4 wv = __ldg(wp + i);
        const __nv_bfloat16* bw = reinterpret_cast<const __nv_bfloat16*>(&wv);
        float fw[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) fw[u] = bf2f(bw[u]);
#pragma unroll
        for (int r = 0; r < MAX_R; ++r) {
            if (r < rows) {
                uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)r * H) + i);
                const __nv_bfloat16* bx = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
                for (int u = 0; u < 8; ++u) acc[r] = fmaf(bf2f(bx[u]), fw[u], acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        if (r < rows) {
            float s = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
            if (lane == 0) out[(int64_t)r * N + n] = s;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_ernie55_lm_head_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out,
    int rows,
    int N,
    int H
) {
    if (rows <= 4) {
        lm_head_bf16_body<4>(x, w, out, rows, N, H);
    } else {
        lm_head_bf16_body<8>(x, w, out, rows, N, H);
    }
}

// BF16 VHA post-mix step 1 (local U^T @ o -> BF16 z [rows, 16, head_dim]): halves VHA AllReduce wire bytes!
// Launch: grid=(ceil(head_dim / 32), rows), block=(32, 16)
extern "C" __global__ void tf_ernie55_vha_u_proj_bf16(
    const __nv_bfloat16* __restrict__ o,  // [rows, H_loc, head_dim]
    const __nv_bfloat16* __restrict__ U,  // [H_loc, 16]
    __nv_bfloat16* __restrict__ z,        // [rows, 16, head_dim]
    int rows,
    int H_loc,
    int head_dim
) {
    int d = blockIdx.x * 32 + threadIdx.x;
    int k = threadIdx.y;  // 0..15
    int r = blockIdx.y;
    if (r >= rows || d >= head_dim || k >= 16) return;
    float sum = 0.0f;
    for (int h = 0; h < H_loc; ++h) {
        sum += bf2f(U[h * 16 + k]) * bf2f(o[((int64_t)r * H_loc + h) * head_dim + d]);
    }
    z[((int64_t)r * 16 + k) * head_dim + d] = f2bf(sum);
}

// BF16 VHA post-mix step 2: o[r, h, d] += sum_k V[h, k] * z_bf16[r, k, d]
// Launch: grid=(ceil(head_dim / 32), rows), block=(32, H_loc_chunk<=16)
extern "C" __global__ void tf_ernie55_vha_v_add_bf16(
    __nv_bfloat16* __restrict__ o,        // [rows, H_loc, head_dim]
    const __nv_bfloat16* __restrict__ V,  // [H_loc, 16]
    const __nv_bfloat16* __restrict__ z,  // [rows, 16, head_dim]
    int rows,
    int H_loc,
    int head_dim
) {
    int d = blockIdx.x * 32 + threadIdx.x;
    int r = blockIdx.y;
    if (r >= rows || d >= head_dim) return;
    for (int h = threadIdx.y; h < H_loc; h += blockDim.y) {
        float add = 0.0f;
        for (int k = 0; k < 16; ++k) {
            add += bf2f(V[h * 16 + k]) * bf2f(z[((int64_t)r * 16 + k) * head_dim + d]);
        }
        int64_t idx = ((int64_t)r * H_loc + h) * head_dim + d;
        o[idx] = f2bf(bf2f(o[idx]) + add);
    }
}

__device__ __forceinline__ void mhc_mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 16-row, 4-warp K-split BF16 Tensor-Core mHC pre-mix for prompt rows >= 64 (3x faster than scalar tiled!)
// Launch: grid=(ceil(rows / 16)), block=(128)
extern "C" __global__ void __launch_bounds__(128) tf_ernie55_mhc_pre_mma(
    const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ fn,
    float* __restrict__ mix_raw,
    float* __restrict__ sqrsum_out,
    int rows,
    int hc_dim
) {
    const int r0 = blockIdx.x * 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    constexpr int BK_ALL = 256;
    constexpr int LD_ALL = 264;
    __shared__ __align__(16) __nv_bfloat16 xs[16 * LD_ALL];
    __shared__ __align__(16) __nv_bfloat16 fs[24 * LD_ALL];
    __shared__ float s_out[4][16 * 24];
    __shared__ float s_sq[4][16];

    float acc[3][4];
#pragma unroll
    for (int j = 0; j < 3; ++j) for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    float sq_local = 0.0f;
    const int sq_r = lane & 15;
    const int sq_h = (lane >> 4) * 32;

    for (int k0 = 0; k0 < hc_dim; k0 += BK_ALL) {
        __syncthreads();
        for (int i = tid; i < 16 * 32; i += 128) {
            int rr = i >> 5, c = (i & 31) * 8, r = r0 + rr;
            uint4 v = (r < rows) ? *reinterpret_cast<const uint4*>(res + (int64_t)r * hc_dim + k0 + c) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(xs + rr * LD_ALL + c) = v;
        }
        for (int i = tid; i < 24 * 64; i += 128) {
            int jj = i >> 6, c = (i & 63) * 4;
            float4 fv = *reinterpret_cast<const float4*>(fn + (int64_t)jj * hc_dim + k0 + c);
            __nv_bfloat16* dst = fs + jj * LD_ALL + c;
            dst[0] = f2bf(fv.x); dst[1] = f2bf(fv.y); dst[2] = f2bf(fv.z); dst[3] = f2bf(fv.w);
        }
        __syncthreads();

        const int kw = warp * 64;
        if (r0 + sq_r < rows) {
            const __nv_bfloat16* xp = xs + sq_r * LD_ALL + kw + sq_h;
#pragma unroll
            for (int c = 0; c < 32; c += 8) {
                uint4 v = *reinterpret_cast<const uint4*>(xp + c);
                const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
                for (int u = 0; u < 8; ++u) { float f = bf2f(b[u]); sq_local += f * f; }
            }
        }
#pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
            uint32_t a[4];
            const __nv_bfloat16* ap = xs + g * LD_ALL + kw + kk + tq * 2;
            a[0] = *reinterpret_cast<const uint32_t*>(ap);
            a[1] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD_ALL);
            a[2] = *reinterpret_cast<const uint32_t*>(ap + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(ap + 8 * LD_ALL + 8);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                uint32_t b[2];
                const __nv_bfloat16* bp = fs + (j * 8 + g) * LD_ALL + kw + kk + tq * 2;
                b[0] = *reinterpret_cast<const uint32_t*>(bp);
                b[1] = *reinterpret_cast<const uint32_t*>(bp + 8);
                mhc_mma16816(acc[j], a, b);
            }
        }
    }
    sq_local += __shfl_xor_sync(0xffffffffu, sq_local, 16);
    if (lane < 16) s_sq[warp][lane] = sq_local;
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int rr = g + (i >= 2 ? 8 : 0);
            int cc = j * 8 + tq * 2 + (i & 1);
            s_out[warp][rr * 24 + cc] = acc[j][i];
        }
    __syncthreads();
    if (tid < 16 && r0 + tid < rows) {
        sqrsum_out[r0 + tid] = s_sq[0][tid] + s_sq[1][tid] + s_sq[2][tid] + s_sq[3][tid];
    }
    for (int idx = tid; idx < 16 * 24; idx += 128) {
        int rr = idx / 24, cc = idx - rr * 24;
        if (r0 + rr < rows) {
            mix_raw[(int64_t)(r0 + rr) * 24 + cc] = s_out[0][idx] + s_out[1][idx] + s_out[2][idx] + s_out[3][idx];
        }
    }
}


// --- SM121a Native FP8 Block-128 Prefill GEMM (82.5 TFLOPS on GB10, 2.6x faster than qmmf!) ---

// Quantize BF16 [M, ldx] (first K cols) -> E4M3 [M, K] + F32 scales [M, K/128]
// Launch: grid=(K/128, M), block=(32)
extern "C" __global__ void tf_ernie55_quant_bf16_to_fp8_b128(
    const __nv_bfloat16* __restrict__ x,
    int ldx,
    uint8_t* __restrict__ out_q,
    float* __restrict__ out_s,
    int M,
    int K
) {
    int kb = blockIdx.x;
    int m = blockIdx.y;
    int lane = threadIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)m * ldx + kb * 128 + lane * 4;
    uint2 raw = *reinterpret_cast<const uint2*>(xr);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float v0 = __bfloat162float(b[0]), v1 = __bfloat162float(b[1]);
    float v2 = __bfloat162float(b[2]), v3 = __bfloat162float(b[3]);
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float scale = fmaxf(amax, 1e-12f) * (1.0f / 448.0f);
    float inv = 1.0f / scale;
    if (lane == 0) out_s[(int64_t)m * (K >> 7) + kb] = scale;
    uint8_t q[4];
    q[0] = __nv_fp8_e4m3(v0 * inv).__x;
    q[1] = __nv_fp8_e4m3(v1 * inv).__x;
    q[2] = __nv_fp8_e4m3(v2 * inv).__x;
    q[3] = __nv_fp8_e4m3(v3 * inv).__x;
    *reinterpret_cast<uint32_t*>(out_q + (int64_t)m * K + kb * 128 + lane * 4) = *reinterpret_cast<uint32_t*>(q);
}

__device__ __forceinline__ void ernie_fp8_mma16832(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ernie_cp_async16(void* smem_ptr, const void* glob_ptr, bool pred) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    if (pred) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(glob_ptr));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" :: "r"(smem_addr), "l"(glob_ptr));
    }
}

__device__ __forceinline__ void ernie_ldm_x4(uint32_t* dst, const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

__device__ __forceinline__ void ernie_ldm_x2(uint32_t* dst, const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1]) : "r"(addr));
}

__device__ __forceinline__ int ernie_swz64(int row, int col16) {
    return (row << 6) | ((col16 ^ (row & 3)) << 4);
}

// Launch: grid=(ceil(N/128), ceil(M/64)), block=(256)
extern "C" __global__ void __launch_bounds__(256, 2) tf_ernie55_fp8_gemm_b128_mma(
    const uint8_t* __restrict__ Xq,
    const float* __restrict__ Sx,
    const uint8_t* __restrict__ Wq,
    const float* __restrict__ Sw,
    __nv_bfloat16* __restrict__ Y,
    int M,
    int N,
    int K
) {
    int num_m = (M + 63) >> 6;
    int num_n = (N + 127) >> 7;
    int bid = blockIdx.y * num_n + blockIdx.x;
    int grp = 8;
    int grp_tiles = grp * num_n;
    int g_id = bid / grp_tiles;
    int first_m = g_id * grp;
    int gm = (num_m - first_m < grp) ? (num_m - first_m) : grp;
    int in_g = bid - g_id * grp_tiles;
    int bm = first_m + (in_g % gm);
    int bn = in_g / gm;
    int n0 = bn * 128;
    int m0 = bm * 64;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wm = warp >> 2;
    int wn = warp & 3;
    int g = lane >> 2;
    int tq = lane & 3;

    __shared__ __align__(16) uint8_t sA[3][64 * 64];
    __shared__ __align__(16) uint8_t sB[3][128 * 64];

    float acc[2][4][4];
    float sub[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = sub[mi][ni][r] = 0.0f;

    int kb_cnt = K >> 7;
    int nb = n0 >> 7;
    int num_stages = K >> 6;

    auto load_stage = [&](int stg, int k0) {
        {
            int row = tid >> 2;
            int col16 = tid & 3;
            bool pa = (m0 + row < M);
            ernie_cp_async16(sA[stg] + ernie_swz64(row, col16), Xq + (int64_t)(pa ? (m0 + row) : 0) * K + k0 + (col16 << 4), pa);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 256;
            int row = idx >> 2;
            int col16 = idx & 3;
            bool pb = (n0 + row < N);
            ernie_cp_async16(sB[stg] + ernie_swz64(row, col16), Wq + (int64_t)(pb ? (n0 + row) : 0) * K + k0 + (col16 << 4), pb);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    if (num_stages > 0) load_stage(0, 0);
    if (num_stages > 1) load_stage(1, 64);

    int a_row_off = lane & 15;
    int a_col_off = (lane >> 4) & 1;
    int b_row_off = lane & 7;
    int b_col_off = (lane >> 3) & 1;

    for (int s = 0; s < num_stages; ++s) {
        int cur = s % 3;
        int pf = s + 2;
        if (pf < num_stages) {
            load_stage(pf % 3, pf * 64);
            asm volatile("cp.async.wait_group 2;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

#pragma unroll
        for (int kk16 = 0; kk16 < 4; kk16 += 2) {
            uint32_t a_reg[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int ar = wm * 32 + mi * 16 + a_row_off;
                ernie_ldm_x4(a_reg[mi], sA[cur] + ernie_swz64(ar, kk16 + a_col_off));
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int br = wn * 32 + ni * 8 + b_row_off;
                uint32_t b_reg[2];
                ernie_ldm_x2(b_reg, sB[cur] + ernie_swz64(br, kk16 + b_col_off));
                ernie_fp8_mma16832(sub[0][ni], a_reg[0], b_reg);
                ernie_fp8_mma16832(sub[1][ni], a_reg[1], b_reg);
            }
        }

        if (s & 1) {
            int kb = s >> 1;
            float sw = Sw[(int64_t)nb * kb_cnt + kb];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int mr0 = m0 + wm * 32 + mi * 16 + g;
                int mr1 = mr0 + 8;
                float s0 = (mr0 < M ? Sx[(int64_t)mr0 * kb_cnt + kb] : 0.0f) * sw;
                float s1 = (mr1 < M ? Sx[(int64_t)mr1 * kb_cnt + kb] : 0.0f) * sw;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    acc[mi][ni][0] = fmaf(sub[mi][ni][0], s0, acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(sub[mi][ni][1], s0, acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(sub[mi][ni][2], s1, acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(sub[mi][ni][3], s1, acc[mi][ni][3]);
                    sub[mi][ni][0] = sub[mi][ni][1] = sub[mi][ni][2] = sub[mi][ni][3] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        int mr0 = m0 + wm * 32 + mi * 16 + g;
        int mr1 = mr0 + 8;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int nc = n0 + wn * 32 + ni * 8 + tq * 2;
            if (mr0 < M && nc + 1 < N)
                *reinterpret_cast<__nv_bfloat162*>(Y + (int64_t)mr0 * N + nc) = __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
            if (mr1 < M && nc + 1 < N)
                *reinterpret_cast<__nv_bfloat162*>(Y + (int64_t)mr1 * N + nc) = __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}


// --- SM121a Native BF16 64x64 MMA Prefill GEMM (56.5 TFLOPS, 20.7x faster than qmmf BF16!) ---
__device__ __forceinline__ void ernie_bf16_mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ int ernie_swz32bf(int row, int col16) {
    return (row << 5) | ((col16 ^ (row & 3)) << 3);
}

template <bool OUT_F32>
__device__ __forceinline__ void ernie_bf16_gemm_body(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    void* __restrict__ Y,
    int M, int N, int K
) {
    int n0 = blockIdx.x * 64;
    int m0 = blockIdx.y * 64;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int wm = warp >> 1;
    int wn = warp & 1;
    int g = lane >> 2;
    int tq = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 sA[2][64 * 32];
    __shared__ __align__(16) __nv_bfloat16 sB[2][64 * 32];

    float acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.0f;

    int num_stages = K >> 5;
    auto load_stage = [&](int stg, int k0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 128;
            int row = idx >> 2;
            int col16 = idx & 3;
            bool pa = (m0 + row < M);
            bool pb = (n0 + row < N);
            ernie_cp_async16(sA[stg] + ernie_swz32bf(row, col16), X + (int64_t)(pa ? (m0 + row) : 0) * K + k0 + (col16 << 3), pa);
            ernie_cp_async16(sB[stg] + ernie_swz32bf(row, col16), W + (int64_t)(pb ? (n0 + row) : 0) * K + k0 + (col16 << 3), pb);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    load_stage(0, 0);

    int a_row_off = lane & 15;
    int a_col_off = (lane >> 4) & 1;
    int b_row_off = lane & 7;
    int b_col_off = (lane >> 3) & 1;

    for (int s = 0; s < num_stages; ++s) {
        int cur = s & 1;
        int nxt = cur ^ 1;
        if (s + 1 < num_stages) {
            load_stage(nxt, (s + 1) * 32);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

#pragma unroll
        for (int kk16 = 0; kk16 < 4; kk16 += 2) {
            uint32_t a_reg[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int ar = wm * 32 + mi * 16 + a_row_off;
                ernie_ldm_x4(a_reg[mi], sA[cur] + ernie_swz32bf(ar, kk16 + a_col_off));
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int br = wn * 32 + ni * 8 + b_row_off;
                uint32_t b_reg[2];
                ernie_ldm_x2(b_reg, sB[cur] + ernie_swz32bf(br, kk16 + b_col_off));
                ernie_bf16_mma16816(acc[0][ni], a_reg[0], b_reg);
                ernie_bf16_mma16816(acc[1][ni], a_reg[1], b_reg);
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        int mr0 = m0 + wm * 32 + mi * 16 + g;
        int mr1 = mr0 + 8;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int nc = n0 + wn * 32 + ni * 8 + tq * 2;
            if (OUT_F32) {
                float* Yf = reinterpret_cast<float*>(Y);
                if (mr0 < M && nc + 1 < N) {
                    Yf[(int64_t)mr0 * N + nc] = acc[mi][ni][0];
                    Yf[(int64_t)mr0 * N + nc + 1] = acc[mi][ni][1];
                }
                if (mr1 < M && nc + 1 < N) {
                    Yf[(int64_t)mr1 * N + nc] = acc[mi][ni][2];
                    Yf[(int64_t)mr1 * N + nc + 1] = acc[mi][ni][3];
                }
            } else {
                __nv_bfloat16* Yb = reinterpret_cast<__nv_bfloat16*>(Y);
                if (mr0 < M && nc + 1 < N)
                    *reinterpret_cast<__nv_bfloat162*>(Yb + (int64_t)mr0 * N + nc) = __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
                if (mr1 < M && nc + 1 < N)
                    *reinterpret_cast<__nv_bfloat162*>(Yb + (int64_t)mr1 * N + nc) = __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) tf_ernie55_bf16_gemm_bf16_mma(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Y, int M, int N, int K) {
    ernie_bf16_gemm_body<false>(X, W, Y, M, N, K);
}

extern "C" __global__ void __launch_bounds__(128, 4) tf_ernie55_bf16_gemm_f32_mma(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ Y, int M, int N, int K) {
    ernie_bf16_gemm_body<true>(X, W, Y, M, N, K);
}

struct TFSegEntry {
    uint64_t stash_ptr;
    int32_t row0;
    int32_t n;
    int32_t max_pos;
    int32_t older_ceil;
    int32_t sparse;
    int32_t pad;
};

// Copy per-row DSpark taps to each segment's stash buffer.
// Launch: grid=(rows, 4), block=(256)
extern "C" __global__ void tf_ernie55_stash_taps_tab(
    const __nv_bfloat16* __restrict__ taps,
    const int32_t* __restrict__ row_seg,
    const TFSegEntry* __restrict__ seg_tab,
    int rows,
    int tap_elems
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int seg = row_seg[r];
    if (seg < 0) return;
    uint64_t dst_u64 = seg_tab[seg].stash_ptr;
    if (dst_u64 == 0) return;
    int local_r = r - seg_tab[seg].row0;
    const uint4* src = reinterpret_cast<const uint4*>(taps + (int64_t)r * tap_elems);
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(dst_u64) + (int64_t)local_r * tap_elems);
    int vecs = tap_elems >> 3;
    int chunk = (vecs + gridDim.y - 1) / gridDim.y;
    int i0 = blockIdx.y * chunk;
    int i1 = (i0 + chunk < vecs) ? (i0 + chunk) : vecs;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) dst[i] = src[i];
}
