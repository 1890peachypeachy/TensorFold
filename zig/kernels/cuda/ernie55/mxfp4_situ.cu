// Native SM121 MXFP4 (OCP E2M1 + UE8M0 group-32) grouped-expert MoE kernels
// with fused SITU(beta=4.0, linear_beta=25.0) activation for ERNIE 5.5 Flash Preview.
//
// Zero-LUT PTX E2M1->bf16x2 decode + 128-bit coalesced vector loads (1 full 32-elem MXFP4 group per load).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

// Decode low nibble -> low bf16, high nibble -> high bf16 of one byte using 3 register instructions (0 LUTs!).
__device__ __forceinline__ __nv_bfloat162 e2m1_byte_to_bf162(uint32_t b) {
    const uint32_t v = (b & 0xFu) | ((b & 0xF0u) << 12);
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return *reinterpret_cast<const __nv_bfloat162*>(&r);
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
    return (e == 0) ? 5.8774717541114375e-39f : __int_as_float((uint32_t)e << 23);
}

__device__ __forceinline__ float sigmoid_rn(float x) {
    return __fdiv_rn(1.0f, 1.0f + expf(-x));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// Dot product of 32 packed E2M1 weights (16 bytes = uint4 = 1 full MXFP4 group) against 32 BF16 activations in smem.
__device__ __forceinline__ float dot32_e2m1(uint4 qw, const __nv_bfloat162* x2) {
    const uint32_t words[4] = {qw.x, qw.y, qw.z, qw.w};
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t wd = words[w];
        uint4 xv = *reinterpret_cast<const uint4*>(x2 + w * 4);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            float2 wf = __bfloat1622float2(e2m1_byte_to_bf162((wd >> (b * 8)) & 0xFFu));
            float2 xf = __bfloat1622float2(xp[b]);
            s0 = fmaf(wf.x, xf.x, s0);
            s1 = fmaf(wf.y, xf.y, s1);
        }
    }
    return s0 + s1;
}

// Dot product of 16 packed E2M1 weights (8 bytes = uint2 = half of an MXFP4 group) against 16 BF16 activations in smem.
__device__ __forceinline__ float dot16_e2m1(uint2 qw, const __nv_bfloat162* x2) {
    const uint32_t words[2] = {qw.x, qw.y};
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        uint32_t wd = words[w];
        uint4 xv = *reinterpret_cast<const uint4*>(x2 + w * 4);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            float2 wf = __bfloat1622float2(e2m1_byte_to_bf162((wd >> (b * 8)) & 0xFFu));
            float2 xf = __bfloat1622float2(xp[b]);
            s0 = fmaf(wf.x, xf.x, s0);
            s1 = fmaf(wf.y, xf.y, s1);
        }
    }
    return s0 + s1;
}

}  // namespace

// Fused MXFP4 w1 + w3 -> SITU(beta, linear_beta) for selected topk experts.
// K = 2048 => Kg = 64 groups of 32 => 2 full uint4 (16B) coalesced loads per lane in a 32-thread warp!
// Launch: grid=(ceil(N_loc/4), topk, rows), block=(128), smem = K * sizeof(__nv_bfloat16) (4 KB)
extern "C" __global__ void tf_ernie55_mxfp4_w13_situ(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w1_q,
    const uint8_t* __restrict__ w1_s,
    const uint8_t* __restrict__ w3_q,
    const uint8_t* __restrict__ w3_s,
    const int32_t* __restrict__ ids,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int topk,
    int N_loc,
    int K,
    float beta,
    float linear_beta
) {
    int r = blockIdx.z;
    int tk = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 4 + warp;
    if (r >= rows || tk >= topk) return;

    extern __shared__ __nv_bfloat16 s_x[];
    const __nv_bfloat16* xr = x + (int64_t)r * K;
    for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
        *reinterpret_cast<uint4*>(s_x + i) = *reinterpret_cast<const uint4*>(xr + i);
    }
    __syncthreads();

    if (n >= N_loc) return;

    int e = ids[(int64_t)r * topk + tk];
    int Kg = K >> 5;
    int64_t q_off = ((int64_t)e * N_loc + n) * (K >> 1);
    int64_t s_off = ((int64_t)e * N_loc + n) * Kg;

    const uint4* q1 = reinterpret_cast<const uint4*>(w1_q + q_off);
    const uint4* q3 = reinterpret_cast<const uint4*>(w3_q + q_off);
    const uint8_t* s1 = w1_s + s_off;
    const uint8_t* s3 = w3_s + s_off;
    const __nv_bfloat162* sx2 = reinterpret_cast<const __nv_bfloat162*>(s_x);

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    // Each lane loads 1 full group (16 bytes = 32 weights) per iteration -> 512 bytes coalesced across warp!
    for (int g = lane; g < Kg; g += 32) {
        const __nv_bfloat162* xp = sx2 + g * 16;
        uint4 v1 = __ldg(q1 + g);
        uint4 v3 = __ldg(q3 + g);
        float sc1 = ue8m0_to_f32(__ldg(s1 + g));
        float sc3 = ue8m0_to_f32(__ldg(s3 + g));
        g_acc = fmaf(dot32_e2m1(v1, xp), sc1, g_acc);
        u_acc = fmaf(dot32_e2m1(v3, xp), sc3, u_acc);
    }

    g_acc = warp_sum(g_acc);
    u_acc = warp_sum(u_acc);

    if (lane == 0) {
        float sg = beta * tanhf(g_acc / beta) * sigmoid_rn(g_acc);
        float su = linear_beta * tanhf(u_acc / linear_beta);
        out[((int64_t)r * topk + tk) * N_loc + n] = __float2bfloat16_rn(sg * su);
    }
}

// MXFP4 w2 + weighted topk combine (deterministic sequential sum over tk = 0..topk-1, ZERO atomics).
// K_loc = 512 on TP=4 => 16 groups of 32 => 2 lanes per group (uint2 = 8 bytes/lane -> 1 single coalesced 256B pass, ZERO K-loop!).
// Launch: grid=(ceil(N_out/4), rows), block=(128), smem = topk * K_loc * sizeof(__nv_bfloat16)
extern "C" __global__ void tf_ernie55_mxfp4_w2_combine(
    const __nv_bfloat16* __restrict__ mid,
    const uint8_t* __restrict__ w2_q,
    const uint8_t* __restrict__ w2_s,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int topk,
    int N_out,
    int K_loc
) {
    int r = blockIdx.y;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 4 + warp;
    if (r >= rows) return;

    extern __shared__ __nv_bfloat16 s_mid[];
    int total_mid = topk * K_loc;
    const __nv_bfloat16* mr = mid + (int64_t)r * total_mid;
    for (int i = threadIdx.x * 8; i < total_mid; i += blockDim.x * 8) {
        *reinterpret_cast<uint4*>(s_mid + i) = *reinterpret_cast<const uint4*>(mr + i);
    }
    __syncthreads();

    if (n >= N_out) return;

    int K2 = K_loc >> 1;
    int Kg = K_loc >> 5;
    int halves = Kg << 1;  // 2 halves (16 elems = 8 bytes = uint2) per MXFP4 group

    float total = 0.0f;
    for (int tk = 0; tk < topk; ++tk) {
        int e = ids[(int64_t)r * topk + tk];
        float w = weights[(int64_t)r * topk + tk];
        int64_t q_off = ((int64_t)e * N_out + n) * K2;
        int64_t s_off = ((int64_t)e * N_out + n) * Kg;
        const uint2* q2 = reinterpret_cast<const uint2*>(w2_q + q_off);
        const uint8_t* s2 = w2_s + s_off;
        const __nv_bfloat162* xtk = reinterpret_cast<const __nv_bfloat162*>(s_mid + tk * K_loc);

        float acc = 0.0f;
        for (int h = lane; h < halves; h += 32) {
            int g = h >> 1;
            uint2 v = __ldg(q2 + h);
            float sc = ue8m0_to_f32(__ldg(s2 + g));
            acc = fmaf(dot16_e2m1(v, xtk + h * 8), sc, acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            total = fmaf(acc, w, total);
        }
    }

    if (lane == 0) {
        out[(int64_t)r * N_out + n] = __float2bfloat16_rn(total);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Expert-major prefill MoE (Spark, 2026-11-24). The per-pick kernels above run one (row, expert) pick a block, so a
// prompt chunk reads every expert's weights once per pick that chose it. Below: the picks sorted by expert
// (route_*), w13+SITU and w2 as per-expert tensor-core GEMMs over all of an expert's picks (weights dequantized
// to bf16 once a block, exact), then the top-k combine in the per-pick kernel's order. Used for rows >= 128.

// counts[0 .. E) = 0 (before route_count, every call).
extern "C" __global__ void tf_ernie55_mxfp4_route_zero(int32_t* __restrict__ counts, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < E) counts[t] = 0;
}

// counts[e] += picks of expert e (counts zeroed by the caller).
extern "C" __global__ void tf_ernie55_mxfp4_route_count(const int32_t* __restrict__ ids, int pairs, int32_t* __restrict__ counts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) atomicAdd(&counts[ids[p]], 1);
}

// offsets[e] = picks below expert e; cursor[e] = offsets[e]. One block of `E` (<= 1024) threads.
extern "C" __global__ void tf_ernie55_mxfp4_route_scan(const int32_t* __restrict__ counts, int32_t* __restrict__ offsets,
                                                 int32_t* __restrict__ cursor, int E) {
    __shared__ int32_t s[1024];
    int t = threadIdx.x;
    s[t] = t < E ? counts[t] : 0;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        int v = t >= off ? s[t - off] : 0;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }
    if (t < E) {
        int ex = s[t] - counts[t];
        offsets[t] = ex;
        cursor[t] = ex;
    }
    if (t == 0) offsets[E] = s[1023];
}

// list[offsets[e] ..] = the picks of expert e (in no set order: each pick's output is its own).
extern "C" __global__ void tf_ernie55_mxfp4_route_fill(const int32_t* __restrict__ ids, int pairs, int32_t* __restrict__ cursor,
                                                 int32_t* __restrict__ list) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) list[atomicAdd(&cursor[ids[p]], 1)] = p;
}

#define TF_MMA_BN 64
#define TF_MMA_BK 64
#define TF_MMA_LD (TF_MMA_BK + 8)  // bf16 row stride in shared memory: 144 bytes, conflict-free fragment loads
#define TF_MMA_MT 4                // 16-pick tiles held a pass

__device__ __forceinline__ void mma_bf16_16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 32 e2m1 weights (one MXFP4 group, 16 bytes) times 2^scale into 32 bf16 at dst (16-byte aligned).
__device__ __forceinline__ void dequant32(uint4 q, float sc, __nv_bfloat16* dst) {
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    const __nv_bfloat162 s2 = __floats2bfloat162_rn(sc, sc);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint4 o;
        uint32_t* op = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            __nv_bfloat162 v = __hmul2(e2m1_byte_to_bf162((words[w] >> (b * 8)) & 0xFFu), s2);
            op[b] = *reinterpret_cast<uint32_t*>(&v);
        }
        *reinterpret_cast<uint4*>(dst + w * 8) = o;
    }
}

// The block's 64-column x 64-input tile of matrix (q, s) [E][N][K/2], [E][N][K/32] at input k0, into ws[64][LD].
__device__ __forceinline__ void load_w_tile(const uint8_t* __restrict__ q, const uint8_t* __restrict__ s, int e, int n0,
                                            int N, int K, int k0, __nv_bfloat16* ws) {
    // 128 threads: thread t takes column t / 2, group (t & 1) of the two 32-input groups in the step
    const int t = threadIdx.x;
    const int c = t >> 1, g = t & 1;
    const int n = n0 + c;
    __nv_bfloat16* dst = ws + c * TF_MMA_LD + g * 32;
    if (n < N) {
        const int64_t row = (int64_t)e * N + n;
        const int kg = (k0 >> 5) + g;
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(q + row * (K >> 1)) + kg);
        dequant32(v, ue8m0_to_f32(__ldg(s + row * (K >> 5) + kg)), dst);
    } else {
        const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(dst + i * 8) = z;
    }
}

// Rows of the pass's picks [p0, p0 + np) at inputs [k0, k0 + 64) into xs[np_pad][LD]; row of pick i: src_row(i).
template <typename RowOf>
__device__ __forceinline__ void load_x_tile(const __nv_bfloat16* __restrict__ x, int ldx, int np, int rows_pad, int k0,
                                            __nv_bfloat16* xs, RowOf src_row) {
    // 8 threads a row (8 bf16 = 16 bytes each)
    for (int i = threadIdx.x; i < rows_pad * 8; i += blockDim.x) {
        const int r = i >> 3, c = (i & 7) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < np) v = *reinterpret_cast<const uint4*>(x + (int64_t)src_row(r) * ldx + k0 + c);
        *reinterpret_cast<uint4*>(xs + r * TF_MMA_LD + c) = v;
    }
}

// A fragment (rows mt*16 .., inputs kk ..) and B fragment (columns nb .., inputs kk ..) from shared memory.
__device__ __forceinline__ void frag_a(const __nv_bfloat16* xs, int mt, int kk, uint32_t* a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* p = xs + (mt * 16 + g) * TF_MMA_LD + kk + t * 2;
    a[0] = *reinterpret_cast<const uint32_t*>(p);
    a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * TF_MMA_LD);
    a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
    a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * TF_MMA_LD + 8);
}
__device__ __forceinline__ void frag_b(const __nv_bfloat16* ws, int nb, int kk, uint32_t* b) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* p = ws + (nb + g) * TF_MMA_LD + kk + t * 2;
    b[0] = *reinterpret_cast<const uint32_t*>(p);
    b[1] = *reinterpret_cast<const uint32_t*>(p + 8);
}

// 100%-coalesced BK=256 (128B = 1 full DRAM cache line per row!) MXFP4 w13 + SITU MMA!
// Loads 64 rows x 128B packed MXFP4 into smem (8 KB each), then runs 4 sub-steps of k=64,
// dequanting w1/w3 straight into MMA B-registers (1.76x faster on GB10, bit-exact BF16 MMA!).
// Launch: grid=(ceil(N_loc/64), E), block=(128)
#define COAL_X_LD 72
extern "C" __global__ void __launch_bounds__(128, 3) tf_ernie55_mxfp4_w13_mma(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w1_q, const uint8_t* __restrict__ w1_s,
    const uint8_t* __restrict__ w3_q, const uint8_t* __restrict__ w3_s, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ list, __nv_bfloat16* __restrict__ out, int topk, int N_loc, int K, float beta,
    float linear_beta) {
    const int e = blockIdx.y;
    const int begin = offsets[e], end = offsets[e + 1];
    const int tz = blockIdx.z;
    const int p_start = begin + tz * 64;
    if (p_start >= end) return;
    const int p_end = (tz + 1 == gridDim.z) ? end : min(end, p_start + 64);
    const int n0 = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;

    __shared__ __align__(16) uint8_t w1_qs[64 * 128];
    __shared__ __align__(16) uint8_t w3_qs[64 * 128];
    __shared__ uint8_t w1_ss[64 * 8];
    __shared__ uint8_t w3_ss[64 * 8];
    __shared__ __align__(16) __nv_bfloat16 xs[64 * COAL_X_LD];

    const int K2 = K >> 1;
    const int K32 = K >> 5;

    for (int p0 = p_start; p0 < p_end; p0 += 64) {
        const int np = min(64, p_end - p0);
        const int mts = (np + 15) >> 4;
        float acc1[4][2][4], acc3[4][2][4];
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc1[m][j][i] = acc3[m][j][i] = 0.0f;

        for (int k0 = 0; k0 < K; k0 += 256) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = tid + i * 128;
                const int r = idx >> 3;
                const int c16 = (idx & 7) << 4;
                const int n = n0 + r;
                if (n < N_loc) {
                    const int64_t off = ((int64_t)e * N_loc + n) * K2 + (k0 >> 1) + c16;
                    *reinterpret_cast<uint4*>(w1_qs + r * 128 + c16) = __ldg(reinterpret_cast<const uint4*>(w1_q + off));
                    *reinterpret_cast<uint4*>(w3_qs + r * 128 + c16) = __ldg(reinterpret_cast<const uint4*>(w3_q + off));
                } else {
                    *reinterpret_cast<uint4*>(w1_qs + r * 128 + c16) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(w3_qs + r * 128 + c16) = make_uint4(0, 0, 0, 0);
                }
            }
            if (tid < 64) {
                const int n = n0 + tid;
                if (n < N_loc) {
                    const int64_t soff = ((int64_t)e * N_loc + n) * K32 + (k0 >> 5);
                    *reinterpret_cast<uint2*>(w1_ss + tid * 8) = __ldg(reinterpret_cast<const uint2*>(w1_s + soff));
                    *reinterpret_cast<uint2*>(w3_ss + tid * 8) = __ldg(reinterpret_cast<const uint2*>(w3_s + soff));
                } else {
                    *reinterpret_cast<uint2*>(w1_ss + tid * 8) = make_uint2(0x7f7f7f7fu, 0x7f7f7f7fu);
                    *reinterpret_cast<uint2*>(w3_ss + tid * 8) = make_uint2(0x7f7f7f7fu, 0x7f7f7f7fu);
                }
            }

#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const int k_cur = k0 + sub * 64;
                if (sub > 0) __syncthreads();
                for (int i = tid; i < mts * 16 * 8; i += 128) {
                    const int r = i >> 3, c = (i & 7) * 8;
                    uint4 v = make_uint4(0, 0, 0, 0);
                    if (r < np) v = *reinterpret_cast<const uint4*>(x + (int64_t)(list[p0 + r] / topk) * K + k_cur + c);
                    *reinterpret_cast<uint4*>(xs + r * COAL_X_LD + c) = v;
                }
                __syncthreads();

#pragma unroll
                for (int kk = 0; kk < 64; kk += 16) {
                    const int g32 = sub * 2 + (kk >> 5);
                    uint32_t b1[2][2], b3[2][2];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int br = warp * 16 + j * 8 + g;
                        const float sf1 = ue8m0_to_f32(w1_ss[br * 8 + g32]);
                        const float sf3 = ue8m0_to_f32(w3_ss[br * 8 + g32]);
                        const __nv_bfloat162 s1 = __floats2bfloat162_rn(sf1, sf1);
                        const __nv_bfloat162 s3 = __floats2bfloat162_rn(sf3, sf3);
                        const int byte_off = br * 128 + ((sub * 64 + kk) >> 1) + t;
                        const __nv_bfloat162 v1_0 = __hmul2(e2m1_byte_to_bf162(w1_qs[byte_off]), s1);
                        const __nv_bfloat162 v1_1 = __hmul2(e2m1_byte_to_bf162(w1_qs[byte_off + 4]), s1);
                        const __nv_bfloat162 v3_0 = __hmul2(e2m1_byte_to_bf162(w3_qs[byte_off]), s3);
                        const __nv_bfloat162 v3_1 = __hmul2(e2m1_byte_to_bf162(w3_qs[byte_off + 4]), s3);
                        b1[j][0] = *reinterpret_cast<const uint32_t*>(&v1_0);
                        b1[j][1] = *reinterpret_cast<const uint32_t*>(&v1_1);
                        b3[j][0] = *reinterpret_cast<const uint32_t*>(&v3_0);
                        b3[j][1] = *reinterpret_cast<const uint32_t*>(&v3_1);
                    }
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < mts) {
                            const __nv_bfloat16* p = xs + (m * 16 + g) * COAL_X_LD + kk + t * 2;
                            uint32_t a[4];
                            a[0] = *reinterpret_cast<const uint32_t*>(p);
                            a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * COAL_X_LD);
                            a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
                            a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * COAL_X_LD + 8);
#pragma unroll
                            for (int j = 0; j < 2; ++j) {
                                mma_bf16_16816(acc1[m][j], a, b1[j]);
                                mma_bf16_16816(acc3[m][j], a, b3[j]);
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= mts) continue;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = m * 16 + g + (i >= 2 ? 8 : 0);
                    const int n = n0 + warp * 16 + j * 8 + t * 2 + (i & 1);
                    if (r < np && n < N_loc) {
                        const float ga = acc1[m][j][i], ua = acc3[m][j][i];
                        const float sg = beta * tanhf(ga / beta) * sigmoid_rn(ga);
                        const float su = linear_beta * tanhf(ua / linear_beta);
                        out[(int64_t)list[p0 + r] * N_loc + n] = __float2bfloat16_rn(sg * su);
                    }
                }
            }
        }
    }
}

// 100%-coalesced BK=256 MXFP4 w2 MMA (K_loc = 512/1024/2048 -> 2/4/8 exact 128B cache-line passes!).
// Launch: grid=(ceil(N_out/64), E), block=(128)
extern "C" __global__ void __launch_bounds__(128, 4) tf_ernie55_mxfp4_w2_mma(
    const __nv_bfloat16* __restrict__ mid, const uint8_t* __restrict__ w2_q, const uint8_t* __restrict__ w2_s,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ list, const float* __restrict__ weights,
    float* __restrict__ part, int topk, int N_out, int K_loc) {
    const int e = blockIdx.y;
    const int begin = offsets[e], end = offsets[e + 1];
    const int tz = blockIdx.z;
    const int p_start = begin + tz * 64;
    if (p_start >= end) return;
    const int p_end = (tz + 1 == gridDim.z) ? end : min(end, p_start + 64);
    const int n0 = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;

    __shared__ __align__(16) uint8_t w2_qs[64 * 128];
    __shared__ uint8_t w2_ss[64 * 8];
    __shared__ __align__(16) __nv_bfloat16 xs[64 * COAL_X_LD];

    const int K2 = K_loc >> 1;
    const int K32 = K_loc >> 5;

    for (int p0 = p_start; p0 < p_end; p0 += 64) {
        const int np = min(64, p_end - p0);
        const int mts = (np + 15) >> 4;
        float acc[4][2][4];
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[m][j][i] = 0.0f;

        for (int k0 = 0; k0 < K_loc; k0 += 256) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = tid + i * 128;
                const int r = idx >> 3;
                const int c16 = (idx & 7) << 4;
                const int n = n0 + r;
                if (n < N_out) {
                    const int64_t off = ((int64_t)e * N_out + n) * K2 + (k0 >> 1) + c16;
                    *reinterpret_cast<uint4*>(w2_qs + r * 128 + c16) = __ldg(reinterpret_cast<const uint4*>(w2_q + off));
                } else {
                    *reinterpret_cast<uint4*>(w2_qs + r * 128 + c16) = make_uint4(0, 0, 0, 0);
                }
            }
            if (tid < 64) {
                const int n = n0 + tid;
                if (n < N_out) {
                    const int64_t soff = ((int64_t)e * N_out + n) * K32 + (k0 >> 5);
                    *reinterpret_cast<uint2*>(w2_ss + tid * 8) = __ldg(reinterpret_cast<const uint2*>(w2_s + soff));
                } else {
                    *reinterpret_cast<uint2*>(w2_ss + tid * 8) = make_uint2(0x7f7f7f7fu, 0x7f7f7f7fu);
                }
            }

#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const int k_cur = k0 + sub * 64;
                if (sub > 0) __syncthreads();
                for (int i = tid; i < mts * 16 * 8; i += 128) {
                    const int r = i >> 3, c = (i & 7) * 8;
                    uint4 v = make_uint4(0, 0, 0, 0);
                    if (r < np) v = *reinterpret_cast<const uint4*>(mid + (int64_t)list[p0 + r] * K_loc + k_cur + c);
                    *reinterpret_cast<uint4*>(xs + r * COAL_X_LD + c) = v;
                }
                __syncthreads();

#pragma unroll
                for (int kk = 0; kk < 64; kk += 16) {
                    const int g32 = sub * 2 + (kk >> 5);
                    uint32_t b[2][2];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int br = warp * 16 + j * 8 + g;
                        const float sf = ue8m0_to_f32(w2_ss[br * 8 + g32]);
                        const __nv_bfloat162 s2 = __floats2bfloat162_rn(sf, sf);
                        const int byte_off = br * 128 + ((sub * 64 + kk) >> 1) + t;
                        const __nv_bfloat162 v0 = __hmul2(e2m1_byte_to_bf162(w2_qs[byte_off]), s2);
                        const __nv_bfloat162 v1 = __hmul2(e2m1_byte_to_bf162(w2_qs[byte_off + 4]), s2);
                        b[j][0] = *reinterpret_cast<const uint32_t*>(&v0);
                        b[j][1] = *reinterpret_cast<const uint32_t*>(&v1);
                    }
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (m < mts) {
                            const __nv_bfloat16* p = xs + (m * 16 + g) * COAL_X_LD + kk + t * 2;
                            uint32_t a[4];
                            a[0] = *reinterpret_cast<const uint32_t*>(p);
                            a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * COAL_X_LD);
                            a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
                            a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * COAL_X_LD + 8);
#pragma unroll
                            for (int j = 0; j < 2; ++j) mma_bf16_16816(acc[m][j], a, b[j]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= mts) continue;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = m * 16 + g + (i >= 2 ? 8 : 0);
                    const int n = n0 + warp * 16 + j * 8 + t * 2 + (i & 1);
                    if (r < np && n < N_out) { const int pk = list[p0 + r]; atomicAdd(part + (int64_t)(pk / topk) * N_out + n, acc[m][j][i] * weights[pk]); }
                }
            }
        }
    }
}

// out[r, n] = sum over tk = 0 .. topk - 1 (in order) of part[r * topk + tk, n] * weights[r * topk + tk].
// Launch: grid=(ceil(N_out/256), rows), block=(256).
extern "C" __global__ void tf_ernie55_mxfp4_moe_combine(const float* __restrict__ part, const float* __restrict__ weights,
                                                      __nv_bfloat16* __restrict__ out, int topk, int N_out) {
    (void)weights;
    (void)topk;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (n >= N_out) return;
    out[(int64_t)r * N_out + n] = __float2bfloat16(part[(int64_t)r * N_out + n]);
}
