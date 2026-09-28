#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define WARP_SIZE 32
#define LOG_2PI   1.8378770664093453f

// ─────────────────────────────────────────
// warp reduce
// ─────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Reduce, then broadcast lane 0's result back to the whole warp so every lane
// can maintain the same online-softmax state independently.
__device__ __forceinline__ float warp_reduce_sum_bcast(float val) {
    val = warp_reduce_sum(val);
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float warp_reduce_max_bcast(float val) {
    val = warp_reduce_max(val);
    return __shfl_sync(0xffffffff, val, 0);
}

// ─────────────────────────────────────────
// Main kernel (tiled / online-softmax version)
//
// Each warp handles one pixel and streams over the Gaussian dimension in tiles
// of WARP_SIZE, using a running max / running sum for an online-softmax
// reduction (the same recurrence as the online softmax in FlashAttention).
// Per-thread register usage is O(1) and does not grow with N, so there is no
// compile-time MAX_N limit and no register-array spill when N gets large.
// ─────────────────────────────────────────
template <int BLOCK_B, int BLOCK_N_THREADS>
__global__ void glut_forward_kernel(
    const float* __restrict__ rgb,
    const float* __restrict__ positions,
    const float* __restrict__ prec,
    const float* __restrict__ log_det,
    const float* __restrict__ opacities,
    const float* __restrict__ color_mat,
    const float* __restrict__ color_bias,
    const float* __restrict__ global_mat,
    const float* __restrict__ global_bias,
    float*       __restrict__ output,
    int B, int N,
    bool residual,
    bool weight_norm
) {
    static_assert(BLOCK_N_THREADS == WARP_SIZE, "tiled streaming kernel requires one warp per pixel");

    // weight_norm == true  : partition-of-unity blending (original behaviour).
    //   lw = log[ N(x; mu, Sigma) * opacity ] and the accumulator is divided by
    //   the weight sum, so the output is a convex combination of the local
    //   transforms (renormalisation makes |Sigma| cancel out).
    // weight_norm == false : opacity is each Gaussian's absolute weight.
    //   The pdf is replaced by the normalisation-constant-free kernel
    //   k = exp(-1/2 * d_maha^2) in (0, 1] (no log_det / (2*pi)^{3/2} term) and
    //   the accumulator is NOT renormalised; instead the online-softmax max
    //   shift is undone (multiply by exp(m)) to recover the true sum
    //   sum_n k_n * opacity_n * transform_n.  log_det is unused in this mode.

    int pixel_id = blockIdx.x * BLOCK_B + threadIdx.y;  // y: pixel
    int lane_n   = threadIdx.x;                         // x: gaussian lane within warp

    if (pixel_id >= B) return;

    float r = rgb[pixel_id * 3 + 0];
    float g = rgb[pixel_id * 3 + 1];
    float b = rgb[pixel_id * 3 + 2];

    // Online-softmax state: after all tiles are consumed this is the final reduction.
    float m     = -1e30f;   // running max of log-weight
    float l     = 0.f;      // running sum of exp(log_weight - m)
    float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f;  // running sum of weight * color

    const int num_tiles = (N + BLOCK_N_THREADS - 1) / BLOCK_N_THREADS;

    for (int t = 0; t < num_tiles; t++) {
        int n = lane_n + t * BLOCK_N_THREADS;

        float lw = -1e30f, out_r_n = 0.f, out_g_n = 0.f, out_b_n = 0.f;

        if (n < N) {
            float dr = r - positions[n*3+0];
            float dg = g - positions[n*3+1];
            float db = b - positions[n*3+2];

            float p00 = prec[n*9+0], p01 = prec[n*9+1], p02 = prec[n*9+2];
            float p11 = prec[n*9+4], p12 = prec[n*9+5];
            float p22 = prec[n*9+8];

            float mahal = dr*(p00*dr + p01*dg + p02*db)
                        + dg*(p01*dr + p11*dg + p12*db)
                        + db*(p02*dr + p12*dg + p22*db);

            lw = weight_norm
                 ? (-0.5f * (mahal + log_det[n] + 3.f * LOG_2PI)
                    + __logf(opacities[n] + 1e-8f))
                 : (-0.5f * mahal + __logf(opacities[n] + 1e-8f));

            float m0=color_mat[n*9+0], m1=color_mat[n*9+1], m2=color_mat[n*9+2];
            float m3=color_mat[n*9+3], m4=color_mat[n*9+4], m5=color_mat[n*9+5];
            float m6=color_mat[n*9+6], m7=color_mat[n*9+7], m8=color_mat[n*9+8];

            out_r_n = m0*r + m1*g + m2*b + color_bias[n*3+0];
            out_g_n = m3*r + m4*g + m5*b + color_bias[n*3+1];
            out_b_n = m6*r + m7*g + m8*b + color_bias[n*3+2];
        }

        // This tile's max log-weight, merged with the historical running max.
        float tile_max = warp_reduce_max_bcast(lw);
        float new_m    = fmaxf(m, tile_max);
        float alpha    = __expf(m - new_m);   // rescale factor for the running accumulators

        float w = (n < N) ? __expf(lw - new_m) : 0.f;

        float tile_l = warp_reduce_sum_bcast(w);
        float tile_r = warp_reduce_sum_bcast(w * out_r_n);
        float tile_g = warp_reduce_sum_bcast(w * out_g_n);
        float tile_b = warp_reduce_sum_bcast(w * out_b_n);

        l     = l     * alpha + tile_l;
        acc_r = acc_r * alpha + tile_r;
        acc_g = acc_g * alpha + tile_g;
        acc_b = acc_b * alpha + tile_b;
        m     = new_m;
    }

    // weight_norm : divide by the weight sum (partition of unity).
    // else        : undo the online-softmax max shift so the accumulator holds
    //               the true (un-normalised) sum  sum_n exp(lw_n) * transform_n.
    //               __expf(m) with m <= 0 underflows gracefully to 0 far from
    //               every Gaussian, which is the desired "no local correction".
    float scale = weight_norm ? (1.f / (l + 1e-8f)) : __expf(m);
    float out_r = acc_r * scale;
    float out_g = acc_g * scale;
    float out_b = acc_b * scale;

    // ── write-back ──
    if (lane_n == 0) {
        float gr = global_mat[0]*r + global_mat[1]*g + global_mat[2]*b + global_bias[0];
        float gg = global_mat[3]*r + global_mat[4]*g + global_mat[5]*b + global_bias[1];
        float gb = global_mat[6]*r + global_mat[7]*g + global_mat[8]*b + global_bias[2];

        if (residual) {
            out_r += gr;
            out_g += gg;
            out_b += gb;
        }

        output[pixel_id*3+0] = fminf(fmaxf(out_r, 0.f), 1.f);
        output[pixel_id*3+1] = fminf(fmaxf(out_g, 0.f), 1.f);
        output[pixel_id*3+2] = fminf(fmaxf(out_b, 0.f), 1.f);
    }
}


torch::Tensor glut_forward_cuda(
    torch::Tensor rgb,
    torch::Tensor positions,
    torch::Tensor prec,
    torch::Tensor log_det,
    torch::Tensor opacities,
    torch::Tensor color_mat,
    torch::Tensor color_bias,
    torch::Tensor global_mat,
    torch::Tensor global_bias,
    bool residual,
    bool weight_norm
) {
    int B = rgb.size(0);
    int N = positions.size(0);

    TORCH_CHECK(N > 0, "N must be positive");

    auto output = torch::empty({B, 3}, rgb.options());

    constexpr int BLOCK_B = 32;
    constexpr int BLOCK_N_THREADS = 32;

    dim3 block(BLOCK_N_THREADS, BLOCK_B);  // x = gaussian lane, y = pixel
    dim3 grid((B + BLOCK_B - 1) / BLOCK_B);

    glut_forward_kernel<BLOCK_B, BLOCK_N_THREADS><<<
        grid, block, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        rgb.data_ptr<float>(),
        positions.data_ptr<float>(),
        prec.data_ptr<float>(),
        log_det.data_ptr<float>(),
        opacities.data_ptr<float>(),
        color_mat.data_ptr<float>(),
        color_bias.data_ptr<float>(),
        global_mat.data_ptr<float>(),
        global_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, N, residual, weight_norm
    );

    return output;
}
