import torch
import glut_cuda   # compiled extension


class GaussianLUTCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rgb, positions, prec, log_det, opacities,
                color_mat, color_bias, global_mat, global_bias,
                residual, weight_norm):
        def prep(t):
            return t.contiguous().float()

        out = glut_cuda.forward(
            prep(rgb),
            prep(positions),
            prep(prec.reshape(-1, 9)),
            prep(log_det),
            prep(opacities),
            prep(color_mat.contiguous().reshape(-1, 9)),
            prep(color_bias),
            prep(global_mat.contiguous().reshape(9)),
            prep(global_bias),
            residual,
            weight_norm,
        )
        ctx.save_for_backward(rgb, positions, prec, log_det,
                              opacities, color_mat, color_bias)
        ctx.residual = residual
        ctx.weight_norm = weight_norm
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Inference-only kernel: training always uses the PyTorch autograd path.
        raise NotImplementedError("Use PyTorch autograd for training")


def glut_forward(rgb, positions, prec, log_det, opacities,
                 color_mat, color_bias, global_mat, global_bias,
                 residual, weight_norm=True):
    """Fused Gaussian-LUT forward pass (CUDA, inference only).

    Args:
        weight_norm: True  -> partition-of-unity blending (weights renormalised
                              to sum to 1); ``log_det`` is used.
                     False -> opacity is each Gaussian's absolute weight and the
                              pdf is the constant-free kernel exp(-1/2*d_maha^2);
                              ``log_det`` is ignored.  Matches
                              ``GLUT3D.forward`` with ``weight_norm=False``.
    """
    if rgb.is_cuda:
        return GaussianLUTCUDAFunction.apply(
            rgb, positions, prec, log_det, opacities,
            color_mat, color_bias, global_mat, global_bias,
            residual, weight_norm,
        )
    else:
        # CPU fallback: use the PyTorch implementation instead.
        raise RuntimeError("CUDA kernel requires GPU input")
