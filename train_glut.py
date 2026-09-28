"""
GLUT - fit a single 3D colour LUT with a small continuous Gaussian model.

A LUT (given as an input/target hald-image pair) is approximated by a mixture of
3D Gaussians in the RGB cube.  Each Gaussian owns a centre, a full covariance
(parameterised by its Cholesky factor), an opacity and a local affine colour
transform; an input colour is transformed by blending the local transforms with
opacity-weighted Gaussian responses, optionally on top of a global affine
transform.

Key options:

    * Data loading: the whole hald point cloud is copied to the GPU once and
      mini-batches are drawn by ``torch.randperm`` indexing.  This removes the
      per-step DataLoader / worker / host->device overhead (~2x faster training
      loop for this tiny model).
    * Large-batch compensation (``--ref_batch_size`` / ``--no_batch_compensation``):
      when ``batch_size`` exceeds the reference batch size, epochs are scaled up
      to keep the optimiser-step budget constant and the lr is scaled by
      ``sqrt(batch/ref)``.
    * ``--cuda_eval``: route the evaluation forward pass through the fused CUDA
      kernel ``glut_cuda.glut.glut_forward`` (inference only; requires a GPU and
      the compiled extension).  Numerically equivalent up to float32 rounding.

The CUDA kernel is inference only; back-propagation always uses PyTorch autograd.
"""

import argparse
import gc
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict

from data.hald_dataloader import SingleHaldBatch
from utils.utils import setup_logging, random_everything, initialize_gaussians_grid
from utils.losses import LABspaceLoss
from utils.metrics import calc_deltaE
from utils.hard_samp_mining import CurriculumHardMining

# ---------------------------------------------------------------------------
# Optional CUDA kernel.  Import lazily so the script still runs on CPU-only
# machines or when the extension has not been compiled.
# ---------------------------------------------------------------------------
try:
    from glut_cuda.glut import glut_forward

    _CUDA_KERNEL_AVAILABLE = True
    _CUDA_IMPORT_ERROR = None
except Exception as _cuda_import_error:  # noqa: BLE001 - any import failure disables the kernel
    glut_forward = None
    _CUDA_KERNEL_AVAILABLE = False
    _CUDA_IMPORT_ERROR = _cuda_import_error


class GLUT3D(nn.Module):
    """Continuous 3D color transform represented as a mixture of Gaussians.

    Each Gaussian owns a center in the RGB cube, a full 3x3 covariance
    (parameterised through its Cholesky factor), an opacity/blend weight and a
    local affine color transform ``(matrix, bias)``.  For an input color the
    Gaussians are blended by their (opacity-weighted) pdf value, and the blended
    local transform is optionally added on top of a global affine transform.
    """

    def __init__(self,
                 num_gaussians: int = 32,
                 init_scale: float = 0.15,
                 residual: bool = True,
                 weight_norm: bool = True,
                 epsilon: float = 1e-6,
                 logger: logging.Logger = None):
        """
        Args:
            num_gaussians: number of Gaussians in the mixture.
            init_scale:    initial standard deviation of every Gaussian.
            residual:      if True the local transform is added to a global
                           affine transform, otherwise it is used directly.
            weight_norm:   how the per-Gaussian blend weights are formed.
                           True  (default) - partition-of-unity blending: the
                                 weights are the opacity-weighted Gaussian pdf
                                 renormalised to sum to 1, so the output is a
                                 convex combination of the local transforms.
                           False - no renormalisation: the opacity directly
                                 controls each Gaussian's absolute weight and the
                                 pdf is replaced by the normalisation-constant-
                                 free kernel exp(-1/2 * d_maha^2) in (0, 1], so
                                 weight magnitude is decoupled from |Sigma|.
                                 In uncovered regions the local term vanishes,
                                 hence ``residual`` is forced ON as a fallback.
            epsilon:       small constant for numerical stability.
            logger:        optional logger for model info.
        """
        super().__init__()

        self.logger = logger
        self.num_gaussians = num_gaussians
        self.residual = residual
        self.weight_norm = weight_norm
        self.epsilon = epsilon

        # Without renormalisation the Gaussians only add a local correction that
        # decays to zero far from every center; a global affine fallback is
        # required, so force the residual connection on in that mode.
        if not self.weight_norm and not self.residual:
            self.residual = True
            if self.logger:
                self.logger.warning(
                    "weight_norm=False requires a global fallback: residual forced ON.")

        # 1. Gaussian centers - inside the RGB cube [0, 1]^3.
        self.positions = nn.Parameter(initialize_gaussians_grid(num_gaussians))

        # 2. Covariance via its Cholesky factor L (6 free parameters per Gaussian).
        #    Diagonal is stored in log space and passed through softplus to stay positive.
        self.cholesky_diag = nn.Parameter(
            torch.full((num_gaussians, 3), math.log(init_scale))
        )
        #    Off-diagonal: off[:, 0] = L[1, 0], off[:, 1] = L[2, 0], off[:, 2] = L[2, 1].
        self.cholesky_off = nn.Parameter(torch.zeros(num_gaussians, 3))

        # 3. Opacity / blend weight (logit, passed through sigmoid).
        self.opacities_logit = nn.Parameter(torch.ones(num_gaussians, 1))

        # 4. Per-Gaussian local affine color transform.
        self.color_matrices = nn.Parameter(
            torch.eye(3).unsqueeze(0).repeat(num_gaussians, 1, 1) * 0.1
        )
        self.color_biases = nn.Parameter(torch.zeros(num_gaussians, 3))

        # 5. Global affine color transform.
        self.global_matrix = nn.Parameter(torch.eye(3))
        self.global_bias = nn.Parameter(torch.zeros(3))

        # Activations.
        self.softplus = nn.Softplus()
        self.opacity_activation = nn.Sigmoid()

        # Precomputed constants.
        self.register_buffer('log_2pi', torch.tensor(math.log(2 * math.pi)))
        self.register_buffer('eye_3x3', torch.eye(3))

        if self.logger:
            self.logger.info("Initialize GLUT3D:")
            self.logger.info(f"  - Number of Gaussians: {num_gaussians}")
            self.logger.info(f"  - Initial scale      : {init_scale}")
            self.logger.info(f"  - Residual           : {self.residual}")
            self.logger.info(f"  - Weight norm        : {self.weight_norm}")
            self.logger.info(f"  - Total parameters   : {self.count_parameters():,}")
            self.logger.info("  - Model device       : {}".format(
                "GPU" if torch.cuda.is_available() else "CPU"))

    # ------------------------------------------------------------------
    # Covariance / precision helpers
    # ------------------------------------------------------------------
    def build_cholesky_matrix(self) -> torch.Tensor:
        """Build the lower-triangular Cholesky factor L from the 6 parameters.

        Returns:
            L: [N, 3, 3] lower-triangular matrices with positive diagonal.
        """
        diag = self.softplus(self.cholesky_diag)  # [N, 3], > 0

        L = torch.zeros(self.num_gaussians, 3, 3, device=self.cholesky_diag.device)
        L[:, 0, 0] = diag[:, 0]
        L[:, 1, 1] = diag[:, 1]
        L[:, 2, 2] = diag[:, 2]
        L[:, 1, 0] = self.cholesky_off[:, 0]
        L[:, 2, 0] = self.cholesky_off[:, 1]
        L[:, 2, 1] = self.cholesky_off[:, 2]
        return L

    def get_covariance_matrix(self) -> torch.Tensor:
        """Return the positive-definite covariance matrices Sigma = L L^T. [N, 3, 3]."""
        L = self.build_cholesky_matrix()
        return torch.bmm(L, L.transpose(1, 2))

    def get_precision_matrix(self, cov: torch.Tensor) -> torch.Tensor:
        """Return the precision matrices (inverse covariance). [N, 3, 3]."""
        cov_regularized = cov + self.epsilon * self.eye_3x3.unsqueeze(0)
        try:
            precision = torch.inverse(cov_regularized)
        except Exception:
            precision = torch.pinverse(cov_regularized)
        return precision

    # ------------------------------------------------------------------
    # Gaussian pdf
    # ------------------------------------------------------------------
    def compute_mahalanobis_distance(self,
                                     rgb: torch.Tensor,
                                     positions: torch.Tensor,
                                     precision: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distance (x - mu)^T Sigma^-1 (x - mu). Returns [B, N]."""
        batch_size = rgb.shape[0]

        if positions.dim() == 2:
            positions = positions.unsqueeze(0)

        diff = rgb.unsqueeze(1) - positions        # [B, N, 3]
        diff = diff.unsqueeze(-1)                   # [B, N, 3, 1]
        diff_T = diff.transpose(2, 3)              # [B, N, 1, 3]

        if precision.dim() == 3:
            precision = precision.unsqueeze(0).expand(batch_size, -1, -1, -1)

        intermediate = torch.matmul(diff_T, precision)
        distance_sq = torch.matmul(intermediate, diff)
        return distance_sq.squeeze(-1).squeeze(-1)

    def compute_gaussian_pdf(self,
                             rgb: torch.Tensor,
                             positions: torch.Tensor,
                             covariance: torch.Tensor,
                             precision: torch.Tensor,
                             normalized: bool = True) -> torch.Tensor:
        """Evaluate every Gaussian at every input color. Returns [B, N].

        Args:
            normalized: if True return the true Gaussian pdf (with the
                        1/((2*pi)^{3/2} |Sigma|^{1/2}) normalisation constant);
                        if False return the constant-free kernel
                        exp(-1/2 * d_maha^2) in (0, 1], whose magnitude does not
                        depend on |Sigma| (used when ``weight_norm`` is False).
        """
        mahalanobis_sq = self.compute_mahalanobis_distance(rgb, positions, precision)

        if not normalized:
            return torch.exp(-0.5 * mahalanobis_sq)

        det_cov = torch.det(covariance)                     # [N]
        log_det = torch.log(det_cov + self.epsilon)         # [N]

        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(0)
        log_pdf = -0.5 * (mahalanobis_sq + log_det + 3 * self.log_2pi)
        return torch.exp(log_pdf)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def forward(self,
                rgb: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """Dense forward pass - blends every Gaussian. Pure PyTorch, differentiable."""
        if rgb.dim() == 1:
            rgb = rgb.unsqueeze(0)
        batch_size = rgb.shape[0]

        # 1. Covariance / precision.
        covariance = self.get_covariance_matrix()          # [N, 3, 3]
        precision = self.get_precision_matrix(covariance)  # [N, 3, 3]

        # 2. Blend weights from the (opacity-weighted) Gaussian response.
        opacities = self.opacity_activation(self.opacities_logit)          # [N, 1]
        if self.weight_norm:
            # Partition-of-unity: renormalise so the weights sum to 1.
            influences = self.compute_gaussian_pdf(rgb, self.positions, covariance, precision)
            weighted_influences = influences * opacities.squeeze(1)        # [B, N]
            weights = weighted_influences / (
                weighted_influences.sum(dim=1, keepdim=True) + self.epsilon)
        else:
            # No renormalisation: opacity is the absolute per-Gaussian weight.
            influences = self.compute_gaussian_pdf(
                rgb, self.positions, covariance, precision, normalized=False)
            weights = influences * opacities.squeeze(1)                    # [B, N]

        # 3. Global affine transform.
        rgb_global = torch.einsum('ij,bj->bi', self.global_matrix, rgb) + self.global_bias

        # 4. Per-Gaussian local affine transform.
        rgb_expanded = rgb.unsqueeze(1).expand(-1, self.num_gaussians, -1)
        gauss_transforms = torch.einsum('nij,bnj->bni', self.color_matrices, rgb_expanded)
        gauss_transforms = gauss_transforms + self.color_biases.unsqueeze(0)

        # 5. Weighted blend of the local transforms.
        local_transform = torch.einsum('bn,bni->bi', weights, gauss_transforms)

        # 6. Combine global + local.
        if self.residual:
            transformed_rgb = rgb_global + local_transform
        else:
            transformed_rgb = local_transform
        transformed_rgb = torch.clamp(transformed_rgb, 0, 1)

        # 7. Optionally return intermediate tensors (used by visualisation tools).
        if return_components:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            return {
                'output': transformed_rgb,
                'weights': weights,
                'influences': influences,
                'gauss_transforms': gauss_transforms,
                'rgb_global': rgb_global,
                'covariance': covariance,
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'opacities': opacities,
                'cholesky_diag': self.softplus(self.cholesky_diag),
                'cholesky_off': self.cholesky_off,
            }

        return transformed_rgb.squeeze() if batch_size == 1 else transformed_rgb

    @torch.no_grad()
    def forward_cuda(self, rgb: torch.Tensor) -> torch.Tensor:
        """Inference-only forward pass through the fused CUDA kernel.

        Produces the same result as :meth:`forward` (up to float32 rounding) for
        both blending modes.  The kernel takes the per-Gaussian precision
        matrices, ``log|Sigma|`` and opacities and honours ``weight_norm``
        (``log|Sigma|`` is only used when ``weight_norm`` is True).
        """
        if not _CUDA_KERNEL_AVAILABLE:
            raise RuntimeError("CUDA kernel not available: {}".format(_CUDA_IMPORT_ERROR))
        if not rgb.is_cuda:
            raise RuntimeError("CUDA kernel requires a GPU input tensor")

        if rgb.dim() == 1:
            rgb = rgb.unsqueeze(0)
        batch_size = rgb.shape[0]

        covariance = self.get_covariance_matrix()                   # [N, 3, 3]
        precision = self.get_precision_matrix(covariance)           # [N, 3, 3]
        log_det = torch.log(torch.det(covariance) + self.epsilon)   # [N]
        opacities = self.opacity_activation(self.opacities_logit).squeeze(1)  # [N]

        out = glut_forward(
            rgb,
            self.positions, precision, log_det, opacities,
            self.color_matrices, self.color_biases,
            self.global_matrix, self.global_bias,
            self.residual,
            self.weight_norm,
        )
        return out.squeeze() if batch_size == 1 else out

    def forward_sparse(self,
                       rgb: torch.Tensor,
                       top_k: int = 16,
                       return_components: bool = False) -> torch.Tensor:
        """Sparse forward pass - only the ``top_k`` nearest Gaussians are blended."""
        if rgb.dim() == 1:
            rgb = rgb.unsqueeze(0)
        batch_size = rgb.shape[0]

        # Step 1: cheap L2 screening to pick candidate Gaussians (no gradient).
        with torch.no_grad():
            l2_dist = torch.norm(rgb.unsqueeze(1) - self.positions.unsqueeze(0), dim=2)  # [B, N]
            _, top_k_indices = torch.topk(
                -l2_dist, k=min(top_k, self.num_gaussians), dim=1)                        # [B, K]
            selected_positions = self.positions[top_k_indices]                            # [B, K, 3]

        # Step 2: full Gaussian pdf, but only for the selected Gaussians.
        covariance = self.get_covariance_matrix()          # [N, 3, 3]
        precision = self.get_precision_matrix(covariance)  # [N, 3, 3]
        selected_precision = precision[top_k_indices]       # [B, K, 3, 3]
        selected_covariance = covariance[top_k_indices]     # [B, K, 3, 3]

        selected_influences = self.compute_gaussian_pdf(
            rgb, selected_positions, selected_covariance, selected_precision,
            normalized=self.weight_norm)

        selected_opacities = self.opacity_activation(self.opacities_logit)[top_k_indices]  # [B, K, 1]
        selected_opacities = selected_opacities.squeeze(-1)                                 # [B, K]

        # Step 3: form the blend weights over the selected Gaussians.
        selected_weighted = selected_influences * selected_opacities                        # [B, K]
        if self.weight_norm:
            norm_factor = selected_weighted.sum(dim=1, keepdim=True) + self.epsilon         # [B, 1]
            selected_weights = selected_weighted / norm_factor                              # [B, K]
        else:
            selected_weights = selected_weighted                                            # [B, K]

        # Step 4: global affine transform.
        rgb_global = torch.einsum('ij,bj->bi', self.global_matrix, rgb) + self.global_bias

        # Step 5: local affine transform for the selected Gaussians.
        selected_color_matrices = self.color_matrices[top_k_indices]  # [B, K, 3, 3]
        selected_color_biases = self.color_biases[top_k_indices]      # [B, K, 3]
        rgb_expanded = rgb.unsqueeze(1).expand(-1, top_k, -1)         # [B, K, 3]
        selected_transforms = torch.einsum(
            'bkij,bkj->bki', selected_color_matrices, rgb_expanded)
        selected_transforms = selected_transforms + selected_color_biases

        # Step 6: weighted blend.
        local_transform = torch.sum(
            selected_transforms * selected_weights.unsqueeze(-1), dim=1)  # [B, 3]

        # Step 7: combine global + local.
        if self.residual:
            transformed_rgb = rgb_global + local_transform
        else:
            transformed_rgb = local_transform
        transformed_rgb = torch.clamp(transformed_rgb, 0, 1)

        if return_components:
            return {
                'output': transformed_rgb,
                'sparse_weights': selected_weights,
                'top_k_indices': top_k_indices,
                'selected_influences': selected_influences,
                'rgb_global': rgb_global,
            }

        return transformed_rgb.squeeze() if batch_size == 1 else transformed_rgb

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_gaussian_info(self) -> Dict[str, np.ndarray]:
        """Return numpy snapshots of every Gaussian's geometry / parameters."""
        covariance = self.get_covariance_matrix()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        stds = torch.sqrt(eigenvalues)
        volumes = (4 * math.pi / 3) * torch.prod(stds, dim=1)

        return {
            'positions': self.positions.detach().cpu().numpy(),
            'covariance': covariance.detach().cpu().numpy(),
            'eigenvalues': eigenvalues.detach().cpu().numpy(),
            'eigenvectors': eigenvectors.detach().cpu().numpy(),
            'stds': stds.detach().cpu().numpy(),
            'volumes': volumes.detach().cpu().numpy(),
            'opacities': self.opacity_activation(self.opacities_logit).detach().cpu().numpy(),
            'color_matrices': self.color_matrices.detach().cpu().numpy(),
            'color_biases': self.color_biases.detach().cpu().numpy(),
            'cholesky_diag': self.softplus(self.cholesky_diag).detach().cpu().numpy(),
            'cholesky_off': self.cholesky_off.detach().cpu().numpy(),
        }

    def count_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class GLUTTrainer:
    """Thin training / evaluation wrapper around :class:`GLUT3D`."""

    def __init__(self, model: GLUT3D):
        self.model = model
        self.lab_loss = LABspaceLoss(losstype='hc')

    def train_step(self, epoch,
                   rgb_input, rgb_target,
                   lambda_sparse: float = 0.001,
                   lambda_lab: float = 1.0):
        """Single optimisation step (forward + loss + backward).

        Args:
            lambda_sparse: weight of the opacity-sparsity (entropy) regulariser.
            lambda_lab:    weight of the LAB-space perceptual loss.

        Returns:
            dict of scalar loss values.
        """
        output = self.model(rgb_input)

        # Main loss - L1 works well for color mapping.
        main_loss = F.l1_loss(output, rgb_target)
        lab_loss = self.lab_loss(output, rgb_target)

        # Sparsity regulariser: push opacities towards 0 or 1 (low entropy).
        sparse_loss = 0.0
        if lambda_sparse > 0:
            opacities = self.model.opacity_activation(self.model.opacities_logit)
            entropy = -opacities * torch.log(opacities + 1e-8) - \
                (1 - opacities) * torch.log(1 - opacities + 1e-8)
            sparse_loss = torch.mean(entropy)

        total_loss = main_loss + lambda_sparse * sparse_loss + lambda_lab * lab_loss

        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'sparse_loss': sparse_loss.item() if lambda_sparse > 0 else 0.0,
            'lab_loss': lab_loss.item() if lambda_lab > 0 else 0.0,
        }

    def evaluate(self, rgb_input, rgb_target,
                 use_sparse: bool = False, topk: int = 16, use_cuda: bool = False):
        """Evaluate the model on one batch and return error metrics.

        Args:
            use_sparse: use the sparse (top-k) forward pass.
            topk:       number of Gaussians for the sparse forward pass.
            use_cuda:   use the fused CUDA kernel for the dense forward pass
                        (ignored when ``use_sparse`` is True or the input is not
                        on the GPU).
        """
        self.model.eval()

        with torch.no_grad():
            if use_sparse:
                output = self.model.forward_sparse(rgb_input, topk)
            elif use_cuda and rgb_input.is_cuda:
                output = self.model.forward_cuda(rgb_input)
            else:
                output = self.model(rgb_input)

            l1_error = F.l1_loss(output, rgb_target).item()
            mse = F.mse_loss(output, rgb_target).item()

            out_np = output.cpu().numpy()
            tgt_np = rgb_target.cpu().numpy()
            deltaE00 = calc_deltaE(out_np, tgt_np)
            deltaE76 = calc_deltaE(out_np, tgt_np, method='CIE 1976')
            max_error = np.max(calc_deltaE(out_np, tgt_np, reduction='none'))

            return {
                'output': output,
                'l1_error': l1_error,
                'l2_error': mse,
                'deltaE00': deltaE00,
                'deltaE76': deltaE76,
                'max_error': max_error,
            }

    def save_checkpoint(self, path: str, epoch: int):
        """Save a minimal checkpoint (weights + reconstruction hyper-params)."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'num_gaussians': self.model.num_gaussians,
            'residual': self.model.residual,
            'weight_norm': self.model.weight_norm,
        }
        torch.save(checkpoint, path)


def materialize_dataset(dataset, device):
    """Move an entire hald dataset onto ``device`` as two dense tensors.

    A hald dataset is a fixed point cloud (even a level-16 hald is only a few
    hundred MB), so it can stay resident on the GPU for the whole run.
    ``SingleHaldBatch`` already stores the pixels as contiguous ``[M, 3]``
    tensors, so this is a single host->device copy - no per-sample
    ``__getitem__`` loop and no DataLoader workers.

    Returns:
        (inputs, targets): two ``[M, 3]`` float tensors on ``device``.
    """
    inputs = dataset.intensities.to(device).contiguous().float()
    targets = dataset.outputs.to(device).contiguous().float()
    return inputs, targets


def main(args):
    """Full training run."""
    print("=" * 60)
    print("Continuous 3D Color Transform with Gaussian Model")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(
        log_dir=args.save_dir,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True,
    )

    logger.info("Training Configuration:")
    logger.info(args.__dict__)

    # Resolve the evaluation backend.
    use_cuda_eval = args.cuda_eval
    if use_cuda_eval and not _CUDA_KERNEL_AVAILABLE:
        logger.warning("--cuda_eval requested but the CUDA kernel could not be "
                       "imported ({}); falling back to the PyTorch forward pass."
                       .format(_CUDA_IMPORT_ERROR))
        use_cuda_eval = False
    if use_cuda_eval and device.type != "cuda":
        logger.warning("--cuda_eval requested but no GPU is available; "
                       "falling back to the PyTorch forward pass.")
        use_cuda_eval = False
    logger.info("Evaluation backend: {}".format("CUDA kernel" if use_cuda_eval else "PyTorch"))

    # 1. Datasets.
    logger.info("1. Creating color transformation dataset...")
    logger.info("Input hald image:{}".format(args.train_input))
    logger.info("Target hald image:{}".format(args.train_target))

    train_dataset = SingleHaldBatch(args.train_input, args.train_target)
    val_dataset = SingleHaldBatch(args.test_input, args.test_target)

    # Keep the whole point cloud resident on the GPU and sample mini-batches with
    # torch.randperm indexing.  This removes the per-step DataLoader iterator /
    # worker IPC / host->device copy, which dominates wall time for this tiny
    # model.  (It also changes the shuffling RNG, so exact bit-equivalence with
    # the DataLoader-based train_glut.py no longer holds - metrics stay within
    # run-to-run noise.)
    train_inputs, train_targets = materialize_dataset(train_dataset, device)
    val_inputs, val_targets = materialize_dataset(val_dataset, device)
    n_train = train_inputs.shape[0]
    n_val = val_inputs.shape[0]

    logger.info("Data resident on {}. Train points: {}, Val points: {}, LUT difference: {}".format(
        device, n_train, n_val, val_dataset.error))

    # 2. Model.
    logger.info("2. Initializing continuous Gaussian LUT model...")
    model = GLUT3D(
        num_gaussians=args.num_gaussians,
        init_scale=args.init_scale,
        residual=args.residual,
        weight_norm=True,
        logger=logger,
    ).to(device)

    trainer = GLUTTrainer(model)

    # 3. Large-batch compensation (relative to --ref_batch_size).
    #    num_epochs / learning_rate are assumed to be tuned at ref_batch_size.
    #    When batch_size > ref_batch_size and compensation is enabled:
    #      * epochs are scaled by batch/ref, so the total number of optimiser
    #        steps matches the reference budget;
    #      * lr is scaled by sqrt(batch/ref) (the sqrt rule works well for Adam).
    #    batch_size <= ref_batch_size leaves everything untouched.
    scale = args.batch_size / args.ref_batch_size
    if (not args.no_batch_compensation) and scale > 1.0:
        num_epochs_eff = max(1, int(round(args.num_epochs * scale)))
        lr_eff = args.learning_rate * math.sqrt(scale)
        logger.info("[Batch compensation] batch={} vs ref={} (x{:g}): "
                    "epochs {} -> {}, lr {:.2e} -> {:.2e}".format(
                        args.batch_size, args.ref_batch_size, scale,
                        args.num_epochs, num_epochs_eff, args.learning_rate, lr_eff))
    else:
        num_epochs_eff = args.num_epochs
        lr_eff = args.learning_rate

    steps_per_epoch = math.ceil(n_train / args.batch_size)
    total_steps = num_epochs_eff * steps_per_epoch

    # 4. Optimizer / scheduler / curriculum hard-example mining.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_eff)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    curriculum_miner = CurriculumHardMining(
        total_epochs=num_epochs_eff,
        warmup_epochs=3,
        mining_start_epoch=5,
        initial_mining_ratio=0.1,
        final_mining_ratio=0.4,
    )

    # 5. Training loop.
    logger.info("3. Training Gaussian color transform model...")
    logger.info("   steps/epoch={}, total_steps={}".format(steps_per_epoch, total_steps))
    logger.info("-" * 60)

    train_losses = []
    iteration = 0

    for epoch in range(num_epochs_eff):
        model.train()
        # Shuffle by indexing the GPU-resident tensors (replaces DataLoader shuffle).
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, args.batch_size):
            iteration += 1
            idx = perm[start:start + args.batch_size]
            batch_input = train_inputs[idx]
            batch_target = train_targets[idx]

            optimizer.zero_grad(set_to_none=True)

            batch_input, batch_target, mining_stats = curriculum_miner(
                epoch, model, batch_input, batch_target)

            losses = trainer.train_step(
                epoch, batch_input, batch_target,
                lambda_sparse=args.lambda_sparse,
                lambda_lab=args.lambda_lab,
            )
            # Gradient clipping to avoid blow-ups.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(losses['total_loss'])

            if (iteration + 1) % 100 == 0 or iteration == 0:
                logger.info(f"Epoch [{epoch + 1:2d}] | "
                            f"Iter [{iteration + 1:3d}] | "
                            f"lr {scheduler.get_last_lr()[0]:.6f} | "
                            f"Total {losses['total_loss']:.6f} | "
                            f"l1 {losses['main_loss']:.6f} | "
                            f"lab {losses['lab_loss']:.6f} | "
                            f"sparse {losses['sparse_loss']:.6f}")

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == num_epochs_eff:
            logger.info("4. Evaluating model performance...")
            logger.info("-" * 60)
            l1_error_total, max_error_total, mse_total = 0.0, 0.0, 0.0
            deltaE00_total, deltaE76_total = 0.0, 0.0
            num_batches = 0

            for start in range(0, n_val, args.batch_size_val):
                val_rgb_input = val_inputs[start:start + args.batch_size_val]
                val_rgb_target = val_targets[start:start + args.batch_size_val]
                eval_metrics = trainer.evaluate(val_rgb_input, val_rgb_target,
                                                use_cuda=use_cuda_eval)
                l1_error_total += eval_metrics['l1_error']
                max_error_total = np.maximum(max_error_total, eval_metrics['max_error'])
                mse_total += eval_metrics['l2_error']
                deltaE00_total += eval_metrics['deltaE00']
                deltaE76_total += eval_metrics['deltaE76']
                num_batches += 1

            average_l1_error = l1_error_total / num_batches
            average_mse = mse_total / num_batches
            average_deltaE00 = deltaE00_total / num_batches
            average_deltaE76 = deltaE76_total / num_batches
            average_psnr = 10 * math.log10(1.0 ** 2 / average_mse) if average_mse > 0 else float('inf')

            logger.info('--- Validation after Epoch: {} | l1_error {:.6f} | psnr {:.2f} | '
                        'dE00 {:.4f} | dE76 {:.4f} | max_error {:.6f}'.format(
                            epoch + 1, average_l1_error, average_psnr,
                            average_deltaE00, average_deltaE76, max_error_total))

            gauss_info = model.get_gaussian_info()
            logger.info("Gaussian statistics:")
            logger.info(f"  - Mean position: {gauss_info['positions'].mean(axis=0).round(3)}")
            logger.info("  - Active gaussians (opacity > 0.1): "
                        f"{(gauss_info['opacities'] > 0.1).sum()}/{args.num_gaussians}")

            histogram, _ = np.histogram(gauss_info['opacities'], bins=10, range=(0, 1))
            logger.info(f"  - Opacity histogram: {histogram.tolist()}")

            torch.cuda.empty_cache()
            gc.collect()

    lut_name = os.path.splitext(os.path.basename(args.train_target))[0]

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)

    trainer.save_checkpoint(
        os.path.join(args.save_dir,
                     f'GLUT_{lut_name}_{args.num_gaussians}_psnr{average_psnr:.2f}'
                     f'_dE{average_deltaE76:.4f}.pth'),
        num_epochs_eff)
    logger.info(f"Final model saved to {lut_name}")

    return model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GLUT Training')
    parser.add_argument("--train_input", help="Input RGB map as a hald image",
                        default="./dataset/hald_images/7luts_train/Original_Image.png", type=str)
    parser.add_argument("--train_target", help="Enhanced RGB map as a hald image, after using the desired 3D LUT",
                        default="./dataset/hald_images/7luts_train/LUT01_After_Effects_LUTs_Contrast.png", type=str)
    parser.add_argument("--test_input", help="Input RGB map as a hald image",
                        default="./dataset/hald_images/7luts_test/Original_Image.png", type=str)
    parser.add_argument("--test_target", help="Enhanced RGB map as a hald image, after using the desired 3D LUT",
                        default="./dataset/hald_images/7luts_test/LUT01_After_Effects_LUTs_Contrast.png", type=str)
    parser.add_argument("--num_workers", help="(unused: the dataset is loaded once and kept on the GPU)",
                        default=4, type=int)
    parser.add_argument("--save_dir", help="Directory for logs and checkpoints", default="./results/single_prev", type=str)

    parser.add_argument("--num_gaussians", help="gaussian factor: number of gaussians", default=32, type=int)
    parser.add_argument("--residual", help="Whether to use residual connection in the model", action="store_true")
    parser.add_argument("--init_scale", help="inital scale", default=0.15, type=float)
    parser.add_argument("--lambda_lab", help="weight of lab loss", default=10.0, type=float)
    parser.add_argument("--lambda_sparse", help="weight of sparse loss", default=0.001, type=float)

    parser.add_argument("--batch_size", help="Batch size for training", default=4096, type=int)
    parser.add_argument("--batch_size_val", help="Batch size for evaluation", default=10000, type=int)
    parser.add_argument("--num_epochs", help="Number of training epochs (calibrated at --ref_batch_size)",
                        default=20, type=int)
    parser.add_argument("--learning_rate", help="Learning rate (calibrated at --ref_batch_size)",
                        default=0.001, type=float)
    parser.add_argument("--ref_batch_size", help="Reference batch size that --num_epochs / --learning_rate "
                        "are tuned for; larger --batch_size is compensated relative to this",
                        default=4096, type=int)
    parser.add_argument("--no_batch_compensation", action="store_true",
                        help="Disable the automatic epoch/lr scaling applied when batch_size > ref_batch_size")
    parser.add_argument("--eval_interval", help="Evaluation interval (in epochs)", default=10, type=int)
    parser.add_argument("--random_seed", help="random seed", default=52, type=int)

    parser.add_argument("--cuda_eval", action="store_true",
                        help="Use the fused CUDA kernel for the forward pass during evaluation "
                             "(inference only; requires a GPU and the compiled glut_cuda extension).")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    random_everything(seed=args.random_seed)

    model, trainer = main(args)
