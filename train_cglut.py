"""
Conditional Gaussian-LUT (cGLUT): fit many 3D LUTs with a single model.

A learned per-LUT condition embedding is fed through a small MLP (`param_encoder`)
and a set of per-parameter heads that emit the parameters of a Gaussian mixture
in the RGB cube (centres, covariances, opacities, per-Gaussian affine colour
transforms and a global affine transform).  An input colour is transformed by
blending the local transforms with opacity-weighted Gaussian responses
(partition-of-unity), optionally added on top of the global transform.

"""

import argparse
import gc
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from typing import Dict

from data.hald_dataloader import MultiHaldBatch
from utils.utils import setup_logging, random_everything, all_equal_and_value, initialize_gaussians_grid
from utils.metrics import calc_deltaE
from utils.losses import LABspaceLoss

# ---------------------------------------------------------------------------
# Optional fused CUDA kernel for inference.  Imported lazily so the script also
# runs on CPU-only machines or when the extension has not been compiled.
# ---------------------------------------------------------------------------
try:
    from glut_cuda.glut import glut_forward

    _CUDA_KERNEL_AVAILABLE = True
    _CUDA_IMPORT_ERROR = None
except Exception as _cuda_import_error:  # noqa: BLE001 - any failure disables the kernel
    glut_forward = None
    _CUDA_KERNEL_AVAILABLE = False
    _CUDA_IMPORT_ERROR = _cuda_import_error


# Which Gaussian parameters are shared across LUTs (a single learned tensor) vs.
# generated per-LUT by a head network.  Colour parameters are always per-LUT.
_PARAM_SHARING = {
    #                       positions  covariance  opacity   global
    'color':                dict(share_positions=True,  share_covariance=True,  share_opacity=True,  share_global=False),
    'color_opacity':        dict(share_positions=True,  share_covariance=True,  share_opacity=False, share_global=False),
    'color_opacity_covariance': dict(share_positions=True, share_covariance=False, share_opacity=False, share_global=False),
    'all':                  dict(share_positions=False, share_covariance=False, share_opacity=False, share_global=False),
}


class ConditionalGLUT(nn.Module):
    """Conditional Gaussian-LUT with a per-parameter head architecture.

    ``shared_param`` selects which parameter groups are LUT-agnostic (stored as
    ``base_*`` parameters) and which are produced per LUT by a small head MLP.
    """

    def __init__(self,
                 num_gaussians: int = 64,
                 num_conditions: int = 5,
                 condition_dim: int = 32,
                 num_neurons: int = 128,
                 shared_param: str = 'color_opacity',
                 residual: bool = True,
                 init_scale: float = 0.1,
                 epsilon: float = 1e-8,
                 logger=None):
        """
        Args:
            num_gaussians: number of Gaussians in the mixture.
            num_conditions: number of LUTs (rows of the condition embedding).
            condition_dim: dimensionality of the condition embedding.
            num_neurons: width of the encoder / head MLPs.
            shared_param: one of ``_PARAM_SHARING`` keys.
            residual: add the global affine transform to the local one (always
                True for this script, kept for API compatibility).
            init_scale: initial Gaussian standard deviation (shared-covariance case).
            epsilon: numerical-stability constant.
            logger: optional logger.
        """
        super().__init__()

        if shared_param not in _PARAM_SHARING:
            raise ValueError(f"shared_param must be one of {list(_PARAM_SHARING)}, got {shared_param!r}")
        cfg = _PARAM_SHARING[shared_param]

        self.num_gaussians = num_gaussians
        self.num_conditions = num_conditions
        self.condition_dim = condition_dim
        self.num_neurons = num_neurons
        self.share_positions = cfg['share_positions']
        self.share_covariance = cfg['share_covariance']
        self.share_opacity = cfg['share_opacity']
        self.share_global = cfg['share_global']
        self.residual = residual
        self.logger = logger
        self.epsilon = epsilon

        # --- Gaussian centres ---------------------------------------------------
        if self.share_positions:
            self.base_positions = nn.Parameter(initialize_gaussians_grid(num_gaussians))
        else:
            self.position_head = nn.Sequential(
                nn.Linear(num_neurons, num_neurons), nn.ReLU(),
                nn.Linear(num_neurons, num_gaussians * 3),
            )

        # --- Covariance (via 6 Cholesky parameters per Gaussian) --------------
        if self.share_covariance:
            self.base_cholesky_diag = nn.Parameter(
                torch.full((num_gaussians, 3), math.log(init_scale)))
            self.base_cholesky_off = nn.Parameter(torch.zeros(num_gaussians, 3))
        else:
            self.covariance_head = nn.Sequential(
                nn.Linear(num_neurons, num_neurons), nn.ReLU(),
                nn.Linear(num_neurons, num_gaussians * 6),
            )

        # --- Opacity / blend weight -----------------------------------------
        if self.share_opacity:
            self.base_opacities_logit = nn.Parameter(torch.ones(num_gaussians, 1))
        else:
            self.opacity_head = nn.Sequential(
                nn.Linear(num_neurons, num_neurons), nn.ReLU(),
                nn.Linear(num_neurons, num_gaussians * 1),
            )

        # --- Per-Gaussian affine colour transform (always per-LUT) ----------
        #     12 outputs per Gaussian = 3x3 matrix (9) + bias (3).
        self.color_head = nn.Sequential(
            nn.Linear(num_neurons, num_neurons), nn.ReLU(),
            nn.Linear(num_neurons, num_neurons), nn.ReLU(),
            nn.Linear(num_neurons, num_gaussians * 12),
        )

        # --- Global affine colour transform --------------------------------
        if self.share_global:
            self.base_global_matrix = nn.Parameter(torch.eye(3))
            self.base_global_bias = nn.Parameter(torch.zeros(3))
        else:
            self.global_head = nn.Sequential(
                nn.Linear(num_neurons, num_neurons), nn.ReLU(),
                nn.Linear(num_neurons, 12),
            )

        # --- Condition encoding -------------------------------------------
        self.condition_embedding = nn.Embedding(num_conditions, condition_dim)
        self.param_encoder = nn.Sequential(
            nn.Linear(condition_dim, num_neurons), nn.ReLU(),
            nn.Linear(num_neurons, num_neurons), nn.ReLU(),
            nn.Linear(num_neurons, num_neurons), nn.ReLU(),
        )

        self.opacity_activation = nn.Sigmoid()

        self.register_buffer('eye_3x3', torch.eye(3))
        self.register_buffer('log_2pi', torch.tensor(math.log(2 * math.pi)))

        self._initialize_parameters()

        # Filled in by forward(); used by the sparsity regulariser.
        self.params: Dict[str, torch.Tensor] = {}

        if self.logger:
            self.logger.info("Initialize conditional GLUT (ConditionalGLUT):")
            self.logger.info(f"  - Number of Gaussians: {self.num_gaussians}")
            self.logger.info(f"  - Number of LUTs     : {self.num_conditions}")
            self.logger.info(f"  - Sharing            : {shared_param}")
            self.logger.info(f"  - Total parameters   : {self.count_parameters():,}")
            self.logger.info("  - Model device       : {}".format(
                "GPU" if torch.cuda.is_available() else "CPU"))

    def _initialize_parameters(self):
        """Weight initialisation (Kaiming for heads, Xavier for the encoder)."""
        nn.init.normal_(self.condition_embedding.weight, mean=0.0, std=0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

        for layer in self.param_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Parameter generation
    # ------------------------------------------------------------------
    def generate_parameters(self, condition_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate the full Gaussian-mixture parameter set for a batch of LUTs.

        Args:
            condition_emb: [B, condition_dim] condition embeddings.

        Returns:
            dict with, for batch size B and N Gaussians:
                positions      [B, N, 3]
                covariances    [B, N, 3, 3]  (Sigma = L L^T)
                opacities      [B, N, 1]     (in (0, 1))
                color_matrices [B, N, 3, 3]  (diagonal kept positive via softplus)
                color_biases   [B, N, 3]
                global_matrix  [B, 3, 3]     (diagonal kept positive via softplus)
                global_bias    [B, 3]
        """
        B = condition_emb.shape[0]
        N = self.num_gaussians
        device = condition_emb.device

        shared_features = self.param_encoder(condition_emb)  # [B, num_neurons]
        params: Dict[str, torch.Tensor] = {}

        # --- positions -----------------------------------------------------
        if self.share_positions:
            params['positions'] = self.base_positions.unsqueeze(0).expand(B, -1, -1)
        else:
            params['positions'] = self.position_head(shared_features).view(B, N, 3)

        # --- covariance: build lower-triangular L, then Sigma = L L^T -----
        if self.share_covariance:
            diag = F.softplus(self.base_cholesky_diag).unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]
            off = self.base_cholesky_off.unsqueeze(0).expand(B, -1, -1)                # [B, N, 3]
        else:
            cov_raw = self.covariance_head(shared_features).view(B, N, 6)
            diag = F.softplus(cov_raw[..., :3])   # positive diagonal
            off = cov_raw[..., 3:]

        L = torch.zeros(B, N, 3, 3, device=device, dtype=diag.dtype)
        L[..., 0, 0] = diag[..., 0]
        L[..., 1, 1] = diag[..., 1]
        L[..., 2, 2] = diag[..., 2]
        L[..., 1, 0] = off[..., 0]   # L[1, 0]
        L[..., 2, 0] = off[..., 1]   # L[2, 0]
        L[..., 2, 1] = off[..., 2]   # L[2, 1]
        params['covariances'] = torch.matmul(L, L.transpose(2, 3))  # [B, N, 3, 3]

        # --- opacity -----------------------------------------------------
        if self.share_opacity:
            logit = self.base_opacities_logit.unsqueeze(0).expand(B, -1, -1)
        else:
            logit = self.opacity_head(shared_features).view(B, N, 1)
        params['opacities'] = self.opacity_activation(logit)   # [B, N, 1]

        # --- per-Gaussian colour transform (diagonal >= 0 via softplus) --
        color_p = self.color_head(shared_features).view(B, N, 12)
        color_matrices = color_p[..., :9].view(B, N, 3, 3)
        diag_mask = torch.zeros(3, 3, device=device, dtype=torch.bool)
        diag_mask[range(3), range(3)] = True
        dm = diag_mask.view(1, 1, 3, 3)
        color_matrices = F.softplus(color_matrices) * dm + color_matrices * (~dm)
        params['color_matrices'] = color_matrices
        params['color_biases'] = color_p[..., 9:]

        # --- global affine transform (diagonal >= 0 via softplus) -------
        if self.share_global:
            params['global_matrix'] = self.base_global_matrix.unsqueeze(0).expand(B, -1, -1)
            params['global_bias'] = self.base_global_bias.unsqueeze(0).expand(B, -1)
        else:
            global_flat = self.global_head(shared_features).view(B, 12)
            global_matrix = global_flat[..., :9].view(B, 3, 3)
            dm_g = diag_mask.view(1, 3, 3)
            global_matrix = F.softplus(global_matrix) * dm_g + global_matrix * (~dm_g)
            params['global_matrix'] = global_matrix
            params['global_bias'] = global_flat[..., 9:]

        return params

    # ------------------------------------------------------------------
    # Gaussian responses
    # ------------------------------------------------------------------
    def compute_gaussian_influences(self, rgb: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Gaussian pdf value of every Gaussian at every input colour. [B, N].

        The 3x3 precision matrix and log|Sigma| are computed in closed form from
        the covariance components (no ``torch.linalg.inv`` / ``torch.det``).
        """
        cov = params['covariances']   # [B, N, 3, 3]
        s00 = cov[..., 0, 0]; s01 = cov[..., 0, 1]; s02 = cov[..., 0, 2]
        s11 = cov[..., 1, 1]; s12 = cov[..., 1, 2]; s22 = cov[..., 2, 2]

        det = (s00 * (s11 * s22 - s12 * s12)
               - s01 * (s01 * s22 - s12 * s02)
               + s02 * (s01 * s12 - s11 * s02))
        log_det = torch.log(det.abs() + self.epsilon)
        inv_det = 1.0 / (det + self.epsilon)

        # Precision P = Sigma^-1 (symmetric), via the adjugate matrix.
        p00 = (s11 * s22 - s12 * s12) * inv_det
        p01 = (s02 * s12 - s01 * s22) * inv_det
        p02 = (s01 * s12 - s02 * s11) * inv_det
        p11 = (s00 * s22 - s02 * s02) * inv_det
        p12 = (s01 * s02 - s00 * s12) * inv_det
        p22 = (s00 * s11 - s01 * s01) * inv_det

        diff = rgb.unsqueeze(1) - params['positions']   # [B, N, 3]
        dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]
        mahal = (dx * dx * p00 + dy * dy * p11 + dz * dz * p22
                 + 2.0 * (dx * dy * p01 + dx * dz * p02 + dy * dz * p12))

        log_pdf = -0.5 * (mahal + log_det + 3.0 * self.log_2pi)
        return torch.exp(log_pdf)   # [B, N]

    @staticmethod
    def _blend(rgb, params, influences, residual, epsilon):
        """Blend the per-Gaussian affine transforms and add the global one.

        Shared by forward() and forward_unified() (PyTorch path).  All params
        tensors have batch dim B (broadcast is fine).
        """
        # Opacity-weighted, renormalised weights (partition of unity).
        w = influences * params['opacities'].squeeze(-1)                 # [B, N]
        w = w / (w.sum(dim=1, keepdim=True) + epsilon)                   # [B, N]

        # Local transform: weighted sum of per-Gaussian affine outputs.
        transformed = torch.einsum('bnij,bnj->bni', params['color_matrices'],
                                   rgb.unsqueeze(1).expand(-1, w.shape[1], -1))
        transformed = transformed + params['color_biases']              # [B, N, 3]
        local = torch.sum(transformed * w.unsqueeze(-1), dim=1)         # [B, 3]

        # Global transform.
        rgb_global = torch.einsum('bij,bj->bi', params['global_matrix'], rgb) + params['global_bias']

        out = rgb_global + local if residual else local
        return torch.clamp(out, 0, 1)

    def forward(self, rgb: torch.Tensor, condition_idx: torch.Tensor) -> torch.Tensor:
        """Training / general forward.

        ``condition_idx`` is generated *per pixel*, so ``generate_parameters``
        runs once per element of the batch.  Use :meth:`forward_unified` when a
        whole batch shares one LUT.

        Args:
            rgb: [B, 3] input colours.
            condition_idx: [B] or [B, 1] LUT index for each pixel.
        Returns:
            [B, 3] transformed colours, clamped to [0, 1].
        """
        B = condition_idx.shape[0]
        condition_emb = self.condition_embedding(condition_idx).view(B, -1)  # [B, condition_dim]

        self.params = self.generate_parameters(condition_emb)
        influences = self.compute_gaussian_influences(rgb, self.params)      # [B, N]
        return self._blend(rgb, self.params, influences, self.residual, self.epsilon)

    def forward_unified(self, rgb: torch.Tensor, condition_idx: torch.Tensor,
                        use_cuda: bool = False) -> torch.Tensor:
        """Single-condition inference: run the generator once, reuse for all pixels.

        ``forward`` re-runs ``param_encoder`` + every head for each of the H*W
        pixels even though an image uses one LUT.  Here the generator is called
        once and the (batch-1) parameters are broadcast over the pixel batch.

        Args:
            rgb: [B, 3] pixels, all belonging to the same LUT.
            condition_idx: a scalar / 0-d / [1] tensor, or a [B] tensor holding
                that same value repeated.
            use_cuda: run the blend through the compiled ``glut_cuda`` kernel
                (requires a CUDA ``rgb`` tensor and the built extension).
        Returns:
            [B, 3] transformed colours, clamped to [0, 1].
        """
        unique_idx = all_equal_and_value(condition_idx)
        if unique_idx is None:
            raise ValueError("forward_unified requires all pixels to share one condition_idx")

        idx = torch.as_tensor([unique_idx], device=rgb.device, dtype=torch.long)
        condition_emb = self.condition_embedding(idx).view(1, -1)   # [1, condition_dim]
        params = self.generate_parameters(condition_emb)            # batch dim 1

        if use_cuda:
            return self._forward_unified_cuda(rgb, params)

        # PyTorch path: broadcast the batch-1 params over the pixel batch.
        B, N = rgb.shape[0], self.num_gaussians
        expanded = {
            'positions': params['positions'].expand(B, N, 3),
            'covariances': params['covariances'].expand(B, N, 3, 3),
            'opacities': params['opacities'].expand(B, N, 1),
            'color_matrices': params['color_matrices'].expand(B, N, 3, 3),
            'color_biases': params['color_biases'].expand(B, N, 3),
            'global_matrix': params['global_matrix'].expand(B, 3, 3),
            'global_bias': params['global_bias'].expand(B, 3),
        }
        influences = self.compute_gaussian_influences(rgb, expanded)
        return self._blend(rgb, expanded, influences, self.residual, self.epsilon)

    def _forward_unified_cuda(self, rgb: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the single-condition blend through the fused ``glut_cuda`` kernel.

        The precision matrix and log|Sigma| are derived in closed form exactly as
        in :meth:`compute_gaussian_influences`, so the kernel output matches the
        PyTorch path up to float32 rounding.
        """
        if not _CUDA_KERNEL_AVAILABLE:
            raise RuntimeError(f"CUDA kernel not available: {_CUDA_IMPORT_ERROR}")
        if not rgb.is_cuda:
            raise RuntimeError("forward_unified(use_cuda=True) requires a CUDA rgb tensor")

        cov = params['covariances'][0]   # [N, 3, 3]
        s00 = cov[:, 0, 0]; s01 = cov[:, 0, 1]; s02 = cov[:, 0, 2]
        s11 = cov[:, 1, 1]; s12 = cov[:, 1, 2]; s22 = cov[:, 2, 2]

        det = (s00 * (s11 * s22 - s12 * s12)
               - s01 * (s01 * s22 - s12 * s02)
               + s02 * (s01 * s12 - s11 * s02))
        inv_det = 1.0 / (det + self.epsilon)

        p00 = (s11 * s22 - s12 * s12) * inv_det
        p01 = (s02 * s12 - s01 * s22) * inv_det
        p02 = (s01 * s12 - s02 * s11) * inv_det
        p11 = (s00 * s22 - s02 * s02) * inv_det
        p12 = (s01 * s02 - s00 * s12) * inv_det
        p22 = (s00 * s11 - s01 * s01) * inv_det

        prec = torch.stack([p00, p01, p02, p01, p11, p12, p02, p12, p22], dim=1)  # [N, 9]
        log_det = torch.log(det.abs() + self.epsilon)                             # [N]

        # weight_norm=True: cGLUT uses partition-of-unity blending.
        return glut_forward(
            rgb,
            params['positions'][0].contiguous(),
            prec.contiguous(),
            log_det.contiguous(),
            params['opacities'][0].squeeze(-1).contiguous(),
            params['color_matrices'][0].reshape(-1, 9).contiguous(),
            params['color_biases'][0].contiguous(),
            params['global_matrix'][0].reshape(-1).contiguous(),
            params['global_bias'][0].contiguous(),
            self.residual,
            True,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_gaussian_info(self, idx_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Per-Gaussian geometry for the LUT(s) in ``idx_tensor`` (first sample)."""
        with torch.no_grad():
            B = idx_tensor.shape[0]
            condition_emb = self.condition_embedding(idx_tensor).view(B, -1)
            params = self.generate_parameters(condition_emb)

            positions = params['positions'][0].cpu().numpy()
            covariances = params['covariances'][0].cpu().numpy()
            opacities = params['opacities'][0].cpu().numpy()

            variances, volumes = [], []
            for i in range(self.num_gaussians):
                eigvals = np.maximum(np.linalg.eigvalsh(covariances[i]), 1e-10)
                stds = np.sqrt(eigvals)
                variances.append(stds)
                volumes.append((4.0 / 3.0) * math.pi * float(np.prod(stds)))

            return {
                'positions': positions,
                'variances': np.array(variances),
                'volumes': np.array(volumes),
                'opacities': opacities.flatten(),
            }

    def count_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiGaussianLUTTrainer:
    """Training / evaluation wrapper around :class:`ConditionalGLUT`."""

    def __init__(self, model: ConditionalGLUT):
        self.model = model
        self.lab_loss = LABspaceLoss(losstype='hc')

    def train_step(self, optimizer, rgb_input, rgb_target, lut_idx,
                   lambda_sparse: float = 0.0, lambda_lab: float = 1.0):
        """One optimisation step (forward + loss + backward + clip + step).

        Args:
            lambda_sparse: weight of the opacity entropy (sparsity) regulariser.
            lambda_lab:    weight of the LAB-space perceptual loss.
        Returns:
            dict of scalar loss values.
        """
        optimizer.zero_grad()

        output = self.model(rgb_input, lut_idx)

        main_loss = F.l1_loss(output, rgb_target)          # L1 works well for colour
        lab_loss = self.lab_loss(output, rgb_target)

        # Sparsity: push opacities towards 0 or 1 (low binary entropy).
        sparse_loss = 0.0
        if lambda_sparse > 0:
            op = self.model.params['opacities']
            entropy = -op * torch.log(op + 1e-8) - (1 - op) * torch.log(1 - op + 1e-8)
            sparse_loss = torch.mean(entropy)

        total_loss = main_loss + lambda_sparse * sparse_loss + lambda_lab * lab_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'sparse_loss': sparse_loss.item() if lambda_sparse > 0 else 0.0,
            'lab_loss': lab_loss.item() if lambda_lab > 0 else 0.0,
        }

    def evaluate(self, rgb_input, rgb_target, lut_idx, reduction='mean'):
        """Evaluate on one batch. ``reduction`` is passed straight to the metrics."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(rgb_input, lut_idx)

            if reduction == 'none':
                l1_error = F.l1_loss(output, rgb_target, reduction='none').mean(dim=1).cpu().numpy()
                l2_error = F.mse_loss(output, rgb_target, reduction='none').mean(dim=1).cpu().numpy()
            elif reduction == 'sum':
                l1_error = F.l1_loss(output, rgb_target, reduction='sum').item()
                l2_error = F.mse_loss(output, rgb_target, reduction='sum').item()
            else:
                l1_error = F.l1_loss(output, rgb_target).item()
                l2_error = F.mse_loss(output, rgb_target).item()

            deltaE00 = calc_deltaE(output.cpu().numpy(), rgb_target.cpu().numpy(),
                                   method='CIE 2000', reduction=reduction)
            deltaE76 = calc_deltaE(output.cpu().numpy(), rgb_target.cpu().numpy(),
                                   method='CIE 1976', reduction=reduction)

            return {
                'output': output,
                'l1_error': l1_error,
                'l2_error': l2_error,
                'deltaE00': deltaE00,
                'deltaE76': deltaE76,
            }

    def save_checkpoint(self, path: str, epoch: int):
        """Save weights + the hyper-params needed to rebuild the model."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'num_gaussians': self.model.num_gaussians,
            'num_conditions': self.model.num_conditions,
            'condition_dim': self.model.condition_dim,
            'num_neurons': self.model.num_neurons,
            'residual': self.model.residual,
        }, path)


def main(args):
    """Full training run."""
    print("=" * 60)
    print("Conditional 3D Color Transform with Gaussian Model")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(
        log_dir=args.save_dir, log_level="INFO",
        log_to_file=True, log_to_console=True, logname_prefix="cglut_train")
    logger.info("Training Configuration:")
    logger.info(args.__dict__)

    # 1. Datasets (one DataLoader item = (rgb, target_rgb, lut_idx)).
    logger.info("1. Creating color transformation dataset...")
    train_dataset = MultiHaldBatch(args.lut_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=True, shuffle=True, num_workers=args.num_workers * 2)
    val_dataset = MultiHaldBatch(args.lut_data_test)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val,
                            pin_memory=True, shuffle=False, num_workers=args.num_workers)
    num_luts = train_dataset.num_luts
    logger.info(f"Dataloader ready, number of LUTs: {num_luts}")

    # 2. Model.
    logger.info("2. Initializing conditional Gaussian LUT model...")
    model = ConditionalGLUT(
        num_gaussians=args.num_gaussians,
        num_conditions=num_luts,
        condition_dim=args.num_conditions,
        num_neurons=args.num_neurons,
        shared_param=args.shared_param,
        residual=True,
        init_scale=args.init_scale,
        logger=logger,
    ).to(device)

    trainer = MultiGaussianLUTTrainer(model)

    # 3. Optimizer: shared/base params and the embedding get a 10x smaller lr
    #    than the encoder + head networks.
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'base_' in n],
         'lr': args.learning_rate * 0.1},
        {'params': list(model.condition_embedding.parameters()),
         'lr': args.learning_rate * 0.1},
        {'params': [p for n, p in model.named_parameters() if '_head' in n]
                   + list(model.param_encoder.parameters()),
         'lr': args.learning_rate},
    ]
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader))

    # 4. Training loop.
    logger.info("3. Training conditional Gaussian color transform model...")
    logger.info("-" * 60)
    iteration = 0

    for epoch in range(args.num_epochs):
        model.train()
        for rgb_input, rgb_target, lut_idx in train_loader:
            iteration += 1
            rgb_input = rgb_input.to(device)
            rgb_target = rgb_target.to(device)
            lut_idx = lut_idx.to(device)

            losses = trainer.train_step(
                optimizer, rgb_input, rgb_target, lut_idx,
                lambda_sparse=args.lambda_sparse, lambda_lab=args.lambda_lab)
            scheduler.step()

            if (iteration + 1) % 100 == 0 or iteration == 0:
                logger.info(f"Epoch {epoch + 1:3d} | Iter {iteration + 1:6d} | "
                            f"Total {losses['total_loss']:.6f} | "
                            f"l1 {losses['main_loss']:.6f} | "
                            f"lab {losses['lab_loss']:.6f} | "
                            f"sparse {losses['sparse_loss']:.6f}")

        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.num_epochs:
            _evaluate_all_luts(model, trainer, val_loader, num_luts, args, device, logger, epoch)

        torch.cuda.empty_cache()
        gc.collect()

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)

    lut_name = (f'cglut_g{args.num_gaussians}_e{args.num_conditions}_{num_luts}styles'
                f'_{args.shared_param}.pth')
    trainer.save_checkpoint(os.path.join(args.save_dir, lut_name), args.num_epochs)
    logger.info(f"Final model saved to {lut_name}")
    return model, trainer


def _evaluate_all_luts(model, trainer, val_loader, num_luts, args, device, logger, epoch):
    """Run the validation set and log per-LUT + average metrics."""
    logger.info("4. Evaluating model performance...")
    logger.info("-" * 60)

    l1_all = np.zeros(num_luts, dtype=np.float64)
    l2_all = np.zeros(num_luts, dtype=np.float64)
    dE00_all = np.zeros(num_luts, dtype=np.float64)
    dE76_all = np.zeros(num_luts, dtype=np.float64)
    max_err_all = np.zeros(num_luts, dtype=np.float64)
    count = np.zeros(num_luts, dtype=np.int64)

    for val_rgb_input, val_rgb_target, val_lut_idx in val_loader:
        unique_idx = all_equal_and_value(val_lut_idx)
        val_rgb_input = val_rgb_input.to(device)
        val_rgb_target = val_rgb_target.to(device)
        val_lut_idx = val_lut_idx.to(device)

        if unique_idx is not None:
            m = trainer.evaluate(val_rgb_input, val_rgb_target, val_lut_idx, reduction='sum')
            n = val_rgb_input.shape[0]
            l1_all[unique_idx] += m['l1_error']
            l2_all[unique_idx] += m['l2_error']
            dE00_all[unique_idx] += m['deltaE00']
            dE76_all[unique_idx] += m['deltaE76']
            max_err_all[unique_idx] = np.maximum(max_err_all[unique_idx], m['deltaE00'] / n)
            count[unique_idx] += n
        else:
            m = trainer.evaluate(val_rgb_input, val_rgb_target, val_lut_idx, reduction='none')
            idx = val_lut_idx.squeeze(1).cpu().numpy()
            l1_all += np.bincount(idx, weights=m['l1_error'], minlength=num_luts)
            l2_all += np.bincount(idx, weights=m['l2_error'], minlength=num_luts)
            dE00_all += np.bincount(idx, weights=m['deltaE00'], minlength=num_luts)
            dE76_all += np.bincount(idx, weights=m['deltaE76'], minlength=num_luts)
            np.maximum.at(max_err_all, idx, m['deltaE00'])
            count += np.bincount(idx, minlength=num_luts)

    avg_l1 = l1_all / count
    avg_mse = l2_all / count
    avg_dE00 = dE00_all / count
    avg_dE76 = dE76_all / count
    avg_psnr = 10 * np.log10(1.0 / avg_mse)

    logger.info(f'--- Validation after Epoch: {epoch + 1} ---')
    for i in range(num_luts):
        logger.info('-- LUT {:2d} | l1: {:.6f} | psnr: {:.2f} | dE00: {:.4f} | dE76: {:.4f} | max_err: {:.4f}'.format(
            i, avg_l1[i], avg_psnr[i], avg_dE00[i], avg_dE76[i], max_err_all[i]))
    logger.info('-- Average | l1: {:.6f} | psnr: {:.2f} | dE00: {:.4f} | dE76: {:.4f} | max_err: {:.4f}'.format(
        avg_l1.mean(), avg_psnr.mean(), avg_dE00.mean(), avg_dE76.mean(), max_err_all.mean()))

    for idx in range(num_luts):
        gi = model.get_gaussian_info(torch.tensor([idx], device=device))
        active = int((gi['opacities'] > 0.1).sum())
        logger.info(f"LUT {idx}: mean volume {gi['volumes'].mean():.5f} | "
                    f"active gaussians (opacity>0.1) {active}/{model.num_gaussians}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGLUT Training')

    parser.add_argument("--lut_data", default='./dataset/hald_images/7luts_train', type=str,
                        help="Folder of enhanced hald images (one per LUT) for training")
    parser.add_argument("--lut_data_test", default='./dataset/hald_images/7luts_test', type=str,
                        help="Folder of enhanced hald images (one per LUT) for validation")
    parser.add_argument("--save_dir", default="./results/multiple", type=str,
                        help="Directory for logs and checkpoints")
    parser.add_argument("--num_workers", default=8, type=int, help="DataLoader workers")

    parser.add_argument("--num_gaussians", default=64, type=int, help="Number of Gaussians")
    parser.add_argument("--num_conditions", default=32, type=int, help="Condition embedding dimension")
    parser.add_argument("--num_neurons", default=128, type=int, help="Width of encoder / head MLPs")
    parser.add_argument("--shared_param", default='color_opacity', type=str,
                        choices=list(_PARAM_SHARING),
                        help="Which Gaussian parameters are shared across LUTs")
    parser.add_argument("--init_scale", default=0.15, type=float, help="Initial Gaussian scale")

    parser.add_argument("--lambda_lab", default=10.0, type=float, help="Weight of the LAB loss")
    parser.add_argument("--lambda_sparse", default=0.0, type=float, help="Weight of the opacity sparsity loss")

    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--batch_size_val", default=10000, type=int)
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--eval_interval", default=20, type=int, help="Epochs between validations")
    parser.add_argument("--random_seed", default=52, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    random_everything(seed=args.random_seed)
    model, trainer = main(args)
