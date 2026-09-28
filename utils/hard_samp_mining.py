"""
Hard-example mining for GLUT training.

``HardExampleMiner`` picks the hardest samples in a mini-batch by one of several
strategies; ``CurriculumHardMining`` wraps it with a curriculum schedule (a warm-up
with no mining, then a mining ratio that ramps up over training).  ``train_glut.py``
uses ``CurriculumHardMining`` with the ``error_threshold`` strategy.
"""

import random
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardExampleMiner:
    """Select the hardest samples in a batch (several strategies)."""

    def __init__(self,
                 strategy='error_threshold',
                 threshold_percentile=80,
                 mining_ratio=0.3,
                 buffer_size=10000,
                 update_frequency=10,
                 min_samples=100):
        """
        Args:
            strategy: one of
                'error_threshold' - keep samples above a dynamic error percentile
                'top_k'           - keep the ``mining_ratio`` fraction with largest error
                'uncertainty'     - MC-dropout predictive variance (needs dropout layers)
                'gradient'        - per-sample input-gradient norm
                'mixed'           - error + noise-uncertainty + colour-space sparsity
            threshold_percentile: percentile used by the 'error_threshold' strategy.
            mining_ratio: fraction of the batch to treat as hard.
            buffer_size: size of the replay buffer of past hard samples.
            update_frequency: push to the replay buffer every N calls.
            min_samples: minimum error history before a dynamic threshold is used.
        """
        self.strategy = strategy
        self.threshold_percentile = threshold_percentile
        self.mining_ratio = mining_ratio
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.min_samples = min_samples

        self.hard_example_buffer = deque(maxlen=buffer_size)
        self.error_history = deque(maxlen=1000)
        self.step_counter = 0

    def compute_errors(self, model: nn.Module,
                       rgb_input: torch.Tensor, rgb_target: torch.Tensor) -> torch.Tensor:
        """Per-sample error = mean L1 + 0.1 * angular error (degrees). Returns [B]."""
        with torch.no_grad():
            rgb_output = model(rgb_input)

            l1_errors = torch.abs(rgb_output - rgb_target).mean(dim=1)

            pred_norm = rgb_output / (torch.norm(rgb_output, dim=1, keepdim=True) + 1e-8)
            target_norm = rgb_target / (torch.norm(rgb_target, dim=1, keepdim=True) + 1e-8)
            dot_product = torch.sum(pred_norm * target_norm, dim=1)
            angular_errors = torch.acos(torch.clamp(dot_product, -1, 1)) * 180 / np.pi

            return l1_errors + 0.1 * angular_errors

    def mine_by_error_threshold(self, errors: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Keep samples whose error exceeds a running percentile threshold."""
        self.error_history.extend(errors.cpu().numpy())

        if len(self.error_history) > self.min_samples:
            threshold = torch.tensor(
                np.percentile(self.error_history, self.threshold_percentile),
                device=errors.device)
            hard_mask = errors > threshold

            # Top up with random samples if too few crossed the threshold.
            if hard_mask.sum() < batch_size * self.mining_ratio:
                n_needed = int(batch_size * self.mining_ratio) - hard_mask.sum().item()
                if n_needed > 0:
                    hard_mask[torch.randperm(len(errors))[:n_needed]] = True
        else:
            # Not enough history yet: pick at random.
            n_hard = int(batch_size * self.mining_ratio)
            hard_mask = torch.zeros(len(errors), dtype=torch.bool, device=errors.device)
            hard_mask[torch.randperm(len(errors))[:n_hard]] = True

        return hard_mask

    def mine_by_top_k(self, errors: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Keep the ``mining_ratio`` fraction of the batch with the largest error."""
        k = int(batch_size * self.mining_ratio)
        _, hard_indices = torch.topk(errors, k=min(k, len(errors)))
        hard_mask = torch.zeros(len(errors), dtype=torch.bool, device=errors.device)
        hard_mask[hard_indices] = True
        return hard_mask

    def mine_by_uncertainty(self, model: nn.Module, rgb_input: torch.Tensor,
                            n_iterations: int = 10) -> torch.Tensor:
        """MC-dropout: keep the samples with the highest predictive variance."""
        model.train()  # keep dropout active
        predictions = []
        for _ in range(n_iterations):
            with torch.no_grad():
                predictions.append(model(rgb_input))
        model.eval()

        uncertainty = torch.stack(predictions).var(dim=0).mean(dim=1)  # [B]
        k = int(len(rgb_input) * self.mining_ratio)
        _, hard_indices = torch.topk(uncertainty, k=k)
        hard_mask = torch.zeros(len(rgb_input), dtype=torch.bool, device=rgb_input.device)
        hard_mask[hard_indices] = True
        return hard_mask

    def mine_by_gradient(self, model: nn.Module,
                         rgb_input: torch.Tensor, rgb_target: torch.Tensor) -> torch.Tensor:
        """Keep the samples with the largest input-gradient norm (slow: per-sample)."""
        grad_norms = []
        for i in range(len(rgb_input)):
            single_input = rgb_input[i:i + 1].requires_grad_(True)
            loss = F.l1_loss(model(single_input), rgb_target[i:i + 1])
            grad = torch.autograd.grad(loss, single_input, retain_graph=True)[0]
            grad_norms.append(grad.norm().item())

        grad_norms = torch.tensor(grad_norms, device=rgb_input.device)
        k = int(len(rgb_input) * self.mining_ratio)
        _, hard_indices = torch.topk(grad_norms, k=k)
        hard_mask = torch.zeros(len(rgb_input), dtype=torch.bool, device=rgb_input.device)
        hard_mask[hard_indices] = True
        return hard_mask

    def mine_by_mixed(self, model: nn.Module,
                      rgb_input: torch.Tensor, rgb_target: torch.Tensor,
                      errors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine (normalised) error, noise-perturbation uncertainty and colour-space sparsity."""
        if errors is None:
            errors = self.compute_errors(model, rgb_input, rgb_target)
        error_scores = errors / (errors.max() + 1e-8)

        # Predictive spread under small input noise.
        with torch.no_grad():
            predictions = [model(rgb_input + torch.randn_like(rgb_input) * 0.01) for _ in range(5)]
            uncertainty = torch.stack(predictions).var(dim=0).mean(dim=1)
            uncertainty_scores = uncertainty / (uncertainty.max() + 1e-8)

        # Local density in RGB space: sparse regions score higher.
        from scipy.spatial import KDTree
        rgb_np = rgb_input.cpu().numpy()
        tree = KDTree(rgb_np)
        distances, _ = tree.query(rgb_np, k=min(10, len(rgb_np)))
        density_scores = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
        density_scores = torch.tensor(density_scores, device=rgb_input.device)
        density_scores = 1.0 - density_scores / (density_scores.max() + 1e-8)

        combined_scores = (error_scores + uncertainty_scores + density_scores) / 3
        k = int(len(rgb_input) * self.mining_ratio)
        _, hard_indices = torch.topk(combined_scores, k=k)
        hard_mask = torch.zeros(len(rgb_input), dtype=torch.bool, device=rgb_input.device)
        hard_mask[hard_indices] = True
        return hard_mask

    def update_buffer(self, hard_samples: torch.Tensor,
                      hard_targets: torch.Tensor, hard_errors: torch.Tensor):
        """Append the current hard samples to the replay buffer."""
        for i in range(len(hard_samples)):
            self.hard_example_buffer.append({
                'rgb': hard_samples[i].cpu(),
                'target': hard_targets[i].cpu(),
                'error': hard_errors[i].cpu().item(),
            })

    def sample_from_buffer(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw a random batch from the replay buffer (or (None, None) if too small)."""
        if len(self.hard_example_buffer) < batch_size:
            return None, None
        samples = random.sample(list(self.hard_example_buffer),
                                min(batch_size, len(self.hard_example_buffer)))
        rgb_batch = torch.stack([s['rgb'] for s in samples])
        target_batch = torch.stack([s['target'] for s in samples])
        return rgb_batch, target_batch

    def __call__(self, model: nn.Module,
                 rgb_input: torch.Tensor, rgb_target: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Return (hard_inputs, hard_targets, stats) for one batch."""
        self.step_counter += 1
        errors = self.compute_errors(model, rgb_input, rgb_target)

        if self.strategy == 'error_threshold':
            hard_mask = self.mine_by_error_threshold(errors, len(rgb_input))
        elif self.strategy == 'top_k':
            hard_mask = self.mine_by_top_k(errors, len(rgb_input))
        elif self.strategy == 'uncertainty':
            hard_mask = self.mine_by_uncertainty(model, rgb_input)
        elif self.strategy == 'gradient':
            hard_mask = self.mine_by_gradient(model, rgb_input, rgb_target)
        elif self.strategy == 'mixed':
            hard_mask = self.mine_by_mixed(model, rgb_input, rgb_target, errors)
        else:
            n_hard = int(len(rgb_input) * self.mining_ratio)
            hard_mask = torch.zeros(len(rgb_input), dtype=torch.bool, device=rgb_input.device)
            hard_mask[torch.randperm(len(rgb_input))[:n_hard]] = True

        hard_inputs = rgb_input[hard_mask]
        hard_targets = rgb_target[hard_mask]
        hard_errors = errors[hard_mask]

        if self.step_counter % self.update_frequency == 0:
            self.update_buffer(hard_inputs, hard_targets, hard_errors)

        stats = {
            'hard_samples_ratio': hard_mask.sum().item() / len(rgb_input),
            'mean_hard_error': hard_errors.mean().item() if len(hard_errors) > 0 else 0,
            'buffer_size': len(self.hard_example_buffer),
            'error_threshold': (np.percentile(self.error_history, self.threshold_percentile)
                                if len(self.error_history) > self.min_samples else 0),
        }
        return hard_inputs, hard_targets, stats


class CurriculumHardMining:
    """Curriculum wrapper: no mining during warm-up, then a ramping mining ratio."""

    def __init__(self,
                 total_epochs: int,
                 warmup_epochs: int = 10,
                 mining_start_epoch: int = 20,
                 initial_mining_ratio: float = 0.1,
                 final_mining_ratio: float = 0.5):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.mining_start_epoch = mining_start_epoch
        self.initial_mining_ratio = initial_mining_ratio
        self.final_mining_ratio = final_mining_ratio

        self.miner = HardExampleMiner(
            strategy='error_threshold',
            threshold_percentile=80,
            mining_ratio=initial_mining_ratio,
            update_frequency=50,
        )

    def get_mining_ratio(self, epoch: int) -> float:
        """Mining ratio schedule as a function of the epoch."""
        if epoch < self.warmup_epochs:
            return 0.0
        if epoch < self.mining_start_epoch:
            # Ramp 0 -> initial_mining_ratio.
            progress = (epoch - self.warmup_epochs) / (self.mining_start_epoch - self.warmup_epochs)
            return self.initial_mining_ratio * progress
        # Ramp initial_mining_ratio -> final_mining_ratio.
        progress = min(1.0, (epoch - self.mining_start_epoch)
                       / (self.total_epochs - self.mining_start_epoch))
        return self.initial_mining_ratio + progress * (self.final_mining_ratio - self.initial_mining_ratio)

    def __call__(self, epoch: int, model: nn.Module,
                 rgb_input: torch.Tensor, rgb_target: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Return the (possibly hard-mined) training batch for this step."""
        current_ratio = self.get_mining_ratio(epoch)
        self.miner.mining_ratio = current_ratio

        # Stochastically skip mining so most batches stay unbiased.
        if current_ratio == 0 or np.random.random() > current_ratio:
            return rgb_input, rgb_target, {'mining_ratio': current_ratio}

        hard_inputs, hard_targets, stats = self.miner(model, rgb_input, rgb_target)

        # Top up from the replay buffer if the batch is short.
        if len(hard_inputs) < len(rgb_input) * current_ratio:
            buffer_inputs, buffer_targets = self.miner.sample_from_buffer(
                len(rgb_input) - len(hard_inputs))
            if buffer_inputs is not None:
                hard_inputs = torch.cat([hard_inputs, buffer_inputs.to(hard_inputs.device)])
                hard_targets = torch.cat([hard_targets, buffer_targets.to(hard_targets.device)])

        stats['mining_ratio'] = current_ratio
        return hard_inputs, hard_targets, stats
