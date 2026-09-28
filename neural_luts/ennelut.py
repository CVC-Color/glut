"""
Modified version for ENNELUT: Supports multiple LUT styles with one-hot conditioning
Fixed memory issue: Batch processing for validation
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import os
import sys
import gc
import argparse

# Make the repo root (which holds the sibling `utils`/`data`/`neural_luts`
# packages) importable regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import clean_mem, count_parameters, setup_logging
from data.hald_dataloader import ConditionalLUTFitting, ValidationDataset
from neural_luts.cnilut import validate_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
start_time = None


def rgb_to_lab(rgb):
    """Convert an RGB tensor [B, 3] to CIE-LAB (sRGB input assumed to be in [0, 1])."""
    # 1. Linearise (inverse sRGB gamma).
    mask = rgb > 0.04045
    rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # 2. Linear RGB -> XYZ.
    matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device)
    xyz = torch.matmul(rgb, matrix.T)

    # 3. XYZ -> LAB (D65 white point).
    xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device)
    xyz = xyz / xyz_ref

    mask = xyz > 0.008856
    f_xyz = torch.where(mask, torch.pow(xyz, 1/3), 7.787 * xyz + 16/116)

    l = 116 * f_xyz[..., 1:2] - 16
    a = 500 * (f_xyz[..., 0:1] - f_xyz[..., 1:2])
    b = 200 * (f_xyz[..., 1:2] - f_xyz[..., 2:3])
    
    return torch.cat([l, a, b], dim=-1)

def delta_e_loss(model_output, targets):
    """Mean CIE76 Delta-E between two RGB tensors."""
    lab_output = rgb_to_lab(model_output)
    lab_targets = rgb_to_lab(targets)
    diff = lab_output - lab_targets
    de = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
    return torch.mean(de)


class LipSwish(nn.Module):
    """Swish scaled to have a bounded derivative (needed for invertibility / stability)."""
    def forward(self, x):
        return 0.909 * F.silu(x)


class ResidualComponent(nn.Module):
    """One residual block Ti.

    Conditioning is applied by adding a per-LUT bias (looked up from, or a weighted
    combination of, the embedding matrix E) to the first layer.
    """
    def __init__(self, num_luts, hidden_structure=[32, 64, 32]):
        super().__init__()
        h1, h2, h3 = hidden_structure

        # Per-LUT conditional bias table.
        self.embedding = nn.Embedding(num_luts, h1)

        self.fc1 = nn.Linear(3, h1, bias=False)   # bias comes from the embedding
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.fc3 = nn.Linear(h2, h3, bias=False)
        self.fc4 = nn.Linear(h3, 3, bias=False)

        self.activation = LipSwish()
        self.act_norm = nn.Parameter(torch.ones(1, 3))   # ActNorm factor, init to 1

        # 1/100 weight shrink so the block starts near the identity.
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.div_(100.0)

    def forward(self, x, lut_index=None, one_hot=None):
        """
        lut_index: [N] discrete LUT indices.
        one_hot:   [N, num_luts] continuous weights, for interpolation.
        """
        if one_hot is not None:
            # Interpolation: [N, num_luts] @ [num_luts, h1] -> [N, h1].
            lut_bias = torch.matmul(one_hot, self.embedding.weight)
        else:
            lut_bias = self.embedding(lut_index)

        res = self.fc1(x) + lut_bias
        res = self.activation(res)
        res = self.fc2(res)
        res = self.activation(res)
        res = self.fc3(res)
        res = self.activation(res)
        res = self.fc4(res)

        return x + res * self.act_norm   # ActNorm-scaled residual add


class ENNELUT(nn.Module):
    """Efficient Neural Network Encoding for LUTs (ENNELUT)."""
    def __init__(self, num_styles=512, D=3, hidden_dim=32, a=0.83):
        super().__init__()
        self.num_styles = num_styles
        self.a = a
        struct = [hidden_dim, hidden_dim * 2, hidden_dim]
        self.components = nn.ModuleList([
            ResidualComponent(num_styles, hidden_structure=struct) for _ in range(D)
        ])

    def forward(self, intensity, style_idx=None, one_hot=None):
        """
        intensity: RGB input [N, 3].
        style_idx: style index [N].
        one_hot:   continuous style weights [N, num_styles] (for interpolation).
        """
        batch_size = intensity.shape[0]

        if one_hot is None and style_idx is not None:
            # Accept a plain int as well as a tensor.
            if not isinstance(style_idx, torch.Tensor):
                style_idx = torch.full((batch_size,), style_idx, dtype=torch.long,
                                       device=intensity.device)
        elif one_hot is None:
            raise ValueError("Either style_idx or one_hot must be provided")

        # Pre-process: map [0, 1] -> [-1, 1] -> unbounded via atanh.
        x = 2 * self.a * (intensity - 0.5)
        x = torch.atanh(x)

        for comp in self.components:
            x = comp(x, lut_index=style_idx, one_hot=one_hot)

        # Post-process: back to (0, 1).
        output = (torch.tanh(x) + 1.0) / 2.0
        return torch.clamp(output, 0.0, 1.0)




def main(args):
    """
    Train a Conditional ENNELUT on multiple LUT styles
    
    Args:
        inp_path: Path to input Hald image for training
        target_folder: Folder containing target Hald images for training (multiple styles)
        inp_path_val: Path to input Hald image for validation
        target_folder_val: Folder containing target Hald images for validation
        total_steps: Number of training steps
        lut_size: Tuple of (hidden_units, hidden_blocks)
        save_dir: Directory to save models and logs
        val_batch_size: Batch size for validation to avoid OOM
    """

    total_steps=args.steps
    lut_size=(args.units, args.blocks)
    
    logger = setup_logging(
        log_dir=args.save_dir,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )

    logger.info("Training Configuration:")
    logger.info(args.__dict__)

    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Start ENNELUT {lut_size} fitting with")
    # logger.info(f"Input hald image : {args.input}")
    logger.info(f"Target folder: {args.target_folder}")

    # Create datasets
    train_dataset = ConditionalLUTFitting(args.target_folder)
    val_dataset_raw = ConditionalLUTFitting(args.target_folder_val)
    
    num_styles = train_dataset.num_styles
    logger.info(f"Number of styles: {num_styles}")
    logger.info(f"Style names: {train_dataset.style_names}")
    
    # Create validation datasets for each style with batching
    val_datasets = []
    for style_idx in range(num_styles):
        intensities = val_dataset_raw.intensities
        targets = val_dataset_raw.target_tensors[style_idx]
        val_dataset = ValidationDataset(
            intensities, targets, style_idx, num_styles, 
            batch_size=args.val_batch_size
        )
        val_datasets.append(val_dataset)
    
    img_size = val_dataset_raw.shape
    logger.info(f"Validation image size: {img_size}")
    logger.info(f"Validation pixels per style: {val_dataset_raw.num_pixels}")
    logger.info(f"Validation batches per style: {len(val_datasets[0])}")
    
    # Create dataloader for training
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                            pin_memory=True, num_workers=4)
    
    logger.info(f"Dataloader ready, image size: {img_size}")
    logger.info(f"Training samples: {len(train_dataset)} (pixels × styles)")
    
    # Define the conditional model
    lut_model = ENNELUT(
        num_styles=num_styles,
        D=args.blocks,        # number of residual components
        hidden_dim=args.units # hidden width inside each component
    ).cuda()
    
    lut_model.cuda()
    optimizer = torch.optim.Adam(lr=args.lr, params=lut_model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    
    # from torchinfo import summary
    # rgb = torch.randn(512*512, 3).to(device)
    # lut_idx = torch.randint(low=1, high=num_styles, size=(512*512,)).to(device)
    # summary(lut_model, input_data=[rgb, lut_idx])
    
    # from fvcore.nn import FlopCountAnalysis
    # lut_model.eval()
    # flops = FlopCountAnalysis(lut_model, (rgb, lut_idx))
    # print(flops.total() / 1e9, "GFLOPs")
    
    
    logger.info(f"Created Conditional ENNELUT model {lut_size} -- params={count_parameters(lut_model)}")
    logger.info(f"Model input dimension: 3 (RGB) + {num_styles} (one-hot) ")
    
    # Training loop
    lut_model.train()
    logger.info(f"** Start training for {total_steps} iterations")
    
    step = 0
    best_avg_psnr = 0
    
    while step < total_steps:
        for batch_idx, (conditional_input, targets) in enumerate(train_loader):
            if step >= total_steps:
                break
                
            conditional_input = conditional_input.cuda()
            targets = targets.cuda()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # Split conditional input back to RGB and one-hot
                rgb_input = conditional_input[:, :3]
                one_hot = conditional_input[:, 3:]
                
                model_output = lut_model(rgb_input, one_hot=one_hot)
                
                if args.loss_type == 'l1':
                    loss = torch.mean(torch.abs(model_output - targets))
                elif args.loss_type == 'l2':
                    loss = F.mse_loss(model_output, targets)
                elif args.loss_type == 'de':
                    loss = delta_e_loss(model_output, targets)
            
            if step % 200 == 0:
                logger.info(f"-- Train Step: {step}, loss: {loss.item():.6f}")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Validation every 5000 steps
            if (step % args.eval_interval == 0 and step > 0) or step == total_steps:
                avg_metrics = validate_model(
                    lut_model, val_datasets, img_size, 
                    train_dataset.style_names, logger, step, num_styles
                )
                
                
                lut_model.train()
            
            step += 1
    
    # Final validation
    logger.info("Training complete. Running final validation...")
    final_metrics = validate_model(
        lut_model, val_datasets, img_size, 
        train_dataset.style_names, logger, total_steps, num_styles
    )
    
    # Save final model
    model_path = os.path.join(args.save_dir, 
                             f"ennelut{lut_size[0]}x{lut_size[1]}_{num_styles}styles_psnr{final_metrics['psnr']:.2f}_dE{final_metrics['dE76']:.2f}_ep{total_steps}.pth")
    torch.save({
        'model_state_dict': lut_model.state_dict(),
        'style_names': train_dataset.style_names,
        'num_styles': num_styles,
        'lut_size': lut_size,
        'final_metrics': final_metrics
    }, model_path)
    logger.info(f"Final model saved to {model_path}")
    
    clean_mem()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional ENNELUT fitting')
    parser.add_argument("--target_folder", help="Folder containing target Hald images for different LUT styles", 
                        default="./dataset/hald_images/75luts_train", type=str)
    parser.add_argument("--target_folder_val", help="Folder containing target Hald images for validation", 
                        default="./dataset/hald_images/75luts_test", type=str)
    
    parser.add_argument("--lr", help="learning rate", default=0.04, type=int)
    parser.add_argument("--lr_step", help="decay step size", default=2560, type=int)
    parser.add_argument("--loss_type", help="type of loss", default='l2', type=str)
    
    parser.add_argument("--save_dir", help="Directory to save results", 
                        default="./results/ennelut", type=str)
    parser.add_argument("--steps", help="Number of optimization steps", default=30760, type=int)
    parser.add_argument("--eval_interval", help="eval every n steps", default=40000, type=int)
    
    parser.add_argument("--units", help="Number of neurons per hidden layer", default=32, type=int)
    parser.add_argument("--blocks", help="Number of hidden blocks", default=3, type=int)
    parser.add_argument("--val_batch_size", help="Batch size for validation to avoid OOM", 
                        default=65535, type=int)
    parser.add_argument("--train_batch_size", help="Batch size for validation to avoid OOM", 
                        default=2048, type=int)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    main(args)
