"""
NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement
https://github.com/mv-lab/nilut

Modified version for Conditional NILUT: Supports multiple LUT styles with one-hot conditioning
Fixed memory issue: Batch processing for validation
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import os
import sys
import gc
import argparse

# Make the repo root (which holds the sibling `utils`/`data` packages)
# importable regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import clean_mem, count_parameters, setup_logging
from utils.metrics import calc_metrics_torch, calc_deltaE
from data.hald_dataloader import ConditionalLUTFitting, ValidationDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
start_time = None


class ConditionalNILUT(nn.Module):
    """
    Conditional Neural Implicit 3D Lookup Table
    Input: RGB + one-hot encoding of style (3 + num_styles dimensions)
    Output: Enhanced RGB
    """
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3, 
                 out_features=3, num_styles=1, res=True):
        super().__init__()
        
        self.res = res
        self.num_styles = num_styles
        self.in_features = in_features + num_styles  # RGB + one-hot
        
        self.net = []
        self.net.append(nn.Linear(self.in_features, hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity, style_idx=None, one_hot=None):
        """
        Args:
            intensity: RGB input tensor [N, 3]
            style_idx: style index (int) for one-hot encoding
            one_hot: pre-computed one-hot encoding [N, num_styles]
        """
        batch_size = intensity.shape[0]
        
        # Create one-hot encoding if not provided
        if one_hot is None and style_idx is not None:
            one_hot = torch.zeros(batch_size, self.num_styles).to(intensity.device)
            one_hot[:, style_idx] = 1.0
        elif one_hot is None:
            raise ValueError("Either style_idx or one_hot must be provided")
        
        # print(one_hot)
        
        # Concatenate RGB with one-hot encoding
        conditional_input = torch.cat([intensity, one_hot], dim=-1)
        
        output = self.net(conditional_input)
        if self.res:
            output = output + intensity
        output = torch.clamp(output, 0., 1.)
        
        return output




def validate_model(model, val_dataset, img_size, style_names, logger, step, num_styles):
    """
    Validate model on all styles using batch processing to avoid OOM
    """
    model.eval()
    
    # Initialize accumulators
    all_metrics = {style_idx: {'l1': 0, 'l2': 0, 'psnr': 0, 'dE00': 0, 'dE76': 0, 'max_dE00': 0, 
                               'count': 0, 'all_dE76': [], 'all_psnr': []} 
                  for style_idx in range(num_styles)}
    
    with torch.no_grad():
        for style_idx in range(num_styles):
            style_name = style_names[style_idx]
            val_dataloader = DataLoader(val_dataset[style_idx], batch_size=1, shuffle=False)
            
            logger.info(f'  Validating style {style_idx} ({style_name})...')
            
            # Process batches
            for batch_idx, (batch_intensities, batch_targets, batch_one_hot) in enumerate(val_dataloader):
                # Move to GPU
                batch_intensities = batch_intensities.squeeze(0).cuda()  # Remove batch dimension from dataloader
                batch_targets = batch_targets.squeeze(0).cuda()
                batch_one_hot = batch_one_hot.squeeze(0).cuda()
                
                # Forward pass
                batch_output = model(batch_intensities, one_hot=batch_one_hot)
                
                # Calculate metrics for this batch
                batch_output_np = batch_output.cpu().numpy()
                batch_targets_np = batch_targets.cpu().numpy()
                
                # Since calc_metrics expects images, we need to reshape if it's the full image
                if batch_output_np.shape[0] == img_size[0] * img_size[1]:
                    # This is the full image
                    batch_output_np = batch_output_np.reshape(img_size[0], img_size[1], 3)
                    batch_targets_np = batch_targets_np.reshape(img_size[0], img_size[1], 3)
                    metrics = calc_metrics_torch(batch_targets_np, batch_output_np)
                    
                    # Store metrics
                    for key in ['l1', 'l2', 'psnr', 'dE00', 'dE76', 'max_dE00']:
                        all_metrics[style_idx][key] = metrics[key]
                    all_metrics[style_idx]['count'] = 1
                else:
                    # For partial batches, we need to handle differently
                    # Simple approach: accumulate L1 loss and pixel count
                    l1_loss = np.mean(np.abs(batch_output_np - batch_targets_np))
                    mse_loss = np.mean((batch_output_np - batch_targets_np) ** 2)
                    deltaE00_pixel = calc_deltaE(batch_output_np, batch_targets_np, method='CIE 2000', reduction='none')
                    deltaE00 = np.mean(deltaE00_pixel)
                    deltaE76 = calc_deltaE(batch_output_np, batch_targets_np, method='CIE 1976', reduction='mean')
                    max_dE00 = np.max(deltaE00_pixel)
                    
                    all_metrics[style_idx]['l1'] += l1_loss * batch_output_np.shape[0]
                    all_metrics[style_idx]['l2'] += mse_loss * batch_output_np.shape[0]
                    all_metrics[style_idx]['dE00'] += deltaE00 * batch_output_np.shape[0]
                    all_metrics[style_idx]['dE76'] += deltaE76 * batch_output_np.shape[0]
                    all_metrics[style_idx]['max_dE00'] = np.maximum(all_metrics[style_idx]['max_dE00'], max_dE00)
                    all_metrics[style_idx]['count'] += batch_output_np.shape[0]
                    
                    # For PSNR, we'll compute from MSE
                    # For dE, we'll store all values to compute mean later
                    # This is simplified; for accurate dE you might want to process full images
                    
                    # Approximate PSNR from MSE
                    if mse_loss > 0:
                        psnr_batch = 10 * np.log10(1.0 / mse_loss)
                        all_metrics[style_idx]['psnr'] += psnr_batch * batch_output_np.shape[0]
            
            # Finalize metrics for this style
            if all_metrics[style_idx]['count'] > 0:
                if all_metrics[style_idx]['count'] == 1:
                    # Already have full metrics
                    pass
                else:
                    # Average per-pixel metrics
                    all_metrics[style_idx]['l1'] /= all_metrics[style_idx]['count']
                    all_metrics[style_idx]['l2'] /= all_metrics[style_idx]['count']
                    all_metrics[style_idx]['psnr'] /= all_metrics[style_idx]['count']
                    all_metrics[style_idx]['dE00'] /= all_metrics[style_idx]['count']
                    all_metrics[style_idx]['dE76'] /= all_metrics[style_idx]['count']
                    
    
    # Calculate averages
    avg_metrics = {'psnr': 0, 'dE00': 0, 'dE76': 0, 'l1': 0, 'l2': 0}
    
    logger.info(f'-- Evaluation Step: {step} --')
    for style_idx in range(num_styles):
        metrics = all_metrics[style_idx]
        logger.info(f'  Style {style_idx} ({style_names[style_idx]}): '
                  f'l1 {metrics["l1"]:.6f} | '
                  f'l2 {metrics["l2"]:.6f} | '
                  f'psnr {metrics["psnr"]:.2f} | '
                  f'dE00 {metrics["dE00"]:.4f} | '
                  f'dE76 {metrics["dE76"]:.4f} | '
                  f'max_dE00 {metrics["max_dE00"]:.4f} | ')
        
        for key in avg_metrics:
            if key in metrics:
                avg_metrics[key] += metrics[key]
    
    # Average across styles
    for key in avg_metrics:
        avg_metrics[key] /= num_styles
    
    logger.info(f'  Averages: '
                  f'l1 {avg_metrics["l1"]:.6f} | '
                  f'l2 {avg_metrics["l2"]:.6f} | '
                  f'psnr {avg_metrics["psnr"]:.2f} | '
                  f'dE00 {avg_metrics["dE00"]:.4f} | '
                  f'dE76 {avg_metrics["dE76"]:.4f}')
    
    return avg_metrics


def main(args):
    """
    Train a Conditional NILUT on multiple LUT styles
    
    Args:
        inp_path: Path to input Hald image for training
        target_folder: Folder containing target Hald images for training (multiple styles)
        inp_path_val: Path to input Hald image for validation
        target_folder_val: Folder containing target Hald images for validation
        total_steps: Number of training steps
        lut_size: Tuple of (hidden_units, hidden_layers)
        save_dir: Directory to save models and logs
        val_batch_size: Batch size for validation to avoid OOM
    """

    total_steps=args.steps
    lut_size=(args.units, args.layers)
    
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

    logger.info(f"Start Conditional NILUT {lut_size} fitting with")
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
    lut_model = ConditionalNILUT(
        in_features=3,
        out_features=3,
        hidden_features=lut_size[0],
        hidden_layers=lut_size[1],
        num_styles=num_styles,
        res=True
    )
    lut_model.cuda()
    opt = torch.optim.Adam(lr=1e-3, params=lut_model.parameters())
    
    # from torchinfo import summary
    # rgb = torch.randn(512*512, 3).to(device)
    # lut_idx = torch.randint(low=1, high=num_styles, size=(512*512,)).to(device)
    # summary(lut_model, input_data=[rgb, lut_idx])
    
    # from fvcore.nn import FlopCountAnalysis
    # lut_model.eval()
    # flops = FlopCountAnalysis(lut_model, (rgb, lut_idx))
    # print(flops.total() / 1e9, "GFLOPs")
    
    
    logger.info(f"Created Conditional NILUT model {lut_size} -- params={count_parameters(lut_model)}")
    logger.info(f"Model input dimension: 3 (RGB) + {num_styles} (one-hot) = {3 + num_styles}")
    
    # Training loop
    lut_model.train()
    logger.info(f"** Start training for {total_steps} iterations")
    
    step = 0
    
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
                loss = torch.mean(torch.abs(model_output - targets))
            
            if step % 200 == 0:
                logger.info(f"-- Train Step: {step}, loss: {loss.item():.6f}")
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
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
                             f"cnilut{lut_size[0]}x{lut_size[1]}_{num_styles}styles_psnr{final_metrics['psnr']:.2f}_dE{final_metrics['dE76']:.2f}_ep{total_steps}.pth")
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
    parser = argparse.ArgumentParser(description='Conditional NILUT fitting')
    parser.add_argument("--target_folder", help="Folder containing target Hald images for different LUT styles", 
                        default="./dataset/hald_images/75luts_train", type=str)
    parser.add_argument("--target_folder_val", help="Folder containing target Hald images for validation", 
                        default="./dataset/hald_images/75luts_test", type=str)
    
    parser.add_argument("--save_dir", help="Directory to save results", 
                        default="./results/cnilut", type=str)
    parser.add_argument("--steps", help="Number of optimization steps", default=60000, type=int)
    parser.add_argument("--eval_interval", help="eval every n steps", default=20000, type=int)
    
    parser.add_argument("--units", help="Number of neurons per hidden layer", default=256, type=int)
    parser.add_argument("--layers", help="Number of hidden layers", default=3, type=int)
    parser.add_argument("--val_batch_size", help="Batch size for validation to avoid OOM", 
                        default=262144, type=int)
    parser.add_argument("--train_batch_size", help="Batch size for validation to avoid OOM", 
                        default=65535, type=int)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    main(args)
