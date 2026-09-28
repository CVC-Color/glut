"""
Utils for training and ploting
"""

import torch
import cv2
import gc
import time
import os
import logging
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image


def initialize_gaussians_grid(num_gaussians: int, padding: float = 0.1):
    """Place Gaussian centres on a regular grid inside the RGB cube.

    Args:
        num_gaussians: number of centres to return.
        padding: keep-out margin from the cube boundary.

    Returns:
        positions: [num_gaussians, 3] float tensor of centres.
    """
    n_per_dim = int(round(num_gaussians ** (1 / 3)))

    x = np.linspace(padding, 1 - padding, n_per_dim)
    y = np.linspace(padding, 1 - padding, n_per_dim)
    z = np.linspace(padding, 1 - padding, n_per_dim)
    X, Y, Z = np.meshgrid(x, y, z)
    positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    if len(positions) < num_gaussians:
        # Not enough grid points: top up with random points.
        n_extra = num_gaussians - len(positions)
        extra = np.random.uniform(padding, 1 - padding, (n_extra, 3))
        positions = np.vstack([positions, extra])
    elif len(positions) > num_gaussians:
        # Too many grid points: keep a random subset.
        indices = np.random.choice(len(positions), num_gaussians, replace=False)
        positions = positions[indices]

    return torch.FloatTensor(positions)


def all_equal_and_value(x: torch.Tensor):
    """Return the common scalar value if every element of ``x`` is equal, else None."""
    if x.numel() == 0:
        return None

    v = x.view(-1)
    first = v[0]
    return first.item() if torch.all(v == first) else None


def random_everything(seed=42):
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str = "./logs", 
                  log_level: str = "INFO",
                  log_to_file: bool = True,
                  log_to_console: bool = True,
                  logname_prefix: str = "training"):
    """Configure the root logger with console and/or timestamped-file handlers.

    Args:
        log_dir: directory for the log file.
        log_level: DEBUG | INFO | WARNING | ERROR | CRITICAL.
        log_to_file: also write to ``<log_dir>/<logname_prefix>_<timestamp>.log``.
        log_to_console: also write to stdout.
        logname_prefix: file-name prefix for the log file.

    Returns:
        the ``"GaussianLUT"`` child logger.
    """
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(console_handler)

    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{logname_prefix}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    main_logger = logging.getLogger("GaussianLUT")
    main_logger.info(f"log system initialized, level: {log_level}")
    main_logger.info(f"log directory: {os.path.abspath(log_dir)}")
    return main_logger


def visualize_color_transformation(model, save_path):
    """Scatter-plot the RGB cube before/after the model transform at a few B slices."""
    fig = plt.figure(figsize=(15, 10))

    n_points = 20
    r_vals = np.linspace(0, 1, n_points)
    g_vals = np.linspace(0, 1, n_points)
    b_vals = [0.0, 0.3, 0.6, 0.9]

    for b_idx, b_val in enumerate(b_vals):
        R, G = np.meshgrid(r_vals, g_vals)
        input_grid = np.stack([R.flatten(), G.flatten(), np.full(R.size, b_val)], axis=1)

        with torch.no_grad():
            input_tensor = torch.tensor(input_grid, dtype=torch.float32).cuda()
            output_grid = model(input_tensor).cpu().numpy()

        ax1 = fig.add_subplot(len(b_vals), 2, 2 * b_idx + 1)
        ax1.scatter(R.flatten(), G.flatten(), c=input_grid, s=50, alpha=0.8)
        ax1.set_xlabel('R'); ax1.set_ylabel('G')
        ax1.set_title(f'Input (B={b_val:.1f})')
        ax1.set_xlim(-0.1, 1.1); ax1.set_ylim(-0.1, 1.1); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(len(b_vals), 2, 2 * b_idx + 2)
        ax2.scatter(R.flatten(), G.flatten(), c=output_grid, s=50, alpha=0.8)
        ax2.set_xlabel('R'); ax2.set_ylabel('G')
        ax2.set_title(f'Transformed (B={b_val:.1f})')
        ax2.set_xlim(-0.1, 1.1); ax2.set_ylim(-0.1, 1.1); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def visualize_loss(train_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')

# Timing utilities
def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    
def clean_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# Model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Load/save and plot images
def load_img(filename, norm=True):

    img = np.array(Image.open(filename))
    if norm:   
        img = img / 255.
        img = img.astype(np.float32)
    return img

def save_rgb(img, filename):
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    cv2.imwrite(filename, img)

def add_cmap(img, colormap=cv2.COLORMAP_PARULA):
    img = np.abs(img)
    # if img.max() <= 1:
    img = np.asarray(img * 255, dtype=np.uint8)
    img = cv2.applyColorMap(img, colormap)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_all (images, figsize=(20,10), axis='off', title=None):

    nplots = len(images)
    fig, axs = plt.subplots(1,nplots, figsize=figsize, dpi=80,constrained_layout=True)
    
    for i in range(nplots):
        axs[i].imshow(images[i])
        axs[i].axis(axis)
    plt.show()

 
