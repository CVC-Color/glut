"""
cGLUT LUT interpolation on a single image.

"Blending" LUT A and LUT B at weight ``alpha`` means interpolating their learned
condition embeddings in a trained :class:`ConditionalGLUT` and running the model
once:  ``emb = (1 - alpha) * emb_A + alpha * emb_B``.

The script sweeps ``alpha`` from 0 to 1 for one image, optionally comparing each
blend against the reference ``(1 - alpha) * A(x) + alpha * B(x)`` (a per-pixel
linear blend of the two ``.cube`` LUT outputs) and writing a metrics CSV plus an
alpha-sweep strip image.

LUT indices (``--lut_a`` / ``--lut_b``) are positions in the *sorted* list of
``.cube`` files in ``--lut``; that order must match the order the model was
trained on (``MultiHaldBatch`` also sorts the target file names).
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data.apply_cube import CubeFileTransform
from utils.metrics import calc_metrics_np
from utils.utils import random_everything
from train_cglut import ConditionalGLUT, _PARAM_SHARING


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_shared_param(state_dict) -> str:
    """Recover the ``shared_param`` key from which ``base_*`` tensors are present."""
    flags = {
        'share_positions':  'base_positions' in state_dict,
        'share_covariance': 'base_cholesky_diag' in state_dict,
        'share_opacity':    'base_opacities_logit' in state_dict,
        'share_global':     'base_global_matrix' in state_dict,
    }
    for name, cfg in _PARAM_SHARING.items():
        if cfg == flags:
            return name
    raise ValueError(f"cannot map the checkpoint's sharing pattern {flags} to a known config")


def load_cglut(checkpoint_path: str, device) -> ConditionalGLUT:
    """Build a :class:`ConditionalGLUT` whose architecture is read from the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt['model_state_dict']

    num_conditions = sd['condition_embedding.weight'].shape[0]
    condition_dim = sd['condition_embedding.weight'].shape[1]
    num_neurons = sd['param_encoder.0.weight'].shape[0]
    num_gaussians = ckpt.get('num_gaussians') or sd['color_head.4.weight'].shape[0] // 12
    residual = ckpt.get('residual', True)
    shared_param = _infer_shared_param(sd)

    model = ConditionalGLUT(
        num_gaussians=num_gaussians, num_conditions=num_conditions,
        condition_dim=condition_dim, num_neurons=num_neurons,
        shared_param=shared_param, residual=residual, logger=None,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    print(f"cGLUT: gaussians={num_gaussians}  luts={num_conditions}  "
          f"cond_dim={condition_dim}  neurons={num_neurons}  "
          f"shared={shared_param}  residual={residual}")
    return model


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

@torch.no_grad()
def blend_cglut(model: ConditionalGLUT, image_path: str, idx_a: int, idx_b: int,
                alpha: float, device='cuda') -> np.ndarray:
    """Run ``model`` on ``image_path`` with the condition embedding interpolated
    between LUT ``idx_a`` (alpha=0) and LUT ``idx_b`` (alpha=1).

    Returns an ``[H, W, 3]`` float array in ``[0, 1]``.  The blend uses the model's
    own ``compute_gaussian_influences`` / ``_blend`` so it matches ``model.forward``
    exactly (this is ``forward_unified`` with an interpolated embedding).
    """
    img = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    pixels = torch.from_numpy(img).view(-1, 3).to(device)          # [HW, 3], RGB in [0,1]

    emb_a = model.condition_embedding(torch.tensor([idx_a], device=device))   # [1, D]
    emb_b = model.condition_embedding(torch.tensor([idx_b], device=device))   # [1, D]
    emb = (1.0 - alpha) * emb_a + alpha * emb_b                               # [1, D]

    params = model.generate_parameters(emb)                        # every tensor has batch dim 1
    B = pixels.shape[0]
    expanded = {k: v.expand(B, *v.shape[1:]) for k, v in params.items()}
    influences = model.compute_gaussian_influences(pixels, expanded)          # [HW, N]
    out = model._blend(pixels, expanded, influences, model.residual, model.epsilon)
    return out.view(h, w, 3).cpu().numpy()


def reference_blend(image_path: str, cube_a: str, cube_b: str, alpha: float) -> np.ndarray:
    """Reference: per-pixel linear blend of the two ``.cube`` LUT outputs. ``[H, W, 3]``."""
    img = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    pixels = torch.from_numpy(img).view(-1, 3)

    out_a = CubeFileTransform(cube_a, 'trilinear')(pixels).view(h, w, 3).numpy()
    out_b = CubeFileTransform(cube_b, 'trilinear')(pixels).view(h, w, 3).numpy()
    return np.clip((1.0 - alpha) * out_a + alpha * out_b, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def add_metric_label(img: np.ndarray, value: float) -> np.ndarray:
    """Draw a semi-transparent grey box with a dE00 label in the bottom-right corner."""
    pil = Image.fromarray(img).convert("RGBA")
    iw, ih = pil.size

    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text = f"dE00={value:.2f}"

    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, ih // 12))
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    draw.rectangle([(iw - tw - 2 * pad, ih - th - 2 * pad), (iw, ih)], fill=(128, 128, 128, 150))
    draw.text((iw - tw - pad, ih - th - pad), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(Image.alpha_composite(pil, overlay).convert("RGB"))


def _to_uint8(img_float: np.ndarray, scale: int = 2, label: float = None) -> np.ndarray:
    """[0,1] float -> uint8, optionally downsampled and dE00-labelled."""
    a = (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    if scale > 1:
        a = np.asarray(Image.fromarray(a).resize(
            (max(1, a.shape[1] // scale), max(1, a.shape[0] // scale)), Image.BOX))
    if label is not None:
        a = add_metric_label(a, label)
    return a


def _hstack_with_gaps(tiles, gap=8):
    h = tiles[0].shape[0]
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [sep, t]
    return np.hstack(row)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    random_everything(seed=args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_cglut(args.pretrained_model, device)

    cubes = sorted(f for f in os.listdir(args.lut) if f.lower().endswith('.cube'))
    if not (0 <= args.lut_a < len(cubes)) or not (0 <= args.lut_b < len(cubes)):
        raise ValueError(f"--lut_a / --lut_b must be in [0, {len(cubes) - 1}] "
                         f"({len(cubes)} .cube files in {args.lut})")
    cube_a = os.path.join(args.lut, cubes[args.lut_a])
    cube_b = os.path.join(args.lut, cubes[args.lut_b])

    alphas = [round(float(a), 4) for a in np.linspace(0.0, 1.0, args.n_alpha)]
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    print(f"image: {img_name}   LUT {args.lut_a} ({cubes[args.lut_a]})  ->  "
          f"{args.lut_b} ({cubes[args.lut_b]})   alphas: {alphas}")

    lpips_metric = None
    if args.metrics:
        import lpips
        lpips_metric = lpips.LPIPS(net='vgg').to(device).eval()

    gt_row, pred_row = [], []
    csv_rows = []

    for alpha in alphas:
        pred = blend_cglut(model, args.image, args.lut_a, args.lut_b, alpha, device=device)

        de00 = None
        if args.metrics:
            gt = reference_blend(args.image, cube_a, cube_b, alpha)
            m = calc_metrics_np(pred, gt, lpips_metric)
            de00 = m['dE00']
            csv_rows.append([args.lut_a, args.lut_b, img_name, alpha,
                             f"{m['PSNR']:.4f}", f"{m['SSIM']:.4f}",
                             f"{m['dE00']:.4f}", f"{m['dE76']:.4f}",
                             f"{m['RMSE']:.4f}", f"{m['MAE']:.4f}", f"{m['LPIPS']:.4f}"])
            gt_row.append(_to_uint8(gt, scale=args.scale))
            print(f"  alpha={alpha:.2f}  PSNR={m['PSNR']:.2f}  dE00={m['dE00']:.3f}")
        else:
            print(f"  alpha={alpha:.2f}  done")

        pred_row.append(_to_uint8(pred, scale=args.scale, label=de00))

        if args.save_frames:
            Image.fromarray(_to_uint8(pred, scale=1)).save(
                os.path.join(args.save_dir,
                             f"cglut_{img_name}_{args.lut_a}_to_{args.lut_b}_a{alpha:.2f}.png"))

    # alpha-sweep strip (predictions, and GT above them when metrics are on)
    strip_rows = [_hstack_with_gaps(pred_row)]
    if gt_row:
        strip_rows.insert(0, _hstack_with_gaps(gt_row))
    gap = np.full((8, strip_rows[0].shape[1], 3), 255, dtype=np.uint8)
    strip = strip_rows[0] if len(strip_rows) == 1 else np.vstack([strip_rows[0], gap, strip_rows[1]])
    strip_path = os.path.join(args.save_dir,
                              f"strip_{img_name}_{args.lut_a}_to_{args.lut_b}.png")
    Image.fromarray(strip).save(strip_path)
    print(f"strip saved: {strip_path}")

    if csv_rows:
        csv_path = os.path.join(args.save_dir, f"{img_name}_{args.lut_a}_{args.lut_b}.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['LUT_A', 'LUT_B', 'Image', 'Alpha',
                        'PSNR', 'SSIM', 'dE00', 'dE76', 'RMSE', 'MAE', 'LPIPS'])
            w.writerows(csv_rows)
        print(f"metrics saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cGLUT LUT interpolation (single image)')
    parser.add_argument("--image", default="./dataset/images/a4501-DSC_0354.png", help="input image")
    parser.add_argument("--lut", default="./dataset/cube_files/7luts",
                        help="directory of .cube files; sorted order defines the LUT indices")
    parser.add_argument("--pretrained_model", default="./pretrained_models/cglut/cglut_g32_e64_7styles_psnr49.58_de0.2411_L_SharedGeo.pth",
                        help="trained ConditionalGLUT .pth")
    parser.add_argument("--lut_a", type=int, default=0, help="LUT index at alpha=0")
    parser.add_argument("--lut_b", type=int, default=1, help="LUT index at alpha=1")
    parser.add_argument("--n_alpha", type=int, default=6, help="number of alpha steps in [0, 1]")
    parser.add_argument("--save_dir", default="./results/blend", help="output directory")
    parser.add_argument("--scale", type=int, default=2, help="downsample factor for the strip image")
    parser.add_argument("--metrics", action="store_true",
                        help="also compute the reference blend and PSNR/SSIM/dE/LPIPS metrics")
    parser.add_argument("--save_frames", action="store_true",
                        help="save every blended frame at full resolution")
    parser.add_argument("--random_seed", type=int, default=52)
    args = parser.parse_args()

    run(args)
