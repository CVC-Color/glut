"""
plot_glut.py  —  Publication-quality GLUT RGB-cube figure
===========================================================
Renders Gaussian ellipsoids inside a delimited unit RGB cube.
White background, serif font, PDF output — ready for paper subfigures.

Usage:
    python plot_glut.py \
        --pretrained_model  path/to/model.pth \
        --num_gaussians     32 \
        --output            glut_cube.pdf \
        --elev              28  --azim  45
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

# Make the repo root (which holds train_glut.py) importable regardless of the
# current working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_glut import GLUT3D

matplotlib.rcParams.update({
    'font.family'  : 'serif',
    'font.size'    : 11,
    'pdf.fonttype' : 42,   # embed fonts for publishers
    'ps.fonttype'  : 42,
})


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def ellipsoid_mesh(center, cov, n_u=40, n_v=22, scale=1.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    axes = np.sqrt(eigvals) * scale
    u  = np.linspace(0, 2 * np.pi, n_u)
    v  = np.linspace(0, np.pi,     n_v)
    ug, vg = np.meshgrid(u, v)
    xs = axes[0] * np.cos(ug) * np.sin(vg)
    ys = axes[1] * np.sin(ug) * np.sin(vg)
    zs = axes[2] * np.cos(vg)
    pts = eigvecs @ np.stack([xs.ravel(), ys.ravel(), zs.ravel()])
    X = pts[0].reshape(xs.shape) + center[0]
    Y = pts[1].reshape(ys.shape) + center[1]
    Z = pts[2].reshape(zs.shape) + center[2]
    return X, Y, Z


CUBE_VERTICES = np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
CUBE_EDGE_IDX = [
    (0,1),(0,2),(0,4),(1,3),(1,5),
    (2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
]
CUBE_FACES = [
    [[0,0,0],[1,0,0],[1,1,0],[0,1,0]],
    [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
    [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
    [[0,1,0],[1,1,0],[1,1,1],[0,1,1]],
    [[0,0,0],[0,1,0],[0,1,1],[0,0,1]],
    [[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_glut(model, args):

    with torch.no_grad():
        positions = model.positions.detach().cpu().numpy()
        covs      = model.get_covariance_matrix().detach().cpu().numpy()
        opacities = torch.sigmoid(
            model.opacities_logit).detach().cpu().numpy().flatten()

    fig = plt.figure(figsize=(args.figsize, args.figsize), dpi=args.dpi)
    ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.set_axis_off()

    # Cube faces
    ax.add_collection3d(Poly3DCollection(
        CUBE_FACES,
        facecolor=(0.94, 0.94, 0.97, 0.14),
        edgecolor='none', zorder=0))

    # Cube edges
    for a, b in CUBE_EDGE_IDX:
        va, vb = CUBE_VERTICES[a], CUBE_VERTICES[b]
        ax.plot(*zip(va, vb), color='#aaaaaa', linewidth=0.85,
                linestyle='-', zorder=1)

    # RGB axis arrows — labels placed at explicit absolute coords
    # G and B sit close to the cube face; R has more room to the right
    AXES = [
        ((1,0,0), '#cc3333', 'R', (1.24, 0,    0   )),
        ((0,1,0), '#339933', 'G', (0,    1.08, 0   )),
        ((0,0,1), '#3355cc', 'B', (0,    0,    1.08)),
    ]
    for vec, col, lbl, lpos in AXES:
        ax.quiver(0, 0, 0, *vec, color=col, linewidth=1.8,
                  arrow_length_ratio=0.12, zorder=5)
        ax.text(*lpos, lbl, color=col, fontsize=14,
                fontweight='bold', ha='center', va='center', zorder=6)

    # Gaussians — back-to-front
    view_vec = np.array([
        np.cos(np.radians(args.elev)) * np.cos(np.radians(args.azim)),
        np.cos(np.radians(args.elev)) * np.sin(np.radians(args.azim)),
        np.sin(np.radians(args.elev)),
    ])
    order = np.argsort(positions @ view_vec)

    ls = matplotlib.colors.LightSource(azdeg=260, altdeg=45)

    for i in order:
        op = float(opacities[i])
        if op < args.opacity_thresh:
            continue

        c = positions[i]
        base     = np.clip(c, 0, 1)
        face_rgb = np.clip(base * 0.62 + 0.32, 0, 1)
        wire_rgb = np.clip(base * 0.38 + 0.05, 0, 1)
        dot_rgb  = np.clip(base * 0.48,         0, 1)

        X, Y, Z = ellipsoid_mesh(c, covs[i], scale=args.ellipsoid_scale)

        ax.plot_surface(X, Y, Z,
                        color=(*face_rgb, op * 0.30),
                        edgecolor='none', linewidth=0,
                        antialiased=True, shade=True,
                        lightsource=ls, zorder=3)

        ax.plot_wireframe(X, Y, Z,
                          color=(*wire_rgb, min(op * 0.55, 0.50)),
                          linewidth=0.35, rstride=12, cstride=12,
                          antialiased=True, zorder=4)

        ax.scatter(*c, color=(*dot_rgb, 0.95),
                   s=18, zorder=6, depthshade=False)

    pad = 0.04
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.set_zlim(-pad, 1 + pad)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=args.elev, azim=args.azim)

    plt.tight_layout(pad=0)
    plt.savefig(args.output, dpi=args.dpi,
                bbox_inches='tight', facecolor='white', pad_inches=0.06)
    plt.close()
    print(f"Saved -> {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str,   required=True)
    parser.add_argument('--num_gaussians',    type=int,   default=32)
    parser.add_argument('--residual',         type=bool,  default=True)
    parser.add_argument('--output',           type=str,   default='glut_cube.pdf')
    parser.add_argument('--elev',             type=float, default=28)
    parser.add_argument('--azim',             type=float, default=45)
    parser.add_argument('--figsize',          type=float, default=5.0)
    parser.add_argument('--dpi',              type=int,   default=100)
    parser.add_argument('--opacity_thresh',   type=float, default=0.05)
    parser.add_argument('--ellipsoid_scale',  type=float, default=0.6)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GLUT3D(
        num_gaussians=args.num_gaussians,
        residual=args.residual,
        logger=None
    ).to(device)
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    visualize_glut(model, args)