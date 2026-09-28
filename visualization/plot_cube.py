"""
plot_cube.py  -  Publication-quality LUT RGB-cube figure
==========================================================
Renders color transformation vectors from a .cube LUT file inside
a delimited unit RGB cube. Matches the GLUT cube visualisation style.

Usage:
    python plot_cube.py \
        --lut       path/to/lut.cube \
        --output    lut_cube.pdf \
        --step      4 \
        --elev      28  --azim  45
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.rcParams.update({
    'font.family'  : 'serif',
    'font.size'    : 11,
    'pdf.fonttype' : 42,
    'ps.fonttype'  : 42,
})

# Make the repo root (which holds apply_cube.py) importable regardless of the
# current working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.apply_cube import load_cube_file


# ---------------------------------------------------------------------------
# Cube geometry (shared with GLUT visualiser)
# ---------------------------------------------------------------------------

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
# LUT sampling
# ---------------------------------------------------------------------------

def sample_lut(lut: np.ndarray, lut_size: int, step: int):
    """Return (input_colors, delta) arrays, skipping near-zero displacements."""
    indices = np.arange(0, lut_size, step)
    rg, gg, bg = np.meshgrid(indices, indices, indices, indexing='ij')
    max_idx = lut_size - 1

    src = np.column_stack([
        rg.ravel() / max_idx,
        gg.ravel() / max_idx,
        bg.ravel() / max_idx,
    ])
    dst = np.column_stack([
        lut[rg.ravel(), gg.ravel(), bg.ravel(), 0],
        lut[rg.ravel(), gg.ravel(), bg.ravel(), 1],
        lut[rg.ravel(), gg.ravel(), bg.ravel(), 2],
    ])
    delta = dst - src

    # Drop trivial (identity) entries
    mag = np.linalg.norm(delta, axis=1)
    mask = mag > 1e-4
    return src[mask], delta[mask], mag[mask]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_lut(lut_path: str, args):

    lut, lut_size, _domain_min, _domain_max = load_cube_file(lut_path)
    src, delta, mag = sample_lut(lut, lut_size, step=args.step)

    fig = plt.figure(figsize=(args.figsize, args.figsize), dpi=args.dpi)
    ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.set_axis_off()

    ax.set_xlim(-0.08, 1.32)
    ax.set_ylim(-0.08, 1.32)
    ax.set_zlim(-0.08, 1.32)
    ax.set_box_aspect([1, 1, 1])

    # ── Cube faces
    ax.add_collection3d(Poly3DCollection(
        CUBE_FACES,
        facecolor=(0.94, 0.94, 0.97, 0.14),
        edgecolor='none', zorder=0))

    # ── Cube edges
    for a, b in CUBE_EDGE_IDX:
        va, vb = CUBE_VERTICES[a], CUBE_VERTICES[b]
        ax.plot(*zip(va, vb), color='#aaaaaa', linewidth=0.85, zorder=1)

    # ── RGB axis arrows (overextend the cube)
    # ── RGB axis arrows — strictly within [0, 1]
    AXES = [
        (0, '#cc3333', 'R', ( 1.05,  0.00,  0.00)),
        (1, '#339933', 'G', (-0.04,  1.05,  0.00)),
        (2, '#3355cc', 'B', ( 0.00, -0.04,  1.05)),
    ]
    for dim, col, lbl, lpos in AXES:
        start = [0, 0, 0]
        vec   = [1 if d == dim else 0 for d in range(3)]
        ax.quiver(*start, *vec, color=col, linewidth=1.8,
                  arrow_length_ratio=0.10, zorder=5)
        ax.text(*lpos, lbl, color=col, fontsize=14,
                fontweight='bold', ha='center', va='center', zorder=6)

    # ── Input color points (optional)
    if args.show_points:
        ax.scatter(
            src[:, 0], src[:, 1], src[:, 2],
            c=src,                          # color = input RGB
            s=8,
            alpha=0.7,
            edgecolors='none',
            zorder=4,
            depthshade=False,
        )

    # ── LUT transformation vectors
    # Normalise magnitudes for alpha scaling
    mag_norm = mag / (mag.max() + 1e-8)

    # Scale arrow length for readability
    arrow_scale = args.arrow_scale / (lut_size / args.step)

    for i in range(len(src)):
        color = np.clip(src[i], 0, 1)          # arrow color = input RGB
        alpha = float(np.clip(0.25 + 0.65 * mag_norm[i], 0, 0.9))
        ax.quiver(
            src[i, 0], src[i, 1], src[i, 2],
            delta[i, 0] * arrow_scale,
            delta[i, 1] * arrow_scale,
            delta[i, 2] * arrow_scale,
            color=(*color, alpha),
            linewidth=0.6,
            arrow_length_ratio=0.30,
            zorder=3,
        )

    ax.view_init(elev=args.elev, azim=args.azim)

    plt.tight_layout(pad=0)
    plt.savefig(args.output, dpi=args.dpi,
                bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close()
    print(f"Saved -> {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Publication-quality LUT RGB-cube visualisation')
    parser.add_argument('--lut',          type=str,   required=True,
                        help='.cube LUT file')
    parser.add_argument('--output',       type=str,   default='lut_cube.pdf')
    parser.add_argument('--step',         type=int,   default=4,
                        help='Sample every nth LUT entry (lower = denser)')
    parser.add_argument('--arrow_scale',  type=float, default=1.0,
                        help='Global scale factor for arrow lengths')
    parser.add_argument('--elev',         type=float, default=28)
    parser.add_argument('--azim',         type=float, default=45)
    parser.add_argument('--figsize',      type=float, default=5.0)
    parser.add_argument('--show_points',  action='store_true', default=False,
                        help='Overlay a scatter of input colors at their RGB positions')
    parser.add_argument('--dpi',          type=int,   default=100)
    args = parser.parse_args()

    visualize_lut(args.lut, args)