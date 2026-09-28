"""
Generate hald images: a 2D image whose pixels enumerate every RGB value of an
identity 3D LUT.  Feeding such an image through a real 3D LUT (see
``apply_cube.py``) yields an (input, target) pair for training GLUT / cGLUT.
See https://3dlutcreator.com/3d-lut-creator---materials-and-luts.html
"""

import os
import sys

import numpy as np

# Make the repo root (which holds the sibling `utils` package) importable
# regardless of how this script is invoked (e.g. `python data/generate_hald.py`
# sets sys.path[0] to `data/`, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import save_rgb


def generate_hald(step=2):
    rgb_map = []
    rgb_map_ex = []

    for b in range(0, 256):
        for g in range(0, 256):
            for r in range(0, 256):
                if r % step == 0 and g % step == 0 and b % step == 0:
                    rgb_map.append([b, g, r])
                elif step != 1:
                        rgb_map_ex.append([b, g, r])


    # Double-check that all the values are there.
    rgb_map = np.array(rgb_map)
    print("Number of unique RGB intensities", rgb_map.shape, 4096*4096-rgb_map.shape[0])
    rgb_img = np.reshape(rgb_map, (4096//step, 4096//(step**2), 3))
    print(f"Image size is {4096//step}x{4096//(step**2)}x{3}")
    save_rgb(rgb_img, f"hald_step{step}.png")

    if step != 1:
        rgb_map_ex = np.array(rgb_map_ex)
        print("Number of unique RGB intensities", rgb_map_ex.shape, 4096*4096-rgb_map_ex.shape[0])
        rgb_img_ex = np.reshape(rgb_map_ex, (rgb_map_ex.shape[0]//4096, 4096, 3 ))
        print(f"Image size is {rgb_map_ex.shape[0]//4096}x{4096}x{3}")
        save_rgb(rgb_img_ex, f"hald_step{step}_exclude.png")

    # The image looks weird but it works!

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='generate hald image')
    parser.add_argument("--step", type=int, default=1, help="sampling step of 256 cube")
    args = parser.parse_args()

    generate_hald(step=args.step)
