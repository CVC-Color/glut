"""
Load ``.cube`` 3D LUTs and apply them to RGB points with trilinear interpolation.

``CubeFileTransform`` is used as the "ground-truth" colour transform when
generating hald-image pairs and when evaluating fitted models. Input values are
normalised by the LUT's own ``DOMAIN_MIN``/``DOMAIN_MAX`` before the lookup
(a no-op for the common case of the default ``[0, 1]`` domain).

Run as a script to render a directory of ``.cube`` LUTs onto a hald image,
producing the training-data layout ``MultiHaldBatch``/``SingleHaldBatch`` expect
(one PNG per LUT plus a copy of the hald image as ``Original_Image.png``).
"""

import os
import shutil

import cv2
import numpy as np
import torch


def load_cube_file(file_path):
    """Parse a ``.cube`` 3D LUT file.

    Args:
        file_path: path to the ``.cube`` file.

    Returns:
        lut_data:   [size, size, size, 3] float array of output RGB.
        size:       LUT resolution per axis.
        domain_min: length-3 list, input-range minimum.
        domain_max: length-3 list, input-range maximum.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    data_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        key = parts[0].lower()
        if key == 'lut_1d_size':
            pass  # 1D LUTs are not supported
        elif key == 'lut_3d_size':
            size = int(parts[1])
        elif key == 'domain_min':
            domain_min = [float(parts[1]), float(parts[2]), float(parts[3])]
        elif key == 'domain_max':
            domain_max = [float(parts[1]), float(parts[2]), float(parts[3])]
        else:
            try:
                values = [float(x) for x in parts]
                if len(values) >= 3:
                    data_lines.append(values[:3])
            except ValueError:
                continue

    if size is None:
        size = int(round(len(data_lines) ** (1 / 3)))
        print(f"WARNING: LUT_3D_SIZE not found, inferred size={size} from the row count")

    lut_data = np.array(data_lines).reshape(size, size, size, 3)
    return lut_data, size, domain_min, domain_max


class CubeFileTransform:
    """Callable colour transform backed by a ``.cube`` 3D LUT."""

    def __init__(self, cube_file_path, interpolation='trilinear'):
        """
        Args:
            cube_file_path: path to the ``.cube`` file.
            interpolation: 'trilinear' or 'nearest'.
        """
        self.cube_file_path = cube_file_path
        self.interpolation = interpolation
        self.lut_data, self.size, domain_min, domain_max = load_cube_file(cube_file_path)
        self.domain_min = np.asarray(domain_min, dtype=np.float32)
        self.domain_max = np.asarray(domain_max, dtype=np.float32)

    def _to_domain(self, rgb):
        """Map ``rgb`` from ``[domain_min, domain_max]`` to ``[0, 1]`` (clipped)."""
        rgb = (rgb - self.domain_min) / (self.domain_max - self.domain_min)
        return np.clip(rgb, 0.0, 1.0)

    def _trilinear_interpolation(self, rgb):
        """Vectorised trilinear interpolation. ``rgb``: [N, 3] in [0, 1] -> [N, 3]."""
        rgb = self._to_domain(rgb)
        rgb_scaled = rgb * (self.size - 1)                      # [N, 3]
        idx0 = np.floor(rgb_scaled).astype(np.int32)            # [N, 3]
        idx1 = np.minimum(idx0 + 1, self.size - 1)              # [N, 3]
        frac = rgb_scaled - idx0                                # [N, 3]

        x0, y0, z0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        x1, y1, z1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        fx, fy, fz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]   # keep dims for broadcast

        # 8 corner colours.
        c000 = self.lut_data[x0, y0, z0]; c001 = self.lut_data[x0, y0, z1]
        c010 = self.lut_data[x0, y1, z0]; c011 = self.lut_data[x0, y1, z1]
        c100 = self.lut_data[x1, y0, z0]; c101 = self.lut_data[x1, y0, z1]
        c110 = self.lut_data[x1, y1, z0]; c111 = self.lut_data[x1, y1, z1]

        # Interpolate along z, then y, then x.
        c00 = c000 * (1 - fz) + c001 * fz
        c01 = c010 * (1 - fz) + c011 * fz
        c10 = c100 * (1 - fz) + c101 * fz
        c11 = c110 * (1 - fz) + c111 * fz
        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy
        return c0 * (1 - fx) + c1 * fx

    def _nearest_interpolation(self, rgb):
        """Nearest-neighbour lookup. ``rgb``: [N, 3] in [0, 1] -> [N, 3]."""
        rgb = self._to_domain(rgb)
        idx = np.clip(np.round(rgb * (self.size - 1)).astype(np.int32), 0, self.size - 1)
        return self.lut_data[idx[:, 0], idx[:, 1], idx[:, 2]]

    def __call__(self, rgb_tensor):
        """Apply the LUT.

        Args:
            rgb_tensor: torch.Tensor [N, 3] in [0, 1] (BGR channel order in, as the
                LUTs were authored for OpenCV-loaded images).
        Returns:
            torch.Tensor [N, 3] in [0, 1] on the same device.
        """
        rgb_tensor = rgb_tensor[:, [2, 1, 0]]

        rgb_np = rgb_tensor.cpu().numpy() if torch.is_tensor(rgb_tensor) else rgb_tensor
        rgb_np = np.clip(rgb_np, 0, 1)

        if self.interpolation == 'trilinear':
            output_np = self._trilinear_interpolation(rgb_np)
        else:
            output_np = self._nearest_interpolation(rgb_np)
        output_np = np.clip(output_np, 0, 1)

        return torch.FloatTensor(output_np).to(rgb_tensor.device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render .cube LUTs onto a hald image")
    parser.add_argument("--lut_dir", required=True, help="directory of .cube files")
    parser.add_argument("--hald_image", required=True,
                        help="identity hald image (see generate_hald.py)")
    parser.add_argument("--out_dir", required=True, help="output directory")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.hald_image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = img_rgb.shape
    points_tensor = torch.from_numpy(img_rgb.reshape(-1, 3))

    os.makedirs(args.out_dir, exist_ok=True)
    for lut_file in sorted(os.listdir(args.lut_dir)):
        if not lut_file.lower().endswith('.cube'):
            continue
        print(f"Processing LUT: {lut_file}")
        out_name = (os.path.splitext(lut_file)[0]
                    .replace(' ', '_').replace('(', '').replace(')', '') + ".png")
        transform = CubeFileTransform(os.path.join(args.lut_dir, lut_file), 'trilinear')
        out_rgb = transform(points_tensor).reshape(h, w, 3).numpy()
        out_bgr = cv2.cvtColor((out_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.out_dir, out_name), out_bgr)

    # SingleHaldBatch / MultiHaldBatch expect the input hald image alongside the
    # per-LUT outputs, named Original_Image.png.
    shutil.copyfile(args.hald_image, os.path.join(args.out_dir, 'Original_Image.png'))
