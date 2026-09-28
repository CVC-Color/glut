"""
Batch-test trained LUT models on real images.

For a directory of (input, target) image pairs and a directory of checkpoints,
match each image to its model, run inference, and log PSNR / SSIM / dE metrics.
Supports GLUT / cGLUT.
"""

import csv
import glob
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from data.apply_cube import CubeFileTransform
from blend_lut import load_cglut
from utils.metrics import calc_deltaE, calc_psnr
from train_glut import GLUT3D


class LUTModelTester:
    """
    Unified LUT-model tester (GLUT / cGLUT)
    uses the LUT transforms from apply_cube.py
    """

    # Upper bound for the deltaE76 heatmap colormap in save_visualization; values
    # above this are clipped so a few outlier pixels don't wash out the scale.
    DELTA_E_VIS_MAX = 10.0

    def __init__(self,
                 model_type: str,  # 'GLUT' / 'CGLUT'
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_type: one of 'GLUT' / 'CGLUT'
            device: compute device
        """
        self.model_type = model_type
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Set up logging.
        self.logger = logging.getLogger('LUTTester')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.logger.info(f"Tester initialised. model_type={model_type}, device={device}")

    def load_model(self, model_path: str):
        """
        Load a checkpoint for the configured model type. Architecture is read
        directly from the checkpoint (saved hyper-params / weight shapes)
        rather than from CLI flags.

        Args:
            model_path: path to the .pth file

        Returns:
            model: the loaded model
        """
        if self.model_type == 'CGLUT':
            # load_cglut already builds the model from the checkpoint's own
            # architecture metadata, loads the weights and calls .eval().
            return load_cglut(model_path, self.device)

        # GLUT
        checkpoint = torch.load(model_path, map_location=self.device)
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

        num_gaussians = checkpoint.get('num_gaussians') if isinstance(checkpoint, dict) else None
        num_gaussians = num_gaussians or sd['positions'].shape[0]
        residual = checkpoint.get('residual', True) if isinstance(checkpoint, dict) else True
        weight_norm = checkpoint.get('weight_norm', True) if isinstance(checkpoint, dict) else True

        model = GLUT3D(num_gaussians=num_gaussians, residual=residual, weight_norm=weight_norm)

        # Load the weights.
        try:
            model.load_state_dict(sd)
        except Exception as e:
            self.logger.error(f"  failed to load model: {e}")
            raise

        model = model.to(self.device)
        model.eval()

        return model

    def process_image_with_model(self, model: nn.Module, image_path: str, condition_idx: int) -> np.ndarray:
        """Run the model on one image."""
        # Read the image.
        image = Image.open(image_path).convert('RGB')

        # To tensor.
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Flatten to [H*W, 3].
        h, w = image_tensor.shape[2:]
        pixels = image_tensor.permute(0, 2, 3, 1).reshape(-1, 3)

        # Apply the model.
        with torch.no_grad():
            if self.model_type == 'GLUT':
                output_pixels = model(pixels)
            elif self.model_type == 'CGLUT':
                lut_idx = np.repeat(condition_idx, h*w , axis=0)
                lut_idx = torch.from_numpy(lut_idx).to(self.device)
                output_pixels = model(pixels, lut_idx)

        # Reshape back to an image.
        output_image = output_pixels.reshape(1, h, w, 3).permute(0, 3, 1, 2)
        output_image = output_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return np.clip(output_image, 0, 1)

    def process_image_with_lut(self, lut_transform: CubeFileTransform, image_path: str) -> np.ndarray:
        """
        Run a .cube LUT on one image via CubeFileTransform.

        Args:
            lut_transform: a CubeFileTransform
            image_path: path to the image

        Returns:
            the transformed image [H, W, 3]
        """
        # Read the image.
        image = Image.open(image_path).convert('RGB')

        # To tensor.
        image_tensor = self.transform(image)  # [3, H, W]

        # Flatten to [H*W, 3].
        h, w = image_tensor.shape[1:]
        pixels = image_tensor.view(3, -1).t()  # [H*W, 3]

        # Apply the LUT.
        output_pixels = lut_transform(pixels)  # [H*W, 3]

        # Reshape back to an image.
        output_image = output_pixels.view(h, w, 3).cpu().numpy()

        return np.clip(output_image, 0, 1)

    def calculate_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Image-quality metrics between two images."""
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        # calc_psnr requires matching dtypes.
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # PSNR
        psnr_value = calc_psnr(img1, img2)

        delta_e00 = calc_deltaE(img1, img2, method='CIE 2000', reduction='mean')
        delta_e76 = calc_deltaE(img1, img2, method='CIE 1976', reduction='none')

        # RMSE
        rmse = np.sqrt(np.mean((img1 - img2) ** 2))

        # MAE
        mae = np.mean(np.abs(img1 - img2))

        return {
            'PSNR': psnr_value,
            'dE00': delta_e00,
            'dE76': np.mean(delta_e76),
            'dE76_std':np.std(delta_e76),
            'RMSE': rmse,
            'MAE': mae,
            'SSIM': 0,
            'lpips': 0
        }

    def extract_lut_name_from_cube(self, cube_filename: str) -> str:
        """
        Extract the LUT name from a .cube file name.
        e.g. "LUT001.cube" -> "LUT001"
        """
        basename = os.path.basename(cube_filename)
        name_without_ext = os.path.splitext(basename)[0]
        return name_without_ext

    def find_matching_model(self, lut_name: str, model_files: List[str]) -> Optional[str]:
        """
        Find the checkpoint that matches a LUT name (GLUT only).

        Matching rule:
        - GLUT:  GLUT_{lut_name}_{num_gaussians}_psnr{psnr}_dE{deltaE}.pth,
          with any whitespace in {lut_name} normalized to underscores.
          e.g. GLUT_LUT01_After_Effects_LUTs_Contrast_32_psnr52.64_dE0.3678.pth
          matches "LUT01_After Effects LUTs_Contrast.cube"
        """
        lut_name_norm = re.sub(r'\s+', '_', lut_name)
        pattern = rf'GLUT_{re.escape(lut_name_norm)}_\d+_psnr[\d.]+_dE[\d.]+\.pth'

        for model_file in model_files:
            model_basename = os.path.basename(model_file)

            if re.match(pattern, model_basename, re.IGNORECASE):
                self.logger.info(f"  matched GLUT model: {model_basename}")
                return model_file

        return None

    def run_test(self,
                luts_dir: str,
                models_dir: str,
                test_images_dir: str,
                save_dir: str,
                output_csv: str = None,
                save_image: bool = False) -> Optional[List[Dict]]:
        """
        Run the full test.

        Args:
            luts_dir: folder of reference .cube files
            models_dir: folder of .pth checkpoints
            test_images_dir: folder of test images
            output_csv: output CSV path (auto-named if None)
            save_image: also write comparison images

        Returns:
            a list of per-image result dicts
        """
        # Auto-name the output file.
        if output_csv is None:
            output_csv = f'{self.model_type}_test_results.csv'

        # List the .cube files.
        lut_files = glob.glob(os.path.join(luts_dir, '*.cube'))
        lut_files.sort()

        # List the checkpoints.
        if self.model_type == 'CGLUT':
            model_files = models_dir
        else:
            model_files = glob.glob(os.path.join(models_dir, '*.pth'))
            model_files.sort()

        # List the test images.
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
        image_files.sort()

        self.logger.info("="*60)
        self.logger.info(f"model_type: {self.model_type}")
        self.logger.info(f"LUT dir: {luts_dir}")
        self.logger.info(f"  {len(lut_files)} LUT files")
        self.logger.info(f"model dir: {models_dir}")
        self.logger.info(f"  {len(model_files)} checkpoints")
        self.logger.info(f"test image dir: {test_images_dir}")
        self.logger.info(f"  {len(image_files)} test images")
        self.logger.info("="*60)

        # Result accumulator.
        all_results = []
        matched_count = 0

        # Process every LUT.
        for lut_idx, lut_file in enumerate(lut_files):
            lut_name = self.extract_lut_name_from_cube(lut_file)

            self.logger.info(f"\nLUT: {os.path.basename(lut_file)} (name: {lut_name})")

            # Find the matching checkpoint.
            if self.model_type == 'CGLUT':
                matching_model = model_files
            else:
                matching_model = self.find_matching_model(lut_name, model_files)

            if matching_model is None:
                self.logger.warning(f"  no {self.model_type} model matches LUT {lut_name}, skipping")
                continue

            matched_count += 1
            self.logger.info(f"  matched model: {os.path.basename(matching_model)}")

            # Build the reference LUT transform.
            try:
                lut_transform = CubeFileTransform(
                    cube_file_path=lut_file,
                    interpolation='trilinear'
                )
                self.logger.info(f"  LUT transform ready, size {lut_transform.size}^3")
            except Exception as e:
                self.logger.error(f"  failed to build LUT transform: {e}")
                continue

            # Load the model.
            try:
                model = self.load_model(matching_model)
                self.logger.info("  model loaded")
            except Exception as e:
                self.logger.error(f"  failed to load model: {e}")
                continue

            # Process every test image.
            for image_file in image_files:
                image_name = os.path.basename(image_file)
                self.logger.info(f"    image: {image_name}")

                try:
                    # 1. model output
                    model_output = self.process_image_with_model(model, image_file, lut_idx)

                    # 2. reference (LUT) output
                    lut_output = self.process_image_with_lut(lut_transform, image_file)

                    # Match the two output sizes.
                    if model_output.shape[:2] != lut_output.shape[:2]:
                        from skimage.transform import resize
                        model_output = resize(model_output, lut_output.shape[:2], preserve_range=True)

                    # 3. metrics
                    metrics = self.calculate_metrics(model_output, lut_output)

                    # Store the row.
                    result = {
                        'Model_Type': self.model_type,
                        'LUT_Name': os.path.basename(lut_file),
                        'LUT_ID': lut_name,
                        'Model_File': os.path.basename(matching_model),
                        'Image': image_name,
                        'PSNR': metrics['PSNR'],
                        'SSIM': metrics['SSIM'],
                        'dE00': metrics['dE00'],
                        'dE76': metrics['dE76'],
                        'dE76_std': metrics['dE76_std'],
                        'RMSE': metrics['RMSE'],
                        'MAE': metrics['MAE'],
                        'lpips': metrics['lpips']
                    }

                    all_results.append(result)

                    self.logger.info(f"      PSNR: {metrics['PSNR']:.2f}, DeltaE76: {metrics['dE76']:.2f}")

                    # Optional: save a comparison image.
                    if save_image and len(all_results) < 10:
                        self.save_visualization(
                            lut_output, model_output,  # order: reference first, model output second
                            lut_name, image_name,
                            output_dir=os.path.join(save_dir, "images")
                        )

                except Exception as e:
                    self.logger.error(f"      failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Write the CSV.
        if all_results:
            csv_path = os.path.join(save_dir, output_csv)
            fieldnames = list(all_results[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            self.logger.info(f"\nresults saved to: {output_csv}")
            self.logger.info(f"matched {matched_count}/{len(lut_files)} LUTs")

            # Print the summary.
            self.print_summary(all_results)

            return all_results
        else:
            self.logger.warning("\nno results produced")
            return None

    def save_visualization(self, lut_output: np.ndarray, model_output: np.ndarray,
                          lut_name: str, image_name: str, output_dir: str = './results/image'):
        """
        Save a side-by-side comparison image: [lut_output | model_output | per-pixel deltaE76 heatmap].

        Args:
            lut_output: reference (LUT) image
            model_output: model image
            lut_name: LUT name
            image_name: image name
        """
        os.makedirs(output_dir, exist_ok=True)

        lut_bgr = cv2.cvtColor((lut_output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        model_bgr = cv2.cvtColor((model_output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        h, w = lut_output.shape[:2]
        delta_e76 = calc_deltaE(lut_output, model_output, method='CIE 1976', reduction='none').reshape(h, w)
        delta_e76_norm = np.clip(delta_e76, 0, self.DELTA_E_VIS_MAX) / self.DELTA_E_VIS_MAX
        diff_bgr = cv2.applyColorMap((delta_e76_norm * 255).astype(np.uint8), cv2.COLORMAP_PARULA)

        combined = np.hstack([lut_bgr, model_bgr, diff_bgr])

        output_path = os.path.join(output_dir, f'{self.model_type}_{lut_name}_{image_name[:-4]}_compare.png')
        cv2.imwrite(output_path, combined)

        self.logger.info(f"      comparison saved: {output_path}")

    @staticmethod
    def _mean_std(values: List[float]):
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def print_summary(self, results: List[Dict]):
        """Print aggregate statistics."""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Summary - {self.model_type}")
        self.logger.info("="*60)

        # Overall statistics.
        psnr_mean, psnr_std = self._mean_std([r['PSNR'] for r in results])
        ssim_mean, ssim_std = self._mean_std([r['SSIM'] for r in results])
        de_mean, de_std = self._mean_std([r['dE76'] for r in results])

        self.logger.info(f"\nOverall ({len(results)} tests):")
        self.logger.info(f"  PSNR      : {psnr_mean:.2f} ± {psnr_std:.2f}")
        self.logger.info(f"  SSIM      : {ssim_mean:.4f} ± {ssim_std:.4f}")
        self.logger.info(f"  DeltaE    : {de_mean:.2f} ± {de_std:.2f}")

        # Per-LUT statistics.
        self.logger.info("\nPer-LUT:")
        by_lut = defaultdict(list)
        for r in results:
            by_lut[r['LUT_ID']].append(r)

        for lut_id in sorted(by_lut.keys()):
            rows = by_lut[lut_id]
            lut_psnr_mean, lut_psnr_std = self._mean_std([r['PSNR'] for r in rows])
            lut_de_mean, lut_de_std = self._mean_std([r['dE76'] for r in rows])
            self.logger.info(f"  {lut_id}:")
            self.logger.info(f"    PSNR: {lut_psnr_mean:.2f}±{lut_psnr_std:.2f}")
            self.logger.info(f"    DeltaE: {lut_de_mean:.2f}±{lut_de_std:.2f}")


if __name__ == "__main__":
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Test trained LUT models on images')
    parser.add_argument('--model_type', type=str, default="GLUT", choices=['GLUT', 'CGLUT'],
                        help='model type')
    parser.add_argument('--models_dir', type=str, default="./pretrained_models/glut/7luts",
                        help='directory of .pth checkpoints')
    parser.add_argument('--luts_dir', type=str, default="./dataset/cube_files/7luts",
                        help='directory of .cube files')
    parser.add_argument('--images_dir', type=str, default="./dataset/images" ,
                        help='directory of test images')
    parser.add_argument("--save_dir", help="Input RGB map as a hald image", default="./results/image", type=str)
    parser.add_argument('--output_csv', type=str, default=None,
                        help='output CSV name')
    parser.add_argument('--save_img', action="store_true",
                        help='also save comparison images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='compute device (cuda/cpu)')

    args = parser.parse_args()

    # Build the tester. Model architecture is read from each checkpoint at
    # load time, not passed in here.
    tester = LUTModelTester(
        model_type=args.model_type,
        device=args.device if torch.cuda.is_available() else 'cpu'
    )

    save_dir = os.path.join(args.save_dir, os.path.basename(args.models_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # Run.
    results = tester.run_test(
        luts_dir=args.luts_dir,
        models_dir=args.models_dir,
        test_images_dir=args.images_dir,
        save_dir=save_dir,
        output_csv=args.output_csv,
        save_image=args.save_img,
    )

    if results is not None:
        print("\nDone.")
    else:
        print("\nFailed.")
