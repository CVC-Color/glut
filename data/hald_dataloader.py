import os
import cv2
import numpy as np

import torch
import torchvision

from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose

from utils.utils import load_img
from utils.metrics import np_psnr



# FOR NILUT AND ENNELUT 

class ConditionalLUTFitting(Dataset):
    """
    Dataset for Conditional LUT training
    Reads multiple target images from a folder, each representing a different LUT style
    """
    def __init__(self, out_img_folder, resize=False):
        """
        Args:
            out_img_folder: folder containing multiple target Hald images (different LUT styles)
            resize: whether to resize images
        """
        super().__init__()

        if resize:
            self.resize = Compose([Resize(self.input_img.shape[0] // 2, 
                                         interpolation=torchvision.transforms.InterpolationMode.NEAREST)])
        
        
        # Load input image (common for all styles)
        input_img = load_img(os.path.join(out_img_folder, "Original_Image.png"))
        input_tensor = torch.from_numpy(input_img)
        if resize:
            input_tensor = self.resize(input_tensor)
        self.intensities = input_tensor.reshape((-1, 3))
        
        
        # Get all image files in the target folder
        target_files = os.listdir(out_img_folder)
        target_files = [name for name in target_files if name.endswith('.png')]
        target_files.sort()  # Ensure consistent ordering

        # Load all target images from the folder
        self.style_names = []
        self.target_tensors = []

        for target_file in target_files:
            # Skip Original_Image.png if it exists in the target folder
            if os.path.basename(target_file) == "Original_Image.png":
                continue
                
            target_img = load_img(os.path.join(out_img_folder, target_file))
            
            # Verify dimensions match
            assert input_img.shape == target_img.shape, \
                f"Shape mismatch: input {input_img.shape} vs target {target_img.shape}"
            assert (input_img.max() <= 1) and (target_img.max() <= 1), \
                "Images must be in range [0, 1]"
            
            target_tensor = torch.from_numpy(target_img)
            if resize:
                target_tensor = self.resize(target_tensor)
            target_tensor = target_tensor.reshape((-1, 3))
            self.target_tensors.append(target_tensor)
        
            self.style_names.append(os.path.basename(target_file)[:-4])  # Remove extension
        
        self.num_styles = len(self.target_tensors)
        print(f"Loaded {self.num_styles} styles: {self.style_names}")
        
        if self.num_styles == 0:
            raise ValueError(f"No target images found in {out_img_folder}")
        
        self.shape = input_tensor.shape
        self.dim = self.intensities.shape
        self.num_pixels = self.dim[0]
        
        # Clean up
        del input_tensor, target_tensor, target_img
    
    def __len__(self):
        """Return number of (pixel, style) pairs"""
        return self.num_pixels * self.num_styles
    
    def __getitem__(self, idx):
        """
        Returns a single (pixel, style) training pair
        Each item is: (input_rgb + one_hot, target_rgb)
        """
        pixel_idx = idx // self.num_styles
        style_idx = idx % self.num_styles
        
        # Get input RGB for this pixel
        input_rgb = self.intensities[pixel_idx]
        
        # Get target RGB for this style
        target_rgb = self.target_tensors[style_idx][pixel_idx]
        
        # Create one-hot encoding for style
        one_hot = torch.zeros(self.num_styles)
        one_hot[style_idx] = 1.0
        
        # Conditional input = RGB + one_hot
        conditional_input = torch.cat([input_rgb, one_hot])
        
        return conditional_input, target_rgb


class ValidationDataset(Dataset):
    """
    Dataset for validation that returns batches of pixels for a specific style
    """
    def __init__(self, intensities, targets, style_idx, num_styles, batch_size=10000):
        """
        Args:
            intensities: RGB input tensor [num_pixels, 3]
            targets: target RGB tensor [num_pixels, 3]
            style_idx: style index for this dataset
            num_styles: total number of styles
            batch_size: batch size for iteration
        """
        self.intensities = intensities
        self.targets = targets
        self.style_idx = style_idx
        self.num_styles = num_styles
        self.batch_size = batch_size
        self.num_pixels = intensities.shape[0]
        
        # Pre-compute one-hot for this style (will be expanded in batch)
        self.one_hot_base = torch.zeros(num_styles)
        self.one_hot_base[style_idx] = 1.0
    
    def __len__(self):
        return (self.num_pixels + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_pixels)
        
        batch_intensities = self.intensities[start_idx:end_idx]
        batch_targets = self.targets[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # Create one-hot for this batch
        batch_one_hot = self.one_hot_base.unsqueeze(0).repeat(batch_size_actual, 1)
        
        return batch_intensities, batch_targets, batch_one_hot



# FOR GLUT  
        
class SingleHaldBatch(Dataset):
    def __init__(self, inp_img, out_img, resize=False):
        super().__init__()
        
        img = load_img(inp_img)
        lut = load_img(out_img)

        self.error = np_psnr(img, lut)
        
        assert img.shape == lut.shape
        assert (img.max() <= 1) and (lut.max() <= 1)
        
        if resize:
            self.resize = Compose([Resize(img.shape[0] // 2, interpolation=torchvision.transforms.InterpolationMode.NEAREST)])
            
        
        img = torch.from_numpy(img)
        lut = torch.from_numpy(lut)
        
        if resize: img = self.resize(img)
        if resize: lut = self.resize(lut)
            
        self.shape = img.shape
        
        self.intensities = img.reshape((img.shape[0]*img.shape[1],3))
        self.outputs     = lut.reshape((img.shape[0]*img.shape[1],3))
        self.dim         = self.intensities.shape
        self.length      = self.intensities.shape[0]
            
        del img, lut


    def __dim__(self):
        return self.dim
    
    def __shape__(self):
        return self.shape
    
    def __len__(self):
        return self.intensities.shape[0]
    
    def __error__(self):
        return self.error

    def __getitem__(self, idx):
        
        intensities = self.intensities[idx, :]
        outputs = self.outputs[idx, :]

        return intensities, outputs


# FOR CGLUT 

class MultiHaldBatch(Dataset):
    """
    The order of the target images must be: ground-truth 3D LUT outputs (the first <nluts> elements in the list), following by gt blending results.
    """

    def __init__(self, out_img_folder, resize=False):
        """
        Args:
            out_img_folder: folder containing multiple target Hald images (different LUT styles)
            resize: whether to resize images
        """
        super().__init__()

        if resize:
            self.resize = Compose([Resize(self.input_img.shape[0] // 2, 
                                         interpolation=torchvision.transforms.InterpolationMode.NEAREST)])
        
        
        # Load input image (common for all styles)
        input_img = load_img(os.path.join(out_img_folder, "Original_Image.png"))
        input_tensor = torch.from_numpy(input_img)
        if resize:
            input_tensor = self.resize(input_tensor)
        self.intensities = input_tensor.reshape((-1, 3))
        
        
        # Get all image files in the target folder
        target_files = os.listdir(out_img_folder)
        target_files = [name for name in target_files if name.endswith('.png')]
        target_files.sort()  # Ensure consistent ordering

        # Load all target images from the folder
        self.style_names = []
        self.target_tensors = []

        for target_file in target_files:
            # Skip Original_Image.png if it exists in the target folder
            if os.path.basename(target_file) == "Original_Image.png":
                continue
            # print(target_file)
            target_img = load_img(os.path.join(out_img_folder, target_file))
            
            # Verify dimensions match
            assert input_img.shape == target_img.shape, \
                f"Shape mismatch: input {self.input_img.shape} vs target {target_img.shape}"
            assert (input_img.max() <= 1) and (target_img.max() <= 1), \
                "Images must be in range [0, 1]"
            
            target_tensor = torch.from_numpy(target_img)
            if resize:
                target_tensor = self.resize(target_tensor)
            target_tensor = target_tensor.reshape((-1, 3))
            self.target_tensors.append(target_tensor)
        
            self.style_names.append(os.path.basename(target_file)[:-4])  # Remove extension
        
        self.num_luts = len(self.target_tensors)
        print(f"Loaded {self.num_luts} styles: {self.style_names}")
        
        if self.num_luts == 0:
            raise ValueError(f"No target images found in {out_img_folder}")
        
        # self.shape = input_tensor.shape
        self.dim = self.intensities.shape
        self.num_pixels = self.dim[0]
        
        # Clean up
        del input_tensor, target_tensor, target_img
    
    def __len__(self):
        """Return number of (pixel, style) pairs"""
        return self.num_pixels * self.num_luts
    
    def __getitem__(self, idx):
        """
        Returns a single (pixel, style) training pair
        Each item is: (input_rgb + one_hot, target_rgb)
        """
        pixel_idx = idx // self.num_luts
        style_idx = idx % self.num_luts
        
        # Get input RGB for this pixel
        input_rgb = self.intensities[pixel_idx]
        
        # Get target RGB for this style
        target_rgb = self.target_tensors[style_idx][pixel_idx]

        style_idx = torch.tensor([style_idx], dtype=torch.long)
        
        
        return input_rgb, target_rgb, style_idx



class MultiImagePairs(Dataset):
    def __init__(self, inp_dir, out_dir, shape=(1024, 2048)):
        super(MultiImagePairs, self).__init__()

        images = os.listdir(inp_dir)
        images = [name for name in images if name.endswith('.png')]
        images.sort()

        inp_imgs = []
        out_imgs = []
        indices = []
        self.errors = []

        index = 0
        for image_name in images:
            inp_img = load_img(os.path.join(inp_dir, image_name))
            inp_img = cv2.resize(inp_img, shape, interpolation=cv2.INTER_NEAREST)
            n_points = inp_img.shape[0] * inp_img.shape[1]
            inp_img = inp_img.reshape((n_points, 3))
            inp_imgs.append(inp_img)

            out_img =  load_img(os.path.join(out_dir, image_name))
            out_img = cv2.resize(out_img, shape, interpolation=cv2.INTER_NEAREST)
            if not out_img.shape[0] * out_img.shape[1] == n_points:
                raise RuntimeError("The input and output dimensions are mismatched...")
            out_img = out_img.reshape((n_points, 3))
            out_imgs.append(out_img)

            self.errors.append(np_psnr(inp_img, out_img))
            indices.append(np.repeat(index, n_points, axis=0))
            index += 1

        
        self.num_luts = len(out_imgs)
        self.dim = n_points
        self.inp_imgs = np.concatenate(inp_imgs, axis=0)
        self.out_luts = np.concatenate(out_imgs, axis=0)
        self.indices = np.concatenate(indices, axis=0).reshape(-1, 1)
 
        
    def __len__(self):
        return self.num_luts * self.dim
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.inp_imgs[idx, :])
        lut = torch.from_numpy(self.out_luts[idx, :])
        lut_idx = torch.from_numpy(self.indices[idx])
        
        return img, lut, lut_idx