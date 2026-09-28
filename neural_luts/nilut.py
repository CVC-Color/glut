

"""
NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement
https://github.com/mv-lab/nilut

Fit a complete 3D LUT into a simple NN.
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose
import os
import sys
import gc
import argparse
# from collections import defaultdict

# Make the repo root (which holds the sibling `utils` package) importable
# regardless of how this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NILUT utils
from utils.utils import load_img, clean_mem, count_parameters, setup_logging
from utils.metrics import calc_metrics_torch, np_psnr



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = True
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
start_time = None



class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3, out_features=3, res=True):
        super().__init__()
        
        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
        output = torch.clamp(output, 0.,1.)
        
        return output
    
    
class LUTFitting(Dataset):
    def __init__(self, inp_img, out_img, resize=False):
        super().__init__()
        
        img = load_img(inp_img)
        lut = load_img(out_img)
        
        self.error = np_psnr(img,lut)
        
        assert img.shape == lut.shape
        assert (img.max() <= 1) and (lut.max() <= 1)
        
        if resize:
            self.resize = Compose([Resize(img.shape[0] // 2, interpolation=torchvision.transforms.InterpolationMode.NEAREST)])
            
        # Convert images to pytorch tensors
        img = torch.from_numpy(img)
        if resize: img = self.resize(img)
        lut = torch.from_numpy(lut)
        if resize: lut = self.resize(lut)
            
        self.shape = img.shape
        
        #self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.intensities = img.reshape((img.shape[0]*img.shape[1],3))
        self.outputs     = lut.reshape((img.shape[0]*img.shape[1],3))
        self.dim         = self.intensities.shape
        del img, lut

    def __dim__(self):
        return self.dim
    
    def __shape__(self):
        return self.shape
    
    def __len__(self):
        return 1
    
    def __error__(self):
        return self.error

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.intensities, self.outputs
    
    

def main(inp_path, out_path, inp_path_val, out_path_val, total_steps, lut_size, save_dir):
    """
    Fit a professional 3D LUT into a simple coordinate-based MLP.
    Complete tutorial at: https://github.com/mv-lab/nilut

    - inp_path: Input RGB map as a hald image
    - out_path: Enhanced RGB map as a hald image, after using the desired 3D LUT

    """
    logger = setup_logging(
        log_dir=save_dir,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )

    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Start NILUT {lut_size} fitting with")
    logger.info(f"Input hald image : {inp_path}")
    logger.info(f"Target hald image: {out_path}")

    # Define the dataloader
    lut_images = LUTFitting(inp_path, out_path)
    dataloader = DataLoader(lut_images, batch_size=1, pin_memory=True, num_workers=0)
    
    lut_images_val = LUTFitting(inp_path_val, out_path_val)
    dataloader_val = DataLoader(lut_images_val, batch_size=1, pin_memory=True, num_workers=0)
    
    img_size = lut_images.shape
    img_size_val = lut_images_val.shape
    logger.info(f"Dataloader ready, image size: {img_size}")
    
    # Define the model
    lut_model = NILUT(in_features=3, out_features=3, hidden_features=lut_size[0], hidden_layers=lut_size[1], res=True)
    lut_model.cuda()
    opt = torch.optim.Adam(lr=1e-3, params=lut_model.parameters())

    logger.info(f"Created NILUT model {lut_size} -- params={count_parameters(lut_model)}")
    
    # Load in memory the input and target hald images
    model_input_cpu, ground_truth_cpu = next(iter(dataloader))
    model_input, ground_truth = model_input_cpu.cuda(), ground_truth_cpu.cuda()
    logger.info(f"Training Input/Output shapes: {model_input.shape} / {ground_truth.shape}")
    
    model_input_cpu_val, ground_truth_cpu_val = next(iter(dataloader_val))
    model_input_val, ground_truth_val = model_input_cpu_val.cuda(), ground_truth_cpu_val.cuda()
    logger.info(f"Testing Input/Output shapes: {model_input_val.shape} / {ground_truth_val.shape}")

    lut_model.train()
    
    logger.info(f"** Start training for {total_steps} iterations")

    for step in range(total_steps+1):
        lut_model.train()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            model_output = lut_model(model_input)
            loss = torch.mean(torch.abs(model_output - ground_truth)) # more stable than L2
        
        if step % 200==0:
            logger.info(f"-- Train Step: {step}, loss: {loss}")

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        
        if step % 100==0 and step!=0:
            lut_model.eval()
            with torch.no_grad():
                model_output_val = lut_model(model_input_val)
                
                np_out_val = model_output_val.view(img_size_val[0], img_size_val[1], 3).detach()
                np_gt_val  = ground_truth_val.view(img_size_val[0], img_size_val[1], 3).detach()
                metrics = calc_metrics_torch(np_gt_val, np_out_val)
                logger.info('-- Evaluation Step: {} | l1 {:.6f} | mse {:.6f} | psnr {:.2f} | dE00 {:.4f} | dE76 {:.4f} | max_dE00 {:.4f}'.format(
                        step, metrics['l1'], metrics['l2'], metrics['psnr'], metrics['dE00'], metrics['dE76'], metrics['max_dE00']
                        ))


    torch.save(lut_model.state_dict(), os.path.join(save_dir, 
                                                    f"nilut{lut_size[0]}x{lut_size[1]}_{os.path.basename(out_path)[:-4]}_psnr{metrics['psnr']:.2f}_dE{metrics['dE76']:.4f}.pth"))
    clean_mem()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='NILUT fitting')
    parser.add_argument("--input", help="Input RGB map as a hald image", 
                        default="./dataset/hald_images/7luts_train/Original_Image.png", type=str)
    parser.add_argument("--target", help="Enhanced RGB map as a hald image, after using the desired 3D LUT", 
                        default="./dataset/hald_images/7luts_train/LUT02_After_Effects_LUTs_Blue_Shine.png", type=str)
    parser.add_argument("--input_val", help="Input RGB map as a hald image", 
                        default="./dataset/hald_images/7luts_test/Original_Image.png", type=str)
    parser.add_argument("--target_val", help="Enhanced RGB map as a hald image, after using the desired 3D LUT", 
                        default="./dataset/hald_images/7luts_test/LUT02_After_Effects_LUTs_Blue_Shine.png", type=str)
    
    parser.add_argument("--save_dir", help="Input RGB map as a hald image", default="./results/nilut", type=str)
    parser.add_argument("--steps", help="Number of optimizaation steps", default=10000, type=int)
    parser.add_argument("--units", help="NILUT MLP architecture: number of neurons", default=128, type=int)
    parser.add_argument("--layers", help="NILUT MLP architecture: number of layers", default=2, type=int)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(inp_path=args.input, 
         out_path=args.target,
         inp_path_val=args.input_val, 
         out_path_val=args.target_val,
         total_steps=args.steps,
         lut_size=(args.units, args.layers),
         save_dir=args.save_dir)
