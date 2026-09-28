

"""
FastNILUT: NILUT accelerated with NVIDIA tiny-cuda-nn's fully-fused MLP kernel.

Same architecture/API as NILUT (nilut.py):
    Linear(in, H) -> ReLU -> [Linear(H, H) -> Tanh] x hidden_layers -> Linear(H, out) [-> Sigmoid] -> (+input) -> clamp

The input layer (3 -> H) is tiny and kept as a regular nn.Linear. Everything after it
(the H-wide stack, which is where almost all the FLOPs are) is executed as a single
tcnn.Network using the 'FullyFusedMLP' kernel, which keeps activations resident in
shared memory/registers across all layers instead of round-tripping through global
memory between separate GEMMs -- this is the whole speedup versus stacked nn.Linear.

Requires `tinycudann` (https://github.com/NVlabs/tiny-cuda-nn, bindings/torch).
FullyFusedMLP only supports n_neurons in {16, 32, 64, 128}; for other widths this
falls back to tcnn's 'CutlassMLP' (still fused via cutlass GEMMs, but not as fast as
FullyFusedMLP) with a warning.
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose

import os
import gc
import warnings
import argparse

try:
    import tinycudann as tcnn
except ImportError as e:
    raise ImportError(
        "nilut_tcnn requires the `tinycudann` package (NVIDIA tiny-cuda-nn PyTorch "
        "bindings). Install it from https://github.com/NVlabs/tiny-cuda-nn "
        "(bindings/torch) for a CUDA/PyTorch combination it supports."
    ) from e

# Import NILUT utils
from utils.utils import load_img, clean_mem, count_parameters, setup_logging
from utils.metrics import calc_metrics_np, np_psnr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = None

_FULLY_FUSED_WIDTHS = (16, 32, 64, 128)


class FastNILUT(nn.Module):
    """
    NILUT with its hidden H->H stack executed by tiny-cuda-nn's fully-fused MLP.
    Drop-in replacement for `nilut.NILUT` (same constructor args, same forward signature).
    """
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3, out_features=3, res=True):
        super().__init__()

        self.res = res
        self.hidden_features = hidden_features

        self.input_layer = nn.Linear(in_features, hidden_features)
        self.input_act = nn.ReLU()

        if hidden_features in _FULLY_FUSED_WIDTHS:
            otype = "FullyFusedMLP"
        else:
            warnings.warn(
                f"hidden_features={hidden_features} is not in {_FULLY_FUSED_WIDTHS}, "
                "which is required by tiny-cuda-nn's FullyFusedMLP kernel. "
                "Falling back to 'CutlassMLP' (still fused, but slower than FullyFusedMLP)."
            )
            otype = "CutlassMLP"

        self.body = tcnn.Network(
            n_input_dims=hidden_features,
            n_output_dims=out_features,
            network_config={
                "otype": otype,
                "activation": "Tanh",
                "output_activation": "None" if res else "Sigmoid",
                "n_neurons": hidden_features,
                "n_hidden_layers": hidden_layers,
            },
        )

    def forward(self, intensity):
        h = self.input_act(self.input_layer(intensity))
        output = self.body(h).to(intensity.dtype)
        if self.res:
            output = output + intensity
        output = torch.clamp(output, 0., 1.)

        return output


class LUTFitting(Dataset):
    def __init__(self, inp_img, out_img, resize=False):
        super().__init__()

        img = load_img(inp_img)
        lut = load_img(out_img)

        self.error = np_psnr(img, lut)

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

        self.intensities = img.reshape((img.shape[0]*img.shape[1], 3))
        self.outputs     = lut.reshape((img.shape[0]*img.shape[1], 3))
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
    Fit a professional 3D LUT into a FastNILUT (tiny-cuda-nn fully-fused MLP).
    """
    logger = setup_logging(
        log_dir=save_dir,
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )

    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Start FastNILUT {lut_size} fitting with")
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
    lut_model = FastNILUT(in_features=3, out_features=3, hidden_features=lut_size[0], hidden_layers=lut_size[1], res=True)
    lut_model.to(device)
    # tcnn manages its own fp16 forward/backward internally (master weights stay
    # fp32), so plain fp32 Adam is used here -- no torch.cuda.amp autocast/GradScaler.
    opt = torch.optim.Adam(lr=1e-3, params=lut_model.parameters())

    logger.info(f"Created FastNILUT model {lut_size} -- params={count_parameters(lut_model)}")

    # Load in memory the input and target hald images
    model_input_cpu, ground_truth_cpu = next(iter(dataloader))
    model_input, ground_truth = model_input_cpu.to(device), ground_truth_cpu.to(device)
    logger.info(f"Training Input/Output shapes: {model_input.shape} / {ground_truth.shape}")

    model_input_cpu_val, ground_truth_cpu_val = next(iter(dataloader_val))
    model_input_val, ground_truth_val = model_input_cpu_val.to(device), ground_truth_cpu_val.to(device)
    logger.info(f"Testing Input/Output shapes: {model_input_val.shape} / {ground_truth_val.shape}")

    lut_model.train()

    logger.info(f"** Start training for {total_steps} iterations")

    for step in range(total_steps+1):
        lut_model.train()
        model_output = lut_model(model_input)
        loss = torch.mean(torch.abs(model_output - ground_truth))  # more stable than L2

        if step % 200 == 0:
            logger.info(f"-- Train Step: {step}, loss: {loss}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10000 == 0 and step != 0:
            lut_model.eval()
            with torch.no_grad():
                model_output_val = lut_model(model_input_val)

                np_out_val = model_output_val.view(img_size_val[0], img_size_val[1], 3).detach()
                np_gt_val  = ground_truth_val.view(img_size_val[0], img_size_val[1], 3).detach()
                metrics = calc_metrics_np(np_gt_val, np_out_val)
                logger.info('-- Evaluation Step: {} | l1 {:.6f} | mse {:.6f} | psnr {:.2f} | dE00 {:.4f} | dE76 {:.4f} | max_dE00 {:.4f}'.format(
                        step, metrics['l1'], metrics['l2'], metrics['psnr'], metrics['dE00'], metrics['dE76'], metrics['max_dE00']
                        ))


    torch.save(lut_model.state_dict(), os.path.join(save_dir,
                                                    f"fastnilut{lut_size[0]}x{lut_size[1]}_{os.path.basename(out_path)[:-4]}_psnr{metrics['psnr']:.2f}_dE{metrics['dE76']:.4f}.pth"))
    clean_mem()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FastNILUT (tiny-cuda-nn fully-fused MLP) fitting')
    parser.add_argument("--input", help="Input RGB map as a hald image",
                        default="./dataset/hald_images/7luts_train/Original_Image.png", type=str)
    parser.add_argument("--target", help="Enhanced RGB map as a hald image, after using the desired 3D LUT",
                        default="./dataset/hald_images/7luts_train/LUT02_After_Effects_LUTs_Blue_Shine.png", type=str)
    parser.add_argument("--input_val", help="Input RGB map as a hald image",
                        default="./dataset/hald_images/7luts_test/Original_Image.png", type=str)
    parser.add_argument("--target_val", help="Enhanced RGB map as a hald image, after using the desired 3D LUT",
                        default="./dataset/hald_images/7luts_test/LUT02_After_Effects_LUTs_Blue_Shine.png", type=str)

    parser.add_argument("--save_dir", help="Input RGB map as a hald image", default="./results/fastnilut", type=str)
    parser.add_argument("--steps", help="Number of optimizaation steps", default=10000, type=int)
    parser.add_argument("--units", help="NILUT MLP architecture: number of neurons (16/32/64/128 for FullyFusedMLP)", default=128, type=int)
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
