# GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation

[![arXiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2605.19889)
[![web](https://img.shields.io/badge/Project-Page-orange)](https://color.cvc.uab.cat/glut/)

### Neurips 2026
[Danna Xue](https://dxue321.github.io/), [David Serrano-Lozano](https://davidserra9.github.io/), [Shaolin Su](https://ssl92.github.io/), and [Javier Vazquez-Corral](https://www.jvazquez-corral.net/)

Computer Vision Center, Universitat Autònoma de Barcelona

## About

**Gaussian LUT (GLUT)** is a compact, continuous color representation based on learnable 3D Gaussian primitives. It avoids fixed-resolution grids while providing high accuracy, interpretability, and direct local editing.

**Conditional GLUT (CGLUT)** further extends GLUT to represent multiple color styles within a single model, enabling smooth and controllable style blending.

GLUT also supports efficient, user-friendly **LUT editing**, allowing localized adjustments to specific color regions without global retraining. 



## Code overview

Fit a 3D colour LUT with a small mixture of 3D Gaussians in the RGB cube. Each Gaussian owns a centre, a full covariance, an opacity and a local affine colour transform; an input colour is transformed by blending the local transforms with opacity‑weighted Gaussian responses, on top of a global affine transform.

| Script | What it does |
|---|---|
| `train_glut.py`  | **GLUT** – fit **one** LUT (a single input/target hald‑image pair). |
| `train_cglut.py` | **CGLUT** – fit **many** LUTs with one conditional model (a learned per‑LUT embedding drives an MLP that generates the Gaussian parameters). |
| `blend_lut.py`   | Interpolate two LUTs on one image with a trained CGLUT (interpolate the two condition embeddings); optionally compare against the linear‑blend reference. |
| `interactive_glut_editor.py` | GUI to locally edit a fitted GLUT by nudging the top‑k Gaussians. |
| `test_image.py`  | Batch-evaluate GLUT/CGLUT checkpoints against the reference `.cube` LUTs on a folder of real images (PSNR / ΔE00 / ΔE76, CSV + optional comparison images). |



## 1. Environment

* Python ≥ 3.10, PyTorch ≥ 2.1 with CUDA (tested with 2.x + CUDA 12.6 on an RTX 4090).
* Python packages:

  ```bash
  pip install torch torchvision numpy opencv-python pillow matplotlib scipy \
              colour-science kornia lpips
  ```

### Optional: fused CUDA inference kernel

`train_glut.py --cuda_eval` and `CGLUT.forward_unified(..., use_cuda=True)` run the
forward pass through a hand‑written CUDA kernel. Build it once:

```bash
cd glut_cuda
python setup.py build_ext --inplace
```

Requires `nvcc` (CUDA toolkit). Edit `-arch=sm_89` in `glut_cuda/setup.py` to match
your GPU (e.g. `sm_89` = RTX 40‑series, `sm_86` = RTX 30‑series).
If `import glut_cuda` fails with `libc10.so: cannot open shared object file`,
either `import torch` first (the training scripts already do) or set
`export LD_LIBRARY_PATH=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib`.
Training always uses PyTorch autograd; the kernel is inference‑only.



## 2. Data

Everything is trained on **hald images**: a 2D image whose pixels enumerate every
RGB value of an identity 3D LUT. Applying a real LUT to that identity image gives
an `(input, target)` pair.

```
dataset/
|- 📁 cube_files/                    # .cube files
|- |- 📁 7luts/
|- |- 📁 75luts/
|  |  |- 📘 LUT01.cube
|  |  |- 📘 LUT02.cube
|  |  |- ...
|  |  |- 📘 LUT75.cube
|- |- 📁 225luts/
|- |- ...
|- 📁 hald_images/
|- |- 📁 7luts_train/                  # training set: one folder, many LUTs
|  |  |- Original_Image.png          #   identity hald (the model input)
|  |  |- 🖼️ LUT01.png                   #   target = LUT01 applied to Original_Image
|  |  |- 🖼️ LUT02.png
|  |  |- ...
|- |- 📁 7luts_test/                   # same LUT names, held-out hald resolution / crop
|- |- 📁 75luts_train/                 # 75-LUT set used for the shipped checkpoint
|- |- 📁 75luts_test/
|- |- 📁 ...
```

### Download

Pretrained checkpoints (`pretrained_models/`) and `.cube` files (`dataset/cube_files/`)
are available here:
**[Models & .cube files](https://cvcuab-my.sharepoint.com/:f:/g/personal/dxue_cvc_uab_cat/IgCyxm08DWUcRo5FHYZXi-QIAYA7WqYV6O-6qc1r9oEe4RM?e=sYyKpp)**

### Generating your own data

The easiest way is the end-to-end script, which generates the identity hald
image(s), applies every `.cube` file in a directory to them, and lays out the
result under `dataset/hald_images/` in the folder structure above:

```bash
# train/test split (two hald images: main + held-out complement)
./data/prepare_hald_dataset.sh --lut_dir path/to/cube_files --step 2
# -> dataset/hald_images/<cube_dir_name>_train/
# -> dataset/hald_images/<cube_dir_name>_test/

# single full-coverage set (no train/test split)
./data/prepare_hald_dataset.sh --lut_dir path/to/cube_files --step 1
# -> dataset/hald_images/<cube_dir_name>_fullsize/
```

To render a directory of `.cube` LUTs onto an existing hald image manually
(e.g. one you already have, or to control the output folder yourself), use
`apply_cube.py` directly — it writes one PNG per LUT plus `Original_Image.png`:

```bash
python data/apply_cube.py \
    --lut_dir     path/to/cube_files \
    --hald_image  path/to/hald_img.png \
    --out_dir     dataset/hald_images/my_luts_train
```



## 3. fit a single LUT

```bash
CUDA_VISIBLE_DEVICES=0 python train_glut.py \
    --train_input  dataset/hald_images/train/Original_Image.png \
    --train_target dataset/hald_images/train/LUT02_After_Effects_LUTs_Blue_Shine.png \
    --test_input   dataset/hald_images/test/Original_Image.png \
    --test_target  dataset/hald_images/test/LUT02_After_Effects_LUTs_Blue_Shine.png \
    --num_gaussians 32 \ # Gaussians per LUT.
    --residual 
```

---

## 4. fit many LUTs with one model

```bash
CUDA_VISIBLE_DEVICES=0 python train_cglut.py \
    --lut_data      ./dataset/hald_images/75lut_train \
    --lut_data_test ./dataset/hald_images/75lut_test \
    --num_gaussians 32 \   # Gaussians per LUT.
    --num_conditions 64 \  # dimensionality of the LUT condition embedding. 
    --num_neurons 128 \
    --shared_param 'all'
```

| Flag | Meaning |
|---|---|
| `--num_neurons` | width of the encoder / parameter‑head MLPs. 128 for Large (L), 64 for Small (S). |
| `--shared_param` | which Gaussian parameter groups are shared across LUTs vs. generated per LUT: `color_opacity` (share positions and covariance - 'SharedGeo' setting),  `all` (nothing shared - 'Full' setting). Colour transforms are always per‑LUT. |


The residual connection is always on. The model exposes:

* `forward(rgb, condition_idx)` — per‑pixel condition, used for training/eval.
* `forward_unified(rgb, condition_idx, use_cuda=False)` — one condition for the
  whole batch: run the parameter generator once, reuse it for every pixel
  (much faster for full‑image inference). `use_cuda=True` routes the blend
  through the compiled kernel.




## 5. interpolate two LUTs with CGLUT

Blend LUT *A* and LUT *B* at weight `alpha` by interpolating their learned
condition embeddings in a trained `ConditionalGLUT`
(`emb = (1-alpha)·emb_A + alpha·emb_B`), sweeping `alpha` from 0 to 1 for one
image. The model architecture is read from the checkpoint.

```bash
python blend_lut.py \
    --image            path/to/photo.png \
    --lut              dataset/cube_files/75lut \
    --pretrained_model pretrained_models/cglut/cglut_g32_e64_7styles_..._Full.pth \
    --lut_a 0   \
    --lut_b 3   \
    --n_alpha 6 \
    --save_dir results/blend
```


## 6. edit LUT interactively

`interactive_glut_editor.py` is a Dash web app for editing a trained GLUT by
picking an input color on an exemplar image and dragging it to a desired
output color — no retraining involved.

```bash
python interactive_glut_editor.py

```
Then open `http://127.0.0.1:8050` in a browser.

Workflow:
1. Upload an image via the toolbar button (or pass `--image` at launch).
2. Click a pixel on the original image → sets the source color.
3. Pick the desired output color with the color picker.
4. Tune the `K` and `Strength` sliders.
5. Press **Apply Edit** → the GLUT is updated in place, and the edited image,
   3D cube view, and delta badge refresh instantly.
6. Press **Download All** to get the original / default / edited images as a ZIP.
7. Press **Save Model** to write the edited GLUT checkpoint.
8. Press **Reset** to restore the original parameters.




## Citation

Hope you like it 🤗 

If you find this work interesting or you use it, don't forget to cite our work:

```bibtex
@article{xue2026glut,
  title={GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation},
  author={Xue, Danna and Serrano-Lozano, David and Su, Shaolin and Vazquez-Corral, Javier},
  journal={arXiv preprint arXiv:2605.19889},
  year={2026}
}
```

## Acknowledgements

Thanks to the [NILUT](https://github.com/mv-lab/nilut) project for its code and data.


