import numpy as np
import colour
import cv2

import torch
import torch.nn.functional as F


# Metrics
def calc_metrics_torch(pred, ref, logger=None):
    l1_error = F.l1_loss(pred, ref).item()
    l2_error = F.mse_loss(pred, ref).item()

    psnr = 10 * np.log10(1.0 ** 2 / l2_error)
            
    deltaE00 = calc_deltaE(pred.cpu().numpy(), ref.cpu().numpy(), method='CIE 2000', reduction='mean')
    deltaE76 = calc_deltaE(pred.cpu().numpy(), ref.cpu().numpy(), method='CIE 1976', reduction='mean')

    # max_diff = np.max(l1_error)
    max_diff = np.max(calc_deltaE(pred.cpu().numpy(), ref.cpu().numpy(), reduction ='none'))

    if logger:
        logger.info('-- l1: {:.6f} | mse: {:.6f}  | psnr: {:.2f} | dE00: {:.4f} | dE76: {:.4f} | max_dE00: {:.6f}'.format(
                l1_error, l2_error, psnr, deltaE00, deltaE76, max_diff
                ))
        
    return {
            'l1': l1_error,
            'l2': l2_error,
            'psnr': psnr,
            'dE00': deltaE00,
            'dE76': deltaE76,
            'max_dE00': max_diff,
            }
    

def calc_metrics_np(img1, img2, lpips_metric=None):
    
    psnr = calc_psnr(img1, img2)
    # SSIM
    # if len(img1.shape) == 3:
    #     img1_gray = color.rgb2gray(img1)
    #     img2_gray = color.rgb2gray(img2)
    #     ssim_value = ssim(img1_gray, img2_gray, data_range=1.0)
    # else:
    #     ssim_value = ssim(img1, img2, data_range=1.0)
    
    ssim_value = calculate_ssim(img1, img2, crop_border=0, input_order='HWC')

    delta_e00 = calc_deltaE(img1, img2, method='CIE 2000', reduction='mean')
    delta_e76 = calc_deltaE(img1, img2, method='CIE 1976', reduction='mean')
    
    # RMSE
    rmse = np.sqrt(np.mean((img1 - img2) ** 2))
    
    # MAE
    mae = np.mean(np.abs(img1 - img2))

    if lpips_metric is not None:
        dev = next(lpips_metric.parameters()).device
        lpips = calc_lpips(torch.from_numpy(img1).permute(2, 0, 1).to(dev),
                           torch.from_numpy(img2).permute(2, 0, 1).to(dev),
                           lpips_metric, normalize=True)
    else:
        lpips = 0.0
    return {
        'PSNR': psnr,
        'SSIM': ssim_value,
        'dE00': delta_e00,
        'dE76': delta_e76,
        'RMSE': rmse,
        'MAE': mae,
        'LPIPS': lpips
    }


def np_psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if(mse == 0):  return np.inf
    return 10 * np.log10(1 ** 2 / mse)

def pt_psnr (y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return 10 * torch.log10(1 ** 2 / mse)
    
def __error(a, b):
    return a.astype(np.float64) - b.astype(np.float64)


def __squared_error(a, b):
    return np.power(__error(a, b), 2.0)


def mse(a, b, axis=None):
    return np.mean(__squared_error(a, b), axis=axis)


def calc_psnr(a, b, axis=None):
    if a.dtype != b.dtype:
        raise Exception(
            f"Wrong numpy array type. 2 arrays should have the same dtype: {a.dtype} vs {b.dtype}"
        )
    if a.dtype == np.uint8:
        max_value = 255
    elif a.dtype in (np.float16, np.float32, np.float64):
        max_value = 1
    else:
        raise Exception(f"Wrong numpy array type. Expect float or uint8 but got {a.dtype}")

    return 10 * np.log10(max_value ** 2 / mse(a, b, axis))




COLOURSPACE_ZOO = {'ProPhoto': colour.models.RGB_COLOURSPACE_PROPHOTO_RGB,
                   'DCI-P3': colour.models.RGB_COLOURSPACE_DCI_P3,
                   'DisplayP3': colour.models.RGB_COLOURSPACE_DISPLAY_P3,
                   'P3-D65': colour.models.RGB_COLOURSPACE_P3_D65,
                   'BT2020': colour.models.RGB_COLOURSPACE_BT2020,
                   'sRGB': colour.models.RGB_COLOURSPACE_sRGB,
                   }


def calc_deltaE(source, target, color_space='sRGB', method='CIE 2000', CAT='CAT02', reduction='mean'):
    # method = 'CIE 2000' | 'CIE 1976'
    assert isinstance(color_space, str), "color_space should be string"
    assert COLOURSPACE_ZOO.get(color_space) != None, "color_space should be ProPhoto, sRGB, AdobeRGB, DisplayP3"
    color_space = COLOURSPACE_ZOO.get(color_space)
    source = np.reshape(source, (-1,3))
    target = np.reshape(target, (-1,3))
    source_XYZ = colour.models.RGB_to_XYZ(
        source,
        color_space,
        None,
        chromatic_adaptation_transform=CAT
    )
    target_XYZ = colour.models.RGB_to_XYZ(
        target,
        color_space,
        None,
        chromatic_adaptation_transform=CAT
    )
    source_Lab = colour.models.XYZ_to_Lab(
        source_XYZ,
        color_space.whitepoint,
    )
    target_Lab = colour.models.XYZ_to_Lab(
        target_XYZ,
        color_space.whitepoint,
    )    
    deltaE = colour.delta_E(source_Lab, target_Lab, method=method)

    if reduction == 'sum':
        return np.sum(deltaE)
    elif reduction == 'mean':
        return np.mean(deltaE)
    elif reduction == 'none':
        return deltaE





def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()




def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def calc_lpips(output, target, loss_fn, normalize=True):
    """
    output, target:
        torch.Tensor
        shape: [N, 3, H, W] or [3, H, W]
        value range: [0,1] or [-1,1]

    return:
        scalar LPIPS value
    """

    if output.dim() == 3:
        output = output.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    assert output.shape == target.shape, "Output and target must have same shape"
    assert output.shape[1] == 3, "LPIPS expects RGB images"

    if normalize:
        # Assume inputs are in [0, 1]; LPIPS expects [-1, 1].
        output = output * 2 - 1
        target = target * 2 - 1

    with torch.no_grad():
        lpips_val = loss_fn(output, target)

    return lpips_val.mean().item()