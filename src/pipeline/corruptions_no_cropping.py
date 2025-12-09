import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from PIL import Image
import skimage as sk
from tqdm import tqdm
from skimage.filters import gaussian
# from scipy.ndimage import gaussian_filter as gaussian # edit
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
# from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import map_coordinates
import warnings
from pkg_resources import resource_filename
from torch.autograd import Variable as V
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import logging, sys
from src.utils import log_timer, setup_logging
import traceback

warnings.simplefilter("ignore", UserWarning)

# fixes GLIBCXX not found bug
libdir = os.path.join(sys.prefix, "lib")
prev = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{prev}"

LOGS_DIR = "./logs"

# CODE CITATIONS:
# the classes and functions for Corruptions come from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py

#### corruption helpers ####

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def clipped_zoom(img, zoom_factor):
    """
    Zooms an H×W image and returns the same H×W crop
    """

    H, W = img.shape[:2]          # ← dynamic size
    ch, cw = int(np.ceil(H/zoom_factor)), int(np.ceil(W/zoom_factor))

    top  = (H - ch) // 2
    left = (W - cw) // 2
    img  = scizoom(img[top:top+ch, left:left+cw],
                   (zoom_factor, zoom_factor, 1), order=1)

    trim_t = (img.shape[0] - H) // 2
    trim_l = (img.shape[1] - W) // 2
    return img[trim_t:trim_t+H, trim_l:trim_l+W]

#### corruptions ####

def gaussian_noise(x, severity=1):
    # 1st
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity=1):
    # 2nd
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255

def impulse_noise(x, severity=1):
    # 3rd
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def defocus_blur(x, severity=1):
    # 4th
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))

    return np.clip(channels, 0, 1) * 255

def glass_blur(x, severity=1):
    # 5th
    sigma, max_d, iters = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity-1]

    x = np.uint8(gaussian(np.array(x)/255., sigma=sigma, channel_axis=-1)*255)

    H, W = x.shape[:2]
    for _ in range(iters):
        for h in range(H - max_d - 1, max_d - 1, -1):      
            for w in range(W - max_d - 1, max_d - 1, -1):  
                dx, dy = np.random.randint(-max_d, max_d + 1, 2)
                h2, w2 = h + dy, w + dx
                x[h, w], x[h2, w2] = x[h2, w2], x[h, w]     # swap

    return np.clip(gaussian(x / 255., sigma=sigma,
                            channel_axis=-1), 0, 1) * 255

def motion_blur(x, severity=1, angle_1=-45, angle_2=45):
    # 6th
    # Modified from original function to work with current pytorch libraries and for our datasets

    # map severity to kernel length
    lengths = [10, 15, 20, 25, 30]
    L = lengths[severity - 1]

    # build a horizontal line kernel
    kernel = np.zeros((L, L), dtype=np.float32)
    kernel[L // 2, :] = np.ones(L, dtype=np.float32) / L

    # pick random angle
    angle = np.random.uniform(angle_1, angle_2)
    M = cv2.getRotationMatrix2D((L/2, L/2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (L, L))

    # normalize kernel
    kernel /= kernel.sum()

    # convert to float array [0,1]
    arr = np.array(x, dtype=np.float32) / 255.0
    blurred = np.zeros_like(arr)

    # apply filter to each channel
    for c in range(3):
        blurred[..., c] = cv2.filter2D(arr[..., c], -1, kernel,
                                       borderType=cv2.BORDER_REFLECT)

    # back to uint8 PIL.Image
    blurred = np.clip(blurred * 255, 0, 255).astype(np.uint8)
    return PILImage.fromarray(blurred)

def zoom_blur(x, severity=1):
    # 7th
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

def snow(x, severity=1):
    """
    x: H×W×3 RGB uint8 image (PIL.Image or NumPy array)
    severity: int in 1..5
    returns: corrupted H×W×3 RGB uint8 image
    """
    # 8th
    # Modified from original function to work with current pytorch libraries and for our datasets

    # severity parameters: loc, scale, zoom_factor, threshold, _, _, blend
    loc, scale, zoom_f, thresh, _, _, blend = [
        (0.1, 0.3, 3,   0.5, 10, 4, 0.8),
        (0.2, 0.3, 2,   0.5, 12, 4, 0.7),
        (0.55,0.3, 4,   0.9, 12, 8, 0.7),
        (0.55,0.3, 4.5, 0.85,12, 8, 0.65),
        (0.55,0.3, 2.5, 0.85,12,12, 0.55),
    ][severity - 1]

    # normalize base image to [0,1]
    img = np.asarray(x, dtype=np.float32) / 255.0
    H, W = img.shape[:2]

    # 1) generate monochrome noise
    noise = np.random.normal(loc, scale, (H, W)).astype(np.float32)

    # 2) zoom and crop/pad using clipped_zoom
    noise = clipped_zoom(noise[..., None], zoom_f)[..., 0]

    # 3) threshold & clamp
    noise[noise < thresh] = 0.0
    noise = np.clip(noise, 0.0, 1.0)

    # 4) create 3‑channel PIL image for motion blur
    snow_pil = PILImage.fromarray((noise * 255).astype(np.uint8)).convert("RGB")

    # 5) apply motion blur
    snow_blr = np.array(motion_blur(snow_pil, severity, angle_1=-135, angle_2=-45), dtype=np.float32) / 255.0

    # 6) extract one channel and rotated copy
    snow_ch  = snow_blr[..., :1]            # shape (H, W, 1)
    snow_rot = np.rot90(snow_ch, 2)         # rotated 180°

    # 7) enhance contrast toward gray
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = (gray.astype(np.float32) / 255.0)[..., None]
    enhanced = np.maximum(img, gray * 1.5 + 0.5)

    # 8) blend base + snow layers
    base = blend * img + (1.0 - blend) * enhanced
    out = np.clip(base + snow_ch + snow_rot, 0.0, 1.0)

    # 9) convert back to uint8
    return (out * 255).astype(np.uint8)

def frost(x, severity=1):
    # 9th
    # blend weights: base, overlay
    # using dtype=float32 to reduce memory and speed up

    alpha, beta = [
        (1.0, 0.4),
        (0.8, 0.6),
        (0.7, 0.7),
        (0.65, 0.7),
        (0.6, 0.75),
    ][severity - 1]

    # pick a random frost file from the local frost/ folder
    frost_files = [
        'frost/frost1.png',
        'frost/frost2.png',
        'frost/frost3.png',
        'frost/frost4.jpg',
        'frost/frost5.jpg',
        'frost/frost6.jpg',
    ]
    fname = np.random.choice(frost_files)
    frost_img = cv2.imread(fname, cv2.IMREAD_COLOR)
    if frost_img is None:
        raise FileNotFoundError(f"Could not load frost overlay: {fname}")
    # convert BGR to RGB
    frost_img = frost_img[..., ::-1].astype(np.float32)

    # prepare base image as float32
    base = np.asarray(x, dtype=np.float32)
    if base.ndim == 2:
        base = np.stack([base]*3, axis=-1)
    H, W = base.shape[:2]

    # if overlay is larger than base, take a random H×W crop; otherwise resize it
    h0, w0 = frost_img.shape[:2]
    if h0 >= H and w0 >= W:
        y0 = np.random.randint(0, h0 - H + 1)
        x0 = np.random.randint(0, w0 - W + 1)
        overlay = frost_img[y0:y0+H, x0:x0+W]
    else:
        overlay = cv2.resize(frost_img, (W, H), interpolation=cv2.INTER_AREA)

    # blend and clamp to [0,255]
    out = alpha * base + beta * overlay
    return np.clip(out, 0, 255).astype(np.uint8)

def fog(x, severity=1):
    # 10th
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    x = np.array(x) / 255.
    H, W = x.shape[:2]
    max_val = x.max()
    size = 1 << int(np.ceil(np.log2(max(H, W))))
    x += c[0] * plasma_fractal(mapsize=size, wibbledecay=c[1])[:H, :W][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def brightness(x, severity=1):
    # 11th
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def contrast(x, severity=1):
    # 12th
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(x, severity=1):
    # 13th
    # 1) normalize once
    image = np.array(x, dtype=np.float32) / 255.0
    H, W = image.shape[:2]

    # severity-dependent parameters: (alpha, sigma, alpha_affine)
    c = [
        (H * 2,    H * 0.7,  H * 0.1),
        (H * 2,    H * 0.08, H * 0.2),
        (H * 0.05, H * 0.01, H * 0.02),
        (H * 0.07, H * 0.01, H * 0.02),
        (H * 0.12, H * 0.01, H * 0.02),
    ][severity - 1]
    alpha, sigma, alpha_affine = c

    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center = np.float32(shape_size) // 2
    sq = min(shape_size) // 3
    pts1 = np.float32([
        center + sq,
        [center[0] + sq, center[1] - sq],
        center - sq
    ])
    pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    # displacement fields
    dx = gaussian(np.random.uniform(-1, 1, size=(H, W)), sigma, mode='reflect', truncate=3) * alpha
    dy = gaussian(np.random.uniform(-1, 1, size=(H, W)), sigma, mode='reflect', truncate=3) * alpha
    dx = dx[..., None]
    dy = dy[..., None]

    # meshgrid for indexing
    xg, yg, zg = np.meshgrid(np.arange(W), np.arange(H), np.arange(shape[2]), indexing='xy')
    indices = (yg + dy).reshape(-1, 1), (xg + dx).reshape(-1, 1), zg.reshape(-1, 1)

    # apply elastic warp and convert back
    distorted = map_coordinates(warped, indices, order=1, mode='reflect').reshape(shape)
    return np.clip(distorted * 255, 0, 255).astype(np.uint8)

def pixelate(x, severity=1):
    # 14th
    c = [0.60, 0.50, 0.40, 0.30, 0.25][severity-1]
    W, H = x.size                          # PIL gives (width, height)
    x     = x.resize((max(1,int(W*c)), max(1,int(H*c))), PILImage.BOX)
    return x.resize((W, H), PILImage.BOX)

def jpeg_compression(x, severity=1):
    # 15th
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x

def corrupt_datasets(config: DictConfig):
    """
    Walks through the directory tree at the input dataset (specified via config["dataset_paths"][dataset_key]),
    applies the specified corruption to each image, and saves the result 
    preserving the original folder structure under the dataset root.
    """

    # resolve paths from config
    try:
        if config.splitted_parsing:
            splitted_suffix = "_".join(config.starting_split_path.split("/")[-2:])

            log_filename = f"corruptions_dset-{config.dataset_key}_type-{config.corruption_type}_sev-{config.severity}-starting_path-{splitted_suffix}.log"
            log_filepath = os.path.join(LOGS_DIR, log_filename)
            setup_logging(log_filepath)

            logging.info(f"corrupting dataset {config.dataset_key} with {config.corruption_type} at severity {config.severity} from {splitted_suffix}")
            input_path = os.path.join(config.data_paths.root, config.dataset_key)
            output_base = os.path.join(config.data_paths.corruptions, f"severity_{config.severity}")
            logging.info("paths for corruption set")
        else:
            # setting up logs
            log_filename = f"corruptions_dset-{config.dataset_key}_type-{config.corruption_type}_sev-{config.severity}.log"
            log_filepath = os.path.join(LOGS_DIR, log_filename)
            setup_logging(log_filepath)

            logging.info(f"corrupting dataset {config.dataset_key} with {config.corruption_type} at severity {config.severity}")
            input_path = os.path.join(config.data_paths.root, config.dataset_key)
            output_base = os.path.join(config.data_paths.corruptions, f"severity_{config.severity}")
            logging.info("paths for corruption set")
    except Exception:
        logging.info("error loading paths from config")
        raise 

    # get the corruption function
    try: 
        logging.info(f"available corruptions: {globals()}")
        if config.corruption_type not in globals():
            raise ValueError(f"Unknown corruption type: {config.corruption_type}")
    except Exception:
        raise

    try:
        if config.corruption_type == "gaussian_noise":
            corruption_fn = gaussian_noise
        elif config.corruption_type == "shot_noise":
            corruption_fn = shot_noise
        elif config.corruption_type == "impulse_noise":
            corruption_fn = impulse_noise
        elif config.corruption_type == "defocus_blur":
            corruption_fn = defocus_blur
        elif config.corruption_type == "glass_blur":
            corruption_fn = glass_blur
        elif config.corruption_type == "motion_blur":
            corruption_fn = motion_blur
        elif config.corruption_type == "zoom_blur":
            corruption_fn = zoom_blur
        elif config.corruption_type == "snow":
            corruption_fn = snow
        elif config.corruption_type == "frost":
            corruption_fn = frost
        elif config.corruption_type == "fog":
            corruption_fn = fog
        elif config.corruption_type == "brightness":
            corruption_fn = brightness
        elif config.corruption_type == "contrast":
            corruption_fn = contrast
        elif config.corruption_type == "elastic_transform":
            corruption_fn = elastic_transform
        elif config.corruption_type == "pixelate":
            corruption_fn = pixelate
        elif config.corruption_type == "jpeg_compression":
            corruption_fn = jpeg_compression
        else:
            raise ValueError(f"Unknown corruption type: {config.corruption_type}")

        # define root of corrupted output
        root_output = os.path.join(output_base, config.corruption_type, config.dataset_key)

        # start processing from first root folder if splitted_parsing is not set
        start_processing = False
        if not config.splitted_parsing:
            start_processing = True

        # walk through the input dataset
        folder_count = 0
        for root, dirs, files in tqdm(os.walk(input_path)):
            # start processing from specified path if splitted_parsing is set
            if config.splitted_parsing and not start_processing:
                if os.path.normpath(root) == os.path.normpath(config.starting_split_path):
                    start_processing = True
                else:
                    logging.info(f"skipping directory: {root}")
                    continue  # skip this directory

            rel_dir = os.path.relpath(root, input_path)
            target_dir = os.path.join(root_output, rel_dir)
            os.makedirs(target_dir, exist_ok=True)

            logging.info(f"processing directory: {root}")

            for fname in tqdm(files):
                lower = fname.lower()
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(target_dir, fname)

                # if already exists, skip
                if os.path.exists(dst_path):
                    # logging.info(f"Skipping existing file: {dst}")
                    continue

                # always copy the image list file
                if fname == "image_unida_list.txt":
                    with open(src_path, 'rb') as f_src, open(dst_path, 'wb') as f_dst:
                        f_dst.write(f_src.read())
                    continue

                if lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img = PILImage.open(src_path).convert('RGB')
                    corrupted = corruption_fn(img, config.severity)

                    if isinstance(corrupted, np.ndarray):
                        corrupted = PILImage.fromarray(
                            np.clip(corrupted, 0, 255).astype(np.uint8)
                        )

                    corrupted.save(dst_path)
                else:
                    # copy non-image files directly
                    try:
                        with open(src_path, 'rb') as f_src, open(dst_path, 'wb') as f_dst:
                            f_dst.write(f_src.read())
                    except Exception:
                        pass
            
            logging.info(f"completed processing directory: {root}")

            # validating number of processed folders (if splitted_parsing is set, else, process all)
            if config.splitted_parsing:
                folder_count += 1
                num_folders = config.num_folders # limit of folders to process
                if folder_count >= num_folders:
                    logging.info(f"Reached the limit of {num_folders} folders. Stopping.")
                    break
            
        logging.info(f"corrupted dataset saved to {root_output}")
    except Exception:
        raise
