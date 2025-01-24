import numpy as np
import skimage
import skimage.io as skio
import skimage.exposure
import skimage.transform
import matplotlib.pyplot as plt
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import skimage
import skimage.io as skio
from skimage.transform import resize
import matplotlib.pyplot as plt

# Ensure 'pytorch-CycleGAN-and-pix2pix' is in your Python path.
from options.test_options import TestOptions
from models import create_model

def transform_image_A_to_B(
    input_image: Image.Image,
    weights_path: str,
    device: str = "mps",    # "cpu", "cuda", or "mps"
    resize_to: int = 256
) -> np.ndarray:
    """
    Creates a 'cycle_gan' model for domain A->B, manually loads netG_A from `weights_path`,
    and applies it to `input_image`. Returns a NumPy array representing the image.

    Supports CPU, CUDA, or MPS:
      device="cpu"  => uses CPU
      device="cuda" => uses GPU
      device="mps"  => uses Apple Silicon GPU (macOS 12.3+ w/ PyTorch MPS build)

    :param input_image:   A PIL image (or string path to an image).
    :param weights_path:  Path to the netG_A checkpoint (e.g. '15_net_G_A.pth').
    :param device:        'cpu', 'cuda', or 'mps'.
    :param resize_to:     Resolution for inference (default=256).
    :return:             A NumPy array in domain B.
    """

    # 1. Prepare minimal CLI args so TestOptions won't fail
    original_argv = sys.argv
    sys.argv = [
        original_argv[0],
        "--dataroot", "dummy",         # needed if '--dataroot' is required=True
        "--model", "cycle_gan",        # ensures we have netG_A and netG_B
        "--dataset_mode", "single",    # we feed images manually
        "--gpu_ids", "-1"              # we override device manually below
    ]
    opt = TestOptions().parse()  # parse minimal CLI args
    sys.argv = original_argv     # restore original argv

    # 2. Override needed fields
    opt.isTrain = False
    opt.no_dropout = True
    opt.load_size = resize_to
    opt.crop_size = resize_to
    opt.serial_batches = True
    opt.num_threads = 0
    opt.batch_size = 1

    # 3. Create the model in 'cycle_gan' mode (so we have netG_A)
    model = create_model(opt)
    model.eval()
    device = "mps"

    # 4. Manually load netG_A from your checkpoint
    state_dict = torch.load(weights_path, map_location="mps")
    model.netG_A.load_state_dict(state_dict)

    # 5. If input_image is a path, load as a PIL image
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")

    # Save original input image shape for resizing later
    original_shape = input_image.size  # (Width, Height)

    # 6. Preprocess (CycleGAN typically uses [-1,1])
    transform_pipeline = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform_pipeline(input_image).unsqueeze(0)  # (1, C, H, W)

    # 7. Move the model & data to the chosen device
    if device.lower() == "cuda" and torch.cuda.is_available():
        model.netG_A.to("cuda")
        input_tensor = input_tensor.to("cuda")
    elif device.lower() == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model.netG_A.to("mps")
        input_tensor = input_tensor.to("mps")
    else:
        model.netG_A.to("cpu")
        input_tensor = input_tensor.to("cpu")

    # 8. Forward pass with G_A
    with torch.no_grad():
        fake_B = model.netG_A(input_tensor)

    # 9. Post-process: from [-1,1] -> [0,255], then to NumPy
    fake_B = (0.5 * (fake_B + 1.0)) * 255  # Scale to [0, 255]
    fake_B_numpy = fake_B.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Convert to HWC format

    # 10. Resize to original input image shape
    original_shape_hw = (original_shape[1], original_shape[0])  # PIL gives (Width, Height); skimage uses (Height, Width)
    fake_B_resized = skimage.transform.resize(
        fake_B_numpy, original_shape_hw, anti_aliasing=True, preserve_range=True
    ).astype(np.uint8)  # Stretch output to match input shape

    return fake_B_resized

def rgb_to_hsl_vectorized(image):
    """Convert an RGB image to HSL using vectorized operations."""
    image = np.clip(image, 0, 1)  # Ensure input values are in [0, 1]
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    l = (maxc + minc) / 2

    s = np.zeros_like(l)
    delta = maxc - minc
    s[l > 0] = delta[l > 0] / (1 - np.abs(2 * l[l > 0] - 1))

    h = np.zeros_like(l)
    mask = delta > 0
    r_eq = (maxc == r) & mask
    g_eq = (maxc == g) & mask
    b_eq = (maxc == b) & mask

    h[r_eq] = (g[r_eq] - b[r_eq]) / delta[r_eq] % 6
    h[g_eq] = (b[g_eq] - r[g_eq]) / delta[g_eq] + 2
    h[b_eq] = (r[b_eq] - g[b_eq]) / delta[b_eq] + 4
    h /= 6
    h[h < 0] += 1

    return np.stack([h, l, s], axis=-1)

def hsl_to_rgb_vectorized(hsl_image):
    """Convert an HSL image back to RGB using vectorized operations."""
    h, l, s = hsl_image[..., 0], hsl_image[..., 1], hsl_image[..., 2]

    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c / 2

    z = np.zeros_like(h)
    r, g, b = (
        np.select([h < 1/6, (1/6 <= h) & (h < 2/6), (2/6 <= h) & (h < 3/6),
                   (3/6 <= h) & (h < 4/6), (4/6 <= h) & (h < 5/6), h >= 5/6],
                  [c, x, z, z, x, c]),
        np.select([h < 1/6, (1/6 <= h) & (h < 2/6), (2/6 <= h) & (h < 3/6),
                   (3/6 <= h) & (h < 4/6), (4/6 <= h) & (h < 5/6), h >= 5/6],
                  [x, c, c, x, z, z]),
        np.select([h < 1/6, (1/6 <= h) & (h < 2/6), (2/6 <= h) & (h < 3/6),
                   (3/6 <= h) & (h < 4/6), (4/6 <= h) & (h < 5/6), h >= 5/6],
                  [z, z, x, c, c, x])
    )
    return np.clip(np.stack([r + m, g + m, b + m], axis=-1), 0, 1)
import numpy as np
from skimage import img_as_float, img_as_ubyte, color
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def generate_film_grain(image, grain_intensity=0.2, chroma=0.5, blur_sigma=0.8):
    """
    Simulates realistic film grain with brightness-dependent density and chroma control.

    Parameters:
        image (ndarray): Input image in RGB format (H, W, 3).
        grain_intensity (float): Intensity of the grain, higher values mean more grain.
        chroma (float): Chroma component of the grain (0 = grayscale, 1 = fully chromatic).
        blur_sigma (float): Standard deviation for Gaussian blur to control grain size.

    Returns:
        ndarray: Image with simulated film grain applied.
    """
    # Ensure input is in float format
    image = img_as_float(image)
    #image = skimage.filters.gaussian(image, sigma=1, channel_axis=-1)
    height, width, channels = image.shape

    # Convert to grayscale to calculate brightness
    brightness = color.rgb2gray(image)

    # Generate random noise for grain
    noise = np.random.normal(0, 1, (height, width))

    # Adjust grain intensity inversely based on brightness
    grain_density = 1 - np.clip(brightness, 0.2, 1.0)  # Shadows have more grain
    noise = noise * grain_density * grain_intensity

    # Apply Gaussian blur for resolution-independent grain size
    noise = gaussian_filter(noise, sigma=blur_sigma)

    # Add chroma (color noise) by mixing noise with grayscale
    grain = np.stack([noise, noise, noise], axis=-1)  # Start with grayscale grain
    if chroma > 0:
        color_noise = np.random.normal(0, 1, (height, width, channels)) * grain_intensity
        grain = (1 - chroma) * grain + chroma * color_noise

    # Blend the grain into the original image using soft light
    grainy_image = image + grain
    grainy_image = np.clip(grainy_image, 0, 1)  # Ensure values are within valid range

    return img_as_ubyte(grainy_image)

import numpy as np

def match_histograms_custom(source, reference, nbins=64):
    """
    Match the histogram of the `source` image (grayscale) to that of the 
    `reference` image, using 'nbins' bins to build the histograms.

    Parameters
    ----------
    source : ndarray
        Grayscale image to transform.
    reference : ndarray
        Grayscale reference image whose histogram will be matched.
    nbins : int
        Number of bins to use when computing the histograms and CDFs.

    Returns
    -------
    matched : ndarray
        The source image transformed so that its histogram matches
        that of the reference.
    """

    # 1) Compute the histograms and normalized cumulative distributions for each:
    #    (We use the actual min/max range of each image to confine the bins.)
    s_min, s_max = source.min(), source.max()
    r_min, r_max = reference.min(), reference.max()

    source_hist, source_bin_edges = np.histogram(
        source.ravel(), bins=nbins, range=(s_min, s_max), density=True
    )
    reference_hist, reference_bin_edges = np.histogram(
        reference.ravel(), bins=nbins, range=(r_min, r_max), density=True
    )

    # Cumulative distribution function (CDF)
    source_cdf = np.cumsum(source_hist)
    source_cdf /= source_cdf[-1]  # normalize to [0..1]

    reference_cdf = np.cumsum(reference_hist)
    reference_cdf /= reference_cdf[-1]  # normalize to [0..1]

    # 2) Map each pixel's intensity in 'source' to a CDF value,
    #    then invert that through the reference image's CDF.
    #    We'll use 'np.interp' to do both forward and inverse lookups.
    
    # Bin centers (use edges to get the midpoints for interpolation):
    source_bin_centers = 0.5 * (source_bin_edges[:-1] + source_bin_edges[1:])
    reference_bin_centers = 0.5 * (reference_bin_edges[:-1] + reference_bin_edges[1:])

    # For each pixel in 'source', find its approximate CDF value:
    source_pixels_cdf_vals = np.interp(
        source.ravel(),                # values to interpolate
        source_bin_centers,            # x-coordinates of data points
        source_cdf                     # y-coordinates of data points
    )

    # Now invert that CDF value in the reference image's CDF:
    matched_pixels = np.interp(
        source_pixels_cdf_vals,         # x to find in reference_cdf
        reference_cdf,                  # known x-coordinates
        reference_bin_centers           # known y-coordinates
    )

    # Reshape back to the original image shape
    matched = matched_pixels.reshape(source.shape)
    return matched


import skimage
import numpy as np

import skimage.io as skio
import skimage.filters
import skimage.color
import matplotlib.pyplot as plt

def blur_edges(img):
    """
    Applies a blur effect to the input image, preserving edges.

    Parameters:
        img (ndarray): Input image in RGB format (H, W, 3).

    Returns:
        ndarray: Blurred image with edges preserved.
    """
    img = skimage.img_as_float(img)
    thresholded = skimage.color.rgb2gray(img) > 0.6
    edges = (skimage.filters.sobel(skimage.color.rgb2gray(img)) > 0.2) * thresholded
    edges = np.clip((skimage.filters.gaussian(edges, sigma=1) * 1.2), 0, 1)
    blured = skimage.filters.gaussian(img, sigma=3)

    # Expand dimensions of edges to match img and blured
    edges_expanded = np.expand_dims(edges, axis=-1)

    img = (img * (1 - edges_expanded) + (blured * edges_expanded))
    
    return skimage.img_as_ubyte(img)

def film(input_image, weights_path, resize_to=2048, lum=False):
    """
    Applies the film simulation pipeline to an input image.

    :param input_image: A file path (str) or a PIL.Image.Image object.
    :param weights_path: Path to the CycleGAN weights.
    :param resize_to: Resolution for inference (default=2048).
    :param lum: If True, skips additional processing and returns the raw CycleGAN output.
    :return: Processed image as a NumPy array.
    """
    # Handle input_image being a file path or PIL.Image.Image
    if isinstance(input_image, str):
        img = skio.imread(input_image)
    elif isinstance(input_image, Image.Image):
        img = np.array(input_image)
    else:
        raise ValueError("input_image must be a file path or a PIL.Image.Image object.")

    filmic = transform_image_A_to_B(input_image, weights_path, resize_to=resize_to)
    if lum:
        return filmic
    filmic = skimage.img_as_float(filmic)

    img = skimage.img_as_float(img)
    img = blur_edges(img)
    
    filmic = skimage.transform.resize(filmic, img.shape, anti_aliasing=True)
    img = skimage.img_as_float(img)

    img = match_histograms_custom(img, filmic, nbins=16)

    img_hsl = rgb_to_hsl_vectorized(img)
    filmic_hsl = rgb_to_hsl_vectorized(filmic)

    # Replace H and S channels from filmic to img
    img_hsl[..., 0], img_hsl[..., 2] = filmic_hsl[..., 0], filmic_hsl[..., 2]

    final = hsl_to_rgb_vectorized(img_hsl)
    final = skimage.img_as_ubyte(final)
    final = generate_film_grain(final, grain_intensity=0.1, chroma=0.3, blur_sigma=0.8)

    plt.figure(figsize=(20, 10))
    plt.imshow(final)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    
    return final