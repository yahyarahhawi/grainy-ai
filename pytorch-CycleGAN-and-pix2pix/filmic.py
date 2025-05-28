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
    device: str = "mps",
    resize_to: int = 256,
    generator: str = "resnet_9blocks",
    alpha: float = 1.0  # new argument
) -> np.ndarray:
    """
    Same as before, but includes optional alpha blending between original and GAN output.
    """
    original_argv = sys.argv
    sys.argv = [
        original_argv[0],
        "--dataroot", "dummy",
        "--model", "cycle_gan",
        "--dataset_mode", "single",
        "--gpu_ids", "-1",
        "--netG", generator
    ]
    opt = TestOptions().parse()
    sys.argv = original_argv

    opt.isTrain = False
    opt.no_dropout = True
    opt.load_size = resize_to
    opt.crop_size = resize_to
    opt.serial_batches = True
    opt.num_threads = 0
    opt.batch_size = 1

    model = create_model(opt)
    model.eval()

    state_dict = torch.load(weights_path, map_location=device)
    model.netG_A.load_state_dict(state_dict)

    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")

    original_shape = input_image.size  # (Width, Height)

    transform_pipeline = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform_pipeline(input_image).unsqueeze(0)

    if device.lower() == "cuda" and torch.cuda.is_available():
        model.netG_A.to("cuda")
        input_tensor = input_tensor.to("cuda")
    elif device.lower() == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model.netG_A.to("mps")
        input_tensor = input_tensor.to("mps")
    else:
        model.netG_A.to("cpu")
        input_tensor = input_tensor.to("cpu")

    with torch.no_grad():
        fake_B = model.netG_A(input_tensor)

    fake_B = (0.5 * (fake_B + 1.0)) * 255
    fake_B_numpy = fake_B.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Resize GAN output to original shape
    original_shape_hw = (original_shape[1], original_shape[0])
    fake_B_resized = skimage.transform.resize(
        fake_B_numpy, original_shape_hw, anti_aliasing=True, preserve_range=True
    ).astype(np.uint8)

    # Perform blending if alpha != 1
    if alpha == 1.0:
        return fake_B_resized
    else:
        original_np = np.array(input_image.resize(original_shape_hw)).astype(np.uint8)
        original_float = skimage.img_as_float(original_np)
        filmic_float = skimage.img_as_float(fake_B_resized)
        blended = np.clip((1 - alpha) * original_float + alpha * filmic_float, 0, 1)
        return skimage.img_as_ubyte(blended)

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

def match_bc(target_img, reference_img):
    # Convert images to float for calculations
    target_img = skimage.img_as_float(target_img)
    reference_img = skimage.img_as_float(reference_img)
    
    # Process each channel separately
    matched = np.zeros_like(target_img)
    for channel in range(target_img.shape[2]):
        target_mean, target_std = target_img[..., channel].mean(), target_img[..., channel].std()
        reference_mean, reference_std = reference_img[..., channel].mean(), reference_img[..., channel].std()

        # Apply the adjustment formula
        matched[..., channel] = (target_img[..., channel] - target_mean) / target_std * reference_std + reference_mean
    
    # Clip to valid range and convert back to uint8
    matched = np.clip(matched, 0, 1)  # scikit-image uses range [0, 1] for float images
    return matched

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
def inspect_video(
    video_path, weights_path, resize_to=2048, lum=False, max_frames=-1, output_path="processed_video.mov"
):
    """
    Reads a video from 'video_path', processes each frame using the 'film(...)' pipeline,
    and saves the processed video to 'output_path' in .MOV format with H.264 codec at 15 Mbps.

    :param video_path:  Path to the input video file (e.g. 'input.mp4').
    :param weights_path: Path to your CycleGAN netG_A weights (e.g. '15_net_G_A.pth').
    :param resize_to:   Resolution to use inside CycleGAN (default=2048 in this example).
    :param lum:         If True, 'film(...)' will return only the direct cycleGAN output.
    :param max_frames:  Number of frames to process. If -1, processes the entire video.
    :param output_path: Path to save the processed video (e.g. 'processed_video.mov').
    """
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use H.264 codec for .MOV format
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec

    # Create VideoWriter with a high bitrate (15 Mbps)
    bitrate = 15000 * 1000  # 15 Mbps in bits per second
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=True
    )

    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break  # No more frames or failed to read

        # 1) Convert BGR (OpenCV) to RGB (PIL expects RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2) Convert to PIL Image
        frame_pil = Image.fromarray(frame_rgb)

        # 3) Apply your 'film(...)' pipeline
        processed_rgb = filmic.film(frame_pil, weights_path, resize_to=resize_to, lum=lum)

        # 4) Convert processed result (RGB) back to BGR for OpenCV
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)

        # 5) Write the processed frame to the output video
        out.write(processed_bgr)

        frame_count += 1
        # If max_frames is set and reached, stop processing
        if max_frames != -1 and frame_count >= max_frames:
            print(f"Processed max_frames={max_frames}, stopping.")
            break

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def film(input_image, weights_path, resize_to=2048, lum=False, grain = False, generator= "resnet_9blocks"):
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

    filmic = transform_image_A_to_B(input_image, weights_path, resize_to=resize_to, generator=generator)
    if lum:
        return filmic
    #filmic = skimage.filters.gaussian(filmic, sigma=3, channel_axis=-1)
    filmic = skimage.transform.resize(filmic, img.shape, anti_aliasing=True)

    filmic = skimage.img_as_float(filmic)

    img = skimage.img_as_float(img)
    img = blur_edges(img)
    
    img = skimage.img_as_float(img)

    #img = match_histograms_custom(img, filmic, nbins = 8)

    img_hsl = skimage.color.rgb2lab(img)
    filmic_hsl = skimage.color.rgb2lab(filmic)
    img_hsl[...,0] = match_histograms_custom(img_hsl[..., 0], filmic_hsl[..., 0], nbins=256)

    # Replace H and S channels from filmic to img
    img_hsl[..., 1], img_hsl[..., 2] = filmic_hsl[..., 1], filmic_hsl[..., 2]

    final = skimage.color.lab2rgb(img_hsl)
    final = skimage.img_as_ubyte(final)
    if not grain: return final
    final = generate_film_grain(final, grain_intensity=0.1, chroma=0.3, blur_sigma=0.8)

    """    plt.figure(figsize=(20, 10))
    plt.imshow(final)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()"""
    
    return final
