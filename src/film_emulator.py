"""
Core film emulation functionality.
This module provides the main API for transforming digital images to look like film.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from typing import Union, Optional

import skimage
from skimage import (
    io as skio,
    exposure,
    transform,
    filters,
    color,
    img_as_float,
    img_as_ubyte
)

from options.test_options import TestOptions
from models import create_model


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Film stock configurations with root weight paths
FILM_STOCKS = {
    "vision3500t": {
        "generator": "resnet_15blocks",
        "description": "Tungsten-balanced film for night photography",
        "weights_path": str(PROJECT_ROOT / "weights/Vision 3 500T/v3.pth"),
        "versions": ["v1.pth", "v2.pth", "v3.pth"]
    },
    "ektar100": {
        "generator": "resnet_9blocks", 
        "description": "Vibrant daylight film for landscapes",
        "weights_path": str(PROJECT_ROOT / "weights/Ektar 100/v3.pth"),
        "versions": ["v1.pth", "v2.pth", "v3.pth"]
    },
    "gold200": {
        "generator": "resnet_9blocks",
        "description": "Warm-toned consumer film for everyday photos",
        "weights_path": str(PROJECT_ROOT / "weights/Gold 200/v3.pth"),
        "versions": ["v1.pth", "v2.pth", "v3.pth"]
    },
    "portra400": {
        "generator": "resnet_9blocks",
        "description": "Professional portrait film with natural skin tones",
        "weights_path": str(PROJECT_ROOT / "weights/Portra 400/v3.pth"),
        "versions": ["v1.pth", "v2.pth", "v3.pth"]
    }
}


def find_alternative_weights(film_stock: str) -> Optional[str]:
    """
    Find alternative weight files for a film stock if the default is missing.
    
    Args:
        film_stock: The film stock name
        
    Returns:
        Path to alternative weights or None if none found
    """
    if film_stock not in FILM_STOCKS:
        return None
    
    # First try the root weights directory
    film_config = FILM_STOCKS[film_stock]
    film_name_mapping = {
        "vision3500t": "Vision 3 500T",
        "ektar100": "Ektar 100", 
        "gold200": "Gold 200",
        "portra400": "Portra 400"
    }
    
    if film_stock in film_name_mapping:
        film_dir = PROJECT_ROOT / "weights" / film_name_mapping[film_stock]
        if film_dir.exists():
            # Try available versions in preference order (v3, v2, v1, then highest)
            preferred_versions = ["v3.pth", "v2.pth", "v1.pth"]
            
            for version in preferred_versions:
                version_path = film_dir / version
                if version_path.exists():
                    return str(version_path)
            
            # If no preferred versions, try any available version
            weight_files = list(film_dir.glob("v*.pth"))
            if weight_files:
                # Sort by version number (highest first)
                def get_version_num(path):
                    try:
                        return int(path.stem[1:])  # Remove 'v' and convert to int
                    except:
                        return 0
                
                weight_files.sort(key=get_version_num, reverse=True)
                return str(weight_files[0])
    
    # If grainy-ai weights not found, try old_weights as backup
    old_base_dir = PROJECT_ROOT / "old_weights"
    if old_base_dir.exists():
        # Alternative directory mappings for old structure
        alternatives = {
            "gold200": ["new_gold", "g200", "git-gold-200", "gold_LR1024"],
            "portra400": ["P400", "portra400-15resnet", "p400", "portra", "portra_LR1024"],
            "vision3500t": ["cine800t 15resnet", "C500T", "cinestill", "cinestill_LR1024", "new_cinestill"],
            "ektar100": ["E100", "Fuji"]
        }
        
        if film_stock in alternatives:
            search_dirs = ["New Gen", "Old Gen/Final"]
            
            for search_dir in search_dirs:
                search_path = old_base_dir / search_dir
                if not search_path.exists():
                    continue
                    
                for alt_name in alternatives[film_stock]:
                    alt_path = search_path / alt_name
                    if alt_path.exists():
                        weight_files = list(alt_path.glob("*_net_G_A.pth"))
                        if weight_files:
                            def get_epoch(path):
                                try:
                                    return int(path.stem.split('_')[0])
                                except:
                                    return 0
                            
                            weight_files.sort(key=get_epoch, reverse=True)
                            
                            # Prefer files in 80-100 range, otherwise take the highest
                            for weight_file in weight_files:
                                epoch = get_epoch(weight_file)
                                if 80 <= epoch <= 100:
                                    return str(weight_file)
                            
                            return str(weight_files[0])
    
    return None


class FilmEmulator:
    """
    Main class for film emulation using CycleGAN models.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the film emulator.
        
        Args:
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        self.device = self._get_device(device)
        self._model_cache = {}
    
    def _get_device(self, device: str) -> str:
        """Automatically detect the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    @classmethod
    def available_stocks(cls) -> dict:
        """Return available film stocks and their descriptions."""
        return {k: v["description"] for k, v in FILM_STOCKS.items()}
    
    @classmethod
    def available_versions(cls, film_stock: str) -> list:
        """Return available versions for a specific film stock."""
        if film_stock in FILM_STOCKS:
            return FILM_STOCKS[film_stock].get("versions", [])
        return []
    
    def get_weights_path(self, film_stock: str, version: str = None) -> str:
        """
        Get the weights path for a specific film stock and version.
        
        Args:
            film_stock: The film stock name
            version: Specific version (e.g., "v1", "v1.pth", "v2.pth"). If None, uses default.
            
        Returns:
            Path to the weights file
        """
        if film_stock not in FILM_STOCKS:
            raise ValueError(f"Unknown film stock: {film_stock}")
        
        if version is None:
            # Use default path
            return FILM_STOCKS[film_stock]["weights_path"]
        else:
            # Normalize version format (add .pth if missing)
            if not version.endswith('.pth'):
                version = f"{version}.pth"
            
            # Construct path for specific version
            film_name_mapping = {
                "vision3500t": "Vision 3 500T",
                "ektar100": "Ektar 100", 
                "gold200": "Gold 200",
                "portra400": "Portra 400"
            }
            
            if film_stock in film_name_mapping:
                return str(PROJECT_ROOT / f"weights/{film_name_mapping[film_stock]}/{version}")
            else:
                return FILM_STOCKS[film_stock]["weights_path"]
    
    def transform(
        self,
        input_image: Union[str, Image.Image],
        film_stock: str = "portra400",
        version: str = None,
        weights_path: Optional[str] = None,
        strength: float = 1.0,
        resize_to: int = 1024
    ) -> np.ndarray:
        """
        Transform an image to look like a specific film stock.
        
        Args:
            input_image: Input image path or PIL Image
            film_stock: Film stock name (e.g., "portra400", "gold200")
            weights_path: Path to model weights (auto-determined if None)
            strength: Transformation strength (0.0 to 1.0)
            resize_to: Processing resolution
            
        Returns:
            Transformed image as numpy array
        """
        # Validate film stock
        if film_stock not in FILM_STOCKS:
            raise ValueError(f"Unknown film stock: {film_stock}. Available: {list(FILM_STOCKS.keys())}")
        
        # Load and prepare image
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Image file not found: {input_image}")
            input_image = Image.open(input_image).convert("RGB")
        
        input_image = ImageOps.exif_transpose(input_image)
        original_shape_hw = (input_image.height, input_image.width)
        
        # Determine weights path
        if weights_path is None:
            # Use version-specific path or default
            weights_path = self.get_weights_path(film_stock, version)
            
            # If default path doesn't exist, try to find alternatives
            if not os.path.exists(weights_path):
                alt_path = find_alternative_weights(film_stock)
                if alt_path:
                    weights_path = alt_path
                    print(f"💡 Using alternative weights: {weights_path}")
            
        if not os.path.exists(weights_path):
            print(f"❌ Model weights not found: {weights_path}")
            print(f"💡 Available weight directories in 'weights/':")
            weights_dir = PROJECT_ROOT / "weights"
            if weights_dir.exists():
                for subdir in weights_dir.iterdir():
                    if subdir.is_dir():
                        weight_files = list(subdir.glob("v*.pth"))
                        versions = [f.name for f in weight_files]
                        print(f"   - {subdir.name}/: {versions}")
            else:
                print("   - weights directory not found")
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        # Get generator architecture for this film stock
        generator = FILM_STOCKS[film_stock]["generator"]
        
        # Transform image using CycleGAN
        return self._transform_with_cyclegan(
            input_image, 
            weights_path, 
            generator, 
            strength, 
            resize_to, 
            original_shape_hw
        )
    
    def _transform_with_cyclegan(
        self,
        input_image: Image.Image,
        weights_path: str,
        generator: str,
        alpha: float,
        resize_to: int,
        original_shape_hw: tuple
    ) -> np.ndarray:
        """Internal method to perform CycleGAN transformation."""
        
        # Setup CycleGAN model options
        original_argv = sys.argv
        sys.argv = [
            original_argv[0], "--dataroot", "dummy", "--model", "cycle_gan",
            "--dataset_mode", "single", "--gpu_ids", "-1", "--netG", generator
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
        
        # Check cache first for performance
        cache_key = f"{generator}_{weights_path}"
        if cache_key in self._model_cache:
            model = self._model_cache[cache_key]
        else:
            # Create and load model
            model = create_model(opt)
            model.eval()
            
            # Load weights with proper device handling
            try:
                if self.device == "mps":
                    # For Apple Silicon, load to CPU first then move to MPS
                    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                else:
                    state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback without weights_only for older PyTorch versions
                if self.device == "mps":
                    state_dict = torch.load(weights_path, map_location="cpu")
                else:
                    state_dict = torch.load(weights_path, map_location=self.device)
            
            model.netG_A.load_state_dict(state_dict)
            
            # Cache the model for future use
            self._model_cache[cache_key] = model
            print(f"📦 Cached model: {cache_key}")
        
        # Prepare input tensor
        transform_pipeline = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform_pipeline(input_image).unsqueeze(0)
        
        # Move to appropriate device
        model.netG_A.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        # Generate transformed image
        with torch.no_grad():
            fake_B = model.netG_A(input_tensor)
        
        # Convert to numpy
        fake_B = (0.5 * (fake_B + 1.0)) * 255
        fake_B_numpy = fake_B.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Resize back to original shape
        fake_B_resized = skimage.transform.resize(
            fake_B_numpy,
            original_shape_hw,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)
        
        # Apply blending if strength < 1.0
        if alpha == 1.0:
            return fake_B_resized
        else:
            original_np = np.array(input_image.resize((original_shape_hw[1], original_shape_hw[0]))).astype(np.uint8)
            if original_np.shape != fake_B_resized.shape:
                raise ValueError(f"Shape mismatch: original {original_np.shape}, transformed {fake_B_resized.shape}")
            
            original_float = skimage.img_as_float(original_np)
            filmic_float = skimage.img_as_float(fake_B_resized)
            blended = np.clip((1 - alpha) * original_float + alpha * filmic_float, 0, 1)
            return skimage.img_as_ubyte(blended)


def soft_blend(original: np.ndarray, styled: np.ndarray, alpha: float) -> np.ndarray:
    """
    Soft blend two images with given alpha.
    
    Args:
        original: Original image array
        styled: Styled image array  
        alpha: Blending strength (0.0 to 1.0)
        
    Returns:
        Blended image array
    """
    original_float = skimage.img_as_float(original)
    styled_float = skimage.img_as_float(styled)
    blended = np.clip((1 - alpha) * original_float + alpha * styled_float, 0, 1)
    return skimage.img_as_ubyte(blended)


def apply_grain(image: np.ndarray, grain_level: str = "medium") -> np.ndarray:
    """
    Apply film grain effect to an image.
    
    Args:
        image: Input image array
        grain_level: Grain intensity ("low", "medium", "high")
        
    Returns:
        Image with grain effect
    """
    if grain_level.lower() == "none":
        return image
    
    # Grain parameters
    grain_params = {
        "low": {"amount": 0.02, "size": 0.8},
        "medium": {"amount": 0.04, "size": 1.0}, 
        "high": {"amount": 0.08, "size": 1.2}
    }
    
    params = grain_params.get(grain_level.lower(), grain_params["medium"])
    
    # Generate grain
    h, w = image.shape[:2]
    grain = np.random.normal(0, params["amount"], (h, w))
    grain = gaussian_filter(grain, sigma=params["size"])
    
    # Apply grain
    image_float = skimage.img_as_float(image)
    if len(image.shape) == 3:
        grain = grain[:, :, np.newaxis]
    
    grained = np.clip(image_float + grain, 0, 1)
    return skimage.img_as_ubyte(grained)


# Legacy function for backwards compatibility
def transform_image_A_to_B(
    input_image: Union[str, Image.Image],
    weights_path: str,
    device: str = "cuda",
    resize_to: int = 256,
    generator: str = "resnet_9blocks",
    alpha: float = 1.0
) -> np.ndarray:
    """
    Legacy function for backwards compatibility.
    Use FilmEmulator class for new code.
    """
    emulator = FilmEmulator(device=device)
    return emulator._transform_with_cyclegan(
        input_image if isinstance(input_image, Image.Image) else Image.open(input_image).convert("RGB"),
        weights_path,
        generator,
        alpha,
        resize_to,
        (input_image.height, input_image.width) if isinstance(input_image, Image.Image) else Image.open(input_image).size[::-1]
    )