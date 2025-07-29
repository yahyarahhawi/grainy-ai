#!/usr/bin/env python3
"""
Command-line interface for iPhone Film Emulator.
Transform your iPhone photos to look like classic film stocks.
"""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from film_emulator import FilmEmulator, FILM_STOCKS


def main():
    parser = argparse.ArgumentParser(
        description="Transform iPhone photos to look like classic film stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg --style portra400
  %(prog)s IMG_1234.HEIC --style gold200 --output film_photo.jpg
  %(prog)s photos/ --style vision3500t --batch
  %(prog)s photo.jpg --style ektar100 --strength 0.8 --grain medium
        """
    )
    
    parser.add_argument(
        "input",
        help="Input image file or directory"
    )
    
    parser.add_argument(
        "--style", "--film-stock",
        choices=list(FILM_STOCKS.keys()),
        default="portra400",
        help="Film stock to emulate (default: portra400)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: adds '_film' suffix)"
    )
    
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Transformation strength (0.0-1.0, default: 1.0)"
    )
    
    parser.add_argument(
        "--grain",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="Add film grain effect (default: none)"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Processing resolution (default: 1024)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Processing device (default: auto)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in directory"
    )
    
    parser.add_argument(
        "--weights",
        help="Custom path to model weights"
    )
    
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available film stocks and exit"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    # List available styles
    if args.list_styles:
        print("Available film stocks:")
        for stock, desc in FilmEmulator.available_stocks().items():
            print(f"  {stock:<12} - {desc}")
        return 0
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return 1
    
    # Validate strength
    if not 0.0 <= args.strength <= 1.0:
        print("Error: Strength must be between 0.0 and 1.0", file=sys.stderr)
        return 1
    
    # Initialize emulator
    try:
        emulator = FilmEmulator(device=args.device)
        if not args.quiet:
            print(f"Using device: {emulator.device}")
            print(f"Film stock: {args.style} - {FilmEmulator.available_stocks()[args.style]}")
    except Exception as e:
        print(f"Error initializing emulator: {e}", file=sys.stderr)
        return 1
    
    # Process images
    if input_path.is_file():
        return process_single_image(args, emulator, input_path)
    elif input_path.is_dir() and args.batch:
        return process_directory(args, emulator, input_path)
    else:
        print("Error: Use --batch flag to process directories", file=sys.stderr)
        return 1


def process_single_image(args, emulator, input_path):
    """Process a single image file."""
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        suffix = input_path.suffix if input_path.suffix.lower() != '.heic' else '.jpg'
        output_path = input_path.parent / f"{stem}_{args.style}{suffix}"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if not args.quiet:
            print(f"Processing: {input_path}")
        
        # Transform image
        result = emulator.transform(
            str(input_path),
            film_stock=args.style,
            weights_path=args.weights,
            strength=args.strength,
            resize_to=args.resolution
        )
        
        # Apply grain if requested
        if args.grain != "none":
            from film_emulator import apply_grain
            result = apply_grain(result, args.grain)
        
        # Save result
        Image.fromarray(result).save(output_path, quality=95)
        
        if not args.quiet:
            print(f"Saved: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}", file=sys.stderr)
        return 1


def process_directory(args, emulator, input_dir):
    """Process all images in a directory."""
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for i, img_path in enumerate(image_files, 1):
        
        # Determine output path
        stem = img_path.stem
        suffix = img_path.suffix if img_path.suffix.lower() != '.heic' else '.jpg'
        output_path = input_dir / f"{stem}_{args.style}{suffix}"
        
        try:
            if not args.quiet:
                print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
            
            # Transform image
            result = emulator.transform(
                str(img_path),
                film_stock=args.style,
                weights_path=args.weights,
                strength=args.strength,
                resize_to=args.resolution
            )
            
            # Apply grain if requested
            if args.grain != "none":
                from film_emulator import apply_grain
                result = apply_grain(result, args.grain)
            
            # Save result
            Image.fromarray(result).save(output_path, quality=95)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", file=sys.stderr)
    
    if not args.quiet:
        print(f"Successfully processed {success_count}/{len(image_files)} images")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())