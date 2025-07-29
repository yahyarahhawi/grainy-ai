#!/usr/bin/env python3
"""
Test script to verify iPhone Film Emulator imports work correctly.
Run this before using the notebooks to ensure everything is set up properly.
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports...")
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL (Pillow)")
    except ImportError as e:
        print(f"❌ PIL (Pillow): {e}")
        return False
    
    return True

def test_ml_imports():
    """Test ML framework imports"""
    print("\n🔍 Testing ML framework imports...")
    try:
        import torch
        print(f"✅ torch (version: {torch.__version__})")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ torchvision (version: {torchvision.__version__})")
    except ImportError as e:
        print(f"❌ torchvision: {e}")
        return False
    
    try:
        import skimage
        print(f"✅ scikit-image (version: {skimage.__version__})")
    except ImportError as e:
        print(f"❌ scikit-image: {e}")
        return False
    
    return True

def test_project_structure():
    """Test that project structure is correct"""
    print("\n🔍 Testing project structure...")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    
    # Check for key directories
    src_dir = current_dir / "src"
    if not src_dir.exists():
        print(f"❌ src directory not found at {src_dir}")
        print("💡 Make sure you're running this from the project root directory")
        return False
    print("✅ src directory found")
    
    # Check for key files
    key_files = [
        "src/film_emulator.py",
        "src/models/__init__.py", 
        "src/options/__init__.py",
        "src/utils/__init__.py",
        "examples/inference.ipynb"
    ]
    
    for file_path in key_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            return False
    
    return True

def test_film_emulator_import():
    """Test importing the FilmEmulator class"""
    print("\n🔍 Testing FilmEmulator import...")
    
    # Add src to path
    src_path = Path.cwd() / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        from film_emulator import FilmEmulator, FILM_STOCKS
        print("✅ FilmEmulator imported successfully")
        
        # Test creating instance
        emulator = FilmEmulator(device="cpu")  # Use CPU to avoid device issues
        print(f"✅ FilmEmulator instance created (device: {emulator.device})")
        
        # Test available stocks
        stocks = emulator.available_stocks()
        print(f"✅ Available film stocks: {list(stocks.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ FilmEmulator import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 iPhone Film Emulator - Import Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Basic imports", test_basic_imports),
        ("ML framework imports", test_ml_imports), 
        ("Project structure", test_project_structure),
        ("FilmEmulator import", test_film_emulator_import)
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("📚 You can now run the Jupyter notebooks in the examples/ directory")
        print("💡 Start with: jupyter notebook examples/inference.ipynb")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("💡 Common fixes:")
        print("   • Install dependencies: pip install -r requirements.txt")
        print("   • Run from project root directory (where src/ is located)")
        print("   • Check that all files were moved correctly")

if __name__ == "__main__":
    main()