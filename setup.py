#!/usr/bin/env python3
"""Setup script for iPhone Film Emulator."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="iphone-film-emulator",
    version="1.0.0",
    description="Transform iPhone photos to look like classic film stocks using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/iphone-film-emulator",
    license="MIT",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    install_requires=requirements,
    
    extras_require={
        "web": ["gradio>=3.0.0"],
        "heic": ["pillow-heif"],
        "coreml": ["coremltools"],
        "dev": ["pytest", "flake8", "black"]
    },
    
    entry_points={
        "console_scripts": [
            "film-emulator=apps.cli.film_cli:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires=">=3.8",
    
    keywords="iphone, film, emulation, photography, ai, cyclegan, neural-style-transfer",
    
    project_urls={
        "Bug Reports": "https://github.com/yourusername/iphone-film-emulator/issues",
        "Source": "https://github.com/yourusername/iphone-film-emulator",
        "Documentation": "https://github.com/yourusername/iphone-film-emulator#readme",
    },
    
    include_package_data=True,
    zip_safe=False,
)