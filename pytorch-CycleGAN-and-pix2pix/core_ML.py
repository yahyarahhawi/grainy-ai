#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CycleGAN → Core ML (mlprogram) with baked-in output scaling 0…255
----------------------------------------------------------------
• Keeps input as UI-friendly 0…255 RGB by using ImageType(scale=1/127.5, bias=-1)
• Adds a (x + 1) * 127.5 layer so the Core ML output is a displayable RGB image
  and no post-processing is needed in Swift.

Author: Yahya Rahhawi | 2025-06-10
"""

import sys, argparse, functools
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

# ---------- CLI ----------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Convert a CycleGAN generator weight file to Core ML with 0-255 output."
)
parser.add_argument("--inference_size", type=int, default=256,
                    help="Spatial size (HxW) of the model. Must match training size.")
parser.add_argument("--name", type=str, required=True,
                    help="Base name for the resulting .mlpackage")
parser.add_argument("--weights_path", type=str, required=True,
                    help="Path to the pre-trained PyTorch generator weights (.pth)")
args = parser.parse_args()

# ---------- CycleGAN options stub ----------------------------------------------
# The original repo uses its own Option parser, so we spoof the bare minimum

def count_resnet_blocks(state_dict):
    block_ids = set()
    for k in state_dict:
        if k.startswith("model.") and ".conv_block" in k:
            try:
                idx = int(k.split(".")[1])
                block_ids.add(idx)
            except ValueError:
                pass
    return len(block_ids)

original_argv = sys.argv
sys.argv = [
    original_argv[0],
    "--dataroot", "dummy",
    "--model", "cycle_gan",
    "--dataset_mode", "single",
    "--gpu_ids", "-1",
    "--netG", f"resnet_{count_resnet_blocks(torch.load(args.weights_path, map_location='cpu'))}blocks"
]
from options.test_options import TestOptions
from models import create_model
opt = TestOptions().parse()
sys.argv = original_argv

opt.isTrain  = False
opt.no_dropout = True
opt.batch_size = 1
opt.load_size  = args.inference_size
opt.crop_size  = args.inference_size

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Build & load generator --------------------------------------------
cycleGAN = create_model(opt)
cycleGAN.eval()
state_dict = torch.load(args.weights_path, map_location=device)
cycleGAN.netG_A.load_state_dict(state_dict)
gen = cycleGAN.netG_A.to(device).eval()

print("• Generator loaded")

# ---------- Wrap with 0-255 scaling --------------------------------------------
class To255(nn.Module):
    def forward(self, x):
        return (x + 1.0) * 127.5          # [-1,1] → [0,255]

model = nn.Sequential(gen, To255()).to(device).eval()

# ---------- Trace --------------------------------------------------------------
H = W = args.inference_size
dummy = torch.rand(1, 3, H, W, device=device) * 2 - 1   # [-1,1] dummy input
traced = torch.jit.trace(model, dummy, strict=False)

print("• TorchScript traced")

# ---------- Convert to Core ML --------------------------------------------------
rgb_bias  = [-1.0, -1.0, -1.0]
rgb_scale = 1 / 127.5

mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,

    # ---------- INPUT  (still has shape/bias/scale) ----------
    inputs=[ct.ImageType(
        name="input",
        shape=dummy.shape,
        bias=rgb_bias,
        scale=rgb_scale,
        color_layout="RGB")],

    # ---------- OUTPUT (no shape) ----------
    outputs=[ct.ImageType(
        name="output",
        color_layout="RGB")]
)

out_path = Path(f"{args.name}.mlpackage").resolve()
mlmodel.save(str(out_path))
print(f"✅ Core ML model saved to {out_path}")
