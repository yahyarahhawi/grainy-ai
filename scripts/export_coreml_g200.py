import sys
import torch
import os
import subprocess
import coremltools as ct
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import create_model
from options.test_options import TestOptions

# ───────────────────────────────────────────────────────────────
# Settings
inference_size = 4096
weights_path = "/Users/yahyarahhawi/Developer/Film/grainy-ai/weights/Gold 200/v1.pth"
coreml_model_path = "./g200.mlpackage"
compiled_output_dir = "./compiled_model"

# ───────────────────────────────────────────────────────────────
# Step 1: Inject required CLI arguments for TestOptions
original_argv = sys.argv
sys.argv = [
    original_argv[0],
    '--dataroot', 'dummy',
    '--model', 'cycle_gan',
    '--dataset_mode', 'single',
    '--gpu_ids', '-1',
    '--netG', 'resnet_9blocks'
]
opt = TestOptions().parse()
sys.argv = original_argv

# ───────────────────────────────────────────────────────────────
# Step 2: Set additional options
opt.isTrain = False
opt.no_dropout = True
opt.batch_size = 1
opt.load_size = inference_size
opt.crop_size = inference_size

# ───────────────────────────────────────────────────────────────
# Step 3: Load model
device = torch.device("cpu")
print(f"Using device: {device}")

cycleGAN = create_model(opt)
cycleGAN.eval()

# Load weights
print(f"Loading weights from: {weights_path}")
state_dict = torch.load(weights_path, map_location=device)
cycleGAN.netG_A.load_state_dict(state_dict)
cycleGAN.netG_A = cycleGAN.netG_A.to(device)
model_netG_A = cycleGAN.netG_A
print("Model loaded successfully.")

# ───────────────────────────────────────────────────────────────
# Step 4: Trace the model
dummy_input = torch.randn(1, 3, inference_size, inference_size).to(device)
model_netG_A.eval()
traced_model = torch.jit.trace(model_netG_A, dummy_input)
print("Model tracing complete.")

# ───────────────────────────────────────────────────────────────
# Step 5: Convert to CoreML with float16 precision
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=dummy_input.shape)],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS15
)
coreml_model.save(coreml_model_path)
print(f"Core ML model saved to {coreml_model_path}")

# ───────────────────────────────────────────────────────────────
# Step 6: Compile with coremlc
if not os.path.exists(compiled_output_dir):
    os.makedirs(compiled_output_dir)

compile_command = [
    "xcrun", "coremlc", "compile", coreml_model_path, compiled_output_dir
]

print("Compiling Core ML model...")
try:
    subprocess.run(compile_command, check=True, capture_output=True, text=True)
    print(f"Compiled model saved to {compiled_output_dir}")
except subprocess.CalledProcessError as e:
    print("Error during model compilation:")
    print(e.stderr)
