inference_size = 2048
name = "cinestill"
import sys
import torch
from models import create_model
from options.test_options import TestOptions

# Path to your pre-trained weights
weights_path = "/Users/yahyarahhawi/Developer/Film/weights/Final/cinestill_LR1024/80_net_G_A.pth"

# Override sys.argv to inject minimal CLI args for TestOptions
original_argv = sys.argv
sys.argv = [
    original_argv[0],
    '--dataroot', 'dummy',        # Needed to satisfy required '--dataroot' argument
    '--model', 'cycle_gan',       # Specifies the CycleGAN model
    '--dataset_mode', 'single',   # We are feeding images manually
    '--gpu_ids', '-1',            # Disable GPU explicitly to avoid CUDA dependency
]

# Parse the options
opt = TestOptions().parse()

# Restore sys.argv to its original state
sys.argv = original_argv

# Set additional options manually
opt.isTrain = False  # Indicates we're running in evaluation/test mode
opt.no_dropout = True  # No dropout for inference
opt.batch_size = 1  # Inference only works with batch size = 1
opt.load_size = inference_size  # Resize to 256x256 during preprocessing (adjust as needed)
opt.crop_size = inference_size  # Crop size must match load size for this example

# Set the device to MPS if available, otherwise CPU
device = torch.device("mps")
print(f"Using device: {device}")

# Create the CycleGAN model
cycleGAN = create_model(opt)
cycleGAN.eval()  # Set the model to evaluation mode

# Manually load netG_A weights
state_dict = torch.load(weights_path, map_location=device)
cycleGAN.netG_A.load_state_dict(state_dict)

# Move the model to the device
cycleGAN.netG_A = cycleGAN.netG_A.to(device)

# Access the generator (netG_A)
model_netG_A = cycleGAN.netG_A

# Print a summary to verify everything is working
print("CycleGAN model created successfully!")
print(f"Generator loaded: {model_netG_A}")
# Create a dummy input with the same shape as your input images
dummy_input = torch.randn(1, 3, inference_size, inference_size).to(device)

# Trace the model to produce a TorchScript version
traced_model = torch.jit.trace(model_netG_A, dummy_input)

# Save the TorchScript model (optional, for debugging)
traced_model_path = "./cycleGAN_traced.pt"
traced_model.save(traced_model_path)

print(f"TorchScript model saved to {traced_model_path}")
import coremltools as ct

# Convert the TorchScript model to Core ML
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, inference_size, inference_size))],
    minimum_deployment_target=ct.target.iOS13  # Specify deployment target if needed
)

# Save the Core ML model with the correct extension
coreml_model_path = f"./{name}.mlpackage"
coreml_model.save(coreml_model_path)

print(f"Core ML model saved to {coreml_model_path}")