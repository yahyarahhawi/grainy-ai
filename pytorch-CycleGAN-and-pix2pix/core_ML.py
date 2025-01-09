import sys
import os
import torch
from models import create_model
from options.test_options import TestOptions
import coremltools as ct

def convert_to_coreml(weights_path, inference_size):
    # Extract the base name of the weights file (without extension) to use as the model name
    model_name = os.path.splitext(os.path.basename(weights_path))[0]

    # Override sys.argv to inject minimal CLI args for TestOptions
    sys.argv = [
        sys.argv[0],
        '--dataroot', 'dummy',        # Needed to satisfy required '--dataroot' argument
        '--model', 'cycle_gan',       # Specifies the CycleGAN model
        '--dataset_mode', 'single',   # We are feeding images manually
        '--gpu_ids', '0',             # Assume GPU is available and use it
    ]

    # Parse the options
    opt = TestOptions().parse()

    # Set additional options manually
    opt.isTrain = False  # Indicates we're running in evaluation/test mode
    opt.no_dropout = True  # No dropout for inference
    opt.batch_size = 1  # Inference only works with batch size = 1
    opt.load_size = inference_size  # Resize during preprocessing
    opt.crop_size = inference_size  # Crop size must match load size for this example

    # Set the device to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    # Ensure the output directory exists
    output_dir = "core_ml_models"
    os.makedirs(output_dir, exist_ok=True)

    # Set the Core ML model path
    coreml_model_path = os.path.join(output_dir, f"{model_name}.mlpackage")

    # Convert the TorchScript model to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=(1, 3, inference_size, inference_size))],
        minimum_deployment_target=ct.target.iOS13  # Specify deployment target if needed
    )

    # Save the Core ML model in the designated directory
    coreml_model.save(coreml_model_path)

    print(f"Core ML model saved to {coreml_model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <weights_path> <inference_size>")
        sys.exit(1)

    weights_path = sys.argv[1]
    inference_size = int(sys.argv[2])

    convert_to_coreml(weights_path, inference_size)