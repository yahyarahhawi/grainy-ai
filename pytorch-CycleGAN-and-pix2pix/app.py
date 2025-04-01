import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ExifTags
import numpy as np
import filmic  # ensure filmic.py is in your project and importable
import os

os.chdir(os.path.expanduser("~"))

def get_writable_checkpoint_dir():
    """Create a folder in the user's home directory for checkpoints."""
    home = os.path.expanduser("~")
    checkpoint_dir = os.path.join(home, "filmic_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

checkpoint_dir = get_writable_checkpoint_dir()
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_file.dat")

# Global variables
input_image_path = None
weights_path = None
input_image = None

def load_image():
    global input_image_path, input_image
    try:
        input_image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg"),
                ("JPEG Files", "*.jpeg"),
                ("All Files", "*.*")
            ]
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file dialog:\n{e}")
        return

    if input_image_path:
        try:
            input_image = Image.open(input_image_path).convert("RGB")
            display_image = input_image.copy()
            display_image.thumbnail((300, 300))
            tk_image = ImageTk.PhotoImage(display_image)
            image_label.config(image=tk_image)
            image_label.image = tk_image
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

def load_weights_file():
    global weights_path
    try:
        weights_path = filedialog.askopenfilename(
            title="Select weights file",
            filetypes=[
                ("PyTorch Weights", "*.pth"),
                ("All Files", "*.*")
            ]
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file dialog:\n{e}")
        return

    if weights_path:
        weights_label.config(text=weights_path.split("/")[-1])

def copy_image_to_clipboard(pil_image):
    """Copy a PIL Image to the macOS clipboard using AppKit."""
    try:
        import io
        from AppKit import NSPasteboard, NSPasteboardTypePNG, NSImage, NSApplication
        from Foundation import NSData

        app = NSApplication.sharedApplication()

        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        data = output.getvalue()
        output.close()

        nsdata = NSData.dataWithBytes_length_(data, len(data))
        nsimage = NSImage.alloc().initWithData_(nsdata)

        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        pasteboard.declareTypes_owner_([NSPasteboardTypePNG], None)
        pasteboard.setData_forType_(nsdata, NSPasteboardTypePNG)
    except Exception as e:
        messagebox.showerror("Clipboard Error", f"Could not copy image to clipboard:\n{e}")

def convert_image():
    global input_image, weights_path
    if input_image is None or not weights_path:
        messagebox.showwarning("Missing Input", "Please select both an image and a weights file.")
        return

    try:
        resize_value = int(resize_var.get())
        mode = mode_var.get()
        generator_type = generator_var.get()

        if mode == "transform":
            result = filmic.transform_image_A_to_B(
                input_image, weights_path, resize_to=resize_value, generator=generator_type
            )
        else:
            result = filmic.film2(input_image, weights_path, resize_to=resize_value, lum=False, generator=generator_type)
        
        result_image_full = Image.fromarray(result)
        if rotate_var.get():
            result_image_full = result_image_full.rotate(-90, expand=True)
        
        copy_image_to_clipboard(result_image_full)
        
        preview_image = result_image_full.copy()
        preview_image.thumbnail((1920, 1080))
        
        img_width, img_height = preview_image.size
        window_width = img_width + 20
        window_height = img_height + 20
        
        result_window = tk.Toplevel(root)
        result_window.title("Processed Image")
        result_window.geometry(f"{window_width}x{window_height}")
        
        tk_result = ImageTk.PhotoImage(preview_image)
        img_label = tk.Label(result_window, image=tk_result)
        img_label.pack(pady=10)
        result_window.image = tk_result

        messagebox.showinfo("Success", "Image processed and full-resolution image copied to clipboard!")
        
    except Exception as e:
        messagebox.showerror("Conversion Error", f"An error occurred:\n{e}")

# Setup main window
root = tk.Tk()
root.title("Filmic Converter")

# Create UI elements
load_img_btn = ttk.Button(root, text="Load Image", command=load_image)
load_img_btn.grid(row=0, column=0, padx=10, pady=10)

load_weights_btn = ttk.Button(root, text="Load Weights", command=load_weights_file)
load_weights_btn.grid(row=0, column=1, padx=10, pady=10)

weights_label = ttk.Label(root, text="No weights selected")
weights_label.grid(row=1, column=1, padx=10, pady=5)

image_label = ttk.Label(root)
image_label.grid(row=1, column=0, padx=10, pady=5)

# Option to choose processing mode
mode_var = tk.StringVar(value="transform")
transform_rb = ttk.Radiobutton(root, text="Transform (A to B)", variable=mode_var, value="transform")
film_rb = ttk.Radiobutton(root, text="Film Pipeline", variable=mode_var, value="film")
transform_rb.grid(row=2, column=0, padx=10, pady=5, sticky="w")
film_rb.grid(row=2, column=1, padx=10, pady=5, sticky="w")

# Generator selection dropdown (ResNet 9 vs. ResNet 15)
generator_var = tk.StringVar(value="resnet_9blocks")
generator_label = ttk.Label(root, text="Generator Type:")
generator_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
generator_menu = ttk.OptionMenu(root, generator_var, "resnet_15blocks", "resnet_9blocks", "resnet_15blocks")
generator_menu.grid(row=3, column=1, padx=10, pady=5, sticky="w")

# Add a drop-down menu for resize resolution
resize_var = tk.StringVar(value="256")
resize_label = ttk.Label(root, text="Resize To:")
resize_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
resize_menu = ttk.OptionMenu(root, resize_var, "256", "256", "512", "1024", "2048")
resize_menu.grid(row=4, column=1, padx=10, pady=5, sticky="w")

# Add a checkbox for rotation
rotate_var = tk.BooleanVar(value=True)
rotate_check = ttk.Checkbutton(root, text="Rotate 90° Clockwise", variable=rotate_var)
rotate_check.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

convert_btn = ttk.Button(root, text="Convert", command=convert_image)
convert_btn.grid(row=6, column=0, columnspan=2, padx=10, pady=20)

root.mainloop()