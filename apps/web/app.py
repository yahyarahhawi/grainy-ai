import gradio as gr
import numpy as np
from PIL import Image
import spaces
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from film_emulator import FilmEmulator, soft_blend, apply_grain

# ──────────────────────────────────────────────────────────────────
FILM_STOCKS = ["Vision 3 500T", "Ektar 100", "Gold 200", "Portra 400"]
VERSIONS    = ["v1", "v2", "v3"]

GEN_BY_STOCK = {
    "Vision 3 500T": "resnet_15blocks",   # 15-block model
    "Ektar 100"    : "resnet_9blocks",
    "Gold 200"     : "resnet_9blocks",
    "Portra 400"   : "resnet_9blocks",
}

GRAIN_LEVELS = ["None", "Low", "Medium", "Large"]

# resolution presets
RES_LABELS = ["Low", "Medium", "High"]
RES_PIXELS = {"Low": 1024, "Medium": 2048, "High": 4096}

# Global emulator instance for performance
print("🚀 Initializing Grainy AI Film Emulator...")
GLOBAL_EMULATOR = FilmEmulator(device="auto")
print(f"✅ Emulator ready on device: {GLOBAL_EMULATOR.device}")

# ──────────────────────────────────────────────────────────────────
# Heavy step – runs on remote GPU
# ──────────────────────────────────────────────────────────────────
@spaces.GPU
def run_gan(img: Image.Image, stock: str, version: str, res_label: str):
    size = RES_PIXELS[res_label]
    
    # Map display names to internal film stock names
    stock_mapping = {
        "Vision 3 500T": "vision3500t",
        "Ektar 100": "ektar100", 
        "Gold 200": "gold200",
        "Portra 400": "portra400"
    }
    
    film_stock = stock_mapping.get(stock, stock.lower().replace(" ", ""))
    
    # Transform using the global emulator instance (no reinitialization!)
    styled_np = GLOBAL_EMULATOR.transform(
        img,
        film_stock=film_stock,
        version=version,  # Will automatically add .pth if needed
        strength=1.0,
        resize_to=size
    )
    return styled_np, np.array(img)   # (styled, original)

# ──────────────────────────────────────────────────────────────────
# Fast step – CPU only (blend + optional grain)
# ──────────────────────────────────────────────────────────────────
def update_display(alpha, grain_level, orig_np, styled_np):
    if orig_np is None or styled_np is None:
        return None

    blended = soft_blend(orig_np, styled_np, alpha)

    # ⟵ skip grain generation entirely when level == "None"
    if grain_level == "None":
        return blended

    return apply_grain(blended, grain_level)

# ──────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Grainy AI – Film Emulator") as demo:
    gr.Markdown("## Grainy AI · upload, pick a film stock, tweak intensity & grain")

    with gr.Row():
        img_in  = gr.Image(type="pil", label="Upload image")
        img_out = gr.Image(label="Result")

    stock_dd = gr.Dropdown(FILM_STOCKS, value="Gold 200", label="Film stock")
    ver_dd   = gr.Dropdown(VERSIONS,    value="v1",       label="Model version")

    res_radio = gr.Radio(
        RES_LABELS,
        value="High",
        label="Resolution",
        interactive=True,
    )

    alpha_sl = gr.Slider(
        0, 2, step=0.25, value=1.0,
        label="Alpha (intensity)",
        interactive=True,
    )

    grain_dd = gr.Dropdown(
        GRAIN_LEVELS,
        value="None",                  # ← default now “None”
        label="Film grain",
        interactive=True,
    )

    gen_btn = gr.Button("Generate")

    # hidden caches
    orig_state   = gr.State()
    styled_state = gr.State()

    # heavy GPU call
    def generate(img, stock, version, res_label, alpha, grain_level):
        styled_np, orig_np = run_gan(img, stock, version, res_label)
        out = update_display(alpha, grain_level, orig_np, styled_np)
        return out, orig_np, styled_np

    gen_btn.click(
        generate,
        inputs  =[img_in, stock_dd, ver_dd, res_radio, alpha_sl, grain_dd],
        outputs =[img_out, orig_state, styled_state],
    )

    # fast CPU-only callbacks
    alpha_sl.change(
        update_display,
        inputs =[alpha_sl, grain_dd, orig_state, styled_state],
        outputs= img_out,
        show_progress=False,
    )
    grain_dd.change(
        update_display,
        inputs =[alpha_sl, grain_dd, orig_state, styled_state],
        outputs= img_out,
        show_progress=False,
    )

demo.queue(max_size=20)
demo.launch(server_name="0.0.0.0")