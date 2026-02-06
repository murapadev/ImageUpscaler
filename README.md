# Image Upscaler (Python)

A simple, powerful image upscaler using the **Swin2SR** model from Hugging Face Transformers. 
It supports **GPU acceleration (CUDA)** and **tiled processing** to handle high-resolution images without running out of memory.

## Features

- **2x Upscaling**: Transforms low-res images into higher resolution variants up to 2x.
- **GPU Support**: Automatically uses CUDA if available.
- **Smart Tiling**: Splits large images into tiles to prevent Out-Of-Memory (OOM) errors on limited VRAM.
- **Seamless Stitching**: Handles tile overlap to ensure no artifacts at tile boundaries.

## Installation

1. **Clone the repository** (or download the files).
2. **Install dependencies**:
   It is recommended to use a virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python3 upscale.py input.jpg
```
This will save the output as `input_upscaled.jpg`.

### Custom Output Path

```bash
python3 upscale.py input.jpg output.png
```

### Advanced Options

- `--model`: Specify a custom Hugging Face model ID (default: `caidas/swin2SR-classical-sr-x2-64`).
- `--tile-size`: Set the tile size for processing (default: `512`). Smaller values save memory but may be slower.
- `--tile-overlap`: Set the overlap between tiles (default: `32`).

```bash
# Example for low VRAM
python3 upscale.py input.jpg --tile-size 256
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for speed)
