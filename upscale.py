import argparse
import sys
import os
import gc
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from PIL import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

def cleanup_cuda():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def upscale_tile(model, processor, image, device):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Clean up immediate tensors
    del inputs
    del outputs
    cleanup_cuda()
    
    output = (output * 255.0).round().astype("uint8")
    output = output.transpose([1, 2, 0]) # C H W -> H W C
    return output

def process_in_tiles(model, processor, image, device, tile_size=256, padding=16):
    width, height = image.size
    scale = model.config.upscale
    
    target_height = int(height * scale)
    target_width = int(width * scale)
    
    # Create empty result array
    # Note: For very large images, this numpy array itself might be large.
    # 8k x 8k x 3 bytes ~= 192 MB. This is fine for RAM.
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    print(f"Processing ({width}x{height}) in tiles of size {tile_size} with padding {padding}...")

    total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    processed_count = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Coordinates for input crop (with padding)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(width, x + tile_size + padding)
            y_end = min(height, y + tile_size + padding)
            
            # Crop input tile
            tile = image.crop((x_start, y_start, x_end, y_end))
            
            # Process tile
            try:
                upscaled_tile = upscale_tile(model, processor, tile, device)
            except torch.OutOfMemoryError:
                print(f"\nError: OOM on tile ({x},{y}) with size {tile_size}. Suggest reducing --tile-size further.")
                cleanup_cuda()
                raise

            # Calculate crop coordinates for the upscaled tile to remove padding
            pad_left = x - x_start
            pad_top = y - y_start
            
            actual_tile_w = min(width, x + tile_size) - x
            actual_tile_h = min(height, y + tile_size) - y
            
            out_h, out_w, _ = upscaled_tile.shape
            
            crop_x_start = int(pad_left * scale)
            crop_y_start = int(pad_top * scale)
            
            crop_x_end = crop_x_start + int(actual_tile_w * scale)
            crop_y_end = crop_y_start + int(actual_tile_h * scale)
            
            # Safety clamp
            crop_x_end = min(crop_x_end, out_w)
            crop_y_end = min(crop_y_end, out_h)

            valid_output = upscaled_tile[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            
            # Place in result
            out_x = int(x * scale)
            out_y = int(y * scale)
            
            out_h_final, out_w_final, _ = valid_output.shape
            
            result[out_y:out_y+out_h_final, out_x:out_x+out_w_final] = valid_output
            
            processed_count += 1
            print(f"\rProgress: {processed_count}/{total_tiles} tiles", end="", flush=True)

    print("\nStitching complete.")
    return Image.fromarray(result)

def upscale_image(input_path, output_path, model_name="caidas/swin2SR-classical-sr-x2-64", tile_size=256, tile_overlap=16):
    cleanup_cuda()
    print(f"Loading model: {model_name}...")
    try:
        processor = Swin2SRImageProcessor.from_pretrained(model_name)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    print(f"Loading image: {input_path}...")
    try:
        image = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    # Clean up before processing
    cleanup_cuda()

    # Attempt processing
    use_tiling = False
    
    # 1MP threshold for proactive tiling
    if image.width * image.height > 1024*1024:
        print("Image > 1MP, using tiled processing proactively.")
        use_tiling = True
    
    if not use_tiling:
        try:
            print("Attempting full image processing...")
            inputs = processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            
            # Clean up
            del inputs
            del outputs
            cleanup_cuda()
            
            output = (output * 255.0).round().astype("uint8")
            output = output.transpose([1, 2, 0])
            
            # Handle cropping for full image
            orig_w, orig_h = image.size
            scale = model.config.upscale
            expected_w = int(orig_w * scale)
            expected_h = int(orig_h * scale)
            output_image = Image.fromarray(output[:expected_h, :expected_w])

        except torch.OutOfMemoryError:
            print("OOM detected during full processing. Switching to tiled processing...")
            # Ensure variables are gone
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs
            cleanup_cuda()
            use_tiling = True

    if use_tiling:
        current_tile_size = tile_size
        while current_tile_size >= 64:
            try:
                print(f"Starting tiled processing with tile size: {current_tile_size}")
                output_image = process_in_tiles(model, processor, image, device, tile_size=current_tile_size, padding=tile_overlap)
                break # Success!
            except (torch.OutOfMemoryError, RuntimeError) as e:
                # Check for OOM in RuntimeError message just in case
                if isinstance(e, RuntimeError) and "out of memory" not in str(e):
                    raise e
                    
                print(f"\nOOM detected with tile size {current_tile_size}. clearing cache and retrying...")
                cleanup_cuda()
                current_tile_size = current_tile_size // 2
                if current_tile_size < 64:
                    print("Tile size too small. Aborting.")
                    raise e

    print(f"Saving upscaled image to: {output_path}")
    output_image.save(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale images using Swin2SR")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", nargs="?", help="Path to output image")
    parser.add_argument("--model", default="caidas/swin2SR-classical-sr-x2-64", help="HuggingFace model ID")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for processing (default: 256)")
    parser.add_argument("--tile-overlap", type=int, default=16, help="Overlap padding for tiles (default: 16)")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not output_path:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_upscaled{ext}"

    upscale_image(input_path, output_path, args.model, args.tile_size, args.tile_overlap)
