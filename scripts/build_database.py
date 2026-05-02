"""
Script to pre-compute style embeddings for the retrieval database.

Randomly samples N images from the raw dataset, passes them through the
trained Imprint v2 model, and saves the resulting 128-D embeddings along
with metadata (artist, file path) to a PyTorch file for fast search.

Usage:
    python scripts/build_database.py --num_images 500
"""

import argparse
import os
import random
import sys

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.transforms import get_eval_transforms
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser(description="Build Style Retrieval Database")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print("Loading model...")
    model = ImprintModule.from_config(cfg)
    ckpt_path = "checkpoints/best.ckpt"
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}!")
        print("Using random weights (Search results will be meaningless until you add the real checkpoint).")
    
    model = model.to(device)
    model.eval()

    # 2. Gather Image Paths
    images_dir = cfg["data"]["images_dir"]
    all_images = []
    
    print(f"Scanning {images_dir} for artworks...")
    for artist_folder in os.listdir(images_dir):
        artist_path = os.path.join(images_dir, artist_folder)
        if not os.path.isdir(artist_path):
            continue
            
        for img_file in os.listdir(artist_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append({
                    "artist": artist_folder.replace("_", " "),
                    "filename": img_file,
                    "path": os.path.join(artist_path, img_file)
                })

    if not all_images:
        print(f"Error: No images found in {images_dir}")
        return

    # Subsample if necessary
    if len(all_images) > args.num_images:
        print(f"Randomly sampling {args.num_images} from {len(all_images)} total images...")
        random.seed(42)
        sample_images = random.sample(all_images, args.num_images)
    else:
        sample_images = all_images

    # 3. Compute Embeddings
    print("Computing Style Fingerprints...")
    preprocess = get_eval_transforms(image_size=cfg["data"].get("image_size", 224))
    
    embeddings = []
    metadata = []
    
    with torch.no_grad():
        for item in tqdm(sample_images):
            try:
                img = Image.open(item["path"]).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                out = model.encoder(tensor)
                
                embeddings.append(out.embedding.cpu().squeeze())
                metadata.append({
                    "artist": item["artist"],
                    "filename": item["filename"],
                    "path": item["path"]
                })
            except Exception as e:
                print(f"Skipping {item['filename']} due to error: {e}")

    # 4. Save Database
    if not embeddings:
        print("Failed to compute any embeddings.")
        return
        
    embeddings_tensor = torch.stack(embeddings)
    
    os.makedirs("data", exist_ok=True)
    db_path = "data/style_database.pt"
    
    torch.save({
        "embeddings": embeddings_tensor,
        "metadata": metadata
    }, db_path)
    
    print(f"\n✓ Successfully saved database to {db_path}")
    print(f"Database contains {embeddings_tensor.size(0)} artworks.")


if __name__ == "__main__":
    main()
