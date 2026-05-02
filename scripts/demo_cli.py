"""
Terminal-Based Interactive Presentation Demo for Imprint v2.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.transforms import get_eval_transforms
from src.inference.similarity import StyleSimilarityEngine
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def load_system():
    print(f"{Colors.OKCYAN}[*] Initializing Imprint v2 Architecture...{Colors.ENDC}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = load_config("configs/default.yaml")
    model = ImprintModule.from_config(cfg)
    
    ckpt_path = "checkpoints/best.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"{Colors.FAIL}[!] ERROR: Checkpoint not found at {ckpt_path}{Colors.ENDC}")
        sys.exit(1)
        
    print(f"{Colors.OKCYAN}[*] Loading Multi-Scale Gram Attention Weights...{Colors.ENDC}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    engine = StyleSimilarityEngine(model.encoder, device=device)
    preprocess = get_eval_transforms(image_size=cfg["data"].get("image_size", 224))
    
    db_path = "data/style_database.pt"
    if not os.path.exists(db_path):
        print(f"{Colors.FAIL}[!] ERROR: Database not found. Run build_database.py first.{Colors.ENDC}")
        sys.exit(1)
        
    print(f"{Colors.OKCYAN}[*] Loading Style Fingerprint Database...{Colors.ENDC}")
    db = torch.load(db_path, map_location=device)
    db_embeddings = F.normalize(db["embeddings"], p=2, dim=1)
    db_metadata = db["metadata"]
    
    return engine, preprocess, db_embeddings, db_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Path to the artwork to analyze")
    args = parser.parse_args()

    print_header("IMPRINT v2: TERMINAL DEMO")

    engine, preprocess, db_embeddings, db_metadata = load_system()
    
    print(f"\n{Colors.OKBLUE}>> Analyzing Query Artwork: {args.query}{Colors.ENDC}")
    try:
        img = Image.open(args.query).convert("RGB")
    except Exception as e:
        print(f"{Colors.FAIL}[!] Could not load image: {e}{Colors.ENDC}")
        return
        
    tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        print(f"{Colors.OKBLUE}>> Computing 128-D Style Fingerprint...{Colors.ENDC}")
        out = engine.encoder(tensor)
        query_emb = out.embedding.squeeze()
        
        print(f"{Colors.OKBLUE}>> Searching through {len(db_metadata)} published artworks...{Colors.ENDC}")
        sims = F.cosine_similarity(query_emb.unsqueeze(0), db_embeddings, dim=1)
        top5_sims, top5_idx = torch.topk(sims, k=5)
    
    print_header("TOP 5 STYLISTIC MATCHES")
    
    for i in range(5):
        idx = top5_idx[i].item()
        sim_score = top5_sims[i].item()
        meta = db_metadata[idx]
        
        artist = meta['artist'].title()
        filename = meta['filename']
        
        color = Colors.OKGREEN if i == 0 else Colors.ENDC
        bold = Colors.BOLD if i == 0 else ""
        
        print(f"{color}{bold}{i+1}. {artist:<30} | Match: {sim_score:>6.1%}{Colors.ENDC}")
        print(f"   {Colors.WARNING}File: {filename}{Colors.ENDC}\n")
        
    print(f"{Colors.OKCYAN}Demo Complete.{Colors.ENDC}")


if __name__ == "__main__":
    main()
