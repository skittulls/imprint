import io
import os
import sys

# Ensure imprint_v2 root is in sys.path so src imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.inference.similarity import StyleSimilarityEngine
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config

app = FastAPI(title="Imprint v2 Style Attribution API")

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
cfg = load_config(CONFIG_PATH)

# Initialize Model and load Checkpoint
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.ckpt")

# We build the model from the config first, then load weights if checkpoint exists
print("Building model architecture from config...")
model = ImprintModule.from_config(cfg)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
else:
    print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
    print("Using randomly initialized weights. Ensure you download your checkpoint!")

model = model.to(device)

# Initialize Inference Engine
engine = StyleSimilarityEngine(encoder=model.encoder, device=device)

# Image Preprocessing (uses exactly the same transforms as validation)
image_size = cfg["data"].get("image_size", 224)
preprocess = get_eval_transforms(image_size=image_size)


def process_uploaded_file(file: UploadFile) -> torch.Tensor:
    """Read an uploaded file and convert it to a preprocessed tensor."""
    img = Image.open(io.BytesIO(file.file.read())).convert('RGB')
    return preprocess(img)


@app.post("/similarity")
async def get_similarity(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    try:
        t1 = process_uploaded_file(image_a)
        t2 = process_uploaded_file(image_b)
        
        cos_sim, attention = engine.compute_similarity(t1, t2)
        
        return {
            "cosine_similarity": cos_sim,
            # We no longer rely on a simple 'confidence' derived from euclidean distance.
            # Cosine similarity > 0.6 is a strong stylistic match for our normalized embeddings.
            "match": cos_sim > 0.6,
            "attention_breakdown": attention
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(CHECKPOINT_PATH)}
