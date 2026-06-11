import os
import sys
import glob

import gradio as gr
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.transforms import get_eval_transforms
from src.inference.similarity import StyleSimilarityEngine
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config

# -----------------------------------------------------------------------------
# 1. Load System (Model & Database)
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config("configs/default.yaml")
model = ImprintModule.from_config(cfg)
ckpt_path = "checkpoints/best.ckpt"

if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
else:
    print("WARNING: Checkpoint not found! Using random weights.")

model = model.to(device)
engine = StyleSimilarityEngine(model.encoder, device=device)
preprocess = get_eval_transforms(image_size=cfg["data"].get("image_size", 224))

db_path = "data/style_database.pt"
if os.path.exists(db_path):
    db = torch.load(db_path, map_location=device)
    db_emb = F.normalize(db["embeddings"], p=2, dim=1)
    db_meta = db["metadata"]
else:
    print("WARNING: Database not found!")
    db_emb, db_meta = None, None

# Load example images dynamically
examples_dir = os.path.join(os.path.dirname(__file__), "examples")
# Use relative paths for Gradio to properly serve thumbnails
example_images = [os.path.relpath(p, start=os.getcwd()) for p in glob.glob(os.path.join(examples_dir, "*.jpg"))]
search_examples = [[img] for img in example_images]


# -----------------------------------------------------------------------------
# 2. Gradio Logic Functions
# -----------------------------------------------------------------------------
def search(query_img):
    if query_img is None or db_emb is None:
        return None
        
    img = Image.fromarray(query_img).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = engine.encoder(tensor)
        query_emb = out.embedding.squeeze()
        sims = F.cosine_similarity(query_emb.unsqueeze(0), db_emb, dim=1)
        top5_sims, top5_idx = torch.topk(sims, k=5)
        
    if top5_sims[0].item() < 0.40:
        gr.Warning("⚠️ OUT OF DISTRIBUTION: This artwork's style does not strongly match any known lineage in the database.")
        
    results = []
    for i in range(5):
        idx = top5_idx[i].item()
        meta = db_meta[idx]
        sim = top5_sims[i].item()
        
        try:
            match_img = Image.open(meta["path"]).convert("RGB")
            results.append((match_img, f"{meta['artist'].title()} ({sim:.1%})"))
        except Exception as e:
            print(f"Error loading {meta['path']}: {e}")
            
    return results

def analyze(img1, img2):
    if img1 is None or img2 is None:
        return None, None, None, None
        
    i1 = Image.fromarray(img1).convert("RGB")
    i2 = Image.fromarray(img2).convert("RGB")
    t1, t2 = preprocess(i1).to(device), preprocess(i2).to(device)
    
    with torch.no_grad():
        cos_sim, attn = engine.compute_similarity(t1, t2)
        heatmapA = engine.get_heatmap_overlay(t1, i1)
        heatmapB = engine.get_heatmap_overlay(t2, i2)
        
    df = pd.DataFrame({
        "Feature Scale": ["Layer 2 (Brushstrokes)", "Layer 3 (Motifs)", "Layer 4 (Composition)"],
        "Attention Weight": [
            attn["texture_and_brushstrokes"],
            attn["shapes_and_motifs"],
            attn["composition_and_layout"],
        ],
    })
    
    fig = px.bar(
        df, x="Attention Weight", y="Feature Scale", orientation="h", color="Feature Scale",
        color_discrete_sequence=["#9ca3af", "#6b7280", "#4b5563"],
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=14, color="#9ca3af"),
        title=dict(text="MSGMAtt: Attention Breakdown", font=dict(size=16, color="#d1d5db")),
        xaxis_title="Attention Weight",
        yaxis_title="",
    )
    
    verdict = (
        "HIGH PROBABILITY - Stylistic Match" if cos_sim > 0.7
        else "MODERATE - Possible Match" if cos_sim > 0.4
        else "LOW PROBABILITY - Distinct Styles"
    )
    
    if cos_sim < 0.4:
        gr.Info("OOD Detection: These artworks represent distinct stylistic lineages.")
    
    summary = f"""
    <div style="background-color: var(--block-background-fill); padding: 24px; border-radius: 8px; border: 1px solid var(--border-color-primary); text-align: center; margin-bottom: 24px;">
        <h3 style="margin: 0 0 8px 0; font-family: 'Inter', sans-serif; font-size: 0.9rem; color: var(--body-text-color-subdued); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Cosine Similarity</h3>
        <h1 style="margin: 8px 0; font-family: 'Inter', sans-serif; font-size: 3.5rem; color: var(--body-text-color); font-weight: 700;">{cos_sim:.1%}</h1>
        <p style="margin: 12px 0 0 0; font-size: 1.1rem; font-weight: 600; color: var(--body-text-color); font-family: 'Inter', sans-serif;">{verdict}</p>
    </div>
    """
    
    return summary, fig, heatmapA, heatmapB

# -----------------------------------------------------------------------------
# 3. Gradio UI 
# -----------------------------------------------------------------------------

theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*neutral_800",
    button_primary_background_fill_hover="*neutral_700",
    button_primary_text_color="white",
)

custom_css = """
.header-text {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-color-primary);
}
.header-text h1 {
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--body-text-color);
    margin-bottom: 0.5rem;
}
.header-text p {
    color: var(--body-text-color-subdued);
    font-size: 1.1rem;
}
.gallery-container {
    min-height: 400px;
}
"""

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(elem_classes="header-text"):
            gr.Markdown(
                "# Imprint: Image Style Attributor\n"
                "An academic prototype for few-shot style attribution using multi-scale Gram attention."
            )

    with gr.Tabs():
        with gr.Tab("Gallery Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_img = gr.Image(label="Query Artwork", type="numpy", height=400)
                    search_btn = gr.Button("Search Database", variant="primary", size="lg")
                    
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Top 5 Matches", columns=3, rows=2, object_fit="contain", elem_classes="gallery-container")
            
            if search_examples:
                gr.Examples(
                    examples=search_examples, 
                    inputs=[q_img],
                    outputs=[gallery],
                    fn=search,
                    run_on_click=True
                )
            
            search_btn.click(search, inputs=[q_img], outputs=[gallery])

        with gr.Tab("Deep Style Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    imgA = gr.Image(label="Artwork A", type="numpy", height=300)
                    hmA = gr.Image(label="XAI Heatmap A (Spatial Activations)", type="pil", height=300, interactive=False)
                with gr.Column(scale=1):
                    imgB = gr.Image(label="Artwork B", type="numpy", height=300)
                    hmB = gr.Image(label="XAI Heatmap B (Spatial Activations)", type="pil", height=300, interactive=False)
                with gr.Column(scale=1):
                    analyze_btn = gr.Button("Analyze Style & Generate Heatmaps", variant="primary", size="lg")
                    res_text = gr.HTML()
                    res_plot = gr.Plot()
                    
            analyze_btn.click(analyze, inputs=[imgA, imgB], outputs=[res_text, res_plot, hmA, hmB])

demo.launch(share=True, theme=theme, css=custom_css)
