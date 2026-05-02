import os
import sys

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
        
    results = []
    for i in range(5):
        idx = top5_idx[i].item()
        meta = db_meta[idx]
        sim = top5_sims[i].item()
        
        # We explicitly open the image in memory so Gradio's path security doesn't block it
        try:
            match_img = Image.open(meta["path"]).convert("RGB")
            results.append((match_img, f"{meta['artist'].title()} ({sim:.1%})"))
        except Exception as e:
            print(f"Error loading {meta['path']}: {e}")
            
    return results

def analyze(img1, img2):
    if img1 is None or img2 is None:
        return None, None
        
    i1 = Image.fromarray(img1).convert("RGB")
    i2 = Image.fromarray(img2).convert("RGB")
    t1, t2 = preprocess(i1).to(device), preprocess(i2).to(device)
    
    with torch.no_grad():
        cos_sim, attn = engine.compute_similarity(t1, t2)
        
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
        color_discrete_sequence=["#E63946", "#F4A261", "#2A9D8F"],
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="#f8f6f1",
        plot_bgcolor="#f8f6f1",
        font=dict(family="Georgia, serif", size=14, color="#1a1a1a"),
        title="MSGMAtt: Attention Breakdown",
        xaxis_title="Attention Weight",
        yaxis_title="",
    )
    
    verdict = (
        "🟢 HIGH PROBABILITY (Stylistic Match)" if cos_sim > 0.7
        else "🟡 MODERATE (Possible Match)" if cos_sim > 0.4
        else "🔴 LOW PROBABILITY (Different Styles)"
    )
    
    recommendation = "Select this image for attribution." if cos_sim > 0.7 else "Do not select this image. It is stylistically distinct."
    
    summary = f"""
    <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #ccc; text-align: center; margin-bottom: 20px;">
        <h3 style="margin: 0; font-family: 'Georgia', serif; font-size: 1.2rem; color: #555;">Cosine Similarity</h3>
        <h1 style="margin: 10px 0; font-family: 'Georgia', serif; font-size: 3.5rem; color: #1a1a1a;">{cos_sim:.1%}</h1>
        <p style="margin: 5px 0; font-size: 1.1rem; font-weight: bold;">{verdict}</p>
        <p style="margin: 0; font-size: 1rem; color: #666; font-style: italic;">{recommendation}</p>
    </div>
    """
    
    return summary, fig


# -----------------------------------------------------------------------------
# 3. Gradio UI & Custom CSS
# -----------------------------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

body, .gradio-container {
    font-family: 'EB Garamond', Georgia, serif !important;
    background-color: #f8f6f1 !important;
    color: #1a1a1a !important;
}

#title-block {
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 1rem;
    margin-bottom: 1.8rem;
}

#title-block h1 {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #1a1a1a !important;
    margin-bottom: 0.25rem !important;
}

#title-block p {
    font-size: 0.88rem !important;
    color: #1a1a1a !important;
    font-style: italic !important;
    opacity: 0.6;
    margin: 0 !important;
}

/* Tab nav bar */
#main-tabs > .tab-nav {
    border-bottom: 1px solid #1a1a1a !important;
    background: transparent !important;
    gap: 0 !important;
}

#main-tabs > .tab-nav button {
    font-family: 'Libre Baskerville', Georgia, serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #1a1a1a !important;
    opacity: 0.45;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.4rem !important;
    margin-bottom: -1px;
    transition: opacity 0.15s;
}

#main-tabs > .tab-nav button:hover {
    opacity: 0.75 !important;
}

#main-tabs > .tab-nav button.selected {
    opacity: 1 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #1a1a1a !important;
}

/* Always show all tab content panels */
#main-tabs > .tabitem {
    display: block !important;
}

/* Lay the two tab panels side by side */
#main-tabs {
    display: flex !important;
    flex-direction: column !important;
}

#main-tabs > .tab-nav {
    display: flex !important;
    flex-direction: row !important;
}

#main-tabs > div:not(.tab-nav) {
    display: flex !important;
    flex-direction: row !important;
    gap: 2rem !important;
}

#main-tabs > div:not(.tab-nav) > .tabitem {
    flex: 1 !important;
    border-right: 1px solid rgba(26,26,26,0.12) !important;
    padding-right: 2rem !important;
}

#main-tabs > div:not(.tab-nav) > .tabitem:last-child {
    border-right: none !important;
    padding-right: 0 !important;
}

/* Buttons */
button.lg.primary {
    background-color: #1a1a1a !important;
    color: #f8f6f1 !important;
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 1px !important;
    padding: 0.45rem 1.4rem !important;
    margin-top: 0.6rem !important;
}

button.lg.primary:hover {
    opacity: 0.8 !important;
}

label > span {
    font-size: 0.78rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    opacity: 0.55 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    with gr.Column(elem_id="title-block"):
        gr.Markdown(
            "# IMPRINT: Image Style Attributor\n"
            "An academic prototype for few-shot style attribution using multi-scale Gram attention."
        )

    with gr.Tabs(elem_id="main-tabs"):
        with gr.Tab("Search"):
            with gr.Row():
                q_img = gr.Image(label="Query Artwork")
                gallery = gr.Gallery(label="Top 5 Matches", columns=5)
            search_btn = gr.Button("Search Database", variant="primary")
            search_btn.click(search, inputs=[q_img], outputs=[gallery])

        with gr.Tab("Deep Style Analysis"):
            with gr.Row():
                imgA = gr.Image(label="Artwork A")
                imgB = gr.Image(label="Artwork B")
            analyze_btn = gr.Button("Analyze Style", variant="primary")
            res_text = gr.HTML()
            res_plot = gr.Plot()
            analyze_btn.click(analyze, inputs=[imgA, imgB], outputs=[res_text, res_plot])

# Launch with share=True to bypass Lightning proxy issues
demo.launch(share=True)
