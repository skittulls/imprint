import io
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.transforms import get_eval_transforms
from src.inference.similarity import StyleSimilarityEngine
from src.training.lightning_module import ImprintModule
from src.utils.helpers import load_config

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Imprint v2",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for academic aesthetic
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { font-family: 'Georgia', serif; color: #2C3E50; }
    .stAlert { border-radius: 8px; }
    .css-1d391kg { padding-top: 2rem; }
    .stat-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Model and Database
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    device = torch.device("cpu")
    
    # 1. Load config and model
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    cfg = load_config(cfg_path)
    model = ImprintModule.from_config(cfg)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.ckpt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model_status = "Loaded Successfully"
    else:
        model_status = "WARNING: No checkpoint found. Using random weights."
        
    model = model.to(device)
    engine = StyleSimilarityEngine(model.encoder, device=device)
    preprocess = get_eval_transforms(image_size=cfg["data"].get("image_size", 224))
    
    # 2. Load Database
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "style_database.pt")
    if os.path.exists(db_path):
        db = torch.load(db_path, map_location=device)
        db_embeddings = F.normalize(db["embeddings"], p=2, dim=1) # Ensure normalized
        db_metadata = db["metadata"]
    else:
        db_embeddings = None
        db_metadata = None

    return engine, preprocess, db_embeddings, db_metadata, model_status


engine, preprocess, db_embeddings, db_metadata, status = load_system()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Imprint v2: Multi-Scale Gram Attention")
st.markdown("""
*An academic prototype for few-shot style attribution using texture correlations.*

Unlike traditional classification models that rely on fixed identity labels, **Imprint v2** maps artworks into a continuous 128-dimensional metric space. By extracting Gram matrices across three ResNet depths and aggregating them via self-attention (MSGMAtt), the model learns an interpretable "Style Fingerprint" that captures both fine brushstrokes and macro composition.
""")

if "WARNING" in status:
    st.error(status)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Database Retrieval (Search)", "⚖️ Deep Style Analysis (1-to-1)"])

# =============================================================================
# TAB 1: RETRIEVAL
# =============================================================================
with tab1:
    st.markdown("### Style Fingerprint Retrieval")
    st.write("Upload a query artwork to instantly search the pre-computed dataset for the top 5 closest stylistic matches.")
    
    if db_embeddings is None:
        st.warning("Retrieval database not found. Please run `python scripts/build_database.py` first.")
    else:
        query_file = st.file_uploader("Upload Query Artwork", type=["png", "jpg", "jpeg"], key="retrieval")
        
        if query_file:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(query_file, caption="Query Artwork", use_container_width=True)
                
            with col2:
                with st.spinner("Computing 128-D Fingerprint and searching database..."):
                    # Process query
                    img = Image.open(query_file).convert("RGB")
                    tensor = preprocess(img).unsqueeze(0)
                    
                    # Get embedding
                    out = engine.encoder(tensor)
                    query_emb = out.embedding.squeeze() # (128,)
                    
                    # Compute distances to all database embeddings
                    sims = F.cosine_similarity(query_emb.unsqueeze(0), db_embeddings, dim=1)
                    
                    # Get Top 5
                    top5_sims, top5_idx = torch.topk(sims, k=5)
                    
                    st.markdown("#### Top 5 Stylistic Matches")
                    
                    # Display Results
                    match_cols = st.columns(5)
                    for i in range(5):
                        idx = top5_idx[i].item()
                        sim_score = top5_sims[i].item()
                        meta = db_metadata[idx]
                        
                        with match_cols[i]:
                            try:
                                # Try to load the image to show it
                                match_img = Image.open(meta["path"])
                                st.image(match_img, use_container_width=True)
                            except:
                                st.write("*(Image missing)*")
                                
                            st.markdown(f"**{meta['artist'].title()}**")
                            st.caption(f"Similarity: {sim_score:.1%}")


# =============================================================================
# TAB 2: VERIFICATION & ATTENTION
# =============================================================================
with tab2:
    st.markdown("### Interpretability: MSGMAtt Breakdown")
    st.write("Compare two artworks to see not only *how similar* they are, but *why* they are similar based on the Multi-Scale Gram Attention module.")
    
    col1, col2 = st.columns(2)
    with col1:
        img1_file = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg"], key="imgA")
    with col2:
        img2_file = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg"], key="imgB")
        
    if img1_file and img2_file:
        c1, c2 = st.columns(2)
        img1 = Image.open(img1_file).convert("RGB")
        img2 = Image.open(img2_file).convert("RGB")
        c1.image(img1, caption="Artwork A", use_container_width=True)
        c2.image(img2, caption="Artwork B", use_container_width=True)
        
        with st.spinner("Analyzing Gram Matrix textures..."):
            t1 = preprocess(img1)
            t2 = preprocess(img2)
            
            cos_sim, attention = engine.compute_similarity(t1, t2)
            
        st.markdown("---")
        
        # Results area
        r1, r2 = st.columns([1, 2])
        
        with r1:
            st.markdown(f"""
            <div class="stat-box">
                <h3 style="margin-bottom: 0;">Cosine Similarity</h3>
                <h1 style="font-size: 4rem; color: #2E86C1; margin-top: 10px;">{cos_sim:.1%}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with r2:
            st.markdown("#### What drove this similarity?")
            st.write("The model dynamically weighted the importance of textures at different network depths.")
            
            # Plotly Bar Chart
            df = pd.DataFrame({
                "Feature Scale": ["Layer 2 (Fine Brushstrokes)", "Layer 3 (Shapes/Motifs)", "Layer 4 (Composition)"],
                "Attention Weight": [
                    attention["texture_and_brushstrokes"],
                    attention["shapes_and_motifs"],
                    attention["composition_and_layout"]
                ]
            })
            
            fig = px.bar(
                df, 
                x="Attention Weight", 
                y="Feature Scale", 
                orientation='h',
                color="Feature Scale",
                color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=200)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Imprint v2 | ResNet-18 Backbone | Triplet Margin Loss | Attentive Prototypical Few-Shot Verification")
