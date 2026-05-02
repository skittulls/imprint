"""
Inference engine for computing style similarity between two images.

Wraps the Imprint v2 StyleEncoder to provide:
1. Cosine similarity score [-1, 1].
2. Attention weights for both images, showing whether the model
   relied more on texture (layer2), motifs (layer3), or layout (layer4).
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.models.encoder import StyleEncoder


class StyleSimilarityEngine:
    def __init__(self, encoder: StyleEncoder, device: torch.device = None):
        """
        Args:
            encoder: Pre-trained StyleEncoder.
            device:  Device to run inference on.
        """
        self.device = device or torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def compute_similarity(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute style similarity between two images.

        Args:
            image_a: Tensor of shape (C, H, W)
            image_b: Tensor of shape (C, H, W)

        Returns:
            similarity: Cosine similarity score [-1, 1]
            attention_breakdown: Dictionary showing average attention
                                 weights across both images.
        """
        # Add batch dimension
        if image_a.dim() == 3:
            image_a = image_a.unsqueeze(0)
        if image_b.dim() == 3:
            image_b = image_b.unsqueeze(0)

        image_a = image_a.to(self.device)
        image_b = image_b.to(self.device)

        # Forward pass
        out_a = self.encoder(image_a)
        out_b = self.encoder(image_b)

        # Compute Cosine Similarity (embeddings are already L2 normalized)
        cos_sim = F.cosine_similarity(out_a.embedding, out_b.embedding, dim=1).item()

        # Average the attention weights across both images to provide a single
        # 'explanation' of what the model focused on for this comparison.
        attn_a = out_a.attention_weights.squeeze()
        attn_b = out_b.attention_weights.squeeze()
        avg_attn = (attn_a + attn_b) / 2.0

        # Map to human-readable names
        attention_breakdown = {
            "texture_and_brushstrokes": float(avg_attn[0]),
            "shapes_and_motifs": float(avg_attn[1]),
            "composition_and_layout": float(avg_attn[2]),
        }

        return cos_sim, attention_breakdown
