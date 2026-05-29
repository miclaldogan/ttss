"""Temporal Threat Scoring System (TTSS): end-to-end ViT + BiLSTM training module.

Wraps a partially-unfrozen VitSceneEncoder and a BiLSTMThreatPredictor into a
single nn.Module so both can be optimised jointly with differential learning
rates (ViT fine-tune LR is typically 10× lower than the BiLSTM LR).

Input:  (B, T, 3, H, W)  — ImageNet-normalised frames, already resized to
                            image_size × image_size (default 224×224).
        yolo_features:     Optional (B, T, 8) pre-computed YOLO detection
                           features.  When provided they are concatenated with
                           the ViT CLS embeddings before the BiLSTM (total
                           feature dim = 776); when omitted only ViT features
                           are used (dim = 768).
Output: ThreatPrediction
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ttss.models.detection.vit_scene import VitSceneEncoder
from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor, ThreatPrediction


class EndToEndThreatModel(nn.Module):
    """Joint ViT (partial fine-tune) + BiLSTM threat prediction model.

    Design notes
    ------------
    * ViT blocks 0..(12 - num_unfreeze_blocks - 1) are frozen — their
      parameters receive no gradient and their weights never change.
    * The last *num_unfreeze_blocks* ViT transformer blocks + final LayerNorm
      are trainable and optimised with a lower LR (see TTSSTrainer param groups).
    * YOLO feature extraction is fully offline (pre-computed); the 8-dim
      detection feature vector is concatenated with the 768-dim ViT CLS token.
    * BiLSTM input_dim must match: 776 (with YOLO) or 768 (ViT only).
    """

    def __init__(
        self,
        vit_encoder: VitSceneEncoder,
        bilstm: BiLSTMThreatPredictor,
    ) -> None:
        super().__init__()
        self.vit = vit_encoder
        self.bilstm = bilstm

    def trainable_vit_params(self) -> list[nn.Parameter]:
        """Return only the ViT parameters that require gradients."""
        return [p for p in self.vit.parameters() if p.requires_grad]

    def bilstm_params(self) -> list[nn.Parameter]:
        """Return all BiLSTM parameters."""
        return list(self.bilstm.parameters())

    def forward(
        self,
        frames_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        yolo_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> ThreatPrediction:
        """Run ViT → (concat YOLO) → BiLSTM for a batch of frame sequences.

        ``frames_input`` can be either:
        * A plain tensor ``(B, T, 3, H, W)`` — YOLO features optional via kwarg.
        * A tuple ``(frames, yolo_features)`` — used when the trainer passes the
          packed input through ``model(x)`` where ``x = (frames, yolo_feats)``.

        Args:
            frames_input:  (B, T, 3, H, W) tensor, or (frames, yolo_features) tuple.
            yolo_features: Optional (B, T, 8) pre-computed YOLO features.
            mask:          Optional (B, T) bool mask for BiLSTM padding.
        """
        if isinstance(frames_input, (tuple, list)):
            frames, yolo_features = frames_input[0], frames_input[1]
        else:
            frames = frames_input

        B, T = frames.shape[:2]
        frames_flat = frames.view(B * T, *frames.shape[2:])
        vit_feats = self.vit.forward(frames_flat).view(B, T, -1)  # (B, T, 768)

        if yolo_features is not None:
            features = torch.cat([yolo_features.to(vit_feats.device), vit_feats], dim=-1)
        else:
            features = vit_feats

        return self.bilstm(features, mask=mask)

    @classmethod
    def build(
        cls,
        num_unfreeze_blocks: int = 2,
        device: str = "cuda",
        pretrained: bool = True,
        use_yolo_features: bool = True,
    ) -> "EndToEndThreatModel":
        """Factory: create and load the standard TTSS end-to-end model.

        Args:
            num_unfreeze_blocks: How many ViT tail blocks to fine-tune (0 = all frozen).
            device:              Torch device string.
            pretrained:          Whether to load ImageNet-pretrained ViT weights.
            use_yolo_features:   Set True to expect concatenated YOLO 8-dim features.
        """
        vit = VitSceneEncoder(
            pretrained=pretrained,
            device=device,
            num_unfreeze_blocks=num_unfreeze_blocks,
        )
        vit.load()

        input_dim = 776 if use_yolo_features else 768
        bilstm = BiLSTMThreatPredictor(input_dim=input_dim)

        model = cls(vit, bilstm)
        model = model.to(device)
        return model
