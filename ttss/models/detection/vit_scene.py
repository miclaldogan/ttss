"""Temporal Threat Scoring System (TTSS): ViT scene encoder.

This module exposes a ViT-B/16 scene feature extractor using either timm or the
Hugging Face transformers backend. The CLS token is used as the frame-level
scene representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None

try:
    from transformers import ViTModel
except ImportError:  # pragma: no cover - optional dependency
    ViTModel = None


@dataclass(slots=True)
class SceneEmbedding:
    """Scene encoder output for a single frame."""

    vector: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    label: str | None = None
    frame_id: int = -1


class VitSceneEncoder(nn.Module):
    """Vision Transformer scene encoder used by the detection layer."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        device: str = "cpu",
        embedding_dim: int = 768,
        pretrained: bool = True,
        image_size: int = 224,
        backend: str = "timm",
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.image_size = image_size
        self.backend = backend
        self.model = model

    def load(self) -> None:
        """Instantiate the configured ViT backend on demand."""
        if self.model is not None:
            self.model.to(self.device)
            return

        if self.backend == "timm":
            if timm is None:
                raise RuntimeError("timm is required for the timm ViT backend")
            self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
        elif self.backend == "transformers":
            if ViTModel is None:
                raise RuntimeError(
                    "transformers is required for the Hugging Face ViT backend"
                )
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.model.to(self.device)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames and return CLS embeddings of shape (B, 768)."""
        if self.model is None:
            self.load()

        pixel_values = self.preprocess_frames(frames).to(self.device)
        if self.backend == "transformers":
            outputs = self.model(pixel_values=pixel_values)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            features = self.model.forward_features(pixel_values)
            if isinstance(features, tuple):
                features = features[0]
            if features.ndim == 3:
                cls_embedding = features[:, 0, :]
            else:
                cls_embedding = features
        return cls_embedding

    def preprocess_frames(self, frames: Sequence[Any] | torch.Tensor) -> torch.Tensor:
        """Convert raw frames to normalized tensors suitable for ViT input."""
        if isinstance(frames, torch.Tensor):
            batch = frames.clone().detach().float()
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
        else:
            tensors = [self._frame_to_tensor(frame) for frame in frames]
            batch = torch.stack(tensors, dim=0) if tensors else torch.empty(0)

        if batch.numel() == 0:
            return torch.empty((0, 3, self.image_size, self.image_size), dtype=torch.float32)

        if batch.shape[-1] == 3 and batch.ndim == 4:
            batch = batch.permute(0, 3, 1, 2)
        batch = batch.to(dtype=torch.float32)
        if batch.max() > 1.0:
            batch = batch / 255.0
        batch = F.interpolate(
            batch,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=batch.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=batch.dtype).view(1, 3, 1, 1)
        return (batch - mean) / std

    def encode_frame(self, frame: Any, frame_id: int = 0) -> SceneEmbedding:
        """Encode a single frame into a scene embedding."""
        embedding = self.forward(self.preprocess_frames([frame])).squeeze(0).detach().cpu()
        return SceneEmbedding(vector=embedding, label=None, frame_id=frame_id)

    def encode_batch(self, frames: Sequence[Any]) -> torch.Tensor:
        """Encode a batch of frames and return a tensor of shape (B, 768)."""
        if not frames:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        pixel_values = self.preprocess_frames(frames)
        return self.forward(pixel_values)

    def _frame_to_tensor(self, frame: Any) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().clone()
        else:
            array = np.asarray(frame)
            tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        if tensor.shape[-1] == 1:
            tensor = tensor.repeat(1, 1, 3)
        return tensor
