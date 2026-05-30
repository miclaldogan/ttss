"""Object-Conditioned Cross-Attention fusion for video anomaly detection.

Bidirectional cross-attention between YOLO object features and VideoMAE scene
features.  Crimes are defined by object-scene interactions (person + weapon,
crowd + motion), so explicitly modeling this interaction is the core novelty.

Two attention directions:
  obj → scene: given what objects are present, which scene moments are relevant?
  scene → obj: given scene motion, which objects are anomaly-relevant?

Reference shapes (per batch element):
  yolo  : (B, T,   8)  — object detection summary per clip
  vmae  : (B, T, 768)  — VideoMAE clip embeddings
  output: (B, T, D)    — fused object-scene representation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObjectSceneCrossAttention(nn.Module):
    """Bidirectional cross-attention between object (YOLO) and scene (VideoMAE) features.

    Parameters
    ----------
    yolo_dim:   Dimension of YOLO features (8).
    vmae_dim:   Dimension of VideoMAE features (768).
    hidden_dim: Shared attention dimension. Default 256.
    num_heads:  Number of attention heads. hidden_dim must be divisible by num_heads.
    dropout:    Dropout on attention weights and fusion MLP.
    """

    def __init__(
        self,
        yolo_dim: int = 8,
        vmae_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Project both streams to shared dimension
        self.obj_proj   = nn.Linear(yolo_dim, hidden_dim)
        self.scene_proj = nn.Linear(vmae_dim, hidden_dim)

        # obj → scene: YOLO queries attend over VideoMAE
        self.obj2scene = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # scene → obj: VideoMAE queries attend over YOLO
        self.scene2obj = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Fuse both attended representations
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        yolo: torch.Tensor,
        vmae: torch.Tensor,
    ) -> torch.Tensor:
        """Return fused object-scene representation.

        Args:
            yolo: (B, T, yolo_dim)
            vmae: (B, T, vmae_dim)

        Returns:
            (B, T, hidden_dim)
        """
        obj   = self.obj_proj(yolo)     # (B, T, D)
        scene = self.scene_proj(vmae)   # (B, T, D)

        # YOLO queries over VideoMAE: what scene context matches these objects?
        obj_ctx, _ = self.obj2scene(query=obj, key=scene, value=scene)    # (B, T, D)

        # VideoMAE queries over YOLO: what objects are present during this motion?
        scene_ctx, _ = self.scene2obj(query=scene, key=obj, value=obj)   # (B, T, D)

        # Fuse both directions
        fused = self.fusion(torch.cat([obj_ctx, scene_ctx], dim=-1))     # (B, T, D)

        # Residual: average of projected inputs + fused
        residual = self.dropout((obj + scene) * 0.5)
        return self.norm(fused + residual)                                # (B, T, D)


class ObjectConditionedThreatModel(nn.Module):
    """Two-stream anomaly model: YOLO-gated VideoMAE + temporal motion delta → BiLSTM.

    Architecture
    ------------
    The original cross-attention between 8-dim YOLO and 768-dim VideoMAE was
    fundamentally broken (8-dim queries attending over 768-dim keys = noise).

    New design:
      1. YOLO Gate: sigmoid(Linear(yolo_dim → vmae_dim)) modulates VideoMAE
         features multiplicatively — YOLO tells the model *which* scene features
         matter given detected objects (person/weapon presence boosts relevant dims).
      2. Temporal Motion Delta: vit[t] - vit[t-1] gives explicit motion/change
         signal in VideoMAE feature space.  Anomalies = sudden large deviations.
      3. Input projection: (gated_vmae ‖ motion_delta) → hidden_dim via MLP.
      4. YOLO residual: a separate Linear(yolo_dim → hidden_dim) is added as a
         direct residual so object counts always reach the BiLSTM.
      5. BiLSTM + attention over the hidden_dim sequence.

    Accepts either:
      - a 3-tuple (yolo, vmae, flow) — preferred after feature enrichment
      - a 2-tuple (yolo, vmae)       — backward compat (flow treated as zeros)
      - a single concatenated tensor — backward compat
    """

    def __init__(
        self,
        yolo_dim: int = 32,
        vmae_dim: int = 768,
        flow_dim: int = 16,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        bilstm_layers: int = 2,
        bilstm_hidden: int = 256,
    ) -> None:
        super().__init__()
        self._yolo_dim = yolo_dim
        self._vmae_dim = vmae_dim
        self._flow_dim = flow_dim

        # 1. YOLO gate: multiplicative modulation of VideoMAE features
        self.yolo_gate = nn.Sequential(
            nn.Linear(yolo_dim, vmae_dim),
            nn.Sigmoid(),
        )

        # 2. Input projection: gated_vmae (vmae_dim) + motion_delta (vmae_dim) + flow (flow_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(vmae_dim * 2 + flow_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. YOLO residual: direct path for object-count signal
        self.yolo_residual = nn.Linear(yolo_dim, hidden_dim)

        # 4. BiLSTM
        from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor
        self.bilstm = BiLSTMThreatPredictor(
            input_dim=hidden_dim,
            hidden_dim=bilstm_hidden,
            num_layers=bilstm_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor | tuple) -> object:
        if isinstance(x, (tuple, list)) and len(x) >= 3:
            yolo, vmae, flow = x[0], x[1], x[2]
        elif isinstance(x, (tuple, list)):
            yolo, vmae = x[0], x[1]
            flow = torch.zeros(yolo.shape[0], yolo.shape[1], self._flow_dim,
                               dtype=yolo.dtype, device=yolo.device)
        else:
            yolo = x[..., : self._yolo_dim]
            vmae = x[..., self._yolo_dim : self._yolo_dim + self._vmae_dim]
            rest = x[..., self._yolo_dim + self._vmae_dim :]
            if rest.shape[-1] >= self._flow_dim:
                flow = rest[..., : self._flow_dim]
            else:
                flow = torch.zeros(yolo.shape[0], yolo.shape[1], self._flow_dim,
                                   dtype=yolo.dtype, device=yolo.device)

        # 1. YOLO-gated VideoMAE: object presence modulates scene features
        gate = self.yolo_gate(yolo)          # (B, T, vmae_dim)
        vmae_gated = vmae * gate             # (B, T, vmae_dim)

        # 2. Temporal motion delta: change in VideoMAE embedding over time
        motion = torch.zeros_like(vmae)
        motion[:, 1:] = vmae[:, 1:] - vmae[:, :-1]   # (B, T, vmae_dim); t=0 stays 0

        # 3. Project to hidden_dim (gated_vmae + motion_delta + optical_flow)
        combined = torch.cat([vmae_gated, motion, flow], dim=-1)  # (B, T, vmae_dim*2+flow_dim)
        fused = self.input_proj(combined)                          # (B, T, hidden_dim)

        # 4. Add YOLO residual so object counts have a direct gradient path
        fused = fused + self.yolo_residual(yolo)             # (B, T, hidden_dim)

        return self.bilstm(fused)
