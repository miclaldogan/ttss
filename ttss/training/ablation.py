"""Temporal Threat Scoring System (TTSS): ablation study framework.

Provides ``AblationConfig`` — a single dataclass that fully specifies one
point in the ablation design space — and ``build_ablation_model()`` /
``build_ablation_loss()`` factories that translate configs into runnable
PyTorch objects.

Ablation dimensions
-------------------
1. **Architecture variants** (``ARCH_VARIANTS``): six configs covering
   removal of ViT, attention, bidirectionality and each loss term.
2. **Pre-crime window K sweep** (``WINDOW_K_VALUES``): evaluates EAR and
   AUC at K ∈ {0, 30, 60, 90, 120, 150}.
3. **Feature fusion strategy** (``FUSION_STRATEGIES`` / ``FUSION_INPUT_DIMS``):
   concatenation, additive projection, and attention-weighted fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ttss.models.prediction.bilstm_threat import BiLSTMThreatPredictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Pre-crime window values (frames) used in the K-sweep experiment.
WINDOW_K_VALUES: list[int] = [0, 30, 60, 90, 120, 150]

#: Feature-input dimensionalities for each fusion strategy.
#: concat: YOLO (8) + ViT (768) = 776; additive / attention: projected to 768.
FUSION_INPUT_DIMS: dict[str, int] = {
    "concat": 776,
    "additive": 768,
    "attention": 768,
}

#: Ordered list of fusion strategy keys.
FUSION_STRATEGIES: list[str] = list(FUSION_INPUT_DIMS)


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Complete specification of one ablation experiment point.

    Architecture fields
    -------------------
    variant:
        One of ``"full"``, ``"no_vit"``, ``"no_attention"``,
        ``"unidirectional"``, ``"no_precrime_loss"``,
        ``"no_temporal_consistency"``.
    input_dim:
        Feature dimensionality fed to the ``BiLSTMThreatPredictor``.
        Defaults to 1536 (YOLO-8 + ViT-768 × 2 concatenated).
    hidden_dim:
        BiLSTM hidden size.
    num_layers:
        Number of BiLSTM stacking layers.
    dropout:
        Dropout probability.
    bidirectional:
        ``False`` for the ``"unidirectional"`` variant.
    use_attention:
        ``False`` for the ``"no_attention"`` variant.

    Loss fields
    -----------
    precrime_weight:
        ``PreCrimeDetectionLoss`` weight — set to ``0.0`` to ablate.
    consistency_lambda:
        ``λ2`` in ``composite_threat_loss`` — set to ``0.0`` to ablate
        temporal consistency.

    Evaluation fields
    -----------------
    precrime_window_k:
        Number of pre-crime frames used for EAR computation.

    Fusion field
    ------------
    fusion:
        One of ``"concat"``, ``"additive"``, ``"attention"``.
    """

    variant: str = "full"

    # Model hyperparams
    input_dim: int = 1536
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True
    use_attention: bool = True

    # Loss hyperparams
    precrime_weight: float = 2.0
    consistency_lambda: float = 0.1

    # Evaluation
    precrime_window_k: int = 120

    # Fusion strategy
    fusion: str = "concat"

    # Human-readable description set by factory helpers
    description: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Predefined variant collections
# ---------------------------------------------------------------------------

#: All six architecture ablation configs, keyed by variant name.
ARCH_VARIANTS: dict[str, AblationConfig] = {
    "full": AblationConfig(
        variant="full",
        description="Full TTSS model — all components enabled",
    ),
    "no_vit": AblationConfig(
        variant="no_vit",
        input_dim=8,
        description="No ViT: use YOLOv8 8-dim features only",
    ),
    "no_attention": AblationConfig(
        variant="no_attention",
        use_attention=False,
        description="No attention: replace TemporalAttention with mean pooling",
    ),
    "unidirectional": AblationConfig(
        variant="unidirectional",
        bidirectional=False,
        description="Unidirectional LSTM (bidirectional=False)",
    ),
    "no_precrime_loss": AblationConfig(
        variant="no_precrime_loss",
        precrime_weight=0.0,
        description="No pre-crime loss: precrime_weight=0",
    ),
    "no_temporal_consistency": AblationConfig(
        variant="no_temporal_consistency",
        consistency_lambda=0.0,
        description="No temporal consistency: λ2=0",
    ),
}

#: K-sweep configs — one per K value, all other params at full-model defaults.
WINDOW_SWEEP_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        variant="full",
        precrime_window_k=k,
        description=f"K-sweep K={k}",
    )
    for k in WINDOW_K_VALUES
]

#: Fusion strategy configs.
FUSION_SWEEP_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        variant="full",
        input_dim=FUSION_INPUT_DIMS[strategy],
        fusion=strategy,
        description=f"Fusion={strategy} (input_dim={FUSION_INPUT_DIMS[strategy]})",
    )
    for strategy in FUSION_STRATEGIES
]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_ablation_model(cfg: AblationConfig) -> BiLSTMThreatPredictor:
    """Construct a ``BiLSTMThreatPredictor`` configured by *cfg*.

    Architecture-level ablations (no_vit, no_attention, unidirectional) are
    encoded directly in the model constructor.  Loss-level ablations
    (no_precrime_loss, no_temporal_consistency) do not change the model
    architecture — use ``build_ablation_loss()`` for those.
    """
    return BiLSTMThreatPredictor(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        use_attention=cfg.use_attention,
    )


def build_ablation_loss(cfg: AblationConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a composite loss function configured by *cfg*.

    The returned callable has signature ``(predictions, targets) -> scalar``.
    """
    from ttss.training.losses import composite_threat_loss

    precrime_weight = cfg.precrime_weight
    consistency_lambda = cfg.consistency_lambda

    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return composite_threat_loss(
            predictions,
            targets,
            lambda2=consistency_lambda,
        )

    # Attach config metadata for inspection
    loss_fn.__doc__ = (  # type: ignore[assignment]
        f"AblationLoss(variant={cfg.variant!r}, "
        f"precrime_weight={precrime_weight}, "
        f"consistency_lambda={consistency_lambda})"
    )
    return loss_fn
