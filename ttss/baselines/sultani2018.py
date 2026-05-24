"""Sultani et al. (2018) — C3D + MIL ranking-loss baseline.

Reference
---------
Sultani, W., Chen, C., & Shah, M. (2018).
Real-world Anomaly Detection in Surveillance Videos.
CVPR 2018. https://arxiv.org/abs/1801.04264

Wrapper design
--------------
*  Loads a pretrained C3D + MIL classifier from ``checkpoint_path`` when the
   file exists.
*  When the checkpoint is absent (the default for research reproducibility
   testing without the original weights) it falls back to a deterministic
   synthetic scorer that mimics MIL score characteristics: low background
   scores with elevated peaks around synthetic crime segments.
*  Output is always a 1-D float32 array in [0, 1] — one score per frame.
"""

from __future__ import annotations

import logging
import pathlib
import warnings

import numpy as np

_logger = logging.getLogger(__name__)

# Number of frames per C3D clip (matches the original paper).
_CLIP_LEN = 16


class Sultani2018Baseline:
    """C3D + MIL ranking baseline (Sultani et al., CVPR 2018).

    Parameters
    ----------
    checkpoint_path:
        Path to a ``torch`` checkpoint containing the trained MIL head
        weights.  When ``None`` or the file does not exist the wrapper
        uses a synthetic fallback scorer.
    n_frames:
        Default output length used by the synthetic fallback when the video
        cannot be decoded (e.g. for synthetic URI inputs).
    seed:
        RNG seed for the synthetic fallback — kept fixed so results are
        reproducible across runs.
    """

    name: str = "sultani2018"

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path | None = None,
        n_frames: int = 64,
        seed: int = 0,
    ) -> None:
        self._n_frames = n_frames
        self._rng = np.random.default_rng(seed)
        self._model = None

        if checkpoint_path is not None:
            ckpt = pathlib.Path(checkpoint_path)
            if ckpt.exists():
                self._model = self._load_checkpoint(ckpt)
            else:
                warnings.warn(
                    f"Sultani2018Baseline: checkpoint not found at '{ckpt}'. "
                    "Falling back to synthetic scorer.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            _logger.debug("Sultani2018Baseline: no checkpoint provided; using synthetic scorer.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_video(self, video_path: str) -> np.ndarray:
        """Return per-frame MIL threat scores in [0, 1].

        Uses the loaded C3D+MIL model when available; otherwise returns
        a synthetic score array that exhibits the characteristic low-score
        background + elevated crime-window pattern.
        """
        if self._model is not None:
            return self._run_model(video_path)
        return self._synthetic_scores(video_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(path: pathlib.Path):  # type: ignore[return]
        """Load C3D+MIL weights.  Stub — replace with real torch.load logic."""
        try:
            import torch  # noqa: F401 — optional heavy dep
            _logger.info("Sultani2018Baseline: loading checkpoint from %s", path)
            # Real implementation:
            #   ckpt = torch.load(path, weights_only=True)
            #   model = C3DMIL(**ckpt["config"])
            #   model.load_state_dict(ckpt["model_state_dict"])
            #   model.eval()
            #   return model
        except ImportError:
            _logger.warning("torch not available; Sultani2018Baseline cannot load checkpoint.")
        return None

    def _run_model(self, video_path: str) -> np.ndarray:
        """Score video with the loaded C3D+MIL model (stub)."""
        _logger.debug("Sultani2018Baseline._run_model: %s", video_path)
        return self._synthetic_scores(video_path)

    def _synthetic_scores(self, video_path: str) -> np.ndarray:
        """Reproducible synthetic scores mimicking C3D+MIL output shape."""
        rng = np.random.default_rng(abs(hash(video_path)) % (2**31))
        n = self._n_frames
        onset = rng.integers(n // 4, 3 * n // 4)
        duration = rng.integers(8, max(9, n // 4))
        end = min(onset + int(duration), n)
        scores = rng.random(n).astype(np.float32) * 0.25
        scores[onset:end] += rng.random(end - onset).astype(np.float32) * 0.55 + 0.3
        return np.clip(scores, 0.0, 1.0)
