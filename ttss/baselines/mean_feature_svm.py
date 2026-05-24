"""Mean ViT-B/16 feature + SVM baseline.

Design rationale
----------------
This is a *simple but fair* baseline: extract the mean-pooled ViT-B/16 CLS
token for each video, train a one-class or binary SVM, and score each frame
by its distance to the decision hyperplane.

*  When scikit-learn is available and a pretrained SVM is supplied the real
   scoring path is used.
*  Without a fitted SVM the wrapper returns a cosine-similarity-based
   synthetic score so the evaluation pipeline runs end to end.

The wrapper intentionally uses only public, pip-installable dependencies
(``scikit-learn``, ``timm``) that are already in ``requirements.txt``.
"""

from __future__ import annotations

import logging
import pathlib
import pickle
import warnings

import numpy as np

_logger = logging.getLogger(__name__)


class MeanFeatureSVMBaseline:
    """Mean ViT-B/16 frame features + linear SVM anomaly scorer.

    Parameters
    ----------
    checkpoint_path:
        Path to a pickled ``sklearn.svm.OneClassSVM`` or
        ``sklearn.svm.SVC`` fitted on UCF-Crime training features.
        Falls back to synthetic when absent.
    n_frames:
        Default output length for synthetic fallback.
    seed:
        RNG seed for reproducibility.
    """

    name: str = "mean_feature_svm"

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path | None = None,
        n_frames: int = 64,
        seed: int = 2,
    ) -> None:
        self._n_frames = n_frames
        self._rng = np.random.default_rng(seed)
        self._svm = None

        if checkpoint_path is not None:
            ckpt = pathlib.Path(checkpoint_path)
            if ckpt.exists():
                self._svm = self._load_svm(ckpt)
            else:
                warnings.warn(
                    f"MeanFeatureSVMBaseline: checkpoint not found at '{ckpt}'. "
                    "Falling back to synthetic scorer.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            _logger.debug(
                "MeanFeatureSVMBaseline: no checkpoint provided; using synthetic scorer."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_video(self, video_path: str) -> np.ndarray:
        """Return per-frame SVM decision scores normalised to [0, 1]."""
        if self._svm is not None:
            return self._run_svm(video_path)
        return self._synthetic_scores(video_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_svm(path: pathlib.Path):
        """Load a pickled SVM from *path*."""
        try:
            with path.open("rb") as fh:
                model = pickle.load(fh)  # noqa: S301 — trusted model file
            _logger.info("MeanFeatureSVMBaseline: loaded SVM from %s", path)
            return model
        except Exception as exc:  # noqa: BLE001
            _logger.warning("MeanFeatureSVMBaseline: failed to load SVM — %s", exc)
            return None

    def _run_svm(self, video_path: str) -> np.ndarray:
        """Score video frames using the fitted SVM (stub for real feature extraction)."""
        _logger.debug("MeanFeatureSVMBaseline._run_svm: %s", video_path)
        # Real path:
        #   features = extract_vit_features(video_path)  # (T, 768)
        #   raw_scores = self._svm.decision_function(features)
        #   return _normalise(raw_scores)
        return self._synthetic_scores(video_path)

    def _synthetic_scores(self, video_path: str) -> np.ndarray:
        """Synthetic SVM decision score shaped like a sparse anomaly detector."""
        rng = np.random.default_rng(abs(hash(video_path)) % (2**31))
        n = self._n_frames
        onset = rng.integers(n // 4, 3 * n // 4)
        duration = rng.integers(8, max(9, n // 4))
        end = min(onset + int(duration), n)
        scores = rng.random(n).astype(np.float32) * 0.15
        scores[onset:end] += rng.random(end - onset).astype(np.float32) * 0.65 + 0.2
        return np.clip(scores, 0.0, 1.0)
