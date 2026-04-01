"""Parametric UMAP projector: 512D CLAP → 25D via trained MLP.

The model is a lightweight MLP trained to reproduce non-parametric UMAP
output (see tools/build_umap_25d_parametric.py).  Inference is thread-safe
— the model is used only with torch.no_grad().

Usage:
    projector = UmapProjector.load(Path("data/umap_25d_model/model.pt"))
    coords_25d = projector.project(embedding_512)  # (25,) float32
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Reconstruct the MLP architecture from saved hyperparams."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class UmapProjector:
    """Projects 512D CLAP embeddings to 25D UMAP space via trained MLP.

    Thread-safe: inference uses torch.no_grad() on a frozen model.
    """
    model_path: Path
    _model: nn.Module = field(repr=False)
    _device: str = field(repr=False)

    @staticmethod
    def load(model_path: Path, device: str | None = None) -> UmapProjector:
        """Load trained model from disk.

        Args:
            model_path: Path to the .pt checkpoint.
            device: torch device ("cpu", "mps", "cuda").
                    Defaults to CPU for inference (fast enough for single vectors).
        """
        if device is None:
            device = "cpu"

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = _build_mlp(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        logger.info(
            "Loaded UmapProjector from %s (%dD → %dD, device=%s)",
            model_path, checkpoint["input_dim"], checkpoint["output_dim"], device,
        )
        return UmapProjector(model_path=model_path, _model=model, _device=device)

    def project(self, embedding_512: np.ndarray) -> np.ndarray:
        """Project a single 512D embedding to 25D.

        Args:
            embedding_512: (512,) float32 array.

        Returns:
            (25,) float32 array.
        """
        x = torch.tensor(embedding_512, dtype=torch.float32, device=self._device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = self._model(x)
        return y.squeeze(0).cpu().numpy()

    def project_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Project a batch of 512D embeddings to 25D.

        Args:
            embeddings: (N, 512) float32 array.

        Returns:
            (N, 25) float32 array.
        """
        x = torch.tensor(embeddings, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            y = self._model(x)
        return y.cpu().numpy()
