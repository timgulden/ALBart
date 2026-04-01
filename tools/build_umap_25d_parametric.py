"""Train a parametric UMAP model (512D CLAP → 25D) for real-time projection.

Two-step process:
  1. Fit non-parametric UMAP on all 512D embeddings from PostgreSQL
     and store the resulting 25D coordinates back in the database.
  2. Train a PyTorch MLP to distill the UMAP mapping (512D → 25D),
     so new embeddings can be projected at runtime without re-running UMAP.

The MLP is saved to data/umap_25d_model/model.pt.

Usage:
    python3 tools/build_umap_25d_parametric.py
    python3 tools/build_umap_25d_parametric.py --force     # rebuild even if model exists
    python3 tools/build_umap_25d_parametric.py --epochs 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.effects.database import DatabaseClient, DatabaseConfig
from albart.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = DATA_DIR / "umap_25d_model"
MODEL_PATH = MODEL_DIR / "model.pt"

# UMAP parameters — tuned for global genre structure preservation
UMAP_N_COMPONENTS = 25
UMAP_N_NEIGHBORS = 75
UMAP_MIN_DIST = 0.3
UMAP_METRIC = "cosine"

# MLP architecture
HIDDEN_DIM = 256
INPUT_DIM = 512
OUTPUT_DIM = 25


def load_embeddings_from_db() -> tuple[list[str], np.ndarray]:
    """Load all 512D embeddings from PostgreSQL.

    Returns:
        (track_ids, embeddings) where embeddings is (N, 512) float32.
    """
    db = DatabaseClient(DatabaseConfig())
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT track_id, embedding_512
                FROM tracks
                WHERE embedding_512 IS NOT NULL
                ORDER BY track_id
            """)
            rows = cur.fetchall()

    track_ids = [row[0] for row in rows]
    embeddings = np.array([np.array(row[1], dtype=np.float32) for row in rows])
    return track_ids, embeddings


def store_umap_25d(track_ids: list[str], coords_25d: np.ndarray) -> None:
    """Write 25D UMAP coordinates back to PostgreSQL."""
    db = DatabaseClient(DatabaseConfig())
    with db._conn() as conn:
        with conn.cursor() as cur:
            for tid, coord in zip(track_ids, coords_25d):
                cur.execute(
                    "UPDATE tracks SET umap_25d = %s WHERE track_id = %s",
                    (coord, tid),
                )
    logger.info("Stored %d 25D projections in database", len(track_ids))


def fit_umap(embeddings: np.ndarray) -> np.ndarray:
    """Fit non-parametric UMAP and return (N, 25) coordinates."""
    import umap

    logger.info(
        "Fitting UMAP: n_components=%d, n_neighbors=%d, min_dist=%.2f, metric=%s",
        UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_METRIC,
    )
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    logger.info("UMAP result shape: %s", coords.shape)
    return coords


def build_mlp():
    """Build the distillation MLP: 512 → 256 → 256 → 25."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


def train_mlp(
    embeddings_512: np.ndarray,
    coords_25d: np.ndarray,
    epochs: int = 500,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    patience: int = 50,
) -> None:
    """Train MLP to map 512D → 25D, save to MODEL_PATH."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # Train/val split (deterministic)
    rng = np.random.RandomState(42)
    n = len(embeddings_512)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(embeddings_512[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(coords_25d[train_idx], dtype=torch.float32)
    X_val = torch.tensor(embeddings_512[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(coords_25d[val_idx], dtype=torch.float32)

    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = build_mlp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()

        if epoch % 50 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d  train_mse=%.6f  val_mse=%.6f",
                epoch, epochs, train_loss, val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Restore best and save
    model.load_state_dict(best_state)
    model = model.cpu()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
    }, MODEL_PATH)
    logger.info("Saved model → %s", MODEL_PATH)

    # Final quality report
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
    val_true = coords_25d[val_idx]
    l2_errors = np.linalg.norm(val_pred - val_true, axis=1)
    logger.info(
        "Val L2 error: mean=%.4f  median=%.4f  p95=%.4f  max=%.4f",
        l2_errors.mean(), np.median(l2_errors),
        np.percentile(l2_errors, 95), l2_errors.max(),
    )
    # Coordinate range for context
    coord_range = np.linalg.norm(coords_25d.max(axis=0) - coords_25d.min(axis=0))
    logger.info("Coordinate diagonal: %.4f (error/diagonal ratio: %.4f)",
                coord_range, l2_errors.mean() / coord_range)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train parametric UMAP (512D CLAP → 25D)"
    )
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if model exists")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Max training epochs (default: 500)")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip non-parametric UMAP (use existing 25D in DB)")
    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.force:
        logger.info("Model already exists: %s (use --force to rebuild)", MODEL_PATH)
        return

    # Step 1: Load embeddings
    logger.info("Loading 512D embeddings from PostgreSQL...")
    track_ids, embeddings = load_embeddings_from_db()
    logger.info("Loaded %d tracks with embeddings (%s)", len(track_ids), embeddings.shape)

    if len(track_ids) < 100:
        logger.error("Too few tracks (%d) for reliable UMAP. Need at least 100.", len(track_ids))
        sys.exit(1)

    # Step 2: Non-parametric UMAP → 25D coordinates
    if args.skip_umap:
        logger.info("Skipping UMAP — loading existing 25D from database...")
        db = DatabaseClient(DatabaseConfig())
        coords_25d = np.zeros((len(track_ids), 25), dtype=np.float32)
        with db._conn() as conn:
            with conn.cursor() as cur:
                for i, tid in enumerate(track_ids):
                    cur.execute("SELECT umap_25d FROM tracks WHERE track_id = %s", (tid,))
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        coords_25d[i] = np.array(row[0], dtype=np.float32)
                    else:
                        logger.error("No umap_25d for %s — cannot skip UMAP", tid)
                        sys.exit(1)
    else:
        coords_25d = fit_umap(embeddings)
        store_umap_25d(track_ids, coords_25d)

    # Step 3: Train MLP distillation
    logger.info("Training MLP distillation model...")
    train_mlp(embeddings, coords_25d, epochs=args.epochs)

    logger.info("Done.")


if __name__ == "__main__":
    main()
