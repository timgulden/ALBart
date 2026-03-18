"""Shared utilities: device selection, config loading."""

import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (ALBart/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def get_device() -> str:
    """Return the best available torch device: 'mps', or 'cpu'."""
    if torch.backends.mps.is_available():
        logger.debug("Using MPS device (Apple Silicon)")
        return "mps"
    logger.debug("Using CPU device")
    return "cpu"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and return the config.yaml as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
