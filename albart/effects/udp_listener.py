"""UDP listener for live audio embeddings from the Listener process.

Runs a background thread that receives pickled embedding dicts.
The engine reads the latest value via ``get_latest()``.
"""

from __future__ import annotations

import logging
import pickle
import socket
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UDPListener:
    """Receives live audio embeddings via UDP.

    NOT frozen — the latest embedding is written by the listener thread
    and read by the engine thread, protected by a Lock.
    """
    port: int = 57002
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latest_emb: Optional[np.ndarray] = field(default=None, repr=False)
    _latest_top1: Optional[str] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def start(self) -> None:
        """Start the background listener thread."""
        if self._thread is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass
        sock.bind(("127.0.0.1", self.port))
        sock.settimeout(1.0)

        def _listen():
            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                    payload = pickle.loads(data)
                    with self._lock:
                        self._latest_emb = payload.get("raw")
                        self._latest_top1 = payload.get("top1")
                except socket.timeout:
                    continue
                except Exception:
                    continue

        self._thread = threading.Thread(target=_listen, daemon=True, name="DJ-UDP")
        self._thread.start()
        logger.info("UDP listener started on port %d", self.port)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Thread-safe read of the latest embedding + top-1 track ID."""
        with self._lock:
            return self._latest_emb, self._latest_top1
