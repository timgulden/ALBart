"""ALBart Control Center — web API for DJ, MapView, and Listener.

This module is now a thin entry-point shim.  All logic lives in
``albart.server.app``.

Usage:
    python -m albart.dj_server
    python -m albart.dj_server --port 8765
"""

from albart.server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
