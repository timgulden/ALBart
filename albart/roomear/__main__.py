"""CLI entry point: python3 -m albart.roomear"""

from __future__ import annotations

import argparse
import logging
import sys

from albart.utils import load_config


def resolve_device(name: str) -> int:
    """Resolve device name/index to a sounddevice integer index."""
    import sounddevice as sd

    try:
        return int(name)
    except ValueError:
        devs = sd.query_devices()
        matches = [
            i for i, d in enumerate(devs)
            if name.lower() in d["name"].lower() and d["max_input_channels"] > 0
        ]
        if not matches:
            print(f"No input device matching '{name}'. Available inputs:")
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    print(f"  [{i}] {d['name']}")
            sys.exit(1)
        print(f"Using device [{matches[0]}]: {sd.query_devices(matches[0])['name']}")
        return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoomEar — ambient audio embedding service"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Input audio device name or index (default: system default)",
    )
    parser.add_argument(
        "--port", type=int, default=57002,
        help="UDP broadcast port (default: 57002)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config()
    rt = config.get("runtime", {})

    # Resolve audio device: CLI arg > config > system default
    device_name = args.device or rt.get("audio_device")
    device = resolve_device(device_name) if device_name is not None else None

    from albart.roomear.service import RoomEarService

    service = RoomEarService(config=config, device=device, port=args.port)
    service.run()


if __name__ == "__main__":
    main()
