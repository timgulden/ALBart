"""
Record audio to a WAV file for offline testing.

Usage:
    python tools/record_sample.py --out data/samples/roads_movo.wav --seconds 180 --device "UNA"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 48000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = None
    if args.device is not None:
        try:
            device = int(args.device)
        except ValueError:
            devs = sd.query_devices()
            matches = [i for i, d in enumerate(devs)
                       if args.device.lower() in d["name"].lower()
                       and d["max_input_channels"] > 0]
            if not matches:
                print(f"No input device matching '{args.device}'")
                sys.exit(1)
            device = matches[0]
            print(f"Using device [{device}]: {sd.query_devices(device)['name']}")

    for i in range(args.countdown, 0, -1):
        print(f"  Starting in {i}...", flush=True)
        import time; time.sleep(1)

    print(f"*** RECORDING {args.seconds}s → {args.out} ***", flush=True)
    audio = sd.rec(
        int(args.seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio = audio.flatten()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Done. {len(audio):,} samples  RMS={rms:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, SAMPLE_RATE)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
