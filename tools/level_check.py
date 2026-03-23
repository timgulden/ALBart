"""
tools/level_check.py

Record N x 10-second samples from the input device and report RMS and peak
for each one — raw and optionally after compress_and_normalize().

Usage:
    # BTS Dynamite (default)
    python tools/level_check.py

    # Chopin nocturne
    python tools/level_check.py --track 2d6ml9Qkx8r4EjuUyrdpRV --label Chopin

    # Show levels after compress_and_normalize() — what the embedder sees
    python tools/level_check.py --normalize
    python tools/level_check.py --track 2d6ml9Qkx8r4EjuUyrdpRV --label Chopin --normalize

    # Record one chunk, normalize it, play it back so you can hear it
    python tools/level_check.py --track 2d6ml9Qkx8r4EjuUyrdpRV --label Chopin --audition
    python tools/level_check.py --audition   # BTS
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.utils import compress_and_normalize

SAMPLE_RATE = 48000
CHUNK_SECONDS = 10
TARGET_RMS_RAW = 0.20
TARGET_RMS_NORM = 0.08


def get_input_volume() -> int:
    result = subprocess.run(
        ["osascript", "-e", "input volume of (get volume settings)"],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def set_input_volume(level: int) -> None:
    subprocess.run(["osascript", "-e", f"set volume input volume {level}"], check=True)


def find_input_device(name_fragment: str) -> int | None:
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_fragment.lower() in d["name"].lower():
            return i
    return None


def spotify_play(track_id: str) -> None:
    uri = f"spotify:track:{track_id}"
    script = f'tell application "Spotify" to play track "{uri}"'
    subprocess.run(["osascript", "-e", script], check=True)


def spotify_pause() -> None:
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to pause'], check=True)


def record_chunk(device: int, duration: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    import sounddevice as sd
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    return audio.flatten()


def meter(rms: float, peak: float, full_rms: float = 0.40, width: int = 28) -> str:
    filled = int(min(rms / full_rms, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    clip = " CLIP" if peak >= 0.98 else ("  hot" if peak >= 0.80 else "")
    return f"[{bar}]{clip}"


def play_wav(path: str, label: str) -> None:
    print(f"\nPlaying {label} via afplay...")
    subprocess.run(["afplay", path])


def save_wav(audio: np.ndarray, sr: int, path: str) -> None:
    import soundfile as sf
    sf.write(path, audio, sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Input level checker")
    parser.add_argument("--device", type=str, default="PDP Audio",
                        help="Input device name fragment (default: 'PDP Audio')")
    parser.add_argument("--track", type=str, default="5QDLhrAOJJdNAmCTJ8xMyW",
                        help="Spotify track ID (default: BTS Dynamite)")
    parser.add_argument("--label", type=str, default="BTS Dynamite",
                        help="Label for display (default: 'BTS Dynamite')")
    parser.add_argument("--delay", type=int, default=10,
                        help="Seconds to wait after starting playback (default: 10)")
    parser.add_argument("--chunks", type=int, default=20,
                        help="Number of 10s chunks to record (default: 20)")
    parser.add_argument("--normalize", action="store_true",
                        help="Show levels after compress_and_normalize() — what the embedder sees")
    parser.add_argument("--input-volume", type=int, default=None, metavar="0-100",
                        help="Set macOS input volume before recording and restore afterward")
    parser.add_argument("--audition", action="store_true",
                        help="Record one 10s chunk, play back raw then normalized — hear what the embedder gets")
    args = parser.parse_args()

    if args.audition:
        args.normalize = True
        args.chunks = 1

    try:
        device = int(args.device)
    except ValueError:
        device = find_input_device(args.device)
    if device is None:
        import sounddevice as sd
        print(f"ERROR: No input device matching '{args.device}'. Available:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']}")
        sys.exit(1)

    import sounddevice as sd
    original_volume = get_input_volume()
    if args.input_volume is not None:
        set_input_volume(args.input_volume)
        print(f"Input volume: {original_volume} → {args.input_volume}/100")
    else:
        print(f"Input volume: {original_volume}/100  (use --input-volume to set)")
    print(f"Track:  {args.label}")
    print(f"Device: [{device}] {sd.query_devices(device)['name']}")
    print(f"Starting Spotify playback...")
    spotify_play(args.track)
    print(f"Waiting {args.delay}s for AirPlay to settle...")
    time.sleep(args.delay)

    if args.normalize:
        print(f"\n{'#':>3}  {'RMS(raw)':>9}  {'Peak(raw)':>9}  {'RMS(norm)':>9}  {'Peak(norm)':>9}  Level (raw)")
    else:
        print(f"\n{'#':>3}  {'RMS':>9}  {'Peak':>9}  Level")
    print("-" * 75)

    raw_rms_values = []
    norm_rms_values = []
    last_raw: np.ndarray | None = None
    last_norm: np.ndarray | None = None

    for i in range(1, args.chunks + 1):
        audio = record_chunk(device, CHUNK_SECONDS)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        raw_rms_values.append(rms)
        bar = meter(rms, peak)
        last_raw = audio

        if args.normalize:
            normed = compress_and_normalize(audio, sr=SAMPLE_RATE)
            nrms = float(np.sqrt(np.mean(normed ** 2)))
            npeak = float(np.abs(normed).max())
            norm_rms_values.append(nrms)
            last_norm = normed
            ok = "✓" if nrms >= TARGET_RMS_NORM else "↑ low"
            print(f"{i:>3}  {rms:>9.4f}  {peak:>9.4f}  {nrms:>9.4f}  {npeak:>9.4f}  {bar}  {ok}")
        else:
            ok = "✓" if rms >= TARGET_RMS_RAW and peak < 0.98 else ("⚠ CLIP" if peak >= 0.98 else "↑ low")
            print(f"{i:>3}  {rms:>9.4f}  {peak:>9.4f}  {bar}  {ok}")

    spotify_pause()

    # Summary
    print(f"\n--- Summary: {args.label} ---")
    avg_raw = sum(raw_rms_values) / len(raw_rms_values)
    print(f"  Raw RMS avg:  {avg_raw:.4f}", end="")
    if avg_raw < TARGET_RMS_RAW:
        print(f"  (low — target ≥ {TARGET_RMS_RAW})")
    elif avg_raw > 0.60:
        print(f"  (very hot — likely heavy clipping)")
    else:
        print(f"  ✓")

    if norm_rms_values:
        avg_norm = sum(norm_rms_values) / len(norm_rms_values)
        print(f"  Norm RMS avg: {avg_norm:.4f}", end="")
        if avg_norm < TARGET_RMS_NORM:
            print(f"  (low — signal may be too quiet at source)")
        else:
            print(f"  ✓  (embedder will see this level)")

    if args.input_volume is not None:
        set_input_volume(original_volume)
        print(f"Input volume restored to {original_volume}/100")

    # Audition: play raw then normalized back-to-back
    if args.audition and last_raw is not None and last_norm is not None:
        with tempfile.NamedTemporaryFile(suffix="_raw.wav", delete=False) as f:
            raw_path = f.name
        with tempfile.NamedTemporaryFile(suffix="_norm.wav", delete=False) as f:
            norm_path = f.name

        save_wav(last_raw, SAMPLE_RATE, raw_path)
        # Normalize normalized audio to [-1,1] for playback (it's already ~0.1 RMS, just clip guard)
        playback = np.clip(last_norm, -1.0, 1.0)
        save_wav(playback, SAMPLE_RATE, norm_path)

        print(f"\n--- Audition ---")
        print(f"  Raw chunk saved to:        {raw_path}")
        print(f"  Normalized chunk saved to: {norm_path}")
        play_wav(raw_path, f"{args.label} — RAW")
        play_wav(norm_path, f"{args.label} — NORMALIZED (what embedder gets)")


if __name__ == "__main__":
    main()
