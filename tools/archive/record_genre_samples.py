"""
tools/record_genre_samples.py

Play Spotify tracks via AppleScript and record them simultaneously from
the Movo mic (PDP Audio Device), producing 3-minute WAV samples for
cross-genre recognition benchmarking.

Usage:
    python tools/record_genre_samples.py
    python tools/record_genre_samples.py --duration 180
    python tools/record_genre_samples.py --dry-run       # print track list, don't record
    python tools/record_genre_samples.py --input-volume 75  # fixed volume (0-100)
    python tools/record_genre_samples.py --agc           # automatic gain control

AGC mode (--agc):
    Measures RMS every --agc-interval seconds and adjusts the macOS input
    volume to the nearest discrete level to keep signal near --agc-target RMS.
    Levels only change when the nearest discrete level differs from the current
    one, so stable content stays at a fixed level with no artifacts.

The GENRE_TRACKS list below defines which tracks to record.  Edit it to
change the selection.  Track IDs come from the ALBart database.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Genre track list — edit to change the benchmark set.
# Each entry: (track_id, output_name, genre, display_label)
# output_name → data/samples/{output_name}.wav
# display_label is just for the console; target substring for batch_sweep
# ---------------------------------------------------------------------------
GENRE_TRACKS = [
    # Classic Rock
    ("14uQWXYRflpBP8J6olZ8mH", "gimme_shelter",      "Classic Rock",      "Gimme Shelter"),
    # Jazz
    ("0E8q2Fx2XuzXCO2NSAppkR", "sentimental_mood",   "Jazz",              "In A Sentimental Mood"),
    # Blues
    ("2HUZVffVPXvqnrml0gXggp", "smokestack",         "Blues",             "Smokestack Lightnin'"),
    # Classical Piano
    ("2d6ml9Qkx8r4EjuUyrdpRV", "chopin_nocturne",    "Classical Piano",   "Nocturne No. 1"),
    # Classical Choral
    ("6WdplZr3V1U3ladDMhWBbo", "bach_kreuzstab",     "Classical Choral",  "Kreuzstab"),
    # K-pop
    ("5QDLhrAOJJdNAmCTJ8xMyW", "bts_dynamite",       "K-pop",             "Dynamite"),
    # Lo-fi
    ("11mUt9hPLDxt7EsoYB2Ujc", "mad_blunted_jazz",   "Lo-fi / Hip-hop",   "Mad Blunted Jazz"),
    # Post-punk
    ("2rEejosMFFiQkveFvb9P0Q", "red_right_hand",     "Post-punk / Gothic", "Red Right Hand"),
    # Electronic
    ("6gbmylJ7sB7NFfMfTQHosf", "alberto_balsalm",    "Electronic / IDM",  "Alberto Balsalm"),
]

# Discrete volume levels for AGC — spaced ~3dB apart (×1.12 each step).
# Fewer levels = fewer jumps; more levels = finer control.
AGC_LEVELS = [20, 25, 32, 40, 50, 63, 79, 100]


def get_input_volume() -> int:
    """Return current macOS input volume (0-100)."""
    result = subprocess.run(
        ["osascript", "-e", "input volume of (get volume settings)"],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def set_input_volume(level: int) -> None:
    """Set macOS input volume (0-100)."""
    subprocess.run(["osascript", "-e", f"set volume input volume {level}"], check=True)


def nearest_level(levels: list[int], ideal: float) -> int:
    """Return the level from `levels` closest to `ideal`."""
    return min(levels, key=lambda x: abs(x - ideal))


def find_input_device(name_fragment: str) -> int | None:
    """Return device index of first input device whose name contains name_fragment."""
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_fragment.lower() in d["name"].lower():
            return i
    return None


def spotify_play(track_id: str) -> None:
    """Tell Spotify (via AppleScript) to play a specific track URI."""
    uri = f"spotify:track:{track_id}"
    script = f'tell application "Spotify" to play track "{uri}"'
    subprocess.run(["osascript", "-e", script], check=True)


def spotify_pause() -> None:
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to pause'], check=True)


def spotify_current() -> tuple[str, str]:
    """Return (title, artist) of currently playing Spotify track."""
    script = 'tell application "Spotify" to get {name of current track, artist of current track}'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    parts = result.stdout.strip().split(", ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")


def record(
    device: int,
    duration: int,
    path: Path,
    sample_rate: int = 48000,
    agc: bool = False,
    agc_interval: int = 30,
    agc_target: float = 0.25,
    agc_levels: list[int] = AGC_LEVELS,
) -> None:
    """
    Record `duration` seconds of audio from `device` to a WAV file.

    With agc=True, measures RMS every agc_interval seconds and snaps the
    macOS input volume to the nearest discrete level in agc_levels.  The
    recording segments are concatenated into a single WAV file.
    """
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    dev_name = sd.query_devices(device)["name"]
    print(f"  Recording {duration}s from device [{device}]: {dev_name}")

    if not agc:
        # Simple fixed-level recording
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype="float32", device=device)
        for elapsed in range(duration):
            time.sleep(1)
            if elapsed % 10 == 0 or elapsed == duration - 1:
                bar = "█" * (elapsed + 1) + "░" * (duration - elapsed - 1)
                print(f"  [{bar[:30]}] {elapsed+1}/{duration}s", end="\r")
        print()
        sd.wait()
        segments = [audio.flatten()]
    else:
        # AGC: record in agc_interval-second segments, adjust volume between each
        current_vol = get_input_volume()
        segments = []
        elapsed_total = 0
        seg_num = 0

        print(f"  AGC target RMS={agc_target}  levels={agc_levels}  interval={agc_interval}s")
        print(f"  {'Seg':>4}  {'Time':>6}  {'RMS':>7}  {'Vol':>5}  Action")
        print(f"  {'-'*50}")

        while elapsed_total < duration:
            seg_dur = min(agc_interval, duration - elapsed_total)
            audio = sd.rec(int(seg_dur * sample_rate), samplerate=sample_rate,
                           channels=1, dtype="float32", device=device)
            sd.wait()
            chunk = audio.flatten()
            segments.append(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2))) + 1e-8
            elapsed_total += seg_dur
            seg_num += 1

            # Compute ideal volume and snap to nearest discrete level
            ideal = current_vol * (agc_target / rms)
            ideal = max(agc_levels[0], min(agc_levels[-1], ideal))
            new_vol = nearest_level(agc_levels, ideal)

            action = ""
            if new_vol != current_vol:
                try:
                    set_input_volume(new_vol)
                    action = f"vol {current_vol}→{new_vol}"
                    current_vol = new_vol
                    time.sleep(0.3)  # brief pause for volume to settle before next segment
                except Exception as e:
                    action = f"vol change failed: {e}"
            else:
                action = "—"

            print(f"  {seg_num:>4}  {elapsed_total:>5}s  {rms:>7.4f}  {current_vol:>5}  {action}")

    # Concatenate and save
    full = np.concatenate(segments)
    rms = float(np.sqrt(np.mean(full ** 2)))
    peak = float(np.abs(full).max())
    level_bar = "█" * int(min(rms / 0.30, 1.0) * 20)
    print(f"  Level: [{level_bar:<20}] RMS={rms:.4f}  peak={peak:.4f}")
    if rms < 0.05:
        print("  ⚠️  VERY LOW SIGNAL — recognition will likely fail.")
    elif rms < 0.15:
        print("  ⚠️  Low signal (target RMS ≥ 0.20).")

    sf.write(str(path), full, sample_rate)
    print(f"  Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record genre samples from Spotify via loopback")
    parser.add_argument("--device", type=str, default="PDP Audio",
                        help="Input device name fragment (default: 'PDP Audio')")
    parser.add_argument("--duration", type=int, default=180,
                        help="Recording duration in seconds (default: 180 = 3 min)")
    parser.add_argument("--delay", type=int, default=10,
                        help="Seconds to wait after starting playback before recording "
                             "(default: 10 — absorbs AirPlay latency + room settling)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print track list without recording anything")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Record only tracks whose output_name matches any of these substrings")
    parser.add_argument("--input-volume", type=int, default=None, metavar="0-100",
                        help="Set macOS input volume before recording and restore afterward")
    parser.add_argument("--agc", action="store_true",
                        help="Automatic gain control: adjust input volume every --agc-interval seconds")
    parser.add_argument("--agc-interval", type=int, default=30, metavar="SECONDS",
                        help="AGC measurement interval in seconds (default: 30)")
    parser.add_argument("--agc-target", type=float, default=0.25, metavar="RMS",
                        help="AGC target RMS level (default: 0.25)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"{'Genre':<22} {'Label':<30} {'Output file'}")
        print("-" * 80)
        for tid, name, genre, label in GENRE_TRACKS:
            path = out_dir / f"{name}.wav"
            exists = "✓ exists" if path.exists() else ""
            print(f"{genre:<22} {label:<30} {name}.wav  {exists}")
        return

    # Find device
    try:
        device = int(args.device)
    except ValueError:
        device = find_input_device(args.device)
    if device is None:
        print(f"ERROR: No input device matching '{args.device}'. Available devices:")
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']}")
        sys.exit(1)

    tracks = GENRE_TRACKS
    if args.only:
        tracks = [t for t in tracks if any(pat.lower() in t[1].lower() for pat in args.only)]
        if not tracks:
            print(f"No tracks match --only '{args.only}'")
            sys.exit(1)

    original_volume = get_input_volume()
    if args.input_volume is not None:
        set_input_volume(args.input_volume)
        print(f"Input volume: {original_volume} → {args.input_volume}/100")
    elif not args.agc:
        print(f"Input volume: {original_volume}/100  (use --input-volume N or --agc)")
    else:
        print(f"Input volume: {original_volume}/100 (AGC active, target RMS={args.agc_target})")

    print(f"Recording {len(tracks)} tracks  ({args.duration}s each)")
    print(f"Device: [{device}]  Output: {out_dir}\n")

    for i, (tid, name, genre, label) in enumerate(tracks, 1):
        path = out_dir / f"{name}.wav"
        print(f"[{i}/{len(tracks)}] {genre}: {label}")

        print(f"  Starting Spotify playback...")
        spotify_play(tid)
        time.sleep(args.delay)

        actual_title, actual_artist = spotify_current()
        print(f"  Playing: {actual_title} — {actual_artist}")

        record(
            device, args.duration, path,
            agc=args.agc,
            agc_interval=args.agc_interval,
            agc_target=args.agc_target,
        )

        spotify_pause()
        print()

    if args.input_volume is not None:
        set_input_volume(original_volume)
        print(f"Input volume restored to {original_volume}/100")
    elif args.agc:
        # Restore to whatever volume AGC left it at the start level
        set_input_volume(original_volume)
        print(f"Input volume restored to {original_volume}/100")

    print("All recordings complete.")
    print(f"\nNow run:  python tools/batch_sweep.py")


if __name__ == "__main__":
    main()
