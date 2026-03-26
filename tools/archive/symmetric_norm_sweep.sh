#!/bin/bash
# Test symmetric normalization at multiple targets.
# For each target: rebuild index + sweep 3 key files.
# ~25 min per target.

cd "/Users/tgulden/Documents/AI Projects/ALBart"

SAMPLES="data/samples/bts_dynamite.wav data/samples/red_right_hand.wav data/samples/rrh_movo.wav data/samples/gimme_shelter.wav data/samples/alberto_balsalm.wav data/samples/mad_blunted_jazz.wav"

for TARGET in 0.12 0.15 0.2 0.25; do
    echo ""
    echo "================================================================"
    echo "  SYMMETRIC NORMALIZATION TARGET = $TARGET"
    echo "================================================================"
    echo ""

    echo "--- Rebuilding index with CLAP_NORM_TARGET=$TARGET ---"
    CLAP_NORM_TARGET=$TARGET python3 -m albart.pipeline.run_pipeline --skip-spotify --force 2>&1 | tail -5

    echo ""
    echo "--- Sweep with CLAP_NORM_TARGET=$TARGET ---"
    CLAP_NORM_TARGET=$TARGET python3 -u tools/batch_sweep.py --samples $SAMPLES 2>&1

    echo ""
    echo ">>> TARGET=$TARGET COMPLETE <<<"
    echo ""
done

echo ""
echo "ALL TARGETS COMPLETE"
