"""Pure mood-filtering logic.

Descriptor parsing and state updates live here.  The actual embedding
computation (CLAP text inference) and mask building (cosine similarity
over all tracks) happen in the effects layer.
"""

from __future__ import annotations

from albart.core.commands import ComputeMoodMaskCommand, LogicResult
from albart.core.state import DJState, MoodState


def parse_mood_descriptors(
    lines: list[str],
) -> tuple[list[str], list[str]]:
    """Split raw descriptor lines into positive and negative lists.

    Lines starting with ``NOT:`` (case-insensitive) are negative.

    Returns:
        ``(positive, negative)`` — each a list of stripped strings.
    """
    positive: list[str] = []
    negative: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("NOT:"):
            negative.append(stripped[4:].strip())
        else:
            positive.append(stripped)
    return positive, negative


def should_recompute_mask(old: MoodState, new: MoodState) -> bool:
    """True if the mood changed in a way that requires a new mask."""
    return old.descriptors != new.descriptors or old.threshold != new.threshold


def update_mood(
    state: DJState,
    *,
    descriptors: list[str] | None = None,
    mood_text: str | None = None,
    threshold: float | None = None,
) -> LogicResult:
    """Return new state with updated mood + a recompute command if needed.

    Any parameter left as ``None`` keeps its current value.
    """
    old_mood = state.mood
    new_mood = old_mood.model_copy(update={
        k: v for k, v in {
            "descriptors": tuple(descriptors) if descriptors is not None else None,
            "mood_text": mood_text,
            "threshold": threshold,
        }.items() if v is not None
    })
    new_state = state.model_copy(update={"mood": new_mood})

    commands = []
    if should_recompute_mask(old_mood, new_mood):
        commands.append(ComputeMoodMaskCommand(
            descriptors=new_mood.descriptors,
            threshold=new_mood.threshold,
        ))

    return LogicResult(state=new_state, commands=commands)


def clear_mood(state: DJState) -> LogicResult:
    """Remove all mood filtering."""
    new_state = state.model_copy(update={"mood": MoodState()})
    # Empty descriptors → engine clears the cached mask
    return LogicResult(
        state=new_state,
        commands=[ComputeMoodMaskCommand(descriptors=(), threshold=0.35)],
    )
