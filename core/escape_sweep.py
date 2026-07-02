"""Mechanical escape sweep over the persisted frontier window.

The agentic diagnosis gives a vision model ~6 experiment tool-calls to find a
verified escape; each experiment is milliseconds of emulator compute. A stuck
frontier is usually beaten by a *standard* Sonic move with the right timing —
exactly the space a mechanical battery scans faster and more reliably than a
model guessing timings one call at a time (live-observed: an entire budget spent
hand-tuning one run-up length). The sweep tries a battery of canonical escapes
from every frontier-pinned savestate. Verified escapes flow into the SAME
``verified_experiments`` shape the diagnosis session produces, so the existing
guard compiler consumes them unchanged; the vision model is reserved for spots
where the battery fails.
"""

from core.diagnosis import DiagnosisSession

# Canonical single holds: (actions, hold_frames).
SINGLE_HOLDS = (
    ("RIGHT,B", 30),      # short hop forward
    ("RIGHT,B", 60),      # full jump forward
    ("RIGHT,UP,B", 45),   # high jump
    ("RIGHT,DOWN", 60),   # roll through
    ("RIGHT", 120),       # push through / momentum
)

# Canonical sequences: run-up length is the usual unknown at a pit or wall, and
# Sonic's jump fires on the B *press*, so "run then jump at the edge" needs
# segments. Each ends with a travel segment so a cleared obstacle is actually
# crossed (the verifier needs max_x past the frontier).
SEQUENCES = (
    ({"actions": "RIGHT", "frames": 30}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    ({"actions": "RIGHT", "frames": 60}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    ({"actions": "RIGHT", "frames": 120}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    # Long forward runways: from a standstill Sonic nears top speed in ~3s, and
    # a FORWARD B-less run-up compiles into the position-gated (replay-faithful)
    # guard, unlike back-up sequences which need band-anchored time replay.
    ({"actions": "RIGHT", "frames": 180}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    ({"actions": "RIGHT", "frames": 240}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    ({"actions": "RIGHT", "frames": 60}, {"actions": "RIGHT,UP,B", "frames": 50}, {"actions": "RIGHT", "frames": 100}),
    ({"actions": "RIGHT", "frames": 180}, {"actions": "RIGHT,UP,B", "frames": 50}, {"actions": "RIGHT", "frames": 100}),
    ({"actions": "RIGHT,DOWN", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    # Back up for a longer runway, then jump at speed. Wide gaps need launch
    # SPEED: the runway length is the dominant unknown, so scan it too.
    ({"actions": "LEFT", "frames": 25}, {"actions": "RIGHT", "frames": 55}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    ({"actions": "LEFT", "frames": 60}, {"actions": "RIGHT", "frames": 120}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
    ({"actions": "LEFT", "frames": 120}, {"actions": "RIGHT", "frames": 180}, {"actions": "RIGHT,B", "frames": 45}, {"actions": "RIGHT", "frames": 120}),
)


def sweep_offsets(window, max_offsets=4):
    """Rewind offsets to sweep from, most runway first.

    Frontier-pinned savestates are the moments just before max-x stopped
    improving — the death/stall approach. Earliest pin first: it has the most
    runway to rebuild speed before the obstacle. Without pins (legacy windows),
    fall back to the largest-x snapshots.
    """
    try:
        failure_frame = int(window.get("failure", {}).get("frame", 0))
    except (TypeError, ValueError):
        return []
    snapshots = window.get("snapshots", [])
    pinned = [s for s in snapshots if s.get("frontier")]
    chosen = pinned or sorted(snapshots, key=lambda s: s.get("x_pos", 0), reverse=True)[:max_offsets]
    offsets = []
    for snapshot in sorted(chosen, key=lambda s: s.get("frame", 0))[:max_offsets]:
        try:
            offset = failure_frame - int(snapshot["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if offset >= 0:
            offsets.append(offset)
    return offsets


def sweep_frontier_escapes(
    window,
    session_factory=None,
    max_offsets=4,
    stop_after=2,
    verbose=True,
):
    """Run the escape battery; return (verified_experiments, summary_text).

    Never raises: a sweep failure just means the vision diagnosis runs as
    before. Experiments verify against the window's recorded frontier_x inside
    the session, so every returned experiment is a measured escape.
    """
    build = session_factory or DiagnosisSession
    session = None
    attempted = 0
    try:
        try:
            session = build(window, capture_screenshots=False)
        except TypeError:
            session = build(window)  # factory without the screenshot flag
        offsets = sweep_offsets(window, max_offsets=max_offsets)
        if not offsets:
            return [], "Escape sweep skipped: no usable savestates in the window."
        for offset in offsets:
            for actions, hold in SINGLE_HOLDS:
                session.try_actions(offset, actions, hold)
                attempted += 1
                if len(session.verified_experiments) >= stop_after:
                    break
            if len(session.verified_experiments) < stop_after:
                for segments in SEQUENCES:
                    session.try_action_sequence(offset, list(segments))
                    attempted += 1
                    if len(session.verified_experiments) >= stop_after:
                        break
            if len(session.verified_experiments) >= stop_after:
                break
        experiments = sorted(
            session.verified_experiments, key=lambda e: e.get("max_x", 0), reverse=True
        )
        if experiments:
            best = experiments[0]
            summary = (
                f"Mechanical escape sweep: {attempted} canonical inputs replayed at the frontier; "
                f"{len(experiments)} VERIFIED escape(s) found. Best: '{best.get('actions')}' "
                f"from x={best.get('start_x')} reached x={best.get('max_x')} "
                f"(beats frontier). Compiled directly into a guard candidate."
            )
        else:
            summary = (
                f"Mechanical escape sweep: {attempted} canonical inputs replayed at the frontier; "
                "none beat it. This obstacle needs a non-standard approach."
            )
        if verbose:
            print(summary)
        return experiments, summary
    except Exception as e:
        return [], f"Escape sweep failed: {type(e).__name__}: {e}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
