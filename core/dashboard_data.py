"""Pure data shaping for the Streamlit dashboard.

Everything here is streamlit-free and unit-testable; dashboard.py stays a thin
rendering layer. Built to address the agency design-panel review: surface the
frontier, grade stagnation honestly (no fake denominator), expose liveness,
and turn raw pipeline artifacts into glanceable signals.
"""

import re
import time

# Sonic 1 zone order; acts are 0-based in RAM.
ZONE_NAMES = ("Green Hill", "Marble", "Spring Yard", "Labyrinth", "Star Light", "Scrap Brain")

# A run is presumed stopped when no generation lands for this long. Generations
# take ~1-5 minutes depending on diagnosis/LLM latency.
STALE_AFTER_SECONDS = 600

_GUARD_MARKER = re.compile(r"# (FRONTIER|DIAGNOSIS|LLM)_GUARD zone=(-?\d+) act=(-?\d+) x=(-?\d+)")

_GUARD_KIND_LABELS = {
    "FRONTIER": "recovery move",
    "DIAGNOSIS": "verified escape",
    "LLM": "proposed move",
}


def zone_act_label(zone, act):
    """Human name for a (zone, act) pair; falls back to raw numbers."""
    try:
        zone_index = int(zone)
        act_number = int(act) + 1
    except (TypeError, ValueError):
        return "Unknown act"
    if 0 <= zone_index < len(ZONE_NAMES):
        return f"{ZONE_NAMES[zone_index]} Act {act_number}"
    return f"Zone {zone_index} Act {act_number}"


def champion_entry(history):
    """The highest-fitness generation entry, or None."""
    entries = [e for e in history or [] if isinstance(e, dict)]
    return max(entries, key=lambda e: e.get("fitness", float("-inf")), default=None)


def is_new_champion(history):
    """True when the latest generation set the all-time record."""
    if not history:
        return False
    best = champion_entry(history)
    return best is not None and best.get("generation") == history[-1].get("generation")


def run_liveness(latest_entry, now=None):
    """({'text', 'stale', 'minutes'}) describing how fresh the run data is.

    Without this, a crashed run is indistinguishable from a live one — the
    dashboard said "making steady progress" forever (agency review).
    """
    now = time.time() if now is None else now
    try:
        timestamp = int(latest_entry.get("timestamp"))
    except (AttributeError, TypeError, ValueError):
        return {"text": "no run data yet", "stale": True, "minutes": None}
    minutes = max(0, int((now - timestamp) // 60))
    if minutes < 1:
        text = "updated moments ago"
    elif minutes == 1:
        text = "updated 1 minute ago"
    else:
        text = f"updated {minutes} minutes ago"
    return {"text": text, "stale": (now - timestamp) > STALE_AFTER_SECONDS, "minutes": minutes}


def stagnation_status(count):
    """Grade stagnation WITHOUT a denominator.

    The runtime stagnation limit is a launch parameter the artifacts do not
    record, and the old hardcoded "/5" showed impossible values like 14/5
    (agency review: flagged by four personas).
    """
    try:
        count = max(0, int(count))
    except (TypeError, ValueError):
        count = 0
    if count == 0:
        return {"level": "success", "text": "Progressing — the latest generation improved the champion."}
    if count < 5:
        return {"level": "info", "text": f"{count} generation(s) without improvement."}
    if count < 10:
        return {
            "level": "warning",
            "text": f"{count} generations without improvement — the mutator is grinding at a hard frontier.",
        }
    return {
        "level": "error",
        "text": f"{count} generations without improvement — a stagnation-escape strategy shift is due.",
    }


def frontier_summary(components):
    """The champion's current wall as a glanceable dict, or None.

    ``components.frontier`` carries the authoritative (zone, act, x); this was
    the review's top finding — the user's core question ("WHERE does it die?")
    was answered nowhere on the page.
    """
    frontier = (components or {}).get("frontier")
    if not isinstance(frontier, dict):
        return None
    try:
        zone, act, x = int(frontier["zone"]), int(frontier["act"]), int(frontier["x"])
    except (KeyError, TypeError, ValueError):
        return None
    summary = {"zone": zone, "act": act, "x": x, "label": zone_act_label(zone, act)}
    try:
        target = int((components or {}).get("completion_target", 0) or 0)
    except (TypeError, ValueError):
        target = 0
    if target > 0:
        summary["completion_target"] = target
        summary["progress"] = max(0.0, min(1.0, x / target))
    return summary


def beaten_acts(components):
    """Names of the acts already cleared this run (the actual trophies)."""
    frontier = frontier_summary(components)
    try:
        cleared = int((components or {}).get("levels_cleared", 0) or 0)
    except (TypeError, ValueError):
        cleared = 0
    if cleared <= 0 or frontier is None:
        return []
    zone, act = frontier["zone"], frontier["act"]
    names = []
    for _ in range(cleared):
        act -= 1
        if act < 0:
            zone -= 1
            act = 2  # Sonic 1 zones have 3 acts (Scrap Brain's oddity ignored)
        if zone < 0:
            break
        names.insert(0, zone_act_label(zone, act))
    return names


def chart_series(history, window=200):
    """Chart-ready dict of aligned lists over the last ``window`` generations.

    Adds the cumulative-max champion staircase: a single raw-fitness line hid
    all recent signal under attempt noise once the completion bonus flattened
    the scale (agency review).
    """
    entries = [e for e in history or [] if isinstance(e, dict)][-max(1, int(window)):]
    generations, attempts, champions = [], [], []
    # The staircase must account for records set BEFORE the window.
    running_max = float("-inf")
    for entry in (history or [])[: max(0, len(history or []) - len(entries))]:
        if isinstance(entry, dict):
            running_max = max(running_max, entry.get("fitness", float("-inf")))
    for entry in entries:
        fitness = entry.get("fitness", 0)
        running_max = max(running_max, fitness)
        generations.append(entry.get("generation"))
        attempts.append(fitness)
        champions.append(running_max)
    return {"generation": generations, "attempt": attempts, "champion": champions}


def chart_caption(history):
    """One-sentence text alternative to the chart (accessibility + glance)."""
    if not history:
        return "No generations recorded yet."
    latest = history[-1]
    best = champion_entry(history) or {}
    stagnation = stagnation_status(latest.get("stagnation_counter", 0))
    return (
        f"Champion fitness {best.get('fitness', 0):,.0f} (gen {best.get('generation', '?')}); "
        f"latest attempt {latest.get('fitness', 0):,.0f} (gen {latest.get('generation', '?')}). "
        + stagnation["text"]
    )


def learned_moves(champion_code):
    """The moves Sonic literally learned: guard markers parsed from the champion.

    Returns newest-first dicts with a human label per conquered frontier.
    """
    moves = []
    for kind, zone, act, x in _GUARD_MARKER.findall(champion_code or ""):
        moves.append(
            {
                "kind": kind,
                "zone": int(zone),
                "act": int(act),
                "x": int(x),
                "label": f"{_GUARD_KIND_LABELS[kind]} at {zone_act_label(zone, act)} x={x}",
            }
        )
    return moves


def video_caption(kind, entry):
    """Identity caption for a video player ('which gen, what fitness?')."""
    if not isinstance(entry, dict):
        return f"{kind}: no run recorded yet"
    return (
        f"{kind} — generation {entry.get('generation', '?')}, "
        f"fitness {entry.get('fitness', 0):,.0f}"
    )


def diagnosis_freshness(report, now=None):
    """Age text for the diagnosis report, or None when unavailable."""
    now = time.time() if now is None else now
    try:
        created = int((report or {}).get("created_at"))
    except (TypeError, ValueError):
        return None
    minutes = max(0, int((now - created) // 60))
    return "from moments ago" if minutes < 1 else f"from {minutes} minute(s) ago"
