"""Deterministic frontier-guard candidates.

When the working policy's trace proves it is repeatedly stationary at one
zone/act/x frontier, one candidate slot gets a narrow, mechanical recovery
guard prepended to the working code. This gives the search a hill-climbing
path around full-policy rewrites that would regress earlier acts.
"""

import ast
import re

from core.actions import GENESIS_BUTTONS
from core.policy_validator import validate_policy_source


def build_frontier_guard_candidate(working_code, trace, sample_count=3, x_radius=25):
    """Add one narrow recovery guard when the working policy repeatedly stalls."""
    samples = list(trace or [])[-sample_count:]
    if len(samples) < sample_count:
        return None

    zone = samples[-1].get("zone")
    act = samples[-1].get("act")
    xs = [int(sample.get("x", 0)) for sample in samples]
    velocities = [abs(float(sample.get("x_velocity", 0) or 0)) for sample in samples]
    if any((sample.get("zone"), sample.get("act")) != (zone, act) for sample in samples):
        return None
    if max(xs) - min(xs) > x_radius or max(velocities) >= 0.5:
        return None

    frontier_x = round(sum(xs) / len(xs))
    marker = f"# FRONTIER_GUARD zone={zone} act={act} x={frontier_x}"
    for existing_zone, existing_act, existing_x in re.findall(
        r"# FRONTIER_GUARD zone=(\S+) act=(\S+) x=(-?\d+)",
        working_code,
    ):
        if (
            existing_zone == str(zone)
            and existing_act == str(act)
            and abs(int(existing_x) - frontier_x) <= x_radius * 2
        ):
            return None

    last_action = str(samples[-1].get("action", ""))
    recovery_action = "RIGHT,B" if "DOWN" in last_action or "B" not in last_action else "RIGHT"

    try:
        tree = ast.parse(working_code)
        function = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_action"
        )
        first_body_line = function.body[0].lineno - 1
    except (SyntaxError, StopIteration, IndexError):
        return None

    lines = working_code.splitlines(keepends=True)
    indent = lines[first_body_line][:len(lines[first_body_line]) - len(lines[first_body_line].lstrip())]
    lower = frontier_x - x_radius
    upper = frontier_x + x_radius
    guard = [
        f"{indent}{marker}\n",
        f"{indent}if (\n",
        f"{indent}    state.get(\"zone\") == {zone!r}\n",
        f"{indent}    and state.get(\"act\") == {act!r}\n",
        f"{indent}    and {lower} <= state.get(\"x_pos\", 0) <= {upper}\n",
        f"{indent}    and abs(state.get(\"x_velocity\", 0)) < 0.5\n",
        f"{indent}):\n",
        f"{indent}    return \"{recovery_action}\"\n",
        "\n",
    ]
    candidate = "".join(lines[:first_body_line] + guard + lines[first_body_line:])
    try:
        validate_policy_source(candidate)
    except Exception:
        return None
    return candidate


def frontier_guard_marker(code):
    match = re.search(r"# FRONTIER_GUARD zone=\S+ act=\S+ x=-?\d+", code or "")
    return match.group(0) if match else None


def _insert_guard_lines(working_code, guard_body_lines):
    """Insert guard lines at the top of get_action, preserving indentation."""
    try:
        tree = ast.parse(working_code)
        function = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_action"
        )
        first_body_line = function.body[0].lineno - 1
    except (SyntaxError, StopIteration, IndexError):
        return None

    lines = working_code.splitlines(keepends=True)
    indent = lines[first_body_line][:len(lines[first_body_line]) - len(lines[first_body_line].lstrip())]
    guard = [f"{indent}{line}\n" for line in guard_body_lines] + ["\n"]
    candidate = "".join(lines[:first_body_line] + guard + lines[first_body_line:])
    try:
        validate_policy_source(candidate)
    except Exception:
        return None
    return candidate


def build_diagnosis_guard_candidate(working_code, experiment, x_radius=25):
    """Compile a VERIFIED diagnosis experiment into a mechanical guard.

    The diagnosis session *measured* that holding ``actions`` from around
    ``start_x`` beat the run's frontier. Injecting exactly that input over the
    verified traversal range removes the riskiest step — trusting an LLM to
    translate its own finding into code — from the loop.
    """
    if not isinstance(experiment, dict):
        return None
    try:
        zone = int(experiment["zone"])
        act = int(experiment["act"])
        start_x = int(experiment["start_x"])
        max_x = int(experiment["max_x"])
    except (KeyError, TypeError, ValueError):
        return None
    if max_x <= start_x:
        return None

    marker = f"# DIAGNOSIS_GUARD zone={zone} act={act} x={start_x}"
    for existing_zone, existing_act, existing_x in re.findall(
        r"# DIAGNOSIS_GUARD zone=(\S+) act=(\S+) x=(-?\d+)",
        working_code,
    ):
        if (
            existing_zone == str(zone)
            and existing_act == str(act)
            and abs(int(existing_x) - start_x) <= x_radius * 2
        ):
            # A newer verified escape at the same spot SUPERSEDES the old
            # guard (live-observed: a promoted x-threshold guard blocked its
            # own frame-replay replacement, freezing iteration at the dip).
            stripped = _strip_guard_block(
                working_code,
                f"# DIAGNOSIS_GUARD zone={existing_zone} act={existing_act} x={existing_x}",
            )
            if stripped is None:
                return None  # guard was hand-mangled by a mutation; stay safe
            working_code = stripped

    lower = start_x - x_radius
    # Apply the verified input through the measured traversal, then hand
    # control back to the existing policy at the new frontier.
    upper = max_x
    head = [
        marker,
        "if (",
        f"    state.get(\"zone\") == {zone!r}",
        f"    and state.get(\"act\") == {act!r}",
        f"    and {lower} <= state.get(\"x_pos\", 0) < {upper}",
        "):",
    ]

    segments = experiment.get("segments") or []
    if len(segments) >= 2:
        # Compile a timed sequence as FRAME REPLAY anchored on the first
        # crossing of the sequence's start x. The deterministic emulator and
        # unchanged approach code reproduce the arrival state, so replaying
        # the measured frame counts is faithful — unlike x-threshold dispatch,
        # which live testing showed releases B after the few frames it takes
        # to cross a band, turning the verified full jump into a short hop.
        cleaned = []
        for segment in segments:
            seg_actions = _valid_actions(segment.get("actions") if isinstance(segment, dict) else None)
            try:
                seg_frames = max(1, int(segment["frames"]))
            except (KeyError, TypeError, ValueError):
                return None
            if seg_actions is None:
                return None
            cleaned.append((seg_frames, seg_actions))
        total_frames = sum(frames for frames, _ in cleaned)

        counter = f"_DIAG_REPLAY_{zone}_{act}_{start_x}"
        body = [
            f"global {counter}",
            f"if {counter!r} not in globals():",
            f"    {counter} = -1",
            "if (",
            f"    state.get(\"zone\") == {zone!r}",
            f"    and state.get(\"act\") == {act!r}",
            f"    and {counter} < {total_frames}",
            f"    and ({counter} >= 0 or {lower} <= state.get(\"x_pos\", 0) <= {start_x + x_radius})",
            "):",
            f"    {counter} = {counter} + 1",
        ]
        threshold = 0
        for seg_frames, seg_actions in cleaned[:-1]:
            threshold += seg_frames
            body.append(f"    if {counter} < {threshold}:")
            body.append(f"        return \"{seg_actions}\"")
        body.append(f"    return \"{cleaned[-1][1]}\"")
        guard_lines = [marker] + body
        return _insert_guard_lines(working_code, guard_lines)

    actions = _valid_actions(experiment.get("actions"))
    if actions is None:
        return None
    return _insert_guard_lines(working_code, head + [f"    return \"{actions}\""])


def _valid_actions(actions):
    """Keep only valid Genesis buttons — exactly what the emulator held."""
    tokens = [t.strip().upper() for t in str(actions or "").split(",")]
    valid_tokens = [t for t in tokens if t in GENESIS_BUTTONS]
    return ",".join(valid_tokens) if valid_tokens else None


def _strip_guard_block(code, marker_text, max_block_lines=40):
    """Remove one generated guard block: its marker line through the blank
    line every insertion appends. Returns None when the block no longer has
    that shape (e.g. a mutation rewrote it), so callers can stay conservative.
    """
    lines = code.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == marker_text),
        None,
    )
    if start is None:
        return None
    for offset in range(1, max_block_lines + 1):
        end = start + offset
        if end >= len(lines):
            return None
        if lines[end].strip() == "":
            return "".join(lines[:start] + lines[end + 1:])
    return None


def diagnosis_guard_marker(code):
    match = re.search(r"# DIAGNOSIS_GUARD zone=\S+ act=\S+ x=-?\d+", code or "")
    return match.group(0) if match else None


def build_llm_guard_candidate(working_code, proposal, x_radius=25):
    """Compile an UNVERIFIED structured model proposal into a preserving guard.

    Unlike a free-form rewrite, a proposal can only PREPEND a narrow recovery
    guard to the working policy; it can never regress the code that already
    works. The model decides only WHAT to try (buttons, and optionally how many
    frames to hold them); the WHERE (zone/act/x) comes from authoritative
    emulator state, and the normal evaluation -- not the model -- confirms
    whether it helps. Marked distinctly from a VERIFIED diagnosis guard so the
    two never supersede each other.
    """
    if not isinstance(proposal, dict):
        return None
    try:
        zone = int(proposal["zone"])
        act = int(proposal["act"])
        x = int(proposal["x"])
    except (KeyError, TypeError, ValueError):
        return None
    actions = _valid_actions(proposal.get("action", proposal.get("actions")))
    if actions is None:
        return None
    try:
        hold_frames = int(proposal.get("hold_frames", 0) or 0)
    except (TypeError, ValueError):
        hold_frames = 0
    hold_frames = max(0, min(hold_frames, 300))

    marker = f"# LLM_GUARD zone={zone} act={act} x={x}"
    for existing_zone, existing_act, existing_x in re.findall(
        r"# LLM_GUARD zone=(\S+) act=(\S+) x=(-?\d+)",
        working_code,
    ):
        if (
            existing_zone == str(zone)
            and existing_act == str(act)
            and abs(int(existing_x) - x) <= x_radius * 2
        ):
            return None

    lower = x - x_radius
    upper = x + x_radius
    if hold_frames >= 1:
        # Hold the proposed action for hold_frames once Sonic first reaches the
        # band, mirroring the verified frame-replay guard (the deterministic
        # emulator makes the replay faithful).
        counter = "_LLM_REPLAY_" + f"{zone}_{act}_{x}".replace("-", "n")
        body = [
            marker,
            f"global {counter}",
            f"if {counter!r} not in globals():",
            f"    {counter} = -1",
            "if (",
            f'    state.get("zone") == {zone!r}',
            f'    and state.get("act") == {act!r}',
            f"    and {counter} < {hold_frames}",
            f'    and ({counter} >= 0 or {lower} <= state.get("x_pos", 0) <= {upper})',
            "):",
            f"    {counter} = {counter} + 1",
            f'    return "{actions}"',
        ]
        return _insert_guard_lines(working_code, body)

    body = [
        marker,
        "if (",
        f'    state.get("zone") == {zone!r}',
        f'    and state.get("act") == {act!r}',
        f'    and {lower} <= state.get("x_pos", 0) <= {upper}',
        "):",
        f'    return "{actions}"',
    ]
    return _insert_guard_lines(working_code, body)


def llm_guard_marker(code):
    match = re.search(r"# LLM_GUARD zone=\S+ act=\S+ x=-?\d+", code or "")
    return match.group(0) if match else None


def recently_attempted_frontier_guard(marker, recent_history):
    for entry in recent_history or []:
        recorded_marker = entry.get("frontier_guard_marker")
        if recorded_marker is None:
            recorded_marker = entry.get("components", {}).get("frontier_guard_marker")
        recorded_markers = entry.get("components", {}).get("frontier_guard_markers", [])
        if marker in recorded_markers:
            return True
        if recorded_marker == marker:
            return True
    return False
