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
    # Keep only valid Genesis buttons — exactly the input the emulator
    # actually held during the experiment (unknown tokens were ignored there).
    tokens = [t.strip().upper() for t in str(experiment.get("actions", "")).split(",")]
    valid_tokens = [t for t in tokens if t in GENESIS_BUTTONS]
    if not valid_tokens or max_x <= start_x:
        return None
    actions = ",".join(valid_tokens)

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
            return None

    lower = start_x - x_radius
    # Apply the verified input through the measured traversal, then hand
    # control back to the existing policy at the new frontier.
    upper = max_x
    guard_lines = [
        marker,
        "if (",
        f"    state.get(\"zone\") == {zone!r}",
        f"    and state.get(\"act\") == {act!r}",
        f"    and {lower} <= state.get(\"x_pos\", 0) < {upper}",
        "):",
        f"    return \"{actions}\"",
    ]
    return _insert_guard_lines(working_code, guard_lines)


def diagnosis_guard_marker(code):
    match = re.search(r"# DIAGNOSIS_GUARD zone=\S+ act=\S+ x=-?\d+", code or "")
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
