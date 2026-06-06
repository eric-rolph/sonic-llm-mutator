"""Efficiency comparison: an evolved code policy vs. an in-loop VLM agent.

The evolved policy's runtime throughput is measured on the real emulator
(``--measure``); the in-loop VLM figures are projected from per-call price,
latency and decision cadence (no paid API is ever called here -- it is explicit
arithmetic). See core/efficiency.py for the model.

Examples:
    python efficiency_report.py --measure
    python efficiency_report.py --frames 3285 --usd-per-call 0.005 --latency 2.0
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath("."))

from core.actions import action_string_to_array
from core.efficiency import (
    amortized_crossover_runs,
    evolved_runtime_profile,
    format_comparison_table,
    inloop_projection,
)

# Defaults measured on this repo's champion clearing Green Hill Act 1
# (legacy gym-retro backend). Re-measure with --measure on your own hardware.
DEFAULT_ACT1_FRAMES = 3285
DEFAULT_EVOLVED_FPS = 600.0


def measure_clear(policy_path, backend="auto", max_frames=8000):
    """Run the policy until it clears the first act (zone/act changes) or hits a
    cap, timing the local decide+step loop. Returns (frames, wall_clock_s, cleared)."""
    warnings.filterwarnings("ignore")
    from emulator.sonic_env import SonicEnvWrapper
    from main import load_policy

    env = SonicEnvWrapper(backend=backend)
    policy = load_policy(policy_path)
    prev_zone_act = None
    frames = 0
    cleared = False
    start = time.perf_counter()
    while frames < max_frames:
        state = env.get_state()
        zone_act = (state.get("zone"), state.get("act"))
        if prev_zone_act is not None and zone_act != prev_zone_act:
            cleared = True
            break
        prev_zone_act = zone_act
        try:
            action_string = policy.get_action(state)
        except Exception:
            action_string = "RIGHT"
        if not isinstance(action_string, str):
            action_string = "RIGHT"
        _, _, done, _ = env.step(action_string_to_array(action_string))
        frames += 1
        if done:
            break
    wall = time.perf_counter() - start
    env.close()
    return frames, wall, cleared


def build_report(frames, evolved_wall_s, cadences, usd_per_call, latency_s, training_usd):
    evolved = evolved_runtime_profile(frames, evolved_wall_s)
    inloop = [
        inloop_projection(frames, c, usd_per_call, latency_s, emulator_wall_clock_s=evolved_wall_s)
        for c in cadences
    ]
    lines = [format_comparison_table([evolved, *inloop])]
    lines.append("")
    lines.append(
        f"Assumptions (in-loop): ${usd_per_call:g}/call, {latency_s:g}s latency/call; "
        f"emulator time shared with the evolved run."
    )
    # Break-even vs the cheapest in-loop cadence.
    cheapest = min(inloop, key=lambda p: p.usd_cost)
    crossover = amortized_crossover_runs(training_usd, cheapest.usd_cost)
    if crossover is None:
        lines.append("In-loop is never cheaper per run.")
    elif crossover == 0:
        lines.append(
            f"Training was local-first (~${training_usd:g}), so the evolved policy is "
            f"cheaper from run #1 (in-loop costs {cheapest.usd_cost and f'${cheapest.usd_cost:,.2f}' or '$0'} every run)."
        )
    else:
        lines.append(
            f"With a one-time training cost of ${training_usd:,.2f}, the evolved policy "
            f"breaks even after ~{crossover} runs vs the cheapest in-loop cadence."
        )
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measure", action="store_true", help="measure the evolved policy on the real emulator")
    p.add_argument("--policy", default="policies/champion_policy.py")
    p.add_argument("--backend", default="auto", choices=["auto", "stable", "legacy"])
    p.add_argument("--frames", type=int, default=None, help=f"frames to clear Act 1 (default: {DEFAULT_ACT1_FRAMES})")
    p.add_argument("--evolved-fps", type=float, default=DEFAULT_EVOLVED_FPS, help="used when not measuring")
    p.add_argument("--frames-per-decision", type=int, nargs="+", default=[4, 12],
                   help="in-loop decision cadence(s)")
    p.add_argument("--usd-per-call", type=float, default=0.003)
    p.add_argument("--latency", type=float, default=1.5, help="seconds per in-loop model call")
    p.add_argument("--training-usd", type=float, default=0.0,
                   help="one-time evolutionary training cost (local-first ~= 0)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.measure:
        frames, wall, cleared = measure_clear(args.policy, args.backend)
        if not cleared:
            print(f"(note: did not clear Act 1 within the cap; reporting the {frames}-frame run)\n")
    else:
        frames = args.frames or DEFAULT_ACT1_FRAMES
        wall = frames / max(1e-9, args.evolved_fps)
    print(build_report(frames, wall, args.frames_per_decision, args.usd_per_call, args.latency, args.training_usd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
