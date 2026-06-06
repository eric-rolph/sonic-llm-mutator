"""Cost / throughput model comparing the two LLM-game paradigms.

* **code-evolver** (this project): the LLM is used *offline* to evolve a policy.
  At runtime the evolved ``get_action`` makes every decision locally, so a run
  costs **zero** API calls and incurs no per-decision latency.
* **in-loop VLM**: the model chooses an action every ``frames_per_decision``
  frames, so each run re-pays an API call (money + latency) per decision -- on
  every run, forever.

The evolved-policy runtime figures are measured empirically (see
``efficiency_report.py``); the in-loop figures are a transparent projection from
the per-call price / latency / cadence you supply. Nothing here calls a paid API
-- it is pure arithmetic so the assumptions are explicit and adjustable.
"""

import math
from dataclasses import dataclass


@dataclass
class RunProfile:
    """One run over ``frames`` emulator frames."""

    label: str
    frames: int
    wall_clock_s: float
    api_calls: int
    usd_cost: float

    @property
    def fps(self):
        return self.frames / self.wall_clock_s if self.wall_clock_s > 0 else float("inf")

    @property
    def frames_per_usd(self):
        return self.frames / self.usd_cost if self.usd_cost > 0 else float("inf")


def evolved_runtime_profile(frames, wall_clock_s, label="evolved policy (runtime)"):
    """A run of an already-evolved policy: no runtime API calls, no API cost."""
    return RunProfile(label, int(frames), float(wall_clock_s), api_calls=0, usd_cost=0.0)


def inloop_projection(
    frames,
    frames_per_decision,
    usd_per_call,
    latency_s_per_call,
    emulator_wall_clock_s=0.0,
):
    """Project an in-loop VLM agent over the same ``frames``.

    ``frames_per_decision`` is how many frames the agent advances per model call
    (its frame-skip / decision cadence). Wall-clock includes the per-call latency
    plus any baseline emulator time.
    """
    frames_per_decision = max(1, int(frames_per_decision))
    decisions = math.ceil(int(frames) / frames_per_decision)
    return RunProfile(
        label=f"in-loop VLM (1 call / {frames_per_decision} frames)",
        frames=int(frames),
        wall_clock_s=float(emulator_wall_clock_s) + decisions * float(latency_s_per_call),
        api_calls=decisions,
        usd_cost=decisions * float(usd_per_call),
    )


def amortized_crossover_runs(training_usd, inloop_usd_per_run, evolved_usd_per_run=0.0):
    """Number of runs after which the evolved approach (which front-loads a
    one-time ``training_usd``) is cheaper than paying the in-loop cost every run.

    Returns 0 when training was free (cheaper immediately), or ``None`` when the
    in-loop approach is never more expensive per run.
    """
    per_run_saving = float(inloop_usd_per_run) - float(evolved_usd_per_run)
    if per_run_saving <= 0:
        return None
    return math.ceil(float(training_usd) / per_run_saving)


def _fmt_usd(value):
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def _fmt_count(value):
    return "inf" if value == float("inf") else f"{value:,.0f}"


def format_comparison_table(profiles):
    """Render a list of RunProfile rows as a fixed-width table."""
    columns = ["scenario", "frames", "api_calls", "usd_cost", "wall_clock_s", "frames_per_usd"]
    rows = []
    for p in profiles:
        rows.append({
            "scenario": p.label,
            "frames": f"{p.frames:,}",
            "api_calls": f"{p.api_calls:,}",
            "usd_cost": _fmt_usd(p.usd_cost),
            "wall_clock_s": f"{p.wall_clock_s:,.1f}",
            "frames_per_usd": _fmt_count(p.frames_per_usd),
        })
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    divider = "  ".join("-" * widths[c] for c in columns)
    body = ["  ".join(r[c].ljust(widths[c]) for c in columns) for r in rows]
    return "\n".join([header, divider] + body)
