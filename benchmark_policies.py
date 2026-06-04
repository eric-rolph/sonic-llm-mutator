import argparse
import importlib.util
import json
import os
from pathlib import Path

from main import evaluate_policy


DEFAULT_STATES = [
    "GreenHillZone.Act1",
    "GreenHillZone.Act2",
    "GreenHillZone.Act3",
    "MarbleZone.Act1",
    "SpringYardZone.Act1",
]

DEFAULT_POLICIES = [
    "policies/champion_policy.py",
    "policies/working_policy.py",
    "policies/current_policy.py",
]


class NoVisionMutator:
    def analyze_environment(self, screenshot_path):
        return "UNKNOWN"


def failure_row(state, backend, policy_path, reason):
    return {
        "state": state,
        "backend": backend,
        "policy": policy_label(policy_path),
        "fitness": 0.0,
        "max_x": 0,
        "frames": 0,
        "reason": reason,
        "trace": [],
        "components": {},
    }


def policy_label(policy_path):
    return Path(policy_path).stem


def load_policy(policy_path):
    path = Path(policy_path)
    module_name = f"benchmark_{path.stem}_{abs(hash(str(path.resolve()))) % 1000000}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load policy: {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_action"):
        raise ValueError(f"Policy does not define get_action: {policy_path}")
    return module


def evaluate_policy_on_state(policy_path, state, max_frames, backend="auto", action_repeat=1):
    from emulator.sonic_env import SonicEnvWrapper

    policy = load_policy(policy_path)
    env = SonicEnvWrapper(state=state, record_path=None, backend=backend)
    try:
        fitness, frames, max_x, reason, _, trace, components = evaluate_policy(
            env,
            policy,
            NoVisionMutator(),
            max_frames=max_frames,
            verbose=False,
            action_repeat=action_repeat,
        )
    finally:
        env.close()

    return {
        "state": state,
        "backend": env.backend,
        "policy": policy_label(policy_path),
        "fitness": round(float(fitness), 2),
        "max_x": int(max_x),
        "frames": int(frames),
        "reason": reason,
        "trace": trace,
        "components": components,
    }


def run_benchmark(policy_paths=None, states=None, max_frames=5000, backend="auto", action_repeat=1):
    selected_policies = policy_paths or DEFAULT_POLICIES
    selected_states = states or DEFAULT_STATES
    rows = []

    for state in selected_states:
        for policy_path in selected_policies:
            if not os.path.exists(policy_path):
                rows.append(failure_row(state, backend, policy_path, f"Policy file not found: {policy_path}"))
                continue
            try:
                rows.append(evaluate_policy_on_state(policy_path, state, max_frames, backend=backend, action_repeat=action_repeat))
            except Exception as e:
                rows.append(failure_row(state, backend, policy_path, f"Benchmark failed: {e}"))

    return rows


def format_results_table(rows):
    columns = ["state", "backend", "policy", "fitness", "max_x", "frames", "reason"]
    normalized = []
    for row in rows:
        normalized.append(
            {
                "state": str(row["state"]),
                "backend": str(row.get("backend", "")),
                "policy": str(row["policy"]),
                "fitness": f"{float(row['fitness']):.2f}",
                "max_x": str(row["max_x"]),
                "frames": str(row["frames"]),
                "reason": str(row["reason"]),
            }
        )

    widths = {
        column: max(len(column), *(len(row[column]) for row in normalized))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(row[column].ljust(widths[column]) for column in columns)
        for row in normalized
    ]
    return "\n".join([header, divider] + body)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark Sonic policies across installed states.")
    parser.add_argument("--states", nargs="+", default=DEFAULT_STATES)
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--backend", choices=["auto", "stable", "legacy"], default="auto")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = run_benchmark(args.policies, args.states, args.max_frames, backend=args.backend, action_repeat=args.action_repeat)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_results_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
