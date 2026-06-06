import os
import tempfile
import unittest
from unittest.mock import patch

import main
from main import (
    build_stagnation_escape_context,
    evaluate_working_baseline,
    preserve_frontier_screenshot,
    resolve_end_generation,
    seed_population_baseline,
    select_working_frontier_context,
)


class RecordingArchive:
    def __init__(self):
        self.calls = []

    def record_evaluation(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class NoVisionMutator:
    def analyze_environment(self, screenshot_path):
        return "UNKNOWN"


class FakeEnv:
    def __init__(self, states):
        self.states = list(states)
        self.index = 0

    def reset(self):
        self.index = 0
        return None

    def get_state(self):
        if self.index >= len(self.states):
            return dict(self.states[-1])
        state = dict(self.states[self.index])
        self.index += 1
        return state

    def step(self, action):
        return None, 0, False, {}

    def get_screenshot(self):
        return "baseline_screenshot.png"


class RunResumeTests(unittest.TestCase):
    def write_policy(self, directory, code):
        policy_path = os.path.join(directory, "policy.py")
        with open(policy_path, "w", encoding="utf-8") as f:
            f.write(code)
        return policy_path

    def test_evaluate_working_baseline_seeds_resume_context(self):
        states = [{"x_pos": 100, "y_pos": 100, "rings": 0, "score": 0}] * 700
        env = FakeEnv(states)
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = self.write_policy(
                tmp,
                "def get_action(state):\n    return 'RIGHT'\n",
            )

            context = evaluate_working_baseline(
                env,
                policy_path,
                NoVisionMutator(),
                max_frames=700,
                verbose=False,
            )

        self.assertGreater(context["working_fitness"], -1)
        self.assertEqual(context["last_screenshot"], "baseline_screenshot.png")
        self.assertIn("stopped making forward progress", context["last_failure_reason"])
        self.assertEqual(context["last_trace"][-1]["x"], 100)
        self.assertEqual(context["last_trace"][-1]["y"], 100)
        self.assertEqual(context["last_trace"][-1]["action"], "RIGHT")
        self.assertIn("distance", context["components"])

    def test_evaluate_working_baseline_handles_missing_policy(self):
        context = evaluate_working_baseline(
            env=None,
            working_path="missing.py",
            mutator=NoVisionMutator(),
            max_frames=700,
            verbose=False,
        )

        self.assertEqual(context["working_fitness"], -1.0)
        self.assertEqual(context["last_failure_reason"], "Initial seed run")
        self.assertIsNone(context["last_screenshot"])
        self.assertEqual(context["last_trace"], [])

    def test_seed_population_baseline_records_the_strong_working_policy(self):
        archive = RecordingArchive()
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = self.write_policy(tmp, "def get_action(state):\n    return 'RIGHT'\n")
            recorded = seed_population_baseline(
                archive,
                policy_path,
                {
                    "working_fitness": 47977.37,
                    "components": {"levels_cleared": 1},
                    "last_failure_reason": "stuck in act 2",
                    "last_trace": [{"zone": 0, "act": 1, "x": 1077}],
                },
            )

        self.assertTrue(recorded)
        self.assertEqual(archive.calls[0][1]["fitness"], 47977.37)
        self.assertEqual(archive.calls[0][1]["components"]["levels_cleared"], 1)

    def test_stagnation_escape_preserves_working_policy_context(self):
        trace = [{"zone": 0, "act": 1, "x": 1077}]

        context = build_stagnation_escape_context(47977.37, trace, "failure.png")

        self.assertEqual(context["working_fitness"], 47977.37)
        self.assertEqual(context["last_trace"], trace)
        self.assertEqual(context["last_screenshot"], "failure.png")
        self.assertIn("preserve", context["last_failure_reason"].lower())

    def test_losing_candidate_does_not_replace_working_frontier_context(self):
        working = {
            "failure_reason": "champion stuck in act 2",
            "trace": [{"zone": 0, "act": 1, "x": 1077}],
            "screenshot": "working-frontier.png",
        }
        losing_candidate = {
            "failure_reason": "candidate stuck in act 1",
            "trace": [{"zone": 0, "act": 0, "x": 3061}],
            "screenshot": "candidate.png",
        }

        selected = select_working_frontier_context(working, losing_candidate, promoted=False)

        self.assertEqual(selected, working)
        self.assertIsNot(selected, working)

    def test_promoted_candidate_replaces_working_frontier_context(self):
        working = {"failure_reason": "old", "trace": [], "screenshot": "old.png"}
        promoted = {"failure_reason": "new", "trace": [1], "screenshot": "new.png"}

        self.assertEqual(select_working_frontier_context(working, promoted, promoted=True), promoted)

    def test_frontier_screenshot_is_copied_away_from_shared_candidate_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "latest.png")
            destination = os.path.join(tmp, "working-frontier.png")
            with open(source, "wb") as f:
                f.write(b"champion")

            preserved = preserve_frontier_screenshot(source, destination)
            with open(source, "wb") as f:
                f.write(b"losing candidate")

            with open(preserved, "rb") as f:
                self.assertEqual(f.read(), b"champion")

    def test_additional_generation_limit_is_relative_to_resume_point(self):
        self.assertEqual(resolve_end_generation(start_gen=498, max_generations=500, generations=15), 512)

    def test_cli_passes_bounded_run_options_to_evaluation_loop(self):
        with patch.object(main, "run_evaluation_loop") as run:
            self.assertEqual(main.main(["--generations", "15", "--frames", "2000"]), 0)

        run.assert_called_once_with(generations=15, max_frames=2000)


if __name__ == "__main__":
    unittest.main()
