import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from main import build_frontier_guard_candidate, generate_candidates, load_pool_codes


class FakeMutator:
    def __init__(self):
        self.mutate_calls = 0
        self.crossover_calls = 0
        self.parent_pairs = []

    def mutate_policy(self, code, reason, screenshot, history, temperature, trace):
        self.mutate_calls += 1
        return f"# mutation t={temperature}", "mutated"

    def crossover_policies(self, parent_a, parent_b, history, temperature=0.7):
        self.crossover_calls += 1
        self.parent_pairs.append((parent_a, parent_b))
        return "# crossover", "crossed"


class ExplodingMutator(FakeMutator):
    def mutate_policy(self, *args, **kwargs):
        raise RuntimeError("llm unreachable")


def generate_silently(*args, **kwargs):
    with redirect_stdout(StringIO()):
        return generate_candidates(*args, **kwargs)


class GenerateCandidatesTests(unittest.TestCase):
    def test_build_frontier_guard_candidate_preserves_code_and_adds_narrow_recovery(self):
        working = "def get_action(state):\n    return 'RIGHT,DOWN'\n"
        trace = [
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
        ]

        candidate = build_frontier_guard_candidate(working, trace)

        self.assertIn("FRONTIER_GUARD zone=0 act=1 x=1077", candidate)
        self.assertIn('return "RIGHT,B"', candidate)
        self.assertIn("return 'RIGHT,DOWN'", candidate)

    def test_build_frontier_guard_candidate_requires_repeated_stationary_trace(self):
        working = "def get_action(state):\n    return 'RIGHT'\n"
        moving = [
            {"zone": 0, "act": 1, "x": 1000, "x_velocity": 3.0, "action": "RIGHT"},
            {"zone": 0, "act": 1, "x": 1100, "x_velocity": 3.0, "action": "RIGHT"},
            {"zone": 0, "act": 1, "x": 1200, "x_velocity": 3.0, "action": "RIGHT"},
        ]

        self.assertIsNone(build_frontier_guard_candidate(working, moving))

    def test_stationary_frontier_reserves_one_candidate_without_llm_rewrite(self):
        mutator = FakeMutator()
        working = "def get_action(state):\n    return 'RIGHT,DOWN'\n"
        trace = [
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
        ]

        result = generate_silently(
            mutator, working, "reason", None, [], trace, 2, [], crossover_probability=0.0
        )

        self.assertIn("FRONTIER_GUARD", result[0][0])
        self.assertEqual(result[0][1], "Deterministic frontier guard")
        self.assertEqual(mutator.mutate_calls, 1)

    def test_returns_one_code_reasoning_tuple_per_candidate(self):
        mutator = FakeMutator()
        result = generate_silently(
            mutator, "WORKING", "reason", None, [], [], 3, [], crossover_probability=0.0
        )
        self.assertEqual(len(result), 3)
        for code, reasoning in result:
            self.assertIsInstance(code, str)
            self.assertIsInstance(reasoning, str)
        self.assertEqual(mutator.mutate_calls, 3)
        self.assertEqual(mutator.crossover_calls, 0)

    def test_uses_crossover_when_pool_large_and_probability_high(self):
        mutator = FakeMutator()
        pool = ["# parent a", "# parent b"]
        generate_silently(
            mutator, "WORKING", "reason", None, [], [], 2, pool, crossover_probability=1.0
        )
        self.assertEqual(mutator.crossover_calls, 2)
        self.assertEqual(mutator.mutate_calls, 0)

    def test_prefers_exploration_aware_parent_selector_for_crossover(self):
        mutator = FakeMutator()
        selector_calls = []

        def select_parents():
            selector_calls.append(True)
            return "# archived parent a", "# archived parent b"

        generate_silently(
            mutator,
            "WORKING",
            "reason",
            None,
            [],
            [],
            1,
            ["# legacy a", "# legacy b"],
            crossover_probability=1.0,
            parent_selector=select_parents,
        )

        self.assertEqual(selector_calls, [True])
        self.assertEqual(mutator.parent_pairs, [("# archived parent a", "# archived parent b")])

    def test_failed_request_falls_back_to_working_policy(self):
        result = generate_silently(
            ExplodingMutator(), "WORKING_CODE", "reason", None, [], [], 1, [], crossover_probability=0.0
        )
        code, reasoning = result[0]
        self.assertEqual(code, "WORKING_CODE")
        self.assertIn("Failed", reasoning)

    def test_load_pool_codes_reads_pool_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pool_300.00_aaaa.py").write_text("# a")
            (Path(tmp) / "pool_250.00_bbbb.py").write_text("# b")
            (Path(tmp) / "ignored.py").write_text("# not a pool file")

            codes = load_pool_codes(pool_dir=tmp)

        self.assertEqual(sorted(codes), ["# a", "# b"])


if __name__ == "__main__":
    unittest.main()
