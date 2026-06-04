import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from main import generate_candidates, load_pool_codes


class FakeMutator:
    def __init__(self):
        self.mutate_calls = 0
        self.crossover_calls = 0

    def mutate_policy(self, code, reason, screenshot, history, temperature, trace):
        self.mutate_calls += 1
        return f"# mutation t={temperature}", "mutated"

    def crossover_policies(self, parent_a, parent_b, history, temperature=0.7):
        self.crossover_calls += 1
        return "# crossover", "crossed"


class ExplodingMutator(FakeMutator):
    def mutate_policy(self, *args, **kwargs):
        raise RuntimeError("llm unreachable")


def generate_silently(*args, **kwargs):
    with redirect_stdout(StringIO()):
        return generate_candidates(*args, **kwargs)


class GenerateCandidatesTests(unittest.TestCase):
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
