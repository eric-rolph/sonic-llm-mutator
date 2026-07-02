import unittest

from llm.mutator import extract_policy_code, extract_python_block, slim_history

GOOD = "def get_action(state):\n    return 'RIGHT'"


class SlimHistoryTests(unittest.TestCase):
    def test_drops_paths_and_truncates_reasoning(self):
        entry = {
            "generation": 577,
            "fitness": 54442.86,
            "failure_reason": "lost a life at the frontier",
            "llm_reasoning": "x" * 5000,
            "screenshot": "artifacts/failures/generation_577_best.png",
            "archive_path": "policies/archive/gen_577.py",
            "components": {"frontier": {"zone": 0, "act": 1, "x": 4272}, "levels_cleared": 1,
                           "distance": 28070.0, "speed": 1339.0},
        }
        slim = slim_history([entry])[0]

        self.assertEqual(slim["generation"], 577)
        self.assertEqual(slim["frontier"], {"zone": 0, "act": 1, "x": 4272})
        self.assertEqual(slim["levels_cleared"], 1)
        self.assertEqual(len(slim["llm_reasoning"]), 200)  # truncated
        self.assertNotIn("screenshot", slim)               # paths dropped
        self.assertNotIn("archive_path", slim)
        self.assertNotIn("distance", slim)

    def test_handles_junk_entries(self):
        self.assertEqual(slim_history(None), [])
        self.assertEqual(slim_history(["not-a-dict", 42]), [])


class CrossoverBudgetTests(unittest.TestCase):
    def _mutator(self):
        from llm.mutator import MutatorClient

        class CapturingMutator(MutatorClient):
            def __init__(self):
                self.prompts = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.prompts.append(prompt)
                return "def get_action(state):\n    return 'RIGHT'", "ok"

        return CapturingMutator()

    def test_normal_parents_cross_over_within_budget(self):
        from contextlib import redirect_stdout
        from io import StringIO

        from llm.mutator import PROMPT_CHAR_BUDGET

        mutator = self._mutator()
        with redirect_stdout(StringIO()):
            code, _ = mutator.crossover_policies("def get_action(state):\n    return 'RIGHT'",
                                                 "def get_action(state):\n    return 'LEFT'", [])
        self.assertIn("get_action", code)
        self.assertLessEqual(len(mutator.prompts[0]), PROMPT_CHAR_BUDGET)

    def test_oversized_parents_skip_crossover_explicitly(self):
        # Two grown champions can exceed a local model's context by themselves;
        # a faithful merge is impossible in-window, so skip loudly (the caller
        # already treats a raising crossover as a filled-by-fallback slot)
        # instead of hard-failing the API call.
        mutator = self._mutator()
        huge = "def get_action(state):\n    return 'RIGHT'\n" + ("# pad\n" * 4000)
        with self.assertRaises(ValueError):
            mutator.crossover_policies(huge, huge, [])
        self.assertEqual(mutator.prompts, [])  # no over-budget call was made


class PromptBudgetTests(unittest.TestCase):
    def test_oversized_history_is_dropped_to_fit_the_budget(self):
        from llm.mutator import PROMPT_CHAR_BUDGET, MutatorClient

        class CapturingMutator(MutatorClient):
            def __init__(self):
                self.prompts = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.prompts.append(prompt)
                return "def get_action(state):\n    return 'RIGHT'", "ok"

        # 400 history entries survive slim_history (~200 chars each) and blow
        # far past the budget; the staged trim must drop them rather than send
        # an over-budget prompt that hard-fails on an 8k-context local model.
        history = [
            {"generation": i, "fitness": 1.0, "failure_reason": "R" * 150}
            for i in range(400)
        ]
        mutator = CapturingMutator()
        from contextlib import redirect_stdout
        from io import StringIO
        with redirect_stdout(StringIO()):
            code, _ = mutator.mutate_policy(
                "def get_action(state):\n    return 'RIGHT'",
                "Policy code timeout (infinite loop).",  # routes to micro, skips guard path
                None,
                history,
            )

        self.assertEqual(len(mutator.prompts), 1)
        self.assertLessEqual(len(mutator.prompts[0]), PROMPT_CHAR_BUDGET)
        self.assertIn("def get_action", mutator.prompts[0])  # code never dropped


class ExtractPythonBlockTests(unittest.TestCase):
    def test_skills_style_extraction_skips_truncated_draft(self):
        # Skills responses have no get_action; the generic extractor keeps the
        # last block that PARSES, skipping a truncated reasoning draft.
        resp = (
            "Draft:\n```python\ndef boost(state):\n    if state['x'] ==\n```\n"
            "Final:\n```python\ndef boost(state):\n    return 'RIGHT,B'\n```"
        )
        self.assertEqual(extract_python_block(resp), "def boost(state):\n    return 'RIGHT,B'")

    def test_falls_back_to_last_block_when_nothing_parses(self):
        resp = "```python\ndef a(:\n```\n```python\ndef b(:\n```"
        self.assertEqual(extract_python_block(resp), "def b(:")


class ExtractPolicyCodeTests(unittest.TestCase):
    def test_extracts_single_fenced_python_block(self):
        resp = f"Here is the policy:\n```python\n{GOOD}\n```"
        self.assertEqual(extract_policy_code(resp), GOOD)

    def test_extracts_unlabelled_fence(self):
        resp = f"```\n{GOOD}\n```"
        self.assertEqual(extract_policy_code(resp), GOOD)

    def test_prefers_last_complete_block_over_truncated_draft(self):
        # Reasoning scratchpad: a truncated draft first, then the real policy.
        resp = (
            "Let me draft this.\n```python\ndef get_action(state):\n    if state['x'] ==\n```\n"
            f"Final version:\n```python\n{GOOD}\n```"
        )
        self.assertEqual(extract_policy_code(resp), GOOD)

    def test_skips_trailing_non_policy_block(self):
        # The last block is a state-dict example; the real policy is earlier.
        resp = (
            f"```python\n{GOOD}\n```\n"
            "For reference the state looks like:\n```python\nstate = {'x_pos': 10}\n```"
        )
        self.assertEqual(extract_policy_code(resp), GOOD)

    def test_extracts_bare_code_without_fences(self):
        self.assertEqual(extract_policy_code(GOOD), GOOD)

    def test_falls_back_to_last_block_when_no_valid_policy(self):
        # Nothing defines get_action -> return last block so validator/repair runs
        # instead of silently dropping the output.
        resp = "```python\nx = 1\n```\n```python\ny = 2\n```"
        self.assertEqual(extract_policy_code(resp), "y = 2")

    def test_empty_input_returns_empty(self):
        self.assertEqual(extract_policy_code(""), "")
        self.assertEqual(extract_policy_code(None), "")


if __name__ == "__main__":
    unittest.main()
