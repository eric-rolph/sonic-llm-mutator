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
