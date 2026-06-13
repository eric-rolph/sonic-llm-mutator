import unittest

from llm.mutator import extract_policy_code

GOOD = "def get_action(state):\n    return 'RIGHT'"


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
