import unittest

from llm.prompts import RECOMMENDED_ACTIONS, SYSTEM_PROMPT


class PromptActionsTests(unittest.TestCase):
    def test_prompt_recommends_constrained_macro_actions(self):
        self.assertIn("Recommended macro-actions", SYSTEM_PROMPT)
        self.assertIn("RIGHT", RECOMMENDED_ACTIONS)
        self.assertIn("RIGHT,B", RECOMMENDED_ACTIONS)
        self.assertIn("RIGHT,DOWN", RECOMMENDED_ACTIONS)


if __name__ == "__main__":
    unittest.main()
