import unittest

from core.actions import GENESIS_BUTTONS, action_string_to_array


class ActionsTests(unittest.TestCase):
    def test_button_array_length_matches_button_map(self):
        self.assertEqual(len(action_string_to_array("RIGHT")), len(GENESIS_BUTTONS))
        self.assertEqual(len(GENESIS_BUTTONS), 12)

    def test_parses_multi_button_action(self):
        array = action_string_to_array("RIGHT,B")
        self.assertEqual(array[GENESIS_BUTTONS.index("RIGHT")], 1)
        self.assertEqual(array[GENESIS_BUTTONS.index("B")], 1)
        self.assertEqual(sum(array), 2)

    def test_empty_string_is_no_op(self):
        self.assertEqual(action_string_to_array(""), [0] * 12)
        self.assertEqual(action_string_to_array(None), [0] * 12)

    def test_tokens_are_trimmed_and_upper_cased(self):
        self.assertEqual(action_string_to_array(" right , b "), action_string_to_array("RIGHT,B"))

    def test_unknown_tokens_are_ignored(self):
        array = action_string_to_array("RIGHT,JUMP,FOO")
        self.assertEqual(array[GENESIS_BUTTONS.index("RIGHT")], 1)
        self.assertEqual(sum(array), 1)


if __name__ == "__main__":
    unittest.main()
