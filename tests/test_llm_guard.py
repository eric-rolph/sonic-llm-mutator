import ast
import unittest

from core.frontier import build_llm_guard_candidate, llm_guard_marker

WORKING = "def get_action(state):\n    return 'RIGHT,DOWN'\n"


class BuildLlmGuardTests(unittest.TestCase):
    def test_frame_replay_guard_preserves_working_code(self):
        proposal = {"zone": 0, "act": 1, "x": 4268, "action": "RIGHT,B", "hold_frames": 20}
        candidate = build_llm_guard_candidate(WORKING, proposal)

        self.assertIsNotNone(candidate)
        self.assertIn("# LLM_GUARD zone=0 act=1 x=4268", candidate)
        self.assertIn('return "RIGHT,B"', candidate)
        self.assertIn("return 'RIGHT,DOWN'", candidate)  # original preserved
        self.assertIn("_LLM_REPLAY_0_1_4268 < 20", candidate)  # holds for hold_frames
        ast.parse(candidate)  # compiles

    def test_threshold_guard_when_no_hold_frames(self):
        proposal = {"zone": 0, "act": 1, "x": 100, "action": "RIGHT"}
        candidate = build_llm_guard_candidate("def get_action(state):\n    return 'RIGHT'\n", proposal)

        self.assertIn("# LLM_GUARD zone=0 act=1 x=100", candidate)
        self.assertIn('75 <= state.get("x_pos", 0) <= 125', candidate)  # +/- x_radius
        ast.parse(candidate)

    def test_coordinates_come_from_proposal_actions_sanitised(self):
        # Only valid Genesis buttons survive; junk tokens are dropped.
        proposal = {"zone": 0, "act": 1, "x": 100, "action": "RIGHT, JUMP, B"}
        candidate = build_llm_guard_candidate("def get_action(state):\n    return 'RIGHT'\n", proposal)
        self.assertIn('return "RIGHT,B"', candidate)

    def test_rejects_malformed_or_invalid(self):
        working = "def get_action(state):\n    return 'RIGHT'\n"
        self.assertIsNone(build_llm_guard_candidate(working, None))
        self.assertIsNone(build_llm_guard_candidate(working, {"zone": 0, "act": 1}))  # no x
        self.assertIsNone(  # no valid buttons
            build_llm_guard_candidate(working, {"zone": 0, "act": 1, "x": 100, "action": "JUMP"})
        )

    def test_dedups_overlapping_guard(self):
        existing = (
            "def get_action(state):\n"
            "    # LLM_GUARD zone=0 act=1 x=100\n"
            '    if 75 <= state.get("x_pos", 0) <= 125:\n'
            '        return "RIGHT,B"\n'
            "    return 'RIGHT'\n"
        )
        # A near-identical spot must not stack a second overlapping guard.
        self.assertIsNone(
            build_llm_guard_candidate(existing, {"zone": 0, "act": 1, "x": 110, "action": "RIGHT,B"})
        )

    def test_marker_extraction(self):
        candidate = build_llm_guard_candidate(WORKING, {"zone": 2, "act": 0, "x": 55, "action": "RIGHT"})
        self.assertEqual(llm_guard_marker(candidate), "# LLM_GUARD zone=2 act=0 x=55")
        self.assertIsNone(llm_guard_marker("def get_action(state):\n    return 'RIGHT'\n"))


if __name__ == "__main__":
    unittest.main()
