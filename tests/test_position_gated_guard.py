"""Behavioral test of the position-gated sequence guard.

Live-observed fidelity gap: a verified escape (run to the pit edge at x=4053,
jump, travel -> x=4272) compiled into a band-entry-anchored TIMED replay peaked
at x=4263 and died, because entering the +/-25 band early shifts the whole
timed run-up. The fix gates the run-up by position and anchors the timed jump
exactly at the measured launch x.
"""

import unittest

from core.frontier import build_diagnosis_guard_candidate

WORKING = "def get_action(state):\n    return 'RIGHT,DOWN'\n"

EXPERIMENT = {
    "zone": 0, "act": 1, "start_x": 3928, "max_x": 4272,
    "actions": "RIGHT", "hold_frames": 225,
    "segments": [
        {"actions": "RIGHT", "frames": 60, "start_x": 3928, "start_y": 636},
        {"actions": "RIGHT,B", "frames": 45, "start_x": 4053, "start_y": 620},
        {"actions": "RIGHT", "frames": 120, "start_x": 4125, "start_y": 545},
    ],
}


def load_guard_policy(code):
    namespace = {}
    exec(code, namespace)  # noqa: S102 - deterministic generated code under test
    return namespace["get_action"]


class PositionGatedGuardTests(unittest.TestCase):
    def test_run_up_is_position_gated_and_jump_time_anchored(self):
        code = build_diagnosis_guard_candidate(WORKING, EXPERIMENT)
        self.assertIsNotNone(code)
        get_action = load_guard_policy(code)

        def state(x):
            return {"zone": 0, "act": 1, "x_pos": x}

        # Entering the band EARLY (x=3905, the live failure mode): the guard
        # returns the run-up action by POSITION, no timer consumed.
        for x in (3905, 3960, 4000, 4052):
            self.assertEqual(get_action(state(x)), "RIGHT")

        # Crossing the measured launch x anchors the timed replay: the jump is
        # held for its full measured 45 frames regardless of x.
        actions = [get_action(state(4053 + i * 3)) for i in range(45)]
        self.assertTrue(all(a == "RIGHT,B" for a in actions), actions[:5])

        # Then the travel segment plays out, then control returns to the policy.
        travel = [get_action(state(4150 + i)) for i in range(120)]
        self.assertTrue(all(a == "RIGHT" for a in travel))
        self.assertEqual(get_action(state(4272)), "RIGHT,DOWN")  # guard consumed

    def test_wrong_act_or_far_x_leaves_policy_untouched(self):
        code = build_diagnosis_guard_candidate(WORKING, EXPERIMENT)
        get_action = load_guard_policy(code)
        self.assertEqual(get_action({"zone": 0, "act": 0, "x_pos": 4000}), "RIGHT,DOWN")
        self.assertEqual(get_action({"zone": 0, "act": 1, "x_pos": 100}), "RIGHT,DOWN")

    def test_jump_first_sequences_keep_pure_time_replay(self):
        # A sequence that STARTS with B cannot be position-gated (the press
        # timing is the whole point); it must keep the band-anchored replay.
        experiment = dict(EXPERIMENT)
        experiment["segments"] = [
            {"actions": "RIGHT,B", "frames": 30, "start_x": 3928},
            {"actions": "RIGHT", "frames": 120, "start_x": 3990},
        ]
        code = build_diagnosis_guard_candidate(WORKING, experiment)
        self.assertIsNotNone(code)
        get_action = load_guard_policy(code)
        # First frame in the band starts the replay immediately with the jump.
        self.assertEqual(get_action({"zone": 0, "act": 1, "x_pos": 3910}), "RIGHT,B")

    def test_missing_segment_start_x_falls_back_to_time_replay(self):
        experiment = dict(EXPERIMENT)
        experiment["segments"] = [
            {"actions": "RIGHT", "frames": 60},
            {"actions": "RIGHT,B", "frames": 45},
        ]
        code = build_diagnosis_guard_candidate(WORKING, experiment)
        self.assertIsNotNone(code)
        get_action = load_guard_policy(code)
        self.assertEqual(get_action({"zone": 0, "act": 1, "x_pos": 3928}), "RIGHT")


if __name__ == "__main__":
    unittest.main()
