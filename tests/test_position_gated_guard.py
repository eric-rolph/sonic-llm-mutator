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

    def test_backward_runup_compiles_as_position_phase_machine(self):
        # The live wide-pit escape: back up LEFT, charge RIGHT, jump at the
        # measured edge. Band-anchored TIME replay missed (4917 verified vs
        # 4258 replayed); each travel phase must be gated by POSITION.
        experiment = {
            "zone": 0, "act": 1, "start_x": 3593, "max_x": 4917, "actions": "LEFT",
            "segments": [
                {"actions": "LEFT", "frames": 60, "start_x": 3593, "start_y": 600},
                {"actions": "RIGHT", "frames": 180, "start_x": 3143, "start_y": 600},
                {"actions": "RIGHT,B", "frames": 2, "start_x": 4053, "start_y": 620},
                {"actions": "RIGHT", "frames": 2, "start_x": 4125, "start_y": 545},
            ],
        }
        code = build_diagnosis_guard_candidate(WORKING, experiment)
        self.assertIsNotNone(code)
        self.assertIn("_DIAG_PHASE_0_1_3593", code)
        get_action = load_guard_policy(code)

        def act(x):
            return get_action({"zone": 0, "act": 1, "x_pos": x})

        self.assertEqual(act(3600), "LEFT")     # engaged in band; back up
        self.assertEqual(act(3400), "LEFT")     # still right of back-up point
        self.assertEqual(act(3100), "RIGHT")    # reached it; charge
        self.assertEqual(act(3500), "RIGHT")    # charging through the band again
        self.assertEqual(act(4053), "RIGHT,B")  # measured launch: timed jump
        self.assertEqual(act(4060), "RIGHT,B")  # full held jump
        self.assertEqual(act(4070), "RIGHT")    # timed travel
        self.assertEqual(act(4080), "RIGHT")
        self.assertEqual(act(4272), "RIGHT,DOWN")  # consumed; base policy

    def test_settle_frames_extend_the_final_travel_segment(self):
        # The verified trajectory includes the survival settle (input held,
        # Sonic lived): the guard must replay it too, not hand control back to
        # the base policy 90 frames early.
        experiment = dict(EXPERIMENT)
        experiment["segments"] = [
            {"actions": "RIGHT", "frames": 60, "start_x": 3928, "start_y": 636},
            {"actions": "RIGHT,B", "frames": 45, "start_x": 4053, "start_y": 620},
            {"actions": "RIGHT", "frames": 2, "start_x": 4125, "start_y": 545},
        ]
        experiment["settle_frames"] = 3
        code = build_diagnosis_guard_candidate(WORKING, experiment)
        get_action = load_guard_policy(code)

        def act(x):
            return get_action({"zone": 0, "act": 1, "x_pos": x})

        act(3940)  # engage; run-up (positional)
        jump = [act(4053 + i) for i in range(45)]
        self.assertTrue(all(a == "RIGHT,B" for a in jump))
        # Final travel: 2 scripted + 3 settle = 5 frames before hand-back.
        travel = [act(4150 + i) for i in range(5)]
        self.assertTrue(all(a == "RIGHT" for a in travel), travel)
        self.assertEqual(act(4900), "RIGHT,DOWN")  # consumed

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
