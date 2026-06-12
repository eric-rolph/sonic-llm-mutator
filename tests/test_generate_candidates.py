import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from main import (
    build_diagnosis_guard_candidate,
    build_frontier_guard_candidate,
    generate_candidates,
    load_pool_codes,
)


class FakeMutator:
    def __init__(self):
        self.mutate_calls = 0
        self.crossover_calls = 0
        self.parent_pairs = []
        self.diagnosis_reports = []

    def mutate_policy(self, code, reason, screenshot, history, temperature, trace, diagnosis_report=None):
        self.mutate_calls += 1
        self.diagnosis_reports.append(diagnosis_report)
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

    def test_build_frontier_guard_candidate_does_not_shadow_overlapping_guard(self):
        working = """def get_action(state):
    # FRONTIER_GUARD zone=0 act=1 x=1077
    if 1052 <= state.get("x_pos", 0) <= 1102:
        return "RIGHT,B"
    return "RIGHT,DOWN"
"""
        trace = [
            {"zone": 0, "act": 1, "x": 1078, "x_velocity": 0.0, "action": "RIGHT,B"},
            {"zone": 0, "act": 1, "x": 1078, "x_velocity": 0.0, "action": "RIGHT,B"},
            {"zone": 0, "act": 1, "x": 1078, "x_velocity": 0.0, "action": "RIGHT,B"},
        ]

        self.assertIsNone(build_frontier_guard_candidate(working, trace))

    def test_recent_failed_frontier_guard_is_not_retried_identically(self):
        mutator = FakeMutator()
        working = "def get_action(state):\n    return 'RIGHT,DOWN'\n"
        trace = [
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
        ]
        recent_history = [
            {
                "llm_reasoning": "Deterministic frontier guard",
                "components": {
                    "frontier_guard_markers": ["# FRONTIER_GUARD zone=0 act=1 x=1077"]
                },
            }
        ]

        result = generate_silently(
            mutator,
            working,
            "reason",
            None,
            recent_history,
            trace,
            1,
            [],
            crossover_probability=0.0,
        )

        self.assertEqual(result, [("# mutation t=0.7", "mutated")])
        self.assertEqual(mutator.mutate_calls, 1)

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

    def test_build_diagnosis_guard_compiles_verified_experiment_into_code(self):
        working = "def get_action(state):\n    return 'RIGHT,DOWN'\n"
        experiment = {
            "zone": 0, "act": 1, "start_x": 2404, "max_x": 2520,
            "actions": "RIGHT,B", "hold_frames": 120,
        }

        candidate = build_diagnosis_guard_candidate(working, experiment)

        self.assertIn("# DIAGNOSIS_GUARD zone=0 act=1 x=2404", candidate)
        self.assertIn('return "RIGHT,B"', candidate)
        self.assertIn("2379 <= state.get(\"x_pos\", 0) < 2520", candidate)
        self.assertIn("return 'RIGHT,DOWN'", candidate)  # original preserved

    def test_build_diagnosis_guard_rejects_malformed_and_non_improving(self):
        working = "def get_action(state):\n    return 'RIGHT'\n"
        self.assertIsNone(build_diagnosis_guard_candidate(working, None))
        self.assertIsNone(build_diagnosis_guard_candidate(working, {"zone": 0}))
        self.assertIsNone(
            build_diagnosis_guard_candidate(
                working,
                {"zone": 0, "act": 1, "start_x": 100, "max_x": 100, "actions": "RIGHT"},
            )
        )
        self.assertIsNone(
            build_diagnosis_guard_candidate(
                working,
                {"zone": 0, "act": 1, "start_x": 100, "max_x": 200, "actions": "; import os"},
            )
        )

    def test_new_verified_escape_replaces_overlapping_guard(self):
        # Live-observed freeze: the promoted champion carried an x-threshold
        # guard at x=2404, and its own better frame-replay replacement at
        # x=2393 was blocked by the overlap check. Newer verified escapes at
        # the same spot must supersede the old guard, not be blocked by it.
        working = """def get_action(state):
    # DIAGNOSIS_GUARD zone=0 act=1 x=2400
    if (
        state.get("zone") == 0
        and state.get("act") == 1
        and 2375 <= state.get("x_pos", 0) < 2500
    ):
        return "RIGHT,B"

    return "RIGHT"
"""
        experiment = {
            "zone": 0, "act": 1, "start_x": 2410, "max_x": 2600, "actions": "DOWN,B",
        }

        candidate = build_diagnosis_guard_candidate(working, experiment)

        self.assertIsNotNone(candidate)
        self.assertIn("# DIAGNOSIS_GUARD zone=0 act=1 x=2410", candidate)
        self.assertNotIn("# DIAGNOSIS_GUARD zone=0 act=1 x=2400", candidate)
        self.assertNotIn("2375 <= state.get", candidate)  # old body gone
        self.assertIn('return "DOWN,B"', candidate)
        self.assertIn('return "RIGHT"', candidate)  # original policy intact

    def test_mangled_old_guard_blocks_replacement_conservatively(self):
        # If a mutation rewrote the old guard block (no trailing blank line in
        # our generated shape), stripping is unsafe -- keep the old behavior.
        working = (
            "def get_action(state):\n"
            "    # DIAGNOSIS_GUARD zone=0 act=1 x=2400\n"
            "    return 'RIGHT'\n"  # no blank line: not our generated shape
        )
        experiment = {
            "zone": 0, "act": 1, "start_x": 2410, "max_x": 2600, "actions": "DOWN,B",
        }
        self.assertIsNone(build_diagnosis_guard_candidate(working, experiment))

    def test_build_diagnosis_guard_compiles_sequence_as_frame_replay(self):
        working = "def get_action(state):\n    return 'RIGHT'\n"
        experiment = {
            "zone": 0, "act": 1, "start_x": 2350, "max_x": 2600, "actions": "RIGHT",
            "segments": [
                {"actions": "RIGHT", "frames": 90, "start_x": 2350},
                {"actions": "RIGHT,B", "frames": 40, "start_x": 2460},
                {"actions": "RIGHT", "frames": 60, "start_x": 2530},
            ],
        }

        candidate = build_diagnosis_guard_candidate(working, experiment)

        self.assertIn("# DIAGNOSIS_GUARD zone=0 act=1 x=2350", candidate)
        # Replay anchors on the first crossing of the start x...
        self.assertIn("2325 <= state.get(\"x_pos\", 0) <= 2375", candidate)
        # ...then plays the measured frame counts: jump segment holds B for
        # its full 40 frames (x-thresholds released it after a few frames,
        # turning a verified full jump into a short hop).
        self.assertIn("global _DIAG_REPLAY_0_1_2350", candidate)
        self.assertIn("if _DIAG_REPLAY_0_1_2350 < 90:", candidate)
        self.assertIn("if _DIAG_REPLAY_0_1_2350 < 130:", candidate)
        self.assertIn('return "RIGHT,B"', candidate)
        self.assertIn("_DIAG_REPLAY_0_1_2350 < 190", candidate)  # total budget
        # The compiled candidate must still load in the restricted runtime.
        from core.policy_validator import validate_policy_source
        validate_policy_source(candidate)

    def test_compiled_sequence_guard_replays_through_the_real_loader(self):
        # Golden round-trip: the frame-replay guard's correctness depends on the
        # runtime exec'ing the module once and persisting its global counter
        # (core.policy_loader / PolicyRunner). Load the generated source through
        # the REAL loader and drive get_action frame-by-frame, asserting each
        # segment fires for its measured count -- this fails loudly if the
        # codegen or the runtime's exec/global model ever drift apart.
        from core.policy_loader import load_policy

        working = "def get_action(state):\n    return 'RIGHT'\n"
        experiment = {
            "zone": 0, "act": 1, "start_x": 2400, "max_x": 2600, "actions": "RIGHT",
            "segments": [
                {"actions": "RIGHT", "frames": 3, "start_x": 2400},
                {"actions": "RIGHT,B", "frames": 2, "start_x": 2430},
                {"actions": "RIGHT", "frames": 2, "start_x": 2470},
            ],
        }
        candidate = build_diagnosis_guard_candidate(working, experiment)
        self.assertIsNotNone(candidate)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guard_policy.py"
            path.write_text(candidate, encoding="utf-8")
            policy = load_policy(str(path))

            # Inside the guard's x-band and act, the replay drives the segments
            # by frame count regardless of the exact x each frame.
            state = {"zone": 0, "act": 1, "x_pos": 2400}
            actions = [policy.get_action(dict(state)) for _ in range(7)]

        # 3x RIGHT, then 2x RIGHT,B (the full held jump), then 2x RIGHT, then
        # the counter is spent and control returns to the base policy (RIGHT).
        self.assertEqual(actions[:3], ["RIGHT", "RIGHT", "RIGHT"])
        self.assertEqual(actions[3:5], ["RIGHT,B", "RIGHT,B"])
        self.assertEqual(actions[5:7], ["RIGHT", "RIGHT"])

    def test_sequence_guard_compiles_backward_runups_too(self):
        # Frame replay does not need x-monotonic boundaries, so back-up-then-
        # charge sequences (the longer-runway strategy) compile as well.
        working = "def get_action(state):\n    return 'RIGHT'\n"
        experiment = {
            "zone": 0, "act": 1, "start_x": 2400, "max_x": 2600, "actions": "LEFT",
            "segments": [
                {"actions": "LEFT", "frames": 30, "start_x": 2400},
                {"actions": "RIGHT,B", "frames": 60, "start_x": 2350},
            ],
        }

        candidate = build_diagnosis_guard_candidate(working, experiment)

        self.assertIsNotNone(candidate)
        self.assertIn('return "LEFT"', candidate)
        self.assertIn('return "RIGHT,B"', candidate)

    def test_verified_experiment_takes_the_deterministic_slot(self):
        mutator = FakeMutator()
        experiments = [
            {"zone": 0, "act": 1, "start_x": 2404, "max_x": 2520, "actions": "RIGHT,B"},
            {"zone": 0, "act": 1, "start_x": 2300, "max_x": 2600, "actions": "DOWN,B"},
        ]
        # A stationary trace that would normally produce a frontier guard --
        # the verified experiment must win the slot.
        trace = [
            {"zone": 0, "act": 1, "x": 1077, "x_velocity": 0.0, "action": "RIGHT,DOWN"},
        ] * 3

        result = generate_silently(
            mutator,
            "def get_action(state):\n    return 'RIGHT,DOWN'\n",
            "stuck",
            None,
            [],
            trace,
            2,
            [],
            crossover_probability=0.0,
            verified_experiments=experiments,
        )

        code, reasoning = result[0]
        self.assertEqual(reasoning, "Diagnosed guard (verified input)")
        # The furthest-reaching experiment (max_x 2600) wins.
        self.assertIn("# DIAGNOSIS_GUARD zone=0 act=1 x=2300", code)
        self.assertIn('return "DOWN,B"', code)
        self.assertEqual(mutator.mutate_calls, 1)  # only one LLM slot left

    def test_recently_attempted_diagnosis_guard_is_not_retried(self):
        mutator = FakeMutator()
        experiments = [
            {"zone": 0, "act": 1, "start_x": 2404, "max_x": 2520, "actions": "RIGHT,B"},
        ]
        recent = [
            {"components": {"frontier_guard_markers": ["# DIAGNOSIS_GUARD zone=0 act=1 x=2404"]}}
        ]

        result = generate_silently(
            mutator,
            "def get_action(state):\n    return 'RIGHT'\n",
            "stuck",
            None,
            recent,
            [],
            2,
            [],
            crossover_probability=0.0,
            verified_experiments=experiments,
        )

        reasons = [reasoning for _code, reasoning in result]
        self.assertNotIn("Diagnosed guard (verified input)", reasons)
        self.assertEqual(mutator.mutate_calls, 2)

    def test_diagnosis_report_is_forwarded_to_mutations(self):
        mutator = FakeMutator()

        generate_silently(
            mutator,
            "def get_action(state):\n    return 'RIGHT'\n",
            "stuck",
            None,
            [],
            [],
            2,
            [],
            crossover_probability=0.0,
            diagnosis_report="The wall at x=3061 needs a full jump; RIGHT,B for 40 frames verified.",
        )

        self.assertEqual(mutator.mutate_calls, 2)
        self.assertEqual(
            mutator.diagnosis_reports,
            ["The wall at x=3061 needs a full jump; RIGHT,B for 40 frames verified."] * 2,
        )

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
