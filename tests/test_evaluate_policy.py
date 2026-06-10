import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from core.evaluator import LEVEL_CLEARED_BONUS
from main import evaluate_policy


class StaticPolicy:
    def __init__(self):
        self.calls = 0

    def get_action(self, state):
        self.calls += 1
        return "RIGHT"


class NoVisionMutator:
    def analyze_environment(self, screenshot_path):
        return "UNKNOWN"


class FakeEnv:
    def __init__(self, states, done_after=None):
        self.states = list(states)
        self.done_after = done_after
        self.index = 0
        self.step_count = 0

    def reset(self):
        self.index = 0
        self.step_count = 0
        return None

    def get_state(self):
        if self.index >= len(self.states):
            return dict(self.states[-1])
        state = dict(self.states[self.index])
        self.index += 1
        return state

    def step(self, action):
        self.step_count += 1
        done = self.done_after is not None and self.step_count >= self.done_after
        return None, 0, done, {}

    def get_screenshot(self):
        return "fake_screenshot.png"


class MutatingPolicy:
    def get_action(self, state):
        state["x_pos"] = 1_000_000_000
        state["score"] = 1_000_000_000
        return "RIGHT"


class RaisingPolicy:
    def get_action(self, state):
        raise RuntimeError("broken candidate")


class NonStringPolicy:
    def get_action(self, state):
        return ["RIGHT"]


class EvaluatePolicyTests(unittest.TestCase):
    def evaluate_silently(self, env, max_frames, policy=None, **kwargs):
        return evaluate_policy(
            env,
            policy or StaticPolicy(),
            NoVisionMutator(),
            max_frames=max_frames,
            verbose=False,
            **kwargs,
        )

    def test_preserves_stuck_failure_reason(self):
        states = [{"x_pos": 10, "y_pos": 100, "rings": 0, "score": 0}] * 700
        env = FakeEnv(states)

        _, frames, max_x, reason, _, _, _ = self.evaluate_silently(env, max_frames=700)

        self.assertEqual(max_x, 10)
        self.assertGreater(frames, 500)
        self.assertIn("stopped making forward progress", reason)

    def test_reports_timeout_when_max_frames_reached(self):
        states = [
            {"x_pos": 1, "y_pos": 100, "rings": 0, "score": 0},
            {"x_pos": 2, "y_pos": 100, "rings": 0, "score": 0},
            {"x_pos": 3, "y_pos": 100, "rings": 0, "score": 0},
        ]
        env = FakeEnv(states)

        _, frames, _, reason, _, _, _ = self.evaluate_silently(env, max_frames=3)

        self.assertEqual(frames, 3)
        self.assertEqual(reason, "Timeout reached.")

    def test_reports_fatal_when_environment_ends_without_specific_reason(self):
        states = [{"x_pos": 50, "y_pos": 100, "rings": 0, "score": 0}] * 5
        env = FakeEnv(states, done_after=2)

        _, frames, _, reason, _, _, _ = self.evaluate_silently(env, max_frames=100)

        self.assertEqual(frames, 2)
        self.assertEqual(reason, "Sonic lost a life or hit a fatal obstacle.")

    def test_verbose_false_suppresses_evaluator_messages(self):
        states = [{"x_pos": 10, "y_pos": 100, "rings": 0, "score": 0}] * 700
        env = FakeEnv(states)
        output = StringIO()

        with redirect_stdout(output):
            evaluate_policy(
                env,
                StaticPolicy(),
                NoVisionMutator(),
                max_frames=700,
                verbose=False,
            )

        self.assertEqual(output.getvalue(), "")

    def test_trace_entries_include_motion_action_and_vision_context(self):
        states = [
            {
                "x_pos": index,
                "y_pos": 100 + index,
                "x_velocity": 1,
                "y_velocity": 1,
                "rings": 3,
                "score": 10,
                "vision_context": "CLEAR",
            }
            for index in range(40)
        ]
        env = FakeEnv(states)

        _, _, _, _, _, trace, _ = self.evaluate_silently(env, max_frames=31)

        self.assertGreaterEqual(len(trace), 2)
        self.assertIsInstance(trace[-1], dict)
        self.assertEqual(trace[-1]["x"], 30)
        self.assertEqual(trace[-1]["y"], 130)
        self.assertEqual(trace[-1]["action"], "RIGHT")
        self.assertEqual(trace[-1]["vision_context"], "UNKNOWN")
        self.assertIn("x_velocity", trace[-1])
        self.assertIn("frame", trace[-1])

    def test_action_repeat_reuses_policy_action_for_multiple_frames(self):
        states = [
            {"x_pos": index, "y_pos": 100, "rings": 0, "score": 0}
            for index in range(20)
        ]
        env = FakeEnv(states)
        policy = StaticPolicy()

        _, frames, _, reason, _, _, _ = self.evaluate_silently(
            env,
            max_frames=5,
            policy=policy,
            action_repeat=3,
        )

        self.assertEqual(frames, 5)
        self.assertEqual(policy.calls, 2)
        self.assertEqual(reason, "Timeout reached.")

    def test_trace_cadence_not_skipped_by_misaligned_action_repeat(self):
        # action_repeat=7 does not divide the 30-frame trace interval, so the
        # old `frames_alive % 30 == 0` check recorded only the frame-0 entry.
        # The elapsed-frame cadence must keep sampling throughout the run.
        states = [
            {"x_pos": index * 100, "y_pos": 100, "rings": 0, "score": 0}
            for index in range(60)
        ]
        env = FakeEnv(states)

        _, frames, _, _, _, trace, _ = self.evaluate_silently(
            env,
            max_frames=210,
            action_repeat=7,
        )

        self.assertEqual(frames, 210)
        self.assertGreaterEqual(len(trace), 3)
        recorded_frames = [entry["frame"] for entry in trace]
        self.assertEqual(recorded_frames, sorted(recorded_frames))

    def test_context_screenshots_are_bounded_to_a_small_ring(self):
        class RecordingShotEnv(FakeEnv):
            def __init__(self, states):
                super().__init__(states)
                self.context_shots = []

            def get_screenshot(self, filepath=None):
                if filepath is not None:
                    self.context_shots.append(filepath)
                    return filepath
                return "final.png"

        # Long enough for several 300-frame vision polls.
        states = [{"x_pos": i * 10, "y_pos": 100, "rings": 0, "score": 0} for i in range(1100)]
        env = RecordingShotEnv(states)

        self.evaluate_silently(env, max_frames=1000)

        self.assertGreater(len(env.context_shots), 3)  # polled several times...
        self.assertLessEqual(len(set(env.context_shots)), 3)  # ...but onto <=3 files
        for path in env.context_shots:
            self.assertRegex(path, r"context_slot[012]\.png$")

    def test_snapshot_sink_receives_every_authoritative_state(self):
        class RecordingSink:
            def __init__(self):
                self.calls = []

            def record(self, env, frame, state):
                self.calls.append((frame, dict(state)))

        states = [
            {"x_pos": i * 10, "y_pos": 100, "rings": 0, "score": 0}
            for i in range(20)
        ]
        env = FakeEnv(states)
        sink = RecordingSink()

        evaluate_policy(
            env, StaticPolicy(), NoVisionMutator(), max_frames=10, verbose=False, snapshot_sink=sink
        )

        # Initial state plus one per stepped frame, frames monotonic; cadence
        # is the sink's own concern (FailureSnapshotRing tests cover it).
        self.assertEqual(len(sink.calls), 11)
        frames = [frame for frame, _ in sink.calls]
        self.assertEqual(frames, sorted(frames))
        self.assertEqual(sink.calls[0][0], 0)
        self.assertEqual(sink.calls[-1][0], 10)

    def test_cached_vision_context_applies_synchronously_without_api_call(self):
        class CachedVisionMutator:
            def __init__(self):
                self.analyze_calls = 0

            def cached_vision_context(self, location_key):
                return "SPIKES"

            def analyze_environment(self, screenshot_path):
                self.analyze_calls += 1
                return "CLEAR"

        states = [
            {"x_pos": i, "y_pos": 100, "zone": 0, "act": 0, "screen_x": i, "rings": 0, "score": 0}
            for i in range(400)
        ]
        env = FakeEnv(states)
        mutator = CachedVisionMutator()

        _, _, _, _, _, trace, _ = evaluate_policy(
            env, StaticPolicy(), mutator, max_frames=350, verbose=False
        )

        # The cache answered every poll: no API/thread round trip, and the
        # policy saw the cached label in its state by the next trace entry.
        self.assertEqual(mutator.analyze_calls, 0)
        self.assertEqual(trace[-1]["vision_context"], "SPIKES")

    def test_vision_poll_stores_result_under_location_key(self):
        class StoringVisionMutator:
            def __init__(self):
                self.stored = []

            def cached_vision_context(self, location_key):
                return None

            def store_vision_context(self, location_key, label):
                self.stored.append((location_key, label))

            def analyze_environment(self, screenshot_path):
                return "ENEMY"

        states = [
            {"x_pos": i, "y_pos": 100, "zone": 0, "act": 1, "screen_x": i, "rings": 0, "score": 0}
            for i in range(400)
        ]
        env = FakeEnv(states)
        mutator = StoringVisionMutator()

        evaluate_policy(env, StaticPolicy(), mutator, max_frames=350, verbose=False)

        self.assertGreaterEqual(len(mutator.stored), 1)
        location_key, label = mutator.stored[0]
        self.assertRegex(location_key, r"^zone-0-act-1-sx-\d+$")
        self.assertEqual(label, "ENEMY")

    def test_proactive_vision_disabled_by_env_var(self):
        class CountingVisionMutator:
            def __init__(self):
                self.calls = 0

            def analyze_environment(self, screenshot_path):
                self.calls += 1
                return "CLEAR"

        states = [{"x_pos": i, "y_pos": 100, "rings": 0, "score": 0} for i in range(400)]
        env = FakeEnv(states)
        mutator = CountingVisionMutator()

        with mock.patch.dict(os.environ, {"SONIC_PROACTIVE_VISION": "0"}):
            evaluate_policy(env, StaticPolicy(), mutator, max_frames=350, verbose=False)

        self.assertEqual(mutator.calls, 0)  # no proactive polling when disabled

    def test_level_transition_counts_clear_and_resets_progress(self):
        # Act 0: x climbs to 1000. Then (zone, act) flips and x resets -- the old
        # global-max stuck detector would have killed the run here.
        act0 = [{"x_pos": (i + 1) * 100, "zone": 0, "act": 0, "rings": 0, "score": 0} for i in range(10)]
        act1 = [{"x_pos": (i + 1) * 50, "zone": 0, "act": 1, "rings": 0, "score": 0} for i in range(10)]
        env = FakeEnv(act0 + act1)

        _, frames, max_x, reason, _, _, components = self.evaluate_silently(env, max_frames=20)

        self.assertEqual(frames, 20)
        self.assertEqual(reason, "Timeout reached.")  # not "stopped making forward progress"
        self.assertEqual(components["levels_cleared"], 1)
        self.assertEqual(components["levels"], LEVEL_CLEARED_BONUS)
        self.assertEqual(max_x, 500)  # furthest in the *current* act, not the 1000 from act 0

    def test_policy_cannot_mutate_authoritative_state_used_for_fitness(self):
        states = [{"x_pos": 10, "y_pos": 100, "rings": 0, "score": 0}] * 3
        env = FakeEnv(states)

        fitness, _, max_x, _, _, _, components = self.evaluate_silently(
            env,
            max_frames=1,
            policy=MutatingPolicy(),
        )

        self.assertEqual(max_x, 10)
        self.assertEqual(components["score"], 0)
        self.assertLess(fitness, 1_000_000)

    def test_scores_authoritative_post_step_state(self):
        class PostStepStateEnv(FakeEnv):
            def step(self, action):
                self.step_count += 1
                self.index = 1
                return None, 0, True, {}

        env = PostStepStateEnv(
            [
                {"x_pos": 0, "y_pos": 100, "rings": 0, "score": 0},
                {"x_pos": 100, "y_pos": 100, "rings": 2, "score": 50},
            ],
            done_after=1,
        )

        _, _, max_x, _, _, _, components = self.evaluate_silently(env, max_frames=10)

        self.assertEqual(max_x, 100)
        self.assertEqual(components["rings"], 2)
        self.assertGreater(components["score"], 0)

    def test_policy_exception_invalidates_candidate_without_stepping(self):
        env = FakeEnv([{"x_pos": 10, "y_pos": 100, "rings": 0, "score": 0}] * 3)

        fitness, frames, max_x, reason, _, _, components = self.evaluate_silently(
            env,
            max_frames=3,
            policy=RaisingPolicy(),
        )

        self.assertEqual(fitness, 0.0)
        self.assertEqual(frames, 0)
        self.assertEqual(max_x, 0)
        self.assertEqual(env.step_count, 0)
        self.assertIn("exception", reason.lower())
        self.assertIn("runtime_error", components)

    def test_non_string_action_invalidates_candidate_without_stepping(self):
        env = FakeEnv([{"x_pos": 10, "y_pos": 100, "rings": 0, "score": 0}] * 3)

        fitness, frames, max_x, reason, _, _, components = self.evaluate_silently(
            env,
            max_frames=3,
            policy=NonStringPolicy(),
        )

        self.assertEqual(fitness, 0.0)
        self.assertEqual(frames, 0)
        self.assertEqual(max_x, 0)
        self.assertEqual(env.step_count, 0)
        self.assertIn("non-string", reason.lower())
        self.assertIn("runtime_error", components)

    def test_policy_runner_startup_failure_invalidates_candidate(self):
        env = FakeEnv([{"x_pos": 10, "y_pos": 100, "rings": 0, "score": 0}] * 3)

        with mock.patch("core.evaluation.PolicyRunner", side_effect=RuntimeError("spawn failed")):
            fitness, frames, max_x, reason, _, _, components = self.evaluate_silently(
                env,
                max_frames=3,
            )

        self.assertEqual((fitness, frames, max_x), (0.0, 0, 0))
        self.assertIn("failed to start", reason.lower())
        self.assertIn("runtime_error", components)


if __name__ == "__main__":
    unittest.main()
