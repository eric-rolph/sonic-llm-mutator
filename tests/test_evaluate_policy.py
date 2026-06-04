import unittest
from contextlib import redirect_stdout
from io import StringIO

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


if __name__ == "__main__":
    unittest.main()
