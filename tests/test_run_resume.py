import os
import tempfile
import unittest

from main import evaluate_working_baseline


class NoVisionMutator:
    def analyze_environment(self, screenshot_path):
        return "UNKNOWN"


class FakeEnv:
    def __init__(self, states):
        self.states = list(states)
        self.index = 0

    def reset(self):
        self.index = 0
        return None

    def get_state(self):
        if self.index >= len(self.states):
            return dict(self.states[-1])
        state = dict(self.states[self.index])
        self.index += 1
        return state

    def step(self, action):
        return None, 0, False, {}

    def get_screenshot(self):
        return "baseline_screenshot.png"


class RunResumeTests(unittest.TestCase):
    def write_policy(self, directory, code):
        policy_path = os.path.join(directory, "policy.py")
        with open(policy_path, "w", encoding="utf-8") as f:
            f.write(code)
        return policy_path

    def test_evaluate_working_baseline_seeds_resume_context(self):
        states = [{"x_pos": 100, "y_pos": 100, "rings": 0, "score": 0}] * 700
        env = FakeEnv(states)
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = self.write_policy(
                tmp,
                "def get_action(state):\n    return 'RIGHT'\n",
            )

            context = evaluate_working_baseline(
                env,
                policy_path,
                NoVisionMutator(),
                max_frames=700,
                verbose=False,
            )

        self.assertGreater(context["working_fitness"], -1)
        self.assertEqual(context["last_screenshot"], "baseline_screenshot.png")
        self.assertIn("stopped making forward progress", context["last_failure_reason"])
        self.assertEqual(context["last_trace"][-1], (100, 100))

    def test_evaluate_working_baseline_handles_missing_policy(self):
        context = evaluate_working_baseline(
            env=None,
            working_path="missing.py",
            mutator=NoVisionMutator(),
            max_frames=700,
            verbose=False,
        )

        self.assertEqual(context["working_fitness"], -1.0)
        self.assertEqual(context["last_failure_reason"], "Initial seed run")
        self.assertIsNone(context["last_screenshot"])
        self.assertEqual(context["last_trace"], [])


if __name__ == "__main__":
    unittest.main()
