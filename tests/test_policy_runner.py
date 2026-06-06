import os
import tempfile
import time
import unittest

from core.policy_runner import PolicyRunner, PolicyTimeout
from main import load_policy


class FastPolicy:
    def get_action(self, state):
        return "RIGHT"


class RaisingPolicy:
    def get_action(self, state):
        raise ValueError("boom")


class HangingPolicy:
    """Simulates a runaway policy without an unkillable infinite loop.

    Sleeping well past the test's timeout exercises the same code path as a real
    infinite loop, but the daemon worker still wakes up eventually so it cannot
    leak indefinitely during the test run.
    """

    def get_action(self, state):
        time.sleep(5)
        return "RIGHT"


class PolicyRunnerTests(unittest.TestCase):
    def test_returns_policy_action(self):
        runner = PolicyRunner(FastPolicy())
        try:
            self.assertEqual(runner.get_action({}, timeout=1.0), "RIGHT")
        finally:
            runner.close()

    def test_reuses_single_worker_across_many_calls(self):
        runner = PolicyRunner(FastPolicy())
        try:
            for _ in range(100):
                self.assertEqual(runner.get_action({}, timeout=1.0), "RIGHT")
        finally:
            runner.close()

    def test_propagates_policy_exception(self):
        runner = PolicyRunner(RaisingPolicy())
        try:
            with self.assertRaises(ValueError):
                runner.get_action({}, timeout=1.0)
        finally:
            runner.close()

    def test_timeout_raises_and_marks_runner_broken(self):
        runner = PolicyRunner(HangingPolicy())
        start = time.monotonic()
        with self.assertRaises(PolicyTimeout):
            runner.get_action({}, timeout=0.1)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 2.0)
        self.assertTrue(runner.broken)
        with self.assertRaises(PolicyTimeout):
            runner.get_action({}, timeout=0.1)

    def test_loaded_generated_policy_runs_in_killable_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policy.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("def get_action(state):\n    return 'RIGHT'\n")
            runner = PolicyRunner(load_policy(path))
            try:
                self.assertEqual(runner.get_action({}, timeout=3.0), "RIGHT")
                self.assertTrue(runner.process_isolation)
                self.assertTrue(runner.memory_limited)
            finally:
                runner.close()

        self.assertFalse(runner.worker_alive)

    def test_timed_out_generated_policy_process_is_terminated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policy.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "def get_action(state):\n"
                    "    for _ in range(1000000000):\n"
                    "        pass\n"
                    "    return 'RIGHT'\n"
                )
            runner = PolicyRunner(load_policy(path))
            with self.assertRaises(PolicyTimeout):
                runner.get_action({}, timeout=0.2)

            self.assertTrue(runner.process_isolation)
            self.assertFalse(runner.worker_alive)


if __name__ == "__main__":
    unittest.main()
