import time
import unittest

from core.policy_runner import PolicyRunner, PolicyTimeout


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

        # Should give up promptly, not wait for the 5s sleep.
        self.assertLess(elapsed, 2.0)
        self.assertTrue(runner.broken)

        # A broken runner refuses further work instead of hanging again.
        with self.assertRaises(PolicyTimeout):
            runner.get_action({}, timeout=0.1)


if __name__ == "__main__":
    unittest.main()
