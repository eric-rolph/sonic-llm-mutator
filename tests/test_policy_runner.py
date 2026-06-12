import os
import tempfile
import time
import unittest

from core.policy_runner import PolicyRunner, PolicyTimeout
from main import load_policy


class SkillsSandboxTests(unittest.TestCase):
    """A policy's imported skills must run under the SAME restricted builtins
    as the policy -- not the real builtins module (the historical asymmetry)."""

    def _run_with_skills(self, tmp, skills_src, policy_src):
        os.makedirs(os.path.join(tmp, "policies"), exist_ok=True)
        with open(os.path.join(tmp, "policies", "skills.py"), "w", encoding="utf-8") as f:
            f.write(skills_src)
        policy_path = os.path.join(tmp, "policy.py")
        with open(policy_path, "w", encoding="utf-8") as f:
            f.write(policy_src)
        previous = os.getcwd()
        os.chdir(tmp)  # load_policy resolves policies/skills.py relative to cwd
        try:
            policy = load_policy(policy_path)
            return policy.get_action({"x_pos": 0})
        finally:
            os.chdir(previous)

    def test_imported_skill_runs_and_is_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            action = self._run_with_skills(
                tmp,
                "def boost(state):\n    return 'RIGHT,B'\n",
                "import policies.skills as skills\n"
                "def get_action(state):\n    return skills.boost(state)\n",
            )
        self.assertEqual(action, "RIGHT,B")

    def test_skills_module_executes_under_restricted_builtins(self):
        # White-box: the validator blocks *naming* open/eval in skill source,
        # but the deeper guarantee is that the skills module's __builtins__ is
        # the restricted set (not the real builtins module) -- so even a
        # validator gap can't reach open/eval/__import__ at runtime.
        from core.policy_loader import SAFE_POLICY_BUILTINS, _restricted_skills_module

        with tempfile.TemporaryDirectory() as tmp:
            skills_path = os.path.join(tmp, "skills.py")
            with open(skills_path, "w", encoding="utf-8") as f:
                f.write("def boost(state):\n    return 'RIGHT,B'\n")

            module = _restricted_skills_module(skills_path)
            module_builtins = module.__dict__["__builtins__"]

        self.assertNotIn("open", module_builtins)
        self.assertNotIn("eval", module_builtins)
        self.assertNotIn("exec", module_builtins)
        self.assertIn("abs", module_builtins)  # safe builtins still present
        self.assertEqual(set(SAFE_POLICY_BUILTINS) - {"__import__"}, set(module_builtins) - {"__import__"})
        # __import__ is the deny stub, not the real importer.
        with self.assertRaises(ImportError):
            module_builtins["__import__"]("os")

    def test_from_import_form_also_restricted(self):
        with tempfile.TemporaryDirectory() as tmp:
            action = self._run_with_skills(
                tmp,
                "def boost(state):\n    return 'RIGHT,B'\n",
                "from policies import skills\n"
                "def get_action(state):\n    return skills.boost(state)\n",
            )
            self.assertEqual(action, "RIGHT,B")


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
