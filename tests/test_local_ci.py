import subprocess
import sys
import unittest
from unittest.mock import patch

import local_ci


class LocalCITests(unittest.TestCase):
    def test_run_local_ci_uses_current_python_interpreter_for_unit_tests(self):
        completed = subprocess.CompletedProcess(
            args=[sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        with patch.object(local_ci.os.path, "exists", return_value=True), patch.object(
            local_ci.subprocess, "run", return_value=completed
        ) as run:
            self.assertTrue(local_ci.run_local_ci())

        self.assertEqual(run.call_args.args[0][0], sys.executable)
        self.assertEqual(run.call_args.args[0][1:], ["-m", "unittest", "discover", "-s", "tests"])

    def test_main_returns_nonzero_when_ci_fails(self):
        with patch.object(local_ci, "run_local_ci", return_value=False):
            self.assertEqual(local_ci.main(), 1)


if __name__ == "__main__":
    unittest.main()
