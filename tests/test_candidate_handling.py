import os
import tempfile
import unittest

from main import build_policy_load_failure, clear_candidate_recording


class CandidateHandlingTests(unittest.TestCase):
    def test_build_policy_load_failure_uses_specific_reason(self):
        fitness, frames, max_x, reason, screenshot, trace, components = build_policy_load_failure(
            SyntaxError("bad syntax")
        )

        self.assertEqual(fitness, 0.0)
        self.assertEqual(frames, 0)
        self.assertEqual(max_x, 0)
        self.assertIn("Policy failed to load", reason)
        self.assertIn("bad syntax", reason)
        self.assertIsNone(screenshot)
        self.assertEqual(trace, [])
        self.assertEqual(components["load_error"], "bad syntax")

    def test_clear_candidate_recording_removes_stale_bk2(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale_path = os.path.join(tmp, "candidate_0.bk2")
            with open(stale_path, "w", encoding="utf-8") as f:
                f.write("stale")

            clear_candidate_recording(tmp, 0)

            self.assertFalse(os.path.exists(stale_path))


if __name__ == "__main__":
    unittest.main()
