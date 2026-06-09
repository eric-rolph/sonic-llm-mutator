"""Real-emulator smoke test against the bundled homebrew Airstriker ROM.

The unit suite stubs the emulator, so backend API drift (reset/step signature
changes, info-dict shape) only surfaces in real training runs. Both gym-retro
and stable-retro legally bundle the Airstriker-Genesis homebrew ROM, so this
test can run a real rollout in CI without any commercial ROM. It skips
entirely when no backend is installed (the default dev environment).
"""

import os
import tempfile
import unittest

from core.actions import action_string_to_array


def _backend_available():
    for module_name in ("stable_retro", "retro"):
        try:
            __import__(module_name)
            return True
        except ImportError:
            continue
    return False


@unittest.skipUnless(_backend_available(), "no retro emulator backend installed")
class EmulatorSmokeTests(unittest.TestCase):
    def test_airstriker_rollout_steps_screenshots_and_reports_state(self):
        from emulator.sonic_env import SonicEnvWrapper

        env = SonicEnvWrapper(game="Airstriker-Genesis", state="Level1", record_path=None)
        try:
            self.assertIn(env.backend, {"stable", "legacy"})

            action = action_string_to_array("B")
            frames = 0
            for _ in range(60):
                obs, reward, done, info = env.step(action)
                frames += 1
                if done:
                    break

            self.assertGreater(frames, 0)
            self.assertEqual(env.frame_count, frames)

            state = env.get_state()
            for key in ("x_pos", "y_pos", "rings", "lives", "score", "zone", "act"):
                self.assertIn(key, state)

            with tempfile.TemporaryDirectory() as tmp:
                shot_path = os.path.join(tmp, "smoke.png")
                saved = env.get_screenshot(shot_path)
                self.assertEqual(saved, shot_path)
                self.assertTrue(os.path.exists(shot_path))
                self.assertGreater(os.path.getsize(shot_path), 0)

            env.reset()
            self.assertEqual(env.frame_count, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
