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

    def test_savestate_round_trip_restores_emulator_state(self):
        # Savestates are the foundation of agentic failure diagnosis: capture,
        # diverge for 30 frames, restore, and the authoritative variables must
        # match the captured moment. Also proves stepping works after a load.
        from emulator.sonic_env import SonicEnvWrapper

        env = SonicEnvWrapper(game="Airstriker-Genesis", state="Level1", record_path=None)
        try:
            action = action_string_to_array("B")
            for _ in range(5):
                env.step(action)
            saved = env.save_emulator_state()
            state_at_save = env.get_state()

            for _ in range(30):
                env.step(action_string_to_array("RIGHT"))

            env.load_emulator_state(saved)
            restored = env.get_state()
            for key in ("x_pos", "y_pos", "rings", "lives", "score"):
                self.assertEqual(state_at_save[key], restored[key], key)

            obs, reward, done, info = env.step(action)
            self.assertIsNotNone(obs)
        finally:
            env.close()

    def test_diagnosis_env_coexists_with_training_env(self):
        # The bug live testing caught: retro allows ONE emulator instance per
        # process, so an in-process diagnosis env can never start while the
        # training env exists. The child-process proxy must coexist, accept a
        # savestate captured by the training env, and run experiments on it.
        import tempfile

        from core.diagnosis import ProcessDiagnosisEnv
        from emulator.sonic_env import SonicEnvWrapper

        training_env = SonicEnvWrapper(game="Airstriker-Genesis", state="Level1", record_path=None)
        try:
            for _ in range(5):
                training_env.step(action_string_to_array("B"))
            saved = training_env.save_emulator_state()
            state_at_save = training_env.get_state()

            diagnosis_env = ProcessDiagnosisEnv(
                factory_spec="tests.test_emulator_smoke:make_airstriker_env"
            )
            try:
                diagnosis_env.load_emulator_state(saved)
                restored = diagnosis_env.get_state()
                for key in ("x_pos", "y_pos", "rings", "lives", "score"):
                    self.assertEqual(state_at_save[key], restored[key], key)

                obs, reward, done, info = diagnosis_env.step(action_string_to_array("B"))
                self.assertIsNone(obs)  # frame stripped at the process boundary

                with tempfile.TemporaryDirectory() as tmp:
                    shot = os.path.join(tmp, "diag.png")
                    self.assertEqual(diagnosis_env.get_screenshot(shot), shot)
                    self.assertGreater(os.path.getsize(shot), 0)
            finally:
                diagnosis_env.close()

            # The training env is still healthy after the child ran alongside.
            training_env.step(action_string_to_array("B"))
        finally:
            training_env.close()


def make_airstriker_env():
    """Child-process factory for the coexistence test (must be importable)."""
    from emulator.sonic_env import SonicEnvWrapper

    return SonicEnvWrapper(game="Airstriker-Genesis", state="Level1", record_path=None)


if __name__ == "__main__":
    unittest.main()
