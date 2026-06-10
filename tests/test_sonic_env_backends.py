import unittest

from emulator.sonic_env import (
    SonicEnvWrapper,
    make_retro_env,
    normalize_reset_result,
    normalize_step_result,
)


class FakeLegacyEnv:
    action_space = object()

    def __init__(self):
        self.closed = False

    def reset(self):
        return "legacy_obs"

    def step(self, action):
        return "legacy_next", 1.0, True, {"x": 10, "y": 20}

    def close(self):
        self.closed = True


class FakeGymnasiumEnv:
    action_space = object()

    def __init__(self):
        self.closed = False

    def reset(self):
        return "stable_obs", {"reset": True}

    def step(self, action):
        return "stable_next", 2.0, False, True, {"x": 30, "y": 40}

    def close(self):
        self.closed = True


class FakeModule:
    def __init__(self, env):
        self.env = env
        self.calls = []

    def make(self, **kwargs):
        self.calls.append(kwargs)
        return self.env


class SonicEnvBackendTests(unittest.TestCase):
    def test_normalize_reset_result_supports_legacy_and_gymnasium(self):
        self.assertEqual(normalize_reset_result("obs"), ("obs", {}))
        self.assertEqual(normalize_reset_result(("obs", {"a": 1})), ("obs", {"a": 1}))

    def test_normalize_step_result_supports_legacy_and_gymnasium(self):
        self.assertEqual(
            normalize_step_result(("obs", 1.0, True, {"x": 1})),
            ("obs", 1.0, True, {"x": 1}),
        )
        self.assertEqual(
            normalize_step_result(("obs", 1.0, False, True, {"x": 1})),
            ("obs", 1.0, True, {"x": 1}),
        )

    def test_make_retro_env_passes_game_state_and_record_path(self):
        module = FakeModule(FakeLegacyEnv())

        env = make_retro_env(
            module,
            game="SonicTheHedgehog-Genesis",
            state="GreenHillZone.Act1",
            record_path="artifacts/videos/tmp",
        )

        self.assertIs(env, module.env)
        self.assertEqual(
            module.calls[0],
            {
                "game": "SonicTheHedgehog-Genesis",
                "state": "GreenHillZone.Act1",
                "record": "artifacts/videos/tmp",
            },
        )

    def test_make_retro_env_passes_backend_specific_options(self):
        module = FakeModule(FakeLegacyEnv())

        env = make_retro_env(
            module,
            game="SonicTheHedgehog-Genesis",
            state=None,
            players=2,
            use_restricted_actions="ALL",
        )

        self.assertIs(env, module.env)
        self.assertEqual(
            module.calls[0],
            {
                "game": "SonicTheHedgehog-Genesis",
                "state": None,
                "players": 2,
                "use_restricted_actions": "ALL",
            },
        )

    def test_make_retro_env_defaults_stable_backend_to_headless(self):
        # stable-retro's render_mode defaults to "human", which needs a real
        # display (pyglet/GLU) the moment reset() runs. Training is headless.
        module = FakeModule(FakeGymnasiumEnv())
        module.__name__ = "stable_retro"

        make_retro_env(module, game="Airstriker-Genesis", state="Level1")

        self.assertIn("render_mode", module.calls[0])
        self.assertIsNone(module.calls[0]["render_mode"])

    def test_make_retro_env_respects_explicit_render_mode_on_stable(self):
        module = FakeModule(FakeGymnasiumEnv())
        module.__name__ = "stable_retro"

        make_retro_env(
            module,
            game="Airstriker-Genesis",
            state="Level1",
            render_mode="rgb_array",
        )

        self.assertEqual(module.calls[0]["render_mode"], "rgb_array")

    def test_make_retro_env_never_sends_render_mode_to_legacy_backend(self):
        # Legacy gym-retro's make() has no render_mode parameter.
        module = FakeModule(FakeLegacyEnv())

        make_retro_env(module, game="Airstriker-Genesis", state="Level1")

        self.assertNotIn("render_mode", module.calls[0])

    def test_wrapper_normalizes_gymnasium_reset_and_step(self):
        module = FakeModule(FakeGymnasiumEnv())
        wrapper = SonicEnvWrapper(
            state="GreenHillZone.Act1",
            backend="stable",
            retro_module=module,
        )

        self.assertEqual(wrapper.backend, "stable")
        self.assertEqual(wrapper.obs, "stable_obs")
        obs, reward, done, info = wrapper.step([0] * 12)

        self.assertEqual(obs, "stable_next")
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertEqual(info["x"], 30)


if __name__ == "__main__":
    unittest.main()
