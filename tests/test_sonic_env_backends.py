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


class FakeEmulator:
    def __init__(self):
        self.state = b"live-state"

    def get_state(self):
        return self.state

    def set_state(self, state_bytes):
        self.state = state_bytes

    def get_screen(self):
        return f"screen-for-{self.state.decode('ascii')}"


class FakeData:
    def __init__(self, variables):
        self.variables = variables

    def lookup_all(self):
        return dict(self.variables)


class FakeSavestateEnv(FakeGymnasiumEnv):
    def __init__(self):
        super().__init__()
        self.em = FakeEmulator()
        self.data = FakeData({"x": 1500, "y": 320, "rings": 7, "lives": 3})


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

    def make_savestate_wrapper(self):
        return SonicEnvWrapper(
            state="GreenHillZone.Act1",
            backend="stable",
            retro_module=FakeModule(FakeSavestateEnv()),
        )

    def test_save_emulator_state_returns_raw_savestate(self):
        wrapper = self.make_savestate_wrapper()

        self.assertEqual(wrapper.save_emulator_state(), b"live-state")

    def test_load_emulator_state_refreshes_obs_info_and_rebaselines_velocity(self):
        wrapper = self.make_savestate_wrapper()

        obs = wrapper.load_emulator_state(b"snapshot-42")

        self.assertEqual(obs, "screen-for-snapshot-42")
        self.assertEqual(wrapper.obs, "screen-for-snapshot-42")
        self.assertEqual(wrapper.info["x"], 1500)
        state = wrapper.get_state()
        self.assertEqual(state["x_pos"], 1500)
        self.assertEqual(state["y_pos"], 320)
        self.assertEqual(state["rings"], 7)
        # The first state after a seek must not report a velocity computed
        # against wherever the emulator was before the load.
        self.assertEqual(state["x_velocity"], 0)
        self.assertEqual(state["y_velocity"], 0)

    def test_savestate_api_raises_clearly_without_emulator_handle(self):
        module = FakeModule(FakeGymnasiumEnv())  # no .em on this fake
        wrapper = SonicEnvWrapper(
            state="GreenHillZone.Act1",
            backend="stable",
            retro_module=module,
        )

        with self.assertRaises(RuntimeError):
            wrapper.save_emulator_state()


if __name__ == "__main__":
    unittest.main()
