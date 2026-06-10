import importlib
import os

import cv2

DEFAULT_GAME = "SonicTheHedgehog-Genesis"
DEFAULT_STATE = "GreenHillZone.Act1"
DEFAULT_BACKEND = "auto"


def normalize_backend_name(backend):
    requested = (backend or os.environ.get("SONIC_RETRO_BACKEND", DEFAULT_BACKEND)).lower()
    aliases = {
        "gym-retro": "legacy",
        "retro": "legacy",
        "legacy": "legacy",
        "stable-retro": "stable",
        "stable_retro": "stable",
        "stable": "stable",
        "auto": "auto",
    }
    if requested not in aliases:
        raise ValueError(f"Unsupported retro backend: {backend}")
    return aliases[requested]


def resolve_backend_module(backend=None):
    normalized = normalize_backend_name(backend)
    if normalized == "auto":
        try:
            return importlib.import_module("stable_retro"), "stable"
        except ImportError:
            return importlib.import_module("retro"), "legacy"
    if normalized == "stable":
        return importlib.import_module("stable_retro"), "stable"
    return importlib.import_module("retro"), "legacy"


def make_retro_env(module, game=DEFAULT_GAME, state=DEFAULT_STATE, record_path=None, **extra_kwargs):
    kwargs = {"game": game, "state": state, **extra_kwargs}
    if record_path is not None:
        kwargs["record"] = record_path

    is_stable = getattr(module, "__name__", "") == "stable_retro"
    if is_stable and "render_mode" not in kwargs:
        # stable-retro defaults to render_mode="human", which opens a pyglet/
        # OpenGL viewer on reset() -- impossible on headless training boxes,
        # containers, and CI. Training reads frames from the observation, so
        # default to no rendering; callers can still pass render_mode through.
        kwargs["render_mode"] = None

    try:
        return module.make(**kwargs)
    except Exception:
        if is_stable and not str(game).endswith("-v0"):
            retry_kwargs = dict(kwargs)
            retry_kwargs["game"] = f"{game}-v0"
            return module.make(**retry_kwargs)
        raise


def normalize_reset_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def normalize_step_result(result):
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    if isinstance(result, tuple) and len(result) == 4:
        return result
    raise ValueError(f"Unsupported emulator step result: {result!r}")


class SonicEnvWrapper:
    def __init__(self, state=DEFAULT_STATE, record_path=None, backend=None, game=DEFAULT_GAME, retro_module=None):
        self.module, self.backend = (retro_module, normalize_backend_name(backend)) if retro_module else resolve_backend_module(backend)
        self.env = make_retro_env(self.module, game=game, state=state, record_path=record_path)
        self.obs, self.info = normalize_reset_result(self.env.reset())
        self.frame_count = 0

    def step(self, action):
        """
        Executes an action and advances the environment.
        action: array of shape (12,) representing button presses
        [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
        For Genesis: usually B, A, Mode, Start, Up, Down, Left, Right, C, Y, X, Z
        Stable-retro standard Genesis mapping: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
        """
        self.obs, reward, done, self.info = normalize_step_result(self.env.step(action))
        self.frame_count += 1
        return self.obs, reward, done, self.info

    def reset(self):
        self.obs, self.info = normalize_reset_result(self.env.reset())
        self.frame_count = 0
        if hasattr(self, 'last_x'):
            del self.last_x
            del self.last_y
        return self.obs

    def _emulator(self):
        """The raw retro emulator handle, common to gym-retro and stable-retro."""
        unwrapped = getattr(self.env, "unwrapped", self.env)
        emulator = getattr(unwrapped, "em", None)
        if emulator is None:
            raise RuntimeError("This emulator backend does not expose savestates (no .em handle).")
        return emulator

    def save_emulator_state(self):
        """Return a raw savestate of the whole machine (not just tracked RAM)."""
        return self._emulator().get_state()

    def load_emulator_state(self, state_bytes):
        """Restore a savestate and refresh obs/info without advancing a frame.

        Velocity tracking re-baselines to the restored position so the first
        ``get_state`` after a load reports zero velocity instead of the delta
        from wherever the emulator happened to be before the seek.
        """
        emulator = self._emulator()
        emulator.set_state(state_bytes)
        self.obs = emulator.get_screen()
        unwrapped = getattr(self.env, "unwrapped", self.env)
        data = getattr(unwrapped, "data", None)
        self.info = dict(data.lookup_all()) if data is not None else {}
        self.last_x = self.info.get('x', 0)
        self.last_y = self.info.get('y', 0)
        return self.obs

    def get_screenshot(self, filepath="artifacts/failures/latest_screenshot.png"):
        """Saves current observation as an image."""
        # Convert RGB (from retro) to BGR (for OpenCV)
        if self.obs is not None:
            bgr_img = cv2.cvtColor(self.obs, cv2.COLOR_RGB2BGR)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv2.imwrite(filepath, bgr_img)
            return filepath
        return None

    def get_state(self):
        """Returns the current relevant RAM values as a dict."""
        current_x = self.info.get('x', 0)
        current_y = self.info.get('y', 0)

        # Calculate velocity if previous state exists
        if not hasattr(self, 'last_x'):
            self.last_x = current_x
            self.last_y = current_y

        x_vel = current_x - self.last_x
        y_vel = current_y - self.last_y

        self.last_x = current_x
        self.last_y = current_y

        return {
            "x_pos": current_x,
            "y_pos": current_y,
            "x_velocity": x_vel,
            "y_velocity": y_vel,
            "screen_x": self.info.get('screen_x', 0),
            "screen_y": self.info.get('screen_y', 0),
            "screen_x_end": self.info.get('screen_x_end', 0),
            "rings": self.info.get('rings', 0),
            "lives": self.info.get('lives', 3),
            "score": self.info.get('score', 0),
            # Level identity, so the policy can branch per zone/act and the
            # runner can detect when a level has been cleared.
            "zone": self.info.get('zone', 0),
            "act": self.info.get('act', 0),
            "level_end_bonus": self.info.get('level_end_bonus', 0),
        }

    def close(self):
        self.env.close()

if __name__ == "__main__":
    env = SonicEnvWrapper()
    print(f"Environment initialized with backend: {env.backend}")
    env.reset()
    action = env.env.action_space.sample()
    obs, rew, done, info = env.step(action)
    print(f"Step taken. Info: {info}")
    env.close()
