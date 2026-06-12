"""Importable fake env for ProcessDiagnosisEnv tests.

The diagnosis worker runs in a *spawned* child process, so its env factory
must be an importable top-level callable — the in-test fake classes used
elsewhere cannot cross the process boundary.
"""

import os


class StubDiagnosisEnv:
    def __init__(self):
        self.x = 0

    def load_emulator_state(self, state_bytes):
        self.x = int(state_bytes.decode("ascii").rsplit("-", 1)[1])

    def get_state(self):
        return {"x_pos": self.x, "y_pos": 100, "zone": 0, "act": 1, "rings": 3, "lives": 3}

    def step(self, action):
        # Holding RIGHT (index 7) advances; anything else stalls. The fake
        # observation is a large object to prove the proxy strips it.
        self.x += 10 if action[7] else 0
        return bytearray(512 * 1024), 0.0, False, {"x": self.x}

    def get_screenshot(self, filepath="stub_screenshot.png"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(b"png")
        return filepath

    def close(self):
        pass


def make_stub_env():
    return StubDiagnosisEnv()


def make_broken_env():
    raise RuntimeError("stub factory exploded")
