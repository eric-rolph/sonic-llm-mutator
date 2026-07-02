"""An escape only verifies if Sonic SURVIVES it.

Live-observed loophole: a jump peaked past the frontier (x=4272 > 4268) but
Sonic was falling into a wider pit; the death landed after the experiment
horizon (ended_early only fired at lives==0, and a lives-drop mid-experiment
went undetected), so a doomed trajectory was verified and compiled into the
champion. Experiments now track lives and settle VERIFY_SETTLE_FRAMES past the
scripted input before verifying.
"""

import os
import tempfile
import unittest

from core.diagnosis import DiagnosisSession, FailureSnapshotRing, load_failure_window


class DeathZoneEnv:
    """x advances 10/frame on RIGHT; crossing death_x costs a life (x resets)."""

    def __init__(self, death_x=None):
        self.x = 0
        self.lives = 3
        self.death_x = death_x

    def save_emulator_state(self):
        return f"{self.x}:{self.lives}".encode("ascii")

    def load_emulator_state(self, blob):
        x, lives = blob.decode("ascii").split(":")
        self.x, self.lives = int(x), int(lives)

    def get_state(self):
        return {"x_pos": self.x, "y_pos": 100, "zone": 0, "act": 1, "rings": 0, "lives": self.lives}

    def step(self, action):
        if action[7]:  # RIGHT
            self.x += 10
        if self.death_x is not None and self.x >= self.death_x:
            self.lives -= 1
            self.x = 0  # respawn
        return None, 0, False, {}

    def get_screenshot(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(b"png")
        return filepath

    def close(self):
        pass


def make_session(tmp, env):
    ring = FailureSnapshotRing(interval=60, capacity=10)
    env.x = 4000
    ring.record(env, 0, env.get_state(), act_max_x=4000)
    window_dir = ring.persist(
        directory=os.path.join(tmp, "window"),
        failure_reason="lost a life at the frontier",
        final_state={"x_pos": 300, "y_pos": 100, "zone": 0, "act": 1, "rings": 0, "lives": 2},
        failure_frame=600,
    )
    window = load_failure_window(window_dir)
    window["failure"]["frontier_x"] = 4268
    return DiagnosisSession(
        window,
        env_factory=lambda: env,
        screenshot_dir=os.path.join(tmp, "shots"),
        capture_screenshots=False,
    )


class SettledVerificationTests(unittest.TestCase):
    def test_survivable_escape_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = DeathZoneEnv(death_x=None)  # nothing lethal
            session = make_session(tmp, env)
            result = session.try_actions(600, "RIGHT", 40)  # 4000 -> 4400 > 4268

        self.assertTrue(result["passed_frontier_x"])
        self.assertEqual(len(session.verified_experiments), 1)

    def test_doomed_escape_that_peaks_past_frontier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Death zone at 4500: the input (40 frames) peaks at 4400 > 4268
            # ALIVE, but the settle window carries Sonic into the death zone --
            # exactly the wider-pit loophole.
            env = DeathZoneEnv(death_x=4500)
            session = make_session(tmp, env)
            result = session.try_actions(600, "RIGHT", 40)

        self.assertFalse(result["passed_frontier_x"])
        self.assertIn("DIED", result["text"])
        self.assertEqual(session.verified_experiments, [])

    def test_death_during_the_input_itself_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = DeathZoneEnv(death_x=4300)  # dies mid-input after passing 4268
            session = make_session(tmp, env)
            result = session.try_actions(600, "RIGHT", 60)

        self.assertFalse(result["passed_frontier_x"])
        self.assertEqual(session.verified_experiments, [])

    def test_sequence_settles_with_last_segment_and_rejects_doomed(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = DeathZoneEnv(death_x=4500)
            session = make_session(tmp, env)
            result = session.try_action_sequence(
                600,
                [{"actions": "RIGHT", "frames": 20}, {"actions": "RIGHT", "frames": 20}],
            )

        self.assertFalse(result["passed_frontier_x"])
        self.assertIn("DIED", result["text"])

    def test_extra_life_cannot_mask_a_later_death(self):
        # Lives 3 -> 4 (1-up) -> 3 (death): comparing to the START lives saw
        # 3 >= 3 and verified a fatal trajectory (agency review). Per-frame
        # decrement detection catches the death.
        class OneUpThenDeathEnv(DeathZoneEnv):
            def step(self, action):
                if action[7]:
                    self.x += 10
                if self.x == 4100:
                    self.lives += 1  # 1-up monitor
                if self.x >= 4500:
                    self.lives -= 1
                    self.x = 0
                return None, 0, False, {}

        with tempfile.TemporaryDirectory() as tmp:
            env = OneUpThenDeathEnv()
            session = make_session(tmp, env)
            result = session.try_actions(600, "RIGHT", 40)  # peaks 4400, dies in settle

        self.assertFalse(result["passed_frontier_x"])
        self.assertEqual(session.verified_experiments, [])

    def test_cross_act_experiment_never_verifies(self):
        # x is only comparable within one act: a snapshot from a different
        # zone/act can "beat" frontier_x with a meaningless coordinate.
        class WrongActEnv(DeathZoneEnv):
            def get_state(self):
                state = super().get_state()
                state["act"] = 0  # failure window is act 1
                return state

        with tempfile.TemporaryDirectory() as tmp:
            env = WrongActEnv()
            session = make_session(tmp, env)
            # make_session stamps failure zone=0 act=1; window snapshots were
            # recorded via env (act=0), so the seek lands in the wrong act.
            session.window["failure"]["zone"] = 0
            session.window["failure"]["act"] = 1
            result = session.try_actions(600, "RIGHT", 40)  # x 4000 -> 4400 > 4268

        self.assertFalse(result["passed_frontier_x"])
        self.assertEqual(session.verified_experiments, [])

    def test_sequence_survivable_still_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = DeathZoneEnv(death_x=None)
            session = make_session(tmp, env)
            result = session.try_action_sequence(
                600,
                [{"actions": "RIGHT", "frames": 20}, {"actions": "RIGHT", "frames": 20}],
            )

        self.assertTrue(result["passed_frontier_x"])
        self.assertEqual(len(session.verified_experiments), 1)


if __name__ == "__main__":
    unittest.main()
