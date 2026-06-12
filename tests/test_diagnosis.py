import json
import os
import tempfile
import unittest

from core.diagnosis import (
    TRY_ACTIONS_MAX_FRAMES,
    DiagnosisSession,
    FailureSnapshotRing,
    ProcessDiagnosisEnv,
    load_failure_window,
    window_key,
)


class FakeSavestateEnv:
    """Savestate-capable env whose x position is encoded in the state bytes."""

    def __init__(self):
        self.x = 0
        self.saved = 0
        self.closed = False
        self.step_error = None

    def save_emulator_state(self):
        self.saved += 1
        return f"state-x-{self.x}".encode("ascii")

    def load_emulator_state(self, state_bytes):
        self.x = int(state_bytes.decode("ascii").rsplit("-", 1)[1])

    def get_state(self):
        return {"x_pos": self.x, "y_pos": 100, "zone": 0, "act": 1, "rings": 3, "lives": 3}

    def step(self, action):
        if self.step_error is not None:
            raise self.step_error
        # Holding RIGHT (index 7) advances; anything else stalls.
        self.x += 10 if action[7] else 0
        return None, 0, False, {}

    def get_screenshot(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(b"png")
        return filepath

    def close(self):
        self.closed = True


class BrokenSavestateEnv(FakeSavestateEnv):
    def save_emulator_state(self):
        raise RuntimeError("no savestates on this backend")


class FailureSnapshotRingTests(unittest.TestCase):
    def fill_ring(self, env, frames):
        ring = FailureSnapshotRing(interval=60, capacity=10)
        for frame in frames:
            env.x = frame  # make position track the frame for easy assertions
            ring.record(env, frame, env.get_state())
        return ring

    def test_record_honours_cadence_and_capacity(self):
        env = FakeSavestateEnv()
        ring = self.fill_ring(env, range(0, 1200, 30))  # every 30 frames offered

        # Cadence: only every 60 frames captured; capacity: last 10 kept.
        self.assertEqual(len(ring.snapshots), 10)
        frames = [snapshot["frame"] for snapshot in ring.snapshots]
        self.assertEqual(frames, list(range(600, 1200, 60)))

    def test_backend_without_savestates_disables_ring_silently(self):
        env = BrokenSavestateEnv()
        ring = FailureSnapshotRing(interval=1, capacity=5)

        self.assertFalse(ring.record(env, 0, env.get_state()))
        self.assertTrue(ring.disabled)
        self.assertFalse(ring.record(env, 100, env.get_state()))
        self.assertEqual(ring.snapshots, [])

    def test_frontier_x_resets_on_act_transition(self):
        # Live-observed bug: the baseline cleared Act 1 at x~9767, then failed
        # in Act 2 at x~2478 — and real Act-2 escapes (x=3418) were judged
        # against the phantom Act-1 frontier. The frontier must be per-act.
        env = FakeSavestateEnv()
        ring = FailureSnapshotRing(interval=60, capacity=10)

        ring.record(env, 0, {"x_pos": 9000, "y_pos": 1, "zone": 0, "act": 0, "rings": 0, "lives": 3})
        ring.record(env, 60, {"x_pos": 9767, "y_pos": 1, "zone": 0, "act": 0, "rings": 0, "lives": 3})
        ring.record(env, 120, {"x_pos": 100, "y_pos": 1, "zone": 0, "act": 1, "rings": 0, "lives": 3})
        ring.record(env, 180, {"x_pos": 2478, "y_pos": 1, "zone": 0, "act": 1, "rings": 0, "lives": 3})
        ring.record(env, 240, {"x_pos": 2191, "y_pos": 1, "zone": 0, "act": 1, "rings": 0, "lives": 3})

        with tempfile.TemporaryDirectory() as tmp:
            ring.persist(tmp, failure_reason="stuck", failure_frame=240)
            window = load_failure_window(tmp)

        self.assertEqual(window["failure"]["frontier_x"], 2478)

    def test_persist_and_load_round_trip(self):
        env = FakeSavestateEnv()
        ring = self.fill_ring(env, [0, 60, 120])
        final_state = {"x_pos": 150, "y_pos": 90, "zone": 0, "act": 1, "rings": 0, "lives": 2}

        with tempfile.TemporaryDirectory() as tmp:
            directory = ring.persist(tmp, failure_reason="stuck", final_state=final_state, failure_frame=150)
            window = load_failure_window(directory)

            self.assertEqual(window["failure_reason"], "stuck")
            self.assertEqual(window["failure"]["frame"], 150)
            self.assertEqual(window["failure"]["x_pos"], 150)
            self.assertEqual(len(window["snapshots"]), 3)
            self.assertEqual(window["snapshots"][0]["frame"], 0)
            with open(window["snapshots"][2]["path"], "rb") as f:
                self.assertEqual(f.read(), b"state-x-120")
            self.assertIsNotNone(window_key(window))

    def test_persist_replaces_previous_window(self):
        env = FakeSavestateEnv()
        with tempfile.TemporaryDirectory() as tmp:
            self.fill_ring(env, [0, 60]).persist(tmp, failure_reason="first", failure_frame=100)
            self.fill_ring(env, [600]).persist(tmp, failure_reason="second", failure_frame=700)

            window = load_failure_window(tmp)

            self.assertEqual(window["failure_reason"], "second")
            self.assertEqual(len(window["snapshots"]), 1)
            stale = [name for name in os.listdir(tmp) if name == "0.state"]
            self.assertEqual(stale, [])

    def test_empty_ring_persists_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(FailureSnapshotRing().persist(tmp))

    def test_load_tolerates_missing_and_corrupt_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_failure_window(os.path.join(tmp, "missing")))

            with open(os.path.join(tmp, "window.json"), "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertIsNone(load_failure_window(tmp))

            # Valid manifest whose blobs are gone is also unusable.
            with open(os.path.join(tmp, "window.json"), "w", encoding="utf-8") as f:
                json.dump({"snapshots": [{"frame": 0, "file": "0.state"}]}, f)
            self.assertIsNone(load_failure_window(tmp))


class DiagnosisSessionTests(unittest.TestCase):
    def make_session(self, tmp, env=None):
        ring_env = FakeSavestateEnv()
        ring = FailureSnapshotRing(interval=60, capacity=10)
        for frame in (0, 60, 120, 180):
            ring_env.x = frame * 2  # x advances faster than frames
            ring.record(ring_env, frame, ring_env.get_state())
        window_dir = os.path.join(tmp, "window")
        ring.persist(
            window_dir,
            failure_reason="Sonic got stuck",
            final_state={"x_pos": 400, "y_pos": 100, "zone": 0, "act": 1, "rings": 0, "lives": 3},
            failure_frame=200,
        )
        window = load_failure_window(window_dir)
        session_env = env or FakeSavestateEnv()
        session = DiagnosisSession(
            window,
            env_factory=lambda: session_env,
            screenshot_dir=os.path.join(tmp, "shots"),
        )
        return session, session_env

    def test_describe_window_lists_offsets_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, _ = self.make_session(tmp)
            text = session.describe_window()

        self.assertIn("offset=200", text)  # frame 0 is 200 frames before failure
        self.assertIn("offset=20", text)   # frame 180
        self.assertIn("Failure moment: frame=200 x=400", text)

    def test_view_frame_seeks_nearest_snapshot_and_screenshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp)

            # 70 frames before failure(200) = frame 130 -> nearest at-or-before is 120.
            result = session.view_frame(70)

            self.assertTrue(result["ok"])
            self.assertEqual(env.x, 240)  # snapshot at frame 120 had x=240
            self.assertIn('"x_pos": 240', result["text"])
            self.assertTrue(os.path.exists(result["screenshot"]))
            self.assertEqual(session.last_screenshot, result["screenshot"])

    def test_try_actions_reports_progress_past_frontier_x(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp)

            # From frame 180 (x=360), holding RIGHT for 30 frames -> x=660 > 400.
            result = session.try_actions(20, "RIGHT", 30)

            self.assertTrue(result["ok"])
            self.assertTrue(result["passed_frontier_x"])
            self.assertIn("x: 360 -> 660", result["text"])
            self.assertIn("YES", result["text"])

            # Stalling input never beats the frontier.
            result = session.try_actions(20, "DOWN", 30)
            self.assertFalse(result["passed_frontier_x"])

    def test_experiments_are_judged_against_the_true_frontier_not_the_resting_x(self):
        # Live testing showed Sonic can bounce far backward before the stuck
        # detector fires: the run reached x=1000 but died at x=400. An
        # experiment reaching 660 must NOT be reported as beating the run.
        ring_env = FakeSavestateEnv()
        ring = FailureSnapshotRing(interval=60, capacity=10)
        for frame, x in ((0, 0), (60, 1000), (120, 240), (180, 360)):
            ring_env.x = x
            ring.record(ring_env, frame, ring_env.get_state())
        with tempfile.TemporaryDirectory() as tmp:
            window_dir = os.path.join(tmp, "window")
            ring.persist(
                window_dir,
                failure_reason="Sonic got stuck",
                final_state={"x_pos": 400, "y_pos": 100, "zone": 0, "act": 1, "rings": 0, "lives": 3},
                failure_frame=200,
            )
            window = load_failure_window(window_dir)
            self.assertEqual(window["failure"]["frontier_x"], 1000)

            session = DiagnosisSession(
                window,
                env_factory=lambda: FakeSavestateEnv(),
                screenshot_dir=os.path.join(tmp, "shots"),
            )
            result = session.try_actions(20, "RIGHT", 30)  # x 360 -> 660

        self.assertTrue(result["ok"])
        self.assertFalse(result["passed_frontier_x"])
        self.assertIn("frontier x=1000", result["text"])

    def test_verified_escapes_are_recorded_for_guard_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp)

            session.try_actions(20, "RIGHT", 30)  # x 360 -> 660, beats 400
            session.try_actions(20, "DOWN", 30)   # stalls, no escape

        self.assertEqual(len(session.verified_experiments), 1)
        experiment = session.verified_experiments[0]
        self.assertEqual(experiment["actions"], "RIGHT")
        self.assertEqual(experiment["start_x"], 360)
        self.assertEqual(experiment["max_x"], 660)
        self.assertEqual(experiment["zone"], 0)
        self.assertEqual(experiment["act"], 1)
        self.assertEqual(experiment["hold_frames"], 30)

    def test_try_action_sequence_plays_segments_and_records_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp)

            result = session.try_action_sequence(
                20,
                [
                    {"actions": "RIGHT", "frames": 20},   # x 360 -> 560
                    {"actions": "RIGHT,B", "frames": 10},  # x 560 -> 660
                ],
            )

        self.assertTrue(result["ok"], result["text"])
        self.assertTrue(result["passed_frontier_x"])
        self.assertIn("'RIGHT' x20 (from x=360", result["text"])
        self.assertIn("'RIGHT,B' x10 (from x=560", result["text"])

        self.assertEqual(len(session.verified_experiments), 1)
        experiment = session.verified_experiments[0]
        self.assertEqual(len(experiment["segments"]), 2)
        self.assertEqual(experiment["segments"][0]["start_x"], 360)
        self.assertEqual(experiment["segments"][1]["start_x"], 560)
        self.assertEqual(experiment["max_x"], 660)

    def test_try_action_sequence_rejects_empty_and_caps_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp)

            self.assertFalse(session.try_action_sequence(20, [])["ok"])
            self.assertFalse(session.try_action_sequence(20, ["not-a-dict"])["ok"])

            result = session.try_action_sequence(
                20, [{"actions": "RIGHT", "frames": 10_000}]
            )
            self.assertTrue(result["ok"])
            self.assertIn("x600", result["text"])  # capped at SEQUENCE_MAX_FRAMES

    def test_try_actions_caps_hold_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp)

            result = session.try_actions(20, "RIGHT", 10_000)

            self.assertTrue(result["ok"])
            self.assertIn(f"for {TRY_ACTIONS_MAX_FRAMES} frames", result["text"])

    def test_failed_step_returns_error_and_recovers_on_next_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, env = self.make_session(tmp, env=FakeSavestateEnv())
            env.step_error = RuntimeError("emulator crashed")

            result = session.try_actions(20, "RIGHT", 30)
            self.assertFalse(result["ok"])
            self.assertIn("emulator crashed", result["text"])
            self.assertTrue(env.closed)  # broken env was dropped

            # The session lazily rebuilds an env; with our factory returning the
            # same object, clear the fault and confirm the next call works.
            env.step_error = None
            env.closed = False
            result = session.view_frame(20)
            self.assertTrue(result["ok"])

    def test_view_frame_with_empty_window_is_safe(self):
        session = DiagnosisSession({"snapshots": [], "failure": {}}, env_factory=FakeSavestateEnv)
        result = session.view_frame(60)
        self.assertFalse(result["ok"])


class ProcessDiagnosisEnvTests(unittest.TestCase):
    """The emulator allows one instance per process, so diagnosis envs live
    in a spawned child; these tests drive the proxy against an importable
    stub env across a real process boundary."""

    def test_proxy_round_trips_state_steps_and_screenshots(self):
        env = ProcessDiagnosisEnv(factory_spec="tests._diagnosis_env_stub:make_stub_env")
        try:
            env.load_emulator_state(b"state-x-240")
            self.assertEqual(env.get_state()["x_pos"], 240)

            obs, reward, done, info = env.step([0] * 7 + [1] + [0] * 4)  # RIGHT
            self.assertIsNone(obs)  # heavy frame stripped before pickling
            self.assertEqual(info["x"], 250)

            with tempfile.TemporaryDirectory() as tmp:
                shot = os.path.join(tmp, "proxy.png")
                self.assertEqual(env.get_screenshot(shot), shot)
                self.assertTrue(os.path.exists(shot))
        finally:
            env.close()

    def test_proxy_surfaces_child_errors_as_runtime_errors(self):
        env = ProcessDiagnosisEnv(factory_spec="tests._diagnosis_env_stub:make_stub_env")
        try:
            with self.assertRaises(RuntimeError):
                env.load_emulator_state(b"not-a-valid-state")
            # The worker survives a failed call and serves the next one.
            env.load_emulator_state(b"state-x-10")
            self.assertEqual(env.get_state()["x_pos"], 10)
        finally:
            env.close()

    def test_broken_factory_fails_construction_with_reason(self):
        with self.assertRaises(RuntimeError) as ctx:
            ProcessDiagnosisEnv(factory_spec="tests._diagnosis_env_stub:make_broken_env")
        self.assertIn("stub factory exploded", str(ctx.exception))

    def test_diagnosis_session_works_end_to_end_over_the_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            ring_env = FakeSavestateEnv()
            ring = FailureSnapshotRing(interval=60, capacity=10)
            for frame in (0, 60, 120, 180):
                ring_env.x = frame * 2
                ring.record(ring_env, frame, ring_env.get_state())
            window_dir = os.path.join(tmp, "window")
            ring.persist(
                window_dir,
                failure_reason="Sonic got stuck",
                final_state={"x_pos": 400, "y_pos": 100, "zone": 0, "act": 1, "rings": 0, "lives": 3},
                failure_frame=200,
            )
            session = DiagnosisSession(
                load_failure_window(window_dir),
                env_factory=lambda: ProcessDiagnosisEnv(
                    factory_spec="tests._diagnosis_env_stub:make_stub_env"
                ),
                screenshot_dir=os.path.join(tmp, "shots"),
            )
            try:
                view = session.view_frame(20)
                self.assertTrue(view["ok"], view["text"])
                self.assertIn('"x_pos": 360', view["text"])

                experiment = session.try_actions(20, "RIGHT", 30)
                self.assertTrue(experiment["ok"], experiment["text"])
                self.assertTrue(experiment["passed_frontier_x"])
                self.assertTrue(os.path.exists(experiment["screenshot"]))
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
