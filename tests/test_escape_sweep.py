import unittest
from contextlib import redirect_stdout
from io import StringIO

from core.escape_sweep import SEQUENCES, SINGLE_HOLDS, sweep_frontier_escapes, sweep_offsets

WINDOW = {
    "failure_reason": "Sonic lost a life at the frontier (zone 0 act 1, x=4268)",
    "failure": {"frame": 5389, "x_pos": 332, "frontier_x": 4268, "zone": 0, "act": 1},
    "snapshots": [
        {"frame": 4620, "x_pos": 3928, "frontier": True},
        {"frame": 4920, "x_pos": 4268, "frontier": True},
        {"frame": 5280, "x_pos": 113},   # post-respawn, not pinned
        {"frame": 5340, "x_pos": 207},
    ],
}


class FakeSession:
    """Scriptable stand-in for DiagnosisSession."""

    def __init__(self, window, capture_screenshots=True, verify_on_call=None):
        self.window = window
        self.capture_screenshots = capture_screenshots
        self.verified_experiments = []
        self.calls = []
        self.closed = False
        self._verify_on_call = verify_on_call  # call index that "passes"

    def _experiment(self, kind, offset, payload):
        index = len(self.calls)
        self.calls.append((kind, offset, payload))
        if self._verify_on_call is not None and index == self._verify_on_call:
            self.verified_experiments.append(
                {"zone": 0, "act": 1, "start_x": 3928, "actions": "RIGHT,B",
                 "hold_frames": 45, "max_x": 4400, "frames_before_failure": offset}
            )
            return {"ok": True, "passed_frontier_x": True, "text": "YES"}
        return {"ok": True, "passed_frontier_x": False, "text": "no"}

    def try_actions(self, offset, actions, hold_frames):
        return self._experiment("hold", offset, (actions, hold_frames))

    def try_action_sequence(self, offset, segments):
        return self._experiment("seq", offset, tuple(s["actions"] for s in segments))

    def close(self):
        self.closed = True


def run_sweep(verify_on_call=None, window=WINDOW, **kwargs):
    sessions = []

    def factory(w, capture_screenshots=True):
        session = FakeSession(w, capture_screenshots, verify_on_call)
        sessions.append(session)
        return session

    with redirect_stdout(StringIO()):
        experiments, summary = sweep_frontier_escapes(window, session_factory=factory, **kwargs)
    return experiments, summary, sessions[0]


class SweepOffsetsTests(unittest.TestCase):
    def test_prefers_pinned_snapshots_most_runway_first(self):
        offsets = sweep_offsets(WINDOW)
        self.assertEqual(offsets, [5389 - 4620, 5389 - 4920])  # pins only, earliest first

    def test_falls_back_to_largest_x_without_pins(self):
        window = dict(WINDOW)
        window["snapshots"] = [
            {"frame": 5280, "x_pos": 113},
            {"frame": 5340, "x_pos": 207},
        ]
        self.assertEqual(sweep_offsets(window), [5389 - 5280, 5389 - 5340][:2] or [])
        self.assertTrue(all(o >= 0 for o in sweep_offsets(window)))


class SweepTests(unittest.TestCase):
    def test_no_escape_runs_full_battery_at_every_pin(self):
        experiments, summary, session = run_sweep(verify_on_call=None)
        self.assertEqual(experiments, [])
        self.assertIn("none beat it", summary)
        per_offset = len(SINGLE_HOLDS) + len(SEQUENCES)
        self.assertEqual(len(session.calls), 2 * per_offset)  # both pins swept
        self.assertTrue(session.closed)
        self.assertFalse(session.capture_screenshots)  # no PNG churn

    def test_early_exit_once_verified(self):
        experiments, summary, session = run_sweep(verify_on_call=1, stop_after=1)
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["max_x"], 4400)
        self.assertIn("VERIFIED", summary)
        self.assertEqual(len(session.calls), 2)  # stopped right after the hit

    def test_verified_shape_matches_guard_compiler(self):
        from core.frontier import build_diagnosis_guard_candidate

        experiments, _, _ = run_sweep(verify_on_call=0, stop_after=1)
        guard = build_diagnosis_guard_candidate(
            "def get_action(state):\n    return 'RIGHT'\n", experiments[0]
        )
        self.assertIsNotNone(guard)
        self.assertIn("# DIAGNOSIS_GUARD zone=0 act=1 x=3928", guard)

    def test_empty_window_is_a_clean_skip(self):
        experiments, summary, _ = run_sweep(window={"failure": {"frame": 100}, "snapshots": []})
        self.assertEqual(experiments, [])
        self.assertIn("skipped", summary)


class MaybeDiagnoseIntegrationTests(unittest.TestCase):
    def test_sweep_hit_short_circuits_vision_diagnosis(self):
        import main

        class NeverCalledMutator:
            def diagnose_failure(self, *a, **k):
                raise AssertionError("vision diagnosis must not run when the sweep verifies")

        window_calls = []

        def factory(w, capture_screenshots=True):
            window_calls.append(w)
            return FakeSession(w, capture_screenshots, verify_on_call=0)

        cache = {}
        with redirect_stdout(StringIO()):
            result = main.maybe_diagnose_frontier(
                NeverCalledMutator(),
                {"failure_reason": WINDOW["failure_reason"], "window": "somewhere", "trace": []},
                cache,
                emulator_available=True,
                session_factory=factory,
                report_path="artifacts/diagnosis/_test_report.json",
                window_loader=lambda d: WINDOW,
            )

        self.assertTrue(result["verified_experiments"])
        self.assertIn("Mechanical escape sweep", result["report"])
        self.assertEqual(cache["result"], result)  # cached like a vision result


if __name__ == "__main__":
    unittest.main()
