"""The frontier must survive into the diagnosis window.

Live-observed failure mode: Sonic dies AT the frontier (x=4268), respawns at a
checkpoint (x~0), and the stuck detector ends the run ~500 frames later. The
trailing snapshot ring then holds only post-respawn savestates, so no diagnosis
experiment can ever pass frontier_x -- diagnosis was structurally unable to find
an escape, and the failure was misreported as "stuck" at the respawn x.
"""

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from core.diagnosis import FailureSnapshotRing
from core.evaluation import evaluate_policy


class FakeSaveStateEnv:
    def save_emulator_state(self):
        return b"state-bytes"


def record_run(ring, samples):
    """samples: list of (frame, x, zone, act) recorded with act_max_x tracking."""
    env = FakeSaveStateEnv()
    act_max = 0
    prev_zone_act = None
    for frame, x, zone, act in samples:
        if prev_zone_act != (zone, act):
            prev_zone_act = (zone, act)
            act_max = 0
        act_max = max(act_max, x)
        ring.record(
            env, frame,
            {"x_pos": x, "y_pos": 100, "zone": zone, "act": act, "rings": 0, "lives": 3},
            act_max_x=act_max,
        )


class FrontierPinningTests(unittest.TestCase):
    def test_frontier_snapshot_survives_death_respawn_eviction(self):
        ring = FailureSnapshotRing(interval=60, capacity=4, frontier_capacity=2)
        # Progress to x=4200, then die/respawn: 8 post-respawn captures evict
        # the whole trailing window.
        progress = [(f * 60, f * 60, 0, 1) for f in range(1, 71)]          # x rises to 4200
        respawn = [(4200 + f * 60, 10 * f, 0, 1) for f in range(1, 9)]     # x crawls from ~10
        record_run(ring, progress + respawn)

        pinned_xs = [s["info"]["x_pos"] for s in ring.frontier_snapshots]
        self.assertTrue(pinned_xs, "no frontier snapshots pinned")
        self.assertGreaterEqual(max(pinned_xs), 4140)  # near the 4200 frontier
        # Trailing window is all post-respawn (x < 100); pins carry the frontier.
        trailing_max = max(s["info"]["x_pos"] for s in ring.snapshots)
        self.assertLess(trailing_max, 100)

    def test_persist_merges_pins_sorted_and_flagged(self):
        ring = FailureSnapshotRing(interval=60, capacity=3, frontier_capacity=2)
        progress = [(f * 60, f * 60, 0, 1) for f in range(1, 21)]           # to x=1200
        respawn = [(1200 + f * 60, 5, 0, 1) for f in range(1, 6)]
        record_run(ring, progress + respawn)

        with tempfile.TemporaryDirectory() as tmp:
            out = ring.persist(directory=tmp, failure_reason="test")
            with open(os.path.join(out, "window.json"), encoding="utf-8") as f:
                manifest = json.load(f)

        frames = [s["frame"] for s in manifest["snapshots"]]
        self.assertEqual(frames, sorted(frames))  # _nearest_snapshot ordering
        frontier_entries = [s for s in manifest["snapshots"] if s.get("frontier")]
        self.assertTrue(frontier_entries)
        self.assertGreaterEqual(max(s["x_pos"] for s in frontier_entries), 1140)
        self.assertEqual(manifest["failure"]["frontier_x"], 1200)

    def test_pins_reset_when_act_changes(self):
        ring = FailureSnapshotRing(interval=60, capacity=3, frontier_capacity=2)
        act1 = [(f * 60, f * 500, 0, 0) for f in range(1, 5)]   # act 0 to x=2000
        act2 = [(300 + f * 60, f * 30, 0, 1) for f in range(1, 4)]  # act 1 fresh
        record_run(ring, act1 + act2)

        for snapshot in ring.frontier_snapshots:
            self.assertEqual(snapshot["info"]["act"], 1)  # act-0 pins dropped


class StaticPolicy:
    def get_action(self, state):
        return "RIGHT"


class NoVisionMutator:
    def analyze_environment(self, screenshot_path):
        return "UNKNOWN"


class DeathRespawnEnv:
    """x climbs to 1000 (3 lives), dies (2 lives), respawns at x=10 and stalls."""

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return None

    def _x(self):
        if self.step_count < 100:
            return 10 * self.step_count  # to x=1000
        return 10  # post-respawn stall

    def get_state(self):
        lives = 3 if self.step_count < 100 else 2
        return {"x_pos": self._x(), "y_pos": 100, "zone": 0, "act": 1,
                "rings": 0, "score": 0, "lives": lives}

    def step(self, action):
        self.step_count += 1
        return None, 0, False, {}

    def get_screenshot(self, filepath=None):
        return "shot.png"


class DeathBehindFrontierClassificationTests(unittest.TestCase):
    def test_death_then_respawn_reports_frontier_not_respawn_x(self):
        env = DeathRespawnEnv()
        with redirect_stdout(StringIO()):
            _, _, max_x, reason, _, _, components = evaluate_policy(
                env, StaticPolicy(), NoVisionMutator(), max_frames=2000, verbose=False,
            )

        self.assertEqual(max_x, 990)
        self.assertIn("lost a life at the frontier", reason)
        self.assertIn("x=990", reason)          # the REAL frontier
        self.assertNotIn("got stuck", reason)   # not misreported as a stall
        self.assertEqual(components["frontier"], {"zone": 0, "act": 1, "x": 990})

    def test_true_stall_still_reports_stuck(self):
        class StallEnv(DeathRespawnEnv):
            def _x(self):
                return min(10 * self.step_count, 500)  # stalls AT its max

            def get_state(self):
                state = super().get_state()
                state["lives"] = 3  # no death
                return state

        with redirect_stdout(StringIO()):
            _, _, _, reason, _, _, components = evaluate_policy(
                StallEnv(), StaticPolicy(), NoVisionMutator(), max_frames=2000, verbose=False,
            )

        self.assertIn("got stuck", reason)
        self.assertEqual(components["frontier"]["x"], 500)


if __name__ == "__main__":
    unittest.main()
