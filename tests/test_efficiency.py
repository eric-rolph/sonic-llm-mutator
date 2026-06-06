import math
import unittest

from core.efficiency import (
    amortized_crossover_runs,
    evolved_runtime_profile,
    format_comparison_table,
    inloop_projection,
)


class EfficiencyModelTests(unittest.TestCase):
    def test_evolved_runtime_has_no_api_cost(self):
        p = evolved_runtime_profile(frames=3285, wall_clock_s=5.0)
        self.assertEqual(p.api_calls, 0)
        self.assertEqual(p.usd_cost, 0.0)
        self.assertAlmostEqual(p.fps, 657.0)
        self.assertEqual(p.frames_per_usd, float("inf"))  # zero cost -> infinite frames/$

    def test_inloop_projection_counts_one_call_per_cadence(self):
        p = inloop_projection(frames=3285, frames_per_decision=12, usd_per_call=0.003, latency_s_per_call=1.5)
        self.assertEqual(p.api_calls, math.ceil(3285 / 12))  # 274
        self.assertAlmostEqual(p.usd_cost, 274 * 0.003)
        self.assertAlmostEqual(p.wall_clock_s, 274 * 1.5)
        self.assertLess(p.frames_per_usd, float("inf"))

    def test_inloop_emulator_baseline_is_added_to_wall_clock(self):
        p = inloop_projection(100, 10, 0.01, 2.0, emulator_wall_clock_s=5.0)
        self.assertEqual(p.api_calls, 10)
        self.assertAlmostEqual(p.wall_clock_s, 5.0 + 10 * 2.0)

    def test_amortization_crossover(self):
        # Free (local-first) training -> cheaper immediately.
        self.assertEqual(amortized_crossover_runs(0.0, 0.82), 0)
        # Paid training -> break even after ceil(T / per-run saving).
        self.assertEqual(amortized_crossover_runs(10.0, 0.82), math.ceil(10.0 / 0.82))
        # In-loop not more expensive -> never breaks even.
        self.assertIsNone(amortized_crossover_runs(5.0, 0.0))

    def test_comparison_table_renders_rows(self):
        rows = [
            evolved_runtime_profile(3285, 5.0),
            inloop_projection(3285, 12, 0.003, 1.5),
        ]
        table = format_comparison_table(rows)
        self.assertIn("scenario", table)
        self.assertIn("frames_per_usd", table)
        self.assertIn("inf", table)  # evolved runtime = infinite frames per dollar
        self.assertIn("evolved policy", table)
        self.assertIn("in-loop VLM", table)


if __name__ == "__main__":
    unittest.main()
