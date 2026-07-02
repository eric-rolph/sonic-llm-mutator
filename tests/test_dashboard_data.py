"""Pure dashboard data-shaping helpers (agency design-panel review fixes)."""

import unittest

from core.dashboard_data import (
    beaten_acts,
    champion_entry,
    chart_caption,
    chart_series,
    diagnosis_freshness,
    frontier_summary,
    is_new_champion,
    learned_moves,
    run_liveness,
    stagnation_status,
    video_caption,
    zone_act_label,
)


class ZoneActLabelTests(unittest.TestCase):
    def test_names_zones_and_one_based_acts(self):
        self.assertEqual(zone_act_label(0, 1), "Green Hill Act 2")
        self.assertEqual(zone_act_label(3, 0), "Labyrinth Act 1")
        self.assertEqual(zone_act_label(9, 0), "Zone 9 Act 1")
        self.assertEqual(zone_act_label(None, "?"), "Unknown act")


class LivenessTests(unittest.TestCase):
    def test_fresh_run_is_live(self):
        result = run_liveness({"timestamp": 1000}, now=1030)
        self.assertFalse(result["stale"])
        self.assertIn("moments ago", result["text"])

    def test_crashed_run_is_flagged_stale(self):
        # The review's walkthrough: a run dead for an hour looked identical to
        # a live one, and the sidebar said "steady progress" forever.
        result = run_liveness({"timestamp": 1000}, now=1000 + 3600)
        self.assertTrue(result["stale"])
        self.assertEqual(result["minutes"], 60)

    def test_missing_data_is_stale(self):
        self.assertTrue(run_liveness(None)["stale"])


class StagnationStatusTests(unittest.TestCase):
    def test_no_fake_denominator_and_graded_levels(self):
        # The old widget hardcoded "/5" and showed impossible values like 14/5.
        for count, level in ((0, "success"), (3, "info"), (7, "warning"), (14, "error")):
            status = stagnation_status(count)
            self.assertEqual(status["level"], level, count)
            self.assertNotIn("/5", status["text"])

    def test_junk_counts_are_safe(self):
        self.assertEqual(stagnation_status(None)["level"], "success")
        self.assertEqual(stagnation_status("14")["level"], "error")


class FrontierSummaryTests(unittest.TestCase):
    COMPONENTS = {
        "frontier": {"zone": 0, "act": 1, "x": 4929},
        "completion_target": 9700,
        "levels_cleared": 1,
    }

    def test_surfaces_the_core_question(self):
        summary = frontier_summary(self.COMPONENTS)
        self.assertEqual(summary["label"], "Green Hill Act 2")
        self.assertEqual(summary["x"], 4929)
        self.assertAlmostEqual(summary["progress"], 4929 / 9700)

    def test_legacy_entries_without_frontier_are_none(self):
        self.assertIsNone(frontier_summary({}))
        self.assertIsNone(frontier_summary(None))
        self.assertIsNone(frontier_summary({"frontier": {"zone": 0}}))

    def test_beaten_acts_names_the_trophies(self):
        self.assertEqual(beaten_acts(self.COMPONENTS), ["Green Hill Act 1"])
        self.assertEqual(beaten_acts({}), [])


class ChampionAndChartTests(unittest.TestCase):
    HISTORY = [
        {"generation": 1, "fitness": 100.0, "timestamp": 10, "stagnation_counter": 0},
        {"generation": 2, "fitness": 300.0, "timestamp": 20, "stagnation_counter": 0},
        {"generation": 3, "fitness": 200.0, "timestamp": 30, "stagnation_counter": 1},
    ]

    def test_champion_entry_and_new_champion_flag(self):
        self.assertEqual(champion_entry(self.HISTORY)["generation"], 2)
        self.assertFalse(is_new_champion(self.HISTORY))       # latest didn't set the record
        self.assertTrue(is_new_champion(self.HISTORY[:2]))    # gen 2 did
        self.assertFalse(is_new_champion([]))

    def test_chart_series_includes_champion_staircase(self):
        series = chart_series(self.HISTORY)
        self.assertEqual(series["attempt"], [100.0, 300.0, 200.0])
        self.assertEqual(series["champion"], [100.0, 300.0, 300.0])  # cumulative max

    def test_windowed_series_keeps_prewindow_records(self):
        # The staircase must not forget a record set before the window.
        series = chart_series(self.HISTORY, window=1)
        self.assertEqual(series["attempt"], [200.0])
        self.assertEqual(series["champion"], [300.0])

    def test_chart_caption_is_a_text_alternative(self):
        caption = chart_caption(self.HISTORY)
        self.assertIn("300", caption)
        self.assertIn("200", caption)
        self.assertEqual(chart_caption([]), "No generations recorded yet.")


class LearnedMovesTests(unittest.TestCase):
    def test_parses_guard_markers_into_human_labels(self):
        code = (
            "def get_action(state):\n"
            "    # FRONTIER_GUARD zone=0 act=1 x=1077\n"
            "    # DIAGNOSIS_GUARD zone=0 act=1 x=3928\n"
            "    # LLM_GUARD zone=0 act=1 x=4268\n"
            "    return 'RIGHT'\n"
        )
        moves = learned_moves(code)
        self.assertEqual(len(moves), 3)
        self.assertEqual(moves[1]["label"], "verified escape at Green Hill Act 2 x=3928")

    def test_no_guards_no_moves(self):
        self.assertEqual(learned_moves("def get_action(state):\n    return 'RIGHT'\n"), [])
        self.assertEqual(learned_moves(""), [])


class CaptionTests(unittest.TestCase):
    def test_video_caption_carries_identity(self):
        caption = video_caption("Champion", {"generation": 590, "fitness": 55746.24})
        self.assertIn("590", caption)
        self.assertIn("55,746", caption)
        self.assertIn("no run recorded", video_caption("Champion", None))

    def test_diagnosis_freshness(self):
        self.assertIn("minute", diagnosis_freshness({"created_at": 0}, now=600))
        self.assertIsNone(diagnosis_freshness({}, now=600))


if __name__ == "__main__":
    unittest.main()
