import json
import random
import tempfile
import unittest
from pathlib import Path

from core.population import (
    PopulationArchive,
    behavior_descriptor,
    build_obstacle_key,
    p_ucb_score,
)


def policy_returning(action):
    return f'def get_action(state):\n    return "{action}"\n'


class PopulationHelpersTests(unittest.TestCase):
    def test_build_obstacle_key_clusters_failure_by_level_position_and_type(self):
        trace = [{"zone": 0, "act": 1, "x": 1077, "action": "RIGHT,DOWN"}]

        key = build_obstacle_key("Sonic got stuck: stopped making forward progress", trace)

        self.assertEqual(key, "zone-0-act-1-x-1000-stuck")

    def test_behavior_descriptor_uses_distinct_literal_actions(self):
        code = """
def get_action(state):
    if state["rings"]:
        return "RIGHT,B"
    return "RIGHT"
"""
        self.assertEqual(behavior_descriptor(code), "RIGHT|RIGHT,B")

    def test_p_ucb_score_rewards_an_underexplored_equal_fitness_candidate(self):
        underexplored = p_ucb_score(normalized_fitness=0.5, visits=0, total_visits=20)
        overused = p_ucb_score(normalized_fitness=0.5, visits=10, total_visits=20)

        self.assertGreater(underexplored, overused)


class PopulationArchiveTests(unittest.TestCase):
    def test_duplicate_policy_updates_one_record_and_keeps_best_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            code = policy_returning("RIGHT")

            archive.record_evaluation(
                code,
                fitness=100.0,
                components={"distance": 100},
                failure_reason="stuck",
                trace=[{"zone": 0, "act": 0, "x": 300}],
                reasoning="first",
            )
            archive.record_evaluation(
                code,
                fitness=90.0,
                components={"distance": 90},
                failure_reason="fatal",
                trace=[{"zone": 0, "act": 0, "x": 500}],
                reasoning="second",
            )

            records = archive.load_records()
            metadata = json.loads((Path(tmp) / "index.json").read_text())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["evaluations"], 2)
        self.assertEqual(records[0]["fitness"], 100.0)
        self.assertEqual(records[0]["reasoning"], "first")
        self.assertEqual(len(metadata["candidates"]), 1)

    def test_elite_candidates_preserve_behavior_and_obstacle_specialists(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            archive.record_evaluation(
                policy_returning("RIGHT"),
                fitness=300.0,
                failure_reason="stuck",
                trace=[{"zone": 0, "act": 0, "x": 100}],
            )
            archive.record_evaluation(
                policy_returning("RIGHT,B"),
                fitness=100.0,
                failure_reason="fatal",
                trace=[{"zone": 0, "act": 1, "x": 1100}],
            )
            archive.record_evaluation(
                policy_returning("RIGHT,DOWN"),
                fitness=90.0,
                failure_reason="stuck",
                trace=[{"zone": 0, "act": 2, "x": 2100}],
            )
            archive.record_evaluation(
                policy_returning("LEFT"),
                fitness=80.0,
                failure_reason="stuck",
                trace=[{"zone": 0, "act": 0, "x": 100}],
            )

            elite = archive.elite_candidates(limit=3)

        descriptors = {record["behavior_descriptor"] for record in elite}
        obstacles = {record["obstacle_key"] for record in elite}
        self.assertIn("RIGHT,B", descriptors)
        self.assertIn("RIGHT,DOWN", descriptors)
        self.assertIn("zone-0-act-1-x-1000-fatal", obstacles)
        self.assertIn("zone-0-act-2-x-2000-stuck", obstacles)

    def test_select_parent_codes_returns_distinct_policies_and_tracks_visits(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            archive.record_evaluation(policy_returning("RIGHT"), fitness=300.0)
            archive.record_evaluation(policy_returning("RIGHT,B"), fitness=200.0)
            archive.record_evaluation(policy_returning("RIGHT,DOWN"), fitness=100.0)

            parent_a, parent_b = archive.select_parent_codes(rng=random.Random(7))
            records = archive.load_records()

        self.assertNotEqual(parent_a, parent_b)
        self.assertEqual(sum(record["selection_visits"] for record in records), 2)

    def test_select_parent_codes_skips_invalid_and_missing_elites(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            invalid = archive.record_evaluation("not valid python", fitness=400.0)
            missing = archive.record_evaluation(policy_returning("LEFT"), fitness=300.0)
            valid_a = archive.record_evaluation(policy_returning("RIGHT"), fitness=200.0)
            valid_b = archive.record_evaluation(policy_returning("RIGHT,B"), fitness=100.0)
            (Path(tmp) / missing["code_path"]).unlink()

            parents = archive.select_parent_codes(rng=random.Random(7), elite_limit=2)
            records = archive.load_records()

        self.assertIsNotNone(parents)
        self.assertEqual(set(parents), {policy_returning("RIGHT"), policy_returning("RIGHT,B")})
        visits = {record["policy_id"]: record["selection_visits"] for record in records}
        self.assertEqual(visits[invalid["policy_id"]], 0)
        self.assertEqual(visits[missing["policy_id"]], 0)
        self.assertEqual(visits[valid_a["policy_id"]], 1)
        self.assertEqual(visits[valid_b["policy_id"]], 1)

    def test_index_stays_slim_and_details_hold_full_context(self):
        # The index is rewritten on every evaluation, so traces and unbounded
        # text must never accumulate in it.
        long_trace = [{"zone": 0, "act": 0, "x": i, "action": "RIGHT"} for i in range(50)]
        long_reasoning = "why " * 500
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            record = archive.record_evaluation(
                policy_returning("RIGHT"),
                fitness=100.0,
                components={"distance": 100},
                failure_reason="stuck",
                trace=long_trace,
                reasoning=long_reasoning,
            )

            index = json.loads((Path(tmp) / "index.json").read_text())
            indexed = index["candidates"][0]
            details = archive.load_details(record["policy_id"])

        self.assertNotIn("trace", indexed)
        self.assertNotIn("components", indexed)
        self.assertLessEqual(len(indexed["reasoning"]), 300)
        self.assertEqual(details["trace"], long_trace)
        self.assertEqual(details["reasoning"], long_reasoning)
        self.assertEqual(details["components"], {"distance": 100})

    def test_legacy_records_with_inline_traces_migrate_to_details_on_save(self):
        legacy_trace = [{"zone": 0, "act": 1, "x": 1077}]
        with tempfile.TemporaryDirectory() as tmp:
            legacy_record = {
                "policy_id": "legacy0000000000",
                "code_hash": "0" * 64,
                "code_path": "policies/legacy0000000000.py",
                "fitness": 50.0,
                "components": {"distance": 50},
                "failure_reason": "stuck",
                "trace": legacy_trace,
                "reasoning": "legacy reasoning",
                "behavior_descriptor": "RIGHT",
                "obstacle_key": "zone-0-act-1-x-1000-stuck",
                "evaluations": 1,
                "selection_visits": 0,
            }
            (Path(tmp)).mkdir(exist_ok=True)
            (Path(tmp) / "index.json").write_text(
                json.dumps({"candidates": [legacy_record]}), encoding="utf-8"
            )

            archive = PopulationArchive(tmp)
            archive.record_evaluation(policy_returning("RIGHT,B"), fitness=10.0)

            index = json.loads((Path(tmp) / "index.json").read_text())
            migrated = index["candidates"][0]
            details = archive.load_details("legacy0000000000")

        self.assertNotIn("trace", migrated)
        self.assertNotIn("components", migrated)
        self.assertEqual(migrated["fitness"], 50.0)
        self.assertEqual(details["trace"], legacy_trace)
        self.assertEqual(details["components"], {"distance": 50})
        self.assertEqual(details["reasoning"], "legacy reasoning")

    def test_select_parent_codes_floors_out_degenerate_ancestors(self):
        # Live runs: the exploration bonus kept resurrecting ~2k-fitness
        # ancestors against a 54k champion; their offspring died instantly.
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            archive.record_evaluation(policy_returning("RIGHT"), fitness=54000.0)
            archive.record_evaluation(policy_returning("RIGHT,B"), fitness=50000.0)
            degenerate = archive.record_evaluation(policy_returning("LEFT"), fitness=2000.0)

            for seed in range(10):
                parents = archive.select_parent_codes(rng=random.Random(seed))
                self.assertIsNotNone(parents)
                self.assertNotIn(policy_returning("LEFT"), parents)

            records = archive.load_records()
            visits = {record["policy_id"]: record["selection_visits"] for record in records}
            self.assertEqual(visits[degenerate["policy_id"]], 0)

    def test_select_parent_codes_does_not_record_visits_without_two_valid_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = PopulationArchive(tmp)
            archive.record_evaluation("not valid python", fitness=300.0)
            archive.record_evaluation(policy_returning("RIGHT"), fitness=200.0)

            parents = archive.select_parent_codes(rng=random.Random(7))
            records = archive.load_records()

        self.assertIsNone(parents)
        self.assertEqual(sum(record["selection_visits"] for record in records), 0)


if __name__ == "__main__":
    unittest.main()
