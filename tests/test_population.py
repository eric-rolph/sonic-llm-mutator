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


if __name__ == "__main__":
    unittest.main()
