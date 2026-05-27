import unittest

import benchmark_policies


class BenchmarkPoliciesTests(unittest.TestCase):
    def test_default_states_include_primary_and_generalization_levels(self):
        self.assertEqual(benchmark_policies.DEFAULT_STATES[0], "GreenHillZone.Act1")
        self.assertIn("GreenHillZone.Act2", benchmark_policies.DEFAULT_STATES)
        self.assertIn("GreenHillZone.Act3", benchmark_policies.DEFAULT_STATES)
        self.assertIn("MarbleZone.Act1", benchmark_policies.DEFAULT_STATES)

    def test_format_results_table_contains_core_metrics(self):
        rows = [
            {
                "state": "GreenHillZone.Act1",
                "policy": "champion",
                "fitness": 12345.678,
                "max_x": 9700,
                "frames": 3000,
                "reason": "Timeout reached.",
            }
        ]

        table = benchmark_policies.format_results_table(rows)

        self.assertIn("state", table)
        self.assertIn("policy", table)
        self.assertIn("fitness", table)
        self.assertIn("max_x", table)
        self.assertIn("frames", table)
        self.assertIn("GreenHillZone.Act1", table)
        self.assertIn("champion", table)
        self.assertIn("12345.68", table)
        self.assertIn("Timeout reached.", table)

    def test_policy_label_uses_filename_stem(self):
        self.assertEqual(
            benchmark_policies.policy_label("policies/champion_policy.py"),
            "champion_policy",
        )


if __name__ == "__main__":
    unittest.main()
