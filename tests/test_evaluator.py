import unittest

from core.evaluator import (
    COMPLETION_BONUS,
    DISTANCE_WEIGHT,
    FITNESS_FORMULA,
    LEVEL_CLEARED_BONUS,
    RING_WEIGHT,
    SCORE_WEIGHT,
    SPEED_WEIGHT,
    calculate_fitness,
)


class MultiLevelFitnessTests(unittest.TestCase):
    def test_defaults_match_single_level_behaviour(self):
        # No levels_cleared / cumulative_distance -> original computation.
        _, components = calculate_fitness(x_max=3000, frames_alive=1500, rings=2, score=100)
        self.assertEqual(components["levels_cleared"], 0)
        self.assertEqual(components["levels"], 0)
        self.assertEqual(components["distance"], 3000 * DISTANCE_WEIGHT)

    def test_clearing_a_level_beats_nearly_finishing_one(self):
        cleared, c1 = calculate_fitness(
            x_max=200, frames_alive=4000, rings=0, score=0,
            levels_cleared=1, cumulative_distance=9700,
        )
        almost, _ = calculate_fitness(x_max=9699, frames_alive=4000, rings=0, score=0)
        self.assertGreater(cleared, almost)
        self.assertEqual(c1["levels_cleared"], 1)
        self.assertEqual(c1["levels"], LEVEL_CLEARED_BONUS)

    def test_cumulative_distance_counts_toward_distance_and_speed(self):
        _, banked = calculate_fitness(
            x_max=100, frames_alive=2000, rings=0, score=0,
            levels_cleared=1, cumulative_distance=9700,
        )
        _, fresh = calculate_fitness(x_max=100, frames_alive=2000, rings=0, score=0)
        self.assertGreater(banked["distance"], fresh["distance"])
        self.assertGreater(banked["speed"], fresh["speed"])

    def test_more_levels_cleared_dominates_even_with_max_secondary_rewards(self):
        two_levels, _ = calculate_fitness(
            x_max=0, frames_alive=8000, rings=0, score=0,
            levels_cleared=2, cumulative_distance=19400,
        )
        one_level_loaded, _ = calculate_fitness(
            x_max=9700, frames_alive=8000, rings=999, score=999999,
            levels_cleared=1, cumulative_distance=9700,
        )
        self.assertGreater(two_levels, one_level_loaded)


class FitnessFormulaSyncTests(unittest.TestCase):
    def test_components_use_the_named_weights(self):
        # A concrete example whose every term is non-zero, checked against the
        # weight constants so the formula and the calculation cannot drift apart.
        _, components = calculate_fitness(x_max=9700, frames_alive=2000, rings=5, score=300)
        self.assertEqual(components["distance"], 9700 * DISTANCE_WEIGHT)
        self.assertEqual(components["speed"], (9700 / 2000) * SPEED_WEIGHT)
        self.assertEqual(components["rings"], 5 * RING_WEIGHT)
        self.assertEqual(components["score"], 300 * SCORE_WEIGHT)
        self.assertEqual(components["completion"], COMPLETION_BONUS)

    def test_displayed_formula_mentions_each_weight(self):
        for weight in (DISTANCE_WEIGHT, SPEED_WEIGHT, RING_WEIGHT, SCORE_WEIGHT, COMPLETION_BONUS):
            self.assertIn(f"{weight:g}", FITNESS_FORMULA)


class FitnessTests(unittest.TestCase):
    def test_equal_distance_rewards_fewer_frames(self):
        slow, slow_components = calculate_fitness(
            x_max=5000,
            frames_alive=2500,
            rings=0,
            score=0,
        )
        fast, fast_components = calculate_fitness(
            x_max=5000,
            frames_alive=1250,
            rings=0,
            score=0,
        )

        self.assertGreater(fast, slow)
        self.assertGreater(fast_components["speed"], slow_components["speed"])

    def test_speed_beats_rings_and_game_score_at_same_distance(self):
        slow_collector, _ = calculate_fitness(
            x_max=3000,
            frames_alive=3000,
            rings=100,
            score=1000,
        )
        fast_runner, _ = calculate_fitness(
            x_max=3000,
            frames_alive=1500,
            rings=0,
            score=0,
        )

        self.assertGreater(fast_runner, slow_collector)

    def test_distance_still_dominates_small_speed_advantage(self):
        farther_slow, _ = calculate_fitness(
            x_max=4500,
            frames_alive=3000,
            rings=0,
            score=0,
        )
        shorter_fast, _ = calculate_fitness(
            x_max=3000,
            frames_alive=1000,
            rings=0,
            score=0,
        )

        self.assertGreater(farther_slow, shorter_fast)

    def test_completion_threshold_adds_bonus_component(self):
        _, incomplete_components = calculate_fitness(
            x_max=9699,
            frames_alive=3000,
            rings=0,
            score=0,
        )
        complete, complete_components = calculate_fitness(
            x_max=9700,
            frames_alive=3000,
            rings=0,
            score=0,
        )

        self.assertEqual(incomplete_components["completion"], 0)
        self.assertGreater(complete_components["completion"], 0)
        self.assertGreater(complete, 9700 * 2)

    def test_completion_threshold_can_be_state_specific(self):
        _, incomplete_components = calculate_fitness(
            x_max=499,
            frames_alive=300,
            rings=0,
            score=0,
            completion_x=500,
        )
        complete, complete_components = calculate_fitness(
            x_max=500,
            frames_alive=300,
            rings=0,
            score=0,
            completion_x=500,
        )

        self.assertEqual(incomplete_components["completion"], 0)
        self.assertGreater(complete_components["completion"], 0)
        self.assertGreater(complete, 500 * 2)


if __name__ == "__main__":
    unittest.main()
