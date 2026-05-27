import unittest

from core.evaluator import calculate_fitness


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


if __name__ == "__main__":
    unittest.main()
