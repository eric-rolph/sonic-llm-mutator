import unittest

from main import promotion_confirmed, retest_candidate_fitness


class PromotionConfirmedTests(unittest.TestCase):
    def test_confirmed_only_when_both_runs_beat_the_bar(self):
        self.assertTrue(promotion_confirmed(60.0, 55.0, 50.0))
        self.assertTrue(promotion_confirmed(60.0, 60.0, 50.0))

    def test_rejected_when_retest_regresses_to_or_below_bar(self):
        # The fluke case we are guarding against: high original, retest at/below.
        self.assertFalse(promotion_confirmed(60.0, 50.0, 50.0))
        self.assertFalse(promotion_confirmed(60.0, 45.0, 50.0))

    def test_rejected_when_original_did_not_beat_bar(self):
        self.assertFalse(promotion_confirmed(50.0, 60.0, 50.0))


class RetestCandidateFitnessTests(unittest.TestCase):
    def test_returns_none_when_policy_cannot_load(self):
        # Missing file -> load fails -> None, so the caller keeps the original
        # single-eval decision rather than crashing the run.
        self.assertIsNone(
            retest_candidate_fitness(object(), None, "does/not/exist.py", 10, 1)
        )


if __name__ == "__main__":
    unittest.main()
