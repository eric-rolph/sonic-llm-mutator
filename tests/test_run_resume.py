import os
import tempfile
import unittest
from unittest.mock import patch

import main
from main import (
    build_stagnation_escape_context,
    candidate_beats_current_best,
    candidate_is_promotable,
    choose_generation_archive_path,
    derive_resume_state,
    evaluate_working_baseline,
    persist_frontier_window,
    preserve_frontier_screenshot,
    render_candidate_video,
    resolve_end_generation,
    resolve_working_fitness_floor,
    seed_population_baseline,
    select_working_frontier_context,
)


class RecordingArchive:
    def __init__(self):
        self.calls = []

    def record_evaluation(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class NoVisionMutator:
    def analyze_environment(self, screenshot_path):
        return "UNKNOWN"


class FakeEnv:
    def __init__(self, states):
        self.states = list(states)
        self.index = 0

    def reset(self):
        self.index = 0
        return None

    def get_state(self):
        if self.index >= len(self.states):
            return dict(self.states[-1])
        state = dict(self.states[self.index])
        self.index += 1
        return state

    def step(self, action):
        return None, 0, False, {}

    def get_screenshot(self):
        return "baseline_screenshot.png"


class RunResumeTests(unittest.TestCase):
    def write_policy(self, directory, code):
        policy_path = os.path.join(directory, "policy.py")
        with open(policy_path, "w", encoding="utf-8") as f:
            f.write(code)
        return policy_path

    def test_evaluate_working_baseline_seeds_resume_context(self):
        states = [{"x_pos": 100, "y_pos": 100, "rings": 0, "score": 0}] * 700
        env = FakeEnv(states)
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = self.write_policy(
                tmp,
                "def get_action(state):\n    return 'RIGHT'\n",
            )

            context = evaluate_working_baseline(
                env,
                policy_path,
                NoVisionMutator(),
                max_frames=700,
                verbose=False,
            )

        self.assertGreater(context["working_fitness"], -1)
        self.assertEqual(context["last_screenshot"], "baseline_screenshot.png")
        self.assertIn("stopped making forward progress", context["last_failure_reason"])
        self.assertEqual(context["last_trace"][-1]["x"], 100)
        self.assertEqual(context["last_trace"][-1]["y"], 100)
        self.assertEqual(context["last_trace"][-1]["action"], "RIGHT")
        self.assertIn("distance", context["components"])

    def test_evaluate_working_baseline_handles_missing_policy(self):
        context = evaluate_working_baseline(
            env=None,
            working_path="missing.py",
            mutator=NoVisionMutator(),
            max_frames=700,
            verbose=False,
        )

        self.assertEqual(context["working_fitness"], -1.0)
        self.assertEqual(context["last_failure_reason"], "Initial seed run")
        self.assertIsNone(context["last_screenshot"])
        self.assertEqual(context["last_trace"], [])

    def test_seed_population_baseline_records_the_strong_working_policy(self):
        archive = RecordingArchive()
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = self.write_policy(tmp, "def get_action(state):\n    return 'RIGHT'\n")
            recorded = seed_population_baseline(
                archive,
                policy_path,
                {
                    "working_fitness": 47977.37,
                    "components": {"levels_cleared": 1},
                    "last_failure_reason": "stuck in act 2",
                    "last_trace": [{"zone": 0, "act": 1, "x": 1077}],
                },
            )

        self.assertTrue(recorded)
        self.assertEqual(archive.calls[0][1]["fitness"], 47977.37)
        self.assertEqual(archive.calls[0][1]["components"]["levels_cleared"], 1)

    def test_baseline_forwards_snapshot_sink_to_evaluation(self):
        class RecordingSink:
            def __init__(self):
                self.calls = 0

            def record(self, env, frame, state):
                self.calls += 1

        states = [{"x_pos": 100, "y_pos": 100, "rings": 0, "score": 0}] * 700
        env = FakeEnv(states)
        sink = RecordingSink()
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = self.write_policy(tmp, "def get_action(state):\n    return 'RIGHT'\n")

            evaluate_working_baseline(
                env, policy_path, NoVisionMutator(), max_frames=700, verbose=False, snapshot_sink=sink
            )

        self.assertGreater(sink.calls, 0)

    def test_persist_frontier_window_never_raises(self):
        self.assertIsNone(persist_frontier_window(None, "stuck"))

        class ExplodingRing:
            def persist(self, failure_reason=""):
                raise RuntimeError("disk full")

        from contextlib import redirect_stdout
        from io import StringIO

        with redirect_stdout(StringIO()):
            self.assertIsNone(persist_frontier_window(ExplodingRing(), "stuck"))

    def test_persist_frontier_window_writes_a_loadable_window(self):
        from core.diagnosis import FailureSnapshotRing, load_failure_window

        class SavestateEnv:
            def save_emulator_state(self):
                return b"state-bytes"

        ring = FailureSnapshotRing(interval=1, capacity=3)
        ring.record(SavestateEnv(), 0, {"x_pos": 10, "y_pos": 1, "zone": 0, "act": 1, "rings": 0, "lives": 3})
        with tempfile.TemporaryDirectory() as tmp:
            directory = persist_frontier_window(ring, "Sonic got stuck", directory=tmp)
            window = load_failure_window(directory)

        self.assertEqual(window["failure_reason"], "Sonic got stuck")
        self.assertEqual(len(window["snapshots"]), 1)

    def test_stagnation_escape_preserves_working_policy_context(self):
        trace = [{"zone": 0, "act": 1, "x": 1077}]

        context = build_stagnation_escape_context(47977.37, trace, "failure.png")

        self.assertEqual(context["working_fitness"], 47977.37)
        self.assertEqual(context["last_trace"], trace)
        self.assertEqual(context["last_screenshot"], "failure.png")
        self.assertIn("preserve", context["last_failure_reason"].lower())

    def test_losing_candidate_does_not_replace_working_frontier_context(self):
        working = {
            "failure_reason": "champion stuck in act 2",
            "trace": [{"zone": 0, "act": 1, "x": 1077}],
            "screenshot": "working-frontier.png",
        }
        losing_candidate = {
            "failure_reason": "candidate stuck in act 1",
            "trace": [{"zone": 0, "act": 0, "x": 3061}],
            "screenshot": "candidate.png",
        }

        selected = select_working_frontier_context(working, losing_candidate, promoted=False)

        self.assertEqual(selected, working)
        self.assertIsNot(selected, working)

    def test_promoted_candidate_replaces_working_frontier_context(self):
        working = {"failure_reason": "old", "trace": [], "screenshot": "old.png"}
        promoted = {"failure_reason": "new", "trace": [1], "screenshot": "new.png"}

        self.assertEqual(select_working_frontier_context(working, promoted, promoted=True), promoted)

    def test_frontier_screenshot_is_copied_away_from_shared_candidate_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "latest.png")
            destination = os.path.join(tmp, "working-frontier.png")
            with open(source, "wb") as f:
                f.write(b"champion")

            preserved = preserve_frontier_screenshot(source, destination)
            with open(source, "wb") as f:
                f.write(b"losing candidate")

            with open(preserved, "rb") as f:
                self.assertEqual(f.read(), b"champion")

    def test_additional_generation_limit_is_relative_to_resume_point(self):
        self.assertEqual(resolve_end_generation(start_gen=498, max_generations=500, generations=15), 512)

    def test_cli_passes_bounded_run_options_to_evaluation_loop(self):
        with patch.object(main, "run_evaluation_loop") as run:
            self.assertEqual(main.main(["--generations", "15", "--frames", "2000"]), 0)

        run.assert_called_once_with(generations=15, max_frames=2000, n_candidates=2)

    def test_cli_candidates_flag_widens_the_search(self):
        with patch.object(main, "run_evaluation_loop") as run:
            self.assertEqual(main.main(["--candidates", "3"]), 0)

        run.assert_called_once_with(generations=None, max_frames=12000, n_candidates=3)

    def test_resume_state_uses_last_generation_and_persisted_stagnation(self):
        state = derive_resume_state(
            [
                {"generation": 7, "fitness": 100.0, "stagnation_counter": 1},
                {"generation": 8, "fitness": 90.0, "stagnation_counter": 4},
            ]
        )

        self.assertEqual(state["all_time_champion_fitness"], 100.0)
        self.assertEqual(state["start_generation"], 9)
        self.assertEqual(state["stagnation_counter"], 4)

    def test_resume_state_ignores_malformed_numeric_fields(self):
        state = derive_resume_state(
            [{"generation": "bad", "fitness": "bad", "stagnation_counter": "bad"}]
        )

        self.assertEqual(state["all_time_champion_fitness"], -1.0)
        self.assertEqual(state["start_generation"], 1)
        self.assertEqual(state["stagnation_counter"], 0)

    def test_resume_state_ignores_non_finite_numeric_fields(self):
        state = derive_resume_state(
            [{"generation": float("inf"), "fitness": float("nan"), "stagnation_counter": 0}]
        )

        self.assertEqual(state["all_time_champion_fitness"], -1.0)
        self.assertEqual(state["start_generation"], 1)

    def test_empty_resume_state_uses_initial_defaults(self):
        self.assertEqual(
            derive_resume_state([]),
            {
                "all_time_champion_fitness": -1.0,
                "start_generation": 1,
                "stagnation_counter": 0,
                "champion_max_frames": None,
            },
        )

    def test_resume_state_reports_champion_frame_budget(self):
        state = derive_resume_state(
            [
                {"generation": 7, "fitness": 100.0, "max_frames": 12000},
                {"generation": 8, "fitness": 90.0, "max_frames": 2000},
            ]
        )

        self.assertEqual(state["all_time_champion_fitness"], 100.0)
        self.assertEqual(state["champion_max_frames"], 12000)

    def test_resume_state_handles_legacy_entries_without_frame_budget(self):
        state = derive_resume_state([{"generation": 7, "fitness": 100.0}])

        self.assertIsNone(state["champion_max_frames"])

    def test_fitness_floor_applies_when_budgets_match(self):
        bar = resolve_working_fitness_floor(
            baseline_fitness=80.0,
            champion_fitness=100.0,
            champion_max_frames=12000,
            current_max_frames=12000,
            verbose=False,
        )
        self.assertEqual(bar, 100.0)

    def test_fitness_floor_skipped_when_frame_budget_changed(self):
        # Resuming with a smaller --frames must not leave the bar at a
        # champion fitness no candidate can physically reach.
        bar = resolve_working_fitness_floor(
            baseline_fitness=80.0,
            champion_fitness=100.0,
            champion_max_frames=12000,
            current_max_frames=2000,
            verbose=False,
        )
        self.assertEqual(bar, 80.0)

    def test_fitness_floor_keeps_legacy_behaviour_without_recorded_budget(self):
        bar = resolve_working_fitness_floor(
            baseline_fitness=80.0,
            champion_fitness=100.0,
            champion_max_frames=None,
            current_max_frames=2000,
            verbose=False,
        )
        self.assertEqual(bar, 100.0)

    def test_dry_run_candidate_cannot_be_promoted(self):
        self.assertFalse(candidate_is_promotable(None, object(), {}))

    def test_runtime_broken_candidate_cannot_be_promoted(self):
        self.assertFalse(
            candidate_is_promotable(
                object(),
                object(),
                {"runtime_error": "RuntimeError: broken"},
            )
        )

    def test_evaluated_runtime_valid_candidate_can_be_promoted(self):
        self.assertTrue(candidate_is_promotable(object(), object(), {"distance": 10}))

    def test_generation_history_archives_best_candidate_path(self):
        self.assertEqual(
            choose_generation_archive_path("policies/candidate_1.py", "policies/working_policy.py"),
            "policies/candidate_1.py",
        )

    def test_generation_history_falls_back_to_working_path_without_candidate(self):
        self.assertEqual(
            choose_generation_archive_path(None, "policies/working_policy.py"),
            "policies/working_policy.py",
        )

    def test_render_candidate_video_waits_for_renderer_and_reports_success(self):
        completed = main.subprocess.CompletedProcess(args=[], returncode=0)
        with patch.object(main.os.path, "exists", side_effect=[True, False, True]), patch.object(
            main.subprocess, "run", return_value=completed
        ) as run:
            self.assertTrue(render_candidate_video("candidate.bk2", "latest.mp4"))

        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["check"], False)

    def test_render_candidate_video_rejects_failed_render(self):
        completed = main.subprocess.CompletedProcess(args=[], returncode=3)
        with patch.object(main.os.path, "exists", side_effect=[True, False]), patch.object(
            main.subprocess, "run", return_value=completed
        ):
            self.assertFalse(render_candidate_video("candidate.bk2", "latest.mp4"))

    def test_render_candidate_video_rejects_timed_out_render(self):
        with patch.object(main.os.path, "exists", side_effect=[True, False]), patch.object(
            main.subprocess, "run", side_effect=main.subprocess.TimeoutExpired([], 300)
        ):
            self.assertFalse(render_candidate_video("candidate.bk2", "latest.mp4"))

    def test_promotable_candidate_wins_equal_fitness_tie(self):
        self.assertTrue(candidate_beats_current_best(0.0, True, 0.0, False))
        self.assertFalse(candidate_beats_current_best(0.0, False, 0.0, True))

    def test_resume_threshold_never_drops_below_historical_champion(self):
        state = derive_resume_state([{"generation": 8, "fitness": 100.0}])
        self.assertEqual(max(80.0, state["all_time_champion_fitness"]), 100.0)


if __name__ == "__main__":
    unittest.main()
