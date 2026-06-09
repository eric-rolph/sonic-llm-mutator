import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from llm.mutator import (
    MAX_SEMANTIC_LESSONS,
    MutatorClient,
    concise_vision_label,
    dedupe_lessons,
    extract_json_object,
    hazard_category,
    message_text,
    normalize_lesson,
    normalize_vision_context,
    select_relevant_lessons,
    vision_location_key,
)


class _Msg:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class ReasoningModelExtractionTests(unittest.TestCase):
    def test_message_text_prefers_content(self):
        self.assertEqual(message_text(_Msg(content="hello", reasoning_content="ignored")), "hello")

    def test_message_text_falls_back_to_reasoning_when_content_empty(self):
        # Reasoning models (gemma, qwen3) leave content empty and fill reasoning.
        self.assertEqual(message_text(_Msg(content="", reasoning_content="thought")), "thought")
        self.assertEqual(message_text(_Msg(content=None, reasoning_content="r")), "r")

    def test_message_text_empty_when_both_missing(self):
        self.assertEqual(message_text(_Msg(content=None, reasoning_content=None)), "")

    def test_concise_vision_label_drops_connectors_and_uppercases(self):
        self.assertEqual(concise_vision_label("...spikes or enemies"), "SPIKES ENEMIES")
        self.assertEqual(concise_vision_label("The context is clear"), "CLEAR")
        self.assertEqual(concise_vision_label(""), "UNKNOWN")


class VisionCacheTests(unittest.TestCase):
    def make_client(self, cache_path):
        client = object.__new__(MutatorClient)
        client.vision_cache_path = cache_path
        client._vision_cache = None
        return client

    def test_vision_location_key_buckets_camera_position(self):
        state = {"zone": 0, "act": 1, "screen_x": 1337}
        self.assertEqual(vision_location_key(state), "zone-0-act-1-sx-1250")
        # Same bucket -> same key; next bucket -> different key.
        self.assertEqual(
            vision_location_key({"zone": 0, "act": 1, "screen_x": 1499}),
            "zone-0-act-1-sx-1250",
        )
        self.assertNotEqual(
            vision_location_key({"zone": 0, "act": 1, "screen_x": 1500}),
            vision_location_key(state),
        )
        self.assertIsNone(vision_location_key({"zone": "??"}))

    def test_store_and_lookup_round_trip_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "vision_cache.json")
            writer = self.make_client(cache_path)
            writer.store_vision_context("zone-0-act-1-sx-1250", "SPIKES")

            reader = self.make_client(cache_path)
            self.assertEqual(reader.cached_vision_context("zone-0-act-1-sx-1250"), "SPIKES")
            self.assertIsNone(reader.cached_vision_context("zone-0-act-1-sx-1500"))

    def test_unknown_labels_never_poison_a_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "vision_cache.json")
            client = self.make_client(cache_path)
            client.store_vision_context("key", "UNKNOWN")
            client.store_vision_context("key", "")
            client.store_vision_context(None, "SPIKES")
            self.assertIsNone(client.cached_vision_context("key"))
            self.assertFalse(os.path.exists(cache_path))


class MutatorMemoryTests(unittest.TestCase):
    def test_extract_json_object_handles_fenced_json(self):
        text = """Here is the lesson:
```json
{"x": 9767, "hazard": "Pit", "lesson": "When at X=9767, jump."}
```
"""

        self.assertEqual(
            extract_json_object(text),
            {"x": 9767, "hazard": "Pit", "lesson": "When at X=9767, jump."},
        )

    def test_normalize_lesson_rejects_malformed_entries(self):
        self.assertIsNone(normalize_lesson({"x": "oops", "lesson": "jump"}))
        self.assertIsNone(normalize_lesson({"x": 100, "hazard": "Pit"}))
        self.assertEqual(
            normalize_lesson({"x": "287", "hazard": "Pit", "lesson": "Jump"}),
            {"x": 287, "hazard": "Pit", "lesson": "Jump"},
        )

    def test_dedupe_lessons_collapses_nearby_duplicate_lessons(self):
        lessons = [
            {"x": 9767, "hazard": "Pit", "lesson": "Jump to avoid pit"},
            {"x": 9768, "hazard": "Pit", "lesson": "Jump to avoid pit"},
            {"x": 3061, "hazard": "Stuck", "lesson": "Hold RIGHT"},
        ]

        deduped = dedupe_lessons(lessons, x_tolerance=5)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["x"], 9767)
        self.assertEqual(deduped[1]["x"], 3061)

    def test_select_relevant_lessons_filters_by_coordinate_and_limit(self):
        lessons = [
            {"x": 100, "hazard": "Pit", "lesson": "Jump at 100"},
            {"x": 200, "hazard": "Pit", "lesson": "Jump at 200"},
            {"x": 2000, "hazard": "Wall", "lesson": "Do not include"},
        ]

        relevant = select_relevant_lessons(lessons, current_x=150, radius=1000, limit=1)

        self.assertEqual(relevant, ["- Jump at 200"])

    def test_hazard_category_collapses_freeform_names(self):
        self.assertEqual(hazard_category("Wall/Ledge"), "wall")
        self.assertEqual(hazard_category("Vertical stuck loop"), "wall")
        self.assertEqual(hazard_category("Pitfall"), "pit")
        self.assertEqual(hazard_category("Spikes"), "spikes")
        # Falls back to the lesson text when the hazard name is unhelpful.
        self.assertEqual(hazard_category("Unknown", "jump to avoid the pit"), "pit")
        self.assertEqual(hazard_category("Unknown", "no keywords here"), "other")

    def test_dedupe_lessons_collapses_paraphrased_lessons_about_same_obstacle(self):
        # Paraphrases of the same wall at the same spot must not accumulate.
        lessons = [
            {"x": 3061, "hazard": "Wall/Obstacle", "lesson": "Stop spamming RIGHT,B and try a pure jump."},
            {"x": 3060, "hazard": "Wall/Ledge", "lesson": "Increase jump frequency to overcome the wall."},
            {"x": 3062, "hazard": "Vertical stuck loop", "lesson": "Vary the input sequence at the wall."},
        ]

        deduped = dedupe_lessons(lessons, x_tolerance=25)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["x"], 3061)

    def test_dedupe_lessons_keeps_same_x_in_different_acts(self):
        lessons = [
            {"x": 3061, "zone": 0, "act": 0, "hazard": "Wall", "lesson": "Jump the Act 1 wall."},
            {"x": 3061, "zone": 0, "act": 1, "hazard": "Wall", "lesson": "Jump the Act 2 wall."},
            {"x": 3061, "hazard": "Wall", "lesson": "Legacy untagged wall lesson."},
        ]

        deduped = dedupe_lessons(lessons, x_tolerance=25)

        self.assertEqual(len(deduped), 3)

    def test_normalize_lesson_keeps_zone_and_act_when_parseable(self):
        self.assertEqual(
            normalize_lesson({"x": "287", "zone": "0", "act": 1.0, "hazard": "Pit", "lesson": "Jump"}),
            {"x": 287, "zone": 0, "act": 1, "hazard": "Pit", "lesson": "Jump"},
        )
        self.assertEqual(
            normalize_lesson({"x": 287, "zone": "??", "hazard": "Pit", "lesson": "Jump"}),
            {"x": 287, "hazard": "Pit", "lesson": "Jump"},
        )

    def test_select_relevant_lessons_scopes_tagged_lessons_to_their_act(self):
        lessons = [
            {"x": 1000, "zone": 0, "act": 0, "hazard": "Pit", "lesson": "Act 1 pit"},
            {"x": 1000, "zone": 0, "act": 1, "hazard": "Spikes", "lesson": "Act 2 spikes"},
            {"x": 1000, "hazard": "Wall", "lesson": "Legacy lesson applies anywhere"},
        ]

        relevant = select_relevant_lessons(lessons, current_x=1000, zone=0, act=1)

        self.assertEqual(
            relevant,
            ["- [zone 0 act 1] Act 2 spikes", "- Legacy lesson applies anywhere"],
        )

    def test_select_relevant_lessons_without_act_keeps_legacy_behaviour(self):
        lessons = [
            {"x": 1000, "zone": 0, "act": 0, "hazard": "Pit", "lesson": "Act 1 pit"},
            {"x": 1000, "hazard": "Wall", "lesson": "Legacy lesson"},
        ]

        relevant = select_relevant_lessons(lessons, current_x=1000)

        self.assertEqual(len(relevant), 2)

    def test_extract_lesson_stamps_zone_act_from_trace_and_caps_bank(self):
        class LessonMutator(MutatorClient):
            def __init__(self):
                pass

            def _call_micro_model(self, prompt, temperature=0.7):
                return '{"x": 4242, "hazard": "Pit", "lesson": "Jump at 4242"}', "local"

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("memory")
                prefill = [
                    {"x": i * 1000, "hazard": f"Hazard {i}", "lesson": f"Unique lesson {i}"}
                    for i in range(MAX_SEMANTIC_LESSONS + 10)
                ]
                with open("memory/semantic_bank.json", "w", encoding="utf-8") as f:
                    json.dump(prefill, f)

                trace = [{"frame": 30, "x": 4200, "zone": 2, "act": 1}]
                with redirect_stdout(StringIO()):
                    LessonMutator().extract_lesson("Sonic got stuck", trace)

                with open("memory/semantic_bank.json", "r", encoding="utf-8") as f:
                    bank = json.load(f)

                self.assertLessEqual(len(bank), MAX_SEMANTIC_LESSONS)
                newest = bank[-1]
                self.assertEqual(newest["x"], 4242)
                self.assertEqual(newest["zone"], 2)
                self.assertEqual(newest["act"], 1)
            finally:
                os.chdir(previous_cwd)

    def test_normalize_vision_context_handles_empty_model_content(self):
        self.assertEqual(normalize_vision_context(None), "UNKNOWN")
        self.assertEqual(normalize_vision_context(" spikes "), "SPIKES")

    def test_mutate_policy_uses_micro_model_when_screenshot_missing(self):
        class RecordingMutator(MutatorClient):
            def __init__(self):
                self.called = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.called.append("micro")
                return "def get_action(state):\n    return 'RIGHT'", "micro"

            def _call_macro_model(self, prompt, image_path, temperature=0.7):
                self.called.append("macro")
                return "def get_action(state):\n    return 'LEFT'", "macro"

        mutator = RecordingMutator()

        with redirect_stdout(StringIO()):
            code, _ = mutator.mutate_policy(
                current_code="def get_action(state):\n    return 'RIGHT'",
                failure_reason="Sonic lost a life or hit a fatal obstacle.",
                screenshot_path=None,
                recent_history=[],
            )

        self.assertEqual(mutator.called, ["micro"])
        self.assertIn("return 'RIGHT'", code)

    def test_mutate_policy_marks_primary_frontier_and_other_candidate_history(self):
        class RecordingMutator(MutatorClient):
            def __init__(self):
                self.prompt = ""

            def _call_micro_model(self, prompt, temperature=0.7):
                self.prompt = prompt
                return "def get_action(state):\n    return 'RIGHT'", "micro"

        mutator = RecordingMutator()
        with redirect_stdout(StringIO()):
            mutator.mutate_policy(
                current_code="def get_action(state):\n    return 'RIGHT'",
                failure_reason="champion stuck in act 2",
                screenshot_path=None,
                recent_history=[{"failure_reason": "loser stuck in act 1"}],
                coordinate_trace=[{"zone": 0, "act": 1, "x": 1077}],
            )

        self.assertIn("working policy's own frontier", mutator.prompt)
        self.assertIn("Other Evaluated Candidates", mutator.prompt)
        self.assertIn("may not apply to the current code", mutator.prompt)

    def test_mutate_policy_routes_visual_failures_to_vision_and_code_faults_to_micro(self):
        class RecordingMutator(MutatorClient):
            def __init__(self):
                self.called = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.called.append("micro")
                return "code", "micro"

            def _call_macro_model(self, prompt, image_path, temperature=0.7):
                self.called.append("macro")
                return "code", "macro"

        stuck = "Sonic got stuck: stopped making forward progress for 8 seconds. (zone 0 act 1)"
        fatal = "Sonic lost a life or hit a fatal obstacle."
        timeout = "Policy code timeout (likely an infinite loop in get_action)."

        # Visual problems (stuck against geometry, killed by a hazard) need the
        # model to SEE the frame -> vision/macro, when a screenshot is available.
        for reason in (stuck, fatal):
            mutator = RecordingMutator()
            with redirect_stdout(StringIO()):
                mutator.mutate_policy("code", reason, "shot.png", [])
            self.assertEqual(mutator.called, ["macro"], reason)

        # A pure code fault (timeout / infinite loop) -> local code model.
        mutator = RecordingMutator()
        with redirect_stdout(StringIO()):
            mutator.mutate_policy("code", timeout, "shot.png", [])
        self.assertEqual(mutator.called, ["micro"])

        # No frame to look at -> fall back to the local code model regardless.
        mutator = RecordingMutator()
        with redirect_stdout(StringIO()):
            mutator.mutate_policy("code", stuck, None, [])
        self.assertEqual(mutator.called, ["micro"])

    def test_repair_policy_uses_local_model_with_exact_validator_feedback(self):
        class RecordingMutator(MutatorClient):
            def __init__(self):
                self.prompts = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.prompts.append((prompt, temperature))
                return "```python\ndef get_action(state):\n    return 'RIGHT'\n```", "fixed"

        mutator = RecordingMutator()

        with redirect_stdout(StringIO()):
            code, reasoning = mutator.repair_policy(
                "def broken(:\n    pass",
                "Policy syntax error: invalid syntax",
            )

        self.assertEqual(len(mutator.prompts), 1)
        self.assertIn("Policy syntax error: invalid syntax", mutator.prompts[0][0])
        self.assertIn("def broken(", mutator.prompts[0][0])
        self.assertEqual(mutator.prompts[0][1], 0.2)
        self.assertEqual(code, "def get_action(state):\n    return 'RIGHT'")
        self.assertEqual(reasoning, "fixed")

    def test_macro_fallback_preserves_requested_mutation_temperature(self):
        class RecordingMutator(MutatorClient):
            def __init__(self):
                self.macro_client = None
                self.temperatures = []

            def _call_micro_model(self, prompt, temperature=0.7):
                self.temperatures.append(temperature)
                return "code", "local"

        mutator = RecordingMutator()

        with redirect_stdout(StringIO()):
            mutator._call_macro_model("prompt", "shot.png", temperature=0.9)

        self.assertEqual(mutator.temperatures, [0.9])

    def test_macro_request_preserves_requested_mutation_temperature(self):
        mutator = object.__new__(MutatorClient)
        mutator.macro_client = object()
        mutator.macro_model = "vision-model"
        captured = []

        def do_macro_call(prompt, image_path, temperature):
            captured.append(temperature)
            return "code", "reasoning"

        mutator._do_macro_call = do_macro_call

        result = mutator._call_macro_model("prompt", "shot.png", temperature=0.9)

        self.assertEqual(result, ("code", "reasoning"))
        self.assertEqual(captured, [0.9])

    def test_extract_and_save_skills_rejects_unsafe_code_before_writing(self):
        class UnsafeSkillsMutator(MutatorClient):
            def __init__(self):
                pass

            def _call_micro_model(self, prompt, temperature=0.7):
                return "def unsafe(state):\n    writer = open\n    return writer('owned.txt', 'w')", "local"

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("policies")
                with open("policies/skills.py", "w", encoding="utf-8") as f:
                    f.write("def existing(state):\n    return 'RIGHT'\n")

                with redirect_stdout(StringIO()):
                    UnsafeSkillsMutator().extract_and_save_skills("def get_action(state):\n    return 'RIGHT'")

                with open("policies/skills.py", "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), "def existing(state):\n    return 'RIGHT'\n")
                self.assertFalse(os.path.exists("owned.txt"))
            finally:
                os.chdir(previous_cwd)

    def test_extract_and_save_skills_reloads_loaded_module_after_valid_update(self):
        class SafeSkillsMutator(MutatorClient):
            def __init__(self):
                pass

            def _call_micro_model(self, prompt, temperature=0.7):
                return "def updated(state):\n    return 'RIGHT,B'", "local"

        loaded_skills = types.ModuleType("policies.skills")
        previous_module = sys.modules.get("policies.skills")
        sys.modules["policies.skills"] = loaded_skills
        try:
            with tempfile.TemporaryDirectory() as tmp:
                previous_cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    os.makedirs("policies")
                    with patch("llm.mutator.importlib.reload") as reload_module:
                        with redirect_stdout(StringIO()):
                            SafeSkillsMutator().extract_and_save_skills(
                                "def get_action(state):\n    return 'RIGHT'"
                            )

                    reload_module.assert_called_once_with(loaded_skills)
                    with open("policies/skills.py", "r", encoding="utf-8") as f:
                        self.assertIn("def updated(state):", f.read())
                finally:
                    os.chdir(previous_cwd)
        finally:
            if previous_module is None:
                sys.modules.pop("policies.skills", None)
            else:
                sys.modules["policies.skills"] = previous_module

    def test_extract_and_save_skills_preserves_existing_skill_functions(self):
        class DroppingSkillsMutator(MutatorClient):
            def __init__(self):
                pass

            def _call_micro_model(self, prompt, temperature=0.7):
                return "def replacement(state):\n    return 'RIGHT'", "local"

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("policies")
                existing = "def existing(state):\n    return 'RIGHT,B'\n"
                with open("policies/skills.py", "w", encoding="utf-8") as f:
                    f.write(existing)

                with redirect_stdout(StringIO()):
                    DroppingSkillsMutator().extract_and_save_skills(
                        "def get_action(state):\n    return 'RIGHT'"
                    )

                with open("policies/skills.py", "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("def existing(state):", content)
                    self.assertIn("def replacement(state):", content)
            finally:
                os.chdir(previous_cwd)

    def test_extract_and_save_skills_does_not_change_existing_skill_behavior(self):
        class RewritingSkillsMutator(MutatorClient):
            def __init__(self):
                pass

            def _call_micro_model(self, prompt, temperature=0.7):
                return "def existing(state):\n    return 'LEFT'", "local"

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("policies")
                existing = "def existing(state):\n    return 'RIGHT,B'\n"
                with open("policies/skills.py", "w", encoding="utf-8") as f:
                    f.write(existing)

                with redirect_stdout(StringIO()):
                    RewritingSkillsMutator().extract_and_save_skills(
                        "def get_action(state):\n    return 'RIGHT'"
                    )

                with open("policies/skills.py", "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), existing)
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
