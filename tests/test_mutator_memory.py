import unittest
from contextlib import redirect_stdout
from io import StringIO

from llm.mutator import (
    MutatorClient,
    concise_vision_label,
    dedupe_lessons,
    extract_json_object,
    message_text,
    normalize_lesson,
    normalize_vision_context,
    select_relevant_lessons,
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


if __name__ == "__main__":
    unittest.main()
