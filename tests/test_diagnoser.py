import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

from llm.mutator import MutatorClient
from main import diagnosable_failure, maybe_diagnose_frontier, persist_diagnosis_report


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def model_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_calls)


def response_for(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedMacroClient:
    """Plays back a fixed sequence of model messages, recording every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError("ScriptedMacroClient ran out of scripted responses.")
        return response_for(self.script.pop(0))


class FakeSession:
    def __init__(self, tmp):
        self.calls = []
        self.last_screenshot = None
        self.verified_experiments = []
        self._tmp = tmp
        self._shots = 0

    def _shot(self, tag):
        path = os.path.join(self._tmp, f"{tag}_{self._shots}.png")
        self._shots += 1
        with open(path, "wb") as f:
            f.write(b"png-bytes")
        self.last_screenshot = path
        return path

    def describe_window(self):
        return "Available emulator savestates: offset=200 ... offset=20"

    def view_frame(self, frames_before_failure):
        self.calls.append(("view_frame", frames_before_failure))
        return {
            "ok": True,
            "text": f"Viewing {frames_before_failure} frames before the failure.",
            "screenshot": self._shot("view"),
        }

    def try_actions(self, frames_before_failure, actions, hold_frames):
        self.calls.append(("try_actions", frames_before_failure, actions, hold_frames))
        self.verified_experiments.append(
            {"zone": 0, "act": 1, "start_x": 2404, "max_x": 2520, "actions": actions}
        )
        return {
            "ok": True,
            "text": f"Held '{actions}': progressed past the failure x: YES.",
            "screenshot": self._shot("try"),
            "passed_frontier_x": True,
        }

    def close(self):
        self.calls.append(("close",))


def make_mutator(client):
    mutator = object.__new__(MutatorClient)
    mutator.macro_client = client
    mutator.macro_model = "vision-model"
    return mutator


class DiagnoseFailureTests(unittest.TestCase):
    def run_diagnosis(self, script, session=None, **kwargs):
        client = ScriptedMacroClient(script)
        with tempfile.TemporaryDirectory() as tmp:
            session = session or FakeSession(tmp)
            with redirect_stdout(StringIO()):
                result = make_mutator(client).diagnose_failure(
                    session, "Sonic got stuck (zone 0 act 1)", [{"x": 3061}], **kwargs
                )
        return result, client, session

    def test_dispatches_tools_and_returns_finish_report(self):
        script = [
            model_message(tool_calls=[tool_call("c1", "try_actions", {
                "frames_before_failure": 120, "actions": "RIGHT,B", "hold_frames": 40,
            })]),
            model_message(tool_calls=[tool_call("c2", "finish_diagnosis", {
                "report": "Wall at x=3061; RIGHT,B from 120 frames earlier verified to clear it.",
            })]),
        ]

        result, client, session = self.run_diagnosis(script)

        self.assertEqual(
            result["report"],
            "Wall at x=3061; RIGHT,B from 120 frames earlier verified to clear it.",
        )
        self.assertIsNotNone(result["evidence_screenshot"])
        self.assertIn(("try_actions", 120, "RIGHT,B", 40), session.calls)
        # Verified escapes travel with the result for guard compilation.
        self.assertEqual(len(result["verified_experiments"]), 1)
        self.assertEqual(result["verified_experiments"][0]["actions"], "RIGHT,B")

        # Second request must contain the assistant tool-call echo, the tool
        # result, and a follow-up user message carrying the screenshot.
        messages = client.requests[1]["messages"]
        roles = [m["role"] for m in messages]
        self.assertIn("tool", roles)
        tool_message = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(tool_message["tool_call_id"], "c1")
        self.assertIn("progressed past the failure x", tool_message["content"])
        image_messages = [
            m for m in messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
            and any(part.get("type") == "image_url" for part in m["content"])
        ]
        self.assertGreaterEqual(len(image_messages), 1)

    def test_plain_text_response_without_tools_is_the_report(self):
        script = [model_message(content="The pit needs an early jump.")]

        result, client, _ = self.run_diagnosis(script)

        self.assertEqual(result["report"], "The pit needs an early jump.")
        self.assertIn("tools", client.requests[0])

    def test_budget_exhaustion_forces_a_no_tools_final_report(self):
        def looping_tool_response(call_id):
            return model_message(
                tool_calls=[tool_call(call_id, "view_frame", {"frames_before_failure": 60})]
            )

        script = [
            looping_tool_response("c1"),
            looping_tool_response("c2"),
            model_message(content="Final report after budget."),
        ]

        result, client, _ = self.run_diagnosis(script, max_tool_calls=2)

        self.assertEqual(result["report"], "Final report after budget.")
        self.assertIn("tools", client.requests[0])
        self.assertIn("tools", client.requests[1])
        self.assertNotIn("tools", client.requests[2])
        last_user = [m for m in client.requests[2]["messages"] if m["role"] == "user"][-1]
        self.assertIn("budget exhausted", str(last_user["content"]).lower())

    def test_unknown_tool_yields_error_text_not_crash(self):
        script = [
            model_message(tool_calls=[tool_call("c1", "teleport", {"x": 1})]),
            model_message(content="report"),
        ]

        result, client, _ = self.run_diagnosis(script)

        self.assertEqual(result["report"], "report")
        tool_message = next(m for m in client.requests[1]["messages"] if m["role"] == "tool")
        self.assertIn("Unknown tool", tool_message["content"])

    def test_client_error_returns_none(self):
        class ExplodingClient(ScriptedMacroClient):
            def _create(self, **kwargs):
                raise RuntimeError("provider down")

        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(StringIO()):
                result = make_mutator(ExplodingClient([])).diagnose_failure(
                    FakeSession(tmp), "stuck", []
                )

        self.assertIsNone(result)

    def test_no_macro_client_returns_none_without_calls(self):
        mutator = object.__new__(MutatorClient)
        mutator.macro_client = None

        self.assertIsNone(mutator.diagnose_failure(object(), "stuck", []))


class FrontierDiagnosisGatingTests(unittest.TestCase):
    def test_diagnosable_failure_filters_reasons(self):
        self.assertTrue(diagnosable_failure("Sonic got stuck: stopped making forward progress."))
        self.assertTrue(diagnosable_failure("Sonic lost a life or hit a fatal obstacle."))
        self.assertFalse(diagnosable_failure("Policy code timeout (likely an infinite loop)."))
        self.assertFalse(diagnosable_failure("Stagnation plateau: preserve the current working policy."))
        self.assertFalse(diagnosable_failure(None))

    def write_window(self, tmp):
        with open(os.path.join(tmp, "0.state"), "wb") as f:
            f.write(b"state")
        with open(os.path.join(tmp, "window.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "failure_reason": "stuck",
                    "created_at": 1,
                    "failure": {"frame": 100, "x_pos": 400},
                    "snapshots": [{"frame": 0, "file": "0.state", "x_pos": 10}],
                },
                f,
            )
        return tmp

    def test_diagnoses_once_and_caches_for_unchanged_frontier(self):
        class CountingMutator:
            def __init__(self):
                self.calls = 0

            def diagnose_failure(self, session, failure_reason, trace):
                self.calls += 1
                return {"report": "R", "evidence_screenshot": None}

        class NullSession:
            def __init__(self, window):
                self.window = window

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            frontier = {
                "failure_reason": "Sonic got stuck: stopped making forward progress.",
                "trace": [],
                "window": self.write_window(tmp),
            }
            mutator = CountingMutator()
            cache = {}
            # Tests must never write the real artifacts/diagnosis report.
            report_path = os.path.join(tmp, "latest_report.json")

            with redirect_stdout(StringIO()):
                first = maybe_diagnose_frontier(
                    mutator, frontier, cache, session_factory=NullSession, report_path=report_path
                )
                second = maybe_diagnose_frontier(
                    mutator, frontier, cache, session_factory=NullSession, report_path=report_path
                )

        self.assertEqual(first, {"report": "R", "evidence_screenshot": None})
        self.assertEqual(second, first)
        self.assertEqual(mutator.calls, 1)

    def test_gating_skips_disabled_missing_window_and_dry_run(self):
        mutator = SimpleNamespace(diagnose_failure=lambda *a, **k: {"report": "R"})
        frontier = {"failure_reason": "Sonic got stuck", "window": None}

        self.assertIsNone(maybe_diagnose_frontier(mutator, frontier, {}))
        self.assertIsNone(
            maybe_diagnose_frontier(mutator, {"failure_reason": "stuck", "window": "missing-dir"}, {})
        )
        with tempfile.TemporaryDirectory() as tmp:
            real = {"failure_reason": "Sonic got stuck", "window": self.write_window(tmp)}
            self.assertIsNone(
                maybe_diagnose_frontier(mutator, real, {}, emulator_available=False)
            )
            with mock.patch.dict(os.environ, {"SONIC_AGENTIC_DIAGNOSIS": "0"}):
                self.assertIsNone(maybe_diagnose_frontier(mutator, real, {}))

    def test_persist_diagnosis_report_writes_dashboard_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "latest_report.json")

            persist_diagnosis_report(
                {"report": "Verified RIGHT,B clears the wall.", "evidence_screenshot": "artifacts/diagnosis/x.png"},
                "Sonic got stuck",
                report_path=report_path,
            )

            with open(report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertEqual(payload["report"], "Verified RIGHT,B clears the wall.")
        self.assertEqual(payload["evidence_screenshot"], "artifacts/diagnosis/x.png")
        self.assertEqual(payload["failure_reason"], "Sonic got stuck")
        self.assertIn("created_at", payload)

    def test_persist_diagnosis_report_ignores_empty_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "latest_report.json")
            persist_diagnosis_report(None, "stuck", report_path=report_path)
            persist_diagnosis_report({"report": ""}, "stuck", report_path=report_path)
            self.assertFalse(os.path.exists(report_path))

    def test_stuck_frontier_gets_a_fresh_diagnosis_budget(self):
        class CountingMutator:
            def __init__(self):
                self.calls = 0

            def diagnose_failure(self, session, failure_reason, trace):
                self.calls += 1
                return {"report": "R", "evidence_screenshot": None, "verified_experiments": []}

        class NullSession:
            def __init__(self, window):
                self.window = window

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            frontier = {
                "failure_reason": "Sonic got stuck: stopped making forward progress.",
                "trace": [],
                "window": self.write_window(tmp),
            }
            mutator = CountingMutator()
            cache = {}
            report_path = os.path.join(tmp, "latest_report.json")

            def diagnose(stagnation):
                with redirect_stdout(StringIO()):
                    return maybe_diagnose_frontier(
                        mutator,
                        frontier,
                        cache,
                        session_factory=NullSession,
                        report_path=report_path,
                        stagnation_counter=stagnation,
                    )

            diagnose(0)
            diagnose(1)
            diagnose(2)
            self.assertEqual(mutator.calls, 1)  # cached while stagnation grows
            diagnose(3)
            self.assertEqual(mutator.calls, 2)  # fresh budget after 3 stagnant gens
            diagnose(4)
            self.assertEqual(mutator.calls, 2)  # re-cached against the new milestone

    def test_verified_escape_cache_is_served_then_refreshed_while_frontier_stays(self):
        # An UNCHANGED window means the cached "verified escape" never actually
        # promoted -- it must not pin the cache forever (agency review: the old
        # found_escape short-circuit disabled re-diagnosis permanently). The
        # cached result is served while fresh, then a fresh budget runs.
        class CountingMutator:
            def __init__(self):
                self.calls = 0

            def diagnose_failure(self, session, failure_reason, trace):
                self.calls += 1
                return {
                    "report": "R",
                    "evidence_screenshot": None,
                    "verified_experiments": [{"actions": "RIGHT,B"}],
                }

        class NullSession:
            def __init__(self, window):
                self.window = window

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            frontier = {
                "failure_reason": "Sonic got stuck: stopped making forward progress.",
                "trace": [],
                "window": self.write_window(tmp),
            }
            mutator = CountingMutator()
            cache = {}
            report_path = os.path.join(tmp, "latest_report.json")
            with redirect_stdout(StringIO()):
                results = [
                    maybe_diagnose_frontier(
                        mutator,
                        frontier,
                        cache,
                        session_factory=NullSession,
                        report_path=report_path,
                        stagnation_counter=stagnation,
                    )
                    for stagnation in range(8)
                ]

        # Initial diagnosis, then a fresh budget after every
        # REDIAGNOSE_AFTER_STAGNANT_GENERATIONS cached serves: 8 gens -> 3 runs.
        self.assertEqual(mutator.calls, 3)
        # The cached escape is still SERVED between refreshes, never dropped.
        self.assertTrue(all(r and r.get("verified_experiments") for r in results))

    def test_diagnosis_exception_is_cached_as_none(self):
        class ExplodingSessionFactory:
            def __init__(self, window):
                raise RuntimeError("no emulator")

        class CountingMutator:
            def __init__(self):
                self.calls = 0

            def diagnose_failure(self, *args):
                self.calls += 1
                return {"report": "R"}

        with tempfile.TemporaryDirectory() as tmp:
            frontier = {
                "failure_reason": "Sonic got stuck",
                "trace": [],
                "window": self.write_window(tmp),
            }
            mutator = CountingMutator()
            cache = {}
            with redirect_stdout(StringIO()):
                self.assertIsNone(
                    maybe_diagnose_frontier(mutator, frontier, cache, session_factory=ExplodingSessionFactory)
                )
                self.assertIsNone(
                    maybe_diagnose_frontier(mutator, frontier, cache, session_factory=ExplodingSessionFactory)
                )
        self.assertEqual(mutator.calls, 0)
        self.assertIn("key", cache)


if __name__ == "__main__":
    unittest.main()
