import ast
import base64
import importlib
import json
import mimetypes
import os
import re
import sys

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.frontier import (
    build_llm_guard_candidate,
    llm_guard_marker,
    recently_attempted_frontier_guard,
)
from core.fsio import atomic_write_text
from core.policy_validator import validate_skills_source
from core.trace_context import trace_entry_x, trace_entry_zone_act
from llm.prompts import SYSTEM_PROMPT


def _top_level_functions(source):
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source or "").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_uses_global_state(node):
    """True for functions copying the policies' global-state idiom.

    Champions carry injected guard blocks built on ``global`` /
    ``globals()``; extraction routinely copies them into skills, where the
    stricter skills validator rejects the WHOLE library (live-observed: the
    skill library stopped learning after the first guard promotion). Such
    functions are dropped individually instead.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Global):
            return True
        if isinstance(child, ast.Name) and child.id == "globals":
            return True
    return False


def extract_json_object(text):
    """Extract the first valid JSON object from raw or fenced model output."""
    if not text:
        return None

    candidates = [text.strip()]
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            candidates.append(cleaned)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parses(code):
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def _defines_get_action(code):
    """True if ``code`` parses as Python and defines a top-level get_action."""
    if not _parses(code):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_action"
        for node in ast.parse(code).body
    )


def extract_python_block(raw_response, predicate=_parses):
    """Pull working source out of a (possibly reasoning-model) response.

    Reasoning models such as Gemma emit several fenced blocks in their
    scratchpad -- partial drafts, the final version, sometimes an example. The
    old heuristic took the *last* fenced block unconditionally, which frequently
    grabbed a truncated draft and produced a SyntaxError. Instead, return the
    last fenced block satisfying ``predicate``; only if none qualifies fall back
    to the previous last-block/raw behavior so downstream validation/repair
    still gets *something* rather than silently dropping the model's output.
    """
    if not raw_response:
        return ""

    blocks = [b.strip() for b in re.findall(r"```(?:python)?\s*\n?(.*?)```", raw_response, re.DOTALL)]
    blocks = [b for b in blocks if b]

    for block in reversed(blocks):
        if predicate(block):
            return block

    stripped = raw_response.strip()
    if predicate(stripped):
        return stripped

    if blocks:
        return blocks[-1]
    return stripped


def extract_policy_code(raw_response):
    """The last fenced block that parses AND defines a top-level get_action."""
    return extract_python_block(raw_response, predicate=_defines_get_action)


# Hard cap on stored lessons; oldest entries are dropped beyond this.
MAX_SEMANTIC_LESSONS = 100

# Ordered keyword buckets for collapsing the LLM's freeform hazard names
# ("Wall/Ledge", "Vertical stuck loop", "Pitfall", ...) into a small category
# set so paraphrased lessons about the same obstacle deduplicate.
_HAZARD_CATEGORIES = (
    ("pit", ("pit", "fall", "hole", "gap", "chasm")),
    ("spikes", ("spike",)),
    ("enemy", ("enemy", "badnik", "robot", "crab", "buzz", "chopper", "motobug")),
    ("water", ("water", "drown")),
    (
        "wall",
        (
            "wall", "ledge", "obstacle", "stuck", "stall", "stagnation",
            "geometry", "slope", "dip", "loop", "bounce", "deadlock",
            "collision", "barrier", "vertical",
        ),
    ),
)


def hazard_category(hazard, lesson_text=""):
    """Map a freeform hazard description to a coarse category name."""
    for blob in (str(hazard or "").lower(), str(lesson_text or "").lower()):
        for category, needles in _HAZARD_CATEGORIES:
            if any(needle in blob for needle in needles):
                return category
    return "other"


def normalize_lesson(entry):
    """Return a compact semantic lesson dict, or None for malformed entries.

    ``zone``/``act`` are kept when parseable so lessons can be scoped to the
    act their x coordinate belongs to (x resets to ~0 every act). Lessons
    without them are legacy entries treated as applying anywhere.
    """
    if not isinstance(entry, dict):
        return None
    try:
        x = int(float(entry.get("x")))
    except (TypeError, ValueError):
        return None

    lesson = str(entry.get("lesson", "")).strip()
    if not lesson:
        return None

    hazard = str(entry.get("hazard", "Unknown")).strip() or "Unknown"
    normalized = {"x": x, "hazard": hazard, "lesson": lesson}
    for key in ("zone", "act"):
        try:
            normalized[key] = int(float(entry[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return normalized


def lesson_semantic_key(lesson, x_bucket_size=25):
    """Cluster key: same zone/act, same x bucket, same hazard category."""
    bucket = max(1, int(x_bucket_size))
    return (
        lesson.get("zone"),
        lesson.get("act"),
        int(lesson["x"]) // bucket,
        hazard_category(lesson.get("hazard", ""), lesson.get("lesson", "")),
    )


def dedupe_lessons(lessons, x_tolerance=25):
    """Collapse malformed entries and semantically duplicate lessons.

    Earlier dedupe only collapsed *case-identical* text, so every paraphrase of
    "you are stuck at x=3061, jump" accumulated and crowded the prompt's lesson
    budget. Lessons now deduplicate by (zone, act, x bucket, hazard category);
    the first occurrence wins and order is preserved.
    """
    deduped = []
    seen = set()
    for lesson in lessons:
        normalized = normalize_lesson(lesson)
        if normalized is None:
            continue
        key = lesson_semantic_key(normalized, x_tolerance)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def load_semantic_bank(path="memory/semantic_bank.json"):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_bank = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw_bank, list):
        return []
    return dedupe_lessons(raw_bank)


def select_relevant_lessons(lessons, current_x, radius=1000, limit=10, zone=None, act=None):
    """Pick lessons near ``current_x`` that apply to the current zone/act.

    x coordinates are only meaningful within one act, so lessons tagged with a
    zone/act are excluded outside it. Legacy untagged lessons (and callers that
    do not know the current act) keep the old anywhere-matching behaviour.
    """
    relevant = []
    for lesson in dedupe_lessons(lessons):
        if abs(lesson["x"] - current_x) > radius:
            continue
        lesson_zone = lesson.get("zone")
        lesson_act = lesson.get("act")
        if zone is not None and lesson_zone is not None and lesson_zone != zone:
            continue
        if act is not None and lesson_act is not None and lesson_act != act:
            continue
        if lesson_zone is not None and lesson_act is not None:
            relevant.append(f"- [zone {lesson_zone} act {lesson_act}] {lesson['lesson']}")
        else:
            relevant.append(f"- {lesson['lesson']}")
    return relevant[-limit:]


def normalize_vision_context(content):
    if not content:
        return "UNKNOWN"
    normalized = str(content).strip().upper()
    return normalized or "UNKNOWN"


# --- Agentic failure diagnosis -------------------------------------------
# Hard ceiling on model-initiated emulator operations per diagnosis. One
# diagnosis is cached for the whole life of a frontier, so a thorough hunt
# amortizes across every stagnant generation that follows.
DIAGNOSIS_MAX_TOOL_CALLS = 10

DIAGNOSIS_SYSTEM_PROMPT = """You are a game-physics failure analyst with interactive control of a Sega Genesis emulator, paused around the moment a Sonic policy failed.
PRIMARY GOAL: find, by experiment, an input that beats the run's furthest progress -- a result that says "Beat the run's furthest progress: YES". A verified escape is compiled directly into the next candidate policy, so it is worth more than any amount of description.
- try_action_sequence: play TIMED SEGMENTS, e.g. [{"actions":"RIGHT","frames":90},{"actions":"RIGHT,B","frames":40}]. THIS IS USUALLY THE WINNING TOOL: Sonic's jump fires on the B PRESS, so a held "RIGHT,B" jumps exactly once at the start -- "build speed, THEN jump at the edge" is only expressible as a sequence. Vary run-up length to move the jump point. IMPORTANT: every rewind point lies on the FAILING run's own path, so Sonic arrives with the same losing momentum -- if forward attempts keep falling short, back up first to build a longer runway (e.g. [{"actions":"LEFT","frames":90},{"actions":"RIGHT","frames":150},{"actions":"RIGHT,B","frames":40}]).
- try_actions: hold ONE combination for N frames (momentum tests: plain RIGHT from far back, RIGHT,DOWN rolling).
- view_frame: look at the situation N frames before the failure (use sparingly; experiments teach more).
When an experiment reports YES, or you are out of ideas, call finish_diagnosis with a concise report covering:
1. What the obstacle/hazard actually is (from the screenshots).
2. The earliest state cue that predicts it (x range, velocity pattern, vision context).
3. Which inputs you VERIFIED work or fail, with their measured outcomes.
4. A concrete recommendation for the policy code."""

DIAGNOSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "view_frame",
            "description": (
                "Load the emulator savestate nearest to N frames before the failure and look at it. "
                "Returns the authoritative game state and a screenshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frames_before_failure": {
                        "type": "integer",
                        "description": "Frames before the failure moment to view (0 = the failure itself, 60 = one second earlier).",
                    }
                },
                "required": ["frames_before_failure"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "try_actions",
            "description": (
                "Counterfactual experiment: rewind to N frames before the failure, hold a button "
                "combination, and report what actually happens (movement, rings, lives, whether Sonic "
                "progressed past the failure point) plus an end screenshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frames_before_failure": {"type": "integer"},
                    "actions": {
                        "type": "string",
                        "description": "Comma-separated buttons to hold, e.g. 'RIGHT,B'. Valid: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z.",
                    },
                    "hold_frames": {
                        "type": "integer",
                        "description": "How many frames to hold the input (max 300, ~5 seconds).",
                    },
                },
                "required": ["frames_before_failure", "actions", "hold_frames"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "try_action_sequence",
            "description": (
                "Counterfactual experiment with TIMED SEGMENTS: rewind to N frames before the failure, "
                "then play each segment in order (e.g. build speed with RIGHT, then press RIGHT,B to "
                "jump at the edge). Reports measured movement per segment and whether Sonic beat the "
                "run's furthest progress, plus an end screenshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frames_before_failure": {"type": "integer"},
                    "segments": {
                        "type": "array",
                        "description": "Up to 5 segments played in order; 600 frames total.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "actions": {"type": "string", "description": "Buttons held during this segment, e.g. 'RIGHT' or 'RIGHT,B'."},
                                "frames": {"type": "integer", "description": "How many frames to hold this segment."},
                            },
                            "required": ["actions", "frames"],
                        },
                    },
                },
                "required": ["frames_before_failure", "segments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_diagnosis",
            "description": "End the investigation and submit the final diagnosis report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {
                        "type": "string",
                        "description": "The diagnosis: obstacle identity, predictive state cue, verified inputs, recommendation.",
                    }
                },
                "required": ["report"],
            },
        },
    },
]

VISION_SCREEN_BUCKET = 250


def vision_location_key(state, bucket_size=VISION_SCREEN_BUCKET):
    """Stable key for "what the screen shows here": zone, act, camera-x bucket.

    Proactive vision labels used to depend on cloud-call timing, which made the
    `vision_context` a policy sees -- and therefore its fitness -- vary between
    runs of the same policy. Caching labels by location makes re-runs see the
    same context and avoids re-paying API calls for already-seen screens.
    """
    try:
        zone = int(float(state.get("zone", 0)))
        act = int(float(state.get("act", 0)))
        screen_x = int(float(state.get("screen_x", state.get("x_pos", 0))))
    except (TypeError, ValueError):
        return None
    bucket = max(1, int(bucket_size))
    return f"zone-{zone}-act-{act}-sx-{(screen_x // bucket) * bucket}"


def message_text(message):
    """Return a model message's text, falling back to ``reasoning_content``.

    Reasoning models (e.g. gemma, qwen3) routinely leave ``content`` empty and
    place their output in ``reasoning_content``. The micro path already handled
    this; this helper lets the macro/vision paths do the same.
    """
    content = (getattr(message, "content", "") or "").strip()
    if content:
        return content
    return (getattr(message, "reasoning_content", "") or "").strip()


_VISION_STOPWORDS = {
    "OR", "AND", "THE", "A", "AN", "IS", "ARE", "WITH", "OF", "TO", "IN", "ON",
    "NO", "SONIC", "AHEAD", "IMMEDIATE", "CONTEXT", "HAZARD",
}


def concise_vision_label(text, max_words=2):
    """Collapse a (possibly verbose, reasoning-style) reply to a short tag.

    Reasoning models often conclude with the answer, so we keep the trailing
    meaningful words and drop connector/filler tokens.
    """
    words = re.findall(r"[A-Za-z]+", text or "")
    meaningful = [w for w in words if w.upper() not in _VISION_STOPWORDS]
    picks = (meaningful or words)[-max_words:]
    return normalize_vision_context(" ".join(picks)) if picks else "UNKNOWN"


class MutatorClient:
    def __init__(self):
        # Cloud/Macro Model Config
        self.macro_api_key = os.environ.get("MACRO_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        # Default to Google's OpenAI-compatible endpoint if using Gemini directly
        self.macro_base_url = os.environ.get("MACRO_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.macro_model = os.environ.get("MACRO_MODEL", "gemini-2.5-pro")

        # Local/Micro Model Config
        self.micro_base_url = os.environ.get("MICRO_BASE_URL", "http://localhost:1234/v1")
        self.micro_model = os.environ.get("MICRO_MODEL", "local-model")

        self.macro_client = OpenAI(api_key=self.macro_api_key, base_url=self.macro_base_url) if self.macro_api_key else None
        self.micro_client = OpenAI(api_key="not-needed", base_url=self.micro_base_url)

        # Location-keyed cache of proactive vision labels (see
        # vision_location_key). Loaded lazily; persisted as a plain JSON map.
        self.vision_cache_path = os.environ.get(
            "SONIC_VISION_CACHE", "artifacts/vision_cache.json"
        )
        self._vision_cache = None

    def _load_vision_cache(self):
        if self._vision_cache is None:
            try:
                with open(self.vision_cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self._vision_cache = payload if isinstance(payload, dict) else {}
            except (OSError, ValueError):
                self._vision_cache = {}
        return self._vision_cache

    def cached_vision_context(self, location_key):
        """Return the stored vision label for a location, or None."""
        if not location_key:
            return None
        value = self._load_vision_cache().get(location_key)
        return str(value) if value else None

    def store_vision_context(self, location_key, label):
        """Persist a successful vision label; UNKNOWN never poisons a location."""
        if not location_key or not label or label == "UNKNOWN":
            return
        cache = self._load_vision_cache()
        cache[location_key] = str(label)
        try:
            atomic_write_text(
                self.vision_cache_path,
                json.dumps(cache, indent=2, sort_keys=True),
            )
        except OSError as e:
            print(f"Failed to persist vision cache: {e}")

    def write_seed_policy(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        seed_code = """
def get_action(state):
    # Basic Gen 0 Seed Policy
    # Always run right, jump occasionally
    action = "RIGHT"

    # Try to jump if rings are 0 (maybe we hit something)
    if state.get('rings', 1) == 0:
        action += ",B"

    return action
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(seed_code.strip())

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _image_data_url(self, image_path):
        """Build a base64 data URL with the MIME type matching the file.

        Screenshots are written as PNG, so hardcoding image/jpeg can be
        rejected by stricter providers. Derive the type from the extension and
        fall back to PNG.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"
        return f"data:{mime_type};base64,{self._encode_image(image_path)}"

    def _call_macro_model(self, prompt, image_path, temperature=0.7):
        """Calls Cloud LLM for Macro-Mutations (needs vision)."""
        if not image_path:
            print("No screenshot available, falling back to local Micro-Mutation model.")
            return self._call_micro_model(prompt, temperature)
        if not self.macro_client:
            print("No MACRO_API_KEY found, falling back to local Micro-Mutation model.")
            return self._call_micro_model(prompt, temperature)

        print(f"Using Cloud API ({self.macro_model}) for Macro-Mutation.")
        try:
            return self._do_macro_call(prompt, image_path, temperature)
        except Exception as e:
            print(f"Cloud API failed after retries: {e}")
            return self._call_micro_model(prompt, temperature)

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _do_macro_call(self, prompt, image_path, temperature):
        response = self.macro_client.chat.completions.create(
            model=self.macro_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_data_url(image_path)
                            }
                        }
                    ]
                }
            ],
            temperature=temperature,
            # Match the micro path: a reasoning vision model (e.g. gemma) spends
            # tokens thinking before it emits code, so a small cap truncates the
            # policy mid-output and the extracted fragment fails to parse.
            max_tokens=8192,
            timeout=300
        )

        text = message_text(response.choices[0].message)
        if not text:
            raise ValueError("Macro model returned empty content and reasoning.")
        return text, "Cloud vision analysis completed."


    def _call_micro_model(self, prompt, temperature=0.7):
        """Calls Local LLM for Micro-Mutations (code only)."""
        print(f"Using Local API ({self.micro_base_url}) for Micro-Mutation (Temp: {temperature}).")
        try:
            return self._do_micro_call(prompt, temperature)
        except Exception as e:
            print(f"Local inference failed after retries: {e}")
            return "def get_action(state):\n    return 'RIGHT'", "Fallback to simple RIGHT."

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _do_micro_call(self, prompt, temperature):
        response = self.micro_client.chat.completions.create(
            model=self.micro_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=8192,
            timeout=300
        )
        content = message_text(response.choices[0].message)
        if not content:
            raise ValueError("LLM returned an empty string and empty reasoning. Likely a concurrency/queue failure.")

        return content, "Local inference completed."

    def _image_user_message(self, text, image_path):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": self._image_data_url(image_path)}},
            ],
        }

    def _dispatch_diagnosis_tool(self, session, name, args):
        if name == "view_frame":
            return session.view_frame(args.get("frames_before_failure", 0))
        if name == "try_actions":
            return session.try_actions(
                args.get("frames_before_failure", 0),
                str(args.get("actions", "")),
                args.get("hold_frames", 60),
            )
        if name == "try_action_sequence":
            return session.try_action_sequence(
                args.get("frames_before_failure", 0),
                args.get("segments", []),
            )
        return {"ok": False, "text": f"Unknown tool: {name}", "screenshot": None}

    def diagnose_failure(self, session, failure_reason, coordinate_trace=None, max_tool_calls=DIAGNOSIS_MAX_TOOL_CALLS):
        """Interactively replay a failure with the vision model driving tools.

        Returns ``{"report": str, "evidence_screenshot": path-or-None}`` or
        ``None`` on any problem -- the caller falls back to the static-montage
        mutation path, so diagnosis can never take down training.
        """
        if not self.macro_client:
            return None
        try:
            return self._run_diagnosis_loop(session, failure_reason, coordinate_trace, max_tool_calls)
        except Exception as e:
            print(f"Agentic diagnosis failed: {e}")
            return None

    def _run_diagnosis_loop(self, session, failure_reason, coordinate_trace, max_tool_calls):
        recent_trace = json.dumps(list(coordinate_trace or [])[-5:])
        intro = (
            f"A Sonic policy just failed: {failure_reason}\n"
            f"Recent frame trace: {recent_trace}\n\n"
            f"{session.describe_window()}\n\n"
            "Investigate with the tools (prefer try_actions experiments over speculation), "
            "then call finish_diagnosis with your report."
        )
        messages = [
            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": intro},
        ]

        # Show the failure moment up front; does not count against the budget.
        initial_view = session.view_frame(0)
        if initial_view.get("ok") and initial_view.get("screenshot"):
            messages.append(
                self._image_user_message(
                    f"The failure moment itself: {initial_view['text']}",
                    initial_view["screenshot"],
                )
            )

        tool_calls_used = 0
        while True:
            force_finish = tool_calls_used >= max_tool_calls
            request = {
                "model": self.macro_model,
                "messages": messages,
                "temperature": 0.3,
                # Roomy enough that the final report does not truncate
                # mid-sentence (observed at 2048 with gemma in live testing).
                "max_tokens": 3072,
                "timeout": 120,
            }
            if not force_finish:
                request["tools"] = DIAGNOSIS_TOOLS
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool budget exhausted. Write your final diagnosis report now.",
                    }
                )
            response = self.macro_client.chat.completions.create(**request)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if force_finish or not tool_calls:
                report = message_text(message)
                if not report:
                    raise ValueError("Diagnosis model returned an empty report.")
                return {
                    "report": report,
                    "evidence_screenshot": session.last_screenshot,
                    "verified_experiments": list(getattr(session, "verified_experiments", [])),
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            image_followups = []
            for call in tool_calls:
                tool_calls_used += 1
                try:
                    args = json.loads(call.function.arguments or "{}")
                except ValueError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                if call.function.name == "finish_diagnosis":
                    print("  diagnosis: finish_diagnosis")
                    report = str(args.get("report", "")).strip() or message_text(message)
                    if not report:
                        raise ValueError("finish_diagnosis was called without a report.")
                    return {
                        "report": report,
                        "evidence_screenshot": session.last_screenshot,
                        "verified_experiments": list(getattr(session, "verified_experiments", [])),
                    }

                result = self._dispatch_diagnosis_tool(session, call.function.name, args)
                # One line per tool call so the operator can watch the
                # investigation progress (and spot broken tools immediately).
                # The verdict lives at the END of experiment texts, so log the
                # tail as well as the head.
                if result.get("ok"):
                    outcome = "VERIFIED ESCAPE" if result.get("passed_frontier_x") else "ok"
                else:
                    outcome = "ERROR"
                text = str(result.get("text", ""))
                summary = text[:110] + (" ... " + text[-90:] if len(text) > 200 else "")
                print(
                    f"  diagnosis: {call.function.name}({json.dumps(args, sort_keys=True)}) "
                    f"-> {outcome}: {summary}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(result.get("text", "")),
                    }
                )
                if result.get("screenshot"):
                    image_followups.append(
                        self._image_user_message(
                            f"Screenshot from your {call.function.name} call:",
                            result["screenshot"],
                        )
                    )
            messages.extend(image_followups)

    def extract_lesson(self, failure_reason, coordinate_trace):
        prompt = f"""
We failed with the following reason: {failure_reason}
Coordinate trace: {coordinate_trace}

Based on this failure, extract a single, concise, universal rule (a 'lesson learned') for the AI policy to avoid this in the future.
Analyze the coordinate trace. At what approximate X coordinate did the failure occur?
Return ONLY a valid JSON object in this exact format:
{{
    "x": 1500,
    "hazard": "Name of hazard",
    "lesson": "When at X=1500, do X to avoid Y"
}}
"""
        lesson_json_str, _ = self._call_micro_model(prompt, temperature=0.3)
        lesson_data = normalize_lesson(extract_json_object(lesson_json_str))
        if lesson_data is None:
            print("Failed to parse semantic lesson JSON.")
            return

        # Stamp zone/act from the trace (authoritative emulator state) rather
        # than trusting the model: x coordinates only mean anything per-act.
        if coordinate_trace:
            zone, act = trace_entry_zone_act(coordinate_trace[-1])
            if zone is not None:
                lesson_data["zone"] = zone
            if act is not None:
                lesson_data["act"] = act

        os.makedirs("memory", exist_ok=True)
        bank_path = "memory/semantic_bank.json"
        bank = load_semantic_bank(bank_path)
        size_before = len(bank)
        bank.append(lesson_data)
        bank = dedupe_lessons(bank)[-MAX_SEMANTIC_LESSONS:]
        with open(bank_path, "w", encoding="utf-8") as f:
            json.dump(bank, f, indent=4)
        if len(bank) > size_before:
            print(f"Extracted and saved semantic lesson: {lesson_data.get('lesson')}")
        else:
            print(
                "Extracted lesson deduplicated into an existing one for this "
                f"location/hazard: {lesson_data.get('lesson')}"
            )

    def analyze_environment(self, image_path):
        """Uses the Cloud VLM to proactively tag the current visual environment.

        Failures are swallowed and reported as "UNKNOWN" so the run continues,
        which is why this is deliberately *not* wrapped in @retry: the except
        block below would suppress every retry anyway.
        """
        if not self.macro_client:
            return "UNKNOWN"

        prompt = "Analyze this screenshot from Sonic the Hedgehog. Reply with ONLY ONE or TWO WORDS describing the most immediate upcoming hazard or context directly in front of sonic (e.g. 'CLEAR', 'ENEMY', 'SPIKES', 'LOOP', 'PLATFORM', 'WALL')."

        try:
            response = self.macro_client.chat.completions.create(
                model=self.macro_model,
                messages=[
                    {"role": "system", "content": "You are a fast visual classifier. Output only 1-2 words. No formatting."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self._image_data_url(image_path)
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=256,  # reasoning models need room to finish before answering
                timeout=30
            )
            return concise_vision_label(message_text(response.choices[0].message))
        except Exception as e:
            print(f"Proactive vision analysis failed: {e}")
            return "UNKNOWN"



    def extract_and_save_skills(self, policy_code):
        skills_path = "policies/skills.py"
        existing_skills = ""
        if os.path.exists(skills_path):
            with open(skills_path, "r", encoding="utf-8") as f:
                existing_skills = f.read()

        prompt = f"""
We have discovered a highly successful AI policy:
```python
{policy_code}
```

Here is the current skill library:
```python
{existing_skills}
```

Extract any clear, reusable logic from the successful policy into standalone Python functions (skills).
Return ONLY valid Python code containing the updated skill library (the existing skills plus any new ones). Do NOT include `get_action`.
"""
        new_skills_code, _ = self._call_micro_model(prompt, temperature=0.3)

        if new_skills_code:
            try:
                # Robust extraction: the last fenced block that actually parses
                # (reasoning models leave truncated drafts in their scratchpad).
                new_skills_code = extract_python_block(new_skills_code)

                existing_functions = _top_level_functions(existing_skills)
                new_functions = _top_level_functions(new_skills_code)

                new_ast = ast.parse(new_skills_code)
                added_funcs_code = []
                dropped_global_state = []
                for node in new_ast.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name not in existing_functions:
                            if _function_uses_global_state(node):
                                dropped_global_state.append(node.name)
                                continue
                            source = ast.get_source_segment(new_skills_code, node)
                            if source:
                                added_funcs_code.append(source)
                if dropped_global_state:
                    print(
                        "Note: dropped extracted skills that copy the policy's "
                        "global-state idiom (not allowed in the skills library): "
                        + ", ".join(sorted(dropped_global_state))
                    )

                changed_functions = {
                    name
                    for name, definition in existing_functions.items()
                    if name in new_functions and new_functions[name] != definition
                }
                if changed_functions:
                    print(
                        "Note: Extracted skills tried to update existing functions (ignored): "
                        + ", ".join(sorted(changed_functions))
                    )

                if not added_funcs_code:
                    print("No new skills extracted.")
                    return

                combined_skills = existing_skills
                if combined_skills and not combined_skills.endswith("\n"):
                    combined_skills += "\n"
                if combined_skills:
                    combined_skills += "\n\n"
                combined_skills += "\n\n".join(added_funcs_code) + "\n"

                validate_skills_source(combined_skills)
                atomic_write_text(skills_path, combined_skills)
                importlib.invalidate_caches()
                loaded_skills = sys.modules.get("policies.skills")
                try:
                    if loaded_skills is None:
                        importlib.import_module("policies.skills")
                    else:
                        importlib.reload(loaded_skills)
                except Exception:
                    if existing_skills:
                        atomic_write_text(skills_path, existing_skills)
                    elif os.path.exists(skills_path):
                        os.remove(skills_path)
                    importlib.invalidate_caches()
                    if loaded_skills is not None:
                        importlib.reload(loaded_skills)
                    raise
                print("Extracted and saved new skills to policies/skills.py")
            except Exception as e:
                print(f"Failed to save extracted skills: {e}")

    def _request_guard_proposal(self, user_prompt, image_path, temperature):
        """One raw model call returning a JSON guard proposal (not code).

        Uses the vision model when a frame is available (a stuck frontier is a
        visual problem), else the local code model. A dedicated system prompt is
        needed because the normal SYSTEM_PROMPT asks for full policy code.
        """
        system = (
            "You choose ONE controller input for a Sonic policy stuck at one spot. "
            "Reply with ONLY a compact JSON object and nothing else."
        )
        use_vision = bool(image_path) and self.macro_client is not None
        client = self.macro_client if use_vision else self.micro_client
        model = self.macro_model if use_vision else self.micro_model
        if use_vision:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": self._image_data_url(image_path)}},
            ]
        else:
            user_content = user_prompt
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=4096,  # reasoning models spend tokens before emitting the JSON
            timeout=180,
        )
        return message_text(response.choices[0].message)

    def _try_structured_guard(
        self, current_code, failure_reason, screenshot_path, zone, act, x,
        diagnosis_text, recent_history, temperature,
    ):
        """Ask for a structured {action, hold_frames} proposal and compile it
        into a champion-preserving guard. Returns guard code, or None to fall
        back to a full rewrite."""
        prompt = f"""The working Sonic policy is STUCK and cannot get past one spot.

Authoritative emulator state at the stall (from RAM -- exact, do not change):
  zone={zone}, act={act}, x_pos={x}

Failure: {failure_reason}
{diagnosis_text}

Look at the attached frame. Choose ONE input to try AT THAT SPOT to get Sonic moving forward past it.
Reply with ONLY this JSON object (no prose, no code):
{{"action": "<comma-separated buttons>", "hold_frames": <int 1-120>, "why": "<one short sentence>"}}

Valid buttons: RIGHT, LEFT, UP, DOWN, B, A, C  (B is jump).
Typical escapes: "RIGHT,B" (jump a wall/gap), "RIGHT,DOWN" (roll through), "RIGHT" (build speed), "RIGHT,UP,B" (higher jump).
hold_frames: how long to hold it -- ~12 for a hop, ~25 for a full jump, ~40 to roll a stretch.

[cache breaker {os.urandom(6).hex()}]"""
        raw = self._request_guard_proposal(prompt, screenshot_path, temperature)
        print(f"Structured guard proposal (raw): {repr(raw)[:200]}")
        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            return None
        proposal = {
            "zone": zone,
            "act": act,
            "x": x,
            "action": parsed.get("action") or parsed.get("actions"),
            "hold_frames": parsed.get("hold_frames", 0),
        }
        guard = build_llm_guard_candidate(current_code, proposal)
        if guard is None:
            return None
        if recently_attempted_frontier_guard(llm_guard_marker(guard), recent_history):
            print("Structured guard already attempted recently; falling back to rewrite.")
            return None
        return guard

    def mutate_policy(self, current_code, failure_reason, screenshot_path, recent_history, temperature=0.7, coordinate_trace=None, diagnosis_report=None, frontier=None):
        history_text = json.dumps(recent_history, indent=2)
        diagnosis_text = ""
        if diagnosis_report:
            diagnosis_text = (
                "Agentic Failure Diagnosis (a vision model interactively replayed this failure on the "
                "emulator and ran counterfactual input experiments; trust its VERIFIED findings over "
                "guesses from the screenshot):\n" + str(diagnosis_report)
            )
        trace_text = ""
        current_x = 0
        current_zone, current_act = None, None
        if coordinate_trace:
            trace_text = f"Recent frame trace leading to failure: {coordinate_trace}"
            if len(coordinate_trace) > 0:
                current_x = trace_entry_x(coordinate_trace[-1])
                current_zone, current_act = trace_entry_zone_act(coordinate_trace[-1])

        # The evaluator's authoritative frontier beats the trace tail: after a
        # death-then-respawn the trace tail sits at the RESPAWN point, and a
        # guard (or lesson lookup) aimed there misses the death spot entirely.
        if isinstance(frontier, dict):
            try:
                current_zone = int(frontier["zone"])
                current_act = int(frontier["act"])
                current_x = int(frontier["x"])
            except (KeyError, TypeError, ValueError):
                pass

        # Prefer a structured, champion-preserving guard: the model proposes only
        # WHAT to try (buttons + hold duration); the coordinates stay authoritative
        # and the working code is never rewritten, only extended. Fall back to a
        # full rewrite when there is no frontier, the fault is a code timeout, or
        # the proposal does not compile into a fresh guard.
        if "timeout" not in failure_reason.lower() and current_zone is not None and current_act is not None:
            try:
                guard = self._try_structured_guard(
                    current_code, failure_reason, screenshot_path,
                    current_zone, current_act, current_x,
                    diagnosis_text, recent_history, temperature,
                )
            except Exception as e:
                print(f"Structured guard proposal failed ({e}); falling back to rewrite.")
                guard = None
            if guard is not None:
                return guard, "LLM structured guard"

        lessons_text = ""
        if os.path.exists("memory/semantic_bank.json"):
            try:
                bank = load_semantic_bank("memory/semantic_bank.json")
                relevant_lessons = select_relevant_lessons(
                    bank,
                    current_x,
                    radius=1000,
                    limit=10,
                    zone=current_zone,
                    act=current_act,
                )
                if relevant_lessons:
                    lessons_text = "Relevant Semantic Memory (CRITICAL - DO NOT VIOLATE):\n" + "\n".join(relevant_lessons)
            except Exception as e:
                print(f"Error loading semantic bank: {e}")

        skills_text = ""
        skills_path = "policies/skills.py"
        if os.path.exists(skills_path):
            try:
                with open(skills_path, "r", encoding="utf-8") as f:
                    skills_content = f.read().strip()
                    if skills_content and not skills_content.startswith("# This file"):
                        skills_text = "Available Skills (from `policies.skills`):\n```python\n" + skills_content + "\n```\nYou can import these via `import policies.skills as skills` and call them like `skills.my_func(state)`. Use them to construct `get_action(state)`."
            except Exception as e:
                print(f"Error loading skills.py: {e}")

        prompt = f"""
Here is the current code that failed:
```python
{current_code}
```

Primary Failure Reason (the working policy's own frontier): {failure_reason}
{trace_text}

{diagnosis_text}

Recent History of Other Evaluated Candidates (background only; these failures
may not apply to the current code):
{history_text}

{lessons_text}

{skills_text}

Note on Vision Context: The emulator now actively looks at the screen every 5 seconds. The immediate upcoming visual context is injected into `state['vision_context']` (e.g., 'ENEMY', 'CLEAR', 'SPIKES'). You can write logic to check this string!

Analyze the failure and rewrite `get_action(state)`.

CRITICAL — preserve progress: the current code already makes real progress before it
fails. Keep its existing working logic and structure intact and change the SMALLEST
amount needed to get past the specific failure shown above. Do NOT delete working
rules or rewrite unrelated sections, or you will regress earlier progress.
Fix the primary working-policy frontier first. Do not modify the current code solely
to address a failure listed in the other-candidate history.
Note that `x_pos` resets to ~0 at the start of each act and `state['zone']`/`state['act']`
tell you which act you are in — when handling a NEW act, prefer general
velocity/vision-based logic over hardcoded x-coordinates (which only apply to one act).

Return ONLY valid Python code, starting with `def get_action(state):`.

[SYSTEM CACHE BREAKER: {os.urandom(8).hex()} - Ignore this random string and DO NOT write it into your code.]
"""

        # Route by failure type. A pure code fault (an infinite loop caught as a
        # timeout), or having no frame to look at, goes to the local code model.
        # Everything else -- Sonic stuck against level geometry, or killed by a
        # hazard -- is a *visual* problem: the model needs to SEE the frame to
        # understand what is blocking or killing it, so it goes to the vision
        # (macro) model. (Earlier this also sent "stuck" to the blind code model,
        # but a 30-generation run showed that left the model unable to get past
        # unfamiliar geometry it could not see -- 0 vision calls, hard plateau.)
        if "timeout" in failure_reason.lower() or not screenshot_path:
            raw_response, reasoning = self._call_micro_model(prompt, temperature)
        else:
            raw_response, reasoning = self._call_macro_model(prompt, screenshot_path, temperature)

        print(f"Raw Response from LLM (mutate): {repr(raw_response)}")
        return extract_policy_code(raw_response), reasoning

    def repair_policy(self, candidate_code, validation_error):
        """Use the local code model once to repair an exact validator failure."""
        prompt = f"""
The generated Sonic policy below failed deterministic preflight validation.

Invalid candidate:
```python
{candidate_code}
```

Exact validator feedback:
{validation_error}

Repair ONLY the validation failure while preserving the candidate's intended
behavior and structure. The result must define a top-level `get_action(state)`.
Imports are forbidden except optional `import policies.skills as skills`.
Do not use filesystem, process, network, dynamic-code-execution, or dunder APIs.

Return ONLY valid Python code, starting with `def get_action(state):` or the
optional allowed skills import followed by that function.
"""
        raw_response, reasoning = self._call_micro_model(prompt, temperature=0.2)
        if "```python" in raw_response:
            raw_response = raw_response.split("```python")[-1].split("```")[0].strip()
        elif "```" in raw_response:
            parts = raw_response.split("```")
            raw_response = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()
        return raw_response, reasoning

    def crossover_policies(self, policy_a_code, policy_b_code, recent_history, temperature=0.7):
        history_text = json.dumps(recent_history, indent=2)

        prompt = f"""
We are performing an Evolutionary Algorithm Crossover. We have two highly successful policies (Parent A and Parent B) that each excel in different areas.

Parent A Code:
```python
{policy_a_code}
```

Parent B Code:
```python
{policy_b_code}
```

Recent Failure History of the population (for context on what hazards exist):
{history_text}

Your task is to merge the best logical traits of Parent A and Parent B into a single, superior offspring policy.
Analyze how they handle jumping, speed, and hazards, and combine their strengths while resolving any conflicting logic.
Return ONLY valid Python code, starting with `def get_action(state):`.

[SYSTEM CACHE BREAKER: {os.urandom(8).hex()} - Ignore this random string and DO NOT write it into your code.]
"""

        raw_response, reasoning = self._call_micro_model(prompt, temperature)
        reasoning = "FunSearch Crossover Offspring"

        print(f"Raw Response from LLM (crossover): {repr(raw_response)}")
        return extract_policy_code(raw_response), reasoning
