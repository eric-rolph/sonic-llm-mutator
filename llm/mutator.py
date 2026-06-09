import ast
import base64
import importlib
import json
import mimetypes
import os
import re
import sys
import tempfile

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from core.policy_validator import validate_skills_source
from core.trace_context import trace_entry_x
from llm.prompts import SYSTEM_PROMPT


def _top_level_functions(source):
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source or "").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _atomic_write_text(filepath, text):
    directory = os.path.dirname(filepath) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".skills-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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


def normalize_lesson(entry):
    """Return a compact semantic lesson dict, or None for malformed entries."""
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
    return {"x": x, "hazard": hazard, "lesson": lesson}


def dedupe_lessons(lessons, x_tolerance=25):
    """Collapse malformed and duplicate nearby lessons while preserving order."""
    deduped = []
    for lesson in lessons:
        normalized = normalize_lesson(lesson)
        if normalized is None:
            continue

        lesson_key = normalized["lesson"].casefold()
        duplicate = False
        for existing in deduped:
            if (
                abs(existing["x"] - normalized["x"]) <= x_tolerance
                and existing["lesson"].casefold() == lesson_key
            ):
                duplicate = True
                break
        if not duplicate:
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


def select_relevant_lessons(lessons, current_x, radius=1000, limit=10):
    relevant = []
    for lesson in dedupe_lessons(lessons):
        if abs(lesson["x"] - current_x) <= radius:
            relevant.append(f"- {lesson['lesson']}")
    return relevant[-limit:]


def normalize_vision_context(content):
    if not content:
        return "UNKNOWN"
    normalized = str(content).strip().upper()
    return normalized or "UNKNOWN"


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

        os.makedirs("memory", exist_ok=True)
        bank_path = "memory/semantic_bank.json"
        bank = load_semantic_bank(bank_path)
        bank.append(lesson_data)
        bank = dedupe_lessons(bank)
        with open(bank_path, "w", encoding="utf-8") as f:
            json.dump(bank, f, indent=4)
        print(f"Extracted and saved semantic lesson: {lesson_data.get('lesson')}")

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
                if "```python" in new_skills_code:
                    new_skills_code = new_skills_code.split("```python")[-1].split("```")[0].strip()
                elif "```" in new_skills_code:
                    parts = new_skills_code.split("```")
                    new_skills_code = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()

                existing_functions = _top_level_functions(existing_skills)
                new_functions = _top_level_functions(new_skills_code)

                new_ast = ast.parse(new_skills_code)
                added_funcs_code = []
                for node in new_ast.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name not in existing_functions:
                            source = ast.get_source_segment(new_skills_code, node)
                            if source:
                                added_funcs_code.append(source)

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
                _atomic_write_text(skills_path, combined_skills)
                importlib.invalidate_caches()
                loaded_skills = sys.modules.get("policies.skills")
                try:
                    if loaded_skills is None:
                        importlib.import_module("policies.skills")
                    else:
                        importlib.reload(loaded_skills)
                except Exception:
                    if existing_skills:
                        _atomic_write_text(skills_path, existing_skills)
                    elif os.path.exists(skills_path):
                        os.remove(skills_path)
                    importlib.invalidate_caches()
                    if loaded_skills is not None:
                        importlib.reload(loaded_skills)
                    raise
                print("Extracted and saved new skills to policies/skills.py")
            except Exception as e:
                print(f"Failed to save extracted skills: {e}")

    def mutate_policy(self, current_code, failure_reason, screenshot_path, recent_history, temperature=0.7, coordinate_trace=None):
        history_text = json.dumps(recent_history, indent=2)
        trace_text = ""
        current_x = 0
        if coordinate_trace:
            trace_text = f"Recent frame trace leading to failure: {coordinate_trace}"
            if len(coordinate_trace) > 0:
                current_x = trace_entry_x(coordinate_trace[-1])

        lessons_text = ""
        if os.path.exists("memory/semantic_bank.json"):
            try:
                bank = load_semantic_bank("memory/semantic_bank.json")
                relevant_lessons = select_relevant_lessons(bank, current_x, radius=1000, limit=10)
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

        # Clean up markdown if the LLM wrapped it anyway
        print(f"Raw Response from LLM (mutate): {repr(raw_response)}")
        if "```python" in raw_response:
            raw_response = raw_response.split("```python")[-1].split("```")[0].strip()
        elif "```" in raw_response:
            parts = raw_response.split("```")
            raw_response = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()

        return raw_response, reasoning

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
        if "```python" in raw_response:
            raw_response = raw_response.split("```python")[-1].split("```")[0].strip()
        elif "```" in raw_response:
            parts = raw_response.split("```")
            raw_response = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()

        return raw_response, reasoning
