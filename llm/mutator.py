import os
import base64
import json
from llm.prompts import SYSTEM_PROMPT
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

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
        with open(filepath, 'w') as f:
            f.write(seed_code.strip())

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_macro_model(self, prompt, image_path):
        """Calls Cloud LLM for Macro-Mutations (needs vision)."""
        if not self.macro_client:
            print("No MACRO_API_KEY found, falling back to local Micro-Mutation model.")
            return self._call_micro_model(prompt)
            
        print(f"Using Cloud API ({self.macro_model}) for Macro-Mutation.")
        try:
            return self._do_macro_call(prompt, image_path)
        except Exception as e:
            print(f"Cloud API failed after retries: {e}")
            return self._call_micro_model(prompt)

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _do_macro_call(self, prompt, image_path):
        base64_image = self._encode_image(image_path)
        
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=2048,
            timeout=60
        )
        
        return response.choices[0].message.content, "Cloud vision analysis completed."


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
            max_tokens=2048,
            timeout=300
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("LLM returned an empty string. Likely a concurrency/queue failure.")
            
        return content, "Local inference completed."

    def extract_lesson(self, failure_reason, coordinate_trace):
        prompt = f"""
We failed with the following reason: {failure_reason}
Coordinate trace: {coordinate_trace}

Based on this failure, extract a single, concise, universal rule (a 'lesson learned') for the AI policy to avoid this in the future.
Format it EXACTLY as: "- When at X=[approximate], [action to take] to avoid [hazard]."
Return ONLY the single bullet point.
"""
        lesson, _ = self._call_micro_model(prompt, temperature=0.3)
        if lesson:
            os.makedirs("memory", exist_ok=True)
            with open("memory/lessons_learned.txt", "a") as f:
                f.write(f"{lesson.strip()}\n")
            print(f"Extracted and saved lesson: {lesson.strip()}")

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def analyze_environment(self, image_path):
        """Uses the Cloud VLM to proactively tag the current visual environment."""
        if not self.macro_client:
            return "UNKNOWN"
            
        base64_image = self._encode_image(image_path)
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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=10,
                timeout=20
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"Proactive vision analysis failed: {e}")
            return "UNKNOWN"



    def mutate_policy(self, current_code, failure_reason, screenshot_path, recent_history, temperature=0.7, coordinate_trace=None):
        history_text = json.dumps(recent_history, indent=2)
        trace_text = ""
        if coordinate_trace:
            trace_text = f"Recent coordinate trace (x, y) leading to failure: {coordinate_trace}"
            
        lessons_text = ""
        if os.path.exists("memory/lessons_learned.txt"):
            with open("memory/lessons_learned.txt", "r") as f:
                lessons_text = "Persisted Lessons Learned (CRITICAL - DO NOT VIOLATE):\n" + f.read()
        
        prompt = f"""
Here is the current code that failed:
```python
{current_code}
```

Failure Reason: {failure_reason}
{trace_text}

Recent History of Failures:
{history_text}

{lessons_text}

Note on Vision Context: The emulator now actively looks at the screen every 5 seconds. The immediate upcoming visual context is injected into `state['vision_context']` (e.g., 'ENEMY', 'CLEAR', 'SPIKES'). You can write logic to check this string!

Analyze the failure and rewrite `get_action(state)`. 
Return ONLY valid Python code, starting with `def get_action(state):`.

[SYSTEM CACHE BREAKER: {os.urandom(8).hex()} - Ignore this random string and DO NOT write it into your code.]
"""
        
        # Decide routing based on failure reason complexity
        if "stuck" in failure_reason.lower() or "timeout" in failure_reason.lower():
            # Likely a physics/logic bug, local model can handle
            raw_response, reasoning = self._call_micro_model(prompt, temperature)
        else:
            # Fatal error, died to enemy/pit. Needs visual analysis.
            raw_response, reasoning = self._call_macro_model(prompt, screenshot_path)
            
        # Clean up markdown if the LLM wrapped it anyway
        print(f"Raw Response from LLM (mutate): {repr(raw_response)}")
        if "```python" in raw_response:
            raw_response = raw_response.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()
            
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
            raw_response = raw_response.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()
            
        return raw_response, reasoning

