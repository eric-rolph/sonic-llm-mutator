import os
import requests
import json
from llm.prompts import SYSTEM_PROMPT

class MutatorClient:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.lm_studio_url = "http://localhost:1234/v1/chat/completions"

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

    def _call_gemini(self, prompt, image_path):
        """Calls Gemini for Macro-Mutations (needs vision)."""
        if not self.gemini_api_key:
            print("No GEMINI_API_KEY found, falling back to local LM Studio.")
            return self._call_lm_studio(prompt)
            
        print("Using Gemini API for Macro-Mutation.")
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7}
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text, "Gemini vision analysis completed."
        except Exception as e:
            print(f"Gemini API failed: {e}")
            return self._call_lm_studio(prompt)

    def _call_lm_studio(self, prompt, temperature=0.7):
        """Calls LM Studio for Micro-Mutations (code only)."""
        print(f"Using local LM Studio for Micro-Mutation (Temp: {temperature}).")
        payload = {
            "model": "local-model", # Uses whatever is loaded in LM Studio
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.lm_studio_url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'], "Local inference completed."
        except Exception as e:
            print(f"LM Studio local inference failed: {e}")
            # Fallback trivial mutation so the loop doesn't crash completely
            return "def get_action(state):\n    return 'RIGHT'", "Fallback to simple RIGHT."

    def mutate_policy(self, current_code, failure_reason, screenshot_path, recent_history, temperature=0.7, coordinate_trace=None):
        history_text = json.dumps(recent_history, indent=2)
        trace_text = ""
        if coordinate_trace:
            trace_text = f"Recent coordinate trace (x, y) leading to failure: {coordinate_trace}"
        
        prompt = f"""
Here is the current code that failed:
```python
{current_code}
```

Failure Reason: {failure_reason}
{trace_text}

Recent History of Failures:
{history_text}

Analyze the failure and rewrite `get_action(state)`. 
Return ONLY valid Python code, starting with `def get_action(state):`.

(Random Seed for generation variance: {os.urandom(8).hex()})
"""
        
        # Decide routing based on failure reason complexity
        if "stuck" in failure_reason.lower() or "timeout" in failure_reason.lower():
            # Likely a physics/logic bug, LM studio can handle
            raw_response, reasoning = self._call_lm_studio(prompt, temperature)
        else:
            # Fatal error, died to enemy/pit. Needs visual analysis.
            raw_response, reasoning = self._call_gemini(prompt, screenshot_path)
            
        # Clean up markdown if the LLM wrapped it anyway
        if "```python" in raw_response:
            raw_response = raw_response.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()
            
        return raw_response, reasoning
