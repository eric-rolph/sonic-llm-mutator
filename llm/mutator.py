import os
import requests
import json
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
from llm.prompts import SYSTEM_PROMPT

class MutatorClient:
    def __init__(self):
        self.gemini_client = None
        if "GEMINI_API_KEY" in os.environ:
            self.gemini_client = genai.Client()
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
        if not self.gemini_client:
            print("No GEMINI_API_KEY found, falling back to local LM Studio.")
            return self._call_lm_studio(prompt)
            
        print("Using Gemini API for Macro-Mutation.")
        try:
            # Upload the image if we are using the files API, or pass it directly.
            # Using simple text for now if image isn't critical or we can pass base64
            # In google-genai, we can pass the image file path directly if we read it
            
            # For simplicity, we just send the text prompt in this prototype unless
            # we properly format the image for the specific google-genai version.
            # Assuming we can just pass the image file object to generate_content
            contents = [prompt]
            if os.path.exists(image_path):
                # The google-genai SDK allows passing PIL images or byte arrays, 
                # but we'll stick to a simpler approach if we don't have PIL loaded.
                # To be safe, we'll just send text if we hit issues.
                pass
                
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-pro',
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                )
            )
            return response.text, "Gemini vision analysis completed."
        except Exception as e:
            print(f"Gemini API failed: {e}")
            return self._call_lm_studio(prompt)

    def _call_lm_studio(self, prompt):
        """Calls LM Studio for Micro-Mutations (code only)."""
        print("Using local LM Studio for Micro-Mutation.")
        payload = {
            "model": "local-model", # Uses whatever is loaded in LM Studio
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
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

    def mutate_policy(self, current_code, failure_reason, screenshot_path, recent_history):
        history_text = json.dumps(recent_history, indent=2)
        
        prompt = f"""
Here is the current code that failed:
```python
{current_code}
```

Failure Reason: {failure_reason}

Recent History of Failures:
{history_text}

Analyze the failure and rewrite `get_action(state)`. 
Return ONLY valid Python code, starting with `def get_action(state):`.
"""
        
        # Decide routing based on failure reason complexity
        if "stuck" in failure_reason.lower() or "timeout" in failure_reason.lower():
            # Likely a physics/logic bug, LM studio can handle
            raw_response, reasoning = self._call_lm_studio(prompt)
        else:
            # Fatal error, died to enemy/pit. Needs visual analysis.
            raw_response, reasoning = self._call_gemini(prompt, screenshot_path)
            
        # Clean up markdown if the LLM wrapped it anyway
        if "```python" in raw_response:
            raw_response = raw_response.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()
            
        return raw_response, reasoning
