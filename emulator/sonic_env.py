import retro
import cv2
import os
import json
import numpy as np

class SonicEnvWrapper:
    def __init__(self, state='GreenHillZone.Act1', record_path=None):
        self.env = retro.make(game='SonicTheHedgehog-Genesis', state=state, record=record_path)
        self.obs = self.env.reset()
        self.info = {}
        self.frame_count = 0
        
    def step(self, action):
        """
        Executes an action and advances the environment.
        action: array of shape (12,) representing button presses
        [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
        For Genesis: usually B, A, Mode, Start, Up, Down, Left, Right, C, Y, X, Z
        Stable-retro standard Genesis mapping: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
        """
        self.obs, reward, done, self.info = self.env.step(action)
        self.frame_count += 1
        return self.obs, reward, done, self.info

    def reset(self):
        self.obs = self.env.reset()
        self.info = {}
        self.frame_count = 0
        return self.obs

    def get_screenshot(self, filepath="artifacts/failures/latest_screenshot.png"):
        """Saves current observation as an image."""
        # Convert RGB (from retro) to BGR (for OpenCV)
        if self.obs is not None:
            bgr_img = cv2.cvtColor(self.obs, cv2.COLOR_RGB2BGR)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv2.imwrite(filepath, bgr_img)
            return filepath
        return None

    def get_state(self):
        """Returns the current relevant RAM values as a dict."""
        # Note: The exact variables depend on the data.json in stable-retro for Sonic 1.
        # Common ones are 'x', 'y', 'screen_x', 'screen_y', 'rings', 'lives', 'score'
        # To get velocity, we might need to calculate it if it's not exposed, or read it if it is.
        # For this prototype, we'll extract what we can and mock what isn't directly in data.json.
        return {
            "x_pos": self.info.get('x', 0),
            "y_pos": self.info.get('y', 0),
            "screen_x": self.info.get('screen_x', 0),
            "screen_y": self.info.get('screen_y', 0),
            "rings": self.info.get('rings', 0),
            "lives": self.info.get('lives', 3),
            "score": self.info.get('score', 0)
        }

    def close(self):
        self.env.close()

if __name__ == "__main__":
    env = SonicEnvWrapper()
    print("Environment initialized.")
    env.reset()
    action = env.env.action_space.sample()
    obs, rew, done, info = env.step(action)
    print(f"Step taken. Info: {info}")
    env.close()
