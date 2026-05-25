from mcp.server.fastmcp import FastMCP
from emulator.sonic_env import SonicEnvWrapper
import os
import time

# Initialize the MCP Server
mcp = FastMCP("Sonic Emulator MCP")

# Global environment instance
# In a real setup, we might want to manage the lifecycle better, but for MCP 
# being invoked as a process, we can keep a singleton-like instance.
_env = None

def get_env():
    global _env
    if _env is None:
        _env = SonicEnvWrapper()
    return _env

@mcp.tool()
def reset_game() -> str:
    """Resets the Sonic game to the beginning of the level."""
    get_env().reset()
    return "Game reset successfully."

@mcp.tool()
def get_game_state() -> str:
    """Returns the current game state variables (X/Y coordinates, rings, score, etc)."""
    env = get_env()
    state = env.get_state()
    return str(state)

@mcp.tool()
def get_screenshot() -> str:
    """
    Captures a screenshot of the current game frame.
    Returns the absolute filepath to the captured image.
    """
    env = get_env()
    # Use absolute path to ensure LLM can read it
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    filepath = os.path.join(base_dir, "artifacts", "failures", f"screenshot_{int(time.time())}.png")
    saved_path = env.get_screenshot(filepath)
    if saved_path:
        return f"Screenshot saved to: {saved_path}"
    return "Failed to capture screenshot."

@mcp.tool()
def step_frames(action_string: str, frames: int = 1) -> str:
    """
    Steps the game forward by applying the given action for the specified number of frames.
    action_string: A comma-separated string of buttons to press (e.g., 'RIGHT,B')
    Valid buttons: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
    """
    env = get_env()
    
    # Genesis mapping: B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
    buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
    action_array = [0] * 12
    
    pressed = [b.strip().upper() for b in action_string.split(',')] if action_string else []
    for p in pressed:
        if p in buttons:
            action_array[buttons.index(p)] = 1
            
    for _ in range(frames):
        obs, reward, done, info = env.step(action_array)
        if done:
            break
            
    return f"Stepped {frames} frames with action {action_string}. Current state: {env.get_state()}"

if __name__ == "__main__":
    # Start the FastMCP stdio server
    mcp.run()
