import os
import time

from mcp.server.fastmcp import FastMCP

from core.actions import action_string_to_array
from core.diagnosis import DiagnosisSession, load_failure_window, window_key
from emulator.sonic_env import SonicEnvWrapper

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


# Diagnosis session over the persisted failure window. Uses its own env (via
# the DiagnosisSession default factory) so seeks/experiments never disturb the
# interactive env above. Rebuilt whenever training persists a newer window.
_diagnosis_session = None

def get_diagnosis_session():
    global _diagnosis_session
    window = load_failure_window()
    if window is None:
        return None, (
            "No persisted failure window found. Run the training loop until a "
            "visual failure becomes the working frontier first."
        )
    if _diagnosis_session is None or window_key(_diagnosis_session.window) != window_key(window):
        if _diagnosis_session is not None:
            _diagnosis_session.close()
        _diagnosis_session = DiagnosisSession(window)
    return _diagnosis_session, None


def _format_session_result(result):
    text = str(result.get("text", ""))
    screenshot = result.get("screenshot")
    if screenshot:
        return f"{text}\nScreenshot: {os.path.abspath(screenshot)}"
    return text

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

    action_array = action_string_to_array(action_string)

    for _ in range(frames):
        obs, reward, done, info = env.step(action_array)
        if done:
            break

    return f"Stepped {frames} frames with action {action_string}. Current state: {env.get_state()}"

@mcp.tool()
def list_failure_window() -> str:
    """
    Lists the emulator savestates captured around the most recent training
    failure (the same window the mutator's agentic diagnosis uses).
    """
    session, error = get_diagnosis_session()
    if error:
        return error
    return session.describe_window()

@mcp.tool()
def view_failure_frame(frames_before_failure: int = 0) -> str:
    """
    Seeks the diagnosis emulator to ~N frames before the most recent failure.
    Returns the authoritative game state and the path to a screenshot.
    """
    session, error = get_diagnosis_session()
    if error:
        return error
    return _format_session_result(session.view_frame(frames_before_failure))

@mcp.tool()
def try_failure_actions(actions: str, frames_before_failure: int = 120, hold_frames: int = 60) -> str:
    """
    Counterfactual experiment at the most recent failure: rewind N frames
    before it, hold a button combination (e.g. 'RIGHT,B') for hold_frames,
    and report what actually happens — including whether Sonic progressed
    past the failure point.
    """
    session, error = get_diagnosis_session()
    if error:
        return error
    return _format_session_result(session.try_actions(frames_before_failure, actions, hold_frames))

if __name__ == "__main__":
    # Start the FastMCP stdio server
    mcp.run()
