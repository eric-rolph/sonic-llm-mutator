"""Shared Sega Genesis controller mapping.

Both the training loop (``main.py``) and the MCP server (``emulator/mcp_server.py``)
need to turn a comma-separated action string such as ``"RIGHT,B"`` into the
12-element button array the retro emulator expects. Keeping the mapping in one
place avoids the two copies drifting apart.
"""

# Standard stable-retro / gym-retro Genesis button order.
GENESIS_BUTTONS = [
    "B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z",
]


def action_string_to_array(action_string):
    """Convert ``"RIGHT,B"`` -> ``[0,1,0,...,1,...]`` (len == 12).

    Tokens are upper-cased and trimmed, so a policy that emits ``"right, b"``
    still works. Unknown tokens are ignored rather than raising, which keeps the
    evaluation loop resilient to noisy LLM output.
    """
    array = [0] * len(GENESIS_BUTTONS)
    if not action_string:
        return array
    for part in action_string.split(","):
        token = part.strip().upper()
        if token in GENESIS_BUTTONS:
            array[GENESIS_BUTTONS.index(token)] = 1
    return array
