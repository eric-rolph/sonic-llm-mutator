import os

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def _numeric(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_trace_entry(frame, state, action_string):
    """Build compact frame context for mutation prompts and history logs."""
    return {
        "frame": int(frame),
        "x": _integer(state.get("x_pos", 0)),
        "y": _integer(state.get("y_pos", 0)),
        "zone": _integer(state.get("zone", 0)),
        "act": _integer(state.get("act", 0)),
        "screen_x": _integer(state.get("screen_x", 0)),
        "screen_y": _integer(state.get("screen_y", 0)),
        "x_velocity": round(_numeric(state.get("x_velocity", 0.0)), 3),
        "y_velocity": round(_numeric(state.get("y_velocity", 0.0)), 3),
        "rings": _integer(state.get("rings", 0)),
        "lives": _integer(state.get("lives", 0)),
        "score": _integer(state.get("score", 0)),
        "vision_context": str(state.get("vision_context", "UNKNOWN") or "UNKNOWN"),
        "action": str(action_string or ""),
    }


def trace_entry_x(entry):
    """Return the x coordinate from either new dict traces or legacy tuple traces."""
    if isinstance(entry, dict):
        return _integer(entry.get("x", entry.get("x_pos", 0)))
    if isinstance(entry, (list, tuple)) and entry:
        return _integer(entry[0])
    return 0


def trace_entry_zone_act(entry):
    """Return ``(zone, act)`` from a trace entry, or ``(None, None)`` when absent.

    Legacy tuple traces carry no level identity, so they yield ``(None, None)``
    rather than a fabricated zone 0 act 0.
    """
    if not isinstance(entry, dict):
        return None, None

    def level_int(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    return level_int(entry.get("zone")), level_int(entry.get("act"))


def build_screenshot_montage(image_paths, output_path, max_images=4, columns=2):
    """Combine recent screenshots into a small grid for visual mutation prompts."""
    if cv2 is None or np is None:
        return None

    images = []
    for image_path in image_paths[-max_images:]:
        if not image_path or not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

    if not images:
        return None

    height, width = images[-1].shape[:2]
    normalized = [cv2.resize(image, (width, height)) for image in images]
    while len(normalized) < max_images:
        normalized.insert(0, np.zeros((height, width, 3), dtype=np.uint8))

    columns = max(1, int(columns))
    rows = []
    for start in range(0, max_images, columns):
        rows.append(np.hstack(normalized[start:start + columns]))

    montage = np.vstack(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, montage)
    return output_path
