"""Shared filesystem helpers."""

import os
import tempfile


def atomic_write_text(filepath, text, prefix=".tmp-"):
    """Write UTF-8 text without exposing a truncated destination on failure."""
    directory = os.path.dirname(filepath) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
