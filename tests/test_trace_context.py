import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch

import core.trace_context as trace_context
from core.trace_context import build_screenshot_montage


class TraceContextTests(unittest.TestCase):
    def test_build_screenshot_montage_combines_recent_frames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for index, value in enumerate((20, 80, 140, 200)):
                image = np.full((8, 8, 3), value, dtype=np.uint8)
                path = Path(tmp_dir) / f"frame_{index}.png"
                cv2.imwrite(str(path), image)
                paths.append(str(path))

            output_path = Path(tmp_dir) / "montage.png"
            result = build_screenshot_montage(paths, str(output_path))
            montage = cv2.imread(str(output_path))

        self.assertEqual(result, str(output_path))
        self.assertEqual(montage.shape, (16, 16, 3))

    def test_build_screenshot_montage_returns_none_without_cv2(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "frame.png"
            path.write_bytes(b"not an image")

            with patch.object(trace_context, "cv2", None):
                result = build_screenshot_montage([str(path)], str(Path(tmp_dir) / "montage.png"))

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
