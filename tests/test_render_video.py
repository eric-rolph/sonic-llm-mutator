import builtins
import importlib
import sys
import unittest
from unittest.mock import patch


class RenderVideoTests(unittest.TestCase):
    def test_import_does_not_require_deprecated_retro_module(self):
        sys.modules.pop("render_video", None)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "retro":
                raise ImportError("legacy retro import should be lazy")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            module = importlib.import_module("render_video")

        self.assertTrue(hasattr(module, "bk2_to_mp4"))


if __name__ == "__main__":
    unittest.main()
