import unittest
from pathlib import Path


class RunPipelineTests(unittest.TestCase):
    def test_powershell_runner_forwards_bounded_run_options(self):
        script = Path("run_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("python -u main.py --generations $Generations --frames $Frames", script)

    def test_powershell_runner_preserves_the_existing_frame_budget_by_default(self):
        script = Path("run_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("[int]$Frames = 12000", script)


if __name__ == "__main__":
    unittest.main()
