import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunPipelineTests(unittest.TestCase):
    def test_powershell_runner_forwards_bounded_run_options(self):
        script = Path("run_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("python -u main.py --generations $Generations --frames $Frames", script)

    def test_powershell_runner_preserves_the_existing_frame_budget_by_default(self):
        script = Path("run_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("[int]$Frames = 12000", script)

    def test_powershell_runner_propagates_main_failure(self):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shutil.copy2("run_pipeline.ps1", root / "run_pipeline.ps1")
            (root / "venv38" / "Scripts").mkdir(parents=True)
            (root / "venv38" / "Scripts" / "Activate.ps1").write_text(
                "", encoding="utf-8"
            )

            retro_path = root / "retro"
            rom_path = (
                retro_path
                / "data"
                / "stable"
                / "SonicTheHedgehog-Genesis"
                / "rom.md"
            )
            rom_path.parent.mkdir(parents=True)
            rom_path.write_text("test rom marker", encoding="utf-8")

            bin_path = root / "bin"
            bin_path.mkdir()
            # Windows resolves `python` to python.cmd via PATHEXT; on Linux
            # pwsh resolves the executable extensionless stub. Without the
            # POSIX stub the real interpreter runs: the `-c` retro check then
            # accidentally succeeds (the fixture's retro/ directory in CWD is
            # importable as a namespace package) and `python -u main.py` fails
            # with CPython's exit code 2 instead of the stub's 23.
            (bin_path / "python.cmd").write_text(
                "@echo off\n"
                'if "%1"=="-c" (\n'
                f"  echo {retro_path}\n"
                "  exit /b 0\n"
                ")\n"
                "exit /b 23\n",
                encoding="ascii",
            )
            posix_stub = bin_path / "python"
            posix_stub.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "-c" ]; then\n'
                f'  echo "{retro_path}"\n'
                "  exit 0\n"
                "fi\n"
                "exit 23\n",
                encoding="ascii",
            )
            posix_stub.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bin_path}{os.pathsep}{env['PATH']}"

            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(root / "run_pipeline.ps1"),
                    "-Generations",
                    "1",
                    "-Frames",
                    "1",
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(23, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("Pipeline Simulation Complete.", result.stdout)


if __name__ == "__main__":
    unittest.main()
