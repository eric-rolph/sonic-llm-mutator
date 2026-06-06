import builtins
import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch


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

    @staticmethod
    def _renderer_dependencies():
        retro = MagicMock()
        retro.Actions.ALL = "all"
        movie = retro.Movie.return_value
        movie.get_game.return_value = "SonicTheHedgehog-Genesis"
        movie.get_state.return_value = b"state"
        movie.players = 1
        movie.step.side_effect = [True, True, False]

        env = MagicMock()
        env.observation_space.shape = (224, 320, 3)
        env.num_buttons = 1

        process = MagicMock()
        process.wait.return_value = 0
        return retro, movie, env, process

    def test_closes_environment_and_encoder_after_success(self):
        import render_video

        retro, _, env, process = self._renderer_dependencies()
        env.step.return_value = (MagicMock(), 0, False, {})

        with patch.object(
            render_video, "resolve_backend_module", return_value=(retro, "")
        ), patch.object(
            render_video, "make_retro_env", return_value=env
        ), patch.object(
            render_video.subprocess, "Popen", return_value=process
        ):
            render_video.bk2_to_mp4("input.bk2", "output.mp4")

        process.stdin.close.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=120)
        env.close.assert_called_once_with()

    def test_closes_environment_and_encoder_after_render_failure(self):
        import render_video

        retro, _, env, process = self._renderer_dependencies()
        env.step.side_effect = RuntimeError("render failed")

        with patch.object(
            render_video, "resolve_backend_module", return_value=(retro, "")
        ), patch.object(
            render_video, "make_retro_env", return_value=env
        ), patch.object(
            render_video.subprocess, "Popen", return_value=process
        ):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                render_video.bk2_to_mp4("input.bk2", "output.mp4")

        process.stdin.close.assert_called_once_with()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=120)
        env.close.assert_called_once_with()

    def test_waits_for_encoder_when_termination_reports_it_already_exited(self):
        import render_video

        retro, _, env, process = self._renderer_dependencies()
        env.step.side_effect = RuntimeError("render failed")
        process.terminate.side_effect = OSError("encoder already exited")

        with patch.object(
            render_video, "resolve_backend_module", return_value=(retro, "")
        ), patch.object(
            render_video, "make_retro_env", return_value=env
        ), patch.object(
            render_video.subprocess, "Popen", return_value=process
        ):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                render_video.bk2_to_mp4("input.bk2", "output.mp4")

        process.wait.assert_called_once_with(timeout=120)
        env.close.assert_called_once_with()

    def test_raises_when_encoder_exits_nonzero(self):
        import render_video

        retro, _, env, process = self._renderer_dependencies()
        env.step.return_value = (MagicMock(), 0, False, {})
        process.wait.return_value = 7

        with patch.object(
            render_video, "resolve_backend_module", return_value=(retro, "")
        ), patch.object(
            render_video, "make_retro_env", return_value=env
        ), patch.object(
            render_video.subprocess, "Popen", return_value=process
        ):
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                render_video.bk2_to_mp4("input.bk2", "output.mp4")


if __name__ == "__main__":
    unittest.main()
