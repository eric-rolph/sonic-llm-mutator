"""Failure-window savestates and the interactive diagnosis session.

During evaluation a :class:`FailureSnapshotRing` captures whole-machine
savestates at a fixed cadence (~10 s of history). When a run becomes the
working frontier its ring is persisted to disk as raw ``.state`` blobs plus a
``window.json`` manifest. A :class:`DiagnosisSession` then lets the vision
model (or a human via the MCP sidecar) seek to any captured moment, read the
authoritative state, take screenshots, and run counterfactual inputs — on a
dedicated non-recording env so experiments never pollute training recordings.
"""

import json
import os
import time

from core.actions import action_string_to_array
from core.fsio import atomic_write_text

SNAPSHOT_INTERVAL = 60        # frames between captures (~1 s at 60 fps)
SNAPSHOT_CAPACITY = 10        # ring depth: ~10 s of history before failure
TRY_ACTIONS_MAX_FRAMES = 300  # hard cap per counterfactual rollout (~5 s)
DEFAULT_WINDOW_DIR = "artifacts/diagnosis/window"
DEFAULT_SCREENSHOT_DIR = "artifacts/diagnosis"

_INFO_KEYS = ("x_pos", "y_pos", "zone", "act", "rings", "lives")


def _info_subset(state):
    state = state or {}
    subset = {}
    for key in _INFO_KEYS:
        try:
            subset[key] = int(float(state.get(key, 0)))
        except (TypeError, ValueError):
            subset[key] = 0
    return subset


class FailureSnapshotRing:
    """Rolling savestates captured during one evaluation episode."""

    def __init__(self, interval=SNAPSHOT_INTERVAL, capacity=SNAPSHOT_CAPACITY):
        self.interval = max(1, int(interval))
        self.capacity = max(1, int(capacity))
        self.snapshots = []
        self.last_seen = None  # (frame, info) of the newest state offered
        self._last_frame = None
        self.disabled = False

    def record(self, env, frame, state):
        """Capture a savestate at the cadence. Never raises into evaluation.

        Every offered state updates ``last_seen`` (the eventual failure
        moment); savestates are only captured at the cadence. A backend
        without savestate support disables the ring on first failure instead
        of paying a try/except per capture forever.
        """
        if self.disabled:
            return False
        self.last_seen = (int(frame), _info_subset(state))
        if self._last_frame is not None and frame - self._last_frame < self.interval:
            return False
        try:
            state_bytes = env.save_emulator_state()
        except Exception:
            self.disabled = True
            return False
        self._last_frame = frame
        self.snapshots.append(
            {
                "frame": int(frame),
                "state_bytes": state_bytes,
                "info": _info_subset(state),
            }
        )
        del self.snapshots[: -self.capacity]
        return True

    def persist(self, directory=DEFAULT_WINDOW_DIR, failure_reason="", final_state=None, failure_frame=None):
        """Write blobs + manifest, replacing any previous window. None if empty."""
        if not self.snapshots:
            return None
        if final_state is None and self.last_seen is not None:
            final_state = self.last_seen[1]
        if failure_frame is None and self.last_seen is not None:
            failure_frame = self.last_seen[0]
        os.makedirs(directory, exist_ok=True)
        for name in os.listdir(directory):
            if name.endswith(".state") or name == "window.json":
                try:
                    os.remove(os.path.join(directory, name))
                except OSError:
                    pass

        failure = _info_subset(final_state)
        if failure_frame is not None:
            failure["frame"] = int(failure_frame)
        manifest = {
            "failure_reason": str(failure_reason or ""),
            "created_at": int(time.time()),
            "failure": failure,
            "snapshots": [],
        }
        for snapshot in self.snapshots:
            filename = f"{snapshot['frame']}.state"
            with open(os.path.join(directory, filename), "wb") as f:
                f.write(snapshot["state_bytes"])
            entry = dict(snapshot["info"])
            entry["frame"] = snapshot["frame"]
            entry["file"] = filename
            manifest["snapshots"].append(entry)

        atomic_write_text(
            os.path.join(directory, "window.json"),
            json.dumps(manifest, indent=2),
        )
        return directory


def load_failure_window(directory=DEFAULT_WINDOW_DIR):
    """Read a persisted window; None when missing, corrupt, or blob-less."""
    try:
        with open(os.path.join(directory, "window.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None

    snapshots = []
    for entry in manifest.get("snapshots", []):
        if not isinstance(entry, dict):
            continue
        path = os.path.join(directory, str(entry.get("file", "")))
        if os.path.isfile(path):
            verified = dict(entry)
            verified["path"] = path
            snapshots.append(verified)
    if not snapshots:
        return None

    window = dict(manifest)
    window["snapshots"] = sorted(snapshots, key=lambda item: int(item.get("frame", 0)))
    window["directory"] = directory
    return window


def window_key(window):
    """Cache identity for one persisted window (changes when re-persisted)."""
    if not window:
        return None
    return f"{window.get('directory', '')}:{window.get('created_at', 0)}:{window.get('failure_reason', '')}"


def _default_env_factory():
    from emulator.sonic_env import SonicEnvWrapper

    return SonicEnvWrapper(record_path=None)


class DiagnosisSession:
    """Seek/inspect/experiment over a persisted failure window.

    Every operation returns a result dict with ``ok``/``text`` (and
    ``screenshot`` where applicable) instead of raising, so a confused model
    or a broken backend can never take down the training loop.
    """

    def __init__(self, window, env_factory=None, screenshot_dir=DEFAULT_SCREENSHOT_DIR):
        self.window = window
        self._env_factory = env_factory or _default_env_factory
        self._env = None
        self.screenshot_dir = screenshot_dir
        self._shot_count = 0
        self.last_screenshot = None

    def _ensure_env(self):
        if self._env is None:
            self._env = self._env_factory()
        return self._env

    def _drop_env(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def failure_frame(self):
        try:
            return int(self.window.get("failure", {}).get("frame", 0))
        except (TypeError, ValueError):
            return 0

    def failure_x(self):
        try:
            return int(self.window.get("failure", {}).get("x_pos", 0))
        except (TypeError, ValueError):
            return 0

    def describe_window(self):
        """Compact text table of the available moments, newest last."""
        failure = self.window.get("failure", {})
        lines = [
            "Available emulator savestates before the failure "
            "(offset = frames before the failure moment):",
        ]
        failure_frame = self.failure_frame()
        for snapshot in self.window.get("snapshots", []):
            offset = failure_frame - int(snapshot.get("frame", 0))
            lines.append(
                f"- offset={offset} frames: x={snapshot.get('x_pos', 0)} y={snapshot.get('y_pos', 0)} "
                f"zone={snapshot.get('zone', 0)} act={snapshot.get('act', 0)} "
                f"rings={snapshot.get('rings', 0)} lives={snapshot.get('lives', 0)}"
            )
        lines.append(
            f"Failure moment: frame={failure.get('frame', '?')} x={failure.get('x_pos', '?')} "
            f"y={failure.get('y_pos', '?')} zone={failure.get('zone', '?')} act={failure.get('act', '?')}"
        )
        return "\n".join(lines)

    def _nearest_snapshot(self, frames_before_failure):
        try:
            requested = max(0, int(frames_before_failure))
        except (TypeError, ValueError):
            requested = 0
        target = self.failure_frame() - requested
        snapshots = self.window.get("snapshots", [])
        at_or_before = [s for s in snapshots if int(s.get("frame", 0)) <= target]
        if at_or_before:
            return at_or_before[-1]
        return snapshots[0] if snapshots else None

    def _seek(self, snapshot):
        env = self._ensure_env()
        with open(snapshot["path"], "rb") as f:
            env.load_emulator_state(f.read())
        return env

    def _take_screenshot(self, env, tag):
        os.makedirs(self.screenshot_dir, exist_ok=True)
        path = os.path.join(self.screenshot_dir, f"diagnosis_{self._shot_count:02d}_{tag}.png")
        self._shot_count += 1
        saved = env.get_screenshot(path)
        if saved:
            self.last_screenshot = saved
        return saved

    def view_frame(self, frames_before_failure):
        """Seek to ~N frames before the failure; return state + screenshot."""
        snapshot = self._nearest_snapshot(frames_before_failure)
        if snapshot is None:
            return {"ok": False, "text": "No savestates are available in this window.", "screenshot": None}
        try:
            env = self._seek(snapshot)
            state = env.get_state()
            shot = self._take_screenshot(env, f"view_{snapshot['frame']}")
            offset = self.failure_frame() - int(snapshot.get("frame", 0))
            text = (
                f"Viewing {offset} frames before the failure. State: "
                + json.dumps(_info_subset(state))
            )
            return {"ok": True, "text": text, "screenshot": shot}
        except Exception as e:
            self._drop_env()
            return {"ok": False, "text": f"view_frame failed: {type(e).__name__}: {e}", "screenshot": None}

    def try_actions(self, frames_before_failure, actions, hold_frames):
        """Seek, hold an action string, and report what actually happened."""
        snapshot = self._nearest_snapshot(frames_before_failure)
        if snapshot is None:
            return {"ok": False, "text": "No savestates are available in this window.", "screenshot": None}
        try:
            hold = max(1, min(int(hold_frames), TRY_ACTIONS_MAX_FRAMES))
        except (TypeError, ValueError):
            hold = 60
        try:
            env = self._seek(snapshot)
            start = _info_subset(env.get_state())
            action = action_string_to_array(actions)
            max_x = start["x_pos"]
            frames_done = 0
            ended_early = False
            for _ in range(hold):
                obs, reward, done, info = env.step(action)
                frames_done += 1
                current_x = _info_subset(env.get_state())["x_pos"]
                max_x = max(max_x, current_x)
                if done:
                    ended_early = True
                    break
            end = _info_subset(env.get_state())
            shot = self._take_screenshot(env, f"try_{snapshot['frame']}")
            offset = self.failure_frame() - int(snapshot.get("frame", 0))
            passed_failure_x = max_x > self.failure_x()
            text = (
                f"Held '{actions}' for {frames_done} frames starting {offset} frames before the failure. "
                f"x: {start['x_pos']} -> {end['x_pos']} (max {max_x}), y: {start['y_pos']} -> {end['y_pos']}, "
                f"rings: {start['rings']} -> {end['rings']}, lives: {start['lives']} -> {end['lives']}. "
                f"Progressed past the failure x ({self.failure_x()}): {'YES' if passed_failure_x else 'no'}."
                + (" The episode ended during this experiment (death or level end)." if ended_early else "")
            )
            return {"ok": True, "text": text, "screenshot": shot, "passed_failure_x": passed_failure_x}
        except Exception as e:
            self._drop_env()
            return {"ok": False, "text": f"try_actions failed: {type(e).__name__}: {e}", "screenshot": None}

    def close(self):
        self._drop_env()
