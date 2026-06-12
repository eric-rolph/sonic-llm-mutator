"""Failure-window savestates and the interactive diagnosis session.

During evaluation a :class:`FailureSnapshotRing` captures whole-machine
savestates at a fixed cadence (~10 s of history). When a run becomes the
working frontier its ring is persisted to disk as raw ``.state`` blobs plus a
``window.json`` manifest. A :class:`DiagnosisSession` then lets the vision
model (or a human via the MCP sidecar) seek to any captured moment, read the
authoritative state, take screenshots, and run counterfactual inputs — on a
dedicated non-recording env so experiments never pollute training recordings.
"""

import importlib
import json
import multiprocessing
import os
import queue
import time

from core.actions import action_string_to_array
from core.fsio import atomic_write_text

SNAPSHOT_INTERVAL = 60        # frames between captures (~1 s at 60 fps)
SNAPSHOT_CAPACITY = 10        # ring depth: ~10 s of history before failure
TRY_ACTIONS_MAX_FRAMES = 300  # hard cap per counterfactual rollout (~5 s)
SEQUENCE_MAX_SEGMENTS = 5     # segments per try_action_sequence experiment
SEQUENCE_MAX_FRAMES = 600     # total frames per sequence experiment (~10 s)

# The emulator methods DiagnosisSession invokes through ProcessDiagnosisEnv.
# This tuple IS the proxy contract -- the single place to edit when the
# session needs a new emulator capability.
DIAGNOSIS_ENV_METHODS = ("load_emulator_state", "get_state", "step", "get_screenshot")
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
        # The run's true frontier WITHIN THE CURRENT ACT. x resets to ~0 every
        # act, so carrying a max across a transition judges Act-2 experiments
        # against Act-1 distances (live-observed: a real escape to x=3418 was
        # called a failure against a phantom frontier of 9767).
        self.max_x_seen = 0
        self._max_x_zone_act = None
        self._last_frame = None
        self.disabled = False

    def record(self, env, frame, state, act_max_x=None):
        """Capture a savestate at the cadence. Never raises into evaluation.

        Every offered state updates ``last_seen`` (the eventual failure
        moment); savestates are only captured at the cadence. A backend
        without savestate support disables the ring on first failure instead
        of paying a try/except per capture forever.

        ``act_max_x`` is the evaluator's authoritative per-act progress —
        prefer it: raw ``x_pos`` keeps reporting the PREVIOUS act's x for a
        while after a transition (live-observed: the ring re-ingested Act 1's
        x=9767 on frames already tagged act=1, despite resetting on the flag).
        The internal zone/act reset remains as a best-effort fallback for
        callers without that accounting.
        """
        if self.disabled:
            return False
        info = _info_subset(state)
        if act_max_x is not None:
            try:
                self.max_x_seen = int(act_max_x)
            except (TypeError, ValueError):
                pass
        else:
            zone_act = (info["zone"], info["act"])
            if self._max_x_zone_act != zone_act:
                self._max_x_zone_act = zone_act
                self.max_x_seen = 0
            self.max_x_seen = max(self.max_x_seen, info["x_pos"])
        self.last_seen = (int(frame), info)
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
        # The run's furthest x. The final resting x can be far behind it
        # (e.g. after a bounce-back), and experiments must be judged against
        # the real frontier, not the spot Sonic happened to die on.
        failure["frontier_x"] = max(self.max_x_seen, failure.get("x_pos", 0))
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
        name = str(entry.get("file", ""))
        # Containment: a window.json is an untrusted artifact (shareable,
        # MCP-touchable). Its file entries must be bare basenames inside the
        # window dir, never "../secret.state" reaching elsewhere on the host
        # and into the emulator via load_emulator_state.
        if not name or name != os.path.basename(name):
            continue
        path = os.path.join(directory, name)
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


def _default_child_env():
    """The real emulator the diagnosis worker process hosts."""
    from emulator.sonic_env import SonicEnvWrapper

    return SonicEnvWrapper(record_path=None)


def _import_callable(spec):
    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _diagnosis_env_worker(factory_spec, request_queue, response_queue):
    try:
        env = _import_callable(factory_spec)()
    except Exception as e:  # noqa: BLE001 - surfaced to the parent
        response_queue.put(("error", f"{type(e).__name__}: {e}"))
        return
    response_queue.put(("ready", None))
    while True:
        request = request_queue.get()
        if request is None:
            break
        method, args, kwargs = request
        try:
            result = getattr(env, method)(*args, **kwargs)
            if method == "step":
                # The observation frame is heavy to pickle and diagnosis only
                # reads RAM state plus screenshots the child writes to disk.
                result = (None,) + tuple(result[1:])
            response_queue.put(("ok", result))
        except Exception as e:  # noqa: BLE001 - surfaced to the parent
            response_queue.put(("error", f"{type(e).__name__}: {e}"))
    try:
        env.close()
    except Exception:
        pass


class ProcessDiagnosisEnv:
    """Hosts the diagnosis emulator in a child process and proxies calls.

    gym-retro (and stable-retro) allow only **one emulator instance per
    process**, and the training env already occupies the training process —
    creating a second in-process env raises RuntimeError, which is exactly
    what broke every diagnosis tool in live testing. A spawned child process
    gets its own instance; the proxy forwards the few methods
    :class:`DiagnosisSession` needs over queues (the same pattern
    ``core.policy_runner`` uses for policy isolation).
    """

    DEFAULT_FACTORY_SPEC = "core.diagnosis:_default_child_env"

    def __init__(self, factory_spec=DEFAULT_FACTORY_SPEC, start_timeout=90.0, call_timeout=60.0):
        self._call_timeout = call_timeout
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_diagnosis_env_worker,
            args=(factory_spec, self._requests, self._responses),
            name="diagnosis-env",
            daemon=True,
        )
        self._process.start()
        try:
            status, payload = self._responses.get(timeout=start_timeout)
        except queue.Empty:
            self._terminate()
            raise RuntimeError("Diagnosis env worker did not start in time.")
        if status != "ready":
            self._terminate()
            raise RuntimeError(f"Diagnosis env worker failed to start: {payload}")

    def _call(self, method, *args, **kwargs):
        self._requests.put((method, args, kwargs))
        try:
            status, payload = self._responses.get(timeout=self._call_timeout)
        except queue.Empty:
            self._terminate()
            raise RuntimeError(f"Diagnosis env call timed out: {method}")
        if status == "error":
            raise RuntimeError(payload)
        return payload

    def __getattr__(self, name):
        # Forward exactly the declared emulator methods to the worker. Adding a
        # method DiagnosisSession needs is a one-line edit to
        # DIAGNOSIS_ENV_METHODS -- the whole reason this proxy exists is that a
        # method the proxy silently failed to expose broke diagnosis in live
        # testing. Private/unknown attributes raise normally.
        if name in DIAGNOSIS_ENV_METHODS:
            def _forward(*args, **kwargs):
                return self._call(name, *args, **kwargs)
            return _forward
        raise AttributeError(name)

    def _terminate(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

    def close(self):
        try:
            self._requests.put_nowait(None)
        except Exception:
            pass
        if self._process is not None:
            self._process.join(timeout=2.0)
        self._terminate()


def _default_env_factory():
    return ProcessDiagnosisEnv()


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
        # Experiments that measurably beat the run's frontier. These are the
        # most valuable diagnosis output: they can be compiled directly into a
        # deterministic guard candidate without trusting an LLM translation.
        self.verified_experiments = []

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

    def frontier_x(self):
        """The run's furthest x; legacy windows fall back to the failure x."""
        failure = self.window.get("failure", {})
        try:
            return int(failure.get("frontier_x", failure.get("x_pos", 0)))
        except (TypeError, ValueError):
            return self.failure_x()

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
            passed_frontier_x = max_x > self.frontier_x()
            if passed_frontier_x and not ended_early:
                self.verified_experiments.append(
                    {
                        "zone": start["zone"],
                        "act": start["act"],
                        "start_x": start["x_pos"],
                        "actions": str(actions),
                        "hold_frames": frames_done,
                        "max_x": max_x,
                        "frames_before_failure": offset,
                    }
                )
            text = (
                f"Held '{actions}' for {frames_done} frames starting {offset} frames before the failure. "
                f"x: {start['x_pos']} -> {end['x_pos']} (max {max_x}), y: {start['y_pos']} -> {end['y_pos']}, "
                f"rings: {start['rings']} -> {end['rings']}, lives: {start['lives']} -> {end['lives']}. "
                f"Beat the run's furthest progress (frontier x={self.frontier_x()}): "
                f"{'YES — VERIFIED ESCAPE, this input will be compiled into a candidate policy' if passed_frontier_x else 'no'}."
                + (" The episode ended during this experiment (death or level end)." if ended_early else "")
            )
            return {"ok": True, "text": text, "screenshot": shot, "passed_frontier_x": passed_frontier_x}
        except Exception as e:
            self._drop_env()
            return {"ok": False, "text": f"try_actions failed: {type(e).__name__}: {e}", "screenshot": None}

    def try_action_sequence(self, frames_before_failure, segments):
        """Play a timed input sequence — the tool single holds cannot express.

        Sonic's jump fires on the B *press*: a held "RIGHT,B" jumps once at
        the start and never again, so "run, THEN jump at the edge" needs
        segments. Segment boundary x positions are measured so a verified
        sequence can compile into a stateless x-threshold guard.
        """
        snapshot = self._nearest_snapshot(frames_before_failure)
        if snapshot is None:
            return {"ok": False, "text": "No savestates are available in this window.", "screenshot": None}

        normalized = []
        total_frames = 0
        for segment in list(segments or [])[:SEQUENCE_MAX_SEGMENTS]:
            if not isinstance(segment, dict):
                continue
            try:
                frames = max(1, int(segment.get("frames", 0)))
            except (TypeError, ValueError):
                continue
            frames = min(frames, SEQUENCE_MAX_FRAMES - total_frames)
            if frames <= 0:
                break
            normalized.append({"actions": str(segment.get("actions", "")), "frames": frames})
            total_frames += frames
        if not normalized:
            return {"ok": False, "text": "No valid segments given. Each segment needs actions and frames.", "screenshot": None}

        try:
            env = self._seek(snapshot)
            start = _info_subset(env.get_state())
            max_x = start["x_pos"]
            ended_early = False
            played = []
            for segment in normalized:
                segment_start = _info_subset(env.get_state())
                action = action_string_to_array(segment["actions"])
                frames_done = 0
                for _ in range(segment["frames"]):
                    obs, reward, done, info = env.step(action)
                    frames_done += 1
                    max_x = max(max_x, _info_subset(env.get_state())["x_pos"])
                    if done:
                        ended_early = True
                        break
                played.append(
                    {
                        "actions": segment["actions"],
                        "frames": frames_done,
                        "start_x": segment_start["x_pos"],
                        "start_y": segment_start["y_pos"],
                    }
                )
                if ended_early:
                    break

            end = _info_subset(env.get_state())
            shot = self._take_screenshot(env, f"seq_{snapshot['frame']}")
            offset = self.failure_frame() - int(snapshot.get("frame", 0))
            passed_frontier_x = max_x > self.frontier_x()
            if passed_frontier_x and not ended_early:
                self.verified_experiments.append(
                    {
                        "zone": start["zone"],
                        "act": start["act"],
                        "start_x": start["x_pos"],
                        "actions": played[0]["actions"],
                        "segments": played,
                        "hold_frames": sum(p["frames"] for p in played),
                        "max_x": max_x,
                        "frames_before_failure": offset,
                    }
                )
            steps_text = "; ".join(
                f"'{p['actions']}' x{p['frames']} (from x={p['start_x']}, y={p['start_y']})" for p in played
            )
            text = (
                f"Played sequence [{steps_text}] starting {offset} frames before the failure. "
                f"x: {start['x_pos']} -> {end['x_pos']} (max {max_x}), y: {start['y_pos']} -> {end['y_pos']}. "
                f"Beat the run's furthest progress (frontier x={self.frontier_x()}): "
                f"{'YES — VERIFIED ESCAPE, this sequence will be compiled into a candidate policy' if passed_frontier_x else 'no'}."
                + (" The episode ended during this experiment (death or level end)." if ended_early else "")
            )
            return {"ok": True, "text": text, "screenshot": shot, "passed_frontier_x": passed_frontier_x}
        except Exception as e:
            self._drop_env()
            return {"ok": False, "text": f"try_action_sequence failed: {type(e).__name__}: {e}", "screenshot": None}

    def close(self):
        self._drop_env()
