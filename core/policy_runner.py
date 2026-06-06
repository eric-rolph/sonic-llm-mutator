"""Run an evolved policy's ``get_action`` under a hard wall-clock timeout.

Generated policies loaded from disk run in a persistent child process so a
timeout can terminate runaway CPU or memory work. Small in-memory test policies
fall back to a daemon thread because they are not reliably serializable across
platforms.
"""

import ctypes
import multiprocessing
import os
import queue
import threading

from core.policy_loader import load_policy

_STOP = object()
POLICY_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


class PolicyTimeout(Exception):
    """Raised when ``policy.get_action`` exceeds its allotted time."""


def _limit_current_process_memory(limit_bytes):
    if os.name == "nt":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _assign_windows_memory_job(process, limit_bytes):
    if os.name != "nt":
        return None

    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )]

    class BASIC_LIMITS(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMITS),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    limits = EXTENDED_LIMITS()
    limits.BasicLimitInformation.LimitFlags = 0x100 | 0x2000
    limits.ProcessMemoryLimit = limit_bytes
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel32.CloseHandle(job)
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(process.sentinel))):
        kernel32.CloseHandle(job)
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    return job


def _process_worker(policy_path, input_queue, output_queue, ready_queue, memory_limit_bytes):
    try:
        _limit_current_process_memory(memory_limit_bytes)
        policy = load_policy(policy_path)
    except Exception as exc:  # noqa: BLE001
        ready_queue.put((False, f"{type(exc).__name__}: {exc}"))
        return
    ready_queue.put((True, "ready"))
    while True:
        state = input_queue.get()
        if state is None:
            return
        try:
            action = policy.get_action(state)
            if type(action) is not str:
                raise TypeError(f"get_action returned non-string {type(action).__name__}")
            if len(action) > 128:
                raise ValueError(
                    f"get_action returned oversized action string ({len(action)} characters)"
                )
            output_queue.put((True, action))
        except Exception as exc:  # noqa: BLE001
            output_queue.put((False, f"{type(exc).__name__}: {exc}"))


class PolicyRunner:
    def __init__(self, policy):
        self.policy = policy
        self._broken = False
        policy_path = getattr(policy, "__file__", None)
        self._process = None
        self._thread = None
        self._job_handle = None
        self._memory_limited = False
        if policy_path and os.path.isfile(policy_path):
            context = multiprocessing.get_context("spawn")
            self._in = context.Queue(maxsize=1)
            self._out = context.Queue(maxsize=1)
            ready_queue = context.Queue(maxsize=1)
            self._process = context.Process(
                target=_process_worker,
                args=(
                    policy_path,
                    self._in,
                    self._out,
                    ready_queue,
                    POLICY_MEMORY_LIMIT_BYTES,
                ),
                name="policy-runner",
                daemon=True,
            )
            self._process.start()
            try:
                self._job_handle = _assign_windows_memory_job(
                    self._process,
                    POLICY_MEMORY_LIMIT_BYTES,
                )
                self._memory_limited = True
            except Exception:
                self._terminate_process()
                raise
            try:
                ready, message = ready_queue.get(timeout=5.0)
            except queue.Empty as exc:
                self._terminate_process()
                raise PolicyTimeout("policy process failed to start") from exc
            if not ready:
                self._terminate_process()
                raise RuntimeError(message)
        else:
            self._in = queue.Queue(maxsize=1)
            self._out = queue.Queue(maxsize=1)
            self._thread = threading.Thread(
                target=self._worker, name="policy-runner", daemon=True
            )
            self._thread.start()

    def _worker(self):
        while True:
            state = self._in.get()
            if state is _STOP:
                return
            try:
                self._out.put((True, self.policy.get_action(state)))
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller thread
                self._out.put((False, exc))

    def get_action(self, state, timeout):
        """Return the policy's action, or raise.

        Raises ``PolicyTimeout`` if the call takes longer than ``timeout``
        seconds (the runner is then permanently broken), or re-raises whatever
        exception the policy itself threw.
        """
        if self._broken:
            raise PolicyTimeout("policy runner abandoned after an earlier timeout")
        self._in.put(state)
        try:
            ok, payload = self._out.get(timeout=timeout)
        except queue.Empty:
            self._broken = True
            self._terminate_process()
            raise PolicyTimeout("policy timed out")
        if ok:
            return payload
        if self.process_isolation:
            raise RuntimeError(payload)
        raise payload

    @property
    def broken(self):
        return self._broken

    @property
    def process_isolation(self):
        return self._process is not None

    @property
    def worker_alive(self):
        if self._process is not None:
            return self._process.is_alive()
        return self._thread is not None and self._thread.is_alive()

    @property
    def memory_limited(self):
        return self._memory_limited

    def _close_job(self):
        if self._job_handle is not None and os.name == "nt":
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._job_handle)
            self._job_handle = None

    def _terminate_process(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._close_job()

    def close(self):
        """Stop the worker if it is still healthy and idle."""
        if self._broken:
            self._terminate_process()
            return
        if self._process is not None:
            try:
                self._in.put_nowait(None)
            except queue.Full:
                pass
            self._process.join(timeout=1.0)
            self._terminate_process()
            return
        try:
            self._in.put_nowait(_STOP)
        except queue.Full:
            pass
