"""Run an evolved policy's ``get_action`` under a hard wall-clock timeout.

LLM-generated policies occasionally contain infinite loops. Python cannot
forcibly kill a thread, so the previous approach (submit to a
``ThreadPoolExecutor`` every frame and ``shutdown(wait=False)`` on timeout) left
the runaway thread spinning a CPU core, and -- because executor workers are not
daemon threads -- could hang the whole process at interpreter exit while the
atexit handler tried to join it.

``PolicyRunner`` instead owns a single **daemon** worker thread that is reused
across frames. If a call exceeds its timeout the runner is marked broken: the
stuck daemon thread is abandoned (it dies with the process instead of blocking
exit) and the caller is expected to stop using this runner and move on to the
next candidate.
"""

import queue
import threading

_STOP = object()


class PolicyTimeout(Exception):
    """Raised when ``policy.get_action`` exceeds its allotted time."""


class PolicyRunner:
    def __init__(self, policy):
        self.policy = policy
        self._in = queue.Queue(maxsize=1)
        self._out = queue.Queue(maxsize=1)
        self._broken = False
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
            # The worker is still stuck on this call. Abandon it; being a daemon
            # thread it will not keep the interpreter alive.
            self._broken = True
            raise PolicyTimeout("policy timed out")
        if ok:
            return payload
        raise payload

    @property
    def broken(self):
        return self._broken

    def close(self):
        """Stop the worker thread if it is still healthy and idle."""
        if self._broken:
            return
        try:
            self._in.put_nowait(_STOP)
        except queue.Full:
            pass
