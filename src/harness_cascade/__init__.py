"""Harness Cascade: headless, provider-native agent handoffs."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import Any

__version__ = "0.3.0"


def run(args: Sequence[str]) -> int:
    if isinstance(args, (str, bytes)):
        raise TypeError("args must be a sequence of arguments, not str or bytes")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "harness_cascade.run must run on Python's main thread "
            "so it can relay stop signals"
        )
    argv = tuple(args)

    child: subprocess.Popen[bytes] | None = None
    pending: int | None = None
    forwarded = False
    spawn_failed = False

    def on_stop(signum: int, _frame: object) -> None:
        nonlocal pending, forwarded
        if forwarded or pending is not None:
            return
        proc = child
        if proc is None:
            # Popen can still be inside C when the signal is delivered, so
            # `child` is not visible yet. Hold the first stop and send it
            # once the process exists.
            pending = signum
            return
        forwarded = True
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    previous: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            # An ignored signal is the caller's choice. Installing a handler
            # would undo it.
            if signal.getsignal(signum) is signal.SIG_IGN:
                continue
            old = signal.signal(signum, on_stop)
            # A handler installed from C is reported as None and cannot be
            # put back. Restore the default action instead of leaving ours.
            previous[signum] = signal.SIG_DFL if old is None else old
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "harness_cascade", *argv],
                start_new_session=True,
            )
        except BaseException:
            spawn_failed = True
            raise
        if pending is not None and not forwarded:
            forwarded = True
            try:
                child.send_signal(pending)
            except ProcessLookupError:
                pass
        status = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if spawn_failed and pending is not None:
            signal.raise_signal(pending)

    # A signal that killed the child shows up as a negative wait status.
    # The CLI reports that same event as 128 plus the signal number.
    if status < 0:
        return 128 - status
    return status
