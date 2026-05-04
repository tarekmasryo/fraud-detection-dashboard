from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)


def _run_safely(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception("background_job_failed")


def submit_background(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run a small reference background task without blocking process shutdown.

    This keeps the public reference simple and prevents idle thread-pool workers
    from holding local tests or CI processes open after the test suite finishes.
    Durable production queues should replace this with Redis/RQ or Celery.
    """
    thread = Thread(
        target=_run_safely,
        args=(fn, *args),
        kwargs=kwargs,
        name="fraud-job",
        daemon=True,
    )
    thread.start()
