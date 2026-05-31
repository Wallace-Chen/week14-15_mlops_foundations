from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator
from uuid import uuid4

logger = logging.getLogger("financial_mlops")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def new_request_id() -> str:
    """Create a lightweight request id for logs and responses."""
    return str(uuid4())


@contextmanager
def latency_timer() -> Iterator[Dict[str, float]]:
    """Measure elapsed time in milliseconds."""
    start = time.perf_counter()
    result: Dict[str, float] = {"latency_ms": 0.0}
    try:
        yield result
    finally:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured JSON log line."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True, default=str))
