from contextlib import contextmanager
from time import perf_counter

from loguru import logger


@contextmanager
def track_latency(stage: str):
    """Record elapsed wall time, including failed dependency calls."""
    started = perf_counter()
    try:
        yield
    finally:
        elapsed = (perf_counter() - started) * 1000
        logger.bind(stage=stage, latency_ms=elapsed).info(
            "{} completed in {:.2f} ms", stage, elapsed
        )
