"""Circular audio buffer — accumulates raw PCM frames for windowed analysis."""

import collections
import threading
from typing import Optional

import numpy as np


class AudioBuffer:
    """Thread-safe circular buffer that stores float32 PCM samples at a fixed sample rate.

    Internally stores the last `capacity_seconds` of audio. Older samples are
    discarded automatically. Callers can snapshot the full buffer at any time
    without blocking incoming writes.
    """

    def __init__(self, capacity_seconds: float, sample_rate: int):
        self._sample_rate = sample_rate
        self._capacity = int(capacity_seconds * sample_rate)
        self._buf: collections.deque[float] = collections.deque(maxlen=self._capacity)
        self._lock = threading.Lock()
        self._total_samples_written = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, samples: np.ndarray) -> None:
        """Append float32 samples to the buffer (thread-safe)."""
        with self._lock:
            self._buf.extend(samples.tolist())
            self._total_samples_written += len(samples)

    def snapshot(self, seconds: Optional[float] = None) -> np.ndarray:
        """Return a copy of up to `seconds` of the most recent audio.

        If `seconds` is None or larger than available data, returns everything.
        """
        with self._lock:
            data = list(self._buf)

        if seconds is not None:
            n = int(seconds * self._sample_rate)
            data = data[-n:]

        return np.array(data, dtype=np.float32)

    def duration_seconds(self) -> float:
        with self._lock:
            return len(self._buf) / self._sample_rate

    def total_samples_written(self) -> int:
        with self._lock:
            return self._total_samples_written

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._total_samples_written = 0
