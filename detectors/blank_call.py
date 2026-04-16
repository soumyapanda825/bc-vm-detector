"""Blank call detector.

A call is declared BLANK when both conditions hold over the analysis window:
  1. RMS energy stays below `silence_dbfs` threshold.
  2. VAD speech ratio is below `speech_ratio_threshold`.

This dual-gate prevents false positives from line noise that would fool
energy-only detectors, and from very quiet speech that would fool VAD-only
detectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio.features import rms_dbfs
from config import settings


@dataclass
class BlankResult:
    is_blank: bool
    rms_db: float
    speech_ratio: float
    silence_duration_s: float
    confidence: float


class BlankCallDetector:
    """Stateful blank call detector.

    Call `update()` with each new audio window snapshot and the current VAD
    speech ratio. When `is_blank()` returns True, the call can be marked as
    BLANK.
    """

    def __init__(
        self,
        silence_dbfs: float = settings.silence_dbfs,
        blank_timeout_s: float = settings.blank_timeout_s,
        speech_ratio_threshold: float = settings.blank_speech_ratio_threshold,
        sample_rate: int = settings.analysis_sample_rate,
    ):
        self._silence_dbfs = silence_dbfs
        self._blank_timeout_s = blank_timeout_s
        self._speech_threshold = speech_ratio_threshold
        self._sample_rate = sample_rate

        # State
        self._silence_start_s: float | None = None  # elapsed time when silence began
        self._total_elapsed_s: float = 0.0
        self._last_result: BlankResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        audio: np.ndarray,
        speech_ratio: float,
        elapsed_s: float,
    ) -> BlankResult:
        """Analyze the latest audio snapshot.

        Args:
            audio: float32 PCM snapshot of recent audio (any length)
            speech_ratio: fraction of recent VAD frames classified as speech
            elapsed_s: total elapsed seconds since call started

        Returns:
            BlankResult with is_blank=True once the silence gate is exceeded.
        """
        self._total_elapsed_s = elapsed_s
        db = rms_dbfs(audio) if len(audio) > 0 else float("-inf")
        frame_silent = db < self._silence_dbfs and speech_ratio < self._speech_threshold

        if frame_silent:
            if self._silence_start_s is None:
                self._silence_start_s = elapsed_s
            silence_dur = elapsed_s - self._silence_start_s
        else:
            self._silence_start_s = None
            silence_dur = 0.0

        is_blank = silence_dur >= self._blank_timeout_s

        # Confidence: linearly scales from 0 at timeout start to 1 at 2× timeout
        if silence_dur >= self._blank_timeout_s:
            over = silence_dur - self._blank_timeout_s
            confidence = min(1.0, 0.5 + over / self._blank_timeout_s)
        else:
            confidence = 0.0

        self._last_result = BlankResult(
            is_blank=is_blank,
            rms_db=db,
            speech_ratio=speech_ratio,
            silence_duration_s=silence_dur,
            confidence=confidence,
        )
        return self._last_result

    def reset(self) -> None:
        self._silence_start_s = None
        self._total_elapsed_s = 0.0
        self._last_result = None
