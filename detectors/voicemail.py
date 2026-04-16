"""Voicemail detector.

Uses a scoring system (0-100) that accumulates evidence across multiple
signals. When the score exceeds `vm_score_threshold` the call is declared
VOICEMAIL.

Score contributors
------------------
+25  Beep tone detected (narrow-band energy in 300-1200 Hz with high ratio)
+20  Short speech burst (0.5-4 s) followed by >= 1 s silence  → greeting pattern
+15  Spectral flatness > 0.55 over the last 3 s  → recording artifacts
+10  Very low zero-crossing rate variance  → highly periodic (recorded) speech
+10  Speech present in first 3 s but not in last 2 s  → greeting → invitation
+20  Two or more of the above patterns observed within 8 s  → corroborating evidence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from audio.features import (
    detect_tone,
    rms_dbfs,
    spectral_flatness,
    zero_crossing_rate,
    frame_audio,
)
from config import settings


class VMState(Enum):
    IDLE = auto()
    GREETING = auto()       # Speech detected — potential greeting
    POST_GREETING = auto()  # Silence after speech — waiting for beep/next clue
    CONFIRMED = auto()      # Score threshold reached


@dataclass
class VoicemailResult:
    is_voicemail: bool
    state: VMState
    score: int
    confidence: float
    signals: dict = field(default_factory=dict)


class VoicemailDetector:
    """Stateful voicemail detector using feature scoring + state machine."""

    def __init__(
        self,
        vm_score_threshold: int = settings.vm_score_threshold,
        beep_freq_low: float = settings.beep_freq_low,
        beep_freq_high: float = settings.beep_freq_high,
        beep_energy_ratio: float = settings.beep_energy_ratio,
        sample_rate: int = settings.analysis_sample_rate,
    ):
        self._threshold = vm_score_threshold
        self._beep_low = beep_freq_low
        self._beep_high = beep_freq_high
        self._beep_ratio = beep_energy_ratio
        self._sr = sample_rate

        # Scoring state
        self._score: int = 0
        self._state: VMState = VMState.IDLE
        self._signals: dict[str, bool] = {}

        # History tracking
        self._speech_burst_count: int = 0
        self._last_speech_end_s: float | None = None
        self._last_silence_after_speech_s: float = 0.0
        self._beep_detected: bool = False
        self._greeting_detected: bool = False

        # Per-update state
        self._prev_speech: bool = False
        self._speech_start_s: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        audio: np.ndarray,
        speech_ratio: float,
        elapsed_s: float,
    ) -> VoicemailResult:
        """Analyze the latest audio window.

        Args:
            audio: float32 PCM snapshot of recent audio
            speech_ratio: fraction of recent VAD frames classified as speech
            elapsed_s: total elapsed seconds since call started

        Returns:
            VoicemailResult (is_voicemail=True once score >= threshold)
        """
        if self._state == VMState.CONFIRMED:
            return self._make_result()

        is_speech = speech_ratio > 0.30

        # --- Track speech burst start / end --------------------------
        if is_speech and not self._prev_speech:
            self._speech_start_s = elapsed_s
        elif not is_speech and self._prev_speech:
            # Speech just ended
            if self._speech_start_s is not None:
                burst_dur = elapsed_s - self._speech_start_s
                self._last_speech_end_s = elapsed_s
                self._handle_speech_burst(burst_dur, elapsed_s)
        self._prev_speech = is_speech

        # --- State machine transitions --------------------------------
        if self._state == VMState.IDLE and is_speech:
            self._state = VMState.GREETING

        elif self._state == VMState.GREETING and not is_speech:
            if self._last_speech_end_s is not None:
                self._state = VMState.POST_GREETING

        # --- Feature checks ------------------------------------------
        self._check_beep(audio, elapsed_s)
        self._check_spectral_flatness(audio, elapsed_s)
        self._check_zcr_variance(audio, elapsed_s)
        self._check_greeting_pattern(is_speech, elapsed_s)

        # --- Corroboration bonus -------------------------------------
        n_signals = sum(self._signals.values())
        if n_signals >= 2 and "corroboration" not in self._signals:
            self._add_score(20, "corroboration")

        # --- Check threshold -----------------------------------------
        if self._score >= self._threshold:
            self._state = VMState.CONFIRMED

        return self._make_result()

    def reset(self) -> None:
        self._score = 0
        self._state = VMState.IDLE
        self._signals.clear()
        self._speech_burst_count = 0
        self._last_speech_end_s = None
        self._last_silence_after_speech_s = 0.0
        self._beep_detected = False
        self._greeting_detected = False
        self._prev_speech = False
        self._speech_start_s = None

    # ------------------------------------------------------------------
    # Feature checks
    # ------------------------------------------------------------------

    def _handle_speech_burst(self, burst_dur: float, elapsed_s: float) -> None:
        """Score a completed speech burst."""
        self._speech_burst_count += 1
        # Greeting-length burst: 0.5–6 s
        if 0.5 <= burst_dur <= 6.0 and "speech_burst" not in self._signals:
            self._add_score(20, "speech_burst")
            self._greeting_detected = True

    def _check_beep(self, audio: np.ndarray, elapsed_s: float) -> None:
        if self._beep_detected or len(audio) < self._sr * 0.1:
            return
        is_tone, freq, ratio = detect_tone(
            audio, self._sr, self._beep_low, self._beep_high, self._beep_ratio
        )
        if is_tone and "beep" not in self._signals:
            self._add_score(25, "beep")
            self._beep_detected = True

    def _check_spectral_flatness(self, audio: np.ndarray, elapsed_s: float) -> None:
        if "flatness" in self._signals or len(audio) < self._sr * 0.5:
            return
        flat = spectral_flatness(audio)
        if flat > 0.55:
            self._add_score(15, "flatness")

    def _check_zcr_variance(self, audio: np.ndarray, elapsed_s: float) -> None:
        if "zcr_variance" in self._signals or len(audio) < self._sr:
            return
        frame_size = int(self._sr * 0.03)  # 30 ms frames
        frames = frame_audio(audio, frame_size, frame_size)
        if len(frames) < 4:
            return
        zcrs = np.array([zero_crossing_rate(f) for f in frames])
        if zcrs.std() < 0.02:
            self._add_score(10, "zcr_variance")

    def _check_greeting_pattern(self, is_speech: bool, elapsed_s: float) -> None:
        """Detect: speech in first 3 s but silence in last 2 s."""
        if "greeting_pattern" in self._signals:
            return
        if (
            self._greeting_detected
            and not is_speech
            and self._last_speech_end_s is not None
            and (elapsed_s - self._last_speech_end_s) >= 1.0
        ):
            self._add_score(10, "greeting_pattern")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_score(self, points: int, signal: str) -> None:
        self._score = min(100, self._score + points)
        self._signals[signal] = True

    def _make_result(self) -> VoicemailResult:
        is_vm = self._state == VMState.CONFIRMED or self._score >= self._threshold
        confidence = min(1.0, self._score / 100.0)
        return VoicemailResult(
            is_voicemail=is_vm,
            state=self._state,
            score=self._score,
            confidence=confidence,
            signals=dict(self._signals),
        )
