"""Analysis pipeline — orchestrates audio intake, VAD, and detectors.

One Pipeline instance is created per call leg (WebSocket connection).
The caller feeds raw audio bytes, and after sufficient evidence the
pipeline emits a CallDecision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from loguru import logger

from audio.buffer import AudioBuffer
from audio.preprocessor import Preprocessor
from config import settings
from detectors.blank_call import BlankCallDetector
from detectors.vad import VAD
from detectors.voicemail import VoicemailDetector


class CallType(str, Enum):
    BLANK = "BLANK"
    VOICEMAIL = "VOICEMAIL"
    LIVE = "LIVE"
    ANALYZING = "ANALYZING"  # not yet decided


@dataclass
class CallDecision:
    call_uuid: str
    result: CallType
    confidence: float
    detected_at_ms: int   # ms elapsed from call start
    details: dict


class Pipeline:
    """Per-call analysis pipeline."""

    _ANALYSIS_WINDOW_S = 4.0  # seconds of audio fed to detectors each cycle
    _MIN_AUDIO_S = 0.5        # need at least this much audio before analysing

    def __init__(self, call_uuid: str):
        self.call_uuid = call_uuid
        self._start_time = time.monotonic()
        self._decided = False
        self._decision: CallDecision | None = None

        self._preprocessor = Preprocessor(
            encoding=settings.input_encoding,
            input_sr=settings.input_sample_rate,
            output_sr=settings.analysis_sample_rate,
        )
        self._buffer = AudioBuffer(
            capacity_seconds=settings.analysis_timeout_s + 5,
            sample_rate=settings.analysis_sample_rate,
        )
        self._vad = VAD(
            sample_rate=settings.analysis_sample_rate,
            frame_ms=settings.frame_ms,
            aggressiveness=settings.vad_aggressiveness,
        )
        self._blank_detector = BlankCallDetector()
        self._vm_detector = VoicemailDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, raw: bytes) -> CallDecision | None:
        """Feed a raw audio chunk into the pipeline.

        Returns a CallDecision when a definitive result is available,
        otherwise None.
        """
        if self._decided:
            return self._decision

        audio = self._preprocessor.process(raw)
        self._buffer.push(audio)
        self._vad.feed(audio)

        elapsed = self._elapsed_s()
        buf_dur = self._buffer.duration_seconds()

        if buf_dur < self._MIN_AUDIO_S:
            return None

        # --- Check analysis timeout → default to LIVE ----------------
        if elapsed >= settings.analysis_timeout_s:
            return self._finalize(
                CallType.LIVE,
                confidence=0.5,
                details={"reason": "analysis_timeout"},
            )

        # --- Run detectors on recent audio window --------------------
        window = self._buffer.snapshot(seconds=self._ANALYSIS_WINDOW_S)
        vad_window_frames = int(
            self._ANALYSIS_WINDOW_S * 1000 / settings.frame_ms
        )
        speech_ratio = self._vad.speech_ratio(last_n_frames=vad_window_frames)

        blank = self._blank_detector.update(window, speech_ratio, elapsed)
        if blank.is_blank:
            return self._finalize(
                CallType.BLANK,
                confidence=blank.confidence,
                details={
                    "rms_db": blank.rms_db,
                    "speech_ratio": blank.speech_ratio,
                    "silence_duration_s": blank.silence_duration_s,
                },
            )

        vm = self._vm_detector.update(window, speech_ratio, elapsed)
        if vm.is_voicemail:
            return self._finalize(
                CallType.VOICEMAIL,
                confidence=vm.confidence,
                details={
                    "vm_score": vm.score,
                    "signals": vm.signals,
                    "vm_state": vm.state.name,
                },
            )

        return None

    @property
    def decided(self) -> bool:
        return self._decided

    @property
    def decision(self) -> CallDecision | None:
        return self._decision

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _elapsed_s(self) -> float:
        return time.monotonic() - self._start_time

    def _finalize(
        self,
        call_type: CallType,
        confidence: float,
        details: dict,
    ) -> CallDecision:
        elapsed_ms = int(self._elapsed_s() * 1000)
        self._decided = True
        self._decision = CallDecision(
            call_uuid=self.call_uuid,
            result=call_type,
            confidence=round(confidence, 4),
            detected_at_ms=elapsed_ms,
            details=details,
        )
        logger.info(
            "Decision for {} → {} (conf={:.2f}, t={}ms)",
            self.call_uuid,
            call_type.value,
            confidence,
            elapsed_ms,
        )
        return self._decision
