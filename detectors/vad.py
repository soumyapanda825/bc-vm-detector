"""WebRTC VAD wrapper with hangover smoothing.

webrtcvad processes fixed-size frames (10, 20, or 30 ms) at 8/16/32/48 kHz.
This wrapper handles frame slicing and applies a hangover window to avoid
choppy speech boundaries.
"""
from __future__ import annotations

import numpy as np
import webrtcvad

from config import settings


class VAD:
    """Stateful VAD processor.

    Accepts arbitrary-length float32 audio chunks at `sample_rate` Hz and
    maintains a rolling buffer of per-frame speech decisions.
    """

    def __init__(
        self,
        sample_rate: int = settings.analysis_sample_rate,
        frame_ms: int = settings.frame_ms,
        aggressiveness: int = settings.vad_aggressiveness,
        hangover_frames: int = 15,
    ):
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"webrtcvad requires 8000/16000/32000/48000 Hz, got {sample_rate}"
            )
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtcvad requires 10/20/30 ms frames, got {frame_ms}")

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._frame_size = int(sample_rate * frame_ms / 1000)  # samples per frame
        self._hangover = hangover_frames

        # Internal state
        self._pending: np.ndarray = np.array([], dtype=np.float32)
        self._decisions: list[bool] = []  # per-frame speech decisions (smoothed)
        self._raw_decisions: list[bool] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, audio: np.ndarray) -> None:
        """Feed float32 PCM samples into the VAD.

        Slices into frames, classifies each, and stores results.
        """
        self._pending = np.concatenate([self._pending, audio])

        while len(self._pending) >= self._frame_size:
            frame = self._pending[: self._frame_size]
            self._pending = self._pending[self._frame_size :]
            self._raw_decisions.append(self._classify_frame(frame))

        self._apply_hangover()

    def speech_ratio(self, last_n_frames: int | None = None) -> float:
        """Fraction of frames classified as speech (after hangover).

        Args:
            last_n_frames: look at only the most recent N frames; None = all.
        """
        decisions = self._decisions
        if last_n_frames is not None:
            decisions = decisions[-last_n_frames:]
        if not decisions:
            return 0.0
        return sum(decisions) / len(decisions)

    def is_speech(self) -> bool:
        """True if the most recent frame was classified as speech."""
        return bool(self._decisions[-1]) if self._decisions else False

    def total_frames(self) -> int:
        return len(self._decisions)

    def reset(self) -> None:
        self._pending = np.array([], dtype=np.float32)
        self._decisions.clear()
        self._raw_decisions.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_frame(self, frame: np.ndarray) -> bool:
        """Convert float32 frame to int16 PCM bytes and run webrtcvad."""
        pcm = (frame * 32767).astype(np.int16).tobytes()
        try:
            return self._vad.is_speech(pcm, self._sample_rate)
        except Exception:
            return False

    def _apply_hangover(self) -> None:
        """Smooth raw VAD decisions: extend speech regions by hangover frames."""
        raw = self._raw_decisions
        smoothed = list(raw)
        for i in range(len(raw)):
            if raw[i]:
                # extend forward
                for j in range(i, min(i + self._hangover, len(raw))):
                    smoothed[j] = True
        self._decisions = smoothed
