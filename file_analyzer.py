"""Offline audio file analyzer.

Processes a local audio file (WAV, FLAC, OGG, etc.) through the same
detection pipeline as the real-time WebSocket service, but uses audio
timestamp instead of wall-clock time so the analysis completes instantly.

Supported formats: anything soundfile handles (WAV, FLAC, OGG, AIFF, CAF).
MP3 is also supported if librosa/audioread is available.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Union

import numpy as np

from audio.buffer import AudioBuffer
from audio.preprocessor import resample
from config import settings
from detectors.blank_call import BlankCallDetector
from detectors.vad import VAD
from detectors.voicemail import VoicemailDetector
from pipeline import CallDecision, CallType

# Process in 20 ms increments — one VAD frame at a time
_FRAME_S: float = settings.frame_ms / 1000.0
# Detectors look at the last 4 seconds of audio
_ANALYSIS_WINDOW_S: float = 4.0
# Require at least 0.5 s of audio before running detectors
_MIN_AUDIO_S: float = 0.5


def analyze_file(
    path: Union[str, Path],
    blank_timeout_s: float = settings.blank_timeout_s,
) -> CallDecision:
    """Analyze an audio file and classify it as BLANK, VOICEMAIL, or LIVE.

    Args:
        path: Path to audio file.
        blank_timeout_s: Continuous silence duration (seconds) to declare BLANK.
                         Defaults to the configured blank_timeout_s (7 s).

    Returns:
        CallDecision containing result, confidence, detected_at_ms, and details.
    """
    call_uuid = str(uuid.uuid4())
    path = Path(path)

    # ------------------------------------------------------------------
    # Load audio
    # ------------------------------------------------------------------
    audio, sample_rate = _load_audio(path)

    # Downmix to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to the analysis rate (16 kHz)
    if sample_rate != settings.analysis_sample_rate:
        audio = resample(audio, sample_rate, settings.analysis_sample_rate)
        sample_rate = settings.analysis_sample_rate

    total_duration_s = len(audio) / sample_rate

    # ------------------------------------------------------------------
    # Initialize pipeline components
    # ------------------------------------------------------------------
    buffer = AudioBuffer(
        capacity_seconds=total_duration_s + 5,
        sample_rate=sample_rate,
    )
    vad = VAD(
        sample_rate=sample_rate,
        frame_ms=settings.frame_ms,
        aggressiveness=settings.vad_aggressiveness,
    )
    blank_detector = BlankCallDetector(blank_timeout_s=blank_timeout_s)
    vm_detector = VoicemailDetector()

    # ------------------------------------------------------------------
    # Process frame by frame, advancing audio time not wall time
    # ------------------------------------------------------------------
    frame_samples = int(_FRAME_S * sample_rate)
    vad_window_frames = int(_ANALYSIS_WINDOW_S * 1000 / settings.frame_ms)

    pos = 0
    while pos < len(audio):
        chunk = audio[pos : pos + frame_samples]

        # Pad the last (short) frame so VAD always gets a full frame
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))

        # Use audio position as elapsed time (not wall clock)
        elapsed_s = pos / sample_rate

        buffer.push(chunk)
        vad.feed(chunk)
        pos += frame_samples

        if elapsed_s < _MIN_AUDIO_S:
            continue

        window = buffer.snapshot(seconds=_ANALYSIS_WINDOW_S)
        speech_ratio = vad.speech_ratio(last_n_frames=vad_window_frames)

        # --- Blank call check ----------------------------------------
        blank = blank_detector.update(window, speech_ratio, elapsed_s)
        if blank.is_blank:
            return CallDecision(
                call_uuid=call_uuid,
                result=CallType.BLANK,
                confidence=blank.confidence,
                detected_at_ms=int(elapsed_s * 1000),
                details={
                    "rms_db": _safe_float(blank.rms_db),
                    "speech_ratio": round(blank.speech_ratio, 4),
                    "silence_duration_s": round(blank.silence_duration_s, 2),
                    "total_duration_s": round(total_duration_s, 2),
                },
            )

        # --- Voicemail check -----------------------------------------
        vm = vm_detector.update(window, speech_ratio, elapsed_s)
        if vm.is_voicemail:
            return CallDecision(
                call_uuid=call_uuid,
                result=CallType.VOICEMAIL,
                confidence=vm.confidence,
                detected_at_ms=int(elapsed_s * 1000),
                details={
                    "vm_score": vm.score,
                    "signals": vm.signals,
                    "vm_state": vm.state.name,
                    "total_duration_s": round(total_duration_s, 2),
                },
            )

    # ------------------------------------------------------------------
    # No definitive signal found — default to LIVE
    # ------------------------------------------------------------------
    return CallDecision(
        call_uuid=call_uuid,
        result=CallType.LIVE,
        confidence=0.5,
        detected_at_ms=int(total_duration_s * 1000),
        details={
            "reason": "no_detection",
            "total_duration_s": round(total_duration_s, 2),
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(v: float, fallback: float = -120.0) -> float:
    """Return a JSON-safe float, replacing -inf / +inf / NaN with `fallback`."""
    import math
    return round(v, 2) if math.isfinite(v) else fallback


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, returning (float32 samples, sample_rate).

    Tries soundfile first; falls back to librosa for formats like MP3.
    """
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return audio, int(sr)
    except Exception:
        pass

    try:
        import librosa  # type: ignore
        audio, sr = librosa.load(str(path), sr=None, mono=False, dtype=np.float32)
        return audio, int(sr)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load audio file {path.name!r}. "
            "Ensure it is a valid WAV, FLAC, OGG, or AIFF file. "
            f"Underlying error: {exc}"
        ) from exc
