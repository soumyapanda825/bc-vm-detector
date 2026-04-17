"""Offline audio file analyzer.

Detection priority
------------------
1. STT keyword match  → VOICEMAIL  (keywords: voicemail / forwarded)
2. 5 s continuous silence → BLANK
3. Otherwise          → LIVE
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
from detectors.stt import transcribe, TranscriptResult
from detectors.vad import VAD
from pipeline import CallDecision, CallType

_FRAME_S: float = settings.frame_ms / 1000.0
_ANALYSIS_WINDOW_S: float = 4.0
_MIN_AUDIO_S: float = 0.5


def analyze_file(
    path: Union[str, Path],
    blank_timeout_s: float = settings.blank_timeout_s,
) -> CallDecision:
    call_uuid = str(uuid.uuid4())
    path = Path(path)

    # ── 1. STT — keyword check ─────────────────────────────────────────
    transcript: TranscriptResult = transcribe(path)
    if transcript.keyword_hit:
        return CallDecision(
            call_uuid=call_uuid,
            result=CallType.VOICEMAIL,
            confidence=0.95,
            detected_at_ms=0,
            details={
                "reason": "keyword_match",
                "matched_keywords": transcript.matched_keywords,
                "transcript": transcript.text,
                "transcript_segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in transcript.segments
                ],
            },
        )

    # ── 2. Load + resample audio ───────────────────────────────────────
    audio, sample_rate = _load_audio(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != settings.analysis_sample_rate:
        audio = resample(audio, sample_rate, settings.analysis_sample_rate)
        sample_rate = settings.analysis_sample_rate

    total_duration_s = len(audio) / sample_rate

    # ── 3. Frame-by-frame blank detection ─────────────────────────────
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

    frame_samples = int(_FRAME_S * sample_rate)
    vad_window_frames = int(_ANALYSIS_WINDOW_S * 1000 / settings.frame_ms)

    pos = 0
    while pos < len(audio):
        chunk = audio[pos: pos + frame_samples]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))

        elapsed_s = pos / sample_rate
        buffer.push(chunk)
        vad.feed(chunk)
        pos += frame_samples

        if elapsed_s < _MIN_AUDIO_S:
            continue

        window = buffer.snapshot(seconds=_ANALYSIS_WINDOW_S)
        speech_ratio = vad.speech_ratio(last_n_frames=vad_window_frames)

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
                    "transcript": transcript.text,
                    "transcript_segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in transcript.segments
                    ],
                },
            )

    # ── 4. Default → LIVE ─────────────────────────────────────────────
    return CallDecision(
        call_uuid=call_uuid,
        result=CallType.LIVE,
        confidence=0.85,
        detected_at_ms=int(total_duration_s * 1000),
        details={
            "reason": "human_voice_detected",
            "total_duration_s": round(total_duration_s, 2),
            "transcript": transcript.text,
            "transcript_segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in transcript.segments
            ],
        },
    )


def _safe_float(v: float, fallback: float = -120.0) -> float:
    import math
    return round(v, 2) if math.isfinite(v) else fallback


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
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
            f"Underlying error: {exc}"
        ) from exc
