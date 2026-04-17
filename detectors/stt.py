"""Speech-to-text using OpenAI Whisper (runs locally, no API key needed).

Audio is loaded with soundfile/librosa and passed as a numpy array so
ffmpeg is NOT required.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

_model = None
_lock  = threading.Lock()

WHISPER_SR = 16_000  # Whisper always expects 16 kHz float32 mono

VOICEMAIL_KEYWORDS = {"voicemail", "forwarded", "voice mail", "voice-mail"}


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import whisper  # type: ignore
                _model = whisper.load_model("base")
    return _model


def _load_audio_np(path: Path) -> np.ndarray:
    """Return float32 mono 16 kHz numpy array without touching ffmpeg."""
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa  # type: ignore
        audio, sr = librosa.load(str(path), sr=None, mono=False, dtype=np.float32)

    # Downmix stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != WHISPER_SR:
        import scipy.signal as sps  # type: ignore
        samples_out = int(len(audio) * WHISPER_SR / sr)
        audio = sps.resample(audio, samples_out).astype(np.float32)

    return audio


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "en"
    keyword_hit: bool = False
    matched_keywords: list[str] = field(default_factory=list)


def transcribe(path: Union[str, Path]) -> TranscriptResult:
    """Transcribe an audio file and check for voicemail keywords."""
    import whisper  # type: ignore

    model   = _get_model()
    audio   = _load_audio_np(Path(path))
    result  = model.transcribe(audio, language="en", fp16=False)

    text       = result.get("text", "").strip()
    text_lower = text.lower()

    segments = [
        TranscriptSegment(
            start=round(s["start"], 2),
            end=round(s["end"], 2),
            text=s["text"].strip(),
        )
        for s in result.get("segments", [])
    ]

    matched = [kw for kw in VOICEMAIL_KEYWORDS if kw in text_lower]

    return TranscriptResult(
        text=text,
        segments=segments,
        language=result.get("language", "en"),
        keyword_hit=bool(matched),
        matched_keywords=matched,
    )
