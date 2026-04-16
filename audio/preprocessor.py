"""Decode and normalize raw audio bytes from FreeSWITCH into float32 PCM.

Supports:
  - pcm_s16le  — signed 16-bit little-endian linear PCM (most common)
  - ulaw       — G.711 μ-law (common on PSTN trunks)

Output is always float32 in [-1.0, 1.0], mono.
Resampling is performed when the input rate differs from the analysis rate.
"""

import audioop
import struct

import numpy as np

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    _HAS_LIBROSA = False


# ---------------------------------------------------------------------------
# μ-law decode table (ITU-T G.711)
# ---------------------------------------------------------------------------
def _build_ulaw_table() -> np.ndarray:
    rows = []
    for b in range(256):
        raw = audioop.ulaw2lin(bytes([b]), 2)
        rows.append(np.frombuffer(raw, dtype=np.int16)[0])
    return np.array(rows, dtype=np.int16)

_ULAW_TABLE: np.ndarray = _build_ulaw_table()


def decode_ulaw(data: bytes) -> np.ndarray:
    """Decode μ-law bytes to int16 numpy array."""
    raw = audioop.ulaw2lin(data, 2)
    return np.frombuffer(raw, dtype=np.int16)


def decode_pcm_s16le(data: bytes) -> np.ndarray:
    """Decode signed 16-bit LE PCM bytes to int16 numpy array."""
    n = len(data) // 2
    return np.frombuffer(data[:n * 2], dtype="<i2")


def to_float32(samples: np.ndarray) -> np.ndarray:
    """Normalize int16 samples to float32 in [-1.0, 1.0]."""
    return samples.astype(np.float32) / 32768.0


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample float32 audio from orig_sr to target_sr.

    Uses librosa if available, otherwise falls back to scipy.
    """
    if orig_sr == target_sr:
        return audio

    if _HAS_LIBROSA:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    # scipy fallback
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


class Preprocessor:
    """Stateless audio preprocessor: bytes → float32 PCM at analysis_sample_rate."""

    def __init__(self, encoding: str, input_sr: int, output_sr: int):
        """
        Args:
            encoding: 'pcm_s16le' or 'ulaw'
            input_sr: sample rate of the incoming stream (e.g. 8000)
            output_sr: desired output sample rate (e.g. 16000)
        """
        if encoding not in ("pcm_s16le", "ulaw"):
            raise ValueError(f"Unsupported encoding: {encoding!r}")
        self._encoding = encoding
        self._input_sr = input_sr
        self._output_sr = output_sr

    def process(self, raw: bytes) -> np.ndarray:
        """Decode raw bytes and return float32 samples at output_sr."""
        if self._encoding == "ulaw":
            samples = decode_ulaw(raw)
        else:
            samples = decode_pcm_s16le(raw)

        audio = to_float32(samples)
        return resample(audio, self._input_sr, self._output_sr)
