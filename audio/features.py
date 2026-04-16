"""Low-level audio feature extraction.

All functions accept a float32 numpy array (mono, any sample rate) and the
sample rate as an integer, and return scalar or array features.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Energy / loudness
# ---------------------------------------------------------------------------

def rms_energy(audio: np.ndarray) -> float:
    """Root-mean-square energy of the signal (linear)."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS energy in dBFS (full-scale). Returns -inf for silence."""
    rms = rms_energy(audio)
    if rms == 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms)


# ---------------------------------------------------------------------------
# Zero-crossing rate
# ---------------------------------------------------------------------------

def zero_crossing_rate(audio: np.ndarray) -> float:
    """Number of zero crossings per second (normalized by length)."""
    if len(audio) < 2:
        return 0.0
    signs = np.sign(audio)
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings / len(audio))


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    """Frequency-weighted centroid of the magnitude spectrum (Hz)."""
    if len(audio) < 2:
        return 0.0
    fft_mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    total = fft_mag.sum()
    if total == 0:
        return 0.0
    return float(np.dot(freqs, fft_mag) / total)


def spectral_flatness(audio: np.ndarray) -> float:
    """Wiener entropy — ratio of geometric to arithmetic mean of spectrum.

    1.0 = white noise, 0.0 = pure tone. Values near 1.0 suggest recordings
    with background hiss (pre-recorded greetings / voicemail systems).
    """
    if len(audio) < 2:
        return 0.0
    mag = np.abs(np.fft.rfft(audio))
    mag = mag[mag > 0]
    if len(mag) == 0:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(mag + 1e-10)))
    arith_mean = np.mean(mag)
    if arith_mean == 0:
        return 0.0
    return float(np.clip(geo_mean / arith_mean, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Tone / beep detection
# ---------------------------------------------------------------------------

def detect_tone(
    audio: np.ndarray,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    energy_ratio_threshold: float = 0.60,
) -> tuple[bool, float, float]:
    """Detect a dominant tone in [freq_low, freq_high] Hz.

    Returns:
        (is_tone, dominant_frequency_hz, band_energy_ratio)
    """
    if len(audio) < 2:
        return False, 0.0, 0.0

    mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)

    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    band_energy = float(mag[band_mask].sum())
    total_energy = float(mag.sum())

    if total_energy == 0:
        return False, 0.0, 0.0

    ratio = band_energy / total_energy

    # Find the dominant peak within the band
    band_mag = mag.copy()
    band_mag[~band_mask] = 0
    peaks, _ = find_peaks(band_mag, height=np.max(band_mag) * 0.5)

    dominant_freq = float(freqs[peaks[0]]) if len(peaks) > 0 else 0.0
    is_tone = ratio >= energy_ratio_threshold

    return is_tone, dominant_freq, ratio


# ---------------------------------------------------------------------------
# Framing utilities
# ---------------------------------------------------------------------------

def frame_audio(
    audio: np.ndarray, frame_size: int, hop_size: int
) -> np.ndarray:
    """Split audio into overlapping frames of shape (n_frames, frame_size)."""
    if len(audio) < frame_size:
        return np.empty((0, frame_size), dtype=audio.dtype)
    n_frames = 1 + (len(audio) - frame_size) // hop_size
    indices = (
        np.arange(frame_size)[None, :]
        + hop_size * np.arange(n_frames)[:, None]
    )
    return audio[indices]
