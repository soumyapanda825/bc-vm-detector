"""Shared pytest fixtures — synthetic audio generators."""

import numpy as np
import pytest

SR = 16000  # analysis sample rate


def make_silence(duration_s: float, sr: int = SR) -> np.ndarray:
    """Pure digital silence."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def make_tone(freq_hz: float, duration_s: float, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    """Pure sine tone."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_noise(duration_s: float, amplitude: float = 0.05, sr: int = SR) -> np.ndarray:
    """White noise at low amplitude (line noise simulation)."""
    rng = np.random.default_rng(42)
    return (amplitude * rng.standard_normal(int(duration_s * sr))).astype(np.float32)


def make_speech_like(duration_s: float, amplitude: float = 0.3, sr: int = SR) -> np.ndarray:
    """Rough speech-like signal: band-limited noise in 300-3400 Hz."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(7)
    noise = rng.standard_normal(int(duration_s * sr)).astype(np.float32)
    sos = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band", output="sos")
    filtered = sosfilt(sos, noise).astype(np.float32)
    filtered /= np.abs(filtered).max() + 1e-9
    return (filtered * amplitude).astype(np.float32)


def audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 to int16 LE PCM bytes (as FreeSWITCH would send)."""
    return (audio * 32767).astype(np.int16).tobytes()


# --- Fixtures ---

@pytest.fixture
def silence_4s():
    return make_silence(4.0)


@pytest.fixture
def silence_6s():
    return make_silence(6.0)


@pytest.fixture
def tone_beep():
    """620 Hz beep — common voicemail invitation tone."""
    return make_tone(620, 0.5)


@pytest.fixture
def speech_burst_2s():
    return make_speech_like(2.0)


@pytest.fixture
def voicemail_audio():
    """Simulated voicemail pattern: greeting (2s) + silence (1s) + beep (0.5s)."""
    greeting = make_speech_like(2.0, amplitude=0.3)
    silence = make_silence(1.0)
    beep = make_tone(620, 0.5, amplitude=0.6)
    return np.concatenate([greeting, silence, beep])
