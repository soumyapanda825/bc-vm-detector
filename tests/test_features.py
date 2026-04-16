"""Tests for low-level audio feature extraction."""

import numpy as np
import pytest

from audio.features import (
    rms_energy,
    rms_dbfs,
    zero_crossing_rate,
    spectral_centroid,
    spectral_flatness,
    detect_tone,
)
from tests.conftest import make_silence, make_tone, make_noise, SR


class TestRMS:
    def test_silence_is_zero(self):
        assert rms_energy(make_silence(0.1)) == pytest.approx(0.0)

    def test_silence_dbfs_is_neg_inf(self):
        assert rms_dbfs(make_silence(0.1)) == float("-inf")

    def test_full_scale_sine_near_neg3(self):
        tone = make_tone(1000, 0.1, amplitude=1.0)
        db = rms_dbfs(tone)
        # Full-scale sine RMS = 1/sqrt(2) ≈ -3.01 dBFS
        assert -4.0 < db < -2.0

    def test_noise_energy_proportional_to_amplitude(self):
        loud = rms_energy(make_noise(0.1, amplitude=0.5))
        quiet = rms_energy(make_noise(0.1, amplitude=0.05))
        assert loud > quiet * 5


class TestZeroCrossingRate:
    def test_silence_is_zero(self):
        assert zero_crossing_rate(make_silence(0.1)) == pytest.approx(0.0)

    def test_high_freq_has_high_zcr(self):
        high = zero_crossing_rate(make_tone(4000, 0.1, sr=SR))
        low = zero_crossing_rate(make_tone(200, 0.1, sr=SR))
        assert high > low


class TestSpectralFeatures:
    def test_tone_centroid_near_frequency(self):
        freq = 1000.0
        tone = make_tone(freq, 0.1)
        centroid = spectral_centroid(tone, SR)
        assert abs(centroid - freq) < 100  # within 100 Hz

    def test_noise_flatness_higher_than_tone(self):
        noise_flat = spectral_flatness(make_noise(0.5, amplitude=0.3))
        tone_flat = spectral_flatness(make_tone(1000, 0.5, amplitude=0.3))
        assert noise_flat > tone_flat


class TestToneDetection:
    def test_detects_voicemail_beep(self):
        beep = make_tone(620, 0.3, amplitude=0.8)
        is_tone, freq, ratio = detect_tone(beep, SR, 300, 1200, 0.6)
        assert is_tone is True
        assert 500 < freq < 750

    def test_silence_not_a_tone(self):
        is_tone, freq, ratio = detect_tone(make_silence(0.3), SR, 300, 1200, 0.6)
        assert is_tone is False

    def test_out_of_band_tone_not_detected(self):
        # 3000 Hz is outside [300, 1200]
        high_tone = make_tone(3000, 0.3, amplitude=0.8)
        is_tone, freq, ratio = detect_tone(high_tone, SR, 300, 1200, 0.6)
        assert is_tone is False
