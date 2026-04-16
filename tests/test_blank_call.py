"""Tests for blank call detection."""

import numpy as np
import pytest

from detectors.blank_call import BlankCallDetector
from tests.conftest import make_silence, make_noise, make_speech_like


class TestBlankCallDetector:
    def test_silence_triggers_blank_after_timeout(self):
        detector = BlankCallDetector(blank_timeout_s=4.0)
        silence = make_silence(5.0)

        # Feed in chunks simulating time progression
        chunk_s = 0.5
        sr = 16000
        chunk_size = int(chunk_s * sr)
        elapsed = 0.0
        result = None
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            elapsed += chunk_s
            result = detector.update(chunk, speech_ratio=0.0, elapsed_s=elapsed)

        assert result is not None
        assert result.is_blank is True
        assert result.confidence > 0.0

    def test_short_silence_does_not_trigger(self):
        detector = BlankCallDetector(blank_timeout_s=4.0)
        silence = make_silence(2.0)

        result = detector.update(silence, speech_ratio=0.0, elapsed_s=2.0)
        assert result.is_blank is False

    def test_speech_resets_silence_counter(self):
        detector = BlankCallDetector(blank_timeout_s=4.0)
        sr = 16000

        # 3 s silence
        detector.update(make_silence(3.0), speech_ratio=0.0, elapsed_s=3.0)
        # 1 s speech — should reset
        detector.update(make_speech_like(1.0), speech_ratio=0.8, elapsed_s=4.0)
        # 3 s silence again — still under 4 s continuous
        result = detector.update(make_silence(3.0), speech_ratio=0.0, elapsed_s=7.0)

        assert result.is_blank is False

    def test_noise_alone_does_not_trigger(self):
        """Low-amplitude line noise should not trigger blank if energy is above threshold."""
        detector = BlankCallDetector(blank_timeout_s=4.0, silence_dbfs=-50.0)
        noise = make_noise(5.0, amplitude=0.2)  # -14 dBFS — above threshold

        result = detector.update(noise, speech_ratio=0.0, elapsed_s=5.0)
        assert result.is_blank is False

    def test_reset_clears_state(self):
        detector = BlankCallDetector(blank_timeout_s=2.0)
        sr = 16000
        chunk_s = 0.5
        chunk_size = int(chunk_s * sr)
        silence = make_silence(3.0)
        result = None
        for step, i in enumerate(range(0, len(silence), chunk_size)):
            result = detector.update(
                silence[i : i + chunk_size],
                speech_ratio=0.0,
                elapsed_s=(step + 1) * chunk_s,
            )
        assert result is not None
        assert result.is_blank is True

        detector.reset()
        result2 = detector.update(silence[:chunk_size], speech_ratio=0.0, elapsed_s=0.5)
        assert result2.is_blank is False
