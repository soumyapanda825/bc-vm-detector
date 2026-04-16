"""Tests for voicemail detection."""

import numpy as np
import pytest

from detectors.voicemail import VoicemailDetector, VMState
from tests.conftest import (
    make_silence,
    make_tone,
    make_speech_like,
    make_noise,
)


SR = 16000


class TestVoicemailDetector:
    def test_beep_tone_increases_score(self):
        detector = VoicemailDetector(vm_score_threshold=100)  # high threshold so we can inspect score
        beep = make_tone(620, 0.5, amplitude=0.7)
        result = detector.update(beep, speech_ratio=0.0, elapsed_s=0.5)
        assert "beep" in result.signals

    def test_greeting_burst_increases_score(self):
        detector = VoicemailDetector(vm_score_threshold=100)
        speech = make_speech_like(2.0)
        result = detector.update(speech, speech_ratio=0.9, elapsed_s=2.0)
        # After silence following the burst
        silence = make_silence(1.5)
        result = detector.update(silence, speech_ratio=0.0, elapsed_s=3.5)
        assert "speech_burst" in result.signals or "greeting_pattern" in result.signals

    def test_full_voicemail_pattern_detected(self):
        """Greeting → silence → beep should reach score threshold."""
        detector = VoicemailDetector(vm_score_threshold=60)

        # Greeting
        speech = make_speech_like(2.0)
        detector.update(speech, speech_ratio=0.85, elapsed_s=2.0)

        # Silence after greeting
        silence = make_silence(1.5)
        detector.update(silence, speech_ratio=0.0, elapsed_s=3.5)

        # Beep
        beep = make_tone(620, 0.4, amplitude=0.7)
        result = detector.update(beep, speech_ratio=0.0, elapsed_s=4.0)

        assert result.is_voicemail is True
        assert result.score >= 60

    def test_continuous_speech_not_voicemail(self):
        """Sustained speech without beep should NOT trigger voicemail quickly."""
        detector = VoicemailDetector(vm_score_threshold=60)
        for i in range(10):
            audio = make_speech_like(1.0)
            result = detector.update(audio, speech_ratio=0.9, elapsed_s=float(i + 1))
            # Should not be flagged after just continuous speech
        assert result.is_voicemail is False

    def test_reset_clears_state(self):
        detector = VoicemailDetector(vm_score_threshold=60)
        speech = make_speech_like(2.0)
        detector.update(speech, speech_ratio=0.85, elapsed_s=2.0)
        silence = make_silence(2.0)
        detector.update(silence, speech_ratio=0.0, elapsed_s=4.0)
        beep = make_tone(620, 0.5, amplitude=0.7)
        result = detector.update(beep, speech_ratio=0.0, elapsed_s=4.5)
        first_score = result.score

        detector.reset()
        result2 = detector.update(make_silence(0.1), speech_ratio=0.0, elapsed_s=0.1)
        assert result2.score == 0
        assert result2.state == VMState.IDLE

    def test_silence_only_does_not_trigger_voicemail(self):
        detector = VoicemailDetector(vm_score_threshold=60)
        silence = make_silence(8.0)
        result = detector.update(silence, speech_ratio=0.0, elapsed_s=8.0)
        assert result.is_voicemail is False
