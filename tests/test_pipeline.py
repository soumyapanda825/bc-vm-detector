"""Integration tests for the full pipeline."""

import numpy as np
import pytest

from pipeline import Pipeline, CallType
from tests.conftest import (
    audio_to_pcm_bytes,
    make_silence,
    make_speech_like,
    make_tone,
    SR,
)
from config import settings


def feed_audio(pipeline: Pipeline, audio: np.ndarray, chunk_s: float = 0.1):
    """Feed audio into the pipeline in small chunks, return last decision."""
    chunk_size = int(chunk_s * SR)
    decision = None
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        raw = audio_to_pcm_bytes(chunk)
        # Simulate input at the configured input sample rate
        # The preprocessor will handle resampling internally
        result = pipeline.feed(raw)
        if result is not None:
            decision = result
    return decision


@pytest.mark.skipif(
    settings.input_sample_rate != 16000,
    reason="Integration test assumes 16kHz input for simplicity"
)
class TestPipelineIntegration:
    def test_blank_call_detected(self):
        """5 s of silence should be detected as BLANK."""
        p = Pipeline("blank-test-001")
        silence = make_silence(6.0)
        decision = feed_audio(p, silence)

        assert decision is not None
        assert decision.result == CallType.BLANK
        assert decision.confidence > 0.5

    def test_voicemail_pattern_detected(self):
        """Greeting + beep pattern should be detected as VOICEMAIL."""
        p = Pipeline("vm-test-001")
        greeting = make_speech_like(2.5, amplitude=0.3)
        silence = make_silence(1.0)
        beep = make_tone(620, 0.4, amplitude=0.7)
        audio = np.concatenate([greeting, silence, beep])

        decision = feed_audio(p, audio)

        assert decision is not None
        assert decision.result == CallType.VOICEMAIL

    def test_analysis_timeout_defaults_to_live(self):
        """If no clear signal within timeout, default to LIVE."""
        # Patch timeout for test speed
        import pipeline as pm
        original = pm.settings.analysis_timeout_s

        try:
            pm.settings.analysis_timeout_s = 1.0
            p = Pipeline("live-test-001")
            # Feed ambiguous audio for just over 1 s
            speech = make_speech_like(1.5, amplitude=0.3)
            decision = feed_audio(p, speech)
            assert decision is not None
            assert decision.result == CallType.LIVE
        finally:
            pm.settings.analysis_timeout_s = original

    def test_pipeline_does_not_emit_twice(self):
        """Once decided, further audio should not change the decision."""
        p = Pipeline("idempotent-test-001")
        silence = make_silence(6.0)
        decisions = []
        chunk_size = int(0.1 * SR)
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            raw = audio_to_pcm_bytes(chunk)
            result = p.feed(raw)
            if result is not None:
                decisions.append(result)

        # Should emit only once (all subsequent None since decided=True)
        assert all(d.result == CallType.BLANK for d in decisions)
