# bc-vm-detector

Real-time **Blank Call & Voicemail Detector** for FreeSWITCH audio streams.

Listens to live RTP/PCM audio over WebSocket (via `mod_audio_stream`) and classifies each call leg as `BLANK`, `VOICEMAIL`, or `LIVE` within seconds.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) copy and edit config
cp .env.example .env

# 3. Run the server
python main.py
# or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8765
```

The server starts at `ws://0.0.0.0:8765/{call_uuid}`.

## FreeSWITCH Dialplan

```xml
<extension name="detect_blank_vm">
  <condition field="destination_number" expression="^(\d+)$">
    <action application="answer"/>
    <action application="audio_stream"
            data="ws://127.0.0.1:8765/${uuid} mono 8000"/>
    <action application="sleep" data="15000"/>
  </condition>
</extension>
```

The service sends a JSON result back over the same WebSocket connection:

```json
{
  "type": "result",
  "call_uuid": "abc-123",
  "result": "VOICEMAIL",
  "confidence": 0.85,
  "detected_at_ms": 3800,
  "details": {
    "vm_score": 75,
    "signals": {"beep": true, "speech_burst": true, "greeting_pattern": true}
  }
}
```

## Detection Logic

### Blank Call
- RMS energy below `-50 dBFS` **AND** VAD speech ratio < 5% for 4+ seconds → `BLANK`

### Voicemail (scoring 0–100, threshold 60)
| Signal | Score |
|--------|-------|
| Beep tone detected (300–1200 Hz) | +25 |
| Speech burst 0.5–6 s followed by silence | +20 |
| Spectral flatness > 0.55 (recording artifacts) | +15 |
| Low ZCR variance (periodic/recorded speech) | +10 |
| Greeting pattern (speech then silence) | +10 |
| 2+ signals in first 8 s (corroboration) | +20 |

### Live Call
- If no BLANK or VOICEMAIL decision within 12 s → `LIVE`

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_HOST` | `0.0.0.0` | Bind address |
| `WS_PORT` | `8765` | Port |
| `INPUT_ENCODING` | `pcm_s16le` | `pcm_s16le` or `ulaw` |
| `INPUT_SAMPLE_RATE` | `8000` | FreeSWITCH stream rate |
| `SILENCE_DBFS` | `-50` | Silence energy threshold |
| `BLANK_TIMEOUT_S` | `4.0` | Silence duration for BLANK |
| `VM_SCORE_THRESHOLD` | `60` | VM score threshold |
| `ANALYSIS_TIMEOUT_S` | `12.0` | Fallback to LIVE timeout |
| `WEBHOOK_URL` | `` | Optional HTTP result webhook |

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
bc-vm-detector/
├── main.py              WebSocket server entry point
├── pipeline.py          Per-call analysis orchestrator
├── publisher.py         WebSocket + HTTP webhook result publisher
├── config.py            Settings (pydantic-settings, .env support)
├── audio/
│   ├── buffer.py        Circular audio buffer
│   ├── preprocessor.py  Decode μ-law/PCM, resample
│   └── features.py      RMS, ZCR, spectral features, tone detection
├── detectors/
│   ├── vad.py           WebRTC VAD with hangover smoothing
│   ├── blank_call.py    Blank call detector
│   └── voicemail.py     Voicemail scorer + state machine
├── freeswitch/
│   └── esl_client.py    Optional ESL client for channel variable setting
└── tests/               pytest test suite with synthetic audio fixtures
```
