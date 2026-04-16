# Blank Call & Voicemail Detector — Implementation Plan

## Overview

A real-time audio analysis service that connects to FreeSWITCH audio streams via WebSocket and classifies calls as:
- **Blank Call** — no audio, sustained silence, or sub-threshold energy
- **Voicemail** — automated greeting, beep tone, or pre-recorded message pattern
- **Live Call** — real human speech detected

---

## Architecture

```
FreeSWITCH
    │
    │  mod_audio_stream (WebSocket, RTP/PCM audio)
    ▼
WebSocket Server (FastAPI + websockets)
    │
    ▼
Audio Pipeline
    ├── Chunk Buffer     → accumulate frames into analysis windows
    ├── Preprocessor     → decode μ-law/PCM, normalize, resample to 16kHz
    ├── VAD              → webrtcvad frame-level speech activity
    ├── Energy Analyzer  → RMS, zero-crossing rate, spectral features
    ├── Tone Detector    → detect beep tones (DTMF, voicemail beeps 440–1000 Hz)
    └── Classifier       → state machine → BLANK | VOICEMAIL | LIVE
    │
    ▼
Result Publisher
    ├── WebSocket response back to FreeSWITCH / ESL
    └── HTTP webhook (optional)
```

---

## Detection Logic

### Blank Call Detection
1. Compute RMS energy per 20ms frame.
2. If RMS < `SILENCE_THRESHOLD` for > `BLANK_TIMEOUT` seconds (default 4s) → **BLANK**.
3. Zero-crossing rate check to distinguish true silence from line noise.

### Voicemail Detection — Heuristics Chain
| Signal | Meaning |
|--------|---------|
| Speech burst (0.5–3 s) → silence → beep tone | Greeting + invitation to record |
| Sustained speech with no pauses > 2 s (first 8 s) | Pre-recorded greeting |
| DTMF or 440 Hz / 480 Hz / 620 Hz beep | Voicemail system prompt |
| Short speech onset after long silence | Recorded name playback |

Multiple signals combine in a scoring system (0–100). Score ≥ 60 → **VOICEMAIL**.

### State Machine
```
LISTENING → (silence > 4 s) → BLANK
LISTENING → (speech detected) → ANALYZING
ANALYZING → (vm score ≥ 60) → VOICEMAIL
ANALYZING → (live speech patterns) → LIVE
```

---

## FreeSWITCH Integration

### Option A — mod_audio_stream (recommended)
`mod_audio_stream` streams raw PCM via WebSocket. In dialplan:
```xml
<action application="audio_stream" data="ws://127.0.0.1:8765/{call_uuid}"/>
```
Audio arrives as binary WebSocket frames: 16-bit LE PCM, 8000 Hz (or 16000 Hz), mono.

### Option B — ESL + record_session
Use Event Socket Library to capture audio via `record_session` to a named pipe or temp file, then analyze the file in near-real-time.

### Option C — mod_shout / SoX pipe
Stream to a SoX pipeline for format conversion then into the service.

This implementation supports **Option A** primarily, with **Option B** as fallback.

---

## Project Structure

```
bc-vm-detector/
├── PLAN.md                     ← this file
├── README.md
├── requirements.txt
├── config.py                   ← settings (thresholds, ports, timeouts)
├── main.py                     ← FastAPI + WebSocket entry point
├── audio/
│   ├── __init__.py
│   ├── buffer.py               ← circular audio buffer
│   ├── preprocessor.py         ← decode, normalize, resample
│   └── features.py             ← RMS, ZCR, spectral centroid, tone detection
├── detectors/
│   ├── __init__.py
│   ├── vad.py                  ← WebRTC VAD wrapper
│   ├── blank_call.py           ← blank call detector
│   └── voicemail.py            ← voicemail detector + state machine
├── pipeline.py                 ← orchestrates audio → decision
├── publisher.py                ← sends result via WS response or HTTP webhook
├── freeswitch/
│   ├── __init__.py
│   └── esl_client.py           ← optional ESL fallback integration
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_blank_call.py
    ├── test_voicemail.py
    └── fixtures/               ← sample WAV files for testing
```

---

## Implementation Steps

### Step 1 — Environment & Dependencies
- Python 3.10+
- `fastapi`, `uvicorn[standard]` — async WebSocket server
- `webrtcvad` — Google WebRTC VAD (frame-level)
- `numpy`, `scipy` — signal processing
- `librosa` — resampling, spectral features
- `soundfile` — WAV I/O for tests
- `httpx` — async HTTP webhook publishing
- `loguru` — structured logging
- `pydantic-settings` — config management

### Step 2 — Audio Buffer & Preprocessor
- Accept raw binary frames from FreeSWITCH
- Detect encoding (μ-law vs PCM) from config
- Resample to 16kHz (required by webrtcvad)
- Maintain a sliding window buffer of the last N seconds

### Step 3 — Feature Extraction
- **RMS energy** per frame
- **Zero-crossing rate** — differentiates noise from silence
- **Spectral centroid** — speech vs tone vs silence
- **Tone detection** — FFT peak detection in 300–1200 Hz band for beep identification

### Step 4 — VAD Integration
- webrtcvad operates on 10/20/30 ms frames at 8/16/32/48 kHz
- Frame classifier returns `speech=True/False`
- Smoothed with a 300ms hangover window

### Step 5 — Blank Call Detector
- Sliding window vote: if `speech_ratio < 0.05` over last 4 s → BLANK
- Energy gate: `rms < SILENCE_DB (-50 dBFS)` confirms

### Step 6 — Voicemail Detector
- State machine tracks: `IDLE → GREETING → BEEP_WAIT → CONFIRMED`
- Scoring: +20 beep detected, +20 speech burst 1–4s, +10 silence >1s after speech, +15 spectral flatness (recording artifacts), +10 low ZCR variance
- Score ≥ 60 triggers VOICEMAIL result

### Step 7 — Pipeline & Decision Engine
- Runs detectors in parallel per audio window
- First definitive result (BLANK or VOICEMAIL) is published; pipeline stops
- Timeout: if no decision after 12 s → LIVE (assume human)

### Step 8 — WebSocket Server
- `ws://host:8765/{call_uuid}` — one connection per call leg
- Binary frames in → JSON result out: `{"call_uuid": "...", "result": "BLANK|VOICEMAIL|LIVE", "confidence": 0.92, "detected_at_ms": 3200}`
- HTTP POST webhook to configurable URL on each decision

### Step 9 — Tests
- Unit tests for each detector using synthetic audio (sine waves, silence, speech fixtures)
- Integration test: WebSocket client sends audio file, checks result

### Step 10 — FreeSWITCH Dialplan Example
```xml
<extension name="vm_blank_detect">
  <condition field="destination_number" expression="^(\d+)$">
    <action application="answer"/>
    <action application="audio_stream" data="ws://127.0.0.1:8765/${uuid}"/>
    <action application="sleep" data="15000"/>
    <!-- Handle result via ESL or channel variable set by service -->
  </condition>
</extension>
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WS_HOST` | `0.0.0.0` | WebSocket bind address |
| `WS_PORT` | `8765` | WebSocket port |
| `SAMPLE_RATE` | `16000` | Analysis sample rate (Hz) |
| `FRAME_MS` | `20` | VAD frame size (ms) |
| `SILENCE_DB` | `-50` | RMS threshold for silence (dBFS) |
| `BLANK_TIMEOUT_S` | `4.0` | Silence duration to declare BLANK |
| `VM_SCORE_THRESHOLD` | `60` | Min score to declare VOICEMAIL |
| `ANALYSIS_TIMEOUT_S` | `12.0` | Max analysis time before defaulting to LIVE |
| `WEBHOOK_URL` | `""` | Optional HTTP webhook for results |
| `INPUT_ENCODING` | `pcm_s16le` | `pcm_s16le` or `ulaw` |
| `INPUT_SAMPLE_RATE` | `8000` | FreeSWITCH stream rate |

---

## Sequence Diagram

```
FS        WS Server      Pipeline       Detector(s)      Publisher
 │            │              │               │                │
 │─(connect)─▶│              │               │                │
 │─(audio)───▶│─(enqueue)───▶│               │                │
 │─(audio)───▶│             │─(extract)─────▶│                │
 │─(audio)───▶│             │◀──(score)──────│                │
 │            │             │─(decision)─────────────────────▶│
 │            │◀────────────────────────────(JSON result)─────│
 │◀(result)───│              │               │                │
```
