"""Entry point — FastAPI + WebSocket server.

Each inbound WebSocket connection represents one call leg. The call UUID
is parsed from the URL path: ws://host:8765/{call_uuid}

Protocol
--------
Client → Server: binary frames of raw PCM/μ-law audio (no framing headers)
Server → Client: JSON text messages

JSON message schema:
{
  "type": "result" | "error" | "ack",
  "call_uuid": "<uuid>",
  "result": "BLANK" | "VOICEMAIL" | "LIVE" | "ANALYZING",
  "confidence": 0.0-1.0,
  "detected_at_ms": 3200,
  "details": { ... }
}
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import settings
from file_analyzer import analyze_file
from pipeline import CallDecision, CallType, Pipeline
from publisher import decision_to_dict, send_webhook

# Active pipelines keyed by call_uuid
_active: dict[str, Pipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "BC/VM Detector starting — ws://{}:{}/{{call_uuid}}",
        settings.ws_host,
        settings.ws_port,
    )
    yield
    logger.info("Shutting down — {} active calls", len(_active))
    _active.clear()


app = FastAPI(title="Blank Call & Voicemail Detector", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "active_calls": len(_active)}


@app.post("/analyze")
async def analyze_audio_file(file: UploadFile):
    """Analyze an uploaded audio file and return BLANK / VOICEMAIL / LIVE."""
    _ALLOWED = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".caf"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in _ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(_ALLOWED))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        decision = analyze_file(tmp_path)
        return decision_to_dict(decision)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("File analysis error: {}", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        os.unlink(tmp_path)


@app.get("/calls")
async def list_calls():
    return {
        "calls": [
            {
                "call_uuid": uuid,
                "decided": p.decided,
                "result": p.decision.result.value if p.decided else "ANALYZING",
            }
            for uuid, p in _active.items()
        ]
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/{call_uuid}")
async def ws_call_handler(websocket: WebSocket, call_uuid: str):
    await websocket.accept()
    logger.info("Call connected: {}", call_uuid)

    pipeline = Pipeline(call_uuid=call_uuid)
    _active[call_uuid] = pipeline

    try:
        async for message in websocket.iter_bytes():
            if not message:
                continue

            decision = pipeline.feed(message)

            if decision is not None and not getattr(pipeline, "_result_sent", False):
                pipeline._result_sent = True  # type: ignore[attr-defined]
                payload = _build_payload("result", decision)
                await websocket.send_text(json.dumps(payload))
                asyncio.create_task(send_webhook(decision))

    except WebSocketDisconnect:
        logger.info("Call disconnected: {}", call_uuid)
    except Exception as exc:
        logger.exception("Error handling call {}: {}", call_uuid, exc)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "call_uuid": call_uuid, "message": str(exc)})
            )
        except Exception:
            pass
    finally:
        _active.pop(call_uuid, None)
        logger.info("Call cleaned up: {}", call_uuid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_payload(msg_type: str, decision: CallDecision) -> dict:
    d = decision_to_dict(decision)
    d["type"] = msg_type
    return d


# ---------------------------------------------------------------------------
# Frontend — served last so it never shadows API or WebSocket routes
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.ws_host,
        port=settings.ws_port,
        log_level=settings.log_level.lower(),
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
