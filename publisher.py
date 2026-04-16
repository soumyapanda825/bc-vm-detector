"""Result publisher — sends CallDecision via WebSocket response and/or HTTP webhook."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from config import settings

if TYPE_CHECKING:
    from pipeline import CallDecision


def decision_to_dict(decision: "CallDecision") -> dict:
    return {
        "call_uuid": decision.call_uuid,
        "result": decision.result.value,
        "confidence": decision.confidence,
        "detected_at_ms": decision.detected_at_ms,
        "details": decision.details,
    }


async def send_webhook(decision: "CallDecision") -> None:
    """POST the decision JSON to the configured webhook URL (fire-and-forget)."""
    if not settings.webhook_url:
        return

    payload = decision_to_dict(decision)
    try:
        async with httpx.AsyncClient(timeout=settings.webhook_timeout_s) as client:
            resp = await client.post(settings.webhook_url, json=payload)
            resp.raise_for_status()
            logger.debug("Webhook sent for {} → {}", decision.call_uuid, resp.status_code)
    except Exception as exc:
        logger.warning("Webhook failed for {}: {}", decision.call_uuid, exc)
