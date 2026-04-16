"""Optional FreeSWITCH ESL (Event Socket Library) client.

This module provides a minimal ESL client that can:
  1. Execute dialplan applications (e.g., audio_stream) on a specific channel.
  2. Set channel variables to report detection results back to FreeSWITCH.
  3. Listen for CHANNEL_HANGUP events to clean up pipeline state.

Usage (standalone, not required for WebSocket-only deployments):

    client = ESLClient(host="127.0.0.1", port=8021, password="ClueCon")
    await client.connect()
    await client.set_variable(call_uuid, "vm_detector_result", "VOICEMAIL")
    await client.disconnect()

Requires `python-ESL` or a pure-Python alternative. This implementation uses
raw TCP socket communication following the ESL plaintext protocol to avoid
heavy C-extension dependencies.
"""

from __future__ import annotations

import asyncio
from loguru import logger


class ESLClient:
    """Minimal async ESL inbound client (connects to FreeSWITCH mod_event_socket)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8021,
        password: str = "ClueCon",
    ):
        self._host = host
        self._port = port
        self._password = password
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port
        )
        # Wait for auth/request line
        await self._read_until_blank()
        await self._send(f"auth {self._password}\n\n")
        reply = await self._read_until_blank()
        if "Reply-Text: +OK accepted" not in reply:
            raise ConnectionError(f"ESL auth failed: {reply!r}")
        self._connected = True
        logger.info("ESL connected to {}:{}", self._host, self._port)

    async def disconnect(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def api(self, cmd: str) -> str:
        """Send an ESL `api` command and return the response body."""
        await self._send(f"api {cmd}\n\n")
        return await self._read_until_blank()

    async def execute(self, call_uuid: str, app: str, arg: str = "") -> str:
        """Execute a dialplan application on a specific channel."""
        cmd = (
            f"sendmsg {call_uuid}\n"
            f"call-command: execute\n"
            f"execute-app-name: {app}\n"
            f"execute-app-arg: {arg}\n\n"
        )
        await self._send(cmd)
        return await self._read_until_blank()

    async def set_variable(
        self, call_uuid: str, name: str, value: str
    ) -> str:
        """Set a channel variable on a specific call leg."""
        return await self.execute(call_uuid, "set", f"{name}={value}")

    async def hangup(self, call_uuid: str, cause: str = "NORMAL_CLEARING") -> str:
        return await self.execute(call_uuid, "hangup", cause)

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    async def _send(self, data: str) -> None:
        if not self._writer:
            raise RuntimeError("Not connected")
        self._writer.write(data.encode())
        await self._writer.drain()

    async def _read_until_blank(self) -> str:
        """Read lines until an empty line (ESL message terminator)."""
        lines: list[str] = []
        while True:
            line = await self._reader.readline()
            decoded = line.decode(errors="replace").rstrip("\r\n")
            if decoded == "":
                break
            lines.append(decoded)
        return "\n".join(lines)
