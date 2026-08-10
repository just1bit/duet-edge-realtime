from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import websockets


class Sink:
    async def start(self, hello: dict) -> None: ...
    async def send(self, message: dict) -> None: ...
    async def close(self) -> None: ...


class NDJSONSink(Sink):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.handle = None

    async def start(self, hello: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", buffering=1)
        await self.send(hello)

    async def send(self, message: dict) -> None:
        if self.handle is None:
            raise RuntimeError("sink not started")
        self.handle.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")

    async def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None


class WebSocketSink(Sink):
    def __init__(self, host: str, port: int, queue_frames: int, on_drop=None):
        self.host, self.port = host, port
        self.queue_frames = queue_frames
        self.on_drop = on_drop
        self.server = None
        self.hello: dict | None = None
        self.latest_status: dict | None = None
        self.clients: dict[Any, tuple[asyncio.Queue, asyncio.Task]] = {}

    async def start(self, hello: dict) -> None:
        self.hello = hello
        self.server = await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_frames)
        sender = asyncio.create_task(self._sender(websocket, queue))
        self.clients[websocket] = (queue, sender)
        try:
            await websocket.send(json.dumps(self.hello, separators=(",", ":")))
            if self.latest_status is not None:
                await websocket.send(json.dumps(self.latest_status, separators=(",", ":")))
            await websocket.wait_closed()
        finally:
            self.clients.pop(websocket, None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    async def _sender(self, websocket, queue: asyncio.Queue) -> None:
        while True:
            message = await queue.get()
            await websocket.send(json.dumps(message, separators=(",", ":"), allow_nan=False))

    async def send(self, message: dict) -> None:
        if message.get("type") in {"metrics", "degraded", "eos", "error"}:
            self.latest_status = message
        for queue, _ in list(self.clients.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                if self.on_drop:
                    self.on_drop()
            queue.put_nowait(message)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None


class CompositeSink(Sink):
    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks

    async def start(self, hello: dict) -> None:
        for sink in self.sinks:
            await sink.start(hello)

    async def send(self, message: dict) -> None:
        for sink in self.sinks:
            await sink.send(message)

    async def close(self) -> None:
        for sink in reversed(self.sinks):
            await sink.close()
