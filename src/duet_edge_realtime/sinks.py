from __future__ import annotations

import asyncio
import json
from collections import deque
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


class ViewerMailbox:
    """A control-preserving mailbox with a bounded latest-frame lane."""

    def __init__(self, frame_capacity: int):
        self.frame_capacity = frame_capacity
        self._items: deque[dict] = deque()
        self._frame_count = 0
        self._available = asyncio.Event()

    def full(self) -> bool:
        return self._frame_count >= self.frame_capacity

    def put_nowait(self, message: dict) -> bool:
        dropped = False
        message_type = message.get("type")
        if message_type == "frame":
            if self.full():
                for index, existing in enumerate(self._items):
                    if existing.get("type") == "frame":
                        del self._items[index]
                        self._frame_count -= 1
                        dropped = True
                        break
            self._frame_count += 1
        elif message_type in {
            "state", "metrics", "degraded", "backpressure", "overload"
        }:
            for index, existing in enumerate(self._items):
                if existing.get("type") == message_type:
                    del self._items[index]
                    break
        self._items.append(message)
        self._available.set()
        return dropped

    def get_nowait(self) -> dict:
        if not self._items:
            raise asyncio.QueueEmpty
        message = self._items.popleft()
        if message.get("type") == "frame":
            self._frame_count -= 1
        if not self._items:
            self._available.clear()
        return message

    async def get(self) -> dict:
        while not self._items:
            self._available.clear()
            await self._available.wait()
        return self.get_nowait()


class WebSocketSink(Sink):
    def __init__(self, host: str, port: int, queue_frames: int, on_drop=None):
        self.host, self.port = host, port
        self.queue_frames = queue_frames
        self.on_drop = on_drop
        self.server = None
        self.hello: dict | None = None
        self.latest_status: dict[str, dict] = {}
        self.clients: dict[Any, tuple[ViewerMailbox, asyncio.Task]] = {}

    async def start(self, hello: dict) -> None:
        self.hello = hello
        self.server = await websockets.serve(self._handler, self.host, self.port)

    async def _handler(self, websocket) -> None:
        queue = ViewerMailbox(self.queue_frames)
        sender = asyncio.create_task(self._sender(websocket, queue))
        self.clients[websocket] = (queue, sender)
        try:
            await websocket.send(json.dumps(self.hello, separators=(",", ":")))
            for status in self.latest_status.values():
                await websocket.send(json.dumps(status, separators=(",", ":")))
            await websocket.wait_closed()
        finally:
            self.clients.pop(websocket, None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    async def _sender(self, websocket, queue: ViewerMailbox) -> None:
        while True:
            message = await queue.get()
            await websocket.send(json.dumps(message, separators=(",", ":"), allow_nan=False))

    async def send(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type in {
            "state", "metrics", "degraded", "backpressure", "overload", "eos", "error"
        }:
            self.latest_status[message_type] = message
        for queue, _ in list(self.clients.values()):
            if isinstance(queue, ViewerMailbox):
                dropped = queue.put_nowait(message)
                if dropped and self.on_drop:
                    self.on_drop()
                continue
            if queue.full() and message_type == "frame":
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
