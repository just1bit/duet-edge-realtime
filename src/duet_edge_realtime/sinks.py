from __future__ import annotations

import asyncio
import json
import mimetypes
import time
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
            "state", "input_status", "metrics", "degraded", "backpressure", "overload"
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
    def __init__(
        self,
        host: str,
        port: int,
        queue_frames: int,
        on_drop=None,
        on_connect=None,
        on_disconnect=None,
        on_telemetry=None,
    ):
        self.host, self.port = host, port
        self.queue_frames = queue_frames
        self.on_drop = on_drop
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_telemetry = on_telemetry
        self.server = None
        self.hello: dict | None = None
        self.latest_status: dict[str, dict] = {}
        self.clients: dict[Any, tuple[ViewerMailbox, asyncio.Task]] = {}
        self.client_ids: dict[Any, str] = {}
        self._next_client_id = 1

    async def start(self, hello: dict) -> None:
        self.hello = hello
        self.server = await websockets.serve(self._handler, self.host, self.port)

    async def update_hello(self, hello: dict) -> None:
        """Replace the session hello and publish it to already connected viewers."""
        self.hello = hello
        self.latest_status.clear()
        payload = json.dumps(hello, separators=(",", ":"), allow_nan=False)
        for client in list(self.clients):
            try:
                await client.send(payload)
            except Exception:
                continue

    async def _handler(self, websocket) -> None:
        connected_at = time.monotonic()
        queue = ViewerMailbox(self.queue_frames)
        sender = asyncio.create_task(self._sender(websocket, queue))
        self.clients[websocket] = (queue, sender)
        self.client_ids[websocket] = f"viewer-{self._next_client_id}"
        self._next_client_id += 1
        if self.on_connect:
            self.on_connect()
        try:
            await websocket.send(json.dumps(self.hello, separators=(",", ":")))
            for status in self.latest_status.values():
                await websocket.send(json.dumps(status, separators=(",", ":")))
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if message.get("type") == "client_metrics" and self.on_telemetry:
                    self.on_telemetry(message)
        finally:
            self.clients.pop(websocket, None)
            self.client_ids.pop(websocket, None)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            if self.on_disconnect:
                self.on_disconnect(time.monotonic() - connected_at)

    async def _sender(self, websocket, queue: ViewerMailbox) -> None:
        while True:
            message = await queue.get()
            await websocket.send(json.dumps(message, separators=(",", ":"), allow_nan=False))

    async def send(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type in {
            "state", "input_status", "metrics", "eos", "error"
        }:
            self.latest_status[message_type] = message
        for client, (queue, _) in list(self.clients.items()):
            if isinstance(queue, ViewerMailbox):
                dropped = queue.put_nowait(message)
                if dropped and self.on_drop:
                    self.on_drop(self.client_ids.get(client, "viewer-unknown"))
                continue
            if queue.full() and message_type == "frame":
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                if self.on_drop:
                    self.on_drop(self.client_ids.get(client, "viewer-unknown"))
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


class StaticWebSink(Sink):
    """Small integrated HTTP server for the Viewer and `/health`."""

    def __init__(self, host: str, port: int, web_root: str | Path):
        self.host = host
        self.port = port
        self.web_root = Path(web_root).resolve()
        self.server = None
        self.hello = None

    async def start(self, hello: dict) -> None:
        if not self.web_root.is_dir():
            raise FileNotFoundError(self.web_root)
        self.hello = hello
        self.server = await asyncio.start_server(
            self._handle_request, self.host, self.port
        )

    async def send(self, message: dict) -> None:
        return None

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_request(self, reader, writer) -> None:
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2.0)
            first = request.split(b"\r\n", 1)[0].decode("ascii", "replace")
            method, target, _ = first.split(" ", 2)
            if method != "GET":
                await self._respond(writer, 405, b"method not allowed\n", "text/plain")
                return
            path = target.split("?", 1)[0]
            if path == "/health":
                body = json.dumps({
                    "ok": True,
                    "protocol": self.hello.get("protocol"),
                    "run_id": self.hello.get("run_id"),
                }).encode()
                await self._respond(writer, 200, body, "application/json")
                return
            relative = "index.html" if path == "/" else path.lstrip("/")
            asset = (self.web_root / relative).resolve()
            if not asset.is_relative_to(self.web_root) or not asset.is_file():
                await self._respond(writer, 404, b"not found\n", "text/plain")
                return
            content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            await self._respond(writer, 200, asset.read_bytes(), content_type)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            await self._respond(writer, 400, b"bad request\n", "text/plain")
        except asyncio.TimeoutError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    @staticmethod
    async def _respond(writer, status: int, body: bytes, content_type: str) -> None:
        reasons = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed"}
        header = (
            f"HTTP/1.1 {status} {reasons[status]}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        writer.write(header + body)
        await writer.drain()
