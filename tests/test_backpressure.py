import asyncio
import unittest
from pathlib import Path

from duet_edge_realtime.backends.fake import FakeInferenceBackend
from duet_edge_realtime.config import RealtimeConfig, StreamConfig
from duet_edge_realtime.playout import VirtualClock
from duet_edge_realtime.schemas import MotionWindow
from duet_edge_realtime.service import StreamingService
from duet_edge_realtime.sinks import ViewerMailbox, WebSocketSink

from helpers import identity_motion


class MemorySink:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


class BackpressureTests(unittest.TestCase):
    def test_slow_viewer_drops_only_its_oldest_frame(self):
        dropped = []
        sink = WebSocketSink("127.0.0.1", 8765, 2, lambda client_id: dropped.append(client_id))
        queue = asyncio.Queue(maxsize=2)
        sink.clients[object()] = (queue, None)
        async def scenario():
            await sink.send({"type":"frame","seq":0})
            await sink.send({"type":"frame","seq":1})
            await sink.send({"type":"frame","seq":2})
        asyncio.run(scenario())
        self.assertEqual(dropped, ["viewer-unknown"])
        self.assertEqual([queue.get_nowait()["seq"], queue.get_nowait()["seq"]], [1,2])

    def test_viewer_mailbox_preserves_control_messages(self):
        mailbox = ViewerMailbox(2)
        mailbox.put_nowait({"type":"frame", "seq":0})
        mailbox.put_nowait({"type":"state", "state":"playing"})
        mailbox.put_nowait({"type":"frame", "seq":1})
        self.assertTrue(mailbox.put_nowait({"type":"frame", "seq":2}))
        messages = [mailbox.get_nowait() for _ in range(3)]
        self.assertEqual(messages[0]["type"], "state")
        self.assertEqual([m["seq"] for m in messages[1:]], [1,2])

    def test_reconnect_snapshot_excludes_historical_diagnostics(self):
        sink = WebSocketSink("127.0.0.1", 8765, 2)

        async def scenario():
            await sink.send({"type": "state", "state": "playing"})
            await sink.send({"type": "metrics", "inference_p95_ms": 10.0})
            await sink.send({"type": "degraded", "window_id": 1})
            await sink.send({"type": "backpressure", "window_id": 2})
            await sink.send({"type": "overload", "window_id": 3})

        asyncio.run(scenario())
        self.assertEqual(set(sink.latest_status), {"state", "metrics"})

    def test_inference_fail_policy_emits_overload(self):
        sink = MemorySink()
        service = StreamingService(
            RealtimeConfig(stream=StreamConfig(inference_queue_policy="fail")),
            FakeInferenceBackend(), None, sink, VirtualClock(), Path("unused.json"),
        )
        service._inference_queue.put_nowait(object())
        window = MotionWindow(0, 0, 150, 0, 1, identity_motion(150))
        with self.assertRaisesRegex(RuntimeError, "fail policy"):
            asyncio.run(service._enqueue_inference(window))
        self.assertEqual(sink.messages[-1]["type"], "overload")
        self.assertEqual(service.metrics.overloads, 1)
        service._inference_executor.shutdown()

    def test_inference_block_policy_waits_for_capacity(self):
        async def scenario():
            sink = MemorySink()
            service = StreamingService(
                RealtimeConfig(), FakeInferenceBackend(), None, sink,
                VirtualClock(), Path("unused.json"),
            )
            service._inference_queue.put_nowait(object())
            window = MotionWindow(0, 0, 150, 0, 1, identity_motion(150))
            pending = asyncio.create_task(service._enqueue_inference(window))
            await asyncio.sleep(0)
            service._inference_queue.get_nowait()
            await pending
            self.assertEqual(sink.messages[-1]["type"], "backpressure")
            self.assertEqual(service.metrics.backpressure_waits, 1)
            service._inference_executor.shutdown()
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
