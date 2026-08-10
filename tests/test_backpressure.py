import asyncio
import unittest

from duet_edge_realtime.sinks import WebSocketSink


class BackpressureTests(unittest.TestCase):
    def test_slow_viewer_drops_only_its_oldest_frame(self):
        dropped = []
        sink = WebSocketSink("127.0.0.1", 8765, 2, lambda: dropped.append(1))
        queue = asyncio.Queue(maxsize=2)
        sink.clients[object()] = (queue, None)
        async def scenario():
            await sink.send({"type":"frame","seq":0})
            await sink.send({"type":"frame","seq":1})
            await sink.send({"type":"frame","seq":2})
        asyncio.run(scenario())
        self.assertEqual(len(dropped), 1)
        self.assertEqual([queue.get_nowait()["seq"], queue.get_nowait()["seq"]], [1,2])


if __name__ == "__main__":
    unittest.main()
