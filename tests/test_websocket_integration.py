import asyncio
import json
import os
import unittest

import websockets

from duet_edge_realtime.sinks import WebSocketSink


@unittest.skipUnless(os.environ.get("RUN_NETWORK_TESTS") == "1", "set RUN_NETWORK_TESTS=1 where loopback bind is allowed")
class WebSocketIntegrationTests(unittest.TestCase):
    def test_hello_frame_and_status_reconnect(self):
        async def scenario():
            sink = WebSocketSink("127.0.0.1", 18765, 4)
            hello = {"type":"hello", "protocol":"duet-edge-stream/v1", "parents":[-1]+[0]*23}
            await sink.start(hello)
            try:
                async with websockets.connect("ws://127.0.0.1:18765") as client:
                    self.assertEqual(json.loads(await client.recv())["type"], "hello")
                    await sink.send({"type":"frame", "seq":0, "joints":[[0,0,0]]*24})
                    self.assertEqual(json.loads(await client.recv())["seq"], 0)
                    await sink.send({"type":"metrics", "inference_p95_ms":1.0})
                    self.assertEqual(json.loads(await client.recv())["type"], "metrics")
                async with websockets.connect("ws://127.0.0.1:18765") as client:
                    self.assertEqual(json.loads(await client.recv())["type"], "hello")
                    self.assertEqual(json.loads(await client.recv())["type"], "metrics")
            finally:
                await sink.close()
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
