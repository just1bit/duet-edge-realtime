import unittest

from duet_edge_realtime.schemas import MotionFrame
from duet_edge_realtime.window_buffer import SequenceError, SlidingWindowBuffer

from helpers import identity_motion


class WindowBufferTests(unittest.TestCase):
    def test_three_hop_windows_for_300_frames(self):
        buffer = SlidingWindowBuffer()
        windows = []
        for seq, vector in enumerate(identity_motion(300)):
            window = buffer.push(MotionFrame(seq, seq / 30, vector), seq / 30)
            if window:
                windows.append(window)
        self.assertEqual([(w.start_seq, w.end_seq) for w in windows], [(0,150),(75,225),(150,300)])
        self.assertEqual(buffer.retained_frames, 150)
        self.assertIsNone(buffer.flush(10.0))

    def test_boundaries_and_padded_tail(self):
        for count in (149, 150, 151, 224, 225):
            buffer = SlidingWindowBuffer()
            windows = []
            for seq, vector in enumerate(identity_motion(count)):
                value = buffer.push(MotionFrame(seq, seq / 30, vector), seq / 30)
                if value:
                    windows.append(value)
            if count < 150:
                self.assertEqual(windows, [])
                with self.assertRaises(SequenceError):
                    buffer.flush(count / 30)
            elif count in (150, 225):
                self.assertIsNone(buffer.flush(count / 30))
            else:
                tail = buffer.flush(count / 30)
                self.assertEqual(tail.valid_frames, count - 150)
                self.assertEqual(tail.start_seq, 75)
                self.assertEqual(tail.motion.shape, (150,151))

    def test_sequence_errors(self):
        vector = identity_motion(1)[0]
        for bad in (0, 2):
            buffer = SlidingWindowBuffer()
            buffer.push(MotionFrame(0, 0, vector), 0)
            with self.assertRaises(SequenceError):
                buffer.push(MotionFrame(bad, 0, vector), 0)


if __name__ == "__main__":
    unittest.main()
