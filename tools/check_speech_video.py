"""Check video indexing (plan D1) on real speech: mux a real recording into an mp4 with a few frames,
then parse it with the video parser and print the frame and sound-window counts.

usage: python tools/check_speech_video.py <recording.wav>
"""

from __future__ import annotations

import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from archive_index.audio_io import read_audio
from archive_index.sources import parse_video


def main() -> int:
    recording = Path(sys.argv[1])
    speech = read_audio(recording)
    seconds = len(speech) / 16000
    out = Path(tempfile.mkdtemp()) / "speech_video.mp4"
    container = av.open(str(out), "w")
    video = container.add_stream("mpeg4", rate=5)
    video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
    audio = container.add_stream("aac", rate=16000)
    audio.layout = "mono"
    for i in range(int(seconds) * 5):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8), format="rgb24")
        frame.pts, frame.time_base = i, Fraction(1, 5)
        for packet in video.encode(frame):
            container.mux(packet)
    samples = av.AudioFrame.from_ndarray(speech[None, :].astype(np.float32), format="flt", layout="mono")
    samples.sample_rate, samples.pts = 16000, 0
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    for frame in resampler.resample(samples):
        for packet in audio.encode(frame):
            container.mux(packet)
    for packet in video.encode() + audio.encode():
        container.mux(packet)
    container.close()
    doc = parse_video(out, out.parent)
    frames = [u for u in doc.units if u.image]
    windows = [u for u in doc.units if u.audio is not None]
    print(f"recording {recording.name}: {seconds:.1f} s of speech")
    print(f"frames: {len(frames)}, sound windows: {len(windows)}, first window: {windows[0].locator if windows else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
