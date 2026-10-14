import numpy as np
import pytest

from archive_index.indexer import run_index
from archive_index.store import Store


class FakeEmbedder:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return np.ones(4, dtype=np.float32)

    def image(self, path):
        return np.ones(4, dtype=np.float32)

    def audio(self, samples):
        return np.ones(4, dtype=np.float32)


def _make_video(path, seconds=12):
    import av
    from fractions import Fraction

    container = av.open(str(path), mode="w")
    vstream = container.add_stream("mpeg4", rate=10)
    vstream.width, vstream.height, vstream.pix_fmt = 64, 48, "yuv420p"
    astream = container.add_stream("aac", rate=16000)
    astream.layout = "mono"
    for i in range(seconds * 10):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i % 255, dtype=np.uint8), format="rgb24")
        frame.pts = i
        frame.time_base = Fraction(1, 10)
        for packet in vstream.encode(frame):
            container.mux(packet)
    t = np.arange(16000 * seconds) / 16000.0
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :]
    audio = av.AudioFrame.from_ndarray(tone, format="flt", layout="mono")
    audio.sample_rate = 16000
    audio.pts = 0
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    for frame in resampler.resample(audio):
        for packet in astream.encode(frame):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)
    for packet in astream.encode():
        container.mux(packet)
    container.close()


@pytest.mark.skipif(False, reason="")
def test_video_gives_frame_images_and_sound_windows(tmp_path):
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    try:
        _make_video(projects / "clip.mp4")
    except Exception as exc:  # the bundled encoder may lack a codec on some builds
        pytest.skip(f"cannot encode test video: {exc}")
    store = Store(tmp_path / "v.sqlite3")
    report = run_index(store, FakeEmbedder(), projects_root=tmp_path / "Projects", home=tmp_path / "h",
                       collections=("video",))
    assert report.errors == []
    locators = [r[0] for r in store.conn.execute("SELECT locator FROM chunks WHERE collection='video'")]
    assert any(loc.startswith("frame 00:00") for loc in locators)
    assert any(loc.endswith("s") and "-" in loc for loc in locators)  # sound windows
    store.close()


def test_m4a_decodes_through_the_av_fallback(tmp_path):
    import av
    from fractions import Fraction

    from archive_index.audio_io import _decode_with_av

    f = tmp_path / "voice.m4a"
    c = av.open(str(f), mode="w")
    st = c.add_stream("aac", rate=44100)
    st.layout = "mono"
    samples = (0.2 * np.sin(2 * np.pi * 300 * np.arange(44100 * 2) / 44100)).astype(np.float32)[None, :]
    frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
    frame.sample_rate = 44100
    frame.pts = 0
    frame.time_base = Fraction(1, 44100)
    for pkt in st.encode(frame):
        c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    out = _decode_with_av(f)
    assert out.dtype == np.float32 and abs(len(out) / 16000 - 2.0) < 0.1
