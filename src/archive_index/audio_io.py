"""Decode a recording to mono float32 at 16 kHz and compute overlapping windows.

soundfile handles wav, flac, ogg, opus and mp3; PyAV (bundled ffmpeg) handles what it cannot open,
such as AAC in m4a. Both paths give the same samples format, so the embedder does not care which one ran.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np

RATE = 16_000


def window_ranges(
    duration_s: float,
    window_s: float = 20.0,
    step_s: float = 15.0,
) -> list[tuple[float, float]]:
    """Return overlapping windows that cover the start and end of an audio file."""
    if duration_s <= 0:
        return []
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")
    if duration_s <= window_s:
        return [(0.0, float(duration_s))]

    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start + window_s <= duration_s:
        ranges.append((round(start, 3), round(start + window_s, 3)))
        start += step_s

    tail_start = duration_s - window_s
    if not ranges or tail_start > ranges[-1][0]:
        ranges.append((round(tail_start, 3), round(float(duration_s), 3)))
    return ranges


def _read_with_soundfile(path: Path, target_rate: int = RATE) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("audio file is empty")
    mono = samples.mean(axis=1)
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor)
    return np.asarray(mono, dtype=np.float32)


def read_audio(path: Path) -> np.ndarray:
    try:
        return _read_with_soundfile(path)
    except Exception:
        return _decode_with_av(path)


def _decode_with_av(path: Path) -> np.ndarray:
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("no audio stream")
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None) or []:
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("audio file is empty")
    return np.concatenate(chunks).astype(np.float32)
