"""Reference vectors (plan G3): detect silent changes in the model or its libraries.

--create  writes evals/reference.npz with vectors of fixed texts, images and sound.
default   recomputes them and fails (exit 1) if any cosine similarity to the reference is below 0.999.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REF = Path("evals/reference.npz")
TEXTS = [  # 20 fixed sentences (plan G3): Ukrainian, English and Russian, technical and everyday
    "Антена працює на частоті 1800 МГц.",
    "The relay station forwards packets between two sites.",
    "Перевірити плату RVZ0014A перед закупівлею.",
    "Quarterly budget summary with three cost centers.",
    "Звіт про стан мережі за минулий тиждень.",
    "Базова станція обслуговує сектор з кутом дев'яносто градусів.",
    "Протокол обміну даними використовує шифрування AES-256.",
    "Оновлення прошивки займає близько двадцяти хвилин.",
    "Налаштуйте точку доступу з паролем не коротше дванадцяти символів.",
    "Grandmother baked apple pie for the whole family on Sunday.",
    "The compiler reports an undefined reference to the symbol main.",
    "Monthly sales grew by twelve percent compared with March.",
    "Датчик температури передає показники кожні п'ять секунд.",
    "Пожарная сигнализация срабатывает при повышении температуры.",
    "Схема подключения приёмника показана на рисунке три.",
    "Meeting notes: agreed to ship the beta on Friday afternoon.",
    "Заявка на закупівлю обладнання погоджена фінансовим відділом.",
    "The configuration file lists three ports and two hosts.",
    "Список задач на наступний спринт уточнити з командою.",
    "Ремонт кабельної лінії завершено, сигнал стабільний.",
]
IMAGE_DIR = Path("evals/reference_images")


def _make_images() -> list[Path]:
    from PIL import Image, ImageDraw

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    specs = {
        "solid.png": lambda: Image.new("RGB", (64, 64), (200, 30, 40)),
        "gradient.png": lambda: Image.fromarray(np.stack([np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))] * 3, axis=-1)),
        "checker.png": lambda: _checker(),
        "circle.png": lambda: _shape("ellipse"),
        "stripes.png": lambda: _shape("stripes"),
    }
    for name, make in specs.items():
        path = IMAGE_DIR / name
        make().save(path)
        paths.append(path)
    return paths


def _checker():
    from PIL import Image

    arr = np.indices((64, 64)).sum(axis=0) % 16 < 8
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def _shape(kind: str):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (64, 64), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    if kind == "ellipse":
        draw.ellipse((8, 8, 56, 56), fill=(30, 90, 200))
    else:
        for y in range(0, 64, 8):
            draw.rectangle((0, y, 63, y + 3), fill=(20, 20, 20))
    return img


def _sound() -> list[np.ndarray]:
    """Five fixed signals: two tones, a decaying tone, a chirp and a square wave (plan G3)."""
    t = np.arange(0, 2.0, 1 / 16000, dtype=np.float32)
    chirp = np.sin(2 * np.pi * (200 + 400 * t) * t)
    square = np.sign(np.sin(2 * np.pi * 300 * t))
    return [
        0.3 * np.sin(2 * np.pi * 440 * t),
        0.3 * np.sin(2 * np.pi * 1200 * t),
        0.3 * np.sin(2 * np.pi * 1200 * t) * np.exp(-t),
        0.3 * chirp,
        0.3 * square,
    ]


def compute(embedder) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    texts = embedder.documents(TEXTS, ["ref"] * len(TEXTS))
    for i, v in enumerate(texts):
        out[f"text{i}"] = np.asarray(v, dtype=np.float32)
    for i, path in enumerate(_make_images()):
        out[f"image{i}"] = np.asarray(embedder.image(str(path)), dtype=np.float32)
    for i, samples in enumerate(_sound()):
        out[f"audio{i}"] = np.asarray(embedder.audio(samples), dtype=np.float32)
    assert len(TEXTS) == 20 and len(_make_images()) == 5 and len(_sound()) == 5
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    from archive_index.embed import Embedder

    current = compute(Embedder())
    if args.create or not REF.exists():
        REF.parent.mkdir(parents=True, exist_ok=True)
        np.savez(REF, **current)
        print(f"reference written: {len(current)} vectors to {REF}")
        return 0
    stored = np.load(REF)
    failures = []
    for key, vector in current.items():
        ref = stored[key]
        cos = float(np.dot(vector, ref) / (np.linalg.norm(vector) * np.linalg.norm(ref) + 1e-12))
        status = "ok" if cos >= 0.999 else "CHANGED"
        print(f"{key:8s} cos={cos:.5f} {status}")
        if cos < 0.999:
            failures.append(key)
    if failures:
        print("reference mismatch: rebuild the index (archive index --force) and review the change")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
