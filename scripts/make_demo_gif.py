"""Create a lightweight README tour GIF from a real ResearchFlow screenshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _ease(value: float) -> float:
    return value * value * (3 - 2 * value)


def _interpolate(start: tuple[int, ...], end: tuple[int, ...], value: float) -> tuple[int, ...]:
    progress = _ease(value)
    return tuple(round(a + (b - a) * progress) for a, b in zip(start, end))


def _frame(source: Image.Image, crop: tuple[int, int, int, int], caption: str) -> Image.Image:
    canvas = source.crop(crop).resize((1280, 720), Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((34, 638, 620, 696), radius=16, fill=(18, 36, 29, 225))
    draw.text((58, 653), caption, font=_font(24), fill=(244, 249, 246, 255))
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = Image.open(args.source).convert("RGB")
    scenes = [
        ((0, 0, 1440, 810), "1 · 论文库、任务队列与研究工作台"),
        ((270, 330, 1310, 915), "2 · Agent 五阶段执行状态实时可见"),
        ((330, 500, 1170, 973), "3 · 稳定引用可回溯到论文证据"),
    ]
    frames: list[Image.Image] = []
    durations: list[int] = []
    for index, (crop, caption) in enumerate(scenes):
        frames.append(_frame(source, crop, caption))
        durations.append(1300)
        if index == len(scenes) - 1:
            continue
        next_crop, next_caption = scenes[index + 1]
        for step in range(1, 7):
            value = step / 7
            frames.append(_frame(source, _interpolate(crop, next_crop, value), next_caption))
            durations.append(75)
    frames.append(_frame(source, scenes[0][0], "ResearchFlow · Evidence-grounded Agent"))
    durations.append(1500)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
