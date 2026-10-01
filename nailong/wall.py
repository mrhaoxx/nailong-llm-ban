"""/wall：把一段时间内被撤回的奶龙全部拼成一张大图。

有动图时输出动态 GIF，每一格按原图自己的节奏循环，帧率取所有动图里最快的那个，不降帧。
GIF 逐帧流式写出（只编码相对上一帧变化的区域），内存只保留当前帧和上一帧；
动图格子边播边解码，不预先解出全部帧。体积超出预算时只缩小格子，不减帧。
"""

from __future__ import annotations

import bisect
import io
import math
import statistics
from pathlib import Path
from typing import Iterator

from PIL import GifImagePlugin, Image, ImageChops, ImageDraw, ImageFont

TARGET_WIDTH = 1600
MIN_CELL = 32
MAX_CELL = 240
MAX_BYTES = 20 * 1024 * 1024
# GIF 帧间隔的下限：低于 20ms 时多数播放器会按 100ms 播放
MIN_STEP = 20
# 一轮动画最长多久；更长的动图播到这里会从头开始
MAX_LOOP = 6000
BG = (24, 24, 24)
TILE_BG = (255, 255, 255)
TEXT = (240, 240, 236)
TEXT_2 = (170, 170, 164)
BADGE = (0, 0, 0)


class TooLarge(Exception):
    pass


def _fit(frame: Image.Image, cell: int) -> Image.Image:
    im = frame.convert("RGBA")
    im.thumbnail((cell, cell))
    out = Image.new("RGB", (cell, cell), TILE_BG)
    out.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
    return out


def durations(path: Path) -> list[int]:
    """各帧显示时长（毫秒）。静态图返回 [0]。"""
    try:
        im = Image.open(path)
        n = getattr(im, "n_frames", 1)
        if n <= 1:
            return [0]
        out = []
        for i in range(n):
            im.seek(i)
            out.append(max(MIN_STEP, int(im.info.get("duration") or 100)))
        return out
    except Exception:
        return [0]


class Tile:
    """一格。动图按需 seek 到当前时间对应的帧再缩放，只缓存当前这一帧。"""

    def __init__(self, path: Path, count: int, cell: int, frame_ms: list[int]):
        self.count = count
        self.cell = cell
        self.static: Image.Image | None = None
        self.im: Image.Image | None = None
        try:
            if len(frame_ms) <= 1:
                src = Image.open(path)
                src.seek(0)
                self.static = _fit(src, cell)
                return
            self.im = Image.open(path)
        except Exception:
            self.static = Image.new("RGB", (cell, cell), TILE_BG)
            return
        self.starts = [0]
        for ms in frame_ms[:-1]:
            self.starts.append(self.starts[-1] + ms)
        self.total = sum(frame_ms)
        self.index = -1
        self.current: Image.Image | None = None

    def at(self, t: int) -> Image.Image:
        if self.static is not None:
            return self.static
        i = bisect.bisect_right(self.starts, t % self.total) - 1
        if i != self.index:
            try:
                self.im.seek(i)
                self.current = _fit(self.im, self.cell)
            except Exception:
                self.current = self.current or Image.new("RGB", (self.cell, self.cell), TILE_BG)
            self.index = i
        return self.current


def cell_size(n: int) -> int:
    cols = max(1, math.ceil(math.sqrt(n)))
    return max(MIN_CELL, min(MAX_CELL, (TARGET_WIDTH - 32 - (cols - 1) * 6) // cols))


def timeline(all_durations: list[list[int]]) -> tuple[int, int]:
    """(帧间隔, 一轮时长)。帧间隔取各动图典型帧时长里最短的，保证最快的那张也不丢帧。"""
    animated = [d for d in all_durations if len(d) > 1]
    step = min(max(MIN_STEP, int(statistics.median(d)) // 10 * 10) for d in animated)
    loop = min(MAX_LOOP, max(sum(d) for d in animated))
    return step, max(step, loop // step * step)


def render(items: list[tuple[Path, int]], title: str, subtitle: str) -> tuple[bytes, str]:
    """items: (图片路径, 被撤回次数)，全部放进一张图。返回 (图片数据, 格式)，格式为 "gif"、"jpeg"，
    或 "jpeg-fallback"（有动图但格子缩到最小也超出体积预算，退成了静态图）。"""
    all_durations = [durations(p) for p, _ in items]
    cell = cell_size(len(items))
    if not any(len(d) > 1 for d in all_durations):
        return _render_static(items, title, subtitle, cell), "jpeg"

    step, loop = timeline(all_durations)
    times = list(range(0, loop, step))
    # 先按当前格子渲染前 3 帧估算整段体积，算出预算内的格子大小（体积约与面积成正比）
    probe = _render_gif(items, all_durations, title, subtitle, cell, step, times[:3])
    estimate = len(probe) / min(3, len(times)) * len(times)
    if estimate > MAX_BYTES * 0.9:
        cell = max(MIN_CELL, int(cell * math.sqrt(MAX_BYTES * 0.8 / estimate)))
    while True:
        try:
            return _render_gif(items, all_durations, title, subtitle, cell, step, times, limit=MAX_BYTES), "gif"
        except TooLarge:
            if cell <= MIN_CELL:
                break
            cell = max(MIN_CELL, int(cell * 0.85))
    return _render_static(items, title, subtitle, cell_size(len(items))), "jpeg-fallback"


def _layout(n: int, cell: int, title: str, subtitle: str):
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    gap, pad, head = 6, 16, 64
    W = pad * 2 + cols * cell + (cols - 1) * gap
    H = pad + head + rows * cell + (rows - 1) * gap + pad
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    d.text((pad, pad), title, fill=TEXT, font=ImageFont.load_default(size=26))
    d.text((pad, pad + 34), subtitle, fill=TEXT_2, font=ImageFont.load_default(size=15))
    positions = [(pad + (i % cols) * (cell + gap), pad + head + (i // cols) * (cell + gap)) for i in range(n)]
    return base, positions, ImageFont.load_default(size=max(12, cell // 9))


def _compose(base, tiles: list[Tile], positions, font, t: int) -> Image.Image:
    frame = base.copy()
    d = ImageDraw.Draw(frame)
    for tile, (x, y) in zip(tiles, positions):
        frame.paste(tile.at(t), (x, y))
        if tile.count > 1:
            label = f"x{tile.count}"
            box = d.textbbox((0, 0), label, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            d.rectangle((x, y, x + w + 10, y + h + 10), fill=BADGE)
            d.text((x + 5 - box[0], y + 5 - box[1]), label, fill=(255, 255, 255), font=font)
    return frame


def _render_static(items: list[tuple[Path, int]], title: str, subtitle: str, cell: int) -> bytes:
    base, positions, font = _layout(len(items), cell, title, subtitle)
    tiles = [Tile(p, c, cell, [0]) for p, c in items]
    buf = io.BytesIO()
    _compose(base, tiles, positions, font, 0).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _render_gif(items, all_durations, title, subtitle, cell, step, times, limit: int | None = None) -> bytes:
    base, positions, font = _layout(len(items), cell, title, subtitle)
    tiles = [Tile(p, c, cell, d) for (p, c), d in zip(items, all_durations)]
    frames = (_compose(base, tiles, positions, font, t) for t in times)
    buf = io.BytesIO()
    write_gif(buf, frames, step, limit)
    return buf.getvalue()


_NETSCAPE_LOOP = b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"


def write_gif(fp: io.BytesIO, frames: Iterator[Image.Image], duration: int, limit: int | None = None) -> None:
    """逐帧写 GIF：每帧自带调色板，只编码相对上一帧变化的矩形区域（其余部分保留上一帧）。"""
    prev: Image.Image | None = None
    for frame in frames:
        if prev is None:
            region, offset = frame, (0, 0)
        else:
            # 完全没变化时也写一个 1x1 的帧，占住这段时长
            bbox = ImageChops.difference(frame, prev).getbbox() or (0, 0, 1, 1)
            region, offset = frame.crop(bbox), bbox[:2]
        p = region.quantize(colors=256)
        if prev is None:
            header, _ = GifImagePlugin.getheader(p)
            fp.write(b"".join(header))
            fp.write(_NETSCAPE_LOOP)
        for chunk in GifImagePlugin.getdata(p, offset, duration=duration, disposal=1, include_color_table=True):
            fp.write(chunk)
        prev = frame
        if limit is not None and fp.tell() > limit:
            raise TooLarge
    fp.write(b";")
