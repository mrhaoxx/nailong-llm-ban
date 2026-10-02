"""/wall：把一段时间内被撤回的奶龙全部拼成一张大图。

有动图时输出动态 GIF，每一格按原图自己的节奏循环，帧率取所有动图里最快的那个，不降帧。
GIF 逐帧流式写出（只编码相对上一帧变化的区域），内存只保留当前帧和上一帧；
动图格子边播边解码，不预先解出全部帧。体积超出预算时只缩小格子，不减帧。
"""

from __future__ import annotations

import bisect
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
import statistics
from pathlib import Path
from typing import Iterator

from PIL import GifImagePlugin, Image, ImageDraw, ImageFont

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


# 奶龙墙用的小图：每张被撤回的图预先缩到这个尺寸（动图存动态 WebP，保留全部帧和帧时长；静态图存 JPEG），
# 生成奶龙墙时只读小图，不再逐帧解码几百像素、十几 MB 的原图
THUMB_SIDE = 240
THUMB_WORKERS = min(4, os.cpu_count() or 1)


class TooLarge(Exception):
    pass


def _durations_file(thumb: Path) -> Path:
    return thumb.with_suffix(".json")


def _thumb_files(thumbs_dir: Path, sha: str) -> tuple[Path, Path]:
    return thumbs_dir / f"{sha}.webp", thumbs_dir / f"{sha}.jpg"


def make_thumb(src: Path, thumbs_dir: Path) -> Path:
    """生成（或找到已有的）小图，返回小图路径；失败时返回原图路径。按内容哈希命名，样本被移动也不受影响。"""
    animated_path, static_path = _thumb_files(thumbs_dir, src.stem)
    for existing in (animated_path, static_path):
        if existing.exists():
            return existing
    try:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        im = Image.open(src)
        n = getattr(im, "n_frames", 1)
        if n <= 1:
            im.seek(0)
            _shrink(im).save(static_path.with_suffix(".tmp"), format="JPEG", quality=88)
            static_path.with_suffix(".tmp").replace(static_path)
            return static_path
        frames, ms = [], []
        for i in range(n):
            im.seek(i)
            frames.append(_shrink(im))
            ms.append(max(MIN_STEP, int(im.info.get("duration") or 100)))
        tmp = animated_path.with_suffix(".tmp")
        frames[0].save(tmp, format="WEBP", save_all=True, append_images=frames[1:], duration=ms, loop=0,
                       quality=85, method=2)
        if getattr(Image.open(tmp), "n_frames", 1) != n:
            # 编码器合并了重复帧，帧和时长对不上，不用这份小图
            tmp.unlink()
            return src
        # Pillow 读 WebP 拿不到逐帧时长，另存一份原图的帧时长
        _durations_file(animated_path).write_text(json.dumps(ms))
        tmp.replace(animated_path)
        return animated_path
    except Exception:
        return src


def _shrink(frame: Image.Image) -> Image.Image:
    im = frame.convert("RGBA")
    im.thumbnail((THUMB_SIDE, THUMB_SIDE))
    out = Image.new("RGB", im.size, TILE_BG)
    out.paste(im, (0, 0), im)
    return out


def use_thumbs(items: list[tuple[Path, int]], thumbs_dir: Path) -> list[tuple[Path, int]]:
    """把原图换成小图，缺的小图多线程补齐。"""
    with ThreadPoolExecutor(max_workers=THUMB_WORKERS) as pool:
        paths = list(pool.map(lambda it: make_thumb(it[0], thumbs_dir), items))
    return [(p, count) for p, (_, count) in zip(paths, items)]


def _fit(frame: Image.Image, cell: int) -> Image.Image:
    im = frame.convert("RGBA")
    im.thumbnail((cell, cell))
    out = Image.new("RGB", (cell, cell), TILE_BG)
    out.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
    return out


def _gif_delays(data: bytes) -> list[int] | None:
    """直接从 GIF 的图形控制扩展块读出每帧时长（毫秒），不解码画面。不是 GIF 或格式异常时返回 None。"""
    if data[:6] not in (b"GIF87a", b"GIF89a") or len(data) < 13:
        return None
    pos = 13
    if data[10] & 0x80:  # 全局调色板
        pos += 3 * (2 << (data[10] & 7))
    delays, pending = [], None

    def skip_blocks(i: int) -> int:
        while i < len(data) and data[i]:
            i += data[i] + 1
        return i + 1

    while pos < len(data):
        b = data[pos]
        if b == 0x3B:  # 文件结束
            break
        if b == 0x21:  # 扩展块
            if data[pos + 1] == 0xF9 and pos + 6 < len(data):
                pending = int.from_bytes(data[pos + 4:pos + 6], "little") * 10
            pos = skip_blocks(pos + 2)
        elif b == 0x2C:  # 一帧图像
            flags = data[pos + 9]
            pos += 10
            if flags & 0x80:  # 局部调色板
                pos += 3 * (2 << (flags & 7))
            pos = skip_blocks(pos + 1)  # 跳过 LZW 最小码长和数据子块
            delays.append(max(MIN_STEP, pending or 100))
            pending = None
        else:
            return None
    return delays or None


def durations(path: Path) -> list[int]:
    """各帧显示时长（毫秒）。静态图返回 [0]。小图读旁边的 JSON；GIF 直接读控制块；其它动图格式用 Pillow 逐帧读。"""
    if path.suffix == ".webp" and _durations_file(path).exists():
        try:
            ms = json.loads(_durations_file(path).read_text())
            return ms if len(ms) > 1 else [0]
        except (OSError, ValueError):
            pass
    try:
        with open(path, "rb") as f:
            head = f.read(6)
        if head in (b"GIF87a", b"GIF89a"):
            delays = _gif_delays(path.read_bytes())
            if delays is not None:
                return delays if len(delays) > 1 else [0]
    except OSError:
        return [0]
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
    # 动图排在前面（各组内保持原来的顺序）：每帧变化的区域集中在顶部几行，编码更快、体积更小，格子可以更大
    order = sorted(range(len(items)), key=lambda i: len(all_durations[i]) <= 1)
    items = [items[i] for i in order]
    all_durations = [all_durations[i] for i in order]

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


def _badge(d: ImageDraw.ImageDraw, x: int, y: int, count: int, font) -> None:
    if count <= 1:
        return
    label = f"x{count}"
    box = d.textbbox((0, 0), label, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.rectangle((x, y, x + w + 10, y + h + 10), fill=BADGE)
    d.text((x + 5 - box[0], y + 5 - box[1]), label, fill=(255, 255, 255), font=font)


def _compose(base, tiles: list[Tile], positions, font, t: int) -> Image.Image:
    frame = base.copy()
    d = ImageDraw.Draw(frame)
    for tile, (x, y) in zip(tiles, positions):
        frame.paste(tile.at(t), (x, y))
        _badge(d, x, y, tile.count, font)
    return frame


def _render_static(items: list[tuple[Path, int]], title: str, subtitle: str, cell: int) -> bytes:
    base, positions, font = _layout(len(items), cell, title, subtitle)
    tiles = [Tile(p, c, cell, [0]) for p, c in items]
    buf = io.BytesIO()
    _compose(base, tiles, positions, font, 0).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _render_gif(items, all_durations, title, subtitle, cell, step, times, limit: int | None = None) -> bytes:
    """在同一张画布上逐帧更新：静态格子和角标只画一次，之后每帧只重贴换了帧的动图格子，
    并把这些格子的外接矩形作为本帧要编码的区域。"""
    canvas, positions, font = _layout(len(items), cell, title, subtitle)
    d = ImageDraw.Draw(canvas)
    tiles = [Tile(p, c, cell, dur) for (p, c), dur in zip(items, all_durations)]
    animated = []
    for tile, (x, y) in zip(tiles, positions):
        canvas.paste(tile.at(0), (x, y))
        _badge(d, x, y, tile.count, font)
        if tile.static is None:
            animated.append((tile, x, y))

    def advance(tile: Tile, t: int) -> Image.Image | None:
        """推进到时间 t，换了帧时返回新帧，否则返回 None。每格只被一个线程处理。"""
        before = tile.index
        img = tile.at(t)
        return img if tile.index != before else None

    def frames(pool: ThreadPoolExecutor) -> Iterator[tuple[Image.Image, tuple[int, int, int, int] | None]]:
        yield canvas, None
        for t in times[1:]:
            box = None
            # 各格的解码和缩放互不相关，并行做（Pillow 在解码、缩放时会释放 GIL）
            updates = pool.map(lambda a: advance(a[0], t), animated)
            for (tile, x, y), img in zip(animated, updates):
                if img is None:
                    continue
                canvas.paste(img, (x, y))
                _badge(d, x, y, tile.count, font)
                r = (x, y, x + cell, y + cell)
                box = r if box is None else (min(box[0], r[0]), min(box[1], r[1]), max(box[2], r[2]), max(box[3], r[3]))
            yield canvas, box or (0, 0, 1, 1)

    buf = io.BytesIO()
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        write_gif(buf, frames(pool), step, limit)
    return buf.getvalue()


_NETSCAPE_LOOP = b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"


def write_gif(fp: io.BytesIO, frames: Iterator[tuple[Image.Image, tuple[int, int, int, int] | None]],
              duration: int, limit: int | None = None) -> None:
    """逐帧写 GIF。frames 给出 (整张画面, 本帧变化区域)；区域为 None 时写整张，否则只编码这块（其余保留上一帧）。
    每帧自带调色板；量化用 FASTOCTREE，比默认的 MEDIANCUT 快约 20 倍，画质差别很小。"""
    first = True
    for frame, bbox in frames:
        region, offset = (frame, (0, 0)) if bbox is None else (frame.crop(bbox), bbox[:2])
        p = region.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        if first:
            header, _ = GifImagePlugin.getheader(p)
            fp.write(b"".join(header))
            fp.write(_NETSCAPE_LOOP)
            first = False
        for chunk in GifImagePlugin.getdata(p, offset, duration=duration, disposal=1, include_color_table=True):
            fp.write(chunk)
        if limit is not None and fp.tell() > limit:
            raise TooLarge
    fp.write(b";")
