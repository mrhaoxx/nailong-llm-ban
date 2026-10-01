from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path

import httpx
from PIL import Image, ImageChops, ImageSequence, ImageStat

from .config import FrameConfig


def mface_url(emoji_id: str) -> str:
    """QQ 商城表情的原图地址。"""
    return f"https://gxh.vip.qq.com/club/item/parcel/item/{emoji_id[:2]}/{emoji_id}/raw300.gif"


async def fetch_bytes(client: httpx.AsyncClient, src: str, max_bytes: int) -> bytes:
    if src.startswith("file://"):
        src = src[len("file://") :]
    if not src.startswith(("http://", "https://")):
        data = Path(src).read_bytes()
    else:
        data = b""
        async with client.stream("GET", src) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                data += chunk
                if len(data) > max_bytes:
                    raise ValueError(f"图片超过 {max_bytes} 字节上限")
    if len(data) > max_bytes:
        raise ValueError(f"图片超过 {max_bytes} 字节上限")
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(frame: Image.Image) -> Image.Image:
    # 用彩色缩略图而不是灰度：红/绿这类亮度相近的画面转灰度后几乎一样
    return frame.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR)


def _diff(a: Image.Image, b: Image.Image) -> float:
    mean = ImageStat.Stat(ImageChops.difference(a, b)).mean
    return sum(mean) / len(mean)


def extract_frames(img: Image.Image, cfg: FrameConfig) -> list[Image.Image]:
    """取出去重后的关键帧，再从中均匀挑最多 max_frames 帧。

    表情包常把少数画面循环播放，和所有已保留帧比较（而不只是上一帧），
    这样循环回来的重复画面也会被去掉。
    """
    unique: list[Image.Image] = []
    sigs: list[Image.Image] = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i >= cfg.max_scan:
            break
        f = frame.convert("RGBA")
        if cfg.dedupe_threshold > 0:
            sig = _signature(f)
            if any(_diff(sig, s) < cfg.dedupe_threshold for s in sigs):
                continue
            sigs.append(sig)
        unique.append(f)

    n = max(1, cfg.max_frames)
    if len(unique) > n:
        step = len(unique) / n
        unique = [unique[int(i * step)] for i in range(n)]
    return unique


def to_data_uri(data: bytes, cfg: FrameConfig) -> str:
    """把原图转成适合喂给视觉模型的 PNG data URI。

    动图取去重后的关键帧拼成网格，避免只看第一帧漏判；
    部分 API 不收 GIF，所以统一转 PNG 并缩小尺寸以节省 token。
    """
    frames = extract_frames(Image.open(io.BytesIO(data)), cfg)

    if len(frames) == 1:
        out = frames[0]
    else:
        w, h = frames[0].size
        cols = math.ceil(math.sqrt(len(frames)))
        rows = math.ceil(len(frames) / cols)
        out = Image.new("RGBA", (w * cols, h * rows), (255, 255, 255, 255))
        for i, f in enumerate(frames):
            out.paste(f.resize((w, h)), ((i % cols) * w, (i // cols) * h))

    out.thumbnail((cfg.max_side, cfg.max_side))
    bg = Image.new("RGB", out.size, (255, 255, 255))
    bg.paste(out, mask=out.getchannel("A"))
    buf = io.BytesIO()
    bg.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


BADGE_DEFAULT = (0, 0, 0)
BADGE_YES = (214, 150, 0)
BADGE_NO = (40, 90, 180)


def make_collage(images: list[tuple[str, bytes] | tuple[str, bytes, tuple[int, int, int]]], cell: int = 240) -> bytes:
    """把 (角标文字, 图片[, 角标颜色]) 拼成带角标的网格 JPEG，用于 /rewind、/examples。动图取第一帧。

    默认字体没有中文字形，角标只放编号和 ASCII 字母。
    """
    from PIL import ImageDraw, ImageFont

    cols = max(1, math.ceil(math.sqrt(len(images))))
    rows = max(1, math.ceil(len(images) / cols))
    gap = 6
    sheet = Image.new("RGB", (cols * cell + (cols + 1) * gap, rows * cell + (rows + 1) * gap), (40, 40, 40))
    font = ImageFont.load_default(size=cell // 6)
    draw = ImageDraw.Draw(sheet)
    for i, tile in enumerate(images):
        label, data = str(tile[0]), tile[1]
        badge = tile[2] if len(tile) > 2 else BADGE_DEFAULT
        x = gap + (i % cols) * (cell + gap)
        y = gap + (i // cols) * (cell + gap)
        draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(255, 255, 255))
        try:
            im = Image.open(io.BytesIO(data))
            im.seek(0)
            im = im.convert("RGBA")
            im.thumbnail((cell, cell))
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2), im)
        except Exception:
            draw.text((x + cell // 2 - 10, y + cell // 2 - 20), "?", fill=(0, 0, 0), font=font)
        # 左上角角标：深色底白字，任何背景上都看得清
        box = draw.textbbox((0, 0), label, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        pad = cell // 30 + 4
        draw.rectangle((x, y, x + w + 2 * pad, y + h + 2 * pad), fill=badge)
        draw.text((x + pad - box[0], y + pad - box[1]), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
