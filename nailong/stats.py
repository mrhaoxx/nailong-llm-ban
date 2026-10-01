"""/stats：统计一段时间内的检测、撤回、人工标注，并画成一张英文统计卡片。"""

from __future__ import annotations

import bisect
import io
import json
from collections import Counter
from dataclasses import dataclass, field
import math
from datetime import datetime, timedelta

from PIL import Image, ImageDraw, ImageFont

from .cache import VerdictCache
from .periods import Window
from .store import SampleStore, _ts

@dataclass
class Stats:
    title: str
    since: float
    until: float
    buckets: list[tuple[str, int]]
    bucket_unit: str
    checked: int = 0
    detected: int = 0
    recalled: int = 0
    banned: int = 0
    reverted: int = 0
    model_new: int = 0
    human_labels: int = 0
    # 人工标注相对模型当时判定：确认正确 / 模型漏判（人工是、模型否）/ 模型误判（人工否、模型是）。
    # 管理员通常只标模型判错的图，所以这不是模型准确率，而是被人工抓到的错误
    confirmed: int = 0
    missed: int = 0
    false_alarms: int = 0
    top_senders: list[tuple[int, int]] = field(default_factory=list)
    top_images: list[tuple[str, int]] = field(default_factory=list)


def _buckets(since: float, until: float) -> tuple[list[datetime], timedelta, str, str]:
    """按跨度选柱状图粒度：两天内按小时，三个月内按天，更长按周。返回 (各桶起点, 步长, 单位, 标签格式)。"""
    span = until - since
    start = datetime.fromtimestamp(since)
    if span <= 2 * 86400:
        step, unit, fmt = timedelta(hours=1), "hour", "%H"
        start = start.replace(minute=0, second=0, microsecond=0)
    elif span <= 92 * 86400:
        step, unit, fmt = timedelta(days=1), "day", "%m-%d"
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        step, unit, fmt = timedelta(weeks=1), "week", "%m-%d"
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        start -= timedelta(days=start.weekday())
    end = datetime.fromtimestamp(until)
    edges = []
    while start < end:
        edges.append(start)
        start += step
    return edges or [datetime.fromtimestamp(since)], step, unit, fmt


def compute(
    cache: VerdictCache, store: SampleStore, threshold: float, window: Window, group_id: int | None = None,
) -> Stats:
    seen = cache.seen_since(window.since, group_id, window.until)
    actions = cache.actions_since(window.since, group_id, window.until)
    since = window.since
    if since <= 0:
        # "all"：从有数据的最早时间开始画
        first = [t for t, _, _ in seen] + [a["time"] for a in actions]
        since = min(first) if first else window.until - 86400
    edges, step, unit, fmt = _buckets(since, window.until)
    st = Stats(window.en, since, window.until, [], unit)

    counts = [0] * len(edges)
    st.checked = len(seen)
    image_hits: Counter[str] = Counter()
    for t, sha, hit in seen:
        if not hit:
            continue
        st.detected += 1
        image_hits[sha] += 1
        i = bisect.bisect_right(edges, datetime.fromtimestamp(t)) - 1
        if 0 <= i < len(counts):
            counts[i] += 1
    st.buckets = [(e.strftime(fmt), c) for e, c in zip(edges, counts)]
    st.top_images = image_hits.most_common(4)

    messages = {a["message_id"]: a for a in actions}
    st.recalled = len(messages)
    st.banned = sum(1 for a in messages.values() if a["banned"])
    st.reverted = len({a["message_id"] for a in actions if a["reverted"]})
    st.top_senders = Counter(a["user_id"] for a in messages.values()).most_common(5)

    # labels.jsonl：模型对新图的判定、人工标注（含被纠正前的模型判定）
    lines = store.manifest.open(encoding="utf-8") if store.manifest.exists() else io.StringIO()
    with lines as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = _ts(r.get("time"))
            if t is None or not (window.since <= t < window.until) or r.get("migrated"):
                continue
            if group_id is not None and r.get("group_id") not in (None, group_id):
                continue
            if r.get("source") == "model":
                st.model_new += 1
            elif r.get("source") == "human":
                st.human_labels += 1
                prev = r.get("previous") or {}
                if prev.get("model") and not str(prev["model"]).startswith("admin:"):
                    decision = bool(prev.get("label")) and (prev.get("confidence") or 0) >= threshold
                    human = bool(r.get("label"))
                    if decision == human:
                        st.confirmed += 1
                    elif human:
                        st.missed += 1
                    else:
                        st.false_alarms += 1
    return st


# ---------- 绘图 ----------

SURFACE = (252, 252, 251)
TILE = (243, 242, 238)
GRID = (226, 225, 219)
TEXT = (11, 11, 11)
TEXT_2 = (82, 81, 78)
TEXT_3 = (130, 129, 124)
SERIES = (42, 120, 214)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _rounded_top_bar(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, fill: tuple) -> None:
    """柱子顶端 4px 圆角，底部贴着基线是直角。"""
    r = min(4, (x1 - x0) / 2, (y1 - y0))
    if y1 - y0 < 1:
        return
    draw.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=fill)
    draw.rectangle((x0, max(y0, y1 - r), x1, y1), fill=fill)


def render(st: Stats, store: SampleStore, group_id: int | None) -> bytes:
    W, pad = 1000, 32
    img = Image.new("RGB", (W, 1400), SURFACE)
    d = ImageDraw.Draw(img)
    y = pad

    d.text((pad, y), f"Nailong stats · {st.title}", fill=TEXT, font=_font(34))
    span = f"{datetime.fromtimestamp(st.since):%Y-%m-%d %H:%M} - {datetime.fromtimestamp(st.until):%Y-%m-%d %H:%M}"
    y += 46
    scope = f"Group {group_id}" if group_id else "All groups"
    d.text((pad, y), f"{scope} · {span}", fill=TEXT_2, font=_font(18))
    y += 44

    # 数字卡片：3 x 2
    tiles = [
        ("Images checked", f"{st.checked}", ""),
        ("Nailong detected", f"{st.detected}", f"{st.detected / st.checked:.0%} of images" if st.checked else ""),
        ("Messages recalled", f"{st.recalled}", f"{st.banned} muted · {st.reverted} undone" if st.recalled else ""),
        ("New images judged", f"{st.model_new}", "model calls on unseen images"),
        ("Human labels", f"{st.human_labels}", f"{st.confirmed} confirmed the model" if st.human_labels else ""),
        ("Model corrected", f"{st.missed + st.false_alarms}",
         f"{st.missed} missed nailong · {st.false_alarms} false alarms"),
    ]
    cols, gap = 3, 12
    tw = (W - 2 * pad - (cols - 1) * gap) / cols
    th = 108
    for i, (label, value, sub) in enumerate(tiles):
        x0 = pad + (i % cols) * (tw + gap)
        y0 = y + (i // cols) * (th + gap)
        d.rounded_rectangle((x0, y0, x0 + tw, y0 + th), radius=10, fill=TILE)
        d.text((x0 + 16, y0 + 12), label, fill=TEXT_2, font=_font(17))
        d.text((x0 + 16, y0 + 36), value, fill=TEXT, font=_font(40))
        if sub:
            d.text((x0 + 16, y0 + 82), sub, fill=TEXT_3, font=_font(14))
    y += 2 * th + gap + 32

    # 柱状图：每小时/每天检测到的奶龙
    d.text((pad, y), f"Nailong detected per {st.bucket_unit}", fill=TEXT, font=_font(22))
    y += 36
    ch_top, ch_h = y, 200
    left, right = pad + 36, W - pad
    peak = max((c for _, c in st.buckets), default=0)
    top = max(4, peak)
    nice = next(v for v in (4, 8, 12, 20, 40, 60, 100, 200, 400, 1000, 10**9) if v >= top)
    for k in range(5):
        gy = ch_top + ch_h - ch_h * k / 4
        d.line((left, gy, right, gy), fill=GRID, width=1)
        d.text((pad, gy - 9), f"{nice * k // 4}", fill=TEXT_3, font=_font(14))
    n = max(1, len(st.buckets))
    slot = (right - left) / n
    bw = max(2.0, min(28.0, slot - 2))
    label_every = max(1, math.ceil(n / 12))
    for i, (label, count) in enumerate(st.buckets):
        cx = left + slot * i + slot / 2
        h = ch_h * count / nice
        _rounded_top_bar(d, cx - bw / 2, ch_top + ch_h - h, cx + bw / 2, ch_top + ch_h, SERIES)
        if count and count == peak:
            d.text((cx, ch_top + ch_h - h - 6), str(count), fill=TEXT, font=_font(14), anchor="ms")
        if i % label_every == 0:
            d.text((cx, ch_top + ch_h + 8), label, fill=TEXT_3, font=_font(13), anchor="mt")
    y = ch_top + ch_h + 44

    # 下方两栏：发送排行 / 最常出现的奶龙
    colw = (W - 2 * pad - 24) / 2
    d.text((pad, y), "Top senders (recalled messages)", fill=TEXT, font=_font(22))
    d.text((pad + colw + 24, y), "Most frequent nailong", fill=TEXT, font=_font(22))
    y += 38
    if st.top_senders:
        most = st.top_senders[0][1]
        for i, (uid, count) in enumerate(st.top_senders):
            ry = y + i * 34
            d.text((pad, ry + 4), str(uid), fill=TEXT_2, font=_font(16))
            bx0 = pad + 130
            bx1 = bx0 + (colw - 180) * count / most
            d.rounded_rectangle((bx0, ry + 4, max(bx0 + 4, bx1), ry + 22), radius=4, fill=SERIES)
            d.text((max(bx0 + 4, bx1) + 8, ry + 4), str(count), fill=TEXT, font=_font(16))
    else:
        d.text((pad, y), "Nothing recalled in this period", fill=TEXT_3, font=_font(16))
    tx = pad + colw + 24
    thumb = int((colw - 3 * 10) / 4)
    if st.top_images:
        for i, (sha, count) in enumerate(st.top_images):
            x0 = int(tx + i * (thumb + 10))
            d.rounded_rectangle((x0, y, x0 + thumb, y + thumb), radius=8, fill=TILE)
            path = store.find(sha)
            if path is not None:
                try:
                    im = Image.open(path)
                    im.seek(0)
                    im = im.convert("RGB")
                    im.thumbnail((thumb - 8, thumb - 8))
                    img.paste(im, (x0 + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
                except Exception:
                    pass
            d.text((x0 + thumb / 2, y + thumb + 6), f"x{count}", fill=TEXT_2, font=_font(15), anchor="mt")
    else:
        d.text((tx, y), "No detections yet", fill=TEXT_3, font=_font(16))
    y += max(5 * 34, thumb + 30) + pad

    img = img.crop((0, 0, W, int(y)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
