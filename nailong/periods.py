"""/stats、/wall 的时间范围：today、yesterday、all、任意时长（30m / 3h / 7d / 2w / 1y）、日期或日期范围。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

USAGE = "时间范围可以写：today、yesterday、all、30m、3h、7d、2w、1y、09-01、2026-09-01、09-01~09-15"

_UNITS = {
    "m": (60, "min", "分钟"), "min": (60, "min", "分钟"), "分钟": (60, "min", "分钟"),
    "h": (3600, "h", "小时"), "小时": (3600, "h", "小时"),
    "d": (86400, "d", "天"), "天": (86400, "d", "天"),
    "w": (7 * 86400, "w", "周"), "周": (7 * 86400, "w", "周"),
    "y": (365 * 86400, "y", "年"), "年": (365 * 86400, "y", "年"),
}
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(min|m|h|d|w|y|分钟|小时|天|周|年)$")
_RANGE_RE = re.compile(r"^(\S+?)\s*(?:~|～|到|至|\.\.)\s*(\S+)$")


@dataclass
class Window:
    since: float
    until: float
    en: str   # 用在英文图片上
    zh: str   # 用在中文回复里


def _midnight(d: date) -> float:
    return datetime(d.year, d.month, d.day).timestamp()


def _parse_date(text: str, today: date) -> date | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})", text)
    if m:
        try:
            d = date(today.year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
        # 只写月日且在未来时，指的是去年
        return d if d <= today else date(today.year - 1, d.month, d.day)
    return None


def parse_window(text: str | None, now: float) -> Window | None:
    t = (text or "").strip().lower()
    today = datetime.fromtimestamp(now).date()
    if t in ("", "today", "今天"):
        return Window(_midnight(today), now, "Today", "今天")
    if t in ("yesterday", "昨天"):
        y = today - timedelta(days=1)
        return Window(_midnight(y), _midnight(today), f"Yesterday ({y:%b %d})", "昨天")
    if t in ("all", "全部"):
        return Window(0, now, "All time", "全部")
    if m := _DURATION_RE.match(t):
        seconds, en_unit, zh_unit = _UNITS[m.group(2)]
        n = float(m.group(1))
        if n <= 0:
            return None
        num = m.group(1)
        return Window(now - n * seconds, now, f"Last {num}{en_unit}", f"最近 {num} {zh_unit}")
    start_text, end_text = (m.group(1), m.group(2)) if (m := _RANGE_RE.match(t)) else (t, t)
    start, end = _parse_date(start_text, today), _parse_date(end_text, today)
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    until = min(now, _midnight(end + timedelta(days=1)))
    if start == end:
        return Window(_midnight(start), until, f"{start:%Y-%m-%d}", f"{start.month}月{start.day}日")
    return Window(_midnight(start), until, f"{start:%b %d} - {end:%b %d}",
                  f"{start.month}月{start.day}日至{end.month}月{end.day}日")
