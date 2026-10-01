"""持续学习：从人工标注的样本里挑正反例，附在每次请求里让模型对照本群的标准。

挑选规则：优先挑模型判错、被人工纠正过的难例，其次是最近标注的。
人工标注一变化（数据库里的标注版本号增加）就重选，但两次重选至少间隔 rebuild_seconds；
标注不变时集合和顺序都不变，请求前缀不变，便于服务商的前缀缓存命中。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from .cache import VerdictCache
from .config import Config
from .image import to_data_uri
from .store import ManifestIndex, SampleStore

log = logging.getLogger(__name__)


COUNTS_SETTING = "examples.counts"

# 已知模型的 (上下文窗口 token, 单次请求最多图片数)
KNOWN_LIMITS = {
    "deepseek-flash": (1_048_576, 600),
    "deepseek-v4-flash": (1_048_576, 600),
    "deepseek-v4-flash-vision-exp": (1_048_576, 600),
    "gpt-4o": (128_000, 500),
    "gpt-4o-mini": (128_000, 500),
    "moonshot-v1-8k-vision-preview": (8_192, 50),
    "moonshot-v1-32k-vision-preview": (32_768, 50),
    "moonshot-v1-128k-vision-preview": (131_072, 50),
}
UNKNOWN_LIMIT = (128_000, 50)
# 每张图按上界估算（DeepSeek 文档：单张图最多 1024 token）；例子实际缩到 384px，远小于这个数
TOKENS_PER_IMAGE = 1024


def examples_limit(cfg: Config) -> int:
    """正反例合计最多几张：按所有视觉 provider 中最小的上下文窗口和图片数算，保证切换备用 provider 时也放得下。"""
    limits = []
    for p in cfg.providers:
        if not p.vision:
            continue
        ctx, images = KNOWN_LIMITS.get(p.model, UNKNOWN_LIMIT)
        ctx, images = p.context_window or ctx, p.max_images or images
        # 预留给系统提示、待测图片和输出
        reserve = min(8_192, ctx // 4) + TOKENS_PER_IMAGE
        limits.append(max(0, min(images - 1, (ctx - reserve) // TOKENS_PER_IMAGE)))
    return min(limits) if limits else 0


@dataclass(frozen=True)
class Example:
    sha: str
    is_nailong: bool
    data_uri: str


class ExampleSet:
    def __init__(self, cfg: Config, store: SampleStore, cache: VerdictCache | None = None):
        self.cfg = cfg
        self.store = store
        self.cache = cache
        self.index = ManifestIndex(store.manifest)
        self._examples: list[Example] = []
        self._uris: dict[str, str] = {}
        self._built_at = float("-inf")
        self._version: int | None = None
        self.updated_at: float | None = None
        self._lock = asyncio.Lock()

    def _decision(self, sha: str) -> bool | None:
        rec = self.index.model.get(sha)
        if not rec or rec.get("label") is None:
            return None
        conf = rec.get("confidence")
        return bool(rec["label"]) and (conf if conf is not None else 1.0) >= self.cfg.threshold

    def counts(self) -> tuple[int, int]:
        """(正例数, 反例数)：/examples 设置过的值优先，否则用 config。"""
        saved = self.cache.get_setting(COUNTS_SETTING) if self.cache else None
        if saved:
            return int(saved[0]), int(saved[1])
        return self.cfg.examples.positive, self.cfg.examples.negative

    def current(self) -> list[Example]:
        """机器人当前正在用的例子（不触发重选）。"""
        return self._examples

    def limit(self) -> int:
        return examples_limit(self.cfg)

    def set_counts(self, positive: int, negative: int) -> None:
        if self.cache is None:
            raise RuntimeError("没有可写的设置存储")
        self.cache.set_setting(COUNTS_SETTING, [positive, negative])
        self.invalidate()

    def invalidate(self) -> None:
        """下次 get() 时立即重选。"""
        self._built_at = float("-inf")

    def available(self) -> tuple[int, int]:
        """人工标注的正例、反例总数。"""
        return tuple(sum(1 for f in self.store.dirs[("human", lab)].iterdir() if f.is_file()) for lab in (True, False))

    def select(self, counts: tuple[int, int] | None = None) -> list[tuple[str, bool]]:
        """返回 (sha, 是否奶龙) 列表，正反例交替排列。"""
        positive, negative = counts or self.counts()
        limit = self.limit()
        if positive + negative > limit:
            # 超出上下文窗口能放下的数量时按比例缩减
            scale = limit / (positive + negative)
            positive, negative = int(positive * scale), int(negative * scale)
        self.index.refresh()
        picked: dict[bool, list[str]] = {}
        for label, want in ((True, positive), (False, negative)):
            files = [f for f in self.store.dirs[("human", label)].iterdir() if f.is_file()]
            human_time = lambda f: self.index.human.get(f.stem, {}).get("time") or ""  # noqa: E731
            # 难例（模型当时判错）在前，各自按人工标注时间从新到旧
            files.sort(key=lambda f: (self._decision(f.stem) not in (None, label), human_time(f)), reverse=True)
            # 选中的集合按 sha 排序，只要集合不变，请求前缀就不变
            picked[label] = sorted(f.stem for f in files[:want])
        pos, neg = picked[True], picked[False]
        out: list[tuple[str, bool]] = []
        for i in range(max(len(pos), len(neg))):
            if i < len(pos):
                out.append((pos[i], True))
            if i < len(neg):
                out.append((neg[i], False))
        return out

    async def get(self, force: bool = False, counts: tuple[int, int] | None = None) -> list[Example]:
        """当前的例子集合。

        force=True 时忽略开关和刷新间隔立即重选；counts 指定本次的正反例数量（网页重测用），
        指定 counts 时返回的集合只给调用方用，不替换机器人正在用的例子。
        """
        ex = self.cfg.examples
        if not ex.enabled and not force:
            return []
        if counts is not None:
            chosen = await asyncio.to_thread(self.select, counts)
            return await self._build(chosen, keep=False)
        async with self._lock:
            version = self.cache.labels_version() if self.cache else None
            if not force:
                never_built = self._built_at == float("-inf")
                # 有数据库时只在标注变化后重选；没有时退化为按间隔重选
                changed = version != self._version if self.cache else True
                if not never_built and (not changed or time.monotonic() - self._built_at < ex.rebuild_seconds):
                    return self._examples
            self._built_at = time.monotonic()
            self._version = version
            chosen = await asyncio.to_thread(self.select)
            if [(e.sha, e.is_nailong) for e in self._examples] == chosen:
                return self._examples
            return await self._build(chosen, keep=True)

    async def _build(self, chosen: list[tuple[str, bool]], keep: bool) -> list[Example]:
        small = replace(self.cfg.frames, max_side=self.cfg.examples.max_side)
        uris: dict[str, str] = {}
        examples = []
        for sha, label in chosen:
            uri = self._uris.get(sha)
            if uri is None:
                path = self.store.find(sha)
                if path is None:
                    continue
                try:
                    uri = await asyncio.to_thread(to_data_uri, path.read_bytes(), small)
                except Exception as e:
                    log.warning("例子图片无法读取 %s: %s", sha[:12], e)
                    continue
            uris[sha] = uri
            examples.append(Example(sha, label, uri))
        if keep:
            self._uris = uris
            self._examples = examples
            self.updated_at = time.time()
            log.info(
                "例子已更新：正例 %d 张，反例 %d 张",
                sum(e.is_nailong for e in examples), sum(not e.is_nailong for e in examples),
            )
        return examples
