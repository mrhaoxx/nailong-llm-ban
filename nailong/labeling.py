from __future__ import annotations

from typing import Any

from .cache import Verdict, VerdictCache
from .store import SampleStore


def apply_human_label(
    cache: VerdictCache,
    store: SampleStore | None,
    sha: str,
    is_nailong: bool,
    admin: int | str,
    extra_keys: list[str] | tuple[str, ...] = (),
    meta: dict[str, Any] | None = None,
) -> bool:
    """对一张已知内容哈希的图做人工标注：覆盖缓存判定，并把样本移到 human/。

    先写缓存再移动文件：即使移动失败，机器人也已经按人工标签处理。返回样本文件是否存在。
    """
    content_key = "sha256:" + sha
    keys = [content_key, *sorted(set(cache.keys_for(sha)) | set(extra_keys))]
    prev = cache.get(content_key)
    cache.put(keys, Verdict(is_nailong, 1.0, "人工标注", f"admin:{admin}"))
    cache.bump_labels()
    if store is None or not store.has(sha):
        return False
    store.relabel(sha, is_nailong, {
        "admin": admin,
        "previous": {"label": prev.is_nailong, "model": prev.model, "confidence": prev.confidence} if prev else None,
        "keys": keys[1:],
        **(meta or {}),
    })
    return True
