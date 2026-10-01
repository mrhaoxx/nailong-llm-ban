from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

_EXT = {"GIF": "gif", "PNG": "png", "JPEG": "jpg", "WEBP": "webp", "BMP": "bmp"}


def _ext(data: bytes) -> str:
    # QQ 的文件名后缀经常和实际格式不符（例如 .jpg 其实是 GIF），按内容判断
    try:
        return _EXT.get(Image.open(io.BytesIO(data)).format or "", "bin")
    except Exception:
        return "bin"


class SampleStore:
    """样本集。每张图按内容哈希只存一份，放在对应标注的目录下：

        human/nailong/       管理员标注为奶龙（可信标签）
        human/not_nailong/   管理员标注为非奶龙（可信标签）
        model/nailong/       模型判为奶龙（未经人工核实）
        model/not_nailong/   模型判为非奶龙（未经人工核实）
        labels.jsonl         每一次判定/标注的记录，只追加不修改

    人工标注优先：图一旦有人工标签，模型判定不会再移动它；管理员重新标注则以最新一次为准。
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.dirs = {
            (src, lab): self.root / src / ("nailong" if lab else "not_nailong")
            for src in ("human", "model")
            for lab in (True, False)
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "labels.jsonl"
        self._lock = threading.Lock()

    def _existing(self, digest: str, source: str) -> list[Path]:
        return [
            f
            for (src, _), d in self.dirs.items()
            if src == source
            for f in d.glob(f"{digest}.*")
        ]

    def find(self, digest: str) -> Path | None:
        found = self._existing(digest, "human") + self._existing(digest, "model")
        return found[0] if found else None

    def has(self, digest: str) -> bool:
        return self.find(digest) is not None

    def save(
        self,
        data: bytes | None,
        digest: str | None,
        is_nailong: bool,
        source: str,
        meta: dict[str, Any],
    ) -> Path | None:
        with self._lock:
            path: Path | None = None
            if data is not None and digest is not None:
                if source == "model" and self._existing(digest, "human"):
                    # 已有人工标签，不让模型结果覆盖
                    path = self._existing(digest, "human")[0]
                else:
                    for stale in self._existing(digest, "human") + self._existing(digest, "model"):
                        stale.unlink()
                    path = self.dirs[(source, is_nailong)] / f"{digest}.{_ext(data)}"
                    path.write_bytes(data)
                    log.info("样本已保存 %s", path.relative_to(self.root))

            self._append(digest, path, is_nailong, source, meta)
            return path

    def relabel(self, digest: str, is_nailong: bool, meta: dict[str, Any]) -> Path | None:
        """对已保存的图做人工标注：移动到 human/ 对应目录并追加记录。图不存在时返回 None。"""
        with self._lock:
            existing = self._existing(digest, "human") + self._existing(digest, "model")
            if not existing:
                return None
            src = existing[0]
            path = self.dirs[("human", is_nailong)] / src.name
            if src != path:
                src.rename(path)
            for stale in existing[1:]:
                if stale != path:
                    stale.unlink()
            log.info("样本已人工标注 %s", path.relative_to(self.root))
            self._append(digest, path, is_nailong, "human", meta)
            return path

    def _append(self, digest: str | None, path: Path | None, is_nailong: bool, source: str, meta: dict[str, Any]) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sha256": digest,
            "file": str(path.relative_to(self.root)) if path else None,
            "label": is_nailong,
            "source": source,
            **meta,
        }
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        return None


class ManifestIndex:
    """增量读取 labels.jsonl（只追加），按 sha256 汇总最近的模型记录、人工记录、缓存键和首次记录时间。"""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.model: dict[str, dict[str, Any]] = {}
        self.human: dict[str, dict[str, Any]] = {}
        self.keys: dict[str, set[str]] = {}
        self.first: dict[str, float] = {}

    def refresh(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break  # 写到一半的行，下次再读
                self.offset += len(line)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sha = rec.get("sha256")
                if not sha:
                    continue
                (self.human if rec.get("source") == "human" else self.model)[sha] = rec
                self.keys.setdefault(sha, set()).update(rec.get("keys") or [])
                if (t := _ts(rec.get("time"))) is not None and t < self.first.get(sha, float("inf")):
                    self.first[sha] = t
