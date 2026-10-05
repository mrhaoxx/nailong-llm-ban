"""图库：给模型看的固定拼图，供以图/文字搜图、标签推广、解释图片共用。

为了让 DeepSeek 的前缀缓存一直命中：
- 每张图进图库时分配永久编号，按编号排进 2x2 拼图（512px 一张，按 token 计费的最低档），只在末尾追加；
- 排满的拼图生成一次就存盘，之后直接读文件，字节永远不变；只有最后一张没排满的拼图会随新图变化；
- 图被改判或删掉时不挪位置（否则后面所有拼图都会变），在结果里过滤；
- 拼图不做随机预处理（会破坏缓存）。小图缩到 250px 再排版，针对原图像素对齐的对抗扰动在这一步就失效了。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

from .cache import VerdictCache
from .store import ManifestIndex, SampleStore
from .wall import make_thumb

if TYPE_CHECKING:
    from .detector import Detector, Usage

log = logging.getLogger(__name__)

COLS = 2
PER_GRID = COLS * COLS
SIDE = 512
GAP = 4
CELL = (SIDE - GAP * (COLS + 1)) // COLS
BG = (40, 40, 40)


@dataclass
class SearchResult:
    shas: list[str]
    usage: Usage


class Gallery:
    def __init__(self, cache: VerdictCache, store: SampleStore):
        self.cache = cache
        self.store = store
        self.dir = store.root / "gallery"
        self.thumbs = store.root / "thumbs"
        self.index = ManifestIndex(store.manifest)
        self._uris: dict[int, str] = {}           # 已排满拼图的 data URI（读自磁盘文件）
        self._tail: tuple[int, int, str] | None = None   # (拼图序号, 图数, data URI)
        self._lock = threading.Lock()
        self._font = ImageFont.load_default(size=max(14, CELL // 7))

    # ---------- 图库内容 ----------

    def scope(self) -> set[str]:
        """应该在图库里的图：模型判为奶龙、人工标为奶龙，以及打过人工标签的图。"""
        dirs = (self.store.dirs[("model", True)], self.store.dirs[("human", True)])
        shas = {f.stem for d in dirs for f in d.iterdir() if f.is_file()}
        return shas | {sha for sha, tags in self.cache.all_image_tags().items() if "human" in tags.values()}

    def sync(self) -> int:
        """把还没进图库的图追加到末尾，按首次出现时间排序。返回新增数量。"""
        with self._lock:
            have = {sha for _, sha in self.cache.gallery_list()}
            new = [s for s in self.scope() - have if self.store.find(s) is not None]
            if not new:
                return 0
            self.index.refresh()
            new.sort(key=lambda s: (self.index.first.get(s) or float("inf"), s))
            added = self.cache.gallery_add(new)
            log.info("图库新增 %d 张", added)
            return added

    def active(self) -> set[str]:
        """结果里允许出现的图（后来被改判为非奶龙、又没有人工标签的图仍占着位置，但不返回）。"""
        return self.scope()

    def entries(self) -> list[tuple[int, str]]:
        return self.cache.gallery_list()

    def id_of(self) -> dict[str, int]:
        return {sha: gid for gid, sha in self.entries()}

    # ---------- 图库任务 ----------

    async def _ask(self, detector: Detector, task: str, image: str | None, kind: str) -> tuple[Any, Usage, dict[int, str]]:
        await asyncio.to_thread(self.sync)
        entries = self.entries()
        if not entries:
            raise RuntimeError("图库是空的")
        parts = await asyncio.to_thread(self.parts)
        value, usage = await detector.ask_gallery(parts, task, image, kind)
        return value, usage, dict(entries)

    async def _ranked(self, value: Any, by_id: dict[int, str], limit: int, exclude: str | None = None) -> list[str]:
        if not isinstance(value, list):
            raise RuntimeError(f"回复不是编号数组: {str(value)[:80]}")
        active = await asyncio.to_thread(self.active)
        out: list[str] = []
        for i in value:
            sha = by_id.get(i) if isinstance(i, int) else None
            if sha and sha in active and sha != exclude and sha not in out:
                out.append(sha)
        return out[:limit]

    def tag_hints(self, per_tag: int = 200) -> str:
        """已有标签及其图的编号，附在搜索任务里。只在请求末尾，标签怎么变都不影响图库前缀的缓存。"""
        ids = self.id_of()
        lines = []
        for t in self.cache.tags():
            nums = sorted(ids[s] for s, _ in self.cache.tagged(t["name"]) if s in ids)[:per_tag]
            if nums:
                note = f"（{t['note']}）" if t["note"] else ""
                lines.append(f"- 「{t['name']}」{note}：" + " ".join(f"#{i}" for i in nums))
        if not lines:
            return ""
        return ("已有标签（人工标注或自动推广，供参考；查询提到标签名或说明里的内容时，优先考虑标签里的图，"
                "但仍要看画面是否符合）：\n" + "\n".join(lines) + "\n")

    async def search(self, detector: Detector, query: str, limit: int) -> SearchResult:
        await asyncio.to_thread(self.sync)
        # 标签放在查询前面：连续搜索时标签这段也能命中缓存，只有查询本身是新的
        task = (f"任务：按文字搜图。\n{self.tag_hints()}查询：{query}\n"
                f"找出画面确实符合查询的图，按相关度从高到低，最多 {limit} 张。宁缺毋滥：只沾一点边的不要，"
                "不必凑满，没有符合的就输出 []。"
                "只输出编号数组，例如 [12, 3, 40]。")
        value, usage, by_id = await self._ask(detector, task, None, "search")
        return SearchResult(await self._ranked(value, by_id, limit), usage)

    async def similar(self, detector: Detector, sha: str | None, image: str | None, limit: int) -> SearchResult:
        """以图搜图。图已在图库里时只用编号指代（不必再传图）；否则把图附在末尾。"""
        await asyncio.to_thread(self.sync)
        gid = self.id_of().get(sha) if sha else None
        target = f"图库里的 #{gid}" if gid else "查询图"
        task = (f"任务：以图搜图。找出图库中和{target}最相似的图：同一张图的其它版本、同一个梗或模板、"
                f"同一角色做相同的动作或表情。只是同一个角色、画风相近不算相似。按相似度从高到低，最多 {limit} 张"
                + (f"，不要包含 #{gid} 本身" if gid else "")
                + "。宁缺毋滥，不必凑满，没有相似的就输出 []。只输出编号数组，例如 [12, 3, 40]。")
        value, usage, by_id = await self._ask(detector, task, None if gid else image, "similar")
        return SearchResult(await self._ranked(value, by_id, limit, exclude=sha), usage)

    # ---------- 标签推广 ----------

    def _signature(self, tag: dict[str, Any], ids: dict[str, int], count: int) -> str:
        human = sorted(ids[s] for s, src in self.cache.tagged(tag["name"]) if src == "human" and s in ids)
        rejected = sorted(ids[s] for s in self.cache.rejected(tag["name"]) if s in ids)
        raw = json.dumps([tag["note"], human, rejected, count], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def stale(self, names: list[str] | None = None) -> list[str]:
        """例子、说明、排除或图库变化后还没重新推广的标签。"""
        ids = self.id_of()
        out = []
        for tag in self.cache.tags():
            if names is not None and tag["name"] not in names:
                continue
            if self.cache.get_setting(f"tagsig.{tag['name']}") != self._signature(tag, ids, len(ids)):
                out.append(tag["name"])
        return out

    async def refresh(self, detector: Detector, names: list[str] | None = None) -> Usage | None:
        """把过期的标签一次性重新推广（一次模型调用）。没有过期的返回 None。

        打标签本身只写数据库；推广推迟到有人查看标签或解释图片时才做，连续打很多标签也只算一次。"""
        await asyncio.to_thread(self.sync)
        stale = await asyncio.to_thread(self.stale, names)
        if not stale:
            return None
        ids = self.id_of()
        tags = {t["name"]: t for t in self.cache.tags()}
        lines = []
        for name in stale:
            human = [ids[s] for s, src in self.cache.tagged(name) if src == "human" and s in ids]
            rejected = sorted(ids[s] for s in self.cache.rejected(name) if s in ids)
            line = f"- 「{name}」：{tags[name]['note'] or '（无说明，按例子归纳）'}；例子 " + (" ".join(f"#{i}" for i in human) or "无")
            if rejected:
                line += "；排除 " + " ".join(f"#{i}" for i in rejected)
            lines.append(line)
        task = ("任务：标签推广。每个标签有说明和人工给的例子编号（一定属于该标签），可能还有人工排除的编号（一定不属于）。"
                "对每个标签，参照说明和例子的画面，找出图库中所有属于它的图（宁缺毋滥，拿不准的不要）。\n标签：\n"
                + "\n".join(lines)
                + '\n输出一个 JSON 对象，键是标签名，值是编号数组，例如 {"标签名": [3, 17, 40]}。')
        value, usage, by_id = await self._ask(detector, task, None, "tags")
        if not isinstance(value, dict):
            raise RuntimeError(f"回复不是 JSON 对象: {str(value)[:80]}")
        for name in stale:
            got = value.get(name)
            if not isinstance(got, list):
                continue
            rejected = self.cache.rejected(name)
            shas = [by_id[i] for i in got if isinstance(i, int) and i in by_id and by_id[i] not in rejected]
            self.cache.replace_auto_tags(name, shas)
            self.cache.set_setting(f"tagsig.{name}", self._signature(tags[name], ids, len(ids)))
            log.info("标签「%s」推广到 %d 张", name, len(shas))
        return usage

    async def classify_tags(self, detector: Detector, image: str) -> tuple[list[str], Usage]:
        """不在图库里的图：直接问它属于哪些标签（不写数据库）。"""
        ids = self.id_of()
        lines = []
        for t in self.cache.tags():
            human = [ids[s] for s, src in self.cache.tagged(t["name"]) if src == "human" and s in ids]
            lines.append(f"- 「{t['name']}」：{t['note'] or '（无说明）'}；例子 " + (" ".join(f"#{i}" for i in human) or "无"))
        task = ("任务：判断查询图属于下面哪些标签（可以多个，也可以一个都不属于；拿不准的不要）。标签：\n"
                + "\n".join(lines) + '\n输出 JSON 对象，例如 {"tags": ["标签名"]}。')
        value, usage, _ = await self._ask(detector, task, image, "explain")
        names = {t["name"] for t in self.cache.tags()}
        got = value.get("tags") if isinstance(value, dict) else None
        return [n for n in (got or []) if n in names], usage

    # ---------- 拼图 ----------

    def parts(self) -> list[dict]:
        """给模型的图库内容：一段说明文字 + 全部拼图。前面的部分字节不变，便于命中缓存。"""
        entries = self.entries()
        uris = [self._grid(k, entries[k * PER_GRID:(k + 1) * PER_GRID]) for k in range((len(entries) + PER_GRID - 1) // PER_GRID)]
        # 开头的文字不能带图片数量之类会变的内容，否则新增一张图整个前缀就失效了。
        # 描述清单放在全部拼图之后、按编号排列：新图的描述只追加在末尾；描述写入后不再修改（见 set_caption）
        captions = dict(self.cache.captions())
        listing = "\n".join(f"{gid}. {captions[sha]}" for gid, sha in entries if sha in captions)
        return [{"type": "text", "text": "图库："}] + \
               [{"type": "image_url", "image_url": {"url": u}} for u in uris] + \
               [{"type": "text", "text": "图片描述：\n" + listing}]

    def missing_captions(self) -> list[str]:
        """图库里还没有描述的图。"""
        captions = dict(self.cache.captions())
        return [sha for _, sha in self.entries() if sha not in captions]

    def _grid(self, k: int, chunk: list[tuple[int, str]]) -> str:
        if len(chunk) < PER_GRID:
            # 最后一张没排满：只在图数变化时重新生成
            if self._tail and self._tail[:2] == (k, len(chunk)):
                return self._tail[2]
            uri = _to_uri(self._render(chunk))
            self._tail = (k, len(chunk), uri)
            return uri
        if k not in self._uris:
            path = self.dir / f"{k:05d}.jpg"
            if not path.exists():
                self.dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(self._render(chunk))
                tmp.replace(path)
            self._uris[k] = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
        return self._uris[k]

    def _render(self, chunk: list[tuple[int, str]]) -> bytes:
        g = Image.new("RGB", (SIDE, SIDE), BG)
        d = ImageDraw.Draw(g)
        for i, (gid, sha) in enumerate(chunk):
            x = GAP + (i % COLS) * (CELL + GAP)
            y = GAP + (i // COLS) * (CELL + GAP)
            d.rectangle((x, y, x + CELL - 1, y + CELL - 1), fill=(255, 255, 255))
            path = self.store.find(sha)
            try:
                im = Image.open(make_thumb(path, self.thumbs)) if path else None
                if im is not None:
                    im.seek(0)
                    im = im.convert("RGB")
                    im.thumbnail((CELL, CELL))
                    g.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
            except Exception as e:
                log.warning("图库拼图读取失败 %s: %s", sha[:12], e)
            label = str(gid)
            box = d.textbbox((0, 0), label, font=self._font)
            d.rectangle((x, y, x + box[2] - box[0] + 8, y + box[3] - box[1] + 8), fill=(0, 0, 0))
            d.text((x + 4 - box[0], y + 4 - box[1]), label, fill=(255, 255, 255), font=self._font)
        buf = io.BytesIO()
        g.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


def _to_uri(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()
