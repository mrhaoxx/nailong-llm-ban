"""样本人工标注网页：浏览 nailong_images 下的图片，标注是/否奶龙，并可用不同 prompt 重测。

和机器人共用同一个缓存数据库和样本目录，标注立即对机器人生效，无需重启。
重测只读取图片、调用模型，不改动缓存和样本目录。
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import mimetypes
import re
import secrets
import socket
import threading
import time
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from . import detector as detector_mod
from .cache import VerdictCache
from .gallery import Gallery
from .config import Config, load_config
from .detector import Detector
from .examples import COUNTS_SETTING, examples_limit
from .image import to_data_uri
from .labeling import apply_human_label
from .store import ManifestIndex, SampleStore

log = logging.getLogger("nailong.web")

DIRS = ["model/nailong", "model/not_nailong", "human/nailong", "human/not_nailong"]
# 虚拟分类：inconsistent = 人工标签与模型当时的判定不同；human = 全部人工标注
VIEWS = [*DIRS, "inconsistent", "human"]
SORTS = {"added", "last_seen", "count", "confidence"}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COOKIE = "nailong_token"
PAGE = (Path(__file__).parent / "webui.html").read_text(encoding="utf-8")


class Retester:
    """在后台事件循环里用指定 prompt 重新判定一批图片，结果保存到 save_dir/retests/。"""

    def __init__(self, cfg: Config, store: SampleStore, out_dir: Path):
        self.cfg = cfg
        self.store = store
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.detector: Detector = asyncio.run_coroutine_threadsafe(self._make_detector(), self.loop).result()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    async def _make_detector(self) -> Detector:
        # 重测不读写缓存，给一个内存库占位；store 只用来挑例子
        return Detector(self.cfg, VerdictCache(":memory:"), self.store)

    def running(self) -> dict[str, Any] | None:
        with self.lock:
            return next((j for j in self.jobs.values() if j["status"] == "running"), None)

    def start(
        self, prompt: str, scope: str, entries: list[dict[str, Any]], use_examples: bool, counts: tuple[int, int]
    ) -> dict[str, Any]:
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise RuntimeError("已有重测在运行")
            job_id, n = time.strftime("%Y%m%d-%H%M%S"), 1
            while job_id in self.jobs or (self.out_dir / f"{job_id}.json").exists():
                n += 1
                job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{n}"
            job = {
                "id": job_id,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "prompt": prompt,
                "scope": scope,
                "use_examples": use_examples,
                "example_counts": list(counts),
                "examples": [],
                "threshold": self.cfg.threshold,
                "providers": [f"{p.name}/{p.model}" for p in self.detector.providers],
                "total": len(entries),
                "done": 0,
                "status": "running",
                "results": [None] * len(entries),
                "metrics": None,
            }
            self.jobs[job["id"]] = job
        asyncio.run_coroutine_threadsafe(self._run(job, entries), self.loop)
        return job

    async def _run(self, job: dict[str, Any], entries: list[dict[str, Any]]) -> None:
        sem = asyncio.Semaphore(6)
        examples = []
        if job["use_examples"] and self.detector.examples:
            examples = await self.detector.examples.get(force=True, counts=tuple(job["example_counts"]))
        job["examples"] = [{"sha": e.sha, "label": e.is_nailong} for e in examples]
        example_shas = {e.sha for e in examples}

        async def one(i: int, e: dict[str, Any]) -> None:
            r = {"sha": e["sha"], "human": e["human_label"], "old": e["model_decision"],
                 "new": None, "confidence": None, "reason": "", "model": "", "error": None}
            if e["sha"] in example_shas:
                # 例子本身已经在请求里给出了答案，不再判定，也不计入统计
                r["example"] = True
                job["results"][i] = r
                job["done"] += 1
                return
            async with sem:
                try:
                    data = await asyncio.to_thread(e["path"].read_bytes)
                    uri = await asyncio.to_thread(to_data_uri, data, self.cfg.frames)
                    v = await self.detector.classify(uri, job["prompt"], examples)
                    if v is None:
                        r["error"] = "模型调用失败"
                    else:
                        r.update(new=v.is_nailong and v.confidence >= self.cfg.threshold,
                                 confidence=v.confidence, reason=v.reason, model=v.model)
                except Exception as ex:  # 单张失败不影响整批
                    r["error"] = f"{type(ex).__name__}: {ex}"[:200]
            job["results"][i] = r
            job["done"] += 1

        try:
            await asyncio.gather(*(one(i, e) for i, e in enumerate(entries)))
            job["metrics"] = compute_metrics(job["results"])
            job["status"] = "done"
        except Exception as ex:
            log.exception("重测失败")
            job["status"] = "failed"
            job["error"] = str(ex)
        (self.out_dir / f"{job['id']}.json").write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("重测 %s 完成: %s", job["id"], job["metrics"])

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            if job_id in self.jobs:
                job = dict(self.jobs[job_id])
                job["results"] = [r for r in job["results"] if r is not None]
                return job
        path = self.out_dir / f"{job_id}.json"
        if re.fullmatch(r"[0-9-]+", job_id) and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def history(self) -> list[dict[str, Any]]:
        runs: dict[str, dict[str, Any]] = {}
        for f in self.out_dir.glob("*.json"):
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs[job["id"]] = job
        with self.lock:
            runs.update({k: v for k, v in self.jobs.items() if v["status"] == "running"})
        keep = ("id", "time", "scope", "use_examples", "example_counts", "total", "done", "status", "metrics", "prompt", "providers")
        return [{k: j.get(k) for k in keep} for j in sorted(runs.values(), key=lambda j: j["id"], reverse=True)]


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """新判定（以及模型当时的判定）相对人工标签的准确率。用作例子的图不计入。"""
    results = [r for r in results if not r.get("example")]

    def score(field: str) -> dict[str, Any]:
        rows = [r for r in results if r["human"] is not None and r[field] is not None]
        tp = sum(r["human"] and r[field] for r in rows)
        fn = sum(r["human"] and not r[field] for r in rows)
        fp = sum(not r["human"] and r[field] for r in rows)
        tn = sum(not r["human"] and not r[field] for r in rows)
        return {
            "n": len(rows), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "accuracy": (tp + tn) / len(rows) if rows else None,
        }

    return {
        "new": score("new"),
        "old": score("old"),
        "changed": sum(r["new"] is not None and r["old"] is not None and r["new"] != r["old"] for r in results),
        "errors": sum(r["new"] is None for r in results),
        "positive": sum(bool(r["new"]) for r in results),
    }


class App:
    def __init__(self, cfg: Config, token: str):
        if not cfg.save_dir:
            raise ValueError("配置里 save_dir 为空，没有样本可以标注")
        self.cfg = cfg
        self.token = token
        self.store = SampleStore(cfg.save_dir)
        self.root = self.store.root.resolve()
        self.cache = VerdictCache(cfg.cache_path)
        self.index = ManifestIndex(self.store.manifest)
        self.lock = threading.Lock()
        self.retester = Retester(cfg, self.store, self.root / "retests")
        self.gallery = Gallery(self.cache, self.store)

    def example_counts(self) -> tuple[int, int]:
        """线上机器人使用的例子数量：/examples 设置过的值优先，否则用 config。"""
        saved = self.cache.get_setting(COUNTS_SETTING)
        if saved:
            return int(saved[0]), int(saved[1])
        return self.cfg.examples.positive, self.cfg.examples.negative

    def _decision(self, rec: dict[str, Any]) -> bool | None:
        """模型记录对应的实际处理结果（与机器人一致：是奶龙且置信度达到阈值）。"""
        if not rec or rec.get("label") is None:
            return None
        return bool(rec["label"]) and (rec.get("confidence") if rec.get("confidence") is not None else 1.0) >= self.cfg.threshold

    def entries(self, view: str) -> list[dict[str, Any]]:
        dirs = {"inconsistent": DIRS[2:], "human": DIRS[2:], "all": DIRS,
                "nailong": ["model/nailong", "human/nailong"]}.get(view, [view])
        sightings = self.cache.sightings()
        tags = self.cache.all_image_tags()
        captions = dict(self.cache.captions())
        with self.lock:
            self.index.refresh()
            out = []
            for d in dirs:
                human_label = {"human/nailong": True, "human/not_nailong": False}.get(d)
                for f in (self.root / d).iterdir():
                    if not f.is_file():
                        continue
                    sha = f.stem
                    m = self.index.model.get(sha) or {}
                    decision = self._decision(m)
                    if view == "inconsistent" and (decision is None or decision == human_label):
                        continue
                    s = sightings.get(sha)
                    added = self.index.first.get(sha) or (s.first_seen if s else None) or f.stat().st_mtime
                    h = self.index.human.get(sha)
                    out.append({
                        "sha": sha,
                        "path": f,
                        "category": d,
                        "human_label": human_label,
                        "model_decision": decision,
                        "model": {k: m.get(k) for k in ("model", "confidence", "reason", "label")} if m else None,
                        "human": {k: h.get(k) for k in ("admin", "time")} if h else None,
                        "confidence": m.get("confidence"),
                        "added": added,
                        "last_seen": s.last_seen if s else added,
                        "count": s.count if s else 0,
                        "group_id": m.get("group_id") or (h or {}).get("group_id"),
                        "user_id": m.get("user_id") or (h or {}).get("user_id"),
                        "tags": tags.get(sha, {}),
                        "desc": captions.get(sha, ""),
                    })
        return out

    @staticmethod
    def sort(entries: list[dict[str, Any]], key: str, order: str) -> list[dict[str, Any]]:
        desc = order != "asc"
        # 没有置信度的（例如人工直接附图标注的）始终排在最后
        present = [e for e in entries if e[key] is not None]
        missing = [e for e in entries if e[key] is None]
        present.sort(key=lambda e: (e[key], e["added"]), reverse=desc)
        return present + missing

    @staticmethod
    def public(e: dict[str, Any]) -> dict[str, Any]:
        item = {k: v for k, v in e.items() if k != "path"}
        item["url"] = f"/img/{e['category']}/{e['path'].name}"
        return item

    def stats(self) -> dict[str, int]:
        counts = {d: sum(1 for f in (self.root / d).iterdir() if f.is_file()) for d in DIRS}
        counts["inconsistent"] = len(self.entries("inconsistent"))
        return counts

    def list_images(self, view: str, sort: str, order: str, offset: int, limit: int) -> dict[str, Any]:
        entries = self.sort(self.entries(view), sort, order)
        return {"total": len(entries), "items": [self.public(e) for e in entries[offset : offset + limit]]}

    def search(self, query: str, limit: int) -> dict[str, Any]:
        """和群里 /search 一样：模型直接看图库拼图找图（与机器人共用同一套拼图，前缀缓存互通）。"""
        future = asyncio.run_coroutine_threadsafe(
            self.gallery.search(self.retester.detector, query, limit), self.retester.loop)
        result = future.result(timeout=180)
        entries = {e["sha"]: e for e in self.entries("all")}
        items = [self.public(entries[sha]) for sha in result.shas if sha in entries]
        u = result.usage
        return {"total": len(items), "items": items, "gallery": len(self.gallery.entries()),
                "usage": {"prompt": u.prompt, "cached": u.cached, "completion": u.completion, "seconds": round(u.seconds, 1)}}

    # ---------- 标签 ----------

    def tag_list(self) -> list[dict[str, Any]]:
        stale = set(self.gallery.stale())
        return [{**t, "stale": t["name"] in stale} for t in self.cache.tags()]

    def add_tag(self, sha: str, name: str, note: str | None) -> bool:
        """和群里 /tag 一样只写数据库；不是奶龙的图打标签后进图库，在后台补写描述。"""
        path = self.store.find(sha)
        if path is None:
            return False
        self.cache.upsert_tag(name, note)
        self.cache.set_image_tag(sha, name, "human")
        log.info("网页给 %s 打上标签「%s」", sha[:12], name)
        if sha not in dict(self.cache.captions()):
            asyncio.run_coroutine_threadsafe(self._caption(sha, path), self.retester.loop)
        return True

    async def _caption(self, sha: str, path: Path) -> None:
        try:
            uri = await asyncio.to_thread(to_data_uri, path.read_bytes(), self.cfg.frames)
            if text := await self.retester.detector.describe(uri):
                self.cache.set_caption(sha, text)
        except Exception as e:
            log.warning("补写描述失败 %s: %s", sha[:12], e)

    def tagged(self, name: str) -> dict[str, Any]:
        """某标签下的图。标签过期时先推广一次（一次模型调用），返回用量。"""
        future = asyncio.run_coroutine_threadsafe(self.gallery.refresh(self.retester.detector, [name]), self.retester.loop)
        usage = future.result(timeout=300)
        entries = {e["sha"]: e for e in self.entries("all")}
        items = [self.public(entries[sha]) for sha, _ in self.cache.tagged(name) if sha in entries]
        out: dict[str, Any] = {"total": len(items), "items": items, "note": self.cache.tag_note(name) or ""}
        if usage:
            out["usage"] = {"prompt": usage.prompt, "cached": usage.cached, "completion": usage.completion,
                            "seconds": round(usage.seconds, 1)}
        return out

    def locate(self) -> dict[str, str]:
        return {f.stem: f"/img/{d}/{f.name}" for d in DIRS for f in (self.root / d).iterdir() if f.is_file()}

    def label(self, sha: str, is_nailong: bool) -> str | None:
        with self.lock:
            self.index.refresh()
            if not self.store.has(sha):
                return None
            src = self.index.model.get(sha) or {}
            apply_human_label(
                self.cache, self.store, sha, is_nailong, "webui",
                extra_keys=sorted(self.index.keys.get(sha, ())),
                meta={k: src[k] for k in ("group_id", "user_id", "message_id") if k in src},
            )
            return "human/" + ("nailong" if is_nailong else "not_nailong")


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    def _token_ok(given: str) -> bool:
        return hmac.compare_digest(given.encode(), app.token.encode())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, status: int, body: bytes, ctype: str, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if not ctype.startswith("image/") else "private, max-age=86400")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _authorized(self, query: dict[str, list[str]]) -> bool:
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                given = cookie[_COOKIE].value if _COOKIE in cookie else ""
            except CookieError:
                given = ""
            return _token_ok(given) or _token_ok((query.get("token") or [""])[0])

        def _check(self) -> tuple[str, dict[str, list[str]]] | None:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            token = (query.get("token") or [""])[0]
            if token and _token_ok(token):
                # 用链接里的 token 换 cookie，然后跳回不带 token 的地址
                self._send(
                    HTTPStatus.SEE_OTHER, b"", "text/plain",
                    {"Location": url.path or "/", "Set-Cookie": f"{_COOKIE}={app.token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"},
                )
                return None
            if not self._authorized(query):
                self._send(HTTPStatus.UNAUTHORIZED, "需要带 token 的访问链接，见服务端日志".encode(), "text/plain; charset=utf-8")
                return None
            return url.path, query

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1 << 20:
                raise ValueError("请求太大")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError
            return body

        def do_GET(self) -> None:
            checked = self._check()
            if checked is None:
                return
            path, query = checked
            q = lambda k, d="": (query.get(k) or [d])[0]  # noqa: E731
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/stats":
                self._json(app.stats())
            elif path == "/api/images":
                view, sort, order = q("cat"), q("sort", "added"), q("order", "desc")
                if view not in VIEWS or sort not in SORTS:
                    return self._json({"error": "参数错误"}, 400)
                try:
                    offset = max(0, int(q("offset", "0")))
                    limit = max(1, min(int(q("limit", "60")), 200))
                except ValueError:
                    return self._json({"error": "参数错误"}, 400)
                self._json(app.list_images(view, sort, order, offset, limit))
            elif path == "/api/search":
                text = q("q").strip()
                if not text:
                    return self._json({"error": "参数错误"}, 400)
                try:
                    limit = max(1, min(int(q("limit", "30")), 60))
                except ValueError:
                    return self._json({"error": "参数错误"}, 400)
                try:
                    self._json(app.search(text, limit))
                except Exception as e:
                    log.warning("搜索失败: %s", e)
                    self._json({"error": f"搜索失败：{e}"[:200]}, 502)
            elif path == "/api/prompt":
                self._json({
                    "prompt": detector_mod.SYSTEM_PROMPT,
                    "examples": {"enabled": app.cfg.examples.enabled, "positive": app.example_counts()[0],
                                 "negative": app.example_counts()[1], "max": examples_limit(app.cfg)},
                    "threshold": app.cfg.threshold,
                    "providers": [f"{p.name}/{p.model}" for p in app.retester.detector.providers],
                })
            elif path == "/api/retests":
                self._json(app.retester.history())
            elif path == "/api/retest":
                job = app.retester.get(q("id"))
                if job is None:
                    return self._json({"error": "没有这次重测"}, 404)
                urls = app.locate()
                for r in job["results"] + job.get("examples", []):
                    r["url"] = urls.get(r["sha"])
                self._json(job)
            elif path == "/api/tags":
                self._json(app.tag_list())
            elif path == "/api/tagged":
                name = q("tag").strip()
                if app.cache.tag_note(name) is None:
                    return self._json({"error": "没有这个标签"}, 404)
                try:
                    self._json(app.tagged(name))
                except Exception as e:
                    log.warning("标签推广失败: %s", e)
                    self._json({"error": f"标签推广失败：{e}"[:200]}, 502)
            elif path.startswith("/img/"):
                target = (app.root / unquote(path[len("/img/") :])).resolve()
                if not target.is_relative_to(app.root) or target.parent.relative_to(app.root).as_posix() not in DIRS or not target.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self._send(200, target.read_bytes(), ctype)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            checked = self._check()
            if checked is None:
                return
            path, _ = checked
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError):
                return self._json({"error": "参数错误"}, 400)
            if path == "/api/label":
                sha, is_nailong = body.get("sha"), body.get("label")
                if not isinstance(sha, str) or not _SHA_RE.match(sha) or not isinstance(is_nailong, bool):
                    return self._json({"error": "参数错误"}, 400)
                category = app.label(sha, is_nailong)
                if category is None:
                    return self._json({"error": "图片不存在"}, 404)
                log.info("网页标注 %s -> %s", sha[:12], category)
                self._json({"ok": True, "category": category})
            elif path in ("/api/tag", "/api/untag", "/api/tagnote", "/api/tagdelete"):
                sha, name, note = body.get("sha"), body.get("tag"), body.get("note")
                name = name.strip() if isinstance(name, str) else ""
                if not name or len(name) > 30 or any(c.isspace() for c in name) or (note is not None and not isinstance(note, str)):
                    return self._json({"error": "标签名不能为空、不能有空格、最多 30 字"}, 400)
                if path in ("/api/tag", "/api/untag") and (not isinstance(sha, str) or not _SHA_RE.match(sha)):
                    return self._json({"error": "参数错误"}, 400)
                if path == "/api/tag":
                    if not app.add_tag(sha, name, note.strip() if note and note.strip() else None):
                        return self._json({"error": "图片不存在"}, 404)
                elif app.cache.tag_note(name) is None:
                    return self._json({"error": "没有这个标签"}, 404)
                elif path == "/api/untag":
                    # 记成 rejected：以后推广时也不会再自动加回来
                    app.cache.set_image_tag(sha, name, "rejected")
                elif path == "/api/tagnote":
                    app.cache.upsert_tag(name, (note or "").strip())
                else:
                    app.cache.delete_tag(name)
                    app.cache.set_setting(f"tagsig.{name}", None)
                    log.info("网页删除标签「%s」", name)
                self._json({"ok": True, "tags": app.cache.image_tags(sha) if isinstance(sha, str) else None})
            elif path == "/api/retest":
                prompt, scope = body.get("prompt"), body.get("scope")
                sort, order, limit = body.get("sort", "added"), body.get("order", "desc"), body.get("limit", 100)
                use_examples = body.get("examples", False)
                default_pos, default_neg = app.example_counts()
                positive, negative = body.get("positive", default_pos), body.get("negative", default_neg)
                if (not isinstance(prompt, str) or not prompt.strip() or scope not in VIEWS or sort not in SORTS
                        or not isinstance(limit, int) or limit < 1 or not isinstance(use_examples, bool)
                        or not all(isinstance(x, int) and x >= 0 for x in (positive, negative))
                        or positive + negative > examples_limit(app.cfg)):
                    return self._json({"error": "参数错误"}, 400)
                entries = app.sort(app.entries(scope), sort, order)[:limit]
                if not entries:
                    return self._json({"error": "这个范围里没有图片"}, 400)
                try:
                    job = app.retester.start(prompt, scope, entries, use_examples, (positive, negative))
                except RuntimeError as e:
                    return self._json({"error": str(e)}, 409)
                log.info("开始重测 %s：%s 共 %d 张", job["id"], scope, len(entries))
                self._json({"id": job["id"], "total": job["total"]})
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="奶龙样本标注网页")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = cfg.web.token or secrets.token_urlsafe(16)
    app = App(cfg, token)
    server = ThreadingHTTPServer((cfg.web.host, cfg.web.port), make_handler(app))
    host = socket.gethostname() if cfg.web.host in ("0.0.0.0", "::") else cfg.web.host
    log.info("标注网页已启动: http://%s:%d/?token=%s", host, cfg.web.port, token)
    if not cfg.web.token:
        log.info("token 为本次启动随机生成，重启后会变；想固定可在 config.yaml 的 web.token 里设置")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
