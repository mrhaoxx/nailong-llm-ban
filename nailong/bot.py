from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .cache import Action, VerdictCache
from .config import Config, load_config
from .detector import Detector, ImageRef, Usage
from .examples import TOKENS_PER_IMAGE
from .image import BADGE_NO, BADGE_YES, make_collage, mface_url, sha256, to_data_uri
from .labeling import apply_human_label
from .onebot import OneBot, OneBotError
from .periods import USAGE as WINDOW_USAGE, Window, parse_window
from .stats import compute as compute_stats, render as render_stats
from .wall import render as render_wall
from .store import SampleStore

log = logging.getLogger("nailong")

# 管理员指令：是奶龙 / 否奶龙（也接受 不是奶龙，可带 / 或 # 前缀）
_COMMAND_RE = re.compile(r"^[/#]?(是|否|不是)奶龙$")
# /rewind [数量]：列出本群最近处理过的奶龙图，按编号改判为非奶龙
_REWIND_RE = re.compile(r"^[/#]rewind(?:\s+(\d+))?$", re.IGNORECASE)
_IDS_RE = re.compile(r"^[\d\s,，、]+$")
# /examples 查看当前例子；/examples N 或 /examples 正例数 反例数 调整数量
_EXAMPLES_RE = re.compile(r"^[/#]examples(?:\s+(\d+)(?:\s+(\d+))?)?$", re.IGNORECASE)
# /test + 图片：用线上设置检测，不处理、不缓存；/status [条数]：最近调用的上下文用量
_TEST_RE = re.compile(r"^[/#]test$", re.IGNORECASE)
_STATUS_RE = re.compile(r"^[/#]status(?:\s+(\d+))?$", re.IGNORECASE)
_FORGET_RE = re.compile(r"^[/#]forget$", re.IGNORECASE)
_STATS_RE = re.compile(r"^[/#]stats(?:\s+(.+))?$", re.IGNORECASE)
_HELP_RE = re.compile(r"^[/#](help|帮助)$", re.IGNORECASE)
_WALL_RE = re.compile(r"^[/#]wall(?:\s+(.+))?$", re.IGNORECASE)
TEST_MAX_IMAGES = 4
REPORT_SECONDS = 120
STATS_SECONDS = 300

HELP_TEXT = """奶龙检测器 · 管理员指令
/test ＋图片（或回复带图消息）：用线上设置检测一次，只报告不处理
/status [条数]：最近几次模型调用的上下文用量和缓存命中
/stats [时间范围]：统计卡片（检测、撤回、排行、模型被人工纠正的漏判和误判），私聊时统计所有群
/forget ＋图片（或回复带图消息）：清除这张图的模型判定缓存，下次出现重新判定
/wall [时间范围]：把这段时间被撤回的奶龙拼成一张大图（有动图时是动态 GIF），私聊时包含所有群
/examples：查看当前例子；/examples 8 或 /examples 10 6：调整正反例数量（群聊）
/rewind [数量]：列出最近处理过的奶龙图，回复编号改判为非奶龙（群聊）
是奶龙 / 否奶龙 ＋图片（或回复带图消息）：人工标注
/help：显示本说明
时间范围：today（默认）、yesterday、all、30m、3h、7d、2w、1y、09-01、09-01~09-15
指令和回复会在一段时间后自动撤回。"""
EXAMPLES_SHOW_SECONDS = 120
EXAMPLES_SET_SECONDS = 30
REWIND_DEFAULT = 9
REWIND_MAX = 25
REWIND_TIMEOUT = 120
REWIND_LINGER = 5


@dataclass
class RewindSession:
    group_id: int
    admin_id: int
    items: dict[int, Action]
    # 交互结束后要撤回的消息：管理员指令、拼图、回复、确认
    messages: list[int] = field(default_factory=list)
    timer: asyncio.Task[None] | None = None


class NailongBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache = VerdictCache(cfg.cache_path)
        self.store = SampleStore(cfg.save_dir) if cfg.save_dir else None
        self.detector = Detector(cfg, self.cache, self.store)
        self.onebot = OneBot(cfg.onebot, self.on_event)
        self.rewinds: dict[tuple[int, int], RewindSession] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        if not cfg.groups:
            log.warning("未配置 groups，不会处理任何群消息")

    async def run(self) -> None:
        try:
            await self.onebot.run()
        finally:
            await self.detector.close()
            self.cache.close()

    def extract_images(self, message: Any, context: dict[str, Any] | None = None) -> list[ImageRef]:
        if not isinstance(message, list):
            # 字符串（CQ 码）格式不解析，请在 NapCat 里把消息格式设为 array
            return []
        refs: list[ImageRef] = []
        for seg in message:
            data = seg.get("data") or {}
            if seg.get("type") == "image":
                file = data.get("file") or ""
                keys = image_keys(file)
                refs.append(ImageRef(keys=keys, url=data.get("url"), resolve=self._resolver(file), context=context or {}))
            elif seg.get("type") == "mface" and data.get("emoji_id"):
                emoji_id = str(data["emoji_id"])
                refs.append(
                    ImageRef(keys=[f"mface:{emoji_id}"], url=data.get("url") or mface_url(emoji_id), context=context or {})
                )
        return refs

    def _resolver(self, file: str):
        async def resolve() -> str | None:
            if not file:
                return None
            try:
                info = await self.onebot.call("get_image", file=file)
            except OneBotError as e:
                log.warning("get_image 失败: %s", e)
                return None
            return info.get("url") or info.get("file")

        return resolve

    async def on_event(self, event: dict[str, Any]) -> None:
        if event.get("post_type") != "message":
            return
        message_type = event.get("message_type")
        if message_type == "group" and event.get("group_id") not in self.cfg.groups:
            return
        if message_type not in ("group", "private"):
            return
        user_id = event.get("user_id")

        if user_id in self.cfg.admins:
            if await self.handle_admin_command(event):
                return
            cmd = parse_command(event.get("message"))
            if cmd is not None:
                await self.handle_label(event, cmd)
            # 管理员的消息不做检测
            return
        if message_type != "group" or user_id == event.get("self_id"):
            return
        if self.cfg.skip_admins and (event.get("sender") or {}).get("role") in ("admin", "owner"):
            return

        refs = self.extract_images(event.get("message"), message_context(event))
        if not refs:
            return

        verdicts = await asyncio.gather(*(self.detector.check(r) for r in refs))
        hits = [(r, v) for r, v in zip(refs, verdicts) if v and v.is_nailong and v.confidence >= self.cfg.threshold]
        if not hits:
            return

        group_id = event["group_id"]
        hit = hits[0][1]
        log.info(
            "检测到奶龙 群=%s 用户=%s 消息=%s 置信度=%.2f (%s)",
            group_id, user_id, event.get("message_id"), hit.confidence, hit.model,
        )
        await self.punish(group_id, user_id, event["message_id"], [r for r, _ in hits])

    async def collect_images(self, event: dict[str, Any]) -> tuple[list[ImageRef], dict[str, Any] | None, str | None]:
        """指令消息里附带的图，加上被回复消息里的图。返回 (图片, 被回复的消息, 被回复消息 id)。"""
        message = event.get("message")
        refs = self.extract_images(message, message_context(event))
        target: dict[str, Any] | None = None
        reply_id = next(
            (seg["data"]["id"] for seg in message if seg.get("type") == "reply"),
            None,
        ) if isinstance(message, list) else None
        if reply_id is not None:
            try:
                target = await self.onebot.call("get_msg", message_id=int(reply_id))
            except (OneBotError, ValueError) as e:
                log.warning("获取被回复的消息失败: %s", e)
            if target:
                refs += self.extract_images(target.get("message"), message_context(target))
        return refs, target, reply_id

    async def handle_label(self, event: dict[str, Any], is_nailong: bool) -> None:
        """管理员回复一条带图消息，或在指令里直接附图，来纠正判定。"""
        refs, target, reply_id = await self.collect_images(event)
        if not refs:
            await self.reply(event, "没找到图片：请回复一条带图片的消息，或在指令后附上图片")
            return

        results = await asyncio.gather(*(self.detector.label(r, is_nailong, event["user_id"]) for r in refs))
        ok = sum(results)
        await self.reply(event, f"已将 {ok} 张图片标记为{'奶龙' if is_nailong else '非奶龙'}")

        # 标记为奶龙时，顺手处理被回复的那条群消息
        if is_nailong and target and event.get("message_type") == "group":
            target_user = target.get("user_id") or (target.get("sender") or {}).get("user_id")
            if target_user and target_user not in self.cfg.admins and target_user != event.get("self_id"):
                target_refs = self.extract_images(target.get("message"))
                await self.punish(event["group_id"], target_user, int(reply_id), target_refs)

    async def reply(self, event: dict[str, Any], text: str, extra: list[dict[str, Any]] | None = None) -> int | None:
        """回复一条消息，返回机器人这条消息的 id（发送失败为 None）。"""
        message = [
            {"type": "reply", "data": {"id": str(event["message_id"])}},
            *(extra or []),
            {"type": "text", "data": {"text": text}},
        ]
        try:
            if event.get("message_type") == "group":
                data = await self.onebot.call("send_group_msg", group_id=event["group_id"], message=message)
            else:
                data = await self.onebot.call("send_private_msg", user_id=event["user_id"], message=message)
            return (data or {}).get("message_id")
        except OneBotError as e:
            log.warning("回复失败: %s", e)
            return None

    async def punish(self, group_id: int, user_id: int, message_id: int, refs: list[ImageRef] | None = None) -> None:
        act = self.cfg.action
        recalled = banned = False
        if act.recall:
            try:
                await self.onebot.call("delete_msg", message_id=message_id)
                recalled = True
            except OneBotError as e:
                log.warning("撤回失败（机器人是否为管理员？）: %s", e)
        if act.ban_seconds > 0:
            try:
                await self.onebot.call("set_group_ban", group_id=group_id, user_id=user_id, duration=act.ban_seconds)
                banned = True
            except OneBotError as e:
                log.warning("禁言失败: %s", e)
        # 记下触发处理的每张图，供 /rewind 改判
        for sha in dict.fromkeys(s for r in refs or [] if (s := self.cache.sha_for(*r.keys))):
            self.cache.log_action(group_id, user_id, message_id, sha, recalled, banned)
        if act.notice:
            try:
                await self.onebot.call(
                    "send_group_msg",
                    group_id=group_id,
                    message=[
                        {"type": "at", "data": {"qq": str(user_id)}},
                        {"type": "text", "data": {"text": " " + act.notice}},
                    ],
                )
            except OneBotError as e:
                log.warning("发送提示失败: %s", e)


    async def handle_admin_command(self, event: dict[str, Any]) -> bool:
        """管理员指令。/test、/status 群聊私聊都可用；/examples、/rewind 只在群里。返回 True 表示消息已被消费。"""
        text = plain_text(event.get("message"))
        if _HELP_RE.match(text):
            await self.say(event, HELP_TEXT, REPORT_SECONDS)
            return True
        if _TEST_RE.match(text):
            await self.test_command(event)
            return True
        if _FORGET_RE.match(text):
            await self.forget_command(event)
            return True
        for regex, handler in ((_STATS_RE, self.stats_command), (_WALL_RE, self.wall_command)):
            if m := regex.match(text):
                window = parse_window(m.group(1), time.time())
                if window is None:
                    await self.say(event, f"看不懂时间范围「{m.group(1)}」。{WINDOW_USAGE}", REPORT_SECONDS)
                else:
                    await handler(event, window)
                return True
        if m := _STATUS_RE.match(text):
            await self.status_command(event, int(m.group(1) or 5))
            return True
        if event.get("message_type") != "group":
            return False
        if m := _EXAMPLES_RE.match(text):
            await self.examples_command(event, m)
            return True
        return await self.handle_rewind(event, text)

    async def say(self, event: dict[str, Any], text: str, seconds: float, extra: list[dict[str, Any]] | None = None) -> None:
        """回复后 seconds 秒连同指令一起撤回，避免刷屏。"""
        reply_id = await self.reply(event, text, extra)
        self._spawn(self._recall_later([i for i in (event["message_id"], reply_id) if i is not None], seconds))

    # ---------- /test ----------

    async def test_command(self, event: dict[str, Any]) -> None:
        refs, _, _ = await self.collect_images(event)
        if not refs:
            return await self.say(event, "用法：发送 /test 并附上图片，或回复一条带图消息发送 /test", REPORT_SECONDS)
        examples = await self.detector.examples.get() if self.detector.examples else []
        example_labels = {e.sha: e.is_nailong for e in examples}
        threshold = self.cfg.threshold
        lines = []
        for i, ref in enumerate(refs[:TEST_MAX_IMAGES], 1):
            head = f"#{i} " if len(refs) > 1 else ""
            data = await self.detector.download(ref)
            if data is None:
                lines.append(f"{head}图片下载失败")
                continue
            sha = sha256(data)
            cached = self.cache.lookup(*ref.keys, "sha256:" + sha, sha=sha)
            try:
                uri = await asyncio.to_thread(to_data_uri, data, self.cfg.frames)
            except Exception as e:
                lines.append(f"{head}图片无法解析：{e}")
                continue
            v = await self.detector.classify(uri)
            if v is None:
                lines.append(f"{head}模型调用失败")
                continue
            act = v.is_nailong and v.confidence >= threshold
            lines.append(f"{head}{'是奶龙' if v.is_nailong else '不是奶龙'} · 置信度 {v.confidence:.2f}"
                         f"（阈值 {threshold}，{'会' if act else '不会'}撤回）")
            lines.append(f"理由：{v.reason}")
            if v.usage:
                lines.append(f"{v.model} · 例子 {v.usage.examples} 张 · {v.usage.seconds:.1f}s")
                lines.append(format_usage(v.usage))
            if cached:
                lines.append(f"缓存里的判定：{'是' if cached.is_nailong else '否'} {cached.confidence:.2f}（{cached.model}）")
            if sha in example_labels:
                lines.append(f"注意：这张图本身就在例子里（例子里标为：{'是' if example_labels[sha] else '否'}）")
        if len(refs) > TEST_MAX_IMAGES:
            lines.append(f"（只检测了前 {TEST_MAX_IMAGES} 张）")
        await self.say(event, "\n".join(lines), REPORT_SECONDS)

    # ---------- /forget ----------

    async def forget_command(self, event: dict[str, Any]) -> None:
        refs, _, _ = await self.collect_images(event)
        if not refs:
            return await self.say(event, "用法：发送 /forget 并附上图片，或回复一条带图消息发送 /forget", REPORT_SECONDS)
        lines = []
        for i, ref in enumerate(refs[:TEST_MAX_IMAGES], 1):
            head = f"#{i} " if len(refs) > 1 else ""
            sha = self.cache.sha_for(*ref.keys)
            if sha is None and (data := await self.detector.download(ref)) is not None:
                sha = sha256(data)
            keys = list(dict.fromkeys([*ref.keys, *(["sha256:" + sha, *self.cache.keys_for(sha)] if sha else [])]))
            current = self.cache.lookup(*keys, sha=sha)
            if current is not None and current.model.startswith("admin:"):
                lines.append(f"{head}这张图有人工标注（{'是' if current.is_nailong else '否'}），/forget 不清除人工标注；"
                             "要改判请用 是奶龙 / 否奶龙")
                continue
            n = self.cache.forget(keys)
            lines.append(f"{head}已清除 {n} 条模型判定，下次出现时会重新判定" if n else f"{head}缓存里没有这张图的判定")
        await self.say(event, "\n".join(lines), REPORT_SECONDS)

    # ---------- /stats ----------

    async def stats_command(self, event: dict[str, Any], window: Window) -> None:
        if self.store is None:
            return await self.say(event, "未开启样本保存（save_dir），无法统计", REPORT_SECONDS)
        group_id = event.get("group_id") if event.get("message_type") == "group" else None
        st = await asyncio.to_thread(compute_stats, self.cache, self.store, self.cfg.threshold, window, group_id)
        card = await asyncio.to_thread(render_stats, st, self.store, group_id)
        image = {"type": "image", "data": {"file": "base64://" + base64.b64encode(card).decode()}}
        names = []
        for uid, count in st.top_senders[:3]:
            name = str(uid)
            if group_id is not None:
                try:
                    info = await self.onebot.call("get_group_member_info", group_id=group_id, user_id=uid)
                    name = f"{info.get('card') or info.get('nickname') or uid}（{uid}）"
                except OneBotError:
                    pass
            names.append(f"{name} {count} 次")
        text = f"{window.zh}：检测 {st.checked} 张，识别为奶龙 {st.detected} 张，撤回 {st.recalled} 条"
        if names:
            text += "\n发奶龙最多：" + "、".join(names)
        await self.say(event, text, STATS_SECONDS, [image])

    # ---------- /wall ----------

    async def wall_command(self, event: dict[str, Any], window: Window) -> None:
        if self.store is None:
            return await self.say(event, "未开启样本保存（save_dir），无法生成奶龙墙", REPORT_SECONDS)
        group_id = event.get("group_id") if event.get("message_type") == "group" else None
        actions = [a for a in self.cache.actions_since(window.since, group_id, window.until) if not a["reverted"]]
        counts = Counter(a["sha"] for a in actions)
        latest: dict[str, int] = {}
        for a in actions:
            latest[a["sha"]] = max(latest.get(a["sha"], 0), a["time"])
        items = []
        for sha in sorted(latest, key=latest.get, reverse=True):
            path = self.store.find(sha)
            # 事后被人工标为非奶龙的不上墙
            if path is None or path.parent == self.store.dirs[("human", False)]:
                continue
            items.append((path, counts[sha]))
        name = window.zh
        if not items:
            return await self.say(event, f"{name}没有被撤回的奶龙", REPORT_SECONDS)

        progress = None
        if len(items) > 10:
            progress = await self.reply(event, f"正在生成奶龙墙（{len(items)} 张），图多时可能要几十秒…")
        scope = f"group {group_id}" if group_id else "all groups"
        data, fmt = await asyncio.to_thread(
            render_wall, items, f"Nailong wall · {window.en}",
            f"{len(items)} images · {sum(c for _, c in items)} recalls · {scope}",
        )
        if progress is not None:
            self._spawn(self._recall_later([progress], 0))
        image = {"type": "image", "data": {"file": "base64://" + base64.b64encode(data).decode()}}
        text = f"{name}被撤回的奶龙 {len(items)} 张"
        if fmt == "jpeg-fallback":
            text += "（图片太多，20MB 内放不下动图，已改为静态图）"
        await self.say(event, text, STATS_SECONDS, [image])

    # ---------- /status ----------

    async def status_command(self, event: dict[str, Any], count: int) -> None:
        d = self.detector
        count = min(max(count, 1), 20)
        uptime = int(time.time() - d.started)
        lines = [f"运行 {uptime // 3600} 小时 {uptime % 3600 // 60} 分 · 模型 {'、'.join(f'{p.name}/{p.model}' for p in d.providers)}"]

        ex = d.examples
        if ex is not None and self.cfg.examples.enabled:
            current = ex.current()
            want_pos, want_neg = ex.counts()
            have_pos, have_neg = ex.available()
            lines.append(
                f"例子：使用 正 {sum(e.is_nailong for e in current)} / 反 {sum(not e.is_nailong for e in current)}"
                f"（设置 {want_pos}/{want_neg}，上限合计 {ex.limit()}，人工标注 是 {have_pos} / 否 {have_neg}）"
            )
            if ex.updated_at:
                ago = int(time.time() - ex.updated_at)
                lines.append(f"例子上次变化：{ago // 60} 分 {ago % 60} 秒前（标注变化后最快 {self.cfg.examples.rebuild_seconds:g} 秒内生效）")
            # 前缀 = 系统提示 + 例子；带同样数量例子的调用里命中缓存的部分就是已缓存的前缀
            same = [u for u in d.usage if u.examples == len(current) and u.cached is not None]
            if same:
                lines.append(f"前缀（系统提示 + {len(current)} 张例子）：已缓存约 {max(u.cached for u in same)} token")
            else:
                lines.append(f"前缀（系统提示 + {len(current)} 张例子）：还没有带这组例子的调用")
        else:
            lines.append("例子：未开启")

        recent = list(d.usage)[-count:]
        if recent:
            lines.append(f"最近 {len(recent)} 次调用（输入 = 缓存 + 新）：")
            for u in reversed(recent):
                kind = {"detect": "检测", "test": "测试"}.get(u.kind, u.kind)
                lines.append(f"{time.strftime('%H:%M:%S', time.localtime(u.time))} {kind} 例子{u.examples} · "
                             f"{format_usage(u)} · {u.seconds:.1f}s")
        else:
            lines.append("启动以来还没有调用过模型")

        hour = [u for u in d.usage if time.time() - u.time < 3600]
        if hour:
            prompt = sum(u.prompt for u in hour)
            cached = sum(u.cached or 0 for u in hour)
            lines.append(f"近 1 小时：调用 {len(hour)} 次，输入 {prompt} token，缓存命中 {cached / prompt:.0%}，"
                         f"处理 {self.cache.count_actions(time.time() - 3600)} 条消息" if prompt else f"近 1 小时：调用 {len(hour)} 次")
        await self.say(event, "\n".join(lines), REPORT_SECONDS)

    # ---------- /examples ----------

    async def examples_command(self, event: dict[str, Any], m: re.Match[str]) -> None:
        def say(text: str, seconds: float, extra: list[dict[str, Any]] | None = None) -> Any:
            return self.say(event, text, seconds, extra)

        ex = self.detector.examples
        if ex is None:
            return await say("未开启样本保存（save_dir），没有例子可用", EXAMPLES_SET_SECONDS)
        enabled_note = "" if self.cfg.examples.enabled else "\n注意：config 中 examples.enabled 为 false，持续学习未开启。"

        if m.group(1) is not None:
            positive = int(m.group(1))
            negative = int(m.group(2)) if m.group(2) is not None else positive
            limit = ex.limit()
            if positive + negative > limit:
                return await say(
                    f"正反例合计最多 {limit} 张（按模型上下文窗口、每张图最多 {TOKENS_PER_IMAGE} token 估算）",
                    EXAMPLES_SET_SECONDS,
                )
            ex.set_counts(positive, negative)
            used = await ex.get(force=True) if self.cfg.examples.enabled else []
            have_pos, have_neg = ex.available()
            log.info("管理员 %s 把例子数量调整为 正 %d / 反 %d", event["user_id"], positive, negative)
            return await say(
                f"已设置为正例 {positive} 张、反例 {negative} 张。人工标注共有 是 {have_pos} / 否 {have_neg} 张，"
                f"实际使用 正 {sum(e.is_nailong for e in used)} / 反 {sum(not e.is_nailong for e in used)} 张。{enabled_note}",
                EXAMPLES_SET_SECONDS,
            )

        if not self.cfg.examples.enabled:
            return await say("持续学习未开启（config 中 examples.enabled 为 false）", EXAMPLES_SET_SECONDS)
        examples = await ex.get()
        if not examples:
            return await say("还没有人工标注，暂无例子", EXAMPLES_SET_SECONDS)
        tiles = []
        for i, e in enumerate(examples, 1):
            path = self.store.find(e.sha) if self.store else None
            if path is not None:
                data = await asyncio.to_thread(path.read_bytes)
                tiles.append((f"{i} {'Y' if e.is_nailong else 'N'}", data, BADGE_YES if e.is_nailong else BADGE_NO))
        collage = await asyncio.to_thread(make_collage, tiles)
        image = {"type": "image", "data": {"file": "base64://" + base64.b64encode(collage).decode()}}
        want_pos, want_neg = ex.counts()
        have_pos, have_neg = ex.available()
        await say(
            f"当前使用的例子：正例 {sum(e.is_nailong for e in examples)} 张、反例 {sum(not e.is_nailong for e in examples)} 张"
            f"（设置 {want_pos}/{want_neg}，人工标注共 是 {have_pos} / 否 {have_neg}）。"
            f"黄色 Y = 是奶龙，蓝色 N = 不是。用 /examples 正例数 反例数 调整，如 /examples 8 8。",
            EXAMPLES_SHOW_SECONDS, [image],
        )

    # ---------- /rewind ----------

    async def handle_rewind(self, event: dict[str, Any], text: str) -> bool:
        """处理 /rewind 指令和它后续的编号回复。返回 True 表示消息已被消费。"""
        key = (event["group_id"], event["user_id"])
        if m := _REWIND_RE.match(text):
            count = min(max(int(m.group(1) or REWIND_DEFAULT), 1), REWIND_MAX)
            await self.start_rewind(event, count)
            return True
        session = self.rewinds.get(key)
        if session and (text == "取消" or _IDS_RE.match(text)):
            await self.answer_rewind(session, event, text)
            return True
        return False

    async def start_rewind(self, event: dict[str, Any], count: int) -> None:
        group_id, admin_id = event["group_id"], event["user_id"]
        if old := self.rewinds.get((group_id, admin_id)):
            self.end_rewind(old, 0)
        session = RewindSession(group_id, admin_id, {}, [event["message_id"]])

        if self.store is None:
            session.messages.append(await self.reply(event, "未开启样本保存（save_dir），无法生成拼图"))
            self.end_rewind(session, REWIND_LINGER * 2)
            return
        tiles: list[tuple[int, bytes]] = []
        for action in self.cache.recent_actions(group_id, count):
            path = self.store.find(action.sha)
            if path is None:
                continue
            num = len(tiles) + 1
            session.items[num] = action
            tiles.append((str(num), await asyncio.to_thread(path.read_bytes)))
        if not tiles:
            session.messages.append(await self.reply(event, "最近没有处理过的奶龙图"))
            self.end_rewind(session, REWIND_LINGER * 2)
            return

        collage = await asyncio.to_thread(make_collage, tiles)
        image = {"type": "image", "data": {"file": "base64://" + base64.b64encode(collage).decode()}}
        hint = (
            f"最近 {len(tiles)} 张被处理的图。回复编号（如 3 5）把对应图片改判为非奶龙，"
            f"回复 0 或「取消」结束。{REWIND_TIMEOUT // 60} 分钟后自动撤回。"
        )
        session.messages.append(await self.reply(event, hint, [image]))
        self.rewinds[(group_id, admin_id)] = session
        session.timer = self._spawn(self._expire_rewind(session))

    async def answer_rewind(self, session: RewindSession, event: dict[str, Any], text: str) -> None:
        session.messages.append(event["message_id"])
        ids = [int(x) for x in re.findall(r"\d+", text)]
        if text == "取消" or not ids or ids == [0]:
            session.messages.append(await self.reply(event, "已取消"))
            self.end_rewind(session, REWIND_LINGER)
            return
        bad = sorted({i for i in ids if i not in session.items})
        if bad:
            msg = f"没有编号 {' '.join(map(str, bad))}，可选 1-{max(session.items)}，请重新回复"
            session.messages.append(await self.reply(event, msg))
            return

        unban: set[int] = set()
        for i in dict.fromkeys(ids):
            action = session.items[i]
            await asyncio.to_thread(
                apply_human_label, self.cache, self.store, action.sha, False, session.admin_id,
                meta={"group_id": session.group_id, "via": "rewind"},
            )
            self.cache.revert_actions([r["id"] for r in action.records])
            unban.update(r["user_id"] for r in action.records if r["banned"])
        for uid in unban:
            try:
                await self.onebot.call("set_group_ban", group_id=session.group_id, user_id=uid, duration=0)
            except OneBotError as e:
                log.warning("解除禁言失败: %s", e)
        done = f"已将 {' '.join('#' + str(i) for i in dict.fromkeys(ids))} 改判为非奶龙"
        if unban:
            done += f"，并解除 {len(unban)} 人禁言"
        log.info("管理员 %s 通过 /rewind %s", session.admin_id, done)
        session.messages.append(await self.reply(event, done))
        self.end_rewind(session, REWIND_LINGER)

    def end_rewind(self, session: RewindSession, delay: float) -> None:
        """结束会话，delay 秒后撤回本次交互的全部消息。"""
        if self.rewinds.get((session.group_id, session.admin_id)) is session:
            del self.rewinds[(session.group_id, session.admin_id)]
        if session.timer and session.timer is not asyncio.current_task():
            session.timer.cancel()
        self._spawn(self._recall_later([m for m in session.messages if m is not None], delay))

    async def _expire_rewind(self, session: RewindSession) -> None:
        await asyncio.sleep(REWIND_TIMEOUT)
        self.end_rewind(session, 0)

    async def _recall_later(self, message_ids: list[int], delay: float) -> None:
        await asyncio.sleep(delay)
        for mid in message_ids:
            try:
                await self.onebot.call("delete_msg", message_id=mid)
            except OneBotError as e:
                log.debug("撤回 %s 失败: %s", mid, e)

    def _spawn(self, coro: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


_QQ_MD5_RE = re.compile(r"^([0-9A-Fa-f]{32})(?:\.\w+)?$")


def image_keys(file: str) -> list[str]:
    """QQ 图片的缓存键。QQ 文件名是内容的 MD5 加扩展名，同一张图可能以 .png / .jpg 等不同扩展名出现，
    所以另外按 MD5 建一个键，让它们命中同一条缓存。"""
    if not file:
        return []
    keys = [f"file:{file}"]
    if m := _QQ_MD5_RE.match(file):
        keys.append(f"qqmd5:{m.group(1).upper()}")
    return keys


def format_usage(u: Usage) -> str:
    if u.cached is None:
        return f"输入 {u.prompt}（缓存未知）· 输出 {u.completion}"
    return f"输入 {u.prompt} = 缓存 {u.cached} + 新 {u.new} · 输出 {u.completion}"


def plain_text(message: Any) -> str:
    if not isinstance(message, list):
        return ""
    return "".join(seg.get("data", {}).get("text", "") for seg in message if seg.get("type") == "text").strip()


def message_context(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": event.get("group_id"),
        "user_id": event.get("user_id") or (event.get("sender") or {}).get("user_id"),
        "message_id": event.get("message_id"),
    }


def parse_command(message: Any) -> bool | None:
    """解析管理员指令：是奶龙 -> True，否奶龙 -> False，其它 -> None。"""
    m = _COMMAND_RE.match(plain_text(message))
    if not m:
        return None
    return m.group(1) == "是"


def main() -> None:
    parser = argparse.ArgumentParser(description="奶龙检测器")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(NailongBot(cfg).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
