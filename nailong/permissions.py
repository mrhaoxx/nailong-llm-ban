"""指令权限与冷却。

三个级别：admin（config 里的 admins，所有群所有指令）、group_admin（群主/群管理员）、member（所有人）。
每个指令有一个「至少需要的级别」，可以按群设置（群里用 /perm，存在数据库里；或写在 config 的 permissions 里），
默认只有 admin 能用。会改数据的指令最多只能开放到 group_admin。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .cache import VerdictCache
from .config import Config

log = logging.getLogger(__name__)

LEVELS = ("member", "group_admin", "admin")
RANK = {lv: i for i, lv in enumerate(LEVELS)}
LEVEL_NAMES = {"member": "所有人", "group_admin": "群管", "admin": "管理员"}
_ALIASES = {
    "member": "member", "all": "member", "所有人": "member", "成员": "member", "群员": "member",
    "group_admin": "group_admin", "群管": "group_admin", "群管理": "group_admin", "群主": "group_admin",
    "admin": "admin", "管理员": "admin", "bot_admin": "admin",
}


@dataclass(frozen=True)
class Command:
    usage: str
    # 最多能开放到哪一级：只读指令可以到 member，会改数据的最多到 group_admin
    max_open: str
    # (冷却范围 "user" / "group", 秒)；config 里的管理员不受冷却限制
    cooldown: tuple[str, int] | None
    # 回复多少秒后连同指令一起撤回；0 表示保留
    recall: float


COMMANDS: dict[str, Command] = {
    "help": Command("/help", "member", ("group", 60), 120),
    "test": Command("/test ＋图片：用线上设置检测一次", "member", ("user", 30), 0),
    "stats": Command("/stats [时间范围]：统计卡片", "member", ("group", 60), 0),
    "wall": Command("/wall [时间范围]：奶龙墙", "member", ("group", 300), 0),
    "status": Command("/status [条数]：最近模型调用的上下文用量", "member", ("group", 60), 120),
    "examples": Command("/examples：查看当前例子", "member", ("group", 60), 120),
    "examples-set": Command("/examples 正例数 反例数：调整例子数量", "group_admin", None, 30),
    "forget": Command("/forget ＋图片：清除这张图的模型判定", "group_admin", None, 120),
    "rewind": Command("/rewind [数量]：改判最近处理过的奶龙图", "group_admin", None, 0),
    "label": Command("是奶龙 / 否奶龙 ＋图片：人工标注", "group_admin", None, 0),
}
COOLDOWN_NOTICE_SECONDS = 10


def parse_level(text: str) -> str | None:
    return _ALIASES.get(text.strip().lower())


class Permissions:
    def __init__(self, cfg: Config, cache: VerdictCache):
        self.cfg = cfg
        self.cache = cache
        self._last: dict[tuple[str, Any], float] = {}
        self._noticed: set[tuple[str, Any]] = set()
        for gid, table in cfg.permissions.groups.items():
            for name, level in table.items():
                if name not in COMMANDS or parse_level(str(level)) is None:
                    log.warning("config permissions：群 %s 的 %s=%s 无效，已忽略", gid, name, level)
                elif RANK[parse_level(str(level))] < RANK[COMMANDS[name].max_open]:
                    log.warning("config permissions：%s 最多只能开放到 %s，群 %s 的设置已按上限处理",
                                name, COMMANDS[name].max_open, gid)

    def level_of(self, event: dict[str, Any]) -> str:
        if event.get("user_id") in self.cfg.admins:
            return "admin"
        if event.get("message_type") == "group" and (event.get("sender") or {}).get("role") in ("owner", "admin"):
            return "group_admin"
        return "member"

    def _overrides(self, group_id: int) -> dict[str, str]:
        return self.cache.get_setting(f"perm.{group_id}", {}) or {}

    def required(self, group_id: int | None, name: str) -> str:
        """这个群里使用该指令至少需要的级别：/perm 设置 > config > 默认 admin，且不低于该指令的开放上限。"""
        if group_id is None:
            return "admin"
        level = self._overrides(group_id).get(name)
        if level is None:
            raw = self.cfg.permissions.groups.get(group_id, {}).get(name)
            level = parse_level(str(raw)) if raw is not None else None
        level = level or "admin"
        cap = COMMANDS[name].max_open
        return level if RANK[level] >= RANK[cap] else cap

    def allowed(self, event: dict[str, Any], name: str) -> bool:
        level = self.level_of(event)
        if level == "admin":
            return True
        return RANK[level] >= RANK[self.required(event.get("group_id"), name)]

    def set(self, group_id: int, name: str, level: str | None) -> None:
        table = self._overrides(group_id)
        if level is None:
            table.pop(name, None)
        else:
            table[name] = level
        self.cache.set_setting(f"perm.{group_id}", table)

    def recall_seconds(self, name: str) -> float:
        return float(self.cfg.permissions.recall.get(name, COMMANDS[name].recall))

    def cooldown(self, event: dict[str, Any], name: str) -> tuple[int, bool]:
        """(还要等几秒, 这次是否需要提示)。没在冷却时返回 (0, False) 并开始计时。config 管理员不受限制。"""
        cmd = COMMANDS[name]
        if cmd.cooldown is None or self.level_of(event) == "admin":
            return 0, False
        scope, seconds = cmd.cooldown
        key = (name, event.get("user_id") if scope == "user" else event.get("group_id"))
        now = time.monotonic()
        wait = self._last.get(key, float("-inf")) + seconds - now
        if wait > 0:
            first = key not in self._noticed
            self._noticed.add(key)
            return int(wait) + 1, first
        self._last[key] = now
        self._noticed.discard(key)
        return 0, False
