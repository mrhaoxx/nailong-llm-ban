from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Verdict:
    is_nailong: bool
    confidence: float
    reason: str = ""
    model: str = ""
    # 本次调用的 token 用量（只在刚调用模型时有，不写入缓存）
    usage: Any = None
    # 模型报告图中有试图左右审核结果、或在说明角色身份的文字（只用于记录和提示，不改变结论）
    injection: bool = False
    # 一句话画面描述，供图库搜图参考
    desc: str = ""


@dataclass
class Sighting:
    first_seen: int
    last_seen: int
    count: int


@dataclass
class Action:
    """一张被处理过的图，以及它触发的各条撤回/禁言记录。"""

    sha: str
    last_time: int
    records: list[dict]


class VerdictCache:
    """图片判定结果的持久缓存。

    同一张图可能有多个 key（QQ 文件名、商城表情 id、内容 sha256），
    都指向同一条判定，这样重复出现的表情包无需下载也无需再调大模型。
    另外记录每个 key 对应的内容哈希，以及每张图的出现时间和次数。
    """

    def __init__(self, path: str):
        # 机器人和标注网页会在多个线程里共用这个连接
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self._db.executescript(
            """CREATE TABLE IF NOT EXISTS verdict (
                key TEXT PRIMARY KEY,
                is_nailong INTEGER NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS key_sha (
                key TEXT PRIMARY KEY,
                sha TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sighting (
                sha TEXT PRIMARY KEY,
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                sha TEXT NOT NULL,
                recalled INTEGER NOT NULL,
                banned INTEGER NOT NULL,
                reverted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS seen_log (
                time INTEGER NOT NULL,
                group_id INTEGER,
                sha TEXT NOT NULL,
                nailong INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS seen_log_time ON seen_log (time);
            CREATE TABLE IF NOT EXISTS caption (
                sha TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                time INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tag (
                name TEXT PRIMARY KEY,
                note TEXT NOT NULL DEFAULT '',
                created INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_tag (
                sha TEXT NOT NULL,
                tag TEXT NOT NULL,
                source TEXT NOT NULL,        -- human / auto / rejected（人工去掉的自动标签，不再自动加回）
                time INTEGER NOT NULL,
                PRIMARY KEY (sha, tag)
            );
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,   -- 图库里的永久编号，只增不减
                sha TEXT UNIQUE NOT NULL,
                added INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS setting (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );"""
        )
        self._db.commit()

    def get(self, *keys: str) -> Verdict | None:
        with self._lock:
            for key in keys:
                row = self._db.execute(
                    "SELECT is_nailong, confidence, reason, model FROM verdict WHERE key = ?",
                    (key,),
                ).fetchone()
                if row:
                    return Verdict(bool(row[0]), row[1], row[2], row[3])
        return None

    def lookup(self, *keys: str, sha: str | None = None) -> Verdict | None:
        """按 key 查判定，但人工标注优先。

        同一张图可能有多个 key（例如 QQ 文件名扩展名不同），某个 key 上可能残留着旧的模型判定；
        只要这张图的内容哈希上有人工标注，就以人工标注为准。
        """
        hit = self.get(*keys)
        if hit is not None and hit.model.startswith("admin:"):
            return hit
        sha = sha or self.sha_for(*keys)
        if sha:
            content = self.get("sha256:" + sha)
            if content is not None and content.model.startswith("admin:"):
                return content
        return hit

    def forget(self, keys: list[str]) -> int:
        """删除这些 key 上的模型判定（人工标注保留），返回删除条数。"""
        if not keys:
            return 0
        with self._lock:
            n = self._db.execute(
                f"DELETE FROM verdict WHERE key IN ({','.join('?' * len(keys))}) AND model NOT LIKE 'admin:%'", keys
            ).rowcount
            self._db.commit()
        return n

    def put(self, keys: list[str], v: Verdict) -> None:
        now = int(time.time())
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO verdict VALUES (?, ?, ?, ?, ?, ?)",
                [(k, int(v.is_nailong), v.confidence, v.reason, v.model, now) for k in keys],
            )
            self._db.commit()

    def link(self, keys: list[str], sha: str) -> None:
        """记录 QQ 文件名等 key 对应的内容哈希，之后缓存命中时不用下载也能知道是哪张图。"""
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO key_sha VALUES (?, ?)", [(k, sha) for k in keys])
            self._db.commit()

    def sha_for(self, *keys: str) -> str | None:
        with self._lock:
            for key in keys:
                row = self._db.execute("SELECT sha FROM key_sha WHERE key = ?", (key,)).fetchone()
                if row:
                    return row[0]
        return None

    def get_setting(self, key: str, default: Any = None) -> Any:
        """运行时设置（如 /examples 调整的例子数量），机器人和网页共用，重启后仍有效。"""
        with self._lock:
            row = self._db.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO setting VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()

    def bump_labels(self) -> None:
        """人工标注有变化时调用。机器人据此立刻重选例子（网页和机器人是两个进程，靠数据库传递）。"""
        with self._lock:
            self._db.execute(
                """INSERT INTO setting VALUES ('labels.version', '1')
                   ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"""
            )
            self._db.commit()

    def labels_version(self) -> int:
        return int(self.get_setting("labels.version", 0))

    def set_caption(self, sha: str, text: str) -> None:
        """图片的一句话描述，供图库搜图参考。已有描述时不改：描述清单在图库请求里，中途改一条会让它后面的缓存失效。"""
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO caption VALUES (?, ?, ?)",
                (sha, text.strip()[:60], int(time.time())),
            )
            self._db.commit()

    def captions(self) -> list[tuple[str, str]]:
        """全部 (sha, 描述)，按加入时间从早到晚。"""
        with self._lock:
            return self._db.execute("SELECT sha, text FROM caption ORDER BY time, sha").fetchall()

    # ---------- 标签 ----------

    def upsert_tag(self, name: str, note: str | None = None) -> bool:
        """新建标签或更新说明（note 为 None 时不改说明）。返回是否新建。"""
        with self._lock:
            exists = self._db.execute("SELECT 1 FROM tag WHERE name = ?", (name,)).fetchone() is not None
            if not exists:
                self._db.execute("INSERT INTO tag VALUES (?, ?, ?)", (name, note or "", int(time.time())))
            elif note is not None:
                self._db.execute("UPDATE tag SET note = ? WHERE name = ?", (note, name))
            self._db.commit()
        return not exists

    def tags(self) -> list[dict[str, Any]]:
        """全部标签及人工/自动数量，按创建时间。"""
        with self._lock:
            rows = self._db.execute(
                """SELECT t.name, t.note,
                          SUM(CASE WHEN i.source = 'human' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN i.source = 'auto' THEN 1 ELSE 0 END)
                   FROM tag t LEFT JOIN image_tag i ON i.tag = t.name
                   GROUP BY t.name ORDER BY t.created, t.name"""
            ).fetchall()
        return [{"name": n, "note": note, "human": h or 0, "auto": a or 0} for n, note, h, a in rows]

    def tag_note(self, name: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT note FROM tag WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def delete_tag(self, name: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM image_tag WHERE tag = ?", (name,))
            self._db.execute("DELETE FROM tag WHERE name = ?", (name,))
            self._db.commit()

    def set_image_tag(self, sha: str, tag: str, source: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO image_tag VALUES (?, ?, ?, ?)", (sha, tag, source, int(time.time())))
            self._db.commit()

    def image_tags(self, sha: str) -> dict[str, str]:
        """这张图的标签 -> 来源（不含 rejected）。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT tag, source FROM image_tag WHERE sha = ? AND source != 'rejected' ORDER BY time", (sha,)
            ).fetchall()
        return dict(rows)

    def all_image_tags(self) -> dict[str, dict[str, str]]:
        with self._lock:
            rows = self._db.execute("SELECT sha, tag, source FROM image_tag WHERE source != 'rejected'").fetchall()
        out: dict[str, dict[str, str]] = {}
        for sha, tag, source in rows:
            out.setdefault(sha, {})[tag] = source
        return out

    def tagged(self, tag: str) -> list[tuple[str, str]]:
        """某标签下的 (sha, 来源)，人工的在前、各自按时间从新到旧。"""
        with self._lock:
            return self._db.execute(
                """SELECT sha, source FROM image_tag WHERE tag = ? AND source != 'rejected'
                   ORDER BY source = 'auto', time DESC""", (tag,)
            ).fetchall()

    def replace_auto_tags(self, tag: str, shas: list[str]) -> None:
        """用新一轮自动推广的结果替换该标签的自动标签；人工标注和人工拒绝的不动。"""
        now = int(time.time())
        with self._lock:
            self._db.execute("DELETE FROM image_tag WHERE tag = ? AND source = 'auto'", (tag,))
            self._db.executemany(
                "INSERT OR IGNORE INTO image_tag VALUES (?, ?, 'auto', ?)", [(sha, tag, now) for sha in shas]
            )
            self._db.commit()

    def rejected(self, tag: str) -> set[str]:
        with self._lock:
            return {r[0] for r in self._db.execute(
                "SELECT sha FROM image_tag WHERE tag = ? AND source = 'rejected'", (tag,))}

    def gallery_add(self, shas: list[str]) -> int:
        """按给定顺序追加到图库末尾（已在图库里的跳过），返回新增数量。"""
        now = int(time.time())
        with self._lock:
            before = self._db.total_changes
            self._db.executemany("INSERT OR IGNORE INTO gallery (sha, added) VALUES (?, ?)", [(s, now) for s in shas])
            self._db.commit()
            return self._db.total_changes - before

    def gallery_list(self) -> list[tuple[int, str]]:
        """图库里的 (编号, sha)，按编号。"""
        with self._lock:
            return self._db.execute("SELECT id, sha FROM gallery ORDER BY id").fetchall()

    def keys_for(self, sha: str) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT key FROM key_sha WHERE sha = ?", (sha,))]

    def log_action(self, group_id: int, user_id: int, message_id: int, sha: str, recalled: bool, banned: bool) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO action (time, group_id, user_id, message_id, sha, recalled, banned) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (int(time.time()), group_id, user_id, message_id, sha, int(recalled), int(banned)),
            )
            self._db.commit()

    def recent_actions(self, group_id: int, limit: int) -> list[Action]:
        """本群最近处理过的不同图片（按最近一次处理排序），每张图附带它的全部未撤销处理记录。"""
        with self._lock:
            rows = self._db.execute(
                """SELECT id, time, user_id, message_id, sha, recalled, banned FROM action
                   WHERE group_id = ? AND reverted = 0 ORDER BY id DESC""",
                (group_id,),
            ).fetchall()
        by_sha: dict[str, Action] = {}
        for rid, t, uid, mid, sha, recalled, banned in rows:
            if sha not in by_sha:
                if len(by_sha) >= limit:
                    continue
                by_sha[sha] = Action(sha, t, [])
            by_sha[sha].records.append({"id": rid, "user_id": uid, "message_id": mid, "banned": bool(banned)})
        return list(by_sha.values())

    def count_actions(self, since: float) -> int:
        """since 之后处理过的消息数。"""
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(DISTINCT message_id) FROM action WHERE time >= ?", (int(since),)
            ).fetchone()[0]

    def revert_actions(self, ids: list[int]) -> None:
        with self._lock:
            self._db.executemany("UPDATE action SET reverted = 1 WHERE id = ?", [(i,) for i in ids])
            self._db.commit()

    def seen(self, sha: str, group_id: int | None = None, nailong: bool = False) -> None:
        """记录一次出现：更新首次/最近出现时间和次数，并写一行检测日志（供 /stats 按时间统计）。"""
        now = int(time.time())
        with self._lock:
            self._db.execute(
                """INSERT INTO sighting VALUES (?, ?, ?, 1)
                   ON CONFLICT(sha) DO UPDATE SET last_seen = excluded.last_seen, count = count + 1""",
                (sha, now, now),
            )
            self._db.execute("INSERT INTO seen_log VALUES (?, ?, ?, ?)", (now, group_id, sha, int(nailong)))
            self._db.commit()

    def seen_since(self, since: float, group_id: int | None = None, until: float | None = None) -> list[tuple[int, str, bool]]:
        """[since, until) 内的 (时间, sha, 是否判为奶龙)。group_id 为 None 时统计所有群。"""
        sql = "SELECT time, sha, nailong FROM seen_log WHERE time >= ? AND time < ?"
        args: list[Any] = [int(since), int(until if until is not None else time.time() + 1)]
        if group_id is not None:
            sql += " AND group_id = ?"
            args.append(group_id)
        with self._lock:
            return [(t, s, bool(n)) for t, s, n in self._db.execute(sql, args)]

    def actions_since(self, since: float, group_id: int | None = None, until: float | None = None) -> list[dict[str, Any]]:
        """[since, until) 内的撤回/禁言记录。"""
        sql = "SELECT time, user_id, message_id, sha, banned, reverted FROM action WHERE time >= ? AND time < ?"
        args: list[Any] = [int(since), int(until if until is not None else time.time() + 1)]
        if group_id is not None:
            sql += " AND group_id = ?"
            args.append(group_id)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        keys = ("time", "user_id", "message_id", "sha", "banned", "reverted")
        return [dict(zip(keys, r)) for r in rows]

    def sightings(self) -> dict[str, Sighting]:
        with self._lock:
            rows = self._db.execute("SELECT sha, first_seen, last_seen, count FROM sighting").fetchall()
        return {r[0]: Sighting(r[1], r[2], r[3]) for r in rows}

    def close(self) -> None:
        self._db.close()
