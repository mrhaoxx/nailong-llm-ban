from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.client import connect
from websockets.asyncio.connection import Connection
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .config import OneBotConfig

log = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class OneBotError(RuntimeError):
    pass


class OneBot:
    """OneBot v11 WebSocket 连接（NapCat），支持正向和反向两种模式。"""

    def __init__(self, cfg: OneBotConfig, on_event: EventHandler):
        self.cfg = cfg
        self.on_event = on_event
        self._ws: Connection | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def call(self, action: str, timeout: float | None = None, **params: Any) -> Any:
        """调用 NapCat API。超时、断线和失败都抛 OneBotError，调用方只需处理这一种异常。"""
        ws = self._ws
        if ws is None:
            raise OneBotError("NapCat 未连接")
        echo = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        try:
            await ws.send(json.dumps({"action": action, "params": params, "echo": echo}))
            resp = await asyncio.wait_for(fut, timeout or self.cfg.api_timeout)
        except TimeoutError as e:
            raise OneBotError(f"{action} 超时") from e
        except ConnectionClosed as e:
            raise OneBotError(f"{action} 发送时连接断开") from e
        finally:
            self._pending.pop(echo, None)
        if resp.get("status") != "ok":
            raise OneBotError(f"{action} 失败: {resp.get('retcode')} {resp.get('message') or resp.get('wording')}")
        return resp.get("data")

    async def run(self) -> None:
        if self.cfg.mode == "forward":
            await self._run_forward()
        elif self.cfg.mode == "reverse":
            await self._run_reverse()
        else:
            raise ValueError(f"未知的 onebot.mode: {self.cfg.mode}")

    async def _run_forward(self) -> None:
        headers = {"Authorization": f"Bearer {self.cfg.access_token}"} if self.cfg.access_token else None
        # connect() 作为异步迭代器使用时会自动断线重连
        async for ws in connect(self.cfg.url, additional_headers=headers, max_size=None):
            log.info("已连接 NapCat: %s", self.cfg.url)
            try:
                await self._serve(ws)
            except ConnectionClosed:
                pass
            log.warning("与 NapCat 的连接断开，重连中…")

    async def _run_reverse(self) -> None:
        async def handler(ws: ServerConnection) -> None:
            if self.cfg.access_token:
                auth = ws.request.headers.get("Authorization", "") if ws.request else ""
                if auth.removeprefix("Bearer ").removeprefix("Token ").strip() != self.cfg.access_token:
                    log.warning("拒绝 token 不正确的连接: %s", ws.remote_address)
                    await ws.close(1008, "invalid token")
                    return
            log.info("NapCat 已连入: %s", ws.remote_address)
            try:
                await self._serve(ws)
            except ConnectionClosed:
                pass
            log.warning("NapCat 连接断开，等待重连…")

        async with serve(handler, self.cfg.host, self.cfg.port, max_size=None) as server:
            log.info("反向 WebSocket 监听 ws://%s:%d", self.cfg.host, self.cfg.port)
            await server.serve_forever()

    async def _serve(self, ws: Connection) -> None:
        self._ws = ws
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                echo = data.get("echo")
                if echo is not None and echo in self._pending:
                    fut = self._pending[echo]
                    if not fut.done():
                        fut.set_result(data)
                elif "post_type" in data:
                    task = asyncio.create_task(self._dispatch(data))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        finally:
            if self._ws is ws:
                self._ws = None
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(OneBotError("连接已断开"))

    async def _dispatch(self, event: dict[str, Any]) -> None:
        try:
            await self.on_event(event)
        except Exception:
            log.exception("处理事件出错")
