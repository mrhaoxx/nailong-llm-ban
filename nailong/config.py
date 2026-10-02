from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    vision: bool = True
    # 额外透传给 chat.completions.create 的参数，例如 temperature / max_tokens
    params: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    # 上下文窗口（token）和单次请求最多几张图，用来算例子上限；不填则按内置的已知模型取值
    context_window: int | None = None
    max_images: int | None = None


@dataclass
class OneBotConfig:
    # forward: 本程序主动连接 NapCat 的 WebSocket 服务端
    # reverse: 本程序作为服务端，等待 NapCat 的 WebSocket 客户端连进来
    mode: str = "reverse"
    url: str = "ws://127.0.0.1:3001"
    host: str = "0.0.0.0"
    port: int = 8080
    access_token: str = ""
    api_timeout: float = 15.0


@dataclass
class ActionConfig:
    recall: bool = True
    ban_seconds: int = 0
    notice: str = ""


@dataclass
class FrameConfig:
    # 动图最多送给模型几帧，拼成接近正方形的网格
    max_frames: int = 4
    # 拼好后整张图的最长边（像素），帧越多每帧越小
    max_side: int = 768
    # 相邻取样前先去重：两帧缩略图的平均色差（0-255）低于该值视为同一画面，0 表示不去重
    dedupe_threshold: float = 8.0
    # 超长动图最多扫描的帧数，防止几百帧的 GIF 占满 CPU
    max_scan: int = 300


@dataclass
class ExamplesConfig:
    # 从人工标注里挑正反例附在请求中；需要 save_dir
    enabled: bool = True
    positive: int = 4
    negative: int = 4
    # 例子图片的最长边，越小越省 token
    max_side: int = 384
    # 人工标注变化后，最快多久重选一次例子。例子一变请求前缀就变、缓存失效，
    # 连续标注时靠这个间隔合并成一次重选；标注没变化时不会重选
    rebuild_seconds: float = 60


@dataclass
class PermissionsConfig:
    # 按群设置指令权限：{群号: {指令名: member / group_admin / admin}}；群里用 /perm 调整的优先
    groups: dict[int, dict[str, str]] = field(default_factory=dict)
    # 指令回复多少秒后撤回：{指令名: 秒}，0 表示不撤回；不写则用默认值
    recall: dict[str, float] = field(default_factory=dict)


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8081
    # 访问口令；为空则每次启动随机生成，访问地址会打印在日志里
    token: str = ""


@dataclass
class Config:
    onebot: OneBotConfig
    providers: list[ProviderConfig]
    action: ActionConfig
    frames: FrameConfig = field(default_factory=FrameConfig)
    web: WebConfig = field(default_factory=WebConfig)
    examples: ExamplesConfig = field(default_factory=ExamplesConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    # 只处理这些群；为空则不处理任何群
    groups: list[int] = field(default_factory=list)
    # 可以用 是奶龙/否奶龙 指令纠正判定的 QQ 号，他们的消息也不会被检测
    admins: list[int] = field(default_factory=list)
    skip_admins: bool = True
    threshold: float = 0.7
    cache_path: str = "nailong_cache.sqlite3"
    # 样本集目录；为空则不保存
    save_dir: str = "nailong_images"
    # 单张图片下载上限；奶龙动图常常超过 10MB
    max_image_bytes: int = 50 * 1024 * 1024
    log_level: str = "INFO"


def _expand(value: Any) -> Any:
    """递归展开字符串中的 ${ENV_VAR}，方便把 API key 放在环境变量里。"""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _permissions(raw: dict[str, Any]) -> PermissionsConfig:
    groups = {int(gid): {str(k): str(v) for k, v in (table or {}).items()} for gid, table in (raw.get("groups") or {}).items()}
    recall = {str(k): float(v) for k, v in (raw.get("recall") or {}).items()}
    return PermissionsConfig(groups=groups, recall=recall)


def load_config(path: str | Path) -> Config:
    raw = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    providers = [ProviderConfig(**p) for p in raw.pop("providers", [])]
    if not providers:
        raise ValueError("配置中至少需要一个 provider")
    return Config(
        onebot=OneBotConfig(**raw.pop("onebot", {})),
        providers=providers,
        action=ActionConfig(**raw.pop("action", {})),
        frames=FrameConfig(**raw.pop("frames", {})),
        web=WebConfig(**raw.pop("web", {})),
        examples=ExamplesConfig(**raw.pop("examples", {})),
        permissions=_permissions(raw.pop("permissions", None) or {}),
        **raw,
    )
