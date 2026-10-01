# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# 先装依赖，代码变动时这一层可以复用
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY nailong ./nailong
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai
# 配置、缓存数据库和样本集都放在 /data（config 里的相对路径以它为基准）
WORKDIR /data
VOLUME /data
# 8080: OneBot 反向 WebSocket；8081: 标注网页
EXPOSE 8080 8081
CMD ["nailong", "-c", "/data/config.yaml"]
