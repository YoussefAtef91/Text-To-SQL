FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PORT=8000
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /app

# Phase 1: install dependencies using mount optimization
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy app source code
COPY . .

# Phase 2: install project-specific deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "python", "main.py"]

EXPOSE ${PORT}
