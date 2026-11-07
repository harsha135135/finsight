# One image for the FinSight API, the worker (under its watchdog) and the supervisor.
# GPU inference stays on the PC; containers reach it through the Mac's SSH tunnel at host.docker.internal:8000.
FROM python:3.12.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
ENV UV_PYTHON_PREFERENCE=only-system UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra app --no-dev --no-install-project
COPY finsight ./finsight
COPY modelforge ./modelforge
COPY migrations ./migrations
RUN uv sync --locked --extra app --no-dev \
    && useradd --create-home --uid 10001 finsight \
    && mkdir -p /app/.finsight && chown finsight /app/.finsight
USER finsight
ENV PATH=/app/.venv/bin:$PATH FINSIGHT_STATE_DIR=/app/.finsight
EXPOSE 8080
CMD ["uvicorn", "finsight.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
