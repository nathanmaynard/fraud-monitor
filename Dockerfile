FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PORT=8080
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
