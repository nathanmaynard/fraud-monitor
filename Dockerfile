FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# LightGBM links against OpenMP, which the slim image doesn't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PORT=8080
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
