FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install PyTorch CPU-only wheel first to avoid pulling gigabytes of CUDA packages on ARM64/CPU
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./

# Install application dependencies
RUN uv pip install --system -r requirements.txt

# Copy application source code
COPY api/ api/
COPY core/ core/
COPY schemas/ schemas/
COPY services/ services/
COPY main.py .

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
