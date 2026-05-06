FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    OPENAI_API_KEY=dummy

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --prefer-binary -r /workspace/requirements.txt

COPY src /workspace/src
COPY configs /workspace/configs
COPY data /workspace/data

ENV PYTHONPATH=/workspace/src \
    REPO_ROOT=/workspace \
    DATA_DIR=/workspace/data \
    CONFIG_DIR=/workspace/configs \
    REPORT_DIR=/workspace/reports \
    LOG_DIR=/workspace/reports/agent_logs \
    WORKSPACE_DIR=/workspace/data/workspace \
    RAG_DIR=/workspace/data/rag \
    NEMO_CONFIG_DIR=/workspace/configs/nemo

EXPOSE 8000
CMD ["uvicorn", "security_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
