FROM node:20-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PROMPTFOO_DISABLE_TELEMETRY=1 \
    PROMPTFOO_DISABLE_UPDATE=1 \
    PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=true \
    OPENAI_API_KEY=dummy \
    PATH=/venv/bin:$PATH

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    ca-certificates \
    git \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
COPY requirements-evaluator.txt /workspace/requirements-evaluator.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --prefer-binary -r /workspace/requirements-evaluator.txt
RUN npm install -g promptfoo

ENV PYTHONPATH=/workspace/src
CMD ["python", "scripts/run_all.py", "--profile", "full"]
