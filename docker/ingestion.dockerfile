FROM python:3.13-slim

ENV PYTHONPATH="/opt/project"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project

COPY requirements/requirements.txt requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements.txt

# Pre-cache the tiktoken BPE encoding so the worker runs fully offline.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN mkdir -p $TIKTOKEN_CACHE_DIR
COPY docker/tiktoken_cache/cl100k_base.tiktoken $TIKTOKEN_CACHE_DIR/9b5ad71b2ce5302211f9c61530b329a4922fc6a4

COPY . .

ENTRYPOINT ["python", "-m", "ingestion"]
