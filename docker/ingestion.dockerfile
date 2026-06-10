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
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .

ENTRYPOINT ["python", "-m", "ingestion"]
