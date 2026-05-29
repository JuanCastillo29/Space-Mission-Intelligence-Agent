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

COPY alembic.ini .
COPY db/ db/
COPY scripts/seed.py scripts/seed.py

CMD ["sh", "-c", "alembic upgrade head && python scripts/seed.py"]
