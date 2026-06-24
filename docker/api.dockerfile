FROM python:3.13-slim

ENV PYTHONPATH="/opt/project"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
