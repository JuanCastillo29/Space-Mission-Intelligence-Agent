.PHONY: install dev lint format test test-cov ingest query evaluate up down build shell clean migrate seed

install:
	pip install -r requirements/requirements.txt

dev:
	pip install -r requirements/requirements-dev.txt

lint:
	ruff check .
	ruff format --check .
	mypy db scripts --ignore-missing-imports

format:
	ruff check --fix .
	ruff format .

test:
	pytest

test-cov:
	pytest --cov=app --cov=ingestion --cov=retrieval --cov=generation --cov-report=term-missing

ingest:
	python -m ingestion.run

query:
	python -m app.main

evaluate:
	python -m evaluation.run

# ── Database ──
migrate:
	alembic upgrade head

seed:
	python scripts/seed.py

data-quality:
	python -m scripts.data_quality

data-quality-llm:
	python -m scripts.data_quality --llm

# ── Docker ──
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec dev /bin/bash

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
