.PHONY: install dev test lint eval docker-up clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

dev:
	.venv/bin/uvicorn app.main:app --reload

test:
	.venv/bin/pytest tests/ -v

lint:
	.venv/bin/ruff check app/ tests/ evals/

eval:
	.venv/bin/python -m evals.run

docker-up:
	docker compose up --build

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
