PYTHON ?= python
UV ?= uv
APP ?= main:app
HOST ?= 0.0.0.0
PORT ?= 8000
RETRIEVER ?= hybrid
SPLIT ?= dev
DIM ?= 512

.PHONY: help install format format-check lint test evaluate ci run

help:
	@printf "Available targets:\n"
	@printf "  install       Install project dependencies with uv\n"
	@printf "  format        Format Python files and fix lint issues with ruff\n"
	@printf "  format-check  Check formatting and lint rules with ruff\n"
	@printf "  lint          Run ruff linter\n"
	@printf "  test          Run tests if present\n"
	@printf "  evaluate      Run retriever evaluation (RETRIEVER=hybrid SPLIT=dev DIM=512)\n"
	@printf "  ci            Run all CI checks\n"
	@printf "  run           Start the FastAPI app with uvicorn\n"

install:
	$(UV) sync --locked --all-groups

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

format-check:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

lint:
	$(UV) run ruff check .

test:
	@if [ -d tests ]; then \
		$(UV) run $(PYTHON) -m pytest; \
	else \
		printf "No tests directory found; skipping tests.\n"; \
	fi

evaluate:
	EMBED_TRUNCATE_DIM=$(DIM) $(UV) run $(PYTHON) -m test.evaluation --retriever $(RETRIEVER) --split $(SPLIT)

ci: format-check test

run:
	$(UV) run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload
