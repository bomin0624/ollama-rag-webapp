PYTHON ?= python
UV ?= uv
APP ?= main:app
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help install format format-check lint test ci run

help:
	@printf "Available targets:\n"
	@printf "  install       Install project dependencies with uv\n"
	@printf "  format        Format Python files with black and fix lint issues with ruff\n"
	@printf "  format-check  Check formatting (black) and lint rules (ruff)\n"
	@printf "  lint          Run ruff linter\n"
	@printf "  test          Run tests if present\n"
	@printf "  ci            Run all CI checks\n"
	@printf "  run           Start the FastAPI app with uvicorn\n"

install:
	$(UV) sync --locked --all-groups

format:
	$(UV) run black .
	$(UV) run ruff check --fix .

format-check:
	$(UV) run black --check --diff .
	$(UV) run ruff check .

lint:
	$(UV) run ruff check .

test:
	@if [ -d tests ]; then \
		$(UV) run $(PYTHON) -m pytest; \
	else \
		printf "No tests directory found; skipping tests.\n"; \
	fi

ci: format-check test

run:
	$(UV) run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload
