PYTHON ?= python
UV ?= uv
APP ?= main:app
HOST ?= 0.0.0.0
PORT ?= 8000
RETRIEVER ?= hybrid
SPLIT ?= dev
DIM ?= 512
COMPOSE ?= docker compose

# Two spellings: the service name resolves only inside the Compose network.
BACKEND_URL       = http://vllm:8000
LOCAL_BACKEND_URL = http://localhost:8001

# 5-second steps, so 120 is a 10-minute ceiling for vLLM's cold start.
BACKEND_WAIT_TRIES ?= 120

.PHONY: help install format format-check lint test evaluate ci \
	vllm run backend backend-only backend-down wait-backend

# `run` blocks, so a line's steps must stay ordered.
.NOTPARALLEL:

help:
	@printf "Full stack (vLLM in Docker, FastAPI on the host):\n"
	@printf "  vllm          Start vLLM, wait for it, serve the app\n"
	@printf "\nIndividual steps:\n"
	@printf "  install       Install project dependencies with uv\n"
	@printf "  format        Format Python files and fix lint issues with ruff\n"
	@printf "  format-check  Check formatting and lint rules with ruff\n"
	@printf "  lint          Run ruff linter\n"
	@printf "  test          Run the unit tests with pytest\n"
	@printf "  evaluate      Run retriever evaluation (RETRIEVER=hybrid SPLIT=dev DIM=512)\n"
	@printf "  ci            Run all CI checks\n"
	@printf "  run           Start the FastAPI app locally against vLLM\n"
	@printf "  backend       Start the API in Docker too, alongside vLLM\n"
	@printf "  backend-only  Start just the vLLM container\n"
	@printf "  backend-down  Stop the API and vLLM\n"

vllm: backend-only wait-backend run

install:
	$(UV) sync --locked --all-groups

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:
	$(UV) run ruff check .

# What `ci` runs.
format-check: lint
	$(UV) run ruff format --check .

test:
	$(UV) run $(PYTHON) -m pytest

evaluate:
	EMBED_TRUNCATE_DIM=$(DIM) $(UV) run $(PYTHON) -m evaluate.evaluation --retriever $(RETRIEVER) --split $(SPLIT)

ci: format-check test

run:
	LLM_BASE_URL=$${LLM_BASE_URL:-$(LOCAL_BACKEND_URL)} \
		$(UV) run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

backend: backend-down
	LLM_BASE_URL=$(BACKEND_URL) $(COMPOSE) up -d

# Naming the service skips the api container, leaving the GPU to `make run`.
backend-only: backend-down
	$(COMPOSE) up -d vllm

# The app never connects at startup, so a booting vLLM would fail at /query.
wait-backend:
	@printf 'Waiting for vllm at %s ' '$(LOCAL_BACKEND_URL)'; \
	for _ in $$(seq $(BACKEND_WAIT_TRIES)); do \
		if curl -sf -o /dev/null $(LOCAL_BACKEND_URL)/health; then \
			printf 'ready\n'; exit 0; \
		fi; \
		printf '.'; sleep 5; \
	done; \
	printf '\nvllm never answered. Check: %s logs vllm\n' '$(COMPOSE)'; \
	exit 1

backend-down:
	$(COMPOSE) down
