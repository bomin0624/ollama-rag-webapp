PYTHON ?= python
UV ?= uv
APP ?= main:app
HOST ?= 0.0.0.0
PORT ?= 8000
RETRIEVER ?= hybrid
SPLIT ?= dev
DIM ?= 512
BACKEND ?= ollama
COMPOSE ?= docker compose

# Derived from BACKEND so the profile and the URL cannot drift apart. Two
# spellings: service names resolve only inside the Compose network.
BACKEND_URL_ollama       = http://ollama:11434
BACKEND_URL_vllm         = http://vllm:8000
LOCAL_BACKEND_URL_ollama = http://localhost:11434
LOCAL_BACKEND_URL_vllm   = http://localhost:8001

BACKEND_URL       = $(BACKEND_URL_$(BACKEND))
LOCAL_BACKEND_URL = $(LOCAL_BACKEND_URL_$(BACKEND))

# Readiness probe: Ollama answers on /, vLLM on /health.
HEALTH_PATH_ollama =
HEALTH_PATH_vllm   = /health
HEALTH_PATH = $(HEALTH_PATH_$(BACKEND))

# 5-second steps, so 120 is a 10-minute ceiling for vLLM's cold start.
BACKEND_WAIT_TRIES ?= 120

# Mirrors DEFAULT_GENERATE_MODELS["ollama"] in src/config.py.
OLLAMA_MODEL ?= $(or $(GENERATE_MODEL),llama3.1)

.PHONY: help install format format-check lint test evaluate ci \
	ollama vllm run backend backend-only backend-down \
	wait-backend ollama-pull check-backend

# `run` blocks, so a line's steps must stay ordered.
.NOTPARALLEL:

help:
	@printf "Full stacks (backend in Docker, FastAPI on the host):\n"
	@printf "  ollama        Start Ollama, pull $(OLLAMA_MODEL), serve the app\n"
	@printf "  vllm          Start vLLM, wait for it, serve the app\n"
	@printf "\nIndividual steps:\n"
	@printf "  install       Install project dependencies with uv\n"
	@printf "  format        Format Python files and fix lint issues with ruff\n"
	@printf "  format-check  Check formatting and lint rules with ruff\n"
	@printf "  lint          Run ruff linter\n"
	@printf "  test          Run the unit tests with pytest\n"
	@printf "  evaluate      Run retriever evaluation (RETRIEVER=hybrid SPLIT=dev DIM=512)\n"
	@printf "  ci            Run all CI checks\n"
	@printf "  run           Start the FastAPI app locally against BACKEND\n"
	@printf "  backend       Start the API in Docker too, with exactly one backend\n"
	@printf "  backend-only  Start just the backend container\n"
	@printf "  backend-down  Stop the API and every backend\n"

# The two supported lines
ollama: BACKEND = ollama
ollama: backend-only wait-backend ollama-pull run

vllm: BACKEND = vllm
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

# Published port, since Compose service names do not resolve on the host.
run:
	LLM_BACKEND=$${LLM_BACKEND:-$(BACKEND)} \
	LLM_BASE_URL=$${LLM_BASE_URL:-$(LOCAL_BACKEND_URL)} \
		$(UV) run uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

# Profiles filter `down` too, so switching backends needs `--profile "*"`.
backend: check-backend backend-down
	LLM_BACKEND=$(BACKEND) LLM_BASE_URL=$(BACKEND_URL) \
		$(COMPOSE) --profile $(BACKEND) up -d
	@test '$(BACKEND)' != ollama || \
		$(MAKE) --no-print-directory wait-backend ollama-pull BACKEND=ollama

# Naming the service skips the api container, leaving the GPU to `make run`.
backend-only: check-backend backend-down
	$(COMPOSE) --profile $(BACKEND) up -d $(BACKEND)

# The app never connects at startup, so a booting vLLM would fail at /query.
wait-backend: check-backend
	@printf 'Waiting for %s at %s ' '$(BACKEND)' '$(LOCAL_BACKEND_URL)'; \
	for _ in $$(seq $(BACKEND_WAIT_TRIES)); do \
		if curl -sf -o /dev/null $(LOCAL_BACKEND_URL)$(HEALTH_PATH); then \
			printf 'ready\n'; exit 0; \
		fi; \
		printf '.'; sleep 5; \
	done; \
	printf '\n%s never answered. Check: %s logs %s\n' \
		'$(BACKEND)' '$(COMPOSE)' '$(BACKEND)'; \
	exit 1

# Ollama ships no weights; a no-op once ./ollama holds the model.
ollama-pull:
	$(COMPOSE) exec ollama ollama pull $(OLLAMA_MODEL)

check-backend:
	@test -n "$(BACKEND_URL)" || { \
		printf 'Unknown BACKEND=%s. Use ollama or vllm.\n' '$(BACKEND)'; \
		exit 1; \
	}

backend-down:
	$(COMPOSE) --profile "*" down
