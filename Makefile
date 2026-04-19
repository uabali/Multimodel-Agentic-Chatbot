.PHONY: setup qdrant llm app dev check stop clean help

SHELL := /bin/bash

# ── Load .env into Make variables (without exporting to sub-shells by default) ─
-include .env
export

# Defaults (overridden by .env if present)
LLAMA_PORT          ?= 8080
LLAMA_CTX_SIZE      ?= 16384
LLAMA_GPU_LAYERS    ?= auto
LLAMA_PARALLEL      ?= 4
LLAMA_HF_REPO      ?= lmstudio-community/gemma-4-E4B-it-GGUF:Q4_K_M
LLM_SERVER_URL      ?= http://localhost:$(LLAMA_PORT)/v1
QDRANT_URL          ?= http://localhost:6333
APP_PORT            ?= 7860

# ─────────────────────────────────────────────────────────────────────────────
help: ## Show this help
	@awk '/^[a-zA-Z_-]+:.*## / {split($$0,a,":.*## "); printf "  \033[36m%-12s\033[0m %s\n", a[1], a[2]}' $(MAKEFILE_LIST)

# ─────────────────────────────────────────────────────────────────────────────
setup: ## Create venv, install deps, copy .env template
	@echo "── setup ──────────────────────────────────────────"
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env from template — edit it now.")
	@echo "Done. Activate with:  source .venv/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
qdrant: ## Start Qdrant (Docker)
	@echo "── qdrant ─────────────────────────────────────────"
	@docker compose up -d
	@echo -n "Waiting for Qdrant... "
	@for i in $$(seq 1 20); do \
		curl -sf http://localhost:6333/readyz > /dev/null 2>&1 && echo "ready." && break; \
		sleep 0.5; \
	done || echo "timeout (may still be starting)."

# ─────────────────────────────────────────────────────────────────────────────
llm: ## Start llama-server (reads LLAMA_* vars from .env)
	@echo "── llm ────────────────────────────────────────────"
	@./start-llama-server.sh

# ─────────────────────────────────────────────────────────────────────────────
app: ## Start Chainlit app
	@echo "── app ────────────────────────────────────────────"
	. .venv/bin/activate && uv run chainlit run src/main.py --port $(APP_PORT)

# ─────────────────────────────────────────────────────────────────────────────
dev: check-qdrant check-llm app ## Start app after verifying Qdrant + LLM are up

check-qdrant:
	@echo -n "Checking Qdrant at $(QDRANT_URL)... "
	@curl -sf $(QDRANT_URL)/readyz > /dev/null 2>&1 \
		&& echo "OK" \
		|| (echo "FAIL — run 'make qdrant' first" && exit 1)

check-llm:
	@echo -n "Checking LLM at $(LLM_SERVER_URL)/models... "
	@curl -sf $(LLM_SERVER_URL)/models > /dev/null 2>&1 \
		&& echo "OK" \
		|| (echo "FAIL — run 'make llm' in another terminal first" && exit 1)

# ─────────────────────────────────────────────────────────────────────────────
check: ## Health-check all services + verify model ids
	@echo "── check ──────────────────────────────────────────"
	@echo -n "Qdrant ($(QDRANT_URL))... "
	@curl -sf $(QDRANT_URL)/readyz > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@echo -n "LLM    ($(LLM_SERVER_URL))... "
	@curl -sf $(LLM_SERVER_URL)/models > /dev/null 2>&1 && echo "OK" || echo "FAIL"
	@echo ""
	@. .venv/bin/activate && python3 scripts/verify_llm_runtime.py

# ─────────────────────────────────────────────────────────────────────────────
tunnel: ## Share Chainlit via ngrok (NGROK_AUTHTOKEN required in .env)
	@./start-tunnel.sh $(APP_PORT)

# ─────────────────────────────────────────────────────────────────────────────
stop: ## Stop all services
	@echo "── stop ───────────────────────────────────────────"
	@docker compose down 2>/dev/null || true
	@pkill -f 'llama-server.*--port $(LLAMA_PORT)' 2>/dev/null && echo "llama-server stopped." || echo "llama-server was not running."

# ─────────────────────────────────────────────────────────────────────────────
clean: stop ## Stop + remove .venv and caches
	rm -rf .venv .rag_cache __pycache__ src/__pycache__
	@echo "Cleaned."
