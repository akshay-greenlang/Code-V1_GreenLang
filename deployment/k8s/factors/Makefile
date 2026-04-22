# GreenLang Factors API — operator Makefile
#
# One-command targets for the day-to-day deploy workflow.  Assumes
# kubectl + kustomize are on PATH and the active context is the right
# cluster (check with `kubectl config current-context` before running
# anything with `-apply` in the name).
#
# Commonly used:
#   make staging-dry          -- dry-run the staging manifests
#   make staging-apply        -- apply the staging manifests
#   make staging-rollout      -- wait for the staging rollout
#   make staging-smoke        -- run smoke tests against staging URL
#   make prod-dry             -- dry-run the prod manifests
#   make prod-apply           -- apply the prod manifests
#   make prod-rollout         -- wait for the prod rollout
#   make rollback-staging     -- kubectl rollout undo on staging
#   make rollback-prod        -- kubectl rollout undo on prod
#   make promote-electricity  -- kick the v1 Electricity slice signoff
#                                (dry-run by default)

SHELL := /bin/bash
NAMESPACE ?= factors
STAGING_URL ?= https://staging.api.greenlang.io
PROD_URL ?= https://api.greenlang.io
KUSTOMIZE_STAGING := deployment/k8s/factors/overlays/staging
KUSTOMIZE_PROD := deployment/k8s/factors/overlays/prod

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

# ------------------------------------------------------------------ #
# Staging
# ------------------------------------------------------------------ #

.PHONY: staging-dry
staging-dry: ## Dry-run the staging kustomize manifests
	kubectl kustomize $(KUSTOMIZE_STAGING) | kubectl apply --dry-run=client -f - --server-side

.PHONY: staging-apply
staging-apply: ## Apply the staging manifests (idempotent)
	kubectl apply -k $(KUSTOMIZE_STAGING)

.PHONY: staging-rollout
staging-rollout: ## Wait for the staging rollout to complete
	kubectl -n $(NAMESPACE) rollout status deployment/factors-api --timeout=5m
	kubectl -n $(NAMESPACE) rollout status deployment/factors-watch --timeout=3m

.PHONY: staging-smoke
staging-smoke: ## Run the smoke test suite against staging
	GL_FACTORS_STAGING_URL=$(STAGING_URL) python -m pytest tests/factors/smoke/test_staging.py -v

.PHONY: staging-logs
staging-logs: ## Tail the staging factors-api logs
	kubectl -n $(NAMESPACE) logs -l app=factors-api -f --tail=100

# ------------------------------------------------------------------ #
# Production (2-approver gate via GHA is the normal path; these are
# fallback commands for incident response).
# ------------------------------------------------------------------ #

.PHONY: prod-dry
prod-dry: ## Dry-run the prod kustomize manifests
	kubectl kustomize $(KUSTOMIZE_PROD) | kubectl apply --dry-run=client -f - --server-side

.PHONY: prod-apply
prod-apply: ## Apply the prod manifests — normally done by CI after 2-approver gate
	@echo "Prefer the CI path (factors-api-deploy.yml). Proceeding anyway in 10s. Ctrl-C to abort."
	@sleep 10
	kubectl apply -k $(KUSTOMIZE_PROD)

.PHONY: prod-rollout
prod-rollout: ## Wait for the prod rollout
	kubectl -n $(NAMESPACE) rollout status deployment/factors-api --timeout=10m
	kubectl -n $(NAMESPACE) rollout status deployment/factors-watch --timeout=5m

.PHONY: prod-smoke
prod-smoke: ## Smoke test against prod
	GL_FACTORS_STAGING_URL=$(PROD_URL) python -m pytest tests/factors/smoke/test_staging.py -v

.PHONY: rollback-staging
rollback-staging: ## Roll back the latest staging deploy
	kubectl -n $(NAMESPACE) rollout undo deployment/factors-api

.PHONY: rollback-prod
rollback-prod: ## Roll back the latest prod deploy
	kubectl -n $(NAMESPACE) rollout undo deployment/factors-api

# ------------------------------------------------------------------ #
# Edition promotion (the commercial outcome)
# ------------------------------------------------------------------ #

.PHONY: promote-electricity
promote-electricity: ## Dry-run the v1 Electricity slice promotion
	python scripts/factors_v1_promote_electricity.py --dry-run

.PHONY: promote-electricity-live
promote-electricity-live: ## Record the live approval for Electricity slice
	@test -n "$$APPROVER" || (echo "Set APPROVER=<email>"; exit 1)
	python scripts/factors_v1_promote_electricity.py --live \
		--approver $$APPROVER \
		--changelog-reviewed \
		--methodology-signed \
		--legal-confirmed \
		$(if $(GOLD_EVAL),--gold-eval-precision $(GOLD_EVAL))

# ------------------------------------------------------------------ #
# Kong + ingress
# ------------------------------------------------------------------ #

.PHONY: kong-apply
kong-apply: ## Apply the Kong route config
	kubectl apply -f deployment/k8s/factors/kong/factors-api.yaml

# ------------------------------------------------------------------ #
# Local dev (docker-compose)
# ------------------------------------------------------------------ #

.PHONY: local-up
local-up: ## Bring up the local staging stack (docker-compose)
	docker compose -f deployment/docker/docker-compose.factors-staging.yml up -d

.PHONY: local-down
local-down: ## Tear down the local stack
	docker compose -f deployment/docker/docker-compose.factors-staging.yml down -v

.PHONY: local-logs
local-logs: ## Tail the local factors-api logs
	docker compose -f deployment/docker/docker-compose.factors-staging.yml logs -f factors-api
