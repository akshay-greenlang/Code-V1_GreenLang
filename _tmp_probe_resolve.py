import json

from greenlang.factors.service import FactorCatalogService
from greenlang.factors.api_endpoints import build_resolution_explain

svc = FactorCatalogService.from_environment()
edition = svc.repo.resolve_edition(None)
body = {
    "activity": "stationary combustion natural gas",
    "method_profile": "corporate_scope1",
    "jurisdiction": "EU",
}
try:
    payload = build_resolution_explain(svc.repo, edition, body)
except Exception as exc:
    print("RESOLVE ERROR:", type(exc).__name__, exc)
    raise
print(json.dumps(payload, indent=2, default=str)[:3000])
