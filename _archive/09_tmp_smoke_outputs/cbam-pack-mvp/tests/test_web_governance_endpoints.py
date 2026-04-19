from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_governance_pack_tier_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/governance/pack-tiers")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("packs"), list)
    assert payload["packs"]
    assert "pack_slug" in payload["packs"][0]


def test_governance_agent_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/governance/agents")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("agents"), list)
    assert payload["agents"]
    assert "agent_id" in payload["agents"][0]


def test_governance_policy_bundle_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/governance/policy-bundles")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("bundles"), list)
    assert payload["bundles"]
    assert "bundle" in payload["bundles"][0]
