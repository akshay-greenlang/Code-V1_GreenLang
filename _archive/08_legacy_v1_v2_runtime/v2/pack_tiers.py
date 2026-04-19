# -*- coding: utf-8 -*-
"""V2 pack tier lifecycle checks for CI and runtime enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import validate_v2_pack

TIERS = {"experimental", "candidate", "supported", "regulated-critical"}
PROMOTION_STATUS_BY_TIER = {
    "experimental": "pilot-approved",
    "candidate": "candidate-approved",
    "supported": "supported-approved",
    "regulated-critical": "regulated-approved",
}


@dataclass
class PackTierEvaluation:
    pack_slug: str
    tier: str
    ok: bool
    errors: list[str]
    warnings: list[str]


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at root")
    return loaded


def load_tier_registry(registry_path: Path) -> list[dict[str, Any]]:
    payload = _read_yaml(registry_path)
    pilot = payload.get("pilot_packs", [])
    if not isinstance(pilot, list):
        raise ValueError("v2 tier registry field pilot_packs must be a list")
    result: list[dict[str, Any]] = []
    for item in pilot:
        if isinstance(item, dict):
            result.append(item)
    return result


def validate_tier_registry(registry_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        entries = load_tier_registry(registry_path)
    except Exception as exc:
        return [str(exc)]

    seen: set[str] = set()
    for idx, entry in enumerate(entries):
        slug = str(entry.get("pack_slug", "")).strip()
        tier = str(entry.get("tier", "")).strip()
        owner = str(entry.get("owner_team", "")).strip()
        support = str(entry.get("support_channel", "")).strip()
        status = str(entry.get("promotion_status", "")).strip()
        if not slug:
            errors.append(f"entry[{idx}] pack_slug is required")
            continue
        if slug in seen:
            errors.append(f"duplicate pack_slug in tier registry: {slug}")
        seen.add(slug)
        if tier not in TIERS:
            errors.append(f"{slug}: unsupported tier '{tier}'")
        if not owner:
            errors.append(f"{slug}: owner_team is required")
        if not support:
            errors.append(f"{slug}: support_channel is required")
        expected_status = PROMOTION_STATUS_BY_TIER.get(tier)
        if expected_status and status != expected_status:
            errors.append(
                f"{slug}: promotion_status must be '{expected_status}' for tier '{tier}', got '{status}'"
            )
        evidence = entry.get("evidence", {})
        if not isinstance(evidence, dict):
            errors.append(f"{slug}: evidence must be a mapping")
            continue
        for key in ("docs_contract", "signed_artifact", "security_scan", "determinism_report"):
            if key not in evidence:
                errors.append(f"{slug}: evidence.{key} is required")
    return errors


def _entry_by_slug(entries: list[dict[str, Any]], slug: str) -> dict[str, Any] | None:
    for entry in entries:
        if str(entry.get("pack_slug", "")).strip() == slug:
            return entry
    return None


def evaluate_pack_tier(
    *,
    pack_slug: str,
    tier: str,
    owner_team: str,
    support_channel: str,
    signed: bool,
    signatures: list[str],
    evidence: dict[str, Any],
) -> PackTierEvaluation:
    errors: list[str] = []
    warnings: list[str] = []
    if tier not in TIERS:
        errors.append(f"unsupported tier: {tier}")
    if not owner_team.strip():
        errors.append("owner_team is required")
    if not support_channel.strip():
        errors.append("support_channel is required")

    docs_contract = bool(evidence.get("docs_contract", False))
    security_scan = bool(evidence.get("security_scan", False))
    determinism_report = bool(evidence.get("determinism_report", False))

    if tier in {"candidate", "supported", "regulated-critical"} and not docs_contract:
        errors.append("docs_contract evidence is required for candidate+ tiers")
    if tier in {"supported", "regulated-critical"}:
        if not signed:
            errors.append("supported and regulated-critical tiers require signed=true")
        if signed and not signatures:
            errors.append("signed tiers require at least one signature")
        if not security_scan:
            errors.append("security_scan evidence is required for supported+ tiers")
    if tier == "regulated-critical" and not determinism_report:
        errors.append("determinism_report evidence is required for regulated-critical tier")

    if tier == "experimental" and signed:
        warnings.append("experimental tier is signed; consider promoting to candidate/supported")

    return PackTierEvaluation(
        pack_slug=pack_slug,
        tier=tier,
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


def evaluate_pack_path(
    pack_yaml_path: Path,
    registry_path: Path | None = None,
) -> PackTierEvaluation:
    payload = _read_yaml(pack_yaml_path)
    slug = str(payload.get("name", pack_yaml_path.parent.name)).strip()
    contract_version = str(payload.get("contract_version", "")).strip()
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    security = payload.get("security", {}) if isinstance(payload.get("security", {}), dict) else {}

    # Keep legacy packs installable; Phase 2 tier lifecycle applies to v2 contracts.
    if contract_version != "2.0":
        return PackTierEvaluation(
            pack_slug=slug,
            tier=str(metadata.get("quality_tier", "legacy")),
            ok=True,
            errors=[],
            warnings=["legacy/non-v2 pack schema; tier lifecycle checks skipped"],
        )

    contract_finding = validate_v2_pack(pack_yaml_path)
    if not contract_finding.ok:
        return PackTierEvaluation(
            pack_slug=slug,
            tier=str(metadata.get("quality_tier", "unknown")),
            ok=False,
            errors=contract_finding.errors,
            warnings=[],
        )

    tier = str(metadata.get("quality_tier", "")).strip()
    owner = str(metadata.get("owner_team", "")).strip()
    support = str(metadata.get("support_channel", "")).strip()
    signed = bool(security.get("signed", False))
    signatures = security.get("signatures", [])
    signatures_list = [str(item) for item in signatures] if isinstance(signatures, list) else []

    evidence: dict[str, Any] = {
        "docs_contract": False,
        "signed_artifact": signed,
        "security_scan": False,
        "determinism_report": False,
    }
    if registry_path and registry_path.exists():
        entries = load_tier_registry(registry_path)
        entry = _entry_by_slug(entries, slug)
        if entry:
            entry_evidence = entry.get("evidence", {})
            if isinstance(entry_evidence, dict):
                evidence.update(entry_evidence)
            if not owner:
                owner = str(entry.get("owner_team", "")).strip()
            if not support:
                support = str(entry.get("support_channel", "")).strip()

    return evaluate_pack_tier(
        pack_slug=slug,
        tier=tier,
        owner_team=owner,
        support_channel=support,
        signed=signed,
        signatures=signatures_list,
        evidence=evidence,
    )
