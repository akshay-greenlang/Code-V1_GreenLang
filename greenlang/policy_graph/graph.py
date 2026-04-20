# -*- coding: utf-8 -*-
"""
PolicyGraph — v3 Intelligence Layer applicability API
======================================================

The *v3 core thesis*: given an entity, an activity, a jurisdiction, and a
reporting date, return which regulations apply and what they require.

::

    from greenlang.policy_graph import PolicyGraph

    pg = PolicyGraph()
    result = pg.applies_to(
        entity={"type": "corporation", "hq_country": "IN", "revenue_usd": 2_000_000_000,
                 "operates_in": ["EU", "US-CA"]},
        activity={"category": "cbam_covered_goods", "goods": "steel"},
        jurisdiction="EU",
        date="2026-08-15",
    )
    for reg in result.applicable_regulations:
        print(reg.name, reg.deadline, reg.required_factor_classes)

Design notes
------------

- Rules are stored as plain Python callables in the ``DEFAULT_RULES`` list.
  Each rule returns either ``None`` (does not apply) or a
  :class:`RegulationApplicability` instance.
- The rule set is pluggable: applications can register additional rules
  loaded from YAML files in ``packs/eu-compliance/PACK-*/rules/`` via
  :meth:`PolicyGraph.register_rule_file`.  That loader is deliberately
  conservative: it reads key-value metadata, *not* executable logic, and
  synthesises a callable rule from the metadata.  Callers that need true
  Rego-style logic should continue to route through
  ``greenlang.policy_graph.enforcer.PolicyEngine`` (OPA-backed).

This is Phase 2.4 of the FY27 plan.
"""
from __future__ import annotations

import logging
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import Field

from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class RegulationApplicability(GreenLangBase):
    """A single regulation's applicability verdict."""

    name: str = Field(..., description="Short regulation identifier (e.g. 'CBAM', 'CSRD', 'SB-253')")
    full_name: str = Field(..., description="Full regulation name")
    jurisdiction: str = Field(..., description="Jurisdiction identifier (e.g. 'EU', 'US-CA')")
    deadline: Optional[str] = Field(
        default=None,
        description="ISO date of the next reporting deadline, or None if rolling",
    )
    required_factor_classes: List[str] = Field(
        default_factory=list,
        description="Emission-factor coverage labels required (e.g. 'Certified').",
    )
    rationale: str = Field(
        ...,
        description="Plain-text explanation of why the regulation applies.",
    )


class ApplicabilityResult(GreenLangBase):
    """Full verdict of ``PolicyGraph.applies_to()``."""

    applicable_regulations: List[RegulationApplicability] = Field(
        default_factory=list,
        description="Regulations that apply, sorted by name.",
    )
    evaluated_at: str = Field(
        ...,
        description="ISO-8601 timestamp of evaluation (UTC).",
    )
    entity_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Salient entity attributes considered during evaluation.",
    )
    evaluation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Inputs fed to each rule (activity, jurisdiction, date).",
    )


# ---------------------------------------------------------------------------
# Rule type + default rule set
# ---------------------------------------------------------------------------


# Rules take (entity, activity, jurisdiction, date) and return either
# None (does not apply) or a RegulationApplicability instance.
Rule = Callable[
    [Dict[str, Any], Dict[str, Any], str, _date],
    Optional[RegulationApplicability],
]


def _coerce_date(value: Union[str, _date, datetime]) -> _date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, _date):
        return value
    return _date.fromisoformat(str(value))


def _operates_in(entity: Dict[str, Any], jurisdiction: str) -> bool:
    """True if ``entity`` explicitly operates in ``jurisdiction``."""
    hq = (entity.get("hq_country") or "").upper()
    ops = {str(j).upper() for j in (entity.get("operates_in") or [])}
    return jurisdiction.upper() in ops or hq == jurisdiction.upper()


def _rule_cbam(entity, activity, jurisdiction, date) -> Optional[RegulationApplicability]:
    """CBAM applies to imports of covered goods into the EU."""
    activity_category = str(activity.get("category") or "").lower()
    if activity_category != "cbam_covered_goods":
        return None
    if jurisdiction.upper() != "EU" and not _operates_in(entity, "EU"):
        return None
    # Definitive period began 2026-01-01.
    if date < _date(2026, 1, 1):
        return None
    # Next reporting deadline is the end of the current quarter.
    month = date.month
    if month <= 3:
        deadline = _date(date.year, 4, 30)
    elif month <= 6:
        deadline = _date(date.year, 7, 31)
    elif month <= 9:
        deadline = _date(date.year, 10, 31)
    else:
        deadline = _date(date.year + 1, 1, 31)
    return RegulationApplicability(
        name="CBAM",
        full_name="EU Carbon Border Adjustment Mechanism",
        jurisdiction="EU",
        deadline=deadline.isoformat(),
        required_factor_classes=["Certified"],
        rationale=(
            "Entity imports CBAM-covered goods ("
            f"{activity.get('goods', 'unspecified')}) into the EU; CBAM "
            "definitive period is live as of 2026-01-01."
        ),
    )


def _rule_csrd(entity, activity, jurisdiction, date) -> Optional[RegulationApplicability]:
    """CSRD applies to large EU-connected undertakings (employees/turnover thresholds)."""
    if jurisdiction.upper() != "EU" and not _operates_in(entity, "EU"):
        return None
    employees = int(entity.get("employees") or 0)
    turnover_m_eur = float(entity.get("turnover_m_eur") or 0.0)
    balance_sheet_m_eur = float(entity.get("balance_sheet_m_eur") or 0.0)
    # Thresholds broadly aligned with CSRD cascade (simplified).
    is_large = (
        employees >= 250
        or turnover_m_eur >= 50.0
        or balance_sheet_m_eur >= 25.0
    )
    if not is_large:
        return None
    # First reporting year is FY24 for the first wave; default to 2025-04-30
    # as the nearest horizon, rolling forward each year.
    deadline_year = date.year if date < _date(date.year, 4, 30) else date.year + 1
    deadline = _date(deadline_year, 4, 30)
    return RegulationApplicability(
        name="CSRD",
        full_name="EU Corporate Sustainability Reporting Directive",
        jurisdiction="EU",
        deadline=deadline.isoformat(),
        required_factor_classes=["Certified", "Preview"],
        rationale=(
            "Entity operates in the EU and meets CSRD size thresholds "
            f"(employees={employees}, turnover_m_eur={turnover_m_eur}, "
            f"balance_sheet_m_eur={balance_sheet_m_eur})."
        ),
    )


def _rule_sb253(entity, activity, jurisdiction, date) -> Optional[RegulationApplicability]:
    """California SB 253 applies to entities doing business in CA with >= $1B revenue."""
    if jurisdiction.upper() not in {"US-CA", "US", "CA"}:
        return None
    if not _operates_in(entity, "US-CA") and not _operates_in(entity, "CA"):
        return None
    revenue_usd = float(entity.get("revenue_usd") or 0.0)
    if revenue_usd < 1_000_000_000:
        return None
    # Scope 1+2 first disclosure deadline: 2026-08-10.
    # Scope 3 deadline: 2027 (exact TBD in regulation).
    if date <= _date(2026, 8, 10):
        deadline = _date(2026, 8, 10)
        required = ["Certified"]
    else:
        deadline = _date(2027, 8, 10)
        required = ["Certified", "Preview"]
    return RegulationApplicability(
        name="SB-253",
        full_name="California Climate Corporate Data Accountability Act (SB 253)",
        jurisdiction="US-CA",
        deadline=deadline.isoformat(),
        required_factor_classes=required,
        rationale=(
            "Entity does business in California with revenue "
            f"${revenue_usd/1e9:.2f}B >= $1B threshold."
        ),
    )


def _rule_tcfd(entity, activity, jurisdiction, date) -> Optional[RegulationApplicability]:
    """TCFD is mandated for UK premium-listed companies; elsewhere voluntary."""
    hq = (entity.get("hq_country") or "").upper()
    if hq != "GB" and not _operates_in(entity, "GB"):
        return None
    return RegulationApplicability(
        name="TCFD",
        full_name="Task Force on Climate-related Financial Disclosures",
        jurisdiction="UK",
        deadline=None,
        required_factor_classes=["Certified", "Preview"],
        rationale="Entity operates in the UK where TCFD-aligned disclosure is mandated for large listed companies.",
    )


def _rule_ghg_protocol(entity, activity, jurisdiction, date) -> Optional[RegulationApplicability]:
    """GHG Protocol is the universal baseline methodology — always applicable."""
    return RegulationApplicability(
        name="GHG-Protocol",
        full_name="GHG Protocol Corporate Accounting and Reporting Standard",
        jurisdiction="GLOBAL",
        deadline=None,
        required_factor_classes=["Certified"],
        rationale="GHG Protocol is the universal baseline for corporate emissions inventories.",
    )


DEFAULT_RULES: List[Rule] = [
    _rule_cbam,
    _rule_csrd,
    _rule_sb253,
    _rule_tcfd,
    _rule_ghg_protocol,
]


# ---------------------------------------------------------------------------
# PolicyGraph
# ---------------------------------------------------------------------------


class PolicyGraph:
    """v3 Policy Graph: regulation applicability reasoning.

    Example::

        from greenlang.policy_graph import PolicyGraph

        pg = PolicyGraph()
        result = pg.applies_to(
            entity={"type": "corporation", "hq_country": "IN",
                    "operates_in": ["EU"], "employees": 500},
            activity={"category": "cbam_covered_goods", "goods": "steel"},
            jurisdiction="EU",
            date="2026-06-01",
        )
    """

    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self._rules: List[Rule] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self._pack_rules: List[Rule] = []
        logger.info("PolicyGraph initialised with %d default rule(s)", len(self._rules))

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def applies_to(
        self,
        entity: Dict[str, Any],
        activity: Dict[str, Any],
        jurisdiction: str,
        date: Union[str, _date, datetime],
    ) -> ApplicabilityResult:
        """Return the set of regulations that apply to ``(entity, activity, jurisdiction, date)``.

        Args:
            entity: Dict of entity attributes.  Recognised keys include
                ``type``, ``hq_country``, ``operates_in`` (list),
                ``employees``, ``turnover_m_eur``, ``balance_sheet_m_eur``,
                ``revenue_usd``.  Unknown keys are ignored.
            activity: Dict of activity attributes.  Recognised keys
                include ``category`` (e.g. ``'cbam_covered_goods'``),
                ``goods`` / ``fuel_type`` / ``scope``.
            jurisdiction: Upper-case jurisdiction identifier (``"EU"``,
                ``"US"``, ``"US-CA"``, ``"GB"``, ``"GLOBAL"``).
            date: ISO date string, ``datetime.date``, or ``datetime`` —
                the reporting period end used for deadline resolution.
        """
        d = _coerce_date(date)
        verdicts: List[RegulationApplicability] = []
        for rule in self._rules + self._pack_rules:
            try:
                verdict = rule(entity, activity, jurisdiction, d)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Rule %s raised %r; skipping", rule, exc)
                continue
            if verdict is not None:
                verdicts.append(verdict)

        verdicts.sort(key=lambda v: v.name)

        return ApplicabilityResult(
            applicable_regulations=verdicts,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            entity_summary={
                k: entity.get(k)
                for k in (
                    "type", "hq_country", "operates_in", "employees",
                    "turnover_m_eur", "balance_sheet_m_eur", "revenue_usd",
                )
                if entity.get(k) is not None
            },
            evaluation_context={
                "activity": activity,
                "jurisdiction": jurisdiction,
                "date": d.isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Introspection + extension
    # ------------------------------------------------------------------

    def list_rules(self) -> List[str]:
        """Return human-readable identifiers of all registered rules."""
        names = [getattr(r, "__name__", repr(r)) for r in self._rules]
        names.extend(getattr(r, "__name__", repr(r)) for r in self._pack_rules)
        return names

    def register_rule(self, rule: Rule) -> None:
        """Append a custom rule callable to the pack rule set."""
        self._pack_rules.append(rule)
        logger.info("Registered custom PolicyGraph rule: %s", getattr(rule, "__name__", rule))

    def register_rule_file(self, path: Union[str, Path]) -> int:
        """Load declarative rules from a YAML file.

        The YAML file is expected to contain a top-level ``rules:`` list
        where each entry has ``name``, ``full_name``, ``jurisdiction``
        (list of allowed jurisdictions), ``deadline`` (optional ISO date),
        ``required_factor_classes`` (list), ``rationale`` template, and
        ``when`` (dict of simple equality predicates on entity/activity).

        Synthesises one callable per entry and registers it.  Returns
        the number of rules loaded.

        Note: this loader is deliberately narrow.  Anything more complex
        (arithmetic thresholds, boolean combinations) should live in
        Python code next to the native ``DEFAULT_RULES`` or in an OPA
        bundle evaluated via :class:`PolicyEngine`.
        """
        import yaml  # lazy import to avoid hard dep at module load

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Rule file not found: {p}")
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rules_raw = doc.get("rules") or []
        added = 0
        for entry in rules_raw:
            rule = _compile_yaml_rule(entry)
            self._pack_rules.append(rule)
            added += 1
        logger.info("Loaded %d rule(s) from %s", added, p)
        return added


def _compile_yaml_rule(entry: Dict[str, Any]) -> Rule:
    """Turn a YAML rule entry into a Rule callable."""
    name = entry["name"]
    full_name = entry.get("full_name", name)
    allowed_jurisdictions = {
        j.upper() for j in (entry.get("jurisdiction") or [])
    }
    deadline = entry.get("deadline")
    required = list(entry.get("required_factor_classes") or [])
    rationale_template = entry.get(
        "rationale",
        f"Rule {name} matched on declarative predicates.",
    )
    predicates = entry.get("when") or {}

    def _rule_fn(
        entity: Dict[str, Any],
        activity: Dict[str, Any],
        jurisdiction: str,
        date: _date,
    ) -> Optional[RegulationApplicability]:
        if allowed_jurisdictions and jurisdiction.upper() not in allowed_jurisdictions:
            return None
        for key, expected in (predicates.get("entity") or {}).items():
            if entity.get(key) != expected:
                return None
        for key, expected in (predicates.get("activity") or {}).items():
            if activity.get(key) != expected:
                return None
        return RegulationApplicability(
            name=name,
            full_name=full_name,
            jurisdiction=jurisdiction,
            deadline=deadline,
            required_factor_classes=required,
            rationale=rationale_template,
        )

    _rule_fn.__name__ = f"yaml_rule_{name}"
    return _rule_fn
