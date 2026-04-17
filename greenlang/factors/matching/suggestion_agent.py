# -*- coding: utf-8 -*-
"""
Factor suggestion agent (F045).

Recommends the best emission factor for a user's activity, with
confidence scoring, scope/boundary alignment verification, and
"Did you mean?" suggestions for common mismatches.

Enterprise tier: POST /api/v1/factors/suggest

Usage:
    agent = FactorSuggestionAgent(repo, edition_id)
    suggestion = agent.suggest(SuggestionRequest(
        activity_description="diesel combustion US",
        activity_amount=1000.0,
        activity_unit="gallons",
    ))
    print(suggestion.recommended.factor_id)
    print(suggestion.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from greenlang.factors.catalog_repository import FactorCatalogRepository
from greenlang.factors.matching.pipeline import HybridConfig, MatchRequest, run_match

logger = logging.getLogger(__name__)

# Common mismatches: user says X but probably means Y
COMMON_MISMATCHES = {
    "gas": [
        ("natural_gas", "Did you mean natural gas (pipeline gas)?"),
        ("gasoline", "Did you mean gasoline (petrol)?"),
    ],
    "petrol": [
        ("gasoline", "Petrol is the British English term for gasoline."),
    ],
    "fuel oil": [
        ("diesel", "Fuel oil / #2 fuel oil is typically diesel-grade distillate."),
    ],
    "power": [
        ("electricity", "Did you mean purchased electricity (Scope 2)?"),
    ],
    "energy": [
        ("electricity", "Did you mean electricity consumption?"),
        ("natural_gas", "Did you mean natural gas for heating?"),
    ],
    "heating": [
        ("natural_gas", "Heating is commonly natural gas. Specify if diesel/oil."),
    ],
}

# Scope/boundary alignment rules
SCOPE_BOUNDARY_RULES = {
    ("electricity", "2"): "Electricity is typically Scope 2 (indirect).",
    ("diesel", "1"): "Diesel combustion is Scope 1 (direct).",
    ("natural_gas", "1"): "Natural gas combustion is Scope 1 (direct).",
    ("coal", "1"): "Coal combustion is Scope 1 (direct).",
    ("gasoline", "1"): "Gasoline combustion is Scope 1 (direct).",
}


@dataclass
class SuggestionRequest:
    """Input for factor suggestion."""

    activity_description: str
    geography: Optional[str] = None
    scope: Optional[str] = None
    fuel_type: Optional[str] = None
    activity_amount: Optional[float] = None
    activity_unit: Optional[str] = None
    boundary: Optional[str] = None


@dataclass
class FactorCandidate:
    """A candidate emission factor with scoring details."""

    factor_id: str
    fuel_type: str
    geography: str
    scope: str
    unit: str
    co2e_per_unit: float
    match_score: float
    dqs_score: float
    source: str


@dataclass
class SuggestionResult:
    """Output from factor suggestion agent."""

    recommended: Optional[FactorCandidate] = None
    alternatives: List[FactorCandidate] = field(default_factory=list)
    confidence: float = 0.0
    confidence_level: str = "low"  # low, medium, high
    warnings: List[str] = field(default_factory=list)
    did_you_mean: List[str] = field(default_factory=list)
    scope_note: Optional[str] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "confidence": round(self.confidence, 4),
            "confidence_level": self.confidence_level,
            "explanation": self.explanation,
            "warnings": self.warnings,
            "did_you_mean": self.did_you_mean,
        }
        if self.recommended:
            d["recommended"] = {
                "factor_id": self.recommended.factor_id,
                "fuel_type": self.recommended.fuel_type,
                "geography": self.recommended.geography,
                "scope": self.recommended.scope,
                "unit": self.recommended.unit,
                "co2e_per_unit": self.recommended.co2e_per_unit,
                "match_score": self.recommended.match_score,
                "dqs_score": self.recommended.dqs_score,
                "source": self.recommended.source,
            }
        if self.alternatives:
            d["alternatives"] = [
                {
                    "factor_id": a.factor_id,
                    "fuel_type": a.fuel_type,
                    "geography": a.geography,
                    "scope": a.scope,
                    "co2e_per_unit": a.co2e_per_unit,
                    "match_score": a.match_score,
                }
                for a in self.alternatives
            ]
        if self.scope_note:
            d["scope_note"] = self.scope_note
        return d


class FactorSuggestionAgent:
    """
    Intelligent emission factor suggestion agent.

    Combines matching pipeline results with domain knowledge to
    recommend the best factor and flag potential issues.
    """

    def __init__(
        self,
        repo: FactorCatalogRepository,
        edition_id: str,
        *,
        semantic_search_fn: Optional[Callable] = None,
        hybrid_config: Optional[HybridConfig] = None,
    ):
        self._repo = repo
        self._edition_id = edition_id
        self._semantic_fn = semantic_search_fn
        self._hybrid_config = hybrid_config

    def suggest(self, request: SuggestionRequest) -> SuggestionResult:
        """
        Suggest the best emission factor for the given activity.

        Steps:
        1. Run matching pipeline to get candidates
        2. Build FactorCandidate objects with full details
        3. Check scope/boundary alignment
        4. Check for common mismatches ("did you mean?")
        5. Score confidence based on match quality
        6. Return recommendation with alternatives
        """
        result = SuggestionResult()

        # Step 1: Match
        match_req = MatchRequest(
            activity_description=request.activity_description,
            geography=request.geography,
            fuel_type=request.fuel_type,
            scope=request.scope,
            limit=10,
        )
        matches = run_match(
            self._repo,
            self._edition_id,
            match_req,
            semantic_search_fn=self._semantic_fn,
            hybrid_config=self._hybrid_config,
        )

        if not matches:
            result.explanation = "No matching factors found for the given activity."
            result.confidence = 0.0
            result.confidence_level = "low"
            return result

        # Step 2: Build candidates with full details
        candidates = self._build_candidates(matches)

        if not candidates:
            result.explanation = "Matching factors found but could not retrieve details."
            return result

        # Step 3: Recommend top candidate
        best = candidates[0]
        result.recommended = best
        result.alternatives = candidates[1:5]

        # Step 4: Scope/boundary alignment
        scope_note = self._check_scope_alignment(best, request)
        if scope_note:
            result.scope_note = scope_note

        # Step 5: Check for "did you mean?" mismatches
        did_you_mean = self._check_common_mismatches(request.activity_description, best)
        result.did_you_mean = did_you_mean

        # Step 6: Unit compatibility check
        unit_warnings = self._check_unit_compatibility(best, request)
        result.warnings.extend(unit_warnings)

        # Step 7: Geography mismatch warning
        geo_warnings = self._check_geography(best, request)
        result.warnings.extend(geo_warnings)

        # Step 8: Confidence scoring
        confidence, level = self._score_confidence(best, candidates, request)
        result.confidence = confidence
        result.confidence_level = level

        # Step 9: Explanation
        result.explanation = self._build_explanation(best, confidence, request)

        logger.info(
            "Suggestion: query=%r -> %s (confidence=%.2f %s)",
            request.activity_description, best.factor_id,
            confidence, level,
        )
        return result

    def _build_candidates(self, matches: List[Dict[str, Any]]) -> List[FactorCandidate]:
        """Fetch full details for matched factors."""
        candidates = []
        for m in matches:
            fid = m["factor_id"]
            factor = self._repo.get_factor(self._edition_id, fid)
            if not factor:
                continue
            candidates.append(FactorCandidate(
                factor_id=fid,
                fuel_type=factor.fuel_type,
                geography=factor.geography,
                scope=factor.scope.value,
                unit=factor.unit,
                co2e_per_unit=float(factor.gwp_100yr.co2e_total),
                match_score=m.get("score", 0.0),
                dqs_score=float(factor.dqs.overall_score),
                source=factor.provenance.source_org,
            ))
        return candidates

    def _check_scope_alignment(
        self,
        candidate: FactorCandidate,
        request: SuggestionRequest,
    ) -> Optional[str]:
        """Check if factor's scope aligns with expected scope."""
        if request.scope and candidate.scope != request.scope:
            return (
                f"Note: You requested Scope {request.scope} but the recommended "
                f"factor is Scope {candidate.scope}."
            )

        # Check domain knowledge rules
        key = (candidate.fuel_type, candidate.scope)
        if key in SCOPE_BOUNDARY_RULES:
            return SCOPE_BOUNDARY_RULES[key]

        return None

    def _check_common_mismatches(
        self,
        activity: str,
        candidate: FactorCandidate,
    ) -> List[str]:
        """Check for common terminology mismatches."""
        suggestions = []
        activity_lower = activity.lower()
        for keyword, options in COMMON_MISMATCHES.items():
            if keyword in activity_lower:
                for fuel_type, message in options:
                    if fuel_type != candidate.fuel_type:
                        suggestions.append(message)
        return suggestions

    def _check_unit_compatibility(
        self,
        candidate: FactorCandidate,
        request: SuggestionRequest,
    ) -> List[str]:
        """Warn if user's unit doesn't match factor's unit."""
        warnings = []
        if request.activity_unit and request.activity_unit.lower() != candidate.unit.lower():
            warnings.append(
                f"Unit mismatch: you specified '{request.activity_unit}' but "
                f"the factor uses '{candidate.unit}'. Conversion may be needed."
            )
        return warnings

    def _check_geography(
        self,
        candidate: FactorCandidate,
        request: SuggestionRequest,
    ) -> List[str]:
        """Warn if geography doesn't match."""
        warnings = []
        if request.geography and candidate.geography != request.geography:
            warnings.append(
                f"Geography mismatch: you specified '{request.geography}' but "
                f"the best match is for '{candidate.geography}'."
            )
        return warnings

    def _score_confidence(
        self,
        best: FactorCandidate,
        all_candidates: List[FactorCandidate],
        request: SuggestionRequest,
    ) -> Tuple[float, str]:
        """
        Score confidence from 0.0 to 1.0 based on:
        - Match score (primary)
        - DQS score
        - Gap between #1 and #2 candidates
        - Geography match
        """
        score = min(1.0, best.match_score)

        # DQS quality bonus (0-0.1)
        score += min(0.1, best.dqs_score / 50.0)

        # Gap bonus: if top candidate is much better than #2 (0-0.15)
        if len(all_candidates) >= 2:
            gap = best.match_score - all_candidates[1].match_score
            score += min(0.15, gap * 0.5)

        # Geography match bonus (0.05)
        if request.geography and best.geography == request.geography:
            score += 0.05

        # Fuel type explicit match bonus (0.05)
        if request.fuel_type and best.fuel_type.lower() == request.fuel_type.lower():
            score += 0.05

        confidence = max(0.0, min(1.0, score))

        if confidence >= 0.8:
            level = "high"
        elif confidence >= 0.5:
            level = "medium"
        else:
            level = "low"

        return confidence, level

    def _build_explanation(
        self,
        best: FactorCandidate,
        confidence: float,
        request: SuggestionRequest,
    ) -> str:
        """Build human-readable explanation of the suggestion."""
        parts = [
            f"Recommended factor: {best.factor_id}",
            f"Fuel type: {best.fuel_type}, Geography: {best.geography}, Scope: {best.scope}",
            f"CO2e per {best.unit}: {best.co2e_per_unit:.4f}",
            f"Data quality score: {best.dqs_score:.1f}/5.0",
            f"Source: {best.source}",
        ]
        return " | ".join(parts)
