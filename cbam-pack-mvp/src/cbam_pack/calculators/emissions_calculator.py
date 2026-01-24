"""
CBAM Emissions Calculator

Computes embedded emissions (direct + indirect) for CBAM import lines.
Supports both default factors and supplier-specific data.
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from cbam_pack.errors import ErrorLocation
from cbam_pack.factors import EmissionFactorLibrary, EmissionFactor
from cbam_pack.models import (
    ImportLineItem,
    EmissionsResult,
    AggregatedResult,
    Assumption,
    AssumptionType,
    MethodType,
    CBAMConfig,
    AggregationPolicy,
)
from cbam_pack.calculators.unit_normalizer import UnitNormalizer


@dataclass
class CalculationResult:
    """Result of emissions calculations for all lines."""
    line_results: list[EmissionsResult] = field(default_factory=list)
    aggregated_results: list[AggregatedResult] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)


class CBAMCalculator:
    """
    Calculator for CBAM embedded emissions.

    Implements the CBAM methodology for direct and indirect emissions
    using default factors or supplier-specific data.
    """

    # Precision settings
    INTERMEDIATE_PRECISION = Decimal("0.0001")  # 4 decimal places
    OUTPUT_PRECISION = Decimal("0.01")  # 2 decimal places

    def __init__(
        self,
        factor_library: Optional[EmissionFactorLibrary] = None,
        normalizer: Optional[UnitNormalizer] = None,
    ):
        """
        Initialize the calculator.

        Args:
            factor_library: Emission factor library to use
            normalizer: Unit normalizer to use
        """
        self.factor_library = factor_library or EmissionFactorLibrary()
        self.normalizer = normalizer or UnitNormalizer()
        self._assumptions: list[Assumption] = []

    def calculate_line_emissions(
        self,
        line: ImportLineItem,
    ) -> EmissionsResult:
        """
        Calculate emissions for a single import line.

        Args:
            line: The import line item

        Returns:
            EmissionsResult with calculated emissions
        """
        # Normalize quantity to tonnes
        quantity_tonnes = self.normalizer.normalize_to_tonnes(
            value=line.quantity,
            from_unit=line.unit.value,
            line_id=line.line_id,
        )

        # Get default factor
        factor = self.factor_library.get_factor(
            cn_code=line.cn_code,
            country=line.country_of_origin,
        )

        # Determine direct emissions
        if line.supplier_direct_emissions is not None:
            direct_factor = line.supplier_direct_emissions
            method_direct = MethodType.SUPPLIER_SPECIFIC
            factor_direct_ref = f"SUPPLIER-{line.supplier_id or 'UNKNOWN'}"
        else:
            direct_factor = factor.direct_emissions_factor
            method_direct = MethodType.DEFAULT
            factor_direct_ref = factor.factor_id

            # Record assumption
            self._record_assumption(
                assumption_type=AssumptionType.DEFAULT_FACTOR,
                description=f"Used default direct emission factor for {factor.product_type} from {factor.country_name}",
                rationale="No supplier-specific direct emissions data provided",
                line_id=line.line_id,
                factor_ref=factor.factor_id,
            )

        # Determine indirect emissions
        if line.supplier_indirect_emissions is not None:
            indirect_factor = line.supplier_indirect_emissions
            method_indirect = MethodType.SUPPLIER_SPECIFIC
            factor_indirect_ref = f"SUPPLIER-{line.supplier_id or 'UNKNOWN'}-INDIRECT"
        else:
            indirect_factor = factor.indirect_emissions_factor
            method_indirect = MethodType.DEFAULT
            factor_indirect_ref = factor.factor_id + "-INDIRECT"

            # Record assumption (only if not already recorded for direct)
            if method_direct != MethodType.DEFAULT:
                self._record_assumption(
                    assumption_type=AssumptionType.DEFAULT_FACTOR,
                    description=f"Used default indirect emission factor for {factor.product_type} from {factor.country_name}",
                    rationale="No supplier-specific indirect emissions data provided",
                    line_id=line.line_id,
                    factor_ref=factor.factor_id,
                )

        # Calculate emissions
        direct_emissions = (quantity_tonnes * direct_factor).quantize(
            self.INTERMEDIATE_PRECISION, rounding=ROUND_HALF_UP
        )
        indirect_emissions = (quantity_tonnes * indirect_factor).quantize(
            self.INTERMEDIATE_PRECISION, rounding=ROUND_HALF_UP
        )
        total_emissions = direct_emissions + indirect_emissions

        # Calculate intensity (emissions per tonne of product)
        emissions_intensity = (total_emissions / quantity_tonnes).quantize(
            self.INTERMEDIATE_PRECISION, rounding=ROUND_HALF_UP
        )

        return EmissionsResult(
            line_id=line.line_id,
            direct_emissions_tco2e=direct_emissions.quantize(
                self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
            ),
            indirect_emissions_tco2e=indirect_emissions.quantize(
                self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
            ),
            total_emissions_tco2e=total_emissions.quantize(
                self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
            ),
            emissions_intensity=emissions_intensity.quantize(
                self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
            ),
            method_direct=method_direct,
            method_indirect=method_indirect,
            factor_direct_ref=factor_direct_ref,
            factor_indirect_ref=factor_indirect_ref,
        )

    def calculate_all(
        self,
        lines: list[ImportLineItem],
        config: CBAMConfig,
    ) -> CalculationResult:
        """
        Calculate emissions for all import lines.

        Args:
            lines: List of import line items
            config: CBAM configuration

        Returns:
            CalculationResult with all results and statistics
        """
        self._assumptions.clear()
        self.normalizer.clear_log()

        line_results: list[EmissionsResult] = []

        for line in lines:
            result = self.calculate_line_emissions(line)
            line_results.append(result)

        # Aggregate results
        aggregated_results = self._aggregate_results(
            lines=lines,
            results=line_results,
            policy=config.settings.aggregation,
        )

        # Calculate statistics
        statistics = self._calculate_statistics(line_results)

        return CalculationResult(
            line_results=line_results,
            aggregated_results=aggregated_results,
            assumptions=self._assumptions.copy(),
            statistics=statistics,
        )

    def _aggregate_results(
        self,
        lines: list[ImportLineItem],
        results: list[EmissionsResult],
        policy: AggregationPolicy,
    ) -> list[AggregatedResult]:
        """Aggregate results by CN code and country."""
        if policy == AggregationPolicy.PRESERVE_DETAIL:
            # No aggregation
            return []

        # Build lookup for lines by line_id
        line_lookup = {line.line_id: line for line in lines}

        # Group by CN code + country
        groups: dict[tuple[str, str], list[tuple[ImportLineItem, EmissionsResult]]] = {}

        for result in results:
            line = line_lookup[result.line_id]
            key = (line.cn_code, line.country_of_origin)

            if key not in groups:
                groups[key] = []
            groups[key].append((line, result))

        # Aggregate each group
        aggregated: list[AggregatedResult] = []

        for (cn_code, country), items in sorted(groups.items()):
            total_quantity = Decimal("0")
            total_direct = Decimal("0")
            total_indirect = Decimal("0")
            methods_used = set()

            for line, result in items:
                # Get quantity in tonnes
                qty_tonnes = self.normalizer.normalize_to_tonnes(
                    line.quantity, line.unit.value
                )
                total_quantity += qty_tonnes
                total_direct += result.direct_emissions_tco2e
                total_indirect += result.indirect_emissions_tco2e

                if result.method_direct == MethodType.SUPPLIER_SPECIFIC:
                    methods_used.add("supplier_specific")
                else:
                    methods_used.add("default_values")

            total_emissions = total_direct + total_indirect
            weighted_intensity = (
                (total_emissions / total_quantity).quantize(
                    self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
                )
                if total_quantity > 0
                else Decimal("0")
            )

            # Determine method string
            if len(methods_used) == 1:
                method_used = list(methods_used)[0]
            else:
                method_used = "mixed"

            aggregated.append(
                AggregatedResult(
                    cn_code=cn_code,
                    country_of_origin=country,
                    total_quantity_tonnes=total_quantity.quantize(
                        self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
                    ),
                    total_direct_emissions_tco2e=total_direct.quantize(
                        self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
                    ),
                    total_indirect_emissions_tco2e=total_indirect.quantize(
                        self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
                    ),
                    total_emissions_tco2e=total_emissions.quantize(
                        self.OUTPUT_PRECISION, rounding=ROUND_HALF_UP
                    ),
                    weighted_intensity=weighted_intensity,
                    line_count=len(items),
                    method_used=method_used,
                )
            )

        return aggregated

    def _calculate_statistics(
        self,
        results: list[EmissionsResult],
    ) -> dict:
        """Calculate summary statistics."""
        total_direct = sum(r.direct_emissions_tco2e for r in results)
        total_indirect = sum(r.indirect_emissions_tco2e for r in results)
        total_emissions = sum(r.total_emissions_tco2e for r in results)

        lines_with_supplier_direct = sum(
            1 for r in results if r.method_direct == MethodType.SUPPLIER_SPECIFIC
        )
        lines_with_supplier_indirect = sum(
            1 for r in results if r.method_indirect == MethodType.SUPPLIER_SPECIFIC
        )
        lines_using_defaults = sum(
            1
            for r in results
            if r.method_direct == MethodType.DEFAULT
            and r.method_indirect == MethodType.DEFAULT
        )

        total_lines = len(results)
        default_usage_percent = (
            (lines_using_defaults / total_lines * 100) if total_lines > 0 else 0
        )

        return {
            "total_lines": total_lines,
            "total_direct_emissions_tco2e": float(total_direct),
            "total_indirect_emissions_tco2e": float(total_indirect),
            "total_emissions_tco2e": float(total_emissions),
            "lines_with_supplier_direct_data": lines_with_supplier_direct,
            "lines_with_supplier_indirect_data": lines_with_supplier_indirect,
            "lines_using_defaults": lines_using_defaults,
            "default_usage_percent": round(default_usage_percent, 1),
        }

    def _record_assumption(
        self,
        assumption_type: AssumptionType,
        description: str,
        rationale: str,
        line_id: str,
        factor_ref: Optional[str] = None,
    ) -> None:
        """Record an assumption made during calculation."""
        # Check if we already have this assumption for this line
        for assumption in self._assumptions:
            if (
                assumption.type == assumption_type
                and assumption.factor_ref == factor_ref
                and line_id in assumption.applies_to
            ):
                return  # Already recorded

        # Check if we can add to existing assumption
        for assumption in self._assumptions:
            if (
                assumption.type == assumption_type
                and assumption.factor_ref == factor_ref
                and assumption.description == description
            ):
                assumption.applies_to.append(line_id)
                return

        # Create new assumption
        assumption = Assumption(
            type=assumption_type,
            description=description,
            rationale=rationale,
            applies_to=[line_id],
            factor_ref=factor_ref,
        )
        self._assumptions.append(assumption)
