"""
Data Transformation Components
==============================

Reusable transformation components for ETL pipelines.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import re
import logging
import hashlib

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""

    def __init__(self, name: str = "BaseTransformer"):
        self.name = name
        self.records_processed = 0
        self.records_transformed = 0
        self.records_failed = 0
        self.errors: List[str] = []

    @abstractmethod
    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a single record.

        Args:
            record: Input record

        Returns:
            Transformed record or None if transformation fails
        """
        pass

    def transform_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a batch of records.

        Args:
            records: List of input records

        Returns:
            List of transformed records (excludes failed)
        """
        results = []
        for record in records:
            self.records_processed += 1
            try:
                transformed = self.transform(record)
                if transformed:
                    results.append(transformed)
                    self.records_transformed += 1
                else:
                    self.records_failed += 1
            except Exception as e:
                self.records_failed += 1
                self.errors.append(str(e))
                logger.warning(f"Transformation failed: {e}")
        return results

    def get_stats(self) -> Dict[str, int]:
        """Get transformation statistics."""
        return {
            "processed": self.records_processed,
            "transformed": self.records_transformed,
            "failed": self.records_failed,
        }


class CleaningTransformer(BaseTransformer):
    """
    Data cleaning transformer.

    Handles:
    - Whitespace trimming
    - Case normalization
    - Null handling
    - Data type coercion
    """

    def __init__(
        self,
        trim_whitespace: bool = True,
        normalize_case: Optional[str] = None,  # 'upper', 'lower', 'title'
        null_values: List[str] = None,
        strip_chars: str = None,
    ):
        super().__init__("CleaningTransformer")
        self.trim_whitespace = trim_whitespace
        self.normalize_case = normalize_case
        self.null_values = null_values or ['', 'null', 'NULL', 'None', 'N/A', 'n/a', '-']
        self.strip_chars = strip_chars

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply cleaning transformations."""
        cleaned = {}

        for key, value in record.items():
            cleaned_value = self._clean_value(value)
            cleaned[key] = cleaned_value

        return cleaned

    def _clean_value(self, value: Any) -> Any:
        """Clean a single value."""
        if value is None:
            return None

        if isinstance(value, str):
            # Trim whitespace
            if self.trim_whitespace:
                value = value.strip()
                if self.strip_chars:
                    value = value.strip(self.strip_chars)

            # Check for null values
            if value in self.null_values:
                return None

            # Normalize case
            if self.normalize_case == 'upper':
                value = value.upper()
            elif self.normalize_case == 'lower':
                value = value.lower()
            elif self.normalize_case == 'title':
                value = value.title()

        return value


class EnrichmentTransformer(BaseTransformer):
    """
    Data enrichment transformer.

    Handles:
    - Lookup table enrichment
    - Calculated fields
    - Default value injection
    """

    def __init__(
        self,
        lookups: Dict[str, Dict[str, Any]] = None,
        calculated_fields: Dict[str, Callable] = None,
        defaults: Dict[str, Any] = None,
    ):
        super().__init__("EnrichmentTransformer")
        self.lookups = lookups or {}
        self.calculated_fields = calculated_fields or {}
        self.defaults = defaults or {}

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply enrichment transformations."""
        enriched = record.copy()

        # Apply defaults for missing fields
        for field, default in self.defaults.items():
            if field not in enriched or enriched[field] is None:
                enriched[field] = default

        # Apply lookups
        for lookup_name, lookup_config in self.lookups.items():
            source_field = lookup_config.get('source_field')
            target_field = lookup_config.get('target_field')
            lookup_table = lookup_config.get('table', {})

            if source_field in enriched:
                source_value = enriched[source_field]
                if source_value in lookup_table:
                    enriched[target_field] = lookup_table[source_value]

        # Apply calculated fields
        for field_name, calc_func in self.calculated_fields.items():
            try:
                enriched[field_name] = calc_func(enriched)
            except Exception as e:
                logger.warning(f"Calculation failed for {field_name}: {e}")

        return enriched

    def add_lookup(
        self,
        name: str,
        source_field: str,
        target_field: str,
        table: Dict[str, Any]
    ) -> 'EnrichmentTransformer':
        """Add a lookup table."""
        self.lookups[name] = {
            'source_field': source_field,
            'target_field': target_field,
            'table': table,
        }
        return self


class NormalizationTransformer(BaseTransformer):
    """
    Data normalization transformer.

    Handles:
    - Unit standardization
    - Currency conversion
    - Code mapping
    """

    # Standard unit conversions
    UNIT_CONVERSIONS = {
        # Mass
        ('kg', 'g'): 1000,
        ('kg', 'tonne'): 0.001,
        ('kg', 't'): 0.001,
        ('tonne', 'kg'): 1000,
        ('t', 'kg'): 1000,
        ('lb', 'kg'): 0.453592,
        ('kg', 'lb'): 2.20462,

        # Energy
        ('kWh', 'MWh'): 0.001,
        ('MWh', 'kWh'): 1000,
        ('kWh', 'GJ'): 0.0036,
        ('GJ', 'kWh'): 277.778,
        ('MJ', 'kWh'): 0.277778,
        ('kWh', 'MJ'): 3.6,
        ('BTU', 'kWh'): 0.000293071,
        ('therm', 'kWh'): 29.3071,

        # Volume
        ('L', 'm3'): 0.001,
        ('m3', 'L'): 1000,
        ('gal', 'L'): 3.78541,
        ('L', 'gal'): 0.264172,

        # Emission factors
        ('lb/MWh', 'kg/kWh'): 0.000453592,
        ('kg/MWh', 'kg/kWh'): 0.001,
        ('g/kWh', 'kg/kWh'): 0.001,
        ('tCO2e/TJ', 'kgCO2e/kWh'): 0.0036,
    }

    def __init__(
        self,
        unit_mappings: Dict[str, str] = None,
        target_units: Dict[str, str] = None,
        custom_conversions: Dict[tuple, float] = None,
    ):
        super().__init__("NormalizationTransformer")
        self.unit_mappings = unit_mappings or {}  # Field -> unit field name
        self.target_units = target_units or {}  # Field -> target unit
        self.custom_conversions = custom_conversions or {}

        # Merge custom conversions with standard
        self.conversions = {**self.UNIT_CONVERSIONS, **self.custom_conversions}

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply normalization transformations."""
        normalized = record.copy()

        for value_field, unit_field in self.unit_mappings.items():
            if value_field not in normalized:
                continue

            current_value = normalized.get(value_field)
            current_unit = normalized.get(unit_field)
            target_unit = self.target_units.get(value_field)

            if current_value is None or current_unit is None:
                continue

            if target_unit and current_unit != target_unit:
                converted = self.convert_unit(current_value, current_unit, target_unit)
                if converted is not None:
                    normalized[value_field] = converted
                    normalized[unit_field] = target_unit

        return normalized

    def convert_unit(
        self,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str
    ) -> Optional[Union[float, Decimal]]:
        """Convert value between units."""
        if from_unit == to_unit:
            return value

        # Look up direct conversion
        conversion_key = (from_unit, to_unit)
        if conversion_key in self.conversions:
            factor = self.conversions[conversion_key]
            return float(value) * factor

        # Try reverse conversion
        reverse_key = (to_unit, from_unit)
        if reverse_key in self.conversions:
            factor = self.conversions[reverse_key]
            return float(value) / factor

        logger.warning(f"No conversion found: {from_unit} -> {to_unit}")
        return None


class AggregationTransformer(BaseTransformer):
    """
    Data aggregation transformer.

    Handles:
    - Group by aggregations
    - Rolling calculations
    - Percentile calculations
    """

    def __init__(
        self,
        group_by: List[str] = None,
        aggregations: Dict[str, str] = None,  # Field -> agg function (sum, avg, min, max, count)
    ):
        super().__init__("AggregationTransformer")
        self.group_by = group_by or []
        self.aggregations = aggregations or {}
        self._groups: Dict[str, List[Dict]] = {}

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add record to aggregation groups (doesn't transform individual records)."""
        if not self.group_by:
            return record

        # Create group key
        group_key = tuple(record.get(field) for field in self.group_by)
        group_key_str = str(group_key)

        if group_key_str not in self._groups:
            self._groups[group_key_str] = []

        self._groups[group_key_str].append(record)
        return record

    def get_aggregated(self) -> List[Dict[str, Any]]:
        """Get aggregated results."""
        results = []

        for group_key_str, records in self._groups.items():
            if not records:
                continue

            aggregated = {}

            # Copy group by fields
            for field in self.group_by:
                aggregated[field] = records[0].get(field)

            # Calculate aggregations
            for field, agg_func in self.aggregations.items():
                values = [r.get(field) for r in records if r.get(field) is not None]

                if not values:
                    aggregated[f"{field}_{agg_func}"] = None
                    continue

                try:
                    if agg_func == 'sum':
                        aggregated[f"{field}_{agg_func}"] = sum(values)
                    elif agg_func == 'avg':
                        aggregated[f"{field}_{agg_func}"] = sum(values) / len(values)
                    elif agg_func == 'min':
                        aggregated[f"{field}_{agg_func}"] = min(values)
                    elif agg_func == 'max':
                        aggregated[f"{field}_{agg_func}"] = max(values)
                    elif agg_func == 'count':
                        aggregated[f"{field}_{agg_func}"] = len(values)
                except Exception as e:
                    logger.warning(f"Aggregation failed: {e}")

            aggregated['_record_count'] = len(records)
            results.append(aggregated)

        return results


class ValidationTransformer(BaseTransformer):
    """
    Data validation transformer.

    Validates records against rules and optionally fixes or rejects invalid records.
    """

    def __init__(
        self,
        rules: List[Dict[str, Any]] = None,
        reject_invalid: bool = True,
        fix_on_fail: bool = False,
    ):
        super().__init__("ValidationTransformer")
        self.rules = rules or []
        self.reject_invalid = reject_invalid
        self.fix_on_fail = fix_on_fail
        self.rejected: List[Dict[str, Any]] = []

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and optionally transform record."""
        validated = record.copy()
        is_valid = True

        for rule in self.rules:
            field = rule.get('field')
            rule_type = rule.get('type')
            value = validated.get(field)

            passes = self._check_rule(value, rule)

            if not passes:
                if self.fix_on_fail and 'fix' in rule:
                    validated[field] = rule['fix']
                else:
                    is_valid = False
                    self.errors.append(f"Field '{field}' failed rule '{rule_type}'")

        if not is_valid:
            if self.reject_invalid:
                self.rejected.append(record)
                return None
            # Mark as invalid but return
            validated['_is_valid'] = False

        return validated

    def _check_rule(self, value: Any, rule: Dict[str, Any]) -> bool:
        """Check a single validation rule."""
        rule_type = rule.get('type')

        if rule_type == 'required':
            return value is not None and value != ''

        elif rule_type == 'type':
            expected_type = rule.get('expected')
            if expected_type == 'numeric':
                return isinstance(value, (int, float, Decimal))
            elif expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'date':
                return isinstance(value, (date, datetime))
            return True

        elif rule_type == 'range':
            if value is None:
                return rule.get('allow_null', False)
            min_val = rule.get('min')
            max_val = rule.get('max')
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True

        elif rule_type == 'pattern':
            if value is None:
                return rule.get('allow_null', False)
            pattern = rule.get('pattern')
            return bool(re.match(pattern, str(value)))

        elif rule_type == 'enum':
            allowed = rule.get('allowed', [])
            return value in allowed

        elif rule_type == 'custom':
            func = rule.get('func')
            if callable(func):
                return func(value)

        return True


class CompositeTransformer(BaseTransformer):
    """
    Composite transformer that chains multiple transformers.
    """

    def __init__(self, transformers: List[BaseTransformer] = None):
        super().__init__("CompositeTransformer")
        self.transformers = transformers or []

    def add(self, transformer: BaseTransformer) -> 'CompositeTransformer':
        """Add a transformer to the chain."""
        self.transformers.append(transformer)
        return self

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply all transformers in sequence."""
        current = record

        for transformer in self.transformers:
            if current is None:
                return None
            current = transformer.transform(current)

        return current

    def get_stats(self) -> Dict[str, Any]:
        """Get stats from all transformers."""
        stats = super().get_stats()
        stats['transformers'] = {
            t.name: t.get_stats() for t in self.transformers
        }
        return stats
