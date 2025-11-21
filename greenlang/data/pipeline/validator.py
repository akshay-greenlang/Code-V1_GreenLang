# -*- coding: utf-8 -*-
"""
Comprehensive Data Validation Framework

Multi-layer validation system for emission factors:
- URI accessibility checks
- Emission factor range validation
- Source organization verification
- Date freshness checks
- Unit consistency validation
- Schema compliance
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import httpx
import asyncio
import yaml
import re
from pathlib import Path
import logging

from .models import (
from greenlang.determinism import DeterministicClock
    ValidationResult,
    URIAccessibilityCheck,
    FactorRangeCheck,
    DataQualityTier
)

logger = logging.getLogger(__name__)


class ValidationRule(ABC):
    """
    Abstract base class for validation rules.

    Each validation rule implements specific checks and scoring.
    """

    def __init__(self, name: str, weight: float = 1.0, required: bool = True):
        """
        Initialize validation rule.

        Args:
            name: Rule name
            weight: Weight for quality score calculation (0-1)
            required: Whether rule must pass for overall validation
        """
        self.name = name
        self.weight = weight
        self.required = required

    @abstractmethod
    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate factor data.

        Args:
            factor_data: Emission factor data dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        pass

    def calculate_score(self, passed: bool) -> float:
        """Calculate weighted score for this rule."""
        return self.weight * 100 if passed else 0.0


class URIValidator(ValidationRule):
    """
    Validate URI accessibility.

    Checks that source URIs are accessible and return valid responses.
    """

    def __init__(self, timeout: int = 10, check_ssl: bool = True):
        super().__init__(name="URI Accessibility", weight=0.15, required=False)
        self.timeout = timeout
        self.check_ssl = check_ssl
        self.client = None

    async def _init_client(self):
        """Initialize HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.check_ssl,
                follow_redirects=True
            )

    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate URI accessibility."""
        errors = []
        warnings = []

        # Get URI from factor data
        uri = factor_data.get('uri') or factor_data.get('source_uri')

        if not uri:
            errors.append(f"Missing URI for factor {factor_data.get('name', 'unknown')}")
            return False, errors, warnings

        # Validate URI format
        if not self._is_valid_uri_format(uri):
            errors.append(f"Invalid URI format: {uri}")
            return False, errors, warnings

        # Check accessibility
        try:
            await self._init_client()

            start_time = DeterministicClock.now()
            response = await self.client.head(uri, timeout=self.timeout)
            response_time = (DeterministicClock.now() - start_time).total_seconds() * 1000

            if response.status_code == 200:
                if response_time > 5000:  # Slow response
                    warnings.append(f"Slow URI response: {response_time:.0f}ms for {uri}")
                return True, errors, warnings

            elif response.status_code in [301, 302, 307, 308]:
                warnings.append(f"URI redirects (status {response.status_code}): {uri}")
                return True, errors, warnings

            elif response.status_code == 403:
                warnings.append(f"URI forbidden (403) but may be valid: {uri}")
                return True, errors, warnings

            elif response.status_code == 405:
                # HEAD not allowed, try GET
                response = await self.client.get(uri, timeout=self.timeout)
                if response.status_code == 200:
                    return True, errors, warnings

            else:
                errors.append(f"URI not accessible (status {response.status_code}): {uri}")
                return False, errors, warnings

        except httpx.TimeoutException:
            warnings.append(f"URI timeout after {self.timeout}s: {uri}")
            return True, errors, warnings  # Timeout is warning, not error

        except httpx.RequestError as e:
            errors.append(f"URI request failed: {uri} - {str(e)}")
            return False, errors, warnings

        except Exception as e:
            logger.error(f"Unexpected error validating URI {uri}: {e}")
            warnings.append(f"URI validation error: {str(e)}")
            return True, errors, warnings

    def _is_valid_uri_format(self, uri: str) -> bool:
        """Check if URI has valid format."""
        if not uri:
            return False

        # Basic URI pattern
        uri_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$',  # path
            re.IGNORECASE
        )

        return bool(uri_pattern.match(uri))

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()


class DateFreshnessValidator(ValidationRule):
    """
    Validate data freshness.

    Checks that emission factors are not older than specified threshold.
    """

    def __init__(self, max_age_years: int = 3, warning_age_years: int = 2):
        super().__init__(name="Date Freshness", weight=0.20, required=False)
        self.max_age_years = max_age_years
        self.warning_age_years = warning_age_years

    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate data freshness."""
        errors = []
        warnings = []

        # Get last updated date
        last_updated = factor_data.get('last_updated')

        if not last_updated:
            errors.append(f"Missing last_updated date for factor {factor_data.get('name', 'unknown')}")
            return False, errors, warnings

        # Parse date
        try:
            if isinstance(last_updated, str):
                update_date = datetime.strptime(last_updated, '%Y-%m-%d')
            elif isinstance(last_updated, datetime):
                update_date = last_updated
            else:
                errors.append(f"Invalid date format: {last_updated}")
                return False, errors, warnings

            # Calculate age
            age_days = (DeterministicClock.now() - update_date).days
            age_years = age_days / 365.25

            # Check against thresholds
            if age_years > self.max_age_years:
                errors.append(
                    f"Data is stale ({age_years:.1f} years old, max {self.max_age_years} years)"
                )
                return False, errors, warnings

            elif age_years > self.warning_age_years:
                warnings.append(
                    f"Data is aging ({age_years:.1f} years old, consider updating)"
                )

            return True, errors, warnings

        except ValueError as e:
            errors.append(f"Date parsing error: {str(e)}")
            return False, errors, warnings


class RangeValidator(ValidationRule):
    """
    Validate emission factor ranges.

    Checks that emission factor values are within reasonable ranges
    for their category.
    """

    def __init__(self):
        super().__init__(name="Range Validation", weight=0.25, required=True)

        # Define reasonable ranges for common categories (kg CO2e per unit)
        self.ranges = {
            'fuels': {
                'natural_gas': (0.05, 0.3),  # per kWh
                'diesel': (2.0, 4.0),  # per liter
                'gasoline': (2.0, 3.5),  # per liter
                'coal': (1.5, 3.0),  # per kg
                'lpg': (1.5, 3.5),  # per liter
            },
            'electricity': {
                'grid': (0.1, 1.2),  # per kWh
                'renewable': (0.0, 0.1),  # per kWh
            },
            'transportation': {
                'flight': (0.05, 0.5),  # per passenger-km
                'car': (0.05, 0.3),  # per km
                'truck': (0.3, 1.5),  # per ton-km
            },
            'materials': {
                'steel': (1.5, 3.5),  # per kg
                'cement': (0.5, 1.2),  # per kg
                'plastic': (1.5, 4.0),  # per kg
            }
        }

    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate emission factor ranges."""
        errors = []
        warnings = []

        # Extract emission factor value
        value = self._extract_primary_value(factor_data)

        if value is None:
            errors.append("No emission factor value found")
            return False, errors, warnings

        if value <= 0:
            errors.append(f"Invalid emission factor value: {value} (must be > 0)")
            return False, errors, warnings

        # Check against category ranges
        category = factor_data.get('category', '').lower()
        subcategory = factor_data.get('subcategory', '').lower()

        # Get expected range
        expected_range = self._get_expected_range(category, subcategory)

        if expected_range:
            min_val, max_val = expected_range

            # Check if value is way out of range (10x deviation)
            if value < min_val / 10:
                errors.append(
                    f"Value {value} is extremely low for {category}/{subcategory} "
                    f"(expected range: {min_val}-{max_val})"
                )
                return False, errors, warnings

            elif value > max_val * 10:
                errors.append(
                    f"Value {value} is extremely high for {category}/{subcategory} "
                    f"(expected range: {min_val}-{max_val})"
                )
                return False, errors, warnings

            # Warning if outside normal range but within 10x
            elif value < min_val or value > max_val:
                warnings.append(
                    f"Value {value} is outside typical range for {category}/{subcategory} "
                    f"(expected: {min_val}-{max_val}). Please verify."
                )

        # Check for unrealistic values (general bounds)
        if value > 10000:  # Very high emission factor
            errors.append(f"Unrealistically high emission factor: {value}")
            return False, errors, warnings

        return True, errors, warnings

    def _extract_primary_value(self, factor_data: Dict[str, Any]) -> Optional[float]:
        """Extract primary emission factor value from data."""
        # Try various field names
        for key in factor_data.keys():
            if key.startswith('emission_factor_kg_co2e_per_'):
                value = factor_data[key]
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)

        # Fallback to generic fields
        for key in ['emission_factor', 'emission_factor_value']:
            if key in factor_data:
                value = factor_data[key]
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)

        return None

    def _get_expected_range(self, category: str, subcategory: str) -> Optional[Tuple[float, float]]:
        """Get expected range for category/subcategory."""
        if category in self.ranges:
            category_ranges = self.ranges[category]

            # Try exact subcategory match
            for key, range_val in category_ranges.items():
                if key in subcategory or subcategory in key:
                    return range_val

        return None


class UnitValidator(ValidationRule):
    """
    Validate unit consistency.

    Checks that units are valid and consistent with factor type.
    """

    def __init__(self):
        super().__init__(name="Unit Validation", weight=0.15, required=True)

        # Valid units for emission factors
        self.valid_units = {
            'energy': ['kwh', 'mwh', 'gwh', 'mj', 'gj', 'mmbtu', 'therm', 'btu'],
            'mass': ['kg', 'ton', 'tonne', 'mt', 'g', 'lb'],
            'volume': ['liter', 'l', 'gallon', 'gal', 'm3', 'scf', 'mcf'],
            'distance': ['km', 'mile', 'mi', 'passenger_km', 'ton_km'],
            'area': ['sqft', 'sqm', 'm2', 'acre', 'hectare'],
            'generic': ['unit', 'item', 'piece']
        }

    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate units."""
        errors = []
        warnings = []

        # Extract units from field names
        units_found = set()

        for key in factor_data.keys():
            if key.startswith('emission_factor_kg_co2e_per_'):
                unit = key.replace('emission_factor_kg_co2e_per_', '')
                units_found.add(unit.lower())

        # Also check explicit unit field
        if 'unit' in factor_data:
            units_found.add(factor_data['unit'].lower())

        if not units_found:
            errors.append("No units specified for emission factor")
            return False, errors, warnings

        # Validate each unit
        for unit in units_found:
            if not self._is_valid_unit(unit):
                warnings.append(f"Unusual or non-standard unit: {unit}")

        return True, errors, warnings

    def _is_valid_unit(self, unit: str) -> bool:
        """Check if unit is in valid units list."""
        unit_lower = unit.lower()

        for category_units in self.valid_units.values():
            if unit_lower in category_units:
                return True

        return False


class SourceValidator(ValidationRule):
    """
    Validate source organization credibility.

    Checks that source organizations are recognized and credible.
    """

    def __init__(self):
        super().__init__(name="Source Validation", weight=0.10, required=False)

        # Recognized emission factor sources
        self.recognized_sources = {
            'EPA', 'IPCC', 'DEFRA', 'IEA', 'GHG Protocol', 'ISO',
            'DOE', 'EIA', 'NREL', 'Ecoinvent', 'EXIOBASE',
            'UK Government', 'European Commission', 'UNFCCC',
            'Carbon Trust', 'CDP', 'Science Based Targets',
            'World Resources Institute', 'WRI'
        }

    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate source organization."""
        errors = []
        warnings = []

        source = factor_data.get('source')

        if not source:
            warnings.append("No source organization specified")
            return True, errors, warnings

        # Check if source is recognized
        is_recognized = any(
            recognized.lower() in source.lower()
            for recognized in self.recognized_sources
        )

        if not is_recognized:
            warnings.append(
                f"Unrecognized source organization: {source}. "
                "Please verify credibility."
            )

        return True, errors, warnings


class SchemaValidator(ValidationRule):
    """
    Validate against expected schema.

    Checks that required fields are present and have correct types.
    """

    def __init__(self):
        super().__init__(name="Schema Validation", weight=0.15, required=True)

        # Required fields
        self.required_fields = [
            'name',
            'scope',
            'source',
            'uri',
            'last_updated'
        ]

        # Optional but recommended fields
        self.recommended_fields = [
            'standard',
            'data_quality',
            'uncertainty',
            'geographic_scope'
        ]

    async def validate(self, factor_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate schema compliance."""
        errors = []
        warnings = []

        # Check required fields
        for field in self.required_fields:
            if field not in factor_data or not factor_data[field]:
                errors.append(f"Missing required field: {field}")

        # Check recommended fields
        for field in self.recommended_fields:
            if field not in factor_data or not factor_data[field]:
                warnings.append(f"Missing recommended field: {field}")

        # Validate field types
        if 'name' in factor_data and not isinstance(factor_data['name'], str):
            errors.append("Field 'name' must be a string")

        if 'scope' in factor_data and not isinstance(factor_data['scope'], str):
            errors.append("Field 'scope' must be a string")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings


class EmissionFactorValidator:
    """
    Comprehensive emission factor validator.

    Orchestrates multiple validation rules and produces quality scores.
    """

    def __init__(self):
        """Initialize validator with all validation rules."""
        self.rules: List[ValidationRule] = [
            SchemaValidator(),
            RangeValidator(),
            UnitValidator(),
            DateFreshnessValidator(max_age_years=3, warning_age_years=2),
            SourceValidator(),
            URIValidator(timeout=5)
        ]

    async def validate_factor(
        self,
        factor_id: str,
        factor_data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate a single emission factor.

        Args:
            factor_id: Factor identifier
            factor_data: Factor data dictionary

        Returns:
            ValidationResult with detailed results
        """
        start_time = DeterministicClock.now()

        all_errors = []
        all_warnings = []
        rules_passed = []
        rules_failed = []
        rules_warnings = []

        # Run all validation rules
        for rule in self.rules:
            try:
                is_valid, errors, warnings = await rule.validate(factor_data)

                if is_valid:
                    rules_passed.append(rule.name)
                else:
                    rules_failed.append(rule.name)

                if warnings:
                    rules_warnings.append(rule.name)

                # Collect errors and warnings
                for error in errors:
                    all_errors.append({
                        'factor_id': factor_id,
                        'rule': rule.name,
                        'severity': 'error',
                        'message': error
                    })

                for warning in warnings:
                    all_warnings.append({
                        'factor_id': factor_id,
                        'rule': rule.name,
                        'severity': 'warning',
                        'message': warning
                    })

            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                all_errors.append({
                    'factor_id': factor_id,
                    'rule': rule.name,
                    'severity': 'error',
                    'message': f"Validation exception: {str(e)}"
                })
                rules_failed.append(rule.name)

        # Calculate quality score
        quality_score = self._calculate_quality_score(rules_passed, rules_failed, rules_warnings)

        # Overall validation result
        is_valid = len(rules_failed) == 0 or not any(
            rule.required for rule in self.rules if rule.name in rules_failed
        )

        duration = (DeterministicClock.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            validation_id=f"val_{factor_id}_{int(DeterministicClock.now().timestamp())}",
            is_valid=is_valid,
            quality_score=quality_score,
            total_records=1,
            valid_records=1 if is_valid else 0,
            invalid_records=0 if is_valid else 1,
            warning_records=1 if all_warnings else 0,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            rules_warnings=rules_warnings,
            errors=all_errors,
            warnings=all_warnings,
            validation_duration_ms=duration
        )

    async def validate_file(self, yaml_path: str) -> ValidationResult:
        """
        Validate entire YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Aggregated ValidationResult
        """
        start_time = DeterministicClock.now()

        # Load YAML file
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Remove metadata section
        if 'metadata' in data:
            del data['metadata']

        all_results = []
        total_factors = 0
        valid_factors = 0

        # Validate each factor
        for category, category_data in data.items():
            if not isinstance(category_data, dict):
                continue

            for factor_key, factor_data in category_data.items():
                if not isinstance(factor_data, dict):
                    continue

                total_factors += 1
                factor_id = f"{category}_{factor_key}".lower()

                result = await self.validate_factor(factor_id, factor_data)
                all_results.append(result)

                if result.is_valid:
                    valid_factors += 1

        # Aggregate results
        all_errors = []
        all_warnings = []
        all_rules_passed = set()
        all_rules_failed = set()
        all_rules_warnings = set()

        for result in all_results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_rules_passed.update(result.rules_passed)
            all_rules_failed.update(result.rules_failed)
            all_rules_warnings.update(result.rules_warnings)

        # Calculate overall quality score
        if all_results:
            avg_quality_score = sum(r.quality_score for r in all_results) / len(all_results)
        else:
            avg_quality_score = 0.0

        duration = (DeterministicClock.now() - start_time).total_seconds() * 1000

        return ValidationResult(
            validation_id=f"val_file_{Path(yaml_path).stem}_{int(DeterministicClock.now().timestamp())}",
            is_valid=valid_factors == total_factors,
            quality_score=avg_quality_score,
            total_records=total_factors,
            valid_records=valid_factors,
            invalid_records=total_factors - valid_factors,
            warning_records=len(all_warnings),
            rules_passed=list(all_rules_passed),
            rules_failed=list(all_rules_failed),
            rules_warnings=list(all_rules_warnings),
            errors=all_errors,
            warnings=all_warnings,
            validation_duration_ms=duration
        )

    def _calculate_quality_score(
        self,
        passed: List[str],
        failed: List[str],
        warnings: List[str]
    ) -> float:
        """
        Calculate quality score (0-100).

        Scoring:
        - Each passed rule contributes its weight * 100
        - Failed rules contribute 0
        - Warnings reduce score by 5% of rule weight
        """
        total_weight = sum(rule.weight for rule in self.rules)

        if total_weight == 0:
            return 0.0

        score = 0.0

        for rule in self.rules:
            if rule.name in passed:
                rule_score = rule.weight * 100

                # Reduce for warnings
                if rule.name in warnings:
                    rule_score *= 0.95

                score += rule_score

        # Normalize by total weight
        normalized_score = score / total_weight

        return round(min(100.0, max(0.0, normalized_score)), 2)

    async def close(self):
        """Close validators that need cleanup."""
        for rule in self.rules:
            if isinstance(rule, URIValidator):
                await rule.close()
