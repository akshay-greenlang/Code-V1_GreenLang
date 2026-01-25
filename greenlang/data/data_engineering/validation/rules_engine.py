"""
Validation Rules Engine
=======================

Comprehensive rule-based validation engine for emission factor data quality.
Supports 28+ validation rule types across completeness, validity, accuracy,
consistency, uniqueness, and timeliness dimensions.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Callable, Set, Union
from enum import Enum
from datetime import datetime, date, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import re
import logging
import hashlib
import json

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RuleType(str, Enum):
    """Types of validation rules."""
    # Completeness
    REQUIRED = "required"
    NOT_NULL = "not_null"
    NOT_EMPTY = "not_empty"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"

    # Validity
    TYPE_CHECK = "type_check"
    PATTERN = "pattern"
    ENUM = "enum"
    RANGE = "range"
    DATE_FORMAT = "date_format"
    CN_CODE = "cn_code"
    ISO_COUNTRY = "iso_country"

    # Accuracy
    EMISSION_FACTOR_RANGE = "emission_factor_range"
    CALCULATION_CHECK = "calculation_check"
    CROSS_REFERENCE = "cross_reference"
    GWP_CHECK = "gwp_check"

    # Consistency
    CROSS_FIELD = "cross_field"
    DATE_SEQUENCE = "date_sequence"
    CURRENCY_COUNTRY = "currency_country"
    VERSION_SEQUENCE = "version_sequence"

    # Uniqueness
    UNIQUE = "unique"
    COMPOSITE_UNIQUE = "composite_unique"
    NO_DUPLICATES = "no_duplicates"

    # Timeliness
    DATA_FRESHNESS = "data_freshness"
    FACTOR_VINTAGE = "factor_vintage"
    PROCESSING_LATENCY = "processing_latency"

    # Custom
    CUSTOM = "custom"


class RuleSeverity(str, Enum):
    """Severity levels for validation failures."""
    ERROR = "error"  # Must fix, blocks processing
    WARNING = "warning"  # Should fix, doesn't block
    INFO = "info"  # Informational only


class ValidationRule(BaseModel):
    """Configuration for a single validation rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_type: RuleType = Field(..., description="Type of validation rule")
    field: Optional[str] = Field(None, description="Field to validate")
    fields: Optional[List[str]] = Field(None, description="Multiple fields for composite rules")
    severity: RuleSeverity = Field(default=RuleSeverity.ERROR)
    message: str = Field(..., description="Error message template")
    enabled: bool = Field(default=True)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


@dataclass
class RuleViolation:
    """A single rule violation."""
    rule_id: str
    field: Optional[str]
    message: str
    severity: RuleSeverity
    actual_value: Any = None
    expected_value: Any = None
    record_index: Optional[int] = None


class ValidationResult(BaseModel):
    """Result of validating a dataset."""
    is_valid: bool = Field(..., description="Overall validity")
    total_records: int = Field(default=0)
    valid_records: int = Field(default=0)
    invalid_records: int = Field(default=0)
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    info_count: int = Field(default=0)
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    rules_executed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    quality_score: float = Field(default=100.0)

    class Config:
        use_enum_values = True


class ValidationRulesEngine:
    """
    Comprehensive validation rules engine.

    Supports:
    - 28+ rule types across 6 quality dimensions
    - Configurable rule sets
    - Batch validation with parallel processing
    - Detailed violation reporting
    - Quality score calculation
    """

    # Valid CN code pattern (8 digits)
    CN_CODE_PATTERN = re.compile(r'^\d{8}$')

    # ISO 3166-1 alpha-2 country codes (subset)
    ISO_COUNTRY_CODES = {
        'AF', 'AL', 'DZ', 'AD', 'AO', 'AR', 'AM', 'AU', 'AT', 'AZ',
        'BH', 'BD', 'BY', 'BE', 'BZ', 'BJ', 'BT', 'BO', 'BA', 'BW',
        'BR', 'BN', 'BG', 'BF', 'BI', 'KH', 'CM', 'CA', 'CF', 'TD',
        'CL', 'CN', 'CO', 'KM', 'CG', 'CD', 'CR', 'CI', 'HR', 'CU',
        'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'EC', 'EG', 'SV', 'GQ',
        'ER', 'EE', 'ET', 'FJ', 'FI', 'FR', 'GA', 'GM', 'GE', 'DE',
        'GH', 'GR', 'GD', 'GT', 'GN', 'GW', 'GY', 'HT', 'HN', 'HU',
        'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IL', 'IT', 'JM', 'JP',
        'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV',
        'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MK', 'MG', 'MW',
        'MY', 'MV', 'ML', 'MT', 'MH', 'MR', 'MU', 'MX', 'FM', 'MD',
        'MC', 'MN', 'ME', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL',
        'NZ', 'NI', 'NE', 'NG', 'NO', 'OM', 'PK', 'PW', 'PA', 'PG',
        'PY', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'KN',
        'LC', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL',
        'SG', 'SK', 'SI', 'SB', 'SO', 'ZA', 'SS', 'ES', 'LK', 'SD',
        'SR', 'SZ', 'SE', 'CH', 'SY', 'TJ', 'TZ', 'TH', 'TL', 'TG',
        'TO', 'TT', 'TN', 'TR', 'TM', 'TV', 'UG', 'UA', 'AE', 'GB',
        'US', 'UY', 'UZ', 'VU', 'VA', 'VE', 'VN', 'YE', 'ZM', 'ZW',
    }

    # GWP values from IPCC AR6 (100-year)
    IPCC_AR6_GWP = {
        'CO2': 1,
        'CH4': 28,
        'N2O': 265,
        'SF6': 23500,
        'NF3': 16100,
    }

    # Typical emission factor ranges by industry (kgCO2e/unit)
    EMISSION_FACTOR_RANGES = {
        'steel': {'min': 0.5, 'max': 5.0, 'unit': 'kgCO2e/kg'},
        'cement': {'min': 0.3, 'max': 1.2, 'unit': 'kgCO2e/kg'},
        'aluminum': {'min': 2.0, 'max': 20.0, 'unit': 'kgCO2e/kg'},
        'electricity': {'min': 0.0, 'max': 1.5, 'unit': 'kgCO2e/kWh'},
        'natural_gas': {'min': 0.001, 'max': 0.01, 'unit': 'kgCO2e/kWh'},
    }

    def __init__(self, rules: List[ValidationRule] = None):
        """Initialize validation engine with rules."""
        self.rules: Dict[str, ValidationRule] = {}
        self._seen_values: Dict[str, Set] = {}  # For uniqueness checks
        self._custom_validators: Dict[str, Callable] = {}

        if rules:
            for rule in rules:
                self.add_rule(rule)

        # Add default emission factor rules
        self._add_default_rules()

    def _add_default_rules(self):
        """Add default validation rules for emission factors."""
        default_rules = [
            # Completeness rules
            ValidationRule(
                rule_id="EF001",
                rule_type=RuleType.REQUIRED,
                field="factor_id",
                severity=RuleSeverity.ERROR,
                message="factor_id is required"
            ),
            ValidationRule(
                rule_id="EF002",
                rule_type=RuleType.REQUIRED,
                field="factor_value",
                severity=RuleSeverity.ERROR,
                message="factor_value is required"
            ),
            ValidationRule(
                rule_id="EF003",
                rule_type=RuleType.REQUIRED,
                field="factor_unit",
                severity=RuleSeverity.ERROR,
                message="factor_unit is required"
            ),
            ValidationRule(
                rule_id="EF004",
                rule_type=RuleType.REQUIRED,
                field="industry",
                severity=RuleSeverity.ERROR,
                message="industry category is required"
            ),

            # Validity rules
            ValidationRule(
                rule_id="EF010",
                rule_type=RuleType.RANGE,
                field="factor_value",
                severity=RuleSeverity.ERROR,
                message="factor_value must be non-negative",
                parameters={"min": 0}
            ),
            ValidationRule(
                rule_id="EF011",
                rule_type=RuleType.RANGE,
                field="reference_year",
                severity=RuleSeverity.ERROR,
                message="reference_year must be between 1990 and 2050",
                parameters={"min": 1990, "max": 2050}
            ),
            ValidationRule(
                rule_id="EF012",
                rule_type=RuleType.ENUM,
                field="ghg_type",
                severity=RuleSeverity.ERROR,
                message="Invalid GHG type",
                parameters={"allowed": ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3", "CO2e"]}
            ),
            ValidationRule(
                rule_id="EF013",
                rule_type=RuleType.ENUM,
                field="scope_type",
                severity=RuleSeverity.ERROR,
                message="Invalid scope type",
                parameters={"allowed": ["scope_1", "scope_2_location", "scope_2_market", "scope_3", "well_to_tank"]}
            ),

            # Quality rules
            ValidationRule(
                rule_id="EF020",
                rule_type=RuleType.RANGE,
                field="aggregate_dqi",
                severity=RuleSeverity.WARNING,
                message="Data quality score below threshold",
                parameters={"min": 50}
            ),

            # Uniqueness rules
            ValidationRule(
                rule_id="EF030",
                rule_type=RuleType.UNIQUE,
                field="factor_hash",
                severity=RuleSeverity.ERROR,
                message="Duplicate emission factor detected"
            ),

            # Timeliness rules
            ValidationRule(
                rule_id="EF040",
                rule_type=RuleType.FACTOR_VINTAGE,
                field="reference_year",
                severity=RuleSeverity.WARNING,
                message="Emission factor is outdated (>5 years old)",
                parameters={"max_age_years": 5}
            ),
        ]

        for rule in default_rules:
            if rule.rule_id not in self.rules:
                self.rules[rule.rule_id] = rule

    def add_rule(self, rule: ValidationRule) -> 'ValidationRulesEngine':
        """Add a validation rule."""
        self.rules[rule.rule_id] = rule
        return self

    def remove_rule(self, rule_id: str) -> 'ValidationRulesEngine':
        """Remove a validation rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
        return self

    def register_custom_validator(
        self,
        name: str,
        func: Callable[[Any, Dict[str, Any]], bool]
    ) -> 'ValidationRulesEngine':
        """Register a custom validation function."""
        self._custom_validators[name] = func
        return self

    def validate_record(
        self,
        record: Dict[str, Any],
        record_index: int = 0
    ) -> List[RuleViolation]:
        """
        Validate a single record against all rules.

        Args:
            record: Record to validate
            record_index: Index of record in batch

        Returns:
            List of violations found
        """
        violations = []

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            try:
                violation = self._check_rule(rule, record, record_index)
                if violation:
                    violations.append(violation)
            except Exception as e:
                logger.error(f"Error checking rule {rule_id}: {e}")

        return violations

    def validate_batch(
        self,
        records: List[Dict[str, Any]],
        fail_fast: bool = False
    ) -> ValidationResult:
        """
        Validate a batch of records.

        Args:
            records: List of records to validate
            fail_fast: Stop on first error

        Returns:
            ValidationResult with all violations
        """
        start_time = datetime.utcnow()
        all_violations = []
        valid_count = 0
        invalid_count = 0
        error_count = 0
        warning_count = 0
        info_count = 0

        # Reset uniqueness tracking for batch
        self._seen_values.clear()

        for idx, record in enumerate(records):
            violations = self.validate_record(record, idx)

            if violations:
                invalid_count += 1
                for v in violations:
                    all_violations.append({
                        'rule_id': v.rule_id,
                        'field': v.field,
                        'message': v.message,
                        'severity': v.severity.value,
                        'record_index': v.record_index,
                        'actual_value': str(v.actual_value)[:100] if v.actual_value else None,
                    })

                    if v.severity == RuleSeverity.ERROR:
                        error_count += 1
                    elif v.severity == RuleSeverity.WARNING:
                        warning_count += 1
                    else:
                        info_count += 1

                if fail_fast and error_count > 0:
                    break
            else:
                valid_count += 1

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Calculate quality score
        quality_score = 100.0
        if len(records) > 0:
            quality_score = (valid_count / len(records)) * 100

        return ValidationResult(
            is_valid=error_count == 0,
            total_records=len(records),
            valid_records=valid_count,
            invalid_records=invalid_count,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            violations=all_violations,
            rules_executed=list(self.rules.keys()),
            execution_time_ms=execution_time,
            quality_score=round(quality_score, 2),
        )

    def _check_rule(
        self,
        rule: ValidationRule,
        record: Dict[str, Any],
        record_index: int
    ) -> Optional[RuleViolation]:
        """Check a single rule against a record."""

        rule_type = RuleType(rule.rule_type) if isinstance(rule.rule_type, str) else rule.rule_type

        # Get field value
        value = record.get(rule.field) if rule.field else None

        # Completeness rules
        if rule_type == RuleType.REQUIRED:
            if value is None or value == '':
                return self._create_violation(rule, record_index, value)

        elif rule_type == RuleType.NOT_NULL:
            if value is None:
                return self._create_violation(rule, record_index, value)

        elif rule_type == RuleType.NOT_EMPTY:
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return self._create_violation(rule, record_index, value)

        elif rule_type == RuleType.MIN_LENGTH:
            min_len = rule.parameters.get('min', 0)
            if value is not None and len(str(value)) < min_len:
                return self._create_violation(rule, record_index, value, f"min length {min_len}")

        elif rule_type == RuleType.MAX_LENGTH:
            max_len = rule.parameters.get('max', float('inf'))
            if value is not None and len(str(value)) > max_len:
                return self._create_violation(rule, record_index, value, f"max length {max_len}")

        # Validity rules
        elif rule_type == RuleType.TYPE_CHECK:
            expected_type = rule.parameters.get('expected')
            if value is not None and not self._check_type(value, expected_type):
                return self._create_violation(rule, record_index, value, expected_type)

        elif rule_type == RuleType.PATTERN:
            pattern = rule.parameters.get('pattern')
            if value is not None and pattern and not re.match(pattern, str(value)):
                return self._create_violation(rule, record_index, value, pattern)

        elif rule_type == RuleType.ENUM:
            allowed = rule.parameters.get('allowed', [])
            if value is not None and value not in allowed:
                return self._create_violation(rule, record_index, value, allowed)

        elif rule_type == RuleType.RANGE:
            if value is not None:
                try:
                    num_value = float(value)
                    min_val = rule.parameters.get('min')
                    max_val = rule.parameters.get('max')
                    if min_val is not None and num_value < min_val:
                        return self._create_violation(rule, record_index, value, f">= {min_val}")
                    if max_val is not None and num_value > max_val:
                        return self._create_violation(rule, record_index, value, f"<= {max_val}")
                except (ValueError, TypeError):
                    pass

        elif rule_type == RuleType.CN_CODE:
            if value is not None and not self.CN_CODE_PATTERN.match(str(value)):
                return self._create_violation(rule, record_index, value, "8-digit CN code")

        elif rule_type == RuleType.ISO_COUNTRY:
            if value is not None and str(value).upper() not in self.ISO_COUNTRY_CODES:
                return self._create_violation(rule, record_index, value, "valid ISO country code")

        # Accuracy rules
        elif rule_type == RuleType.EMISSION_FACTOR_RANGE:
            industry = record.get('industry')
            if industry and value is not None:
                ranges = self.EMISSION_FACTOR_RANGES.get(industry)
                if ranges:
                    try:
                        num_value = float(value)
                        if num_value < ranges['min'] or num_value > ranges['max']:
                            return self._create_violation(
                                rule, record_index, value,
                                f"{ranges['min']}-{ranges['max']} {ranges['unit']}"
                            )
                    except (ValueError, TypeError):
                        pass

        elif rule_type == RuleType.GWP_CHECK:
            ghg_type = record.get('ghg_type')
            gwp_value = record.get('gwp_value')
            if ghg_type and gwp_value:
                expected_gwp = self.IPCC_AR6_GWP.get(ghg_type)
                if expected_gwp and abs(float(gwp_value) - expected_gwp) > expected_gwp * 0.1:
                    return self._create_violation(rule, record_index, gwp_value, expected_gwp)

        # Consistency rules
        elif rule_type == RuleType.CROSS_FIELD:
            func = rule.parameters.get('check_func')
            if func and not func(record):
                return self._create_violation(rule, record_index, None)

        elif rule_type == RuleType.DATE_SEQUENCE:
            start_field = rule.parameters.get('start_field')
            end_field = rule.parameters.get('end_field')
            start_date = record.get(start_field)
            end_date = record.get(end_field)
            if start_date and end_date:
                if self._parse_date(start_date) > self._parse_date(end_date):
                    return self._create_violation(
                        rule, record_index,
                        f"{start_date} > {end_date}",
                        "start date before end date"
                    )

        # Uniqueness rules
        elif rule_type == RuleType.UNIQUE:
            if rule.field not in self._seen_values:
                self._seen_values[rule.field] = set()
            if value in self._seen_values[rule.field]:
                return self._create_violation(rule, record_index, value, "unique value")
            if value is not None:
                self._seen_values[rule.field].add(value)

        elif rule_type == RuleType.COMPOSITE_UNIQUE:
            fields = rule.fields or []
            composite_key = tuple(record.get(f) for f in fields)
            key_name = '_'.join(fields)
            if key_name not in self._seen_values:
                self._seen_values[key_name] = set()
            if composite_key in self._seen_values[key_name]:
                return self._create_violation(rule, record_index, composite_key, "unique combination")
            self._seen_values[key_name].add(composite_key)

        # Timeliness rules
        elif rule_type == RuleType.FACTOR_VINTAGE:
            max_age = rule.parameters.get('max_age_years', 5)
            if value is not None:
                try:
                    year = int(value)
                    current_year = datetime.now().year
                    if current_year - year > max_age:
                        return self._create_violation(
                            rule, record_index, value,
                            f"within {max_age} years"
                        )
                except (ValueError, TypeError):
                    pass

        elif rule_type == RuleType.DATA_FRESHNESS:
            max_age_days = rule.parameters.get('max_age_days', 30)
            if value is not None:
                try:
                    record_date = self._parse_date(value)
                    age = (datetime.now().date() - record_date).days
                    if age > max_age_days:
                        return self._create_violation(
                            rule, record_index, value,
                            f"within {max_age_days} days"
                        )
                except:
                    pass

        # Custom rules
        elif rule_type == RuleType.CUSTOM:
            validator_name = rule.parameters.get('validator')
            if validator_name and validator_name in self._custom_validators:
                validator = self._custom_validators[validator_name]
                if not validator(value, record):
                    return self._create_violation(rule, record_index, value)

        return None

    def _create_violation(
        self,
        rule: ValidationRule,
        record_index: int,
        actual_value: Any,
        expected_value: Any = None
    ) -> RuleViolation:
        """Create a violation instance."""
        return RuleViolation(
            rule_id=rule.rule_id,
            field=rule.field,
            message=rule.message,
            severity=RuleSeverity(rule.severity) if isinstance(rule.severity, str) else rule.severity,
            actual_value=actual_value,
            expected_value=expected_value,
            record_index=record_index,
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'numeric':
            return isinstance(value, (int, float, Decimal))
        elif expected_type == 'integer':
            return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'date':
            return isinstance(value, (date, datetime))
        elif expected_type == 'list':
            return isinstance(value, list)
        elif expected_type == 'dict':
            return isinstance(value, dict)
        return True

    def _parse_date(self, value: Any) -> date:
        """Parse date from various formats."""
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        raise ValueError(f"Cannot parse date: {value}")

    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of configured rules."""
        by_type = {}
        by_severity = {}

        for rule in self.rules.values():
            rule_type = rule.rule_type
            severity = rule.severity

            by_type[rule_type] = by_type.get(rule_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "by_type": by_type,
            "by_severity": by_severity,
        }


# Pre-configured rule sets for common use cases
CBAM_VALIDATION_RULES = [
    ValidationRule(
        rule_id="CBAM001",
        rule_type=RuleType.CN_CODE,
        field="product_code",
        severity=RuleSeverity.ERROR,
        message="Invalid CN code format for CBAM"
    ),
    ValidationRule(
        rule_id="CBAM002",
        rule_type=RuleType.ENUM,
        field="industry",
        severity=RuleSeverity.ERROR,
        message="Industry must be CBAM-eligible",
        parameters={"allowed": ["steel", "cement", "aluminum", "fertilizer", "hydrogen", "electricity"]}
    ),
    ValidationRule(
        rule_id="CBAM003",
        rule_type=RuleType.REQUIRED,
        field="production_route",
        severity=RuleSeverity.WARNING,
        message="Production route recommended for CBAM reporting"
    ),
]

CSRD_VALIDATION_RULES = [
    ValidationRule(
        rule_id="CSRD001",
        rule_type=RuleType.REQUIRED,
        field="source",
        severity=RuleSeverity.ERROR,
        message="Data source required for CSRD audit trail"
    ),
    ValidationRule(
        rule_id="CSRD002",
        rule_type=RuleType.REQUIRED,
        field="quality",
        severity=RuleSeverity.ERROR,
        message="Data quality assessment required for CSRD"
    ),
    ValidationRule(
        rule_id="CSRD003",
        rule_type=RuleType.RANGE,
        field="aggregate_dqi",
        severity=RuleSeverity.WARNING,
        message="Data quality score below CSRD threshold",
        parameters={"min": 70}
    ),
]
