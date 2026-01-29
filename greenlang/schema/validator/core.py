# -*- coding: utf-8 -*-
"""
Validator Core Engine for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements the main schema validator that orchestrates all validation
phases in the correct order. It serves as the primary entry point for payload
validation against GreenLang schemas.

Validation Pipeline Order:
    1. Parse payload (with safety limits)
    2. Resolve schema (with IR caching)
    3. Structural validation (types, required fields)
    4. Constraint validation (ranges, patterns, enums)
    5. Unit validation (dimensional compatibility)
    6. Rule validation (cross-field rules)
    7. Linting (non-blocking warnings)

Key Features:
    - Zero-hallucination: All calculations are deterministic
    - IR caching for performance (compile once, validate many)
    - Fail-fast option for early termination
    - Max errors limit for bounded output
    - Complete timing breakdown for each phase
    - Batch validation with shared compiled schema

Performance:
    - P95 latency < 25ms for small payloads (<50KB)
    - P95 latency < 150ms for medium payloads (<500KB)
    - Batch validation shares compiled IR for efficiency

Example:
    >>> from greenlang.schema.validator.core import SchemaValidator, validate
    >>> validator = SchemaValidator()
    >>> result = validator.validate(payload, "gl://schemas/activity@1.0.0")
    >>> if result.valid:
    ...     print("Validation passed!")
    ... else:
    ...     for finding in result.findings:
    ...         print(f"{finding.code}: {finding.message}")

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.5
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.schema.compiler.compiler import SchemaCompiler
from greenlang.schema.compiler.ir import SchemaIR, CompilationResult
from greenlang.schema.compiler.parser import parse_payload, ParseResult, ParseError
from greenlang.schema.constants import (
    MAX_FINDINGS,
    MAX_BATCH_ITEMS,
    MAX_BATCH_TIME_SECONDS,
    BATCH_CHUNK_SIZE,
)
from greenlang.schema.models.config import (
    ValidationOptions,
    ValidationProfile,
    UnknownFieldPolicy,
)
from greenlang.schema.models.finding import Finding, Severity, FindingHint
from greenlang.schema.models.report import (
    ValidationReport,
    ValidationSummary,
    TimingInfo,
    BatchValidationReport,
    BatchSummary,
    ItemResult,
)
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.registry.resolver import SchemaRegistry, SchemaSource

# Import validators (these may be partially implemented)
from greenlang.schema.validator.structural import StructuralValidator
from greenlang.schema.validator.constraints import ConstraintValidator
from greenlang.schema.validator.units import UnitValidator, UnitCatalog
from greenlang.schema.validator.rules import RuleValidator
from greenlang.schema.validator.linter import SchemaLinter


logger = logging.getLogger(__name__)


# =============================================================================
# SEVERITY ORDERING FOR FINDING SORTING
# =============================================================================

SEVERITY_ORDER: Dict[str, int] = {
    "error": 0,
    "warning": 1,
    "info": 2,
}


# =============================================================================
# SCHEMA VALIDATOR
# =============================================================================


class SchemaValidator:
    """
    Main schema validator that orchestrates all validation phases.

    This is the primary entry point for validating payloads against GreenLang
    schemas. It coordinates parsing, schema resolution, and all validation
    phases to produce a complete validation report.

    Thread Safety:
        The validator is thread-safe for concurrent validation calls.
        The IR cache uses immutable values and the same payload can be
        validated concurrently.

    Attributes:
        registry: Schema registry for resolving schema references
        catalog: Unit catalog for unit validation
        options: Default validation options
        _compiler: Schema compiler for compiling schemas to IR
        _ir_cache: Cache of compiled schema IRs

    Example:
        >>> validator = SchemaValidator()
        >>> result = validator.validate(
        ...     payload={"energy": 100, "unit": "kWh"},
        ...     schema_ref="gl://schemas/activity@1.0.0"
        ... )
        >>> if result.valid:
        ...     print("Validation passed!")
        ... else:
        ...     for finding in result.findings:
        ...         print(f"{finding.code}: {finding.message}")
    """

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistry] = None,
        unit_catalog: Optional[UnitCatalog] = None,
        options: Optional[ValidationOptions] = None,
    ):
        """
        Initialize the schema validator.

        Args:
            schema_registry: Optional registry for resolving schema references.
                If not provided, schemas must be provided directly.
            unit_catalog: Optional unit catalog for unit validation.
                If not provided, a default catalog is created.
            options: Default validation options.
                If not provided, standard defaults are used.
        """
        self.registry = schema_registry
        self.catalog = unit_catalog or UnitCatalog()
        self.options = options or ValidationOptions()
        self._compiler = SchemaCompiler()
        self._ir_cache: Dict[str, SchemaIR] = {}
        # Linter is created on-demand since it requires an IR

        logger.debug(
            f"SchemaValidator initialized with profile={self.options.profile}, "
            f"registry={'configured' if schema_registry else 'none'}"
        )

    def validate(
        self,
        payload: Union[str, Dict[str, Any]],
        schema_ref: Union[SchemaRef, str],
        options: Optional[ValidationOptions] = None,
    ) -> ValidationReport:
        """
        Validate a single payload against a schema.

        This is the main validation entry point. It orchestrates all validation
        phases and produces a complete report with findings, normalized payload,
        and timing information.

        Validation Phases:
            1. Parse payload (if string)
            2. Resolve and compile schema (with caching)
            3. Structural validation
            4. Constraint validation
            5. Unit validation
            6. Rule validation
            7. Linting (non-blocking)

        Args:
            payload: Payload as YAML/JSON string or pre-parsed dictionary.
            schema_ref: Schema reference as SchemaRef object or URI string
                (e.g., "gl://schemas/activity@1.0.0").
            options: Override default validation options for this call.

        Returns:
            ValidationReport containing:
                - valid: Whether payload passed validation
                - schema_ref: The resolved schema reference
                - schema_hash: SHA-256 hash of the compiled schema
                - summary: Counts of errors, warnings, and info
                - findings: List of all validation findings
                - normalized_payload: Normalized payload (if enabled)
                - fix_suggestions: Suggested fixes (if enabled)
                - timings: Performance timing for each phase

        Raises:
            No exceptions are raised; errors are captured as findings.

        Example:
            >>> result = validator.validate(
            ...     payload='{"energy": 100}',
            ...     schema_ref="gl://schemas/activity@1.0.0",
            ...     options=ValidationOptions(profile=ValidationProfile.STRICT)
            ... )
        """
        start_time = time.perf_counter()
        timings: Dict[str, float] = {}
        effective_options = options or self.options

        logger.info(f"Starting validation with profile={effective_options.profile}")

        # Normalize schema_ref to SchemaRef object
        if isinstance(schema_ref, str):
            schema_ref = self._parse_schema_ref(schema_ref)

        # Phase 1: Parse payload
        parse_start = time.perf_counter()
        parsed_payload, parse_findings = self._parse_payload(payload)
        timings["parse_ms"] = (time.perf_counter() - parse_start) * 1000

        if parse_findings:
            # Parse error - return early with error report
            return self._create_error_report(
                schema_ref=schema_ref,
                findings=parse_findings,
                timings=timings,
                start_time=start_time,
            )

        # Phase 2: Resolve and compile schema
        compile_start = time.perf_counter()
        ir, compile_findings = self._get_or_compile_schema(schema_ref)
        timings["compile_ms"] = (time.perf_counter() - compile_start) * 1000

        if compile_findings:
            return self._create_error_report(
                schema_ref=schema_ref,
                findings=compile_findings,
                timings=timings,
                start_time=start_time,
            )

        # Phase 3-7: Run validation phases
        validate_start = time.perf_counter()
        findings, phase_timings = self._run_validation_phases(
            payload=parsed_payload,
            ir=ir,
            options=effective_options,
        )
        timings["validate_ms"] = (time.perf_counter() - validate_start) * 1000
        timings.update(phase_timings)

        # Sort findings deterministically
        sorted_findings = self._sort_findings(findings)

        # Create summary
        summary = self._create_summary(sorted_findings)

        # Calculate total time
        total_time = (time.perf_counter() - start_time) * 1000
        timings["total_ms"] = total_time

        # Create timing info
        timing_info = TimingInfo(
            parse_ms=timings.get("parse_ms"),
            compile_ms=timings.get("compile_ms"),
            validate_ms=timings.get("validate_ms"),
            normalize_ms=timings.get("normalize_ms"),
            suggest_ms=timings.get("suggest_ms"),
            total_ms=total_time,
        )

        # Build the report
        report = ValidationReport(
            valid=summary.valid,
            schema_ref=schema_ref,
            schema_hash=ir.schema_hash,
            summary=summary,
            findings=sorted_findings,
            normalized_payload=parsed_payload if effective_options.normalize else None,
            fix_suggestions=None,  # TODO: Implement fix suggestions
            timings=timing_info,
        )

        logger.info(
            f"Validation completed: valid={report.valid}, "
            f"errors={summary.error_count}, warnings={summary.warning_count}, "
            f"total_ms={total_time:.2f}"
        )

        return report

    def validate_batch(
        self,
        payloads: List[Union[str, Dict[str, Any]]],
        schema_ref: Union[SchemaRef, str],
        options: Optional[ValidationOptions] = None,
    ) -> BatchValidationReport:
        """
        Validate multiple payloads efficiently.

        Shares the compiled schema IR across all payloads for better performance.
        This is the recommended method when validating many payloads against
        the same schema.

        Args:
            payloads: List of payloads to validate (strings or dicts).
            schema_ref: Schema reference as SchemaRef object or URI string.
            options: Override default validation options.

        Returns:
            BatchValidationReport containing:
                - schema_ref: The resolved schema reference
                - schema_hash: SHA-256 hash of the compiled schema
                - summary: Aggregate summary of all items
                - results: Individual result for each payload

        Raises:
            ValueError: If batch exceeds MAX_BATCH_ITEMS limit.

        Example:
            >>> payloads = [
            ...     {"energy": 100, "unit": "kWh"},
            ...     {"energy": 200, "unit": "MWh"},
            ... ]
            >>> result = validator.validate_batch(
            ...     payloads=payloads,
            ...     schema_ref="gl://schemas/activity@1.0.0"
            ... )
            >>> print(f"Valid: {result.summary.valid_count}/{result.summary.total_items}")
        """
        batch_start = time.perf_counter()
        effective_options = options or self.options

        # Validate batch size
        if len(payloads) > MAX_BATCH_ITEMS:
            raise ValueError(
                f"Batch size {len(payloads)} exceeds maximum {MAX_BATCH_ITEMS}"
            )

        logger.info(
            f"Starting batch validation: {len(payloads)} items, "
            f"profile={effective_options.profile}"
        )

        # Normalize schema_ref
        if isinstance(schema_ref, str):
            schema_ref = self._parse_schema_ref(schema_ref)

        # Compile schema once
        ir, compile_findings = self._get_or_compile_schema(schema_ref)

        if compile_findings:
            # Schema compilation failed - return error for all items
            return self._create_batch_error_report(
                schema_ref=schema_ref,
                compile_findings=compile_findings,
                num_payloads=len(payloads),
                start_time=batch_start,
            )

        # Validate each payload
        results: List[ItemResult] = []
        valid_count = 0
        error_count = 0
        warning_count = 0

        for index, payload in enumerate(payloads):
            # Check batch timeout
            elapsed_seconds = time.perf_counter() - batch_start
            if elapsed_seconds > MAX_BATCH_TIME_SECONDS:
                logger.warning(
                    f"Batch timeout after {index} items ({elapsed_seconds:.1f}s)"
                )
                # Add timeout error for remaining items
                for remaining_index in range(index, len(payloads)):
                    results.append(
                        ItemResult(
                            index=remaining_index,
                            valid=False,
                            findings=[
                                Finding(
                                    code="GLSCHEMA-E803",
                                    severity=Severity.ERROR,
                                    path="",
                                    message=f"Batch timeout exceeded ({MAX_BATCH_TIME_SECONDS}s)",
                                )
                            ],
                        )
                    )
                    error_count += 1
                break

            # Validate single payload
            item_result = self._validate_single_for_batch(
                payload=payload,
                ir=ir,
                index=index,
                options=effective_options,
            )
            results.append(item_result)

            if item_result.valid:
                valid_count += 1
            else:
                error_count += 1

            if item_result.warning_count() > 0:
                warning_count += 1

        # Create batch summary
        batch_summary = BatchSummary(
            total_items=len(payloads),
            valid_count=valid_count,
            error_count=error_count,
            warning_count=warning_count,
        )

        # Create batch report
        report = BatchValidationReport(
            schema_ref=schema_ref,
            schema_hash=ir.schema_hash,
            summary=batch_summary,
            results=results,
        )

        batch_time = (time.perf_counter() - batch_start) * 1000
        logger.info(
            f"Batch validation completed: {valid_count}/{len(payloads)} valid, "
            f"total_ms={batch_time:.2f}"
        )

        return report

    def _parse_schema_ref(self, uri: str) -> SchemaRef:
        """
        Parse a schema URI string into a SchemaRef.

        Args:
            uri: Schema URI (e.g., "gl://schemas/activity@1.0.0")

        Returns:
            SchemaRef object

        Raises:
            ValueError: If URI format is invalid
        """
        return SchemaRef.from_uri(uri)

    def _parse_payload(
        self,
        payload: Union[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Finding]]:
        """
        Parse payload with safety limits.

        Args:
            payload: Payload as string or dict

        Returns:
            Tuple of (parsed_payload, findings)
            If parsing fails, findings contains the error
        """
        if isinstance(payload, dict):
            return payload, []

        try:
            result: ParseResult = parse_payload(payload)
            return result.data, []
        except ParseError as e:
            finding = Finding(
                code=e.code,
                severity=Severity.ERROR,
                path="",
                message=e.message,
                expected=None,
                actual=None,
                hint=FindingHint(
                    category="parse_error",
                    suggested_values=[],
                    docs_url=None,
                ) if e.details else None,
            )
            return {}, [finding]

    def _get_or_compile_schema(
        self,
        schema_ref: SchemaRef,
    ) -> Tuple[Optional[SchemaIR], List[Finding]]:
        """
        Get cached IR or compile schema.

        Implements schema caching for performance. The cache key includes
        schema_id and version.

        Args:
            schema_ref: Schema reference

        Returns:
            Tuple of (compiled_ir, findings)
            If compilation fails, ir is None and findings contains errors
        """
        cache_key = schema_ref.to_cache_key()

        # Check cache
        if cache_key in self._ir_cache:
            logger.debug(f"IR cache hit for {cache_key}")
            return self._ir_cache[cache_key], []

        logger.debug(f"IR cache miss for {cache_key}")

        # Resolve schema from registry
        schema_source: Optional[Dict[str, Any]] = None

        if self.registry is not None:
            try:
                source: SchemaSource = self.registry.resolve(
                    schema_ref.schema_id,
                    schema_ref.version,
                )
                schema_source = source.content
            except Exception as e:
                logger.error(f"Schema resolution failed: {e}")
                finding = Finding(
                    code="GLSCHEMA-E500",
                    severity=Severity.ERROR,
                    path="",
                    message=f"Schema resolution failed: {schema_ref}",
                    expected={"schema_id": schema_ref.schema_id, "version": schema_ref.version},
                    actual=str(e),
                )
                return None, [finding]
        else:
            # No registry - require inline schema
            finding = Finding(
                code="GLSCHEMA-E500",
                severity=Severity.ERROR,
                path="",
                message=f"No schema registry configured and schema not found: {schema_ref}",
            )
            return None, [finding]

        # Compile schema
        result: CompilationResult = self._compiler.compile(
            schema_source=schema_source,
            schema_id=schema_ref.schema_id,
            version=schema_ref.version,
        )

        if not result.success:
            findings = [
                Finding(
                    code="GLSCHEMA-E502",
                    severity=Severity.ERROR,
                    path="",
                    message=f"Schema compilation failed: {error}",
                )
                for error in result.errors
            ]
            return None, findings

        # Cache the IR
        self._ir_cache[cache_key] = result.ir
        logger.debug(f"Cached IR for {cache_key}")

        return result.ir, []

    def _run_validation_phases(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR,
        options: ValidationOptions,
    ) -> Tuple[List[Finding], Dict[str, float]]:
        """
        Run all validation phases and collect findings.

        Executes validation phases in order:
            1. Structural validation
            2. Constraint validation
            3. Unit validation
            4. Rule validation
            5. Linting (non-blocking)

        Respects fail_fast and max_errors settings.

        Args:
            payload: Parsed payload
            ir: Compiled schema IR
            options: Validation options

        Returns:
            Tuple of (all_findings, phase_timings)
        """
        all_findings: List[Finding] = []
        phase_timings: Dict[str, float] = {}

        # Phase 3: Structural validation
        structural_start = time.perf_counter()
        structural_findings = self._run_structural_validation(payload, ir, options)
        phase_timings["structural_ms"] = (time.perf_counter() - structural_start) * 1000
        all_findings.extend(structural_findings)

        if self._should_stop(all_findings, options):
            return all_findings, phase_timings

        # Phase 4: Constraint validation
        constraint_start = time.perf_counter()
        constraint_findings = self._run_constraint_validation(payload, ir, options)
        phase_timings["constraint_ms"] = (time.perf_counter() - constraint_start) * 1000
        all_findings.extend(constraint_findings)

        if self._should_stop(all_findings, options):
            return all_findings, phase_timings

        # Phase 5: Unit validation
        unit_start = time.perf_counter()
        unit_findings = self._run_unit_validation(payload, ir, options)
        phase_timings["unit_ms"] = (time.perf_counter() - unit_start) * 1000
        all_findings.extend(unit_findings)

        if self._should_stop(all_findings, options):
            return all_findings, phase_timings

        # Phase 6: Rule validation
        rule_start = time.perf_counter()
        rule_findings = self._run_rule_validation(payload, ir, options)
        phase_timings["rule_ms"] = (time.perf_counter() - rule_start) * 1000
        all_findings.extend(rule_findings)

        if self._should_stop(all_findings, options):
            return all_findings, phase_timings

        # Phase 7: Linting (non-blocking - always runs)
        lint_start = time.perf_counter()
        lint_findings = self._run_linting(payload, ir, options)
        phase_timings["lint_ms"] = (time.perf_counter() - lint_start) * 1000
        all_findings.extend(lint_findings)

        return all_findings, phase_timings

    def _run_structural_validation(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR,
        options: ValidationOptions,
    ) -> List[Finding]:
        """
        Run structural validation phase.

        Validates:
            - Required field presence
            - Type checking
            - Additional properties policy
            - Property count constraints

        Args:
            payload: Parsed payload
            ir: Compiled schema IR
            options: Validation options

        Returns:
            List of structural findings
        """
        try:
            # Create structural validator with appropriate options
            from greenlang.schema.validator.structural import (
                StructuralValidator,
                ValidationOptions as StructuralValidationOptions,
            )

            structural_options = StructuralValidationOptions(
                profile=options.profile.value if hasattr(options.profile, 'value') else str(options.profile),
                unknown_field_policy=options.unknown_field_policy.value if hasattr(options.unknown_field_policy, 'value') else str(options.unknown_field_policy),
            )

            validator = StructuralValidator(ir, structural_options)
            findings = validator.validate(payload, "")

            # Convert to Finding model objects
            return self._convert_findings(findings)
        except NotImplementedError:
            logger.debug("Structural validation not yet implemented, skipping")
            return []
        except Exception as e:
            logger.error(f"Structural validation error: {e}", exc_info=True)
            return [
                Finding(
                    code="GLSCHEMA-E100",
                    severity=Severity.ERROR,
                    path="",
                    message=f"Structural validation failed: {str(e)}",
                )
            ]

    def _run_constraint_validation(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR,
        options: ValidationOptions,
    ) -> List[Finding]:
        """
        Run constraint validation phase.

        Validates:
            - Numeric constraints (min/max, multipleOf)
            - String constraints (pattern, length, format)
            - Array constraints (minItems, maxItems, uniqueItems)
            - Enum constraints

        Args:
            payload: Parsed payload
            ir: Compiled schema IR
            options: Validation options

        Returns:
            List of constraint findings
        """
        try:
            from greenlang.schema.validator.constraints import (
                ConstraintValidator,
                ValidationOptions as ConstraintValidationOptions,
            )

            constraint_options = ConstraintValidationOptions(
                coercion_policy=options.coercion_policy.value if hasattr(options.coercion_policy, 'value') else str(options.coercion_policy),
            )

            validator = ConstraintValidator(ir, constraint_options)
            findings = self._validate_constraints_recursive(validator, payload, ir, "")

            return self._convert_findings(findings)
        except NotImplementedError:
            logger.debug("Constraint validation not yet implemented, skipping")
            return []
        except Exception as e:
            logger.error(f"Constraint validation error: {e}", exc_info=True)
            return [
                Finding(
                    code="GLSCHEMA-E200",
                    severity=Severity.ERROR,
                    path="",
                    message=f"Constraint validation failed: {str(e)}",
                )
            ]

    def _validate_constraints_recursive(
        self,
        validator: Any,
        payload: Any,
        ir: SchemaIR,
        path: str,
    ) -> List[Any]:
        """
        Recursively validate constraints for all values in payload.

        Args:
            validator: ConstraintValidator instance
            payload: Current value to validate
            ir: Compiled schema IR
            path: Current JSON Pointer path

        Returns:
            List of findings from constraint validation
        """
        findings = []

        if isinstance(payload, dict):
            for key, value in payload.items():
                child_path = f"{path}/{key}"

                # Check numeric constraints
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_constraint = ir.get_numeric_constraint(child_path)
                    if numeric_constraint:
                        try:
                            constraint_findings = validator.validate_numeric(
                                value, numeric_constraint, child_path
                            )
                            findings.extend(constraint_findings)
                        except NotImplementedError:
                            pass

                # Check string constraints
                elif isinstance(value, str):
                    string_constraint = ir.get_string_constraint(child_path)
                    if string_constraint:
                        try:
                            constraint_findings = validator.validate_string(
                                value, string_constraint, child_path
                            )
                            findings.extend(constraint_findings)
                        except NotImplementedError:
                            pass

                # Check enum constraints
                enum_values = ir.get_enum(child_path)
                if enum_values is not None:
                    try:
                        constraint_findings = validator.validate_enum(
                            value, enum_values, child_path
                        )
                        findings.extend(constraint_findings)
                    except NotImplementedError:
                        pass

                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    nested_findings = self._validate_constraints_recursive(
                        validator, value, ir, child_path
                    )
                    findings.extend(nested_findings)

        elif isinstance(payload, list):
            # Check array constraints
            array_constraint = ir.get_array_constraint(path)
            if array_constraint:
                try:
                    constraint_findings = validator.validate_array(
                        payload, array_constraint, path
                    )
                    findings.extend(constraint_findings)
                except NotImplementedError:
                    pass

            # Recurse into array items
            for index, item in enumerate(payload):
                item_path = f"{path}/{index}"
                nested_findings = self._validate_constraints_recursive(
                    validator, item, ir, item_path
                )
                findings.extend(nested_findings)

        return findings

    def _run_unit_validation(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR,
        options: ValidationOptions,
    ) -> List[Finding]:
        """
        Run unit validation phase.

        Validates:
            - Unit presence when required
            - Unit dimensional compatibility
            - Unit catalog membership

        Args:
            payload: Parsed payload
            ir: Compiled schema IR
            options: Validation options

        Returns:
            List of unit findings
        """
        try:
            from greenlang.schema.validator.units import (
                UnitValidator,
                ValidationOptions as UnitValidationOptions,
            )

            unit_options = UnitValidationOptions(
                unit_system=options.unit_system,
            )

            validator = UnitValidator(self.catalog, unit_options)
            findings = self._validate_units_recursive(validator, payload, ir, "")

            return self._convert_findings(findings)
        except NotImplementedError:
            logger.debug("Unit validation not yet implemented, skipping")
            return []
        except Exception as e:
            logger.error(f"Unit validation error: {e}", exc_info=True)
            return [
                Finding(
                    code="GLSCHEMA-E300",
                    severity=Severity.ERROR,
                    path="",
                    message=f"Unit validation failed: {str(e)}",
                )
            ]

    def _validate_units_recursive(
        self,
        validator: Any,
        payload: Any,
        ir: SchemaIR,
        path: str,
    ) -> List[Any]:
        """
        Recursively validate units for all values in payload.

        Args:
            validator: UnitValidator instance
            payload: Current value to validate
            ir: Compiled schema IR
            path: Current JSON Pointer path

        Returns:
            List of findings from unit validation
        """
        findings = []

        if isinstance(payload, dict):
            for key, value in payload.items():
                child_path = f"{path}/{key}"

                # Check for unit specification
                unit_spec = ir.get_unit_spec(child_path)
                if unit_spec:
                    try:
                        unit_findings, _ = validator.validate(
                            value, unit_spec, child_path
                        )
                        findings.extend(unit_findings)
                    except NotImplementedError:
                        pass

                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    nested_findings = self._validate_units_recursive(
                        validator, value, ir, child_path
                    )
                    findings.extend(nested_findings)

        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                item_path = f"{path}/{index}"
                nested_findings = self._validate_units_recursive(
                    validator, item, ir, item_path
                )
                findings.extend(nested_findings)

        return findings

    def _run_rule_validation(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR,
        options: ValidationOptions,
    ) -> List[Finding]:
        """
        Run rule validation phase.

        Evaluates cross-field validation rules defined in the schema.

        Args:
            payload: Parsed payload
            ir: Compiled schema IR
            options: Validation options

        Returns:
            List of rule findings
        """
        if not ir.rule_bindings:
            return []

        try:
            from greenlang.schema.validator.rules import (
                RuleValidator,
                ValidationOptions as RuleValidationOptions,
            )

            rule_options = RuleValidationOptions()
            validator = RuleValidator(ir, rule_options)
            findings = validator.validate(payload, ir.rule_bindings)

            return self._convert_findings(findings)
        except NotImplementedError:
            logger.debug("Rule validation not yet implemented, skipping")
            return []
        except Exception as e:
            logger.error(f"Rule validation error: {e}", exc_info=True)
            return [
                Finding(
                    code="GLSCHEMA-E400",
                    severity=Severity.ERROR,
                    path="",
                    message=f"Rule validation failed: {str(e)}",
                )
            ]

    def _run_linting(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR,
        options: ValidationOptions,
    ) -> List[Finding]:
        """
        Run linting phase (non-blocking).

        Produces warnings for best-practice violations:
            - Unknown fields with close matches (typos)
            - Deprecated field usage
            - Non-canonical casing

        Args:
            payload: Parsed payload
            ir: Compiled schema IR
            options: Validation options

        Returns:
            List of lint findings (always warnings/info, never errors)
        """
        try:
            # Create linter with the IR for this validation
            linter = SchemaLinter(ir, options)
            findings = linter.lint(payload, "")
            return self._convert_findings(findings)
        except NotImplementedError:
            logger.debug("Linting not yet implemented, skipping")
            return []
        except Exception as e:
            logger.debug(f"Linting error (non-blocking): {e}")
            return []

    def _convert_findings(self, findings: List[Any]) -> List[Finding]:
        """
        Convert internal findings to Finding model objects.

        Handles both Finding objects and placeholder Finding classes.

        Args:
            findings: List of findings from validators

        Returns:
            List of Finding model objects
        """
        converted = []
        for f in findings:
            if isinstance(f, Finding):
                converted.append(f)
            elif hasattr(f, 'code') and hasattr(f, 'severity') and hasattr(f, 'path') and hasattr(f, 'message'):
                # Convert placeholder Finding class to model
                severity = f.severity
                if isinstance(severity, str):
                    severity = Severity(severity)

                converted.append(Finding(
                    code=f.code,
                    severity=severity,
                    path=f.path,
                    message=f.message,
                    expected=getattr(f, 'expected', None),
                    actual=getattr(f, 'actual', None),
                    hint=getattr(f, 'hint', None),
                ))
        return converted

    def _should_stop(
        self,
        findings: List[Finding],
        options: ValidationOptions,
    ) -> bool:
        """
        Check if validation should stop (fail_fast or max_errors).

        Args:
            findings: Current list of findings
            options: Validation options

        Returns:
            True if validation should stop early
        """
        error_count = sum(1 for f in findings if f.is_error())

        if options.fail_fast and error_count > 0:
            logger.debug("Stopping validation due to fail_fast")
            return True

        if error_count >= options.max_errors:
            logger.debug(f"Stopping validation due to max_errors ({error_count})")
            return True

        return False

    def _create_summary(
        self,
        findings: List[Finding],
    ) -> ValidationSummary:
        """
        Create summary from findings.

        Args:
            findings: List of validation findings

        Returns:
            ValidationSummary with counts
        """
        return ValidationSummary.from_findings(findings)

    def _sort_findings(
        self,
        findings: List[Finding],
    ) -> List[Finding]:
        """
        Sort findings deterministically.

        Sort order:
            1. Severity (error > warning > info)
            2. Path (lexicographic)
            3. Code (lexicographic)

        Args:
            findings: List of findings to sort

        Returns:
            Sorted list of findings
        """
        return sorted(
            findings,
            key=lambda f: (
                SEVERITY_ORDER.get(f.severity.value if hasattr(f.severity, 'value') else f.severity, 3),
                f.path,
                f.code,
            )
        )

    def _create_error_report(
        self,
        schema_ref: SchemaRef,
        findings: List[Finding],
        timings: Dict[str, float],
        start_time: float,
    ) -> ValidationReport:
        """
        Create error report when validation cannot proceed.

        Args:
            schema_ref: Schema reference
            findings: Error findings
            timings: Timing data so far
            start_time: Validation start time

        Returns:
            ValidationReport with errors
        """
        total_time = (time.perf_counter() - start_time) * 1000
        timings["total_ms"] = total_time

        summary = self._create_summary(findings)

        timing_info = TimingInfo(
            parse_ms=timings.get("parse_ms"),
            compile_ms=timings.get("compile_ms"),
            validate_ms=timings.get("validate_ms"),
            total_ms=total_time,
        )

        return ValidationReport(
            valid=False,
            schema_ref=schema_ref,
            schema_hash="0" * 64,  # Placeholder hash for failed validation
            summary=summary,
            findings=findings,
            normalized_payload=None,
            fix_suggestions=None,
            timings=timing_info,
        )

    def _create_batch_error_report(
        self,
        schema_ref: SchemaRef,
        compile_findings: List[Finding],
        num_payloads: int,
        start_time: float,
    ) -> BatchValidationReport:
        """
        Create batch error report when schema compilation fails.

        Args:
            schema_ref: Schema reference
            compile_findings: Compilation error findings
            num_payloads: Number of payloads in batch
            start_time: Batch start time

        Returns:
            BatchValidationReport with errors for all items
        """
        results = [
            ItemResult(
                index=i,
                valid=False,
                findings=compile_findings,
            )
            for i in range(num_payloads)
        ]

        summary = BatchSummary(
            total_items=num_payloads,
            valid_count=0,
            error_count=num_payloads,
            warning_count=0,
        )

        return BatchValidationReport(
            schema_ref=schema_ref,
            schema_hash="0" * 64,
            summary=summary,
            results=results,
        )

    def _validate_single_for_batch(
        self,
        payload: Union[str, Dict[str, Any]],
        ir: SchemaIR,
        index: int,
        options: ValidationOptions,
    ) -> ItemResult:
        """
        Validate a single payload for batch processing.

        Uses pre-compiled IR for efficiency.

        Args:
            payload: Payload to validate
            ir: Pre-compiled schema IR
            index: Index in batch
            options: Validation options

        Returns:
            ItemResult for this payload
        """
        # Parse payload
        parsed_payload, parse_findings = self._parse_payload(payload)

        if parse_findings:
            return ItemResult(
                index=index,
                valid=False,
                findings=parse_findings,
            )

        # Run validation phases
        findings, _ = self._run_validation_phases(
            payload=parsed_payload,
            ir=ir,
            options=options,
        )

        # Sort findings
        sorted_findings = self._sort_findings(findings)

        # Check if valid
        is_valid = not any(f.is_error() for f in sorted_findings)

        return ItemResult(
            index=index,
            valid=is_valid,
            findings=sorted_findings,
            normalized_payload=parsed_payload if options.normalize else None,
        )

    def clear_cache(self) -> None:
        """
        Clear the IR cache.

        Call this method to force re-compilation of schemas on next validation.
        Useful when schemas have been updated.
        """
        self._ir_cache.clear()
        logger.info("IR cache cleared")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def validate(
    payload: Union[str, Dict[str, Any]],
    schema: Union[SchemaRef, str, Dict[str, Any]],
    profile: ValidationProfile = ValidationProfile.STANDARD,
    **options,
) -> ValidationReport:
    """
    Validate a payload against a schema.

    This is the main entry point for schema validation, providing a simple
    function-based API.

    Args:
        payload: Payload as YAML/JSON string or dict.
        schema: Schema as SchemaRef, URI string, or inline dict.
        profile: Validation profile (STRICT, STANDARD, or PERMISSIVE).
        **options: Additional validation options:
            - normalize: bool - Normalize payload to canonical form
            - emit_patches: bool - Generate fix suggestions
            - patch_level: str - Safety level for patches ("safe", "needs_review", "unsafe")
            - max_errors: int - Maximum errors to collect
            - fail_fast: bool - Stop at first error
            - unit_system: str - Unit system for conversions ("SI", "IMPERIAL")
            - unknown_field_policy: str - Handle unknown fields ("error", "warn", "ignore")
            - coercion_policy: str - Type coercion policy ("off", "safe", "aggressive")

    Returns:
        ValidationReport with validation results.

    Examples:
        # Simple validation
        >>> result = validate(payload, "gl://schemas/activity@1.0.0")

        # With strict profile
        >>> result = validate(
        ...     payload,
        ...     schema_ref,
        ...     profile=ValidationProfile.STRICT,
        ...     normalize=True
        ... )

        # With inline schema
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> result = validate({"name": "test"}, schema)
    """
    # Build validation options
    validation_options = ValidationOptions(
        profile=profile,
        **{k: v for k, v in options.items() if v is not None}
    )

    # Create validator
    validator = SchemaValidator(options=validation_options)

    # Handle inline schema
    if isinstance(schema, dict):
        # For inline schemas, we need to compile directly
        # Create a temporary schema reference
        schema_ref = SchemaRef(schema_id="_inline", version="1.0.0")

        # Compile the schema
        result = validator._compiler.compile(
            schema_source=schema,
            schema_id="_inline",
            version="1.0.0",
        )

        if not result.success:
            return validator._create_error_report(
                schema_ref=schema_ref,
                findings=[
                    Finding(
                        code="GLSCHEMA-E502",
                        severity=Severity.ERROR,
                        path="",
                        message=f"Schema compilation failed: {error}",
                    )
                    for error in result.errors
                ],
                timings={},
                start_time=time.perf_counter(),
            )

        # Cache the IR
        validator._ir_cache[schema_ref.to_cache_key()] = result.ir

        return validator.validate(payload, schema_ref, validation_options)
    else:
        # SchemaRef or URI string
        return validator.validate(payload, schema, validation_options)


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "SchemaValidator",
    "validate",
]
