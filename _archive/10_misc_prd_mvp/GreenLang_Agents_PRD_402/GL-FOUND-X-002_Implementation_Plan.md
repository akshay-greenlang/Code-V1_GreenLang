# GL-FOUND-X-002 Implementation Plan
# GreenLang Schema Compiler & Validator

**Version:** 1.0.0
**Created:** 2026-01-28
**Status:** Planning
**Timeline:** 8+ weeks to MVP

---

## Executive Summary

This document outlines the complete implementation plan for GL-FOUND-X-002, the GreenLang Schema Compiler & Validator. The plan is organized into 7 phases with 42 detailed tasks, designed for parallel execution by AI agents.

### Key Decisions (From Interview)
- **Schema Grammar:** JSON Schema Draft 2020-12 + GreenLang extensions
- **Registry:** Hybrid Git + HTTP cache service
- **Rule DSL:** Extend existing PolicyEngine DSL
- **Canonical Units:** SI units (kWh, kg, m²)
- **Language:** Python core + Rust hot paths (later)
- **Deployment:** Embedded library + optional FastAPI service
- **Error Codes:** GLSCHEMA-* prefix

---

## Phase 0: Foundation & Project Setup
**Duration:** Week 1
**Dependencies:** None
**Parallelizable:** Yes (all tasks)

### Task 0.1: Create Project Structure
**Priority:** P0 (Blocking)
**Estimated Effort:** 2 hours
**Agent Type:** Backend Developer

**Description:**
Create the directory structure and package scaffolding for GL-FOUND-X-002.

**Deliverables:**
```
greenlang/schema/
├── __init__.py
├── compiler/
│   ├── __init__.py
│   ├── parser.py          # Schema parser (YAML/JSON)
│   ├── ast.py             # Schema AST definitions
│   ├── ir.py              # Intermediate Representation
│   ├── compiler.py        # Main compiler orchestration
│   └── resolver.py        # $ref resolution
├── validator/
│   ├── __init__.py
│   ├── core.py            # Validator core engine
│   ├── structural.py      # Structural validation
│   ├── constraints.py     # Constraint validation
│   ├── units.py           # Unit validation
│   └── rules.py           # Cross-field rules
├── normalizer/
│   ├── __init__.py
│   ├── engine.py          # Normalization engine
│   ├── coercions.py       # Type coercions
│   └── canonicalizer.py   # Key/unit canonicalization
├── suggestions/
│   ├── __init__.py
│   ├── engine.py          # Fix suggestion engine
│   ├── patches.py         # JSON Patch generation
│   └── safety.py          # Patch safety classification
├── registry/
│   ├── __init__.py
│   ├── resolver.py        # Schema resolver interface
│   ├── git_backend.py     # Git-backed registry
│   ├── cache.py           # IR cache service
│   └── client.py          # HTTP registry client
├── cli/
│   ├── __init__.py
│   ├── main.py            # CLI entry point
│   ├── commands/
│   │   ├── validate.py    # validate command
│   │   ├── compile.py     # compile command
│   │   ├── lint.py        # lint command
│   │   └── migrate.py     # migrate command
│   └── formatters/
│       ├── json.py        # JSON output
│       ├── table.py       # Table output
│       ├── sarif.py       # SARIF output
│       └── pretty.py      # Colorized output
├── api/
│   ├── __init__.py
│   ├── routes.py          # FastAPI routes
│   ├── models.py          # Request/Response models
│   └── dependencies.py    # Dependency injection
├── models/
│   ├── __init__.py
│   ├── schema_ref.py      # SchemaRef model
│   ├── finding.py         # Validation finding
│   ├── report.py          # Validation report
│   ├── patch.py           # Fix suggestion/patch
│   └── config.py          # Validation options
├── units/
│   ├── __init__.py
│   ├── catalog.py         # Unit catalog
│   ├── dimensions.py      # Dimension definitions
│   ├── conversions.py     # Unit conversions
│   └── packs/
│       ├── si.py          # SI units (core)
│       ├── climate.py     # Climate domain units
│       └── finance.py     # Finance domain units
├── errors.py              # Error codes (GLSCHEMA-*)
├── constants.py           # Constants and limits
└── version.py             # Version info
```

**Acceptance Criteria:**
- [ ] All directories created with `__init__.py`
- [ ] Package installable via `pip install -e .`
- [ ] Imports work: `from greenlang.schema import *`

---

### Task 0.2: Define Error Code Taxonomy
**Priority:** P0 (Blocking)
**Estimated Effort:** 3 hours
**Agent Type:** Backend Developer

**Description:**
Define all error codes with the GLSCHEMA-* prefix following the PRD taxonomy.

**Deliverables:**
- `greenlang/schema/errors.py` with complete error code enum
- Error code documentation

**Error Code Categories:**
```python
# Structural Errors (GLSCHEMA-E1xx)
GLSCHEMA-E100  # MISSING_REQUIRED - Required field missing
GLSCHEMA-E101  # UNKNOWN_FIELD - Unknown field in strict mode
GLSCHEMA-E102  # TYPE_MISMATCH - Type does not match schema
GLSCHEMA-E103  # INVALID_NULL - Null not allowed

# Constraint Errors (GLSCHEMA-E2xx)
GLSCHEMA-E200  # RANGE_VIOLATION - Value outside min/max
GLSCHEMA-E201  # PATTERN_MISMATCH - String doesn't match pattern
GLSCHEMA-E202  # ENUM_VIOLATION - Value not in enum
GLSCHEMA-E203  # LENGTH_VIOLATION - String/array length invalid
GLSCHEMA-E204  # UNIQUE_VIOLATION - Array items not unique

# Unit Errors (GLSCHEMA-E3xx)
GLSCHEMA-E300  # UNIT_MISSING - Required unit not provided
GLSCHEMA-E301  # UNIT_INCOMPATIBLE - Unit dimension mismatch
GLSCHEMA-E302  # UNIT_NONCANONICAL - Unit not in canonical form
GLSCHEMA-E303  # UNIT_UNKNOWN - Unit not in catalog

# Rule Errors (GLSCHEMA-E4xx)
GLSCHEMA-E400  # RULE_VIOLATION - Cross-field rule failed
GLSCHEMA-E401  # CONDITIONAL_REQUIRED - Conditional requirement not met
GLSCHEMA-E402  # CONSISTENCY_ERROR - Consistency check failed

# Schema Errors (GLSCHEMA-E5xx)
GLSCHEMA-E500  # REF_RESOLUTION_FAILED - $ref target not found
GLSCHEMA-E501  # CIRCULAR_REF - Circular reference detected
GLSCHEMA-E502  # SCHEMA_INVALID - Schema itself is invalid
GLSCHEMA-E503  # SCHEMA_VERSION_MISMATCH - Incompatible schema version

# Deprecation Warnings (GLSCHEMA-W6xx)
GLSCHEMA-W600  # DEPRECATED_FIELD - Using deprecated field
GLSCHEMA-W601  # RENAMED_FIELD - Field has been renamed
GLSCHEMA-W602  # REMOVED_FIELD - Field will be removed

# Lint Warnings (GLSCHEMA-W7xx)
GLSCHEMA-W700  # SUSPICIOUS_KEY - Possible typo in field name
GLSCHEMA-W701  # NONCOMPLIANT_CASING - Key doesn't follow naming convention
GLSCHEMA-W702  # UNIT_FORMAT_STYLE - Unit format could be improved

# Limit Errors (GLSCHEMA-E8xx)
GLSCHEMA-E800  # PAYLOAD_TOO_LARGE - Exceeds max payload size
GLSCHEMA-E801  # DEPTH_EXCEEDED - Exceeds max nesting depth
GLSCHEMA-E802  # ITEMS_EXCEEDED - Exceeds max array items
GLSCHEMA-E803  # REFS_EXCEEDED - Too many $ref expansions
GLSCHEMA-E804  # FINDINGS_EXCEEDED - Too many findings
```

**Acceptance Criteria:**
- [ ] All error codes defined with messages
- [ ] Error codes are stable identifiers
- [ ] Codes follow PRD naming convention

---

### Task 0.3: Define Core Data Models
**Priority:** P0 (Blocking)
**Estimated Effort:** 4 hours
**Agent Type:** Backend Developer

**Description:**
Define all Pydantic v2 models for the validator using existing codebase patterns.

**Deliverables:**
- `greenlang/schema/models/schema_ref.py` - SchemaRef model
- `greenlang/schema/models/finding.py` - Finding model
- `greenlang/schema/models/report.py` - ValidationReport model
- `greenlang/schema/models/patch.py` - FixSuggestion model
- `greenlang/schema/models/config.py` - ValidationOptions model

**Key Models:**
```python
class SchemaRef(BaseModel):
    schema_id: str
    version: str
    variant: Optional[str] = None

class ValidationProfile(str, Enum):
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"

class Finding(BaseModel):
    code: str  # GLSCHEMA-E100
    severity: Literal["error", "warning", "info"]
    path: str  # JSON Pointer (RFC 6901)
    message: str
    expected: Optional[Dict[str, Any]] = None
    actual: Optional[Any] = None
    hint: Optional[Dict[str, Any]] = None

class FixSuggestion(BaseModel):
    patch: List[JSONPatchOp]
    preconditions: List[JSONPatchOp]
    confidence: float  # 0.0-1.0
    safety: Literal["safe", "needs_review", "unsafe"]
    rationale: str

class ValidationReport(BaseModel):
    valid: bool
    schema_ref: SchemaRef
    schema_hash: str
    summary: ValidationSummary
    findings: List[Finding]
    normalized_payload: Optional[Dict[str, Any]] = None
    fix_suggestions: Optional[List[FixSuggestion]] = None
    timings_ms: Dict[str, float]

class ValidationOptions(BaseModel):
    profile: ValidationProfile = ValidationProfile.STANDARD
    normalize: bool = True
    emit_patches: bool = True
    patch_level: Literal["safe", "needs_review", "unsafe"] = "safe"
    max_errors: int = 100
    fail_fast: bool = False
    unit_system: str = "SI"
    unknown_field_policy: Literal["error", "warn", "ignore"] = "warn"
    coercion_policy: Literal["off", "safe", "aggressive"] = "safe"
```

**Acceptance Criteria:**
- [ ] All models use Pydantic v2 with validation
- [ ] Models are JSON-serializable
- [ ] Models match PRD specification

---

### Task 0.4: Define Constants and Limits
**Priority:** P0 (Blocking)
**Estimated Effort:** 1 hour
**Agent Type:** Backend Developer

**Description:**
Define all configurable limits and constants per PRD section 6.10.

**Deliverables:**
- `greenlang/schema/constants.py`

**Constants:**
```python
# Size Limits
MAX_PAYLOAD_BYTES = 1_048_576  # 1 MB
MAX_SCHEMA_BYTES = 2_097_152   # 2 MB
MAX_OBJECT_DEPTH = 50
MAX_ARRAY_ITEMS = 10_000
MAX_TOTAL_NODES = 200_000
MAX_REF_EXPANSIONS = 10_000
MAX_FINDINGS = 100

# Regex Limits
MAX_REGEX_LENGTH = 1000
REGEX_TIMEOUT_MS = 100

# Cache Settings
SCHEMA_CACHE_TTL_SECONDS = 3600
SCHEMA_CACHE_MAX_SIZE = 1000

# Performance Targets
P95_LATENCY_SMALL_MS = 25    # <50KB
P95_LATENCY_MEDIUM_MS = 150  # <500KB

# Deprecation
DEPRECATION_WARNING_DAYS = 90

# Batch Limits
MAX_BATCH_ITEMS = 1000
MAX_BATCH_BYTES = 10_485_760  # 10 MB
MAX_BATCH_TIME_SECONDS = 60
```

**Acceptance Criteria:**
- [ ] All limits from PRD defined
- [ ] Limits are configurable via environment variables
- [ ] Clear documentation for each limit

---

### Task 0.5: Setup Testing Infrastructure
**Priority:** P0 (Blocking)
**Estimated Effort:** 3 hours
**Agent Type:** Test Engineer

**Description:**
Setup pytest infrastructure with golden tests and property-based testing (Hypothesis).

**Deliverables:**
```
tests/schema/
├── conftest.py           # Shared fixtures
├── golden/               # Golden test data
│   ├── schemas/          # Test schemas
│   ├── payloads/
│   │   ├── valid/        # Valid payloads
│   │   └── invalid/      # Invalid payloads
│   └── expected/         # Expected outputs
├── unit/
│   ├── test_parser.py
│   ├── test_compiler.py
│   ├── test_validator.py
│   ├── test_normalizer.py
│   └── test_suggestions.py
├── integration/
│   ├── test_cli.py
│   ├── test_api.py
│   └── test_registry.py
├── property/
│   ├── test_normalization_idempotent.py
│   └── test_patch_monotonic.py
└── security/
    ├── test_redos.py
    ├── test_yaml_bombs.py
    └── test_schema_bombs.py
```

**Acceptance Criteria:**
- [ ] pytest configured with markers
- [ ] Hypothesis installed for property tests
- [ ] Golden test loader implemented
- [ ] Coverage reporting enabled

---

## Phase 1: Schema Parser & Compiler
**Duration:** Week 2-3
**Dependencies:** Phase 0
**Parallelizable:** Partial (1.1-1.3 parallel, then 1.4-1.6)

### Task 1.1: Implement Safe YAML/JSON Parser
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement secure YAML/JSON parser with size limits, depth limits, and protection against YAML bombs (billion laughs attack).

**Deliverables:**
- `greenlang/schema/compiler/parser.py`

**Requirements:**
- Use `yaml.SafeLoader` only (no arbitrary Python objects)
- Enforce MAX_PAYLOAD_BYTES before parsing
- Track and enforce MAX_OBJECT_DEPTH during parsing
- Count nodes and enforce MAX_TOTAL_NODES
- Disable YAML anchors/aliases to prevent bombs
- Support both YAML and JSON input detection
- Return structured parse result with metadata

**API:**
```python
class ParseResult(BaseModel):
    data: Dict[str, Any]
    format: Literal["yaml", "json"]
    size_bytes: int
    node_count: int
    max_depth: int
    parse_time_ms: float

def parse_payload(
    content: Union[str, bytes],
    max_bytes: int = MAX_PAYLOAD_BYTES,
    max_depth: int = MAX_OBJECT_DEPTH,
    max_nodes: int = MAX_TOTAL_NODES
) -> ParseResult:
    """Parse YAML/JSON with safety limits."""
```

**Acceptance Criteria:**
- [ ] Parses valid YAML and JSON
- [ ] Rejects payloads exceeding size limit
- [ ] Rejects payloads exceeding depth limit
- [ ] Rejects YAML bombs (billion laughs)
- [ ] Returns structured metadata

**Security Tests:**
- Billion laughs YAML
- Deeply nested structures (depth 100+)
- 10MB+ payloads
- Malformed UTF-8

---

### Task 1.2: Define Schema AST
**Priority:** P0 (Blocking)
**Estimated Effort:** 8 hours
**Agent Type:** Backend Developer

**Description:**
Define the Abstract Syntax Tree (AST) for GreenLang schemas, supporting JSON Schema Draft 2020-12 with GreenLang extensions.

**Deliverables:**
- `greenlang/schema/compiler/ast.py`

**AST Node Types:**
```python
class SchemaNode(BaseModel):
    """Base class for all schema AST nodes."""
    node_id: str  # Unique identifier
    location: Optional[str] = None  # Source location

class SchemaDocument(SchemaNode):
    """Root schema document."""
    schema_id: str
    version: str
    dialect: str = "https://json-schema.org/draft/2020-12/schema"
    root: TypeNode
    definitions: Dict[str, TypeNode] = {}
    gl_extensions: GreenLangExtensions

class TypeNode(SchemaNode):
    """Type definition node."""
    type: Optional[Union[str, List[str]]] = None

class ObjectTypeNode(TypeNode):
    properties: Dict[str, TypeNode] = {}
    required: List[str] = []
    additional_properties: Union[bool, TypeNode] = True
    property_names: Optional[StringTypeNode] = None
    min_properties: Optional[int] = None
    max_properties: Optional[int] = None
    dependencies: Dict[str, Union[List[str], TypeNode]] = {}

class ArrayTypeNode(TypeNode):
    items: Optional[TypeNode] = None
    prefix_items: List[TypeNode] = []
    contains: Optional[TypeNode] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: bool = False

class StringTypeNode(TypeNode):
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    format: Optional[str] = None

class NumericTypeNode(TypeNode):
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: Optional[float] = None
    exclusive_maximum: Optional[float] = None
    multiple_of: Optional[float] = None

class RefNode(TypeNode):
    ref: str  # $ref value
    resolved: Optional[TypeNode] = None  # After resolution

# GreenLang Extensions
class GreenLangExtensions(BaseModel):
    unit: Optional[UnitSpec] = None
    dimension: Optional[str] = None
    rules: List[RuleBinding] = []
    aliases: Dict[str, str] = {}
    deprecated: Optional[DeprecationInfo] = None
    renamed_from: Optional[str] = None

class UnitSpec(BaseModel):
    dimension: str  # e.g., "energy", "mass"
    canonical: str  # e.g., "kWh", "kg"
    allowed: List[str] = []  # Allowed input units

class DeprecationInfo(BaseModel):
    since_version: str
    message: str
    replacement: Optional[str] = None
    removal_version: Optional[str] = None
```

**Acceptance Criteria:**
- [ ] All JSON Schema Draft 2020-12 keywords supported
- [ ] GreenLang extensions ($unit, $dimension, $rules, $aliases) supported
- [ ] AST nodes are immutable after creation
- [ ] AST can be serialized to JSON for debugging

---

### Task 1.3: Implement Schema Resolver ($ref)
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement $ref resolution with cycle detection, supporting local refs, external refs, and the schema registry.

**Deliverables:**
- `greenlang/schema/compiler/resolver.py`

**Requirements:**
- Resolve local refs (`#/definitions/Foo`)
- Resolve external refs (`gl://schemas/emissions/activity@1.3.0`)
- Detect circular references with clear cycle trace
- Enforce MAX_REF_EXPANSIONS limit
- Cache resolved refs within a compilation session

**API:**
```python
class RefResolver:
    def __init__(
        self,
        schema_registry: SchemaRegistry,
        max_expansions: int = MAX_REF_EXPANSIONS
    ):
        self.registry = schema_registry
        self.max_expansions = max_expansions
        self._cache: Dict[str, TypeNode] = {}
        self._resolution_stack: List[str] = []
        self._expansion_count: int = 0

    def resolve(self, ref: str, context: SchemaDocument) -> TypeNode:
        """Resolve a $ref to its target TypeNode."""

    def _detect_cycle(self, ref: str) -> Optional[List[str]]:
        """Check for circular reference, return cycle trace if found."""
```

**Acceptance Criteria:**
- [ ] Local refs resolve correctly
- [ ] External refs fetch from registry
- [ ] Circular refs detected with trace
- [ ] Expansion limit enforced
- [ ] Cached refs reused

**Test Cases:**
- Simple local ref
- Nested refs (A -> B -> C)
- Circular ref (A -> B -> A)
- Deep chain (50+ refs)
- External schema ref

---

### Task 1.4: Implement Schema Compiler (AST -> IR)
**Priority:** P0 (Blocking)
**Estimated Effort:** 10 hours
**Agent Type:** Backend Developer
**Dependencies:** Task 1.2, Task 1.3

**Description:**
Compile parsed schema AST into optimized Intermediate Representation (IR) for fast validation.

**Deliverables:**
- `greenlang/schema/compiler/ir.py`
- `greenlang/schema/compiler/compiler.py`

**IR Structure:**
```python
class SchemaIR(BaseModel):
    """Compiled schema intermediate representation."""
    schema_id: str
    version: str
    schema_hash: str  # SHA-256 of canonical schema
    compiled_at: datetime
    compiler_version: str

    # Flattened property maps for O(1) lookup
    properties: Dict[str, PropertyIR]
    required_paths: Set[str]

    # Precompiled constraints
    numeric_constraints: Dict[str, NumericConstraintIR]
    string_constraints: Dict[str, StringConstraintIR]
    array_constraints: Dict[str, ArrayConstraintIR]

    # Precompiled regexes with safety metadata
    patterns: Dict[str, CompiledPattern]

    # Unit metadata
    unit_specs: Dict[str, UnitSpecIR]

    # Rule bindings
    rule_bindings: List[RuleBindingIR]

    # Deprecation index
    deprecated_fields: Dict[str, DeprecationInfo]
    renamed_fields: Dict[str, str]  # old -> new

class CompiledPattern(BaseModel):
    pattern: str
    compiled: Any  # re.Pattern
    complexity_score: float
    is_safe: bool  # No catastrophic backtracking

class SchemaCompiler:
    def compile(
        self,
        schema_source: Union[str, Dict],
        schema_ref: SchemaRef
    ) -> CompilationResult:
        """Compile schema source to IR."""

    def _compute_schema_hash(self, ast: SchemaDocument) -> str:
        """Compute stable SHA-256 hash of canonical schema."""

    def _analyze_regex_safety(self, pattern: str) -> Tuple[bool, float]:
        """Analyze regex for ReDoS vulnerability."""
```

**Acceptance Criteria:**
- [ ] Compiles valid schemas to IR
- [ ] Computes stable schema_hash
- [ ] Flattens property maps for O(1) lookup
- [ ] Precompiles regexes with safety analysis
- [ ] Emits compilation warnings

---

### Task 1.5: Implement Regex Safety Analyzer
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement regex complexity analysis to detect potential ReDoS patterns.

**Deliverables:**
- `greenlang/schema/compiler/regex_analyzer.py`

**Requirements:**
- Detect nested quantifiers (e.g., `(a+)+`)
- Detect overlapping alternations
- Compute complexity score
- Reject or sandbox dangerous patterns
- Support RE2-compatible subset detection

**API:**
```python
class RegexAnalysisResult(BaseModel):
    pattern: str
    is_safe: bool
    complexity_score: float  # 0.0 (safe) to 1.0 (dangerous)
    vulnerability_type: Optional[str]  # "nested_quantifier", "overlapping_alt"
    recommendation: str

def analyze_regex_safety(pattern: str) -> RegexAnalysisResult:
    """Analyze regex pattern for ReDoS vulnerability."""

def is_re2_compatible(pattern: str) -> bool:
    """Check if pattern is compatible with RE2 (no backtracking)."""
```

**Acceptance Criteria:**
- [ ] Detects known ReDoS patterns
- [ ] Assigns complexity scores
- [ ] Recommends alternatives
- [ ] Timeout protection for analysis itself

**Test Cases:**
- `(a+)+` - nested quantifier
- `(a|a)+` - overlapping alternation
- `^[a-zA-Z0-9]+$` - safe pattern
- `.*.*.*.*x` - backtracking bomb

---

### Task 1.6: Implement Schema Self-Validation
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer
**Dependencies:** Task 1.4

**Description:**
Validate schema documents themselves for governance compliance per PRD 6.8.

**Deliverables:**
- `greenlang/schema/compiler/schema_validator.py`

**Validation Checks:**
1. Reference resolution (no missing refs)
2. Cycle detection with clear trace
3. No duplicate property keys after alias resolution
4. Deprecated field metadata well-formed
5. Unit metadata consistent (unit in catalog, dimension specified)
6. Constraints internally consistent (min <= max)
7. Rule expressions type-checkable

**API:**
```python
class SchemaValidationResult(BaseModel):
    valid: bool
    errors: List[Finding]
    warnings: List[Finding]

def validate_schema(
    schema_source: Union[str, Dict],
    strict: bool = True
) -> SchemaValidationResult:
    """Validate a schema document for governance compliance."""
```

**Acceptance Criteria:**
- [ ] All PRD 6.8 checks implemented
- [ ] Clear error messages with paths
- [ ] Supports both strict and lenient modes

---

## Phase 2: Validator Core
**Duration:** Week 3-4
**Dependencies:** Phase 1
**Parallelizable:** Yes (2.1-2.4 parallel)

### Task 2.1: Implement Structural Validator
**Priority:** P0 (Blocking)
**Estimated Effort:** 8 hours
**Agent Type:** Backend Developer

**Description:**
Implement structural validation (shape, types, required fields) against compiled IR.

**Deliverables:**
- `greenlang/schema/validator/structural.py`

**Validations:**
- Required field presence (including nested)
- Type checking (string, number, integer, boolean, object, array, null)
- Additional properties policy
- Property count constraints
- Array item count constraints

**API:**
```python
class StructuralValidator:
    def __init__(self, ir: SchemaIR, options: ValidationOptions):
        self.ir = ir
        self.options = options

    def validate(
        self,
        payload: Dict[str, Any],
        path: str = ""
    ) -> List[Finding]:
        """Validate payload structure against schema IR."""
```

**Acceptance Criteria:**
- [ ] Validates all structural constraints
- [ ] Reports precise JSON Pointer paths
- [ ] Respects validation profile (strict/standard/permissive)
- [ ] Handles nested objects/arrays

---

### Task 2.2: Implement Constraint Validator
**Priority:** P0 (Blocking)
**Estimated Effort:** 8 hours
**Agent Type:** Backend Developer

**Description:**
Implement constraint validation (ranges, patterns, enums).

**Deliverables:**
- `greenlang/schema/validator/constraints.py`

**Validations:**
- Numeric: min/max, exclusive bounds, multipleOf
- String: pattern, minLength/maxLength, format
- Array: minItems/maxItems, uniqueItems, contains
- Enum validation

**API:**
```python
class ConstraintValidator:
    def __init__(self, ir: SchemaIR, options: ValidationOptions):
        self.ir = ir
        self.options = options

    def validate_numeric(
        self,
        value: Union[int, float],
        constraints: NumericConstraintIR,
        path: str
    ) -> List[Finding]:
        """Validate numeric value against constraints."""

    def validate_string(
        self,
        value: str,
        constraints: StringConstraintIR,
        path: str
    ) -> List[Finding]:
        """Validate string value against constraints."""

    def validate_array(
        self,
        value: List,
        constraints: ArrayConstraintIR,
        path: str
    ) -> List[Finding]:
        """Validate array against constraints."""
```

**Acceptance Criteria:**
- [ ] All constraint types validated
- [ ] Regex matching with timeout protection
- [ ] Clear error messages with expected vs actual
- [ ] Respects coercion policy

---

### Task 2.3: Implement Unit Validator
**Priority:** P0 (Blocking)
**Estimated Effort:** 10 hours
**Agent Type:** Backend Developer

**Description:**
Implement unit validation and conversion, integrating with existing UnitConverter.

**Deliverables:**
- `greenlang/schema/validator/units.py`
- `greenlang/schema/units/catalog.py`
- `greenlang/schema/units/dimensions.py`

**Requirements:**
- Detect missing units when schema requires units
- Detect incompatible units (e.g., kg vs kWh)
- Validate unit is in allowed catalog
- Support multiple input formats:
  - `{ "value": 10, "unit": "kWh" }`
  - `"10 kWh"` (string parsing)
  - Separate fields (`energy_value`, `energy_unit`)

**API:**
```python
class UnitValidator:
    def __init__(self, catalog: UnitCatalog, options: ValidationOptions):
        self.catalog = catalog
        self.options = options

    def validate(
        self,
        value: Any,
        unit_spec: UnitSpecIR,
        path: str
    ) -> Tuple[List[Finding], Optional[NormalizedUnit]]:
        """Validate unit and optionally convert to canonical."""

class UnitCatalog:
    def __init__(self):
        self._units: Dict[str, UnitDefinition] = {}
        self._dimensions: Dict[str, List[str]] = {}

    def register_unit(self, unit: UnitDefinition):
        """Register a unit in the catalog."""

    def is_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are dimensionally compatible."""

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between compatible units."""
```

**Acceptance Criteria:**
- [ ] Validates unit presence
- [ ] Validates dimensional compatibility
- [ ] Converts to canonical SI units
- [ ] Supports all PRD input formats
- [ ] Integrates with existing UnitConverter

---

### Task 2.4: Implement Rule Engine Integration
**Priority:** P1 (High)
**Estimated Effort:** 8 hours
**Agent Type:** Backend Developer

**Description:**
Integrate with existing PolicyEngine DSL for cross-field rule evaluation.

**Deliverables:**
- `greenlang/schema/validator/rules.py`

**Requirements:**
- Reuse expression evaluator from `policy_engine.py`
- Support conditional requirements
- Support consistency checks (sum of components = total)
- Support range dependencies

**API:**
```python
class RuleValidator:
    def __init__(self, ir: SchemaIR, options: ValidationOptions):
        self.ir = ir
        self.options = options
        self._evaluator = ExpressionEvaluator()  # From PolicyEngine

    def validate(
        self,
        payload: Dict[str, Any],
        rules: List[RuleBindingIR]
    ) -> List[Finding]:
        """Evaluate cross-field rules against payload."""

class RuleBindingIR(BaseModel):
    rule_id: str
    severity: Literal["error", "warning", "info"]
    when: Optional[Dict]  # Condition expression
    check: Dict  # Validation expression
    message: str
    message_template: Optional[str]  # With {{ var }} support
```

**Acceptance Criteria:**
- [ ] Evaluates conditional requirements
- [ ] Evaluates consistency checks
- [ ] Supports all PolicyEngine operators
- [ ] Clear error messages with rule ID

---

### Task 2.5: Implement Validator Core Orchestration
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer
**Dependencies:** Task 2.1, 2.2, 2.3, 2.4

**Description:**
Implement the main validator that orchestrates all validation phases.

**Deliverables:**
- `greenlang/schema/validator/core.py`

**Validation Order:**
1. Parse payload (with limits)
2. Resolve schema (with caching)
3. Structural validation
4. Constraint validation
5. Unit validation
6. Rule validation
7. Linting (non-blocking)

**API:**
```python
class SchemaValidator:
    def __init__(
        self,
        schema_registry: SchemaRegistry,
        unit_catalog: UnitCatalog,
        options: ValidationOptions = ValidationOptions()
    ):
        self.registry = schema_registry
        self.catalog = unit_catalog
        self.options = options
        self._ir_cache: Dict[str, SchemaIR] = {}

    def validate(
        self,
        payload: Union[str, Dict],
        schema_ref: SchemaRef
    ) -> ValidationReport:
        """Validate payload against schema."""

    def validate_batch(
        self,
        payloads: List[Union[str, Dict]],
        schema_ref: SchemaRef
    ) -> BatchValidationReport:
        """Validate multiple payloads efficiently."""
```

**Acceptance Criteria:**
- [ ] Orchestrates all validation phases
- [ ] Respects fail_fast option
- [ ] Respects max_errors limit
- [ ] Caches compiled IR
- [ ] Records timing for each phase

---

### Task 2.6: Implement Linter
**Priority:** P2 (Medium)
**Estimated Effort:** 4 hours
**Agent Type:** Backend Developer

**Description:**
Implement non-blocking linting checks (warnings).

**Deliverables:**
- `greenlang/schema/validator/linter.py`

**Lint Checks:**
- Unknown fields with close matches (Levenshtein distance)
- Deprecated field usage
- Non-canonical casing
- Unit formatting suggestions
- Suspicious patterns

**API:**
```python
class SchemaLinter:
    def lint(
        self,
        payload: Dict[str, Any],
        ir: SchemaIR
    ) -> List[Finding]:
        """Run non-blocking lint checks."""

    def _find_close_matches(
        self,
        key: str,
        known_keys: Set[str],
        threshold: int = 2
    ) -> List[str]:
        """Find similar keys within edit distance threshold."""
```

**Acceptance Criteria:**
- [ ] Detects typos with suggestions
- [ ] Flags deprecated field usage
- [ ] All findings are warnings (non-blocking)

---

## Phase 3: Normalization Engine
**Duration:** Week 4-5
**Dependencies:** Phase 2
**Parallelizable:** Yes (3.1-3.3 parallel)

### Task 3.1: Implement Type Coercion Engine
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement safe type coercions (string -> number, "true" -> true).

**Deliverables:**
- `greenlang/schema/normalizer/coercions.py`

**Safe Coercions:**
- `"42"` -> `42` (integer)
- `"3.14"` -> `3.14` (float)
- `"true"/"false"` -> `True`/`False` (boolean)
- `"null"` -> `None` (null)

**API:**
```python
class CoercionEngine:
    def __init__(self, policy: Literal["off", "safe", "aggressive"]):
        self.policy = policy

    def coerce(
        self,
        value: Any,
        target_type: str,
        path: str
    ) -> Tuple[Any, Optional[CoercionRecord]]:
        """Coerce value to target type if safe."""

class CoercionRecord(BaseModel):
    path: str
    original_value: Any
    original_type: str
    coerced_value: Any
    coerced_type: str
    reversible: bool
```

**Acceptance Criteria:**
- [ ] Safe coercions only by default
- [ ] Exact parsing (no rounding)
- [ ] Records all coercions for audit
- [ ] Respects coercion policy

---

### Task 3.2: Implement Unit Canonicalizer
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement unit conversion to canonical SI units.

**Deliverables:**
- `greenlang/schema/normalizer/canonicalizer.py`

**Requirements:**
- Convert to canonical unit (e.g., Wh -> kWh)
- Store original unit in `_meta.original_unit`
- Record all conversions in `_meta.conversions`
- Integrate with UnitConverter

**API:**
```python
class UnitCanonicalizer:
    def __init__(self, catalog: UnitCatalog):
        self.catalog = catalog

    def canonicalize(
        self,
        value: Any,
        unit_spec: UnitSpecIR,
        path: str
    ) -> Tuple[Any, List[ConversionRecord]]:
        """Convert to canonical unit."""

class ConversionRecord(BaseModel):
    path: str
    original_value: float
    original_unit: str
    canonical_value: float
    canonical_unit: str
    conversion_factor: float
```

**Acceptance Criteria:**
- [ ] Converts to SI canonical units
- [ ] Preserves original in metadata
- [ ] Handles all supported dimensions
- [ ] Accurate conversions (no precision loss)

---

### Task 3.3: Implement Key Canonicalizer
**Priority:** P2 (Medium)
**Estimated Effort:** 4 hours
**Agent Type:** Backend Developer

**Description:**
Implement key alias resolution and casing normalization.

**Deliverables:**
- `greenlang/schema/normalizer/keys.py`

**Requirements:**
- Resolve known aliases (schema `aliases` map)
- Normalize casing if schema demands
- Stable key ordering for reproducibility

**API:**
```python
class KeyCanonicalizer:
    def __init__(self, ir: SchemaIR):
        self.ir = ir

    def canonicalize(
        self,
        payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[KeyRename]]:
        """Canonicalize all keys in payload."""

class KeyRename(BaseModel):
    path: str
    original_key: str
    canonical_key: str
    reason: Literal["alias", "casing", "typo_correction"]
```

**Acceptance Criteria:**
- [ ] Resolves aliases from schema
- [ ] Normalizes casing
- [ ] Records all renames
- [ ] Stable output ordering

---

### Task 3.4: Implement Normalization Engine
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer
**Dependencies:** Task 3.1, 3.2, 3.3

**Description:**
Implement the main normalization engine that orchestrates all transformations.

**Deliverables:**
- `greenlang/schema/normalizer/engine.py`

**Normalization Order:**
1. Key canonicalization
2. Default application
3. Type coercion
4. Unit canonicalization
5. Add `_meta` block

**API:**
```python
class NormalizationEngine:
    def __init__(
        self,
        ir: SchemaIR,
        catalog: UnitCatalog,
        options: ValidationOptions
    ):
        self.ir = ir
        self.catalog = catalog
        self.options = options

    def normalize(
        self,
        payload: Dict[str, Any]
    ) -> NormalizationResult:
        """Normalize payload to canonical form."""

class NormalizationResult(BaseModel):
    normalized: Dict[str, Any]
    coercions: List[CoercionRecord]
    conversions: List[ConversionRecord]
    renames: List[KeyRename]
    defaults_applied: List[str]
    meta: Dict[str, Any]
```

**Acceptance Criteria:**
- [ ] Normalization is idempotent: `normalize(normalize(x)) == normalize(x)`
- [ ] Deterministic output
- [ ] Complete audit trail in `_meta`
- [ ] Respects all options

**Property Tests:**
- Idempotency
- Preserves valid payloads
- Does not invent required values

---

## Phase 4: Fix Suggestion Engine
**Duration:** Week 5-6
**Dependencies:** Phase 3
**Parallelizable:** Yes (4.1-4.4 parallel)

### Task 4.1: Implement JSON Patch Generator
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement JSON Patch (RFC 6902) generation for fix suggestions.

**Deliverables:**
- `greenlang/schema/suggestions/patches.py`

**Operations:**
- `add` - Add missing field
- `remove` - Remove unknown field
- `replace` - Fix incorrect value
- `move` - Handle renames
- `test` - Preconditions

**API:**
```python
class JSONPatchOp(BaseModel):
    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str  # JSON Pointer
    value: Optional[Any] = None
    from_: Optional[str] = Field(None, alias="from")

class PatchGenerator:
    def generate_add(self, path: str, value: Any) -> JSONPatchOp:
        """Generate add operation."""

    def generate_replace(self, path: str, old: Any, new: Any) -> List[JSONPatchOp]:
        """Generate replace with test precondition."""

    def generate_rename(self, old_path: str, new_path: str) -> List[JSONPatchOp]:
        """Generate move operation for field rename."""
```

**Acceptance Criteria:**
- [ ] Generates valid RFC 6902 patches
- [ ] Includes test preconditions
- [ ] Patches are serializable to JSON

---

### Task 4.2: Implement Patch Safety Classifier
**Priority:** P1 (High)
**Estimated Effort:** 4 hours
**Agent Type:** Backend Developer

**Description:**
Classify patches by safety level (safe, needs_review, unsafe).

**Deliverables:**
- `greenlang/schema/suggestions/safety.py`

**Safety Rules:**
- **safe**: Add optional field with default, coerce exact primitive, apply declared alias
- **needs_review**: Unit conversion, typo correction (edit distance)
- **unsafe**: Infer missing required value (never emit by default)

**API:**
```python
class PatchSafetyClassifier:
    def classify(
        self,
        patch: JSONPatchOp,
        context: PatchContext
    ) -> Tuple[Literal["safe", "needs_review", "unsafe"], str]:
        """Classify patch safety and return rationale."""

class PatchContext(BaseModel):
    finding: Finding
    ir: SchemaIR
    original_value: Optional[Any]
    suggested_value: Any
    derivation: str  # How the suggestion was derived
```

**Acceptance Criteria:**
- [ ] All patches classified
- [ ] Clear rationale for each classification
- [ ] No unsafe patches emitted by default

---

### Task 4.3: Implement Fix Heuristics
**Priority:** P2 (Medium)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement heuristics for generating fix suggestions per PRD Appendix C.

**Deliverables:**
- `greenlang/schema/suggestions/heuristics.py`

**Heuristics:**
1. Rename field when schema provides `renamed_from`
2. Add optional defaults when default exists
3. Coerce safe primitives with exact parsing
4. Unit conversion when dimension matches
5. Close-match unknown keys (edit distance ≤ 2)

**API:**
```python
class FixHeuristics:
    def suggest_for_missing_required(
        self,
        path: str,
        ir_node: PropertyIR
    ) -> Optional[FixSuggestion]:
        """Suggest fix for missing required field."""

    def suggest_for_type_mismatch(
        self,
        path: str,
        value: Any,
        expected_type: str
    ) -> Optional[FixSuggestion]:
        """Suggest fix for type mismatch."""

    def suggest_for_unknown_field(
        self,
        path: str,
        key: str,
        known_keys: Set[str]
    ) -> Optional[FixSuggestion]:
        """Suggest fix for unknown field (typo correction)."""
```

**Acceptance Criteria:**
- [ ] All PRD heuristics implemented
- [ ] Confidence scores assigned
- [ ] Rationale provided for each suggestion

---

### Task 4.4: Implement Fix Suggestion Engine
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer
**Dependencies:** Task 4.1, 4.2, 4.3

**Description:**
Implement the main fix suggestion engine that generates ordered suggestions.

**Deliverables:**
- `greenlang/schema/suggestions/engine.py`

**API:**
```python
class FixSuggestionEngine:
    def __init__(
        self,
        ir: SchemaIR,
        options: ValidationOptions
    ):
        self.ir = ir
        self.options = options
        self._generator = PatchGenerator()
        self._classifier = PatchSafetyClassifier()
        self._heuristics = FixHeuristics(ir)

    def generate(
        self,
        findings: List[Finding],
        payload: Dict[str, Any]
    ) -> List[FixSuggestion]:
        """Generate fix suggestions for findings."""

    def filter_by_safety(
        self,
        suggestions: List[FixSuggestion],
        max_level: Literal["safe", "needs_review", "unsafe"]
    ) -> List[FixSuggestion]:
        """Filter suggestions by safety level."""
```

**Acceptance Criteria:**
- [ ] Generates suggestions for all fixable findings
- [ ] Filters by patch_level option
- [ ] Orders suggestions by confidence
- [ ] No unsafe patches unless explicitly requested

**Property Tests:**
- Applying safe patches reduces error count
- Patches are valid JSON Patch operations

---

## Phase 5: CLI & API
**Duration:** Week 6-7
**Dependencies:** Phase 4
**Parallelizable:** Yes (5.1 and 5.2 parallel)

### Task 5.1: Implement CLI
**Priority:** P0 (Blocking)
**Estimated Effort:** 10 hours
**Agent Type:** Backend Developer

**Description:**
Implement the CLI with `greenlang schema validate/compile` commands.

**Deliverables:**
- `greenlang/schema/cli/main.py`
- `greenlang/schema/cli/commands/validate.py`
- `greenlang/schema/cli/commands/compile.py`
- `greenlang/schema/cli/commands/lint.py`

**Commands:**
```bash
# Validate
greenlang schema validate <file|-> \
  --schema <schema_ref|path> \
  [--profile standard|strict|permissive] \
  [--format pretty|text|table|json|sarif] \
  [--patch-level safe|needs_review|unsafe] \
  [--return-normalized] \
  [--fail-on-warnings] \
  [-v|-vv|--quiet]

# Compile
greenlang schema compile <schema_ref|path> \
  [--out ir.json] \
  [--format json]

# Lint
greenlang schema lint <schema_ref|path>

# Batch (glob)
greenlang schema validate --glob "data/*.yaml" --schema ...

# Aliases
greenlang validate ... → greenlang schema validate ...
greenlang compile-schema ... → greenlang schema compile ...
```

**API:**
```python
import click

@click.group()
def schema():
    """Schema compiler and validator commands."""
    pass

@schema.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--schema", "-s", required=True)
@click.option("--profile", type=click.Choice(["strict", "standard", "permissive"]))
@click.option("--format", "-f", type=click.Choice(["pretty", "text", "table", "json", "sarif"]))
@click.option("--patch-level", type=click.Choice(["safe", "needs_review", "unsafe"]))
@click.option("--return-normalized", is_flag=True)
@click.option("--fail-on-warnings", is_flag=True)
@click.option("-v", "--verbose", count=True)
@click.option("--quiet", is_flag=True)
def validate(file, schema, profile, format, patch_level, return_normalized, fail_on_warnings, verbose, quiet):
    """Validate a payload against a schema."""
```

**Acceptance Criteria:**
- [ ] All commands implemented
- [ ] Exit codes: 0 = valid, 1 = invalid, 2 = error
- [ ] `--fail-on-warnings` changes exit code behavior
- [ ] Default verbosity: summary + first 5 errors
- [ ] `-v` shows all, `-vv` shows deep, `--quiet` for CI
- [ ] Aliases work

---

### Task 5.2: Implement Output Formatters
**Priority:** P1 (High)
**Estimated Effort:** 8 hours
**Agent Type:** Backend Developer

**Description:**
Implement output formatters for JSON, YAML, Table, SARIF, and colorized pretty output.

**Deliverables:**
- `greenlang/schema/cli/formatters/json.py`
- `greenlang/schema/cli/formatters/table.py`
- `greenlang/schema/cli/formatters/sarif.py`
- `greenlang/schema/cli/formatters/pretty.py`

**Pretty Format Example:**
```
INVALID  emissions/activity@1.3.0  (3 errors, 1 warning)

ERROR GLSCHEMA-E100 at /energy_consumption
  Missing required field
  Expected: { value: number, unit: string }

ERROR GLSCHEMA-E301 at /fuel_type
  Unit 'kg' incompatible with dimension 'volume'
  Expected dimension: volume (L, gallon, m³)
  Actual: mass (kg)

ERROR GLSCHEMA-E200 at /temperature
  Value 150 exceeds maximum 120
  Allowed range: 0 to 120

WARNING GLSCHEMA-W700 at /emmisions
  Unknown field. Did you mean 'emissions'?

Run with -v to see all findings
Use --format json for machine output
```

**Acceptance Criteria:**
- [ ] All formatters implemented
- [ ] SARIF compatible with VS Code / GitHub
- [ ] Pretty output is colorized
- [ ] Table format is compact

---

### Task 5.3: Implement FastAPI Service
**Priority:** P1 (High)
**Estimated Effort:** 8 hours
**Agent Type:** API Developer

**Description:**
Implement the optional HTTP service for remote validation.

**Deliverables:**
- `greenlang/schema/api/routes.py`
- `greenlang/schema/api/models.py`
- `greenlang/schema/api/dependencies.py`

**Endpoints:**
```
POST /v1/schema/validate
POST /v1/schema/validate/batch
POST /v1/schema/compile
GET  /v1/schema/{schema_id}/versions
GET  /v1/schema/{schema_id}/{version}
GET  /health
GET  /metrics
```

**Request/Response:**
```python
class ValidateRequest(BaseModel):
    schema_ref: SchemaRef
    payload: Union[Dict, str]
    options: Optional[ValidationOptions] = None

class ValidateResponse(BaseModel):
    valid: bool
    schema_ref: SchemaRef
    schema_hash: str
    summary: ValidationSummary
    findings: List[Finding]
    normalized_payload: Optional[Dict] = None
    fix_suggestions: Optional[List[FixSuggestion]] = None
    timings_ms: Dict[str, float]

class BatchValidateRequest(BaseModel):
    schema_ref: SchemaRef
    payloads: List[Union[Dict, str]]
    options: Optional[ValidationOptions] = None

class BatchValidateResponse(BaseModel):
    schema_ref: SchemaRef
    schema_hash: str
    summary: BatchSummary
    results: List[ItemResult]
```

**Acceptance Criteria:**
- [ ] All endpoints implemented
- [ ] Request validation with Pydantic
- [ ] Proper error responses
- [ ] Health and metrics endpoints
- [ ] OpenAPI documentation

---

### Task 5.4: Implement IR Cache Service
**Priority:** P2 (Medium)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement HTTP cache service for compiled schema IR.

**Deliverables:**
- `greenlang/schema/registry/cache.py`

**Requirements:**
- LRU cache with configurable size
- TTL-based expiration
- Cache key: `(schema_id, version, compiler_version)`
- Metrics: hit rate, size, evictions
- Background warm-up for popular schemas

**API:**
```python
class IRCacheService:
    def __init__(
        self,
        max_size: int = SCHEMA_CACHE_MAX_SIZE,
        ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS
    ):
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[SchemaIR]:
        """Get IR from cache."""

    def put(self, key: str, ir: SchemaIR):
        """Store IR in cache."""

    def warm_up(self, schema_refs: List[SchemaRef]):
        """Pre-compile and cache popular schemas."""

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
```

**Acceptance Criteria:**
- [ ] LRU eviction policy
- [ ] TTL expiration
- [ ] Background warm-up
- [ ] Metrics exposed

---

## Phase 6: Registry & Orchestrator Integration
**Duration:** Week 7-8
**Dependencies:** Phase 5
**Parallelizable:** Yes

### Task 6.1: Implement Git-Backed Schema Registry
**Priority:** P1 (High)
**Estimated Effort:** 8 hours
**Agent Type:** Backend Developer

**Description:**
Implement Git-backed schema registry that reads schemas from a Git repository.

**Deliverables:**
- `greenlang/schema/registry/git_backend.py`

**Requirements:**
- Read schemas from Git repo (local or remote)
- Version resolution via Git tags
- Schema path convention: `schemas/{domain}/{name}@{version}.yaml`
- Support for latest version lookup
- Caching of fetched schemas

**API:**
```python
class GitSchemaRegistry:
    def __init__(
        self,
        repo_path: str,
        remote_url: Optional[str] = None
    ):
        self.repo_path = repo_path
        self.remote_url = remote_url

    def resolve(
        self,
        schema_id: str,
        version: str
    ) -> SchemaSource:
        """Resolve schema by ID and version."""

    def list_versions(self, schema_id: str) -> List[str]:
        """List available versions for a schema."""

    def get_latest(
        self,
        schema_id: str,
        constraint: Optional[str] = None
    ) -> str:
        """Get latest version matching constraint."""
```

**Acceptance Criteria:**
- [ ] Reads from local Git repo
- [ ] Resolves versions from tags
- [ ] Lists available versions
- [ ] Caches fetched schemas

---

### Task 6.2: Implement Orchestrator Validation Step
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Add `validate` step type to GL-FOUND-X-001 orchestrator.

**Deliverables:**
- `greenlang/orchestrator/steps/validate_step.py`
- Updates to `greenlang/orchestrator/step_registry.py`

**Step Configuration:**
```yaml
steps:
  - id: validate_input
    type: validate
    config:
      schema: gl://schemas/emissions/activity@1.3.0
      profile: standard
      fail_on_warnings: false
      normalize: true
    input:
      payload: ${{ inputs.data }}
    output:
      valid: is_valid
      normalized: normalized_data
      findings: validation_findings
```

**API:**
```python
class ValidateStep(BaseStep):
    step_type = "validate"

    async def execute(
        self,
        context: StepContext,
        inputs: Dict[str, Any]
    ) -> StepResult:
        """Execute validation step."""
```

**Acceptance Criteria:**
- [ ] Step type registered
- [ ] Configurable via YAML
- [ ] Returns valid/invalid status
- [ ] Can branch pipeline on result

---

### Task 6.3: Implement Pre-Run Validation Hook
**Priority:** P1 (High)
**Estimated Effort:** 4 hours
**Agent Type:** Backend Developer

**Description:**
Add automatic pre-run validation hook to orchestrator.

**Deliverables:**
- Updates to `greenlang/orchestrator/pipeline_executor.py`

**Requirements:**
- Validate pipeline inputs against declared schemas
- Configurable at pipeline level
- Fail-fast on validation errors
- Record validation in audit trail

**Configuration:**
```yaml
pipeline:
  validation:
    enabled: true
    profile: strict
    fail_on_warnings: false
  inputs:
    data:
      schema: gl://schemas/emissions/activity@1.3.0
```

**Acceptance Criteria:**
- [ ] Pre-run validation executes
- [ ] Fails pipeline on invalid input
- [ ] Configurable via pipeline YAML
- [ ] Audit trail includes validation

---

### Task 6.4: Implement Schema Migration CLI
**Priority:** P2 (Medium)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Implement migration CLI tool for schema adoption.

**Deliverables:**
- `greenlang/schema/cli/commands/migrate.py`

**Commands:**
```bash
# Analyze existing validation code
greenlang schema migrate analyze --path ./src

# Generate migration report
greenlang schema migrate report --path ./src --output migration.json

# Convert existing JSON Schema to GreenLang schema
greenlang schema migrate convert --input old_schema.json --output new_schema.yaml
```

**Acceptance Criteria:**
- [ ] Detects existing validation patterns
- [ ] Generates migration hints
- [ ] Converts JSON Schema to GreenLang format
- [ ] Estimates migration effort

---

## Phase 7: Testing & Documentation
**Duration:** Week 8+
**Dependencies:** All phases
**Parallelizable:** Yes

### Task 7.1: Create Golden Test Suite
**Priority:** P0 (Blocking)
**Estimated Effort:** 8 hours
**Agent Type:** Test Engineer

**Description:**
Create comprehensive golden test suite per PRD 12.1.

**Deliverables:**
```
tests/schema/golden/
├── schemas/
│   ├── basic/
│   │   ├── string_constraints.yaml
│   │   ├── numeric_constraints.yaml
│   │   ├── object_constraints.yaml
│   │   └── array_constraints.yaml
│   ├── units/
│   │   ├── energy_units.yaml
│   │   └── mass_units.yaml
│   ├── rules/
│   │   ├── conditional_required.yaml
│   │   └── consistency_check.yaml
│   └── edge_cases/
│       ├── deep_nesting.yaml
│       ├── circular_ref.yaml
│       └── large_enum.yaml
├── payloads/
│   ├── valid/
│   └── invalid/
└── expected/
    ├── reports/
    ├── normalized/
    └── patches/
```

**Acceptance Criteria:**
- [ ] 100+ golden tests covering all features
- [ ] Tests for each error code
- [ ] Edge case coverage
- [ ] CI runs golden tests on every commit

---

### Task 7.2: Implement Property-Based Tests
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Test Engineer

**Description:**
Implement property-based tests using Hypothesis.

**Deliverables:**
- `tests/schema/property/test_normalization_idempotent.py`
- `tests/schema/property/test_patch_monotonic.py`

**Properties:**
1. Normalization is idempotent: `normalize(normalize(x)) == normalize(x)`
2. Applying safe patches reduces errors: `len(errors(apply(patches, x))) <= len(errors(x))`
3. Valid payloads remain valid after normalization
4. Determinism: same input -> same output

**Acceptance Criteria:**
- [ ] All properties tested with Hypothesis
- [ ] 1000+ generated test cases per property
- [ ] No failures on CI

---

### Task 7.3: Implement Security Tests
**Priority:** P0 (Blocking)
**Estimated Effort:** 6 hours
**Agent Type:** Test Engineer

**Description:**
Implement security tests per PRD 12.2.

**Deliverables:**
- `tests/schema/security/test_yaml_bombs.py`
- `tests/schema/security/test_redos.py`
- `tests/schema/security/test_schema_bombs.py`

**Test Cases:**
- Billion laughs YAML
- Recursive YAML anchors
- ReDoS patterns
- Deep $ref chains
- Path traversal in schema resolver
- Large payloads
- Malformed UTF-8

**Acceptance Criteria:**
- [ ] All security tests pass
- [ ] Fuzzing with afl-fuzz or similar
- [ ] No crashes on malicious input

---

### Task 7.4: Write API Documentation
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Tech Writer

**Description:**
Write comprehensive API and user documentation.

**Deliverables:**
- `docs/schema/README.md`
- `docs/schema/cli.md`
- `docs/schema/api.md`
- `docs/schema/migration.md`
- `docs/schema/error_codes.md`

**Acceptance Criteria:**
- [ ] All public APIs documented
- [ ] CLI examples for each command
- [ ] Migration guide
- [ ] Error code reference

---

### Task 7.5: Implement Python SDK
**Priority:** P1 (High)
**Estimated Effort:** 6 hours
**Agent Type:** Backend Developer

**Description:**
Create clean Python SDK interface for library usage.

**Deliverables:**
- `greenlang/schema/sdk.py`

**API:**
```python
from greenlang.schema import validate, compile_schema, SchemaRef

# Simple validation
result = validate(
    payload={"energy": 100, "unit": "kWh"},
    schema=SchemaRef(schema_id="emissions/activity", version="1.3.0")
)

if result.valid:
    print("Valid!")
else:
    for finding in result.findings:
        print(f"{finding.code}: {finding.message}")

# With options
result = validate(
    payload=data,
    schema="gl://schemas/emissions/activity@1.3.0",
    profile="strict",
    normalize=True
)

# Get normalized payload
normalized = result.normalized_payload

# Get fix suggestions
for fix in result.fix_suggestions:
    print(f"[{fix.safety}] {fix.rationale}")
```

**Acceptance Criteria:**
- [ ] Clean, Pythonic API
- [ ] Sensible defaults
- [ ] Type hints for IDE support
- [ ] docstrings for all functions

---

## Task Dependency Graph

```
Phase 0 (Foundation)
├── 0.1 Project Structure ──┐
├── 0.2 Error Codes ────────┼──┐
├── 0.3 Data Models ────────┤  │
├── 0.4 Constants ──────────┤  │
└── 0.5 Test Infrastructure ┘  │
                               │
Phase 1 (Compiler) ◄───────────┘
├── 1.1 Parser ─────────┐
├── 1.2 AST ────────────┼──┐
├── 1.3 Resolver ───────┘  │
│                          │
├── 1.4 Compiler ◄─────────┤
├── 1.5 Regex Analyzer ◄───┤
└── 1.6 Schema Validator ◄─┘

Phase 2 (Validator) ◄── Phase 1
├── 2.1 Structural ─────┐
├── 2.2 Constraints ────┼──┐
├── 2.3 Units ──────────┤  │
├── 2.4 Rules ──────────┘  │
│                          │
├── 2.5 Core ◄─────────────┤
└── 2.6 Linter ◄───────────┘

Phase 3 (Normalizer) ◄── Phase 2
├── 3.1 Coercions ──────┐
├── 3.2 Unit Canon ─────┼──┐
├── 3.3 Key Canon ──────┘  │
│                          │
└── 3.4 Engine ◄───────────┘

Phase 4 (Suggestions) ◄── Phase 3
├── 4.1 Patches ────────┐
├── 4.2 Safety ─────────┼──┐
├── 4.3 Heuristics ─────┘  │
│                          │
└── 4.4 Engine ◄───────────┘

Phase 5 (CLI & API) ◄── Phase 4
├── 5.1 CLI ──────────────┐
├── 5.2 Formatters ◄──────┤
├── 5.3 API Service ──────┤
└── 5.4 Cache Service ────┘

Phase 6 (Integration) ◄── Phase 5
├── 6.1 Git Registry ─────┐
├── 6.2 Orchestrator Step ┤
├── 6.3 Pre-Run Hook ─────┤
└── 6.4 Migration CLI ────┘

Phase 7 (Testing & Docs) ◄── All
├── 7.1 Golden Tests
├── 7.2 Property Tests
├── 7.3 Security Tests
├── 7.4 Documentation
└── 7.5 Python SDK
```

---

## Agent Deployment Strategy

### Parallel Execution Groups

**Wave 1 (Week 1):** Phase 0 - All tasks parallel
- Agent 1: Task 0.1, 0.4
- Agent 2: Task 0.2, 0.3
- Agent 3: Task 0.5

**Wave 2 (Week 2-3):** Phase 1 - Partial parallel
- Agent 1: Task 1.1, 1.5
- Agent 2: Task 1.2, 1.6
- Agent 3: Task 1.3, 1.4

**Wave 3 (Week 3-4):** Phase 2 - All parallel
- Agent 1: Task 2.1, 2.5
- Agent 2: Task 2.2, 2.6
- Agent 3: Task 2.3
- Agent 4: Task 2.4

**Wave 4 (Week 4-5):** Phase 3 - Partial parallel
- Agent 1: Task 3.1, 3.4
- Agent 2: Task 3.2
- Agent 3: Task 3.3

**Wave 5 (Week 5-6):** Phase 4 - Partial parallel
- Agent 1: Task 4.1, 4.4
- Agent 2: Task 4.2
- Agent 3: Task 4.3

**Wave 6 (Week 6-7):** Phase 5 - All parallel
- Agent 1: Task 5.1
- Agent 2: Task 5.2
- Agent 3: Task 5.3
- Agent 4: Task 5.4

**Wave 7 (Week 7-8):** Phase 6 - All parallel
- Agent 1: Task 6.1
- Agent 2: Task 6.2
- Agent 3: Task 6.3
- Agent 4: Task 6.4

**Wave 8 (Week 8+):** Phase 7 - All parallel
- Agent 1: Task 7.1
- Agent 2: Task 7.2, 7.3
- Agent 3: Task 7.4
- Agent 4: Task 7.5

---

## Success Metrics

| Metric | Target |
|--------|--------|
| P95 latency (small payload) | < 25ms |
| P95 latency (medium payload) | < 150ms |
| Golden test coverage | 100% |
| Property test coverage | 95%+ |
| Error code coverage | 100% |
| API documentation | 100% |
| Security test pass rate | 100% |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| ReDoS vulnerabilities | High | Regex analyzer, RE2 fallback |
| Schema adoption resistance | High | Migration CLI, compatibility mode |
| Performance bottlenecks | Medium | Caching, Rust hot paths (later) |
| Rule DSL complexity | Medium | Reuse PolicyEngine, limit expression depth |
| Circular ref handling | Low | Early detection, clear traces |

---

## Appendix: File Inventory

**Total New Files:** ~65 files
**Total Lines (estimated):** ~15,000 LOC

**Core Library:** ~8,000 LOC
- compiler/: ~2,000 LOC
- validator/: ~2,500 LOC
- normalizer/: ~1,200 LOC
- suggestions/: ~1,000 LOC
- models/: ~800 LOC
- units/: ~500 LOC

**CLI & API:** ~2,500 LOC
- cli/: ~1,500 LOC
- api/: ~1,000 LOC

**Registry:** ~1,500 LOC
- registry/: ~1,500 LOC

**Tests:** ~3,000 LOC
- unit/: ~1,000 LOC
- integration/: ~500 LOC
- golden/: ~500 LOC
- property/: ~500 LOC
- security/: ~500 LOC
