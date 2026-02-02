# -*- coding: utf-8 -*-
# GL-VCCI Utilities Module
# Shared utility functions and helpers

"""
VCCI Utilities
==============

Shared utility functions, validators, and helpers used across the platform.

Modules:
--------
- validation: Data validation and quality checks
- conversion: Unit conversion and normalization
- formatting: Output formatting and display
- encryption: Encryption and hashing utilities
- datetime_utils: Date/time manipulation
- file_utils: File handling and I/O
- api_utils: API helpers and wrappers

Common Utilities:
----------------
```python
from utils import (
    validate_emission_factor,
    convert_units,
    format_emissions,
    hash_sha256,
    parse_date_range,
    read_csv_safely
)

# Validate emission factor
is_valid = validate_emission_factor(
    value=0.193,
    unit="kgCO2e/kWh",
    uncertainty=0.10
)

# Convert units
result = convert_units(
    value=1000,
    from_unit="kg",
    to_unit="tonnes"
)  # Returns 1.0

# Format emissions for display
formatted = format_emissions(1234.5678)  # "1,234.57 tCO2e"

# SHA-256 hash
hash_value = hash_sha256(data)

# Parse date range
start, end = parse_date_range("2024-01-01", "2024-12-31")
```
"""

__version__ = "1.0.0"

__all__ = [
    # Validation
    # "validate_emission_factor",
    # "validate_data_quality",

    # Conversion
    # "convert_units",
    # "normalize_supplier_name",

    # Formatting
    # "format_emissions",
    # "format_percentage",

    # Encryption
    # "hash_sha256",
    # "encrypt_data",

    # Datetime
    # "parse_date_range",
    # "get_reporting_period",

    # File handling
    # "read_csv_safely",
    # "write_json_safely",
]
