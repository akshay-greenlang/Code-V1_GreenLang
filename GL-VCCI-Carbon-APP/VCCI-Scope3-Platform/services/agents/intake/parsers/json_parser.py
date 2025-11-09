"""
JSON Parser with Validation

JSON parsing with schema validation support.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import jsonschema

from greenlang.security.validators import PathTraversalValidator

from ..exceptions import FileParseError, SchemaValidationError
from ..config import get_config

logger = logging.getLogger(__name__)


class JSONParser:
    """
    JSON parser with schema validation.

    Features:
    - Parse JSON files and strings
    - JSON Schema validation
    - Support for JSON Lines format
    - Nested object flattening
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize JSON parser."""
        self.config = get_config().parser if config is None else config
        logger.info("Initialized JSONParser")

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse JSON file with path traversal protection.

        Args:
            file_path: Path to JSON file

        Returns:
            List of dictionaries (supports both array and single object)

        Raises:
            FileParseError: If parsing fails
        """
        try:
            # Validate path for security
            validated_path = PathTraversalValidator.validate_path(file_path, must_exist=True)

            logger.info(f"Parsing JSON file: {validated_path}")

            with open(validated_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert single object to list
            if isinstance(data, dict):
                records = [data]
            elif isinstance(data, list):
                records = data
            else:
                raise FileParseError(
                    f"Unexpected JSON structure: {type(data)}",
                    details={"file_path": str(file_path)}
                )

            logger.info(f"Successfully parsed {len(records)} records from JSON")
            return records

        except json.JSONDecodeError as e:
            raise FileParseError(
                f"Invalid JSON: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

        except Exception as e:
            raise FileParseError(
                f"Failed to parse JSON file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def parse_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse JSON Lines file (one JSON object per line).

        Args:
            file_path: Path to JSONL file

        Returns:
            List of dictionaries
        """
        try:
            logger.info(f"Parsing JSON Lines file: {file_path}")
            records = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        record['_line_number'] = line_num
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON at line {line_num}: {e}"
                        )
                        continue

            logger.info(f"Successfully parsed {len(records)} records from JSONL")
            return records

        except Exception as e:
            raise FileParseError(
                f"Failed to parse JSONL file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def parse_string(self, json_string: str) -> List[Dict[str, Any]]:
        """
        Parse JSON from string.

        Args:
            json_string: JSON string

        Returns:
            List of dictionaries
        """
        try:
            data = json.loads(json_string)

            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            else:
                raise FileParseError(f"Unexpected JSON structure: {type(data)}")

        except json.JSONDecodeError as e:
            raise FileParseError(
                f"Invalid JSON string: {str(e)}",
                details={"error": str(e)}
            ) from e

    def validate_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate data against JSON Schema.

        Args:
            data: Data to validate
            schema: JSON Schema

        Returns:
            Validation result dictionary
        """
        try:
            jsonschema.validate(instance=data, schema=schema)
            return {
                "valid": True,
                "errors": []
            }

        except jsonschema.ValidationError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "error_path": list(e.path),
            }

        except jsonschema.SchemaError as e:
            raise SchemaValidationError(
                f"Invalid JSON Schema: {str(e)}",
                details={"error": str(e)}
            ) from e

    def validate_file(
        self,
        file_path: Path,
        schema_path: Path,
    ) -> Dict[str, Any]:
        """
        Validate JSON file against schema file.

        Args:
            file_path: Path to JSON data file
            schema_path: Path to JSON Schema file

        Returns:
            Validation result with statistics
        """
        try:
            # Load schema
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)

            # Parse data
            records = self.parse(file_path)

            # Validate each record
            results = []
            for i, record in enumerate(records):
                result = self.validate_schema(record, schema)
                result['record_index'] = i
                results.append(result)

            # Summary statistics
            valid_count = sum(1 for r in results if r['valid'])
            invalid_count = len(results) - valid_count

            return {
                "total_records": len(results),
                "valid_records": valid_count,
                "invalid_records": invalid_count,
                "validation_results": results,
            }

        except Exception as e:
            raise SchemaValidationError(
                f"Schema validation failed: {str(e)}",
                details={
                    "file_path": str(file_path),
                    "schema_path": str(schema_path),
                    "error": str(e)
                }
            ) from e

    def flatten_nested(
        self,
        data: Dict[str, Any],
        parent_key: str = '',
        sep: str = '_'
    ) -> Dict[str, Any]:
        """
        Flatten nested JSON structure.

        Args:
            data: Nested dictionary
            parent_key: Parent key prefix
            sep: Separator for flattened keys

        Returns:
            Flattened dictionary
        """
        items = []

        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(
                    self.flatten_nested(v, new_key, sep=sep).items()
                )
            elif isinstance(v, list):
                # Convert list to comma-separated string
                items.append((new_key, ','.join(str(x) for x in v)))
            else:
                items.append((new_key, v))

        return dict(items)


__all__ = ["JSONParser"]
