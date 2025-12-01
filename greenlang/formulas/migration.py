"""
Formula Migration Utilities

This module provides utilities to migrate formulas from external sources
(YAML files, Python modules) into the versioned formula database.

Supports:
- CSRD esrs_formulas.yaml
- CBAM emission_factors.py
- Custom formula definitions
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import logging
from datetime import datetime

from greenlang.formulas.manager import FormulaManager
from greenlang.formulas.models import FormulaCategory
from greenlang.exceptions import ValidationError, ProcessingError

logger = logging.getLogger(__name__)


class FormulaMigrator:
    """
    Migrate formulas from external sources to versioned database.

    Example:
        >>> migrator = FormulaMigrator(manager)
        >>> migrator.migrate_from_yaml("esrs_formulas.yaml")
        >>> migrator.migrate_from_python("emission_factors.py")
    """

    def __init__(self, manager: FormulaManager):
        """
        Initialize migrator.

        Args:
            manager: FormulaManager instance
        """
        self.manager = manager
        self.migration_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
        }

    def migrate_from_yaml(
        self,
        yaml_path: str,
        created_by: str = "migration",
        auto_activate: bool = True,
    ) -> Dict[str, Any]:
        """
        Migrate formulas from YAML file (CSRD esrs_formulas.yaml format).

        Args:
            yaml_path: Path to YAML file
            created_by: User performing migration
            auto_activate: Automatically activate imported versions

        Returns:
            Migration statistics
        """
        logger.info(f"Starting YAML migration from {yaml_path}")

        try:
            # Load YAML file
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # Reset statistics
            self.migration_stats = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'skipped': 0,
            }

            # Process each formula category
            for category_key, category_data in data.items():
                # Skip metadata section
                if category_key in ['metadata', 'calculation_notes', 'utility_formulas']:
                    continue

                # Determine category
                category = self._map_yaml_category(category_key)

                # Process formulas in this category
                if isinstance(category_data, dict):
                    for formula_key, formula_data in category_data.items():
                        self.migration_stats['total'] += 1

                        try:
                            self._import_yaml_formula(
                                formula_key,
                                formula_data,
                                category,
                                created_by,
                                auto_activate,
                            )
                            self.migration_stats['success'] += 1

                        except Exception as e:
                            logger.error(
                                f"Failed to import {formula_key}: {e}"
                            )
                            self.migration_stats['failed'] += 1

            logger.info(
                f"YAML migration complete: {self.migration_stats['success']} "
                f"success, {self.migration_stats['failed']} failed"
            )

            return self.migration_stats

        except Exception as e:
            logger.error(f"YAML migration failed: {e}")
            raise ProcessingError(f"YAML migration failed: {e}") from e

    def _import_yaml_formula(
        self,
        formula_key: str,
        formula_data: Dict[str, Any],
        category: FormulaCategory,
        created_by: str,
        auto_activate: bool,
    ):
        """Import a single formula from YAML."""
        # Extract formula metadata
        metric_code = formula_data.get('metric_code', formula_key)
        metric_name = formula_data.get('metric_name', formula_key)
        formula_expression = formula_data.get('formula', '')
        calculation_type = formula_data.get('calculation_type', 'sum')
        unit = formula_data.get('unit', '')
        inputs = formula_data.get('inputs', [])

        # Check if formula already exists
        existing = self.manager.get_formula(metric_code)

        if existing:
            logger.info(f"Formula {metric_code} already exists, skipping")
            self.migration_stats['skipped'] += 1
            return

        # Create formula
        formula_id = self.manager.create_formula(
            formula_code=metric_code,
            formula_name=metric_name,
            category=category,
            description=formula_data.get('note', ''),
            standard_reference=formula_data.get('standard_reference', ''),
            created_by=created_by,
        )

        # Create initial version
        version_data = {
            'formula_expression': formula_expression,
            'calculation_type': calculation_type,
            'required_inputs': inputs,
            'output_unit': unit,
            'deterministic': formula_data.get('deterministic', True),
            'zero_hallucination': formula_data.get('zero_hallucination', True),
            'example_calculation': formula_data.get('example', ''),
        }

        version_id = self.manager.create_new_version(
            formula_code=metric_code,
            formula_data=version_data,
            change_notes="Initial import from YAML",
            created_by=created_by,
            auto_activate=auto_activate,
        )

        logger.info(
            f"Imported {metric_code} (formula_id={formula_id}, "
            f"version_id={version_id})"
        )

    def _map_yaml_category(self, yaml_category: str) -> FormulaCategory:
        """Map YAML category to FormulaCategory enum."""
        category_map = {
            'E1_formulas': FormulaCategory.EMISSIONS,
            'E2_formulas': FormulaCategory.EMISSIONS,
            'E3_formulas': FormulaCategory.WATER,
            'E4_formulas': FormulaCategory.EMISSIONS,
            'E5_formulas': FormulaCategory.WASTE,
            'S1_formulas': FormulaCategory.WORKFORCE,
            'S2_formulas': FormulaCategory.WORKFORCE,
            'G1_formulas': FormulaCategory.COMPLIANCE,
        }

        return category_map.get(yaml_category, FormulaCategory.UTILITY)

    def migrate_from_python(
        self,
        python_path: str,
        created_by: str = "migration",
        auto_activate: bool = True,
    ) -> Dict[str, Any]:
        """
        Migrate emission factors from Python module.

        This imports emission factors from CBAM emission_factors.py format.

        Args:
            python_path: Path to Python file
            created_by: User performing migration
            auto_activate: Automatically activate imported versions

        Returns:
            Migration statistics
        """
        logger.info(f"Starting Python migration from {python_path}")

        try:
            # Import Python module dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("emission_factors", python_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Reset statistics
            self.migration_stats = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'skipped': 0,
            }

            # Get emission factors database
            if hasattr(module, 'EMISSION_FACTORS_DB'):
                emission_factors_db = module.EMISSION_FACTORS_DB

                for product_key, product_data in emission_factors_db.items():
                    self.migration_stats['total'] += 1

                    try:
                        self._import_emission_factor(
                            product_key,
                            product_data,
                            created_by,
                            auto_activate,
                        )
                        self.migration_stats['success'] += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to import {product_key}: {e}"
                        )
                        self.migration_stats['failed'] += 1

            logger.info(
                f"Python migration complete: {self.migration_stats['success']} "
                f"success, {self.migration_stats['failed']} failed"
            )

            return self.migration_stats

        except Exception as e:
            logger.error(f"Python migration failed: {e}")
            raise ProcessingError(f"Python migration failed: {e}") from e

    def _import_emission_factor(
        self,
        product_key: str,
        product_data: Dict[str, Any],
        created_by: str,
        auto_activate: bool,
    ):
        """Import a single emission factor as a formula."""
        # Create formula code from product key
        formula_code = f"CBAM_{product_key.upper()}"

        # Check if formula already exists
        existing = self.manager.get_formula(formula_code)

        if existing:
            logger.info(f"Formula {formula_code} already exists, skipping")
            self.migration_stats['skipped'] += 1
            return

        # Create formula
        formula_id = self.manager.create_formula(
            formula_code=formula_code,
            formula_name=product_data['product_name'],
            category=FormulaCategory.EMISSIONS,
            description=product_data.get('notes', ''),
            standard_reference=product_data.get('source', ''),
            created_by=created_by,
        )

        # Create version for direct emissions
        version_data = {
            'formula_expression': f"activity_data * {product_data['default_direct_tco2_per_ton']}",
            'calculation_type': 'database_lookup_and_multiply',
            'required_inputs': ['activity_data'],
            'output_unit': 'tCO2e',
            'deterministic': True,
            'zero_hallucination': True,
        }

        version_id = self.manager.create_new_version(
            formula_code=formula_code,
            formula_data=version_data,
            change_notes="Initial import from Python emission factors",
            created_by=created_by,
            auto_activate=auto_activate,
        )

        logger.info(
            f"Imported {formula_code} (formula_id={formula_id}, "
            f"version_id={version_id})"
        )

    def migrate_custom_formulas(
        self,
        formulas: List[Dict[str, Any]],
        created_by: str = "migration",
        auto_activate: bool = True,
    ) -> Dict[str, Any]:
        """
        Migrate custom formula definitions.

        Args:
            formulas: List of formula definitions
            created_by: User performing migration
            auto_activate: Automatically activate imported versions

        Returns:
            Migration statistics

        Example:
            >>> formulas = [
            >>>     {
            >>>         'formula_code': 'CUSTOM_001',
            >>>         'formula_name': 'Custom Calculation',
            >>>         'category': 'emissions',
            >>>         'formula_expression': 'value1 + value2',
            >>>         'calculation_type': 'sum',
            >>>         'required_inputs': ['value1', 'value2'],
            >>>         'output_unit': 'tCO2e',
            >>>     }
            >>> ]
            >>> migrator.migrate_custom_formulas(formulas)
        """
        logger.info(f"Starting custom formula migration ({len(formulas)} formulas)")

        # Reset statistics
        self.migration_stats = {
            'total': len(formulas),
            'success': 0,
            'failed': 0,
            'skipped': 0,
        }

        for formula_def in formulas:
            try:
                # Check if formula exists
                existing = self.manager.get_formula(formula_def['formula_code'])

                if existing:
                    logger.info(
                        f"Formula {formula_def['formula_code']} already exists, skipping"
                    )
                    self.migration_stats['skipped'] += 1
                    continue

                # Create formula
                formula_id = self.manager.create_formula(
                    formula_code=formula_def['formula_code'],
                    formula_name=formula_def['formula_name'],
                    category=FormulaCategory(formula_def['category']),
                    description=formula_def.get('description', ''),
                    standard_reference=formula_def.get('standard_reference', ''),
                    created_by=created_by,
                )

                # Create version
                version_data = {
                    'formula_expression': formula_def['formula_expression'],
                    'calculation_type': formula_def['calculation_type'],
                    'required_inputs': formula_def.get('required_inputs', []),
                    'optional_inputs': formula_def.get('optional_inputs', []),
                    'output_unit': formula_def.get('output_unit', ''),
                    'deterministic': formula_def.get('deterministic', True),
                    'zero_hallucination': formula_def.get('zero_hallucination', True),
                }

                version_id = self.manager.create_new_version(
                    formula_code=formula_def['formula_code'],
                    formula_data=version_data,
                    change_notes="Initial import from custom definition",
                    created_by=created_by,
                    auto_activate=auto_activate,
                )

                self.migration_stats['success'] += 1

                logger.info(
                    f"Imported {formula_def['formula_code']} "
                    f"(formula_id={formula_id}, version_id={version_id})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to import {formula_def.get('formula_code', 'UNKNOWN')}: {e}"
                )
                self.migration_stats['failed'] += 1

        logger.info(
            f"Custom migration complete: {self.migration_stats['success']} "
            f"success, {self.migration_stats['failed']} failed"
        )

        return self.migration_stats

    def get_migration_summary(self) -> Dict[str, Any]:
        """Get summary of last migration."""
        return {
            **self.migration_stats,
            'success_rate': (
                self.migration_stats['success'] / self.migration_stats['total'] * 100
                if self.migration_stats['total'] > 0
                else 0
            ),
        }
