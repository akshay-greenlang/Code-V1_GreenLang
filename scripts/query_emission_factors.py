#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emission Factors Registry Query Utility

This script provides helper functions to query the emission factors registry
with URI validation and data provenance tracking.

Usage:
    from scripts.query_emission_factors import EmissionFactorRegistry

    registry = EmissionFactorRegistry()
    factor = registry.get_fuel_factor("natural_gas", unit="kwh")
    grid_factor = registry.get_grid_factor("US_WECC_CA")

Author: GreenLang Team
Version: 1.0.0
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import requests
from urllib.parse import urlparse


class EmissionFactorRegistry:
    """
    Query and validate emission factors from the central registry.

    Provides easy access to emission factors with automatic URI validation
    and data provenance tracking.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the emission factors registry.

        Args:
            registry_path: Path to the registry file (YAML or JSON).
                          If None, uses default location.
        """
        if registry_path is None:
            # Default path relative to script location
            base_dir = Path(__file__).parent.parent
            registry_path = base_dir / "data" / "emission_factors_registry.yaml"

        self.registry_path = Path(registry_path)
        self.data = self._load_registry()
        self.cache = {}

    def _load_registry(self) -> Dict[str, Any]:
        """Load the emission factors registry from file."""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Registry file not found: {self.registry_path}"
            )

        if self.registry_path.suffix == ".yaml" or self.registry_path.suffix == ".yml":
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f)
        elif self.registry_path.suffix == ".json":
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {self.registry_path.suffix}")

    def get_fuel_factor(
        self,
        fuel_type: str,
        unit: str = "kwh",
        validate_uri: bool = False
    ) -> Dict[str, Any]:
        """
        Get emission factor for a specific fuel type.

        Args:
            fuel_type: Fuel type (e.g., "natural_gas", "diesel", "coal_bituminous")
            unit: Unit for emission factor (e.g., "kwh", "liter", "gallon", "kg")
            validate_uri: Whether to validate the source URI

        Returns:
            Dictionary containing emission factor and metadata

        Example:
            >>> registry = EmissionFactorRegistry()
            >>> factor = registry.get_fuel_factor("natural_gas", unit="kwh")
            >>> print(factor["emission_factor"])
            0.202
            >>> print(factor["uri"])
            https://www.epa.gov/climateleadership/ghg-emission-factors-hub
        """
        fuels = self.data.get("fuels", {})

        if fuel_type not in fuels:
            raise ValueError(
                f"Fuel type '{fuel_type}' not found. "
                f"Available: {', '.join(fuels.keys())}"
            )

        fuel_data = fuels[fuel_type].copy()

        # Find the emission factor for the requested unit
        factor_key = f"emission_factor_kg_co2e_per_{unit.lower()}"

        if factor_key not in fuel_data:
            available_units = [
                key.replace("emission_factor_kg_co2e_per_", "")
                for key in fuel_data.keys()
                if key.startswith("emission_factor_kg_co2e_per_")
            ]
            raise ValueError(
                f"Unit '{unit}' not available for {fuel_type}. "
                f"Available: {', '.join(available_units)}"
            )

        result = {
            "fuel_type": fuel_type,
            "emission_factor": fuel_data[factor_key],
            "unit": f"kg CO2e per {unit}",
            "name": fuel_data.get("name", fuel_type),
            "scope": fuel_data.get("scope", "Unknown"),
            "source": fuel_data.get("source", "Unknown"),
            "uri": fuel_data.get("uri", ""),
            "standard": fuel_data.get("standard", "Unknown"),
            "last_updated": fuel_data.get("last_updated", "Unknown"),
            "data_quality": fuel_data.get("data_quality", "Unknown"),
            "uncertainty": fuel_data.get("uncertainty", "Unknown"),
            "metadata": {k: v for k, v in fuel_data.items()
                        if k not in ["name", "scope", "source", "uri",
                                   "standard", "last_updated"] and
                        not k.startswith("emission_factor")}
        }

        if validate_uri and result["uri"]:
            result["uri_valid"] = self._validate_uri(result["uri"])

        return result

    def get_grid_factor(
        self,
        grid_region: str,
        validate_uri: bool = False
    ) -> Dict[str, Any]:
        """
        Get emission factor for a specific electricity grid region.

        Args:
            grid_region: Grid region code (e.g., "US_WECC_CA", "UK", "IN")
            validate_uri: Whether to validate the source URI

        Returns:
            Dictionary containing emission factor and metadata

        Example:
            >>> registry = EmissionFactorRegistry()
            >>> factor = registry.get_grid_factor("US_WECC_CA")
            >>> print(factor["emission_factor_kwh"])
            0.234
        """
        grids = self.data.get("grids", {})

        if grid_region not in grids:
            raise ValueError(
                f"Grid region '{grid_region}' not found. "
                f"Available: {', '.join(grids.keys())}"
            )

        grid_data = grids[grid_region].copy()

        result = {
            "grid_region": grid_region,
            "emission_factor_kwh": grid_data.get("emission_factor_kg_co2e_per_kwh"),
            "emission_factor_mwh": grid_data.get("emission_factor_kg_co2e_per_mwh"),
            "name": grid_data.get("name", grid_region),
            "scope": grid_data.get("scope", "Scope 2 - Location-Based"),
            "source": grid_data.get("source", "Unknown"),
            "uri": grid_data.get("uri", ""),
            "standard": grid_data.get("standard", "Unknown"),
            "last_updated": grid_data.get("last_updated", "Unknown"),
            "year": grid_data.get("year", "Unknown"),
            "renewable_share": grid_data.get("renewable_share", 0),
            "region": grid_data.get("region", "Unknown"),
            "metadata": {k: v for k, v in grid_data.items()
                        if k not in ["name", "scope", "source", "uri",
                                   "standard", "last_updated", "year",
                                   "renewable_share", "region"] and
                        not k.startswith("emission_factor")}
        }

        if validate_uri and result["uri"]:
            result["uri_valid"] = self._validate_uri(result["uri"])

        return result

    def get_process_factor(
        self,
        process_name: str,
        validate_uri: bool = False
    ) -> Dict[str, Any]:
        """
        Get emission factor for an industrial process.

        Args:
            process_name: Process name (e.g., "pasteurization", "cement_production")
            validate_uri: Whether to validate the source URI

        Returns:
            Dictionary containing emission factor and metadata
        """
        processes = self.data.get("processes", {})

        if process_name not in processes:
            raise ValueError(
                f"Process '{process_name}' not found. "
                f"Available: {', '.join(processes.keys())}"
            )

        process_data = processes[process_name].copy()

        result = {
            "process_name": process_name,
            "name": process_data.get("name", process_name),
            "scope": process_data.get("scope", "Unknown"),
            "source": process_data.get("source", "Unknown"),
            "uri": process_data.get("uri", ""),
            "standard": process_data.get("standard", "Unknown"),
            "last_updated": process_data.get("last_updated", "Unknown"),
            "process_type": process_data.get("process_type", "Unknown"),
            "data": {k: v for k, v in process_data.items()
                    if k not in ["name", "scope", "source", "uri",
                               "standard", "last_updated", "process_type"]}
        }

        if validate_uri and result["uri"]:
            result["uri_valid"] = self._validate_uri(result["uri"])

        return result

    def search(
        self,
        query: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for emission factors by keyword.

        Args:
            query: Search query (case-insensitive)
            category: Optional category filter ("fuels", "grids", "processes")

        Returns:
            List of matching emission factors

        Example:
            >>> registry = EmissionFactorRegistry()
            >>> results = registry.search("coal")
            >>> for result in results:
            ...     print(result["name"])
        """
        results = []
        query_lower = query.lower()

        categories = [category] if category else ["fuels", "grids", "processes"]

        for cat in categories:
            if cat not in self.data:
                continue

            for key, value in self.data[cat].items():
                # Search in key and name
                if (query_lower in key.lower() or
                    query_lower in value.get("name", "").lower() or
                    query_lower in value.get("description", "").lower()):

                    results.append({
                        "category": cat,
                        "id": key,
                        "name": value.get("name", key),
                        "source": value.get("source", "Unknown"),
                        "uri": value.get("uri", "")
                    })

        return results

    def list_categories(self) -> Dict[str, List[str]]:
        """
        List all available categories and their items.

        Returns:
            Dictionary mapping categories to lists of available items
        """
        return {
            "fuels": list(self.data.get("fuels", {}).keys()),
            "grids": list(self.data.get("grids", {}).keys()),
            "processes": list(self.data.get("processes", {}).keys()),
            "district_energy": list(self.data.get("district_energy", {}).keys()),
            "renewable_generation": list(self.data.get("renewable_generation", {}).keys()),
            "business_travel": list(self.data.get("business_travel", {}).keys())
        }

    def validate_all_uris(self) -> Dict[str, Any]:
        """
        Validate all URIs in the registry.

        Returns:
            Dictionary with validation results
        """
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "details": []
        }

        for category in ["fuels", "grids", "processes", "district_energy",
                        "renewable_generation", "business_travel", "water"]:
            if category not in self.data:
                continue

            for key, value in self.data[category].items():
                uri = value.get("uri", "")
                if uri:
                    results["total"] += 1
                    is_valid = self._validate_uri(uri)

                    if is_valid:
                        results["valid"] += 1
                    else:
                        results["invalid"] += 1

                    results["details"].append({
                        "category": category,
                        "id": key,
                        "name": value.get("name", key),
                        "uri": uri,
                        "valid": is_valid
                    })

        return results

    def _validate_uri(self, uri: str, timeout: int = 5) -> bool:
        """
        Validate that a URI is accessible.

        Args:
            uri: URI to validate
            timeout: Request timeout in seconds

        Returns:
            True if URI is accessible, False otherwise
        """
        # Check if already cached
        if uri in self.cache:
            return self.cache[uri]

        try:
            # Parse URI
            parsed = urlparse(uri)
            if not parsed.scheme or not parsed.netloc:
                self.cache[uri] = False
                return False

            # Make HEAD request to check accessibility
            response = requests.head(uri, timeout=timeout, allow_redirects=True)
            is_valid = response.status_code < 400

            self.cache[uri] = is_valid
            return is_valid

        except Exception as e:
            print(f"Warning: URI validation failed for {uri}: {e}")
            self.cache[uri] = False
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Get registry metadata including version and standards."""
        return self.data.get("metadata", {})

    def export_audit_report(self, output_path: str) -> None:
        """
        Export an audit report with all emission factors and their sources.

        Args:
            output_path: Path to save the audit report (CSV or JSON)
        """
        import csv

        output_path = Path(output_path)

        if output_path.suffix == ".csv":
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Category", "ID", "Name", "Emission Factor",
                    "Unit", "Scope", "Source", "URI", "Standard",
                    "Last Updated", "Data Quality"
                ])

                for category in ["fuels", "grids", "processes"]:
                    if category not in self.data:
                        continue

                    for key, value in self.data[category].items():
                        # Find primary emission factor
                        ef_keys = [k for k in value.keys()
                                  if k.startswith("emission_factor")]
                        ef = value.get(ef_keys[0]) if ef_keys else "N/A"
                        unit = ef_keys[0].replace("emission_factor_", "") if ef_keys else "N/A"

                        writer.writerow([
                            category,
                            key,
                            value.get("name", key),
                            ef,
                            unit,
                            value.get("scope", "Unknown"),
                            value.get("source", "Unknown"),
                            value.get("uri", ""),
                            value.get("standard", "Unknown"),
                            value.get("last_updated", "Unknown"),
                            value.get("data_quality", "Unknown")
                        ])

        elif output_path.suffix == ".json":
            with open(output_path, 'w') as f:
                json.dump(self.data, f, indent=2)

        print(f"Audit report exported to: {output_path}")


def main():
    """Command-line interface for the emission factors registry."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query emission factors registry"
    )
    parser.add_argument(
        "command",
        choices=["get-fuel", "get-grid", "get-process", "search",
                "list", "validate-uris", "export-audit"],
        help="Command to execute"
    )
    parser.add_argument("--fuel", help="Fuel type")
    parser.add_argument("--unit", default="kwh", help="Unit for fuel factor")
    parser.add_argument("--grid", help="Grid region")
    parser.add_argument("--process", help="Process name")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--category", help="Category filter for search")
    parser.add_argument("--validate-uri", action="store_true",
                       help="Validate source URI")
    parser.add_argument("--output", help="Output path for export")
    parser.add_argument("--registry", help="Path to registry file")

    args = parser.parse_args()

    # Initialize registry
    registry = EmissionFactorRegistry(args.registry)

    # Execute command
    if args.command == "get-fuel":
        if not args.fuel:
            print("Error: --fuel is required")
            return

        result = registry.get_fuel_factor(
            args.fuel,
            unit=args.unit,
            validate_uri=args.validate_uri
        )
        print(json.dumps(result, indent=2))

    elif args.command == "get-grid":
        if not args.grid:
            print("Error: --grid is required")
            return

        result = registry.get_grid_factor(
            args.grid,
            validate_uri=args.validate_uri
        )
        print(json.dumps(result, indent=2))

    elif args.command == "get-process":
        if not args.process:
            print("Error: --process is required")
            return

        result = registry.get_process_factor(
            args.process,
            validate_uri=args.validate_uri
        )
        print(json.dumps(result, indent=2))

    elif args.command == "search":
        if not args.query:
            print("Error: --query is required")
            return

        results = registry.search(args.query, args.category)
        print(json.dumps(results, indent=2))

    elif args.command == "list":
        categories = registry.list_categories()
        print(json.dumps(categories, indent=2))

    elif args.command == "validate-uris":
        print("Validating all URIs (this may take a while)...")
        results = registry.validate_all_uris()
        print(json.dumps(results, indent=2))

    elif args.command == "export-audit":
        if not args.output:
            print("Error: --output is required")
            return

        registry.export_audit_report(args.output)


if __name__ == "__main__":
    main()
