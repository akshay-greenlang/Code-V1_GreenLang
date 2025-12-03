#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPA eGRID 2023 Data Loader Script

Loads and validates EPA eGRID 2023 emission factor data for US electricity grid.
Provides functionality for data validation, statistics generation, and export.

Usage:
    python load_egrid_2023.py --validate
    python load_egrid_2023.py --stats
    python load_egrid_2023.py --export-csv
    python load_egrid_2023.py --lookup CA

Author: GreenLang Data Integration Engineer
Date: 2024
"""

import argparse
import json
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EGridValidationResult:
    """Validation result for eGRID data."""
    is_valid: bool
    total_subregions: int
    total_states: int
    total_plants_sampled: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    subregion_summary: List[Dict[str, Any]] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EGridStatistics:
    """Statistics for eGRID data."""
    us_average_co2e_kg_mwh: float
    lowest_intensity_subregion: str
    lowest_intensity_value: float
    highest_intensity_subregion: str
    highest_intensity_value: float
    total_generation_mwh: int
    total_plants: int
    generation_mix: Dict[str, float] = field(default_factory=dict)
    subregion_count: int = 0
    state_coverage: int = 0


class EGridLoader:
    """
    EPA eGRID 2023 Data Loader.

    Features:
    - Load and parse eGRID JSON data
    - Validate data completeness and accuracy
    - Generate statistics and reports
    - Export to CSV format
    - Location-based grid intensity lookups
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize eGRID loader.

        Args:
            data_dir: Directory containing eGRID JSON file
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "core" / "greenlang" / "data" / "factors"
        self.egrid_path = self.data_dir / "epa_egrid_2023.json"
        self.data: Optional[Dict[str, Any]] = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load eGRID data from JSON file.

        Returns:
            True if loaded successfully
        """
        if not self.egrid_path.exists():
            logger.error(f"eGRID file not found: {self.egrid_path}")
            return False

        try:
            with open(self.egrid_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self._loaded = True
            logger.info(f"Loaded eGRID data from {self.egrid_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in eGRID file: {e}")
            return False

    def validate(self) -> EGridValidationResult:
        """
        Validate eGRID data completeness and accuracy.

        Returns:
            EGridValidationResult with validation details
        """
        if not self._loaded:
            if not self.load():
                return EGridValidationResult(
                    is_valid=False,
                    total_subregions=0,
                    total_states=0,
                    total_plants_sampled=0,
                    errors=["Failed to load eGRID data"]
                )

        errors = []
        warnings = []
        subregion_summary = []

        # Validate metadata
        metadata = self.data.get('_metadata', {})
        if not metadata:
            warnings.append("Missing metadata section")

        # Validate subregions
        subregions = self.data.get('subregions', {})
        if not subregions:
            errors.append("No subregions found in data")
            return EGridValidationResult(
                is_valid=False,
                total_subregions=0,
                total_states=0,
                total_plants_sampled=0,
                errors=errors
            )

        total_plants = 0
        all_states = set()

        for code, subregion in subregions.items():
            year_data = subregion.get('2023', {})

            # Required fields
            required_fields = ['co2e_kg_per_mwh', 'generation_mix', 'plant_count']
            missing = [f for f in required_fields if f not in year_data]

            if missing:
                warnings.append(f"Subregion {code} missing fields: {missing}")

            # Validate emission intensity range
            intensity = year_data.get('co2e_kg_per_mwh', 0)
            if intensity < 0:
                errors.append(f"Subregion {code} has negative intensity: {intensity}")
            elif intensity > 1500:
                warnings.append(f"Subregion {code} has unusually high intensity: {intensity}")

            # Validate generation mix
            gen_mix = year_data.get('generation_mix', {})
            mix_total = sum(gen_mix.values())
            if abs(mix_total - 100) > 1:
                warnings.append(f"Subregion {code} generation mix sums to {mix_total}%, not 100%")

            # Track statistics
            total_plants += year_data.get('plant_count', 0)
            all_states.update(subregion.get('states', []))

            subregion_summary.append({
                'code': code,
                'name': subregion.get('name', code),
                'states': subregion.get('states', []),
                'co2e_kg_per_mwh': intensity,
                'plant_count': year_data.get('plant_count', 0),
                'total_generation_mwh': year_data.get('total_generation_mwh', 0)
            })

        # Validate US national average
        us_avg = self.data.get('us_national_average', {}).get('2023', {})
        if not us_avg:
            warnings.append("Missing US national average data")

        # Validate state mappings
        state_averages = self.data.get('state_averages', {})
        expected_states = 50  # US states
        if len(state_averages) < expected_states:
            warnings.append(f"State averages only covers {len(state_averages)} states (expected ~50)")

        return EGridValidationResult(
            is_valid=len(errors) == 0,
            total_subregions=len(subregions),
            total_states=len(all_states),
            total_plants_sampled=total_plants,
            errors=errors,
            warnings=warnings,
            subregion_summary=subregion_summary
        )

    def get_statistics(self) -> EGridStatistics:
        """
        Generate statistics from eGRID data.

        Returns:
            EGridStatistics with summary data
        """
        if not self._loaded:
            self.load()

        subregions = self.data.get('subregions', {})
        us_avg = self.data.get('us_national_average', {}).get('2023', {})

        # Find lowest and highest intensity
        lowest_code, lowest_val = None, float('inf')
        highest_code, highest_val = None, 0
        total_gen = 0
        total_plants = 0
        all_states = set()

        for code, subregion in subregions.items():
            if code == 'US_AVG':
                continue

            year_data = subregion.get('2023', {})
            intensity = year_data.get('co2e_kg_per_mwh', 0)

            if intensity < lowest_val and intensity > 0:
                lowest_val = intensity
                lowest_code = code

            if intensity > highest_val:
                highest_val = intensity
                highest_code = code

            total_gen += year_data.get('total_generation_mwh', 0)
            total_plants += year_data.get('plant_count', 0)
            all_states.update(subregion.get('states', []))

        return EGridStatistics(
            us_average_co2e_kg_mwh=us_avg.get('co2e_kg_per_mwh', 0),
            lowest_intensity_subregion=lowest_code,
            lowest_intensity_value=lowest_val,
            highest_intensity_subregion=highest_code,
            highest_intensity_value=highest_val,
            total_generation_mwh=total_gen,
            total_plants=total_plants,
            generation_mix=us_avg.get('generation_mix', {}),
            subregion_count=len([s for s in subregions if s != 'US_AVG']),
            state_coverage=len(all_states)
        )

    def lookup_by_state(self, state_code: str) -> Optional[Dict[str, Any]]:
        """
        Look up grid intensity for a US state.

        Args:
            state_code: Two-letter state code (e.g., CA, TX)

        Returns:
            Grid intensity data or None
        """
        if not self._loaded:
            self.load()

        state_code = state_code.upper()

        # Check state averages first
        state_data = self.data.get('state_averages', {}).get(state_code, {})
        year_data = state_data.get('2023', {})

        if year_data:
            subregion_code = year_data.get('primary_subregion')
            subregion = self.data.get('subregions', {}).get(subregion_code, {})

            return {
                'state': state_code,
                'primary_subregion': subregion_code,
                'co2e_kg_per_mwh': year_data.get('co2e_kg_per_mwh', subregion.get('2023', {}).get('co2e_kg_per_mwh')),
                'subregion_name': subregion.get('name', subregion_code),
                'generation_mix': subregion.get('2023', {}).get('generation_mix', {})
            }

        return None

    def lookup_by_subregion(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Look up grid intensity for an eGRID subregion.

        Args:
            code: eGRID subregion code (e.g., CAMX, ERCT)

        Returns:
            Subregion data or None
        """
        if not self._loaded:
            self.load()

        code = code.upper()
        subregion = self.data.get('subregions', {}).get(code)

        if subregion:
            year_data = subregion.get('2023', {})
            return {
                'code': code,
                'name': subregion.get('name', code),
                'states': subregion.get('states', []),
                'co2e_kg_per_mwh': year_data.get('co2e_kg_per_mwh', 0),
                'co2_lb_per_mwh': year_data.get('co2_lb_per_mwh', 0),
                'generation_mix': year_data.get('generation_mix', {}),
                'total_generation_mwh': year_data.get('total_generation_mwh', 0),
                'plant_count': year_data.get('plant_count', 0)
            }

        return None

    def export_to_csv(self, output_path: Optional[Path] = None) -> Path:
        """
        Export eGRID data to CSV format.

        Args:
            output_path: Output file path

        Returns:
            Path to created CSV file
        """
        if not self._loaded:
            self.load()

        if output_path is None:
            output_path = self.data_dir.parent / "exports" / f"egrid_2023_{datetime.now().strftime('%Y%m%d')}.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        subregions = self.data.get('subregions', {})

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'subregion_code',
                'subregion_name',
                'states',
                'co2e_kg_per_mwh',
                'co2_lb_per_mwh',
                'nox_lb_per_mwh',
                'so2_lb_per_mwh',
                'coal_pct',
                'natural_gas_pct',
                'nuclear_pct',
                'hydro_pct',
                'wind_pct',
                'solar_pct',
                'total_generation_mwh',
                'plant_count'
            ])

            # Data rows
            for code, subregion in subregions.items():
                year_data = subregion.get('2023', {})
                gen_mix = year_data.get('generation_mix', {})

                writer.writerow([
                    code,
                    subregion.get('name', code),
                    ';'.join(subregion.get('states', [])),
                    year_data.get('co2e_kg_per_mwh', ''),
                    year_data.get('co2_lb_per_mwh', ''),
                    year_data.get('nox_lb_per_mwh', ''),
                    year_data.get('so2_lb_per_mwh', ''),
                    gen_mix.get('coal', ''),
                    gen_mix.get('natural_gas', ''),
                    gen_mix.get('nuclear', ''),
                    gen_mix.get('hydro', ''),
                    gen_mix.get('wind', ''),
                    gen_mix.get('solar', ''),
                    year_data.get('total_generation_mwh', ''),
                    year_data.get('plant_count', '')
                ])

        logger.info(f"Exported eGRID data to {output_path}")
        return output_path

    def print_report(self):
        """Print comprehensive eGRID report."""
        stats = self.get_statistics()
        validation = self.validate()

        report = f"""
================================================================================
EPA eGRID 2023 DATA REPORT
================================================================================

Data Source: EPA Emissions & Generation Resource Integrated Database
Data Year:   2022 (published 2024)
Coverage:    United States electricity generation

--------------------------------------------------------------------------------
SUMMARY STATISTICS
--------------------------------------------------------------------------------
Total Subregions:          {stats.subregion_count}
Total States Covered:      {stats.state_coverage}
Total Power Plants:        {stats.total_plants:,}
Total Generation:          {stats.total_generation_mwh:,} MWh

US Average Intensity:      {stats.us_average_co2e_kg_mwh:.1f} kgCO2e/MWh

Lowest Intensity:          {stats.lowest_intensity_subregion} ({stats.lowest_intensity_value:.1f} kgCO2e/MWh)
Highest Intensity:         {stats.highest_intensity_subregion} ({stats.highest_intensity_value:.1f} kgCO2e/MWh)

--------------------------------------------------------------------------------
US GENERATION MIX
--------------------------------------------------------------------------------
"""
        for fuel, pct in sorted(stats.generation_mix.items(), key=lambda x: x[1], reverse=True):
            bar = '#' * int(pct / 2)
            report += f"  {fuel:15s} {pct:5.1f}% {bar}\n"

        report += """
--------------------------------------------------------------------------------
VALIDATION RESULTS
--------------------------------------------------------------------------------
"""
        report += f"Status:    {'PASSED' if validation.is_valid else 'FAILED'}\n"
        report += f"Subregions: {validation.total_subregions}\n"
        report += f"States:     {validation.total_states}\n"
        report += f"Plants:     {validation.total_plants_sampled:,}\n"

        if validation.errors:
            report += f"\nErrors ({len(validation.errors)}):\n"
            for err in validation.errors:
                report += f"  - {err}\n"

        if validation.warnings:
            report += f"\nWarnings ({len(validation.warnings)}):\n"
            for warn in validation.warnings[:10]:
                report += f"  - {warn}\n"
            if len(validation.warnings) > 10:
                report += f"  ... and {len(validation.warnings) - 10} more\n"

        report += """
--------------------------------------------------------------------------------
SUBREGION INTENSITY RANKING (kgCO2e/MWh)
--------------------------------------------------------------------------------
"""
        # Sort by intensity
        sorted_regions = sorted(
            validation.subregion_summary,
            key=lambda x: x.get('co2e_kg_per_mwh', 0)
        )

        for i, region in enumerate(sorted_regions[:15], 1):
            intensity = region.get('co2e_kg_per_mwh', 0)
            bar = '#' * int(intensity / 50)
            report += f"  {i:2d}. {region['code']:6s} {intensity:6.1f} {bar}\n"

        if len(sorted_regions) > 15:
            report += f"  ... and {len(sorted_regions) - 15} more subregions\n"

        report += """
================================================================================
END OF REPORT
================================================================================
"""
        print(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EPA eGRID 2023 Data Loader")
    parser.add_argument('--validate', action='store_true', help='Validate eGRID data')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--export-csv', action='store_true', help='Export to CSV')
    parser.add_argument('--lookup', type=str, help='Lookup by state code (e.g., CA)')
    parser.add_argument('--subregion', type=str, help='Lookup by subregion code (e.g., CAMX)')
    parser.add_argument('--report', action='store_true', help='Generate full report')

    args = parser.parse_args()

    loader = EGridLoader()

    if args.validate:
        result = loader.validate()
        print(f"\nValidation: {'PASSED' if result.is_valid else 'FAILED'}")
        print(f"Subregions: {result.total_subregions}")
        print(f"States: {result.total_states}")
        print(f"Plants: {result.total_plants_sampled:,}")

        if result.errors:
            print(f"\nErrors: {len(result.errors)}")
            for err in result.errors:
                print(f"  - {err}")

        if result.warnings:
            print(f"\nWarnings: {len(result.warnings)}")
            for warn in result.warnings[:5]:
                print(f"  - {warn}")

        sys.exit(0 if result.is_valid else 1)

    elif args.stats:
        stats = loader.get_statistics()
        print(f"\nUS Average: {stats.us_average_co2e_kg_mwh:.1f} kgCO2e/MWh")
        print(f"Lowest: {stats.lowest_intensity_subregion} ({stats.lowest_intensity_value:.1f})")
        print(f"Highest: {stats.highest_intensity_subregion} ({stats.highest_intensity_value:.1f})")
        print(f"Total Plants: {stats.total_plants:,}")
        print(f"Total Generation: {stats.total_generation_mwh:,} MWh")

    elif args.export_csv:
        path = loader.export_to_csv()
        print(f"Exported to: {path}")

    elif args.lookup:
        result = loader.lookup_by_state(args.lookup)
        if result:
            print(f"\nState: {result['state']}")
            print(f"Primary Subregion: {result['primary_subregion']} ({result['subregion_name']})")
            print(f"Grid Intensity: {result['co2e_kg_per_mwh']:.1f} kgCO2e/MWh")
            print("\nGeneration Mix:")
            for fuel, pct in sorted(result['generation_mix'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {fuel}: {pct:.1f}%")
        else:
            print(f"State not found: {args.lookup}")
            sys.exit(1)

    elif args.subregion:
        result = loader.lookup_by_subregion(args.subregion)
        if result:
            print(f"\nSubregion: {result['code']} ({result['name']})")
            print(f"States: {', '.join(result['states'])}")
            print(f"Grid Intensity: {result['co2e_kg_per_mwh']:.1f} kgCO2e/MWh")
            print(f"Plants: {result['plant_count']:,}")
            print(f"Generation: {result['total_generation_mwh']:,} MWh")
            print("\nGeneration Mix:")
            for fuel, pct in sorted(result['generation_mix'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {fuel}: {pct:.1f}%")
        else:
            print(f"Subregion not found: {args.subregion}")
            sys.exit(1)

    elif args.report:
        loader.print_report()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
