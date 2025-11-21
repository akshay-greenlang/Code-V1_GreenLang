#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Validation Script for GreenLang Pipeline Spec v1.0

This script finds and validates all gl.yaml files in the repository using the
PipelineSpec validator. Provides comprehensive reporting and proper exit codes
for CI/CD integration.

Usage:
    python scripts/validate_all_pipelines.py [options]

Examples:
    python scripts/validate_all_pipelines.py --verbose
    python scripts/validate_all_pipelines.py --file path/to/pipeline.yaml
    python scripts/validate_all_pipelines.py --directory packs/ --parallel
    python scripts/validate_all_pipelines.py --report-format json --output report.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from dataclasses import dataclass, asdict

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from greenlang.sdk.pipeline_spec import PipelineSpec
    from pydantic import ValidationError
except ImportError as e:
    print(f"❌ Error importing GreenLang modules: {e}")
    print("Make sure you've installed GreenLang with: pip install -e .")
    sys.exit(1)


@dataclass
class ValidationResult:
    """Result of validating a single pipeline file."""
    file_path: str
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    pipeline_name: Optional[str] = None
    pipeline_version: Optional[str] = None
    step_count: Optional[int] = None
    validation_time_ms: Optional[float] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class ValidationSummary:
    """Summary of all pipeline validations."""
    total_files: int
    valid_files: int
    invalid_files: int
    total_time_ms: float
    results: List[ValidationResult]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.valid_files / self.total_files) * 100


class PipelineValidator:
    """Validates GreenLang pipeline files against Pipeline Spec v1.0."""

    def __init__(self, verbose: bool = False, strict: bool = False):
        self.verbose = verbose
        self.strict = strict

    def find_pipeline_files(self, directory: Path, pattern: str = "gl.yaml") -> List[Path]:
        """Find all pipeline files in the given directory."""
        files = []

        if directory.is_file():
            files.append(directory)
        else:
            # Find all gl.yaml files
            files.extend(directory.rglob(pattern))

            # Also find YAML files in pipelines directories
            pipeline_dirs = directory.rglob("pipelines")
            for pipeline_dir in pipeline_dirs:
                if pipeline_dir.is_dir():
                    files.extend(pipeline_dir.glob("*.yaml"))
                    files.extend(pipeline_dir.glob("*.yml"))

        # Remove duplicates and sort
        unique_files = sorted(set(files))

        if self.verbose:
            print(f"🔍 Found {len(unique_files)} pipeline files to validate")
            for file_path in unique_files[:10]:  # Show first 10
                print(f"   - {file_path}")
            if len(unique_files) > 10:
                print(f"   ... and {len(unique_files) - 10} more")

        return unique_files

    def validate_single_file(self, file_path: Path) -> ValidationResult:
        """Validate a single pipeline file."""
        start_time = time.time()
        result = ValidationResult(file_path=str(file_path), is_valid=False)

        try:
            # Read and parse YAML
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)

            if not yaml_content:
                result.errors.append("Empty YAML file")
                return result

            # Check if it's a pipeline file (has steps or looks like a pipeline)
            if not self._looks_like_pipeline(yaml_content):
                result.warnings.append("File doesn't appear to be a pipeline specification")
                if not self.strict:
                    result.is_valid = True
                    return result

            # Validate against PipelineSpec
            try:
                pipeline_spec = PipelineSpec(**yaml_content)
                result.is_valid = True
                result.pipeline_name = pipeline_spec.name
                result.pipeline_version = pipeline_spec.version
                result.step_count = len(pipeline_spec.steps)

                # Additional validation checks
                self._additional_validation_checks(pipeline_spec, result)

            except ValidationError as e:
                result.is_valid = False
                result.errors.extend([str(error) for error in e.errors()])

        except yaml.YAMLError as e:
            result.errors.append(f"YAML parsing error: {str(e)}")
        except Exception as e:
            result.errors.append(f"Unexpected error: {str(e)}")

        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000

        return result

    def _looks_like_pipeline(self, yaml_content: Dict) -> bool:
        """Check if the YAML content looks like a pipeline specification."""
        # Must have steps to be a pipeline
        if 'steps' not in yaml_content:
            return False

        # Must have a name
        if 'name' not in yaml_content:
            return False

        # Steps must be a list
        if not isinstance(yaml_content['steps'], list):
            return False

        return True

    def _additional_validation_checks(self, pipeline_spec: PipelineSpec, result: ValidationResult):
        """Perform additional validation checks beyond schema validation."""

        # Check for common issues
        step_names = [step.name for step in pipeline_spec.steps]

        # Check for suspicious patterns
        if len(step_names) == 1:
            result.warnings.append("Pipeline has only one step - consider if this is intentional")

        # Check for parallel steps
        parallel_steps = [step for step in pipeline_spec.steps if step.parallel]
        if len(parallel_steps) > 5:
            result.warnings.append(f"High number of parallel steps ({len(parallel_steps)}) may impact performance")

        # Check for agent references
        agents = [step.agent for step in pipeline_spec.steps]
        unknown_agents = []
        known_agents = {
            'ValidatorAgent', 'FuelAgent', 'CarbonAgent', 'IntensityAgent',
            'BenchmarkAgent', 'GridFactorAgent', 'BuildingProfileAgent',
            'RecommendationAgent', 'ReportAgent'
        }

        for agent in agents:
            if agent not in known_agents and not agent.endswith('Agent'):
                unknown_agents.append(agent)

        if unknown_agents:
            result.warnings.append(f"Unknown agent references: {', '.join(unknown_agents)}")

    def validate_files(self, file_paths: List[Path], parallel: bool = False, max_workers: int = 4) -> ValidationSummary:
        """Validate multiple pipeline files."""
        start_time = time.time()
        results = []

        if parallel and len(file_paths) > 1:
            if self.verbose:
                print(f"🚀 Validating {len(file_paths)} files in parallel (max_workers={max_workers})")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.validate_single_file, file_path): file_path
                    for file_path in file_paths
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if self.verbose:
                            status = "✅" if result.is_valid else "❌"
                            print(f"{status} {file_path}")
                    except Exception as e:
                        results.append(ValidationResult(
                            file_path=str(file_path),
                            is_valid=False,
                            errors=[f"Validation failed: {str(e)}"]
                        ))
        else:
            if self.verbose:
                print(f"🔄 Validating {len(file_paths)} files sequentially")

            for i, file_path in enumerate(file_paths, 1):
                result = self.validate_single_file(file_path)
                results.append(result)

                if self.verbose:
                    status = "✅" if result.is_valid else "❌"
                    print(f"{status} [{i}/{len(file_paths)}] {file_path}")

        total_time = (time.time() - start_time) * 1000
        valid_count = sum(1 for r in results if r.is_valid)

        return ValidationSummary(
            total_files=len(file_paths),
            valid_files=valid_count,
            invalid_files=len(file_paths) - valid_count,
            total_time_ms=total_time,
            results=results
        )


def print_summary(summary: ValidationSummary, verbose: bool = False):
    """Print validation summary to console."""
    print("\n" + "="*60)
    print("📊 PIPELINE VALIDATION SUMMARY")
    print("="*60)
    print(f"Total files validated: {summary.total_files}")
    print(f"Valid pipelines: {summary.valid_files}")
    print(f"Invalid pipelines: {summary.invalid_files}")
    print(f"Success rate: {summary.success_rate:.1f}%")
    print(f"Total time: {summary.total_time_ms:.0f}ms")

    if summary.invalid_files > 0:
        print(f"\n❌ VALIDATION ERRORS ({summary.invalid_files} files):")
        print("-" * 40)
        for result in summary.results:
            if not result.is_valid:
                print(f"\n📁 {result.file_path}")
                for error in result.errors:
                    print(f"   ❌ {error}")

    if verbose and any(r.warnings for r in summary.results):
        print(f"\n⚠️  WARNINGS:")
        print("-" * 40)
        for result in summary.results:
            if result.warnings:
                print(f"\n📁 {result.file_path}")
                for warning in result.warnings:
                    print(f"   ⚠️  {warning}")

    if summary.invalid_files == 0:
        print(f"\n🎉 All pipeline files are valid!")
    print("="*60)


def generate_report(summary: ValidationSummary, format_type: str, output_file: Optional[str] = None):
    """Generate validation report in specified format."""

    if format_type == "json":
        report_data = {
            "summary": asdict(summary),
            "timestamp": time.time(),
            "version": "1.0"
        }
        report_content = json.dumps(report_data, indent=2, default=str)

    elif format_type == "markdown":
        report_content = f"""# Pipeline Validation Report

## Summary
- **Total Files**: {summary.total_files}
- **Valid**: {summary.valid_files}
- **Invalid**: {summary.invalid_files}
- **Success Rate**: {summary.success_rate:.1f}%
- **Total Time**: {summary.total_time_ms:.0f}ms

## Results
"""
        for result in summary.results:
            status = "✅ Valid" if result.is_valid else "❌ Invalid"
            report_content += f"\n### {result.file_path}\n"
            report_content += f"**Status**: {status}\n"
            if result.pipeline_name:
                report_content += f"**Name**: {result.pipeline_name}\n"
            if result.pipeline_version:
                report_content += f"**Version**: {result.pipeline_version}\n"
            if result.step_count:
                report_content += f"**Steps**: {result.step_count}\n"

            if result.errors:
                report_content += "\n**Errors**:\n"
                for error in result.errors:
                    report_content += f"- {error}\n"

            if result.warnings:
                report_content += "\n**Warnings**:\n"
                for warning in result.warnings:
                    report_content += f"- {warning}\n"

    else:
        raise ValueError(f"Unsupported report format: {format_type}")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📄 Report saved to: {output_file}")
    else:
        print(report_content)


def main():
    """Main entry point for the pipeline validation script."""
    parser = argparse.ArgumentParser(
        description="Validate GreenLang pipeline files against Pipeline Spec v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --verbose
  %(prog)s --file path/to/pipeline.yaml
  %(prog)s --directory packs/ --parallel
  %(prog)s --report-format json --output report.json
        """
    )

    parser.add_argument(
        "--directory", "-d",
        type=Path,
        default=Path("."),
        help="Directory to search for pipeline files (default: current directory)"
    )

    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Validate a specific file instead of searching directory"
    )

    parser.add_argument(
        "--pattern",
        default="gl.yaml",
        help="File pattern to search for (default: gl.yaml)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (only errors)"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode - treat warnings as errors"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel validation of multiple files"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--report-format",
        choices=["json", "markdown"],
        help="Generate report in specified format"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file for report (default: print to stdout)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.quiet and args.verbose:
        print("❌ Error: --quiet and --verbose are mutually exclusive")
        sys.exit(1)

    # Initialize validator
    validator = PipelineValidator(verbose=args.verbose and not args.quiet, strict=args.strict)

    # Find files to validate
    if args.file:
        if not args.file.exists():
            print(f"❌ Error: File not found: {args.file}")
            sys.exit(1)
        file_paths = [args.file]
    else:
        file_paths = validator.find_pipeline_files(args.directory, args.pattern)

    if not file_paths:
        print(f"⚠️  No pipeline files found in {args.directory} with pattern '{args.pattern}'")
        sys.exit(0)

    # Validate files
    if not args.quiet:
        print(f"🔍 Validating {len(file_paths)} pipeline files...")

    summary = validator.validate_files(
        file_paths,
        parallel=args.parallel,
        max_workers=args.max_workers
    )

    # Generate report if requested
    if args.report_format:
        generate_report(summary, args.report_format, args.output)

    # Print summary unless in quiet mode
    if not args.quiet:
        print_summary(summary, verbose=args.verbose)
    elif summary.invalid_files > 0:
        # In quiet mode, still show errors
        for result in summary.results:
            if not result.is_valid:
                print(f"❌ {result.file_path}: {'; '.join(result.errors)}")

    # Exit with proper code for CI/CD
    if summary.invalid_files > 0:
        if not args.quiet:
            print(f"\n❌ Validation failed: {summary.invalid_files} invalid pipeline(s)")
        sys.exit(1)
    else:
        if not args.quiet:
            print(f"\n✅ All {summary.valid_files} pipeline files are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()