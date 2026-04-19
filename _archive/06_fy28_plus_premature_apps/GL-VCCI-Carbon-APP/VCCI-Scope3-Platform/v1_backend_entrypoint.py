"""Stable v1 backend entrypoint for native VCCI execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cli.commands.pipeline import PipelineExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VCCI pipeline for GreenLang v1 backend adapter")
    parser.add_argument("--input", required=True, help="Input file or directory path")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--categories", default="1", help="Comma-separated Scope 3 categories")
    parser.add_argument("--report-format", default="ghg-protocol", help="Report output format")
    parser.add_argument("--enable-llm", action="store_true", default=True)
    parser.add_argument("--disable-llm", action="store_true", default=False)
    parser.add_argument("--enable-monte-carlo", action="store_true", default=True)
    parser.add_argument("--disable-monte-carlo", action="store_true", default=False)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_list = [int(cat.strip()) for cat in args.categories.split(",") if cat.strip()]
    enable_llm = args.enable_llm and not args.disable_llm
    enable_monte_carlo = args.enable_monte_carlo and not args.disable_monte_carlo

    executor = PipelineExecutor()
    result = executor.run(
        input_path=input_path,
        output_dir=output_dir,
        categories=category_list or [1],
        enable_llm=enable_llm,
        enable_monte_carlo=enable_monte_carlo,
        report_format=args.report_format,
    )
    print(json.dumps({"run_id": result.get("run_id"), "status": result.get("overall_status")}))
    return 0 if result.get("overall_status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

