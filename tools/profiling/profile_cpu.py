# -*- coding: utf-8 -*-
"""
CPU Profiling Tool with Flame Graph Generation
==============================================

Wrapper around cProfile with flame graph generation for visual performance analysis.

Features:
- Function-level CPU profiling
- Call graph generation
- Flame graph visualization (HTML)
- Hotspot identification
- Comparative profiling (before/after)

Usage:
    # Profile a function
    python profile_cpu.py --function "mymodule.myfunction" --args "arg1,arg2"

    # Profile a script
    python profile_cpu.py --script "path/to/script.py"

    # Profile and generate flame graph
    python profile_cpu.py --script "script.py" --flamegraph

    # Compare two profiles
    python profile_cpu.py --compare baseline.prof current.prof

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import cProfile
import pstats
import io
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
import importlib
import json
from datetime import datetime
from greenlang.determinism import DeterministicClock


class CPUProfiler:
    """CPU profiling with flame graph generation."""

    def __init__(self, output_dir: Path = None):
        """
        Initialize CPU profiler.

        Args:
            output_dir: Directory for output files (default: ./profiles)
        """
        self.output_dir = output_dir or Path("./profiles")
        self.output_dir.mkdir(exist_ok=True)

        self.profiler = cProfile.Profile()
        self.stats = None

    def start(self):
        """Start profiling."""
        self.profiler.enable()

    def stop(self):
        """Stop profiling."""
        self.profiler.disable()

    def profile_function(self, func, *args, **kwargs):
        """
        Profile a specific function.

        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        self.start()
        try:
            result = func(*args, **kwargs)
        finally:
            self.stop()

        return result

    def profile_script(self, script_path: str):
        """
        Profile a Python script.

        Args:
            script_path: Path to script to profile
        """
        with open(script_path) as f:
            code = compile(f.read(), script_path, 'exec')

        self.start()
        try:
            exec(code, {'__name__': '__main__'})
        finally:
            self.stop()

    def save_stats(self, filename: str = None) -> Path:
        """
        Save profiling stats to file.

        Args:
            filename: Output filename (default: profile_TIMESTAMP.prof)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{timestamp}.prof"

        output_path = self.output_dir / filename
        self.profiler.dump_stats(str(output_path))

        return output_path

    def print_stats(self, sort_by: str = "cumulative", limit: int = 50):
        """
        Print profiling statistics.

        Args:
            sort_by: Sort criterion (cumulative, time, calls)
            limit: Number of functions to show
        """
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(sort_by)
        stats.print_stats(limit)

        print(stream.getvalue())
        self.stats = stats

    def get_hotspots(self, limit: int = 10) -> List[dict]:
        """
        Get CPU hotspots (most time-consuming functions).

        Args:
            limit: Number of hotspots to return

        Returns:
            List of hotspot dictionaries
        """
        if self.stats is None:
            self.stats = pstats.Stats(self.profiler)

        hotspots = []
        stats_dict = self.stats.stats

        # Sort by cumulative time
        sorted_stats = sorted(
            stats_dict.items(),
            key=lambda x: x[1][3],  # cumtime
            reverse=True
        )

        for (func, (cc, nc, tt, ct, callers)), _ in zip(sorted_stats[:limit], range(limit)):
            file, line, name = func
            hotspots.append({
                "function": name,
                "file": file,
                "line": line,
                "calls": cc,
                "time": tt,
                "cumtime": ct,
                "percall": tt / cc if cc > 0 else 0
            })

        return hotspots

    def generate_callgraph(self, output_file: str = "callgraph.png"):
        """
        Generate call graph visualization (requires gprof2dot and graphviz).

        Args:
            output_file: Output filename
        """
        # Save stats to temp file
        stats_file = self.save_stats("temp_profile.prof")

        try:
            # Convert to dot format
            dot_file = self.output_dir / "callgraph.dot"
            subprocess.run([
                "gprof2dot",
                "-f", "pstats",
                str(stats_file),
                "-o", str(dot_file)
            ], check=True)

            # Convert to image
            output_path = self.output_dir / output_file
            subprocess.run([
                "dot",
                "-Tpng",
                str(dot_file),
                "-o", str(output_path)
            ], check=True)

            print(f"Call graph saved to: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"Error generating call graph: {e}")
            print("Ensure gprof2dot and graphviz are installed:")
            print("  pip install gprof2dot")
            print("  # Install graphviz from https://graphviz.org/")

    def generate_flamegraph(self, output_file: str = "flamegraph.html"):
        """
        Generate flame graph visualization.

        Args:
            output_file: Output filename
        """
        # Save stats
        stats_file = self.save_stats("temp_profile.prof")

        try:
            # Convert to flamegraph format
            import flameprof

            output_path = self.output_dir / output_file
            flameprof.main([str(stats_file), "-o", str(output_path)])

            print(f"Flame graph saved to: {output_path}")

        except ImportError:
            print("Flame graph generation requires 'flameprof'")
            print("Install with: pip install flameprof")

        except Exception as e:
            print(f"Error generating flame graph: {e}")

    def generate_html_report(self, output_file: str = "profile_report.html"):
        """
        Generate comprehensive HTML report.

        Args:
            output_file: Output filename
        """
        hotspots = self.get_hotspots(limit=20)

        # Calculate total time
        total_time = sum(h["cumtime"] for h in hotspots)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CPU Profile Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-item {{
            display: inline-block;
            margin-right: 30px;
        }}
        .summary-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .summary-value {{
            font-size: 1.5em;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .bar {{
            background: #3498db;
            height: 20px;
            border-radius: 3px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CPU Profile Report</h1>
        <p class="timestamp">Generated: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-label">Total Time</div>
                <div class="summary-value">{total_time:.3f}s</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Functions Profiled</div>
                <div class="summary-value">{len(hotspots)}</div>
            </div>
        </div>

        <h2>Top 20 CPU Hotspots</h2>
        <table>
            <thead>
                <tr>
                    <th>Function</th>
                    <th>File</th>
                    <th>Calls</th>
                    <th>Total Time (s)</th>
                    <th>Cumulative Time (s)</th>
                    <th>Per Call (ms)</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
        """

        for hotspot in hotspots:
            percent = (hotspot["cumtime"] / total_time * 100) if total_time > 0 else 0
            html += f"""
                <tr>
                    <td><code>{hotspot["function"]}</code></td>
                    <td>{hotspot["file"]}:{hotspot["line"]}</td>
                    <td>{hotspot["calls"]}</td>
                    <td>{hotspot["time"]:.4f}</td>
                    <td>{hotspot["cumtime"]:.4f}</td>
                    <td>{hotspot["percall"] * 1000:.2f}</td>
                    <td>
                        <div style="display: flex; align-items: center;">
                            <div class="bar" style="width: {percent}%;"></div>
                            <span style="margin-left: 5px;">{percent:.1f}%</span>
                        </div>
                    </td>
                </tr>
            """

        html += """
            </tbody>
        </table>

        <h2>Recommendations</h2>
        <ul>
        """

        # Add recommendations based on hotspots
        for hotspot in hotspots[:5]:
            if hotspot["cumtime"] > total_time * 0.1:  # > 10% of total time
                html += f"""
            <li><strong>{hotspot["function"]}</strong> consumes {hotspot["cumtime"] / total_time * 100:.1f}%
                of total time. Consider optimization.</li>
                """

        html += """
        </ul>
    </div>
</body>
</html>
        """

        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write(html)

        print(f"HTML report saved to: {output_path}")

    def compare_profiles(self, baseline_file: str, current_file: str):
        """
        Compare two profile files.

        Args:
            baseline_file: Baseline profile file
            current_file: Current profile file
        """
        baseline_stats = pstats.Stats(baseline_file)
        current_stats = pstats.Stats(current_file)

        print("\n" + "=" * 80)
        print("PROFILE COMPARISON")
        print("=" * 80)
        print(f"Baseline: {baseline_file}")
        print(f"Current:  {current_file}")
        print("=" * 80)

        # Get top functions from both
        baseline_dict = baseline_stats.stats
        current_dict = current_stats.stats

        # Compare common functions
        common_funcs = set(baseline_dict.keys()) & set(current_dict.keys())

        changes = []
        for func in common_funcs:
            baseline_time = baseline_dict[func][3]  # cumtime
            current_time = current_dict[func][3]

            if baseline_time > 0:
                percent_change = ((current_time - baseline_time) / baseline_time) * 100
                if abs(percent_change) > 5:  # Only show significant changes
                    changes.append({
                        "function": func[2],
                        "baseline": baseline_time,
                        "current": current_time,
                        "change": percent_change
                    })

        # Sort by absolute change
        changes.sort(key=lambda x: abs(x["change"]), reverse=True)

        print("\nSignificant Changes (> 5%):")
        print(f"{'Function':<50} {'Baseline':<12} {'Current':<12} {'Change':<12}")
        print("-" * 86)

        for change in changes[:20]:
            color = "\033[91m" if change["change"] > 0 else "\033[92m"  # Red if slower, green if faster
            reset = "\033[0m"
            print(
                f"{change['function']:<50} "
                f"{change['baseline']:<12.4f} "
                f"{change['current']:<12.4f} "
                f"{color}{change['change']:>+11.1f}%{reset}"
            )

        print("\n" + "=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="CPU Profiling Tool")

    parser.add_argument("--script", help="Python script to profile")
    parser.add_argument("--function", help="Function to profile (module.function)")
    parser.add_argument("--args", help="Function arguments (comma-separated)")
    parser.add_argument("--output-dir", default="./profiles", help="Output directory")
    parser.add_argument("--flamegraph", action="store_true", help="Generate flame graph")
    parser.add_argument("--callgraph", action="store_true", help="Generate call graph")
    parser.add_argument("--compare", nargs=2, help="Compare two profile files")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")

    args = parser.parse_args()

    profiler = CPUProfiler(output_dir=Path(args.output_dir))

    if args.compare:
        profiler.compare_profiles(args.compare[0], args.compare[1])
        return

    # Profile script or function
    if args.script:
        profiler.profile_script(args.script)
    elif args.function:
        # Import and run function
        module_name, func_name = args.function.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        func_args = []
        if args.args:
            func_args = args.args.split(",")

        profiler.profile_function(func, *func_args)
    else:
        parser.print_help()
        return

    # Print stats
    profiler.print_stats()

    # Save stats
    stats_file = profiler.save_stats()
    print(f"\nProfile stats saved to: {stats_file}")

    # Generate visualizations
    if args.flamegraph:
        profiler.generate_flamegraph()

    if args.callgraph:
        profiler.generate_callgraph()

    if args.report:
        profiler.generate_html_report()


if __name__ == "__main__":
    main()
