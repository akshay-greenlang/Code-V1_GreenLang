"""
Memory Profiling Tool
====================

Track memory usage over time and detect memory leaks.

Features:
- Line-by-line memory profiling
- Memory leak detection
- Heap snapshot comparison
- Memory timeline visualization
- Top memory consumers identification

Usage:
    # Profile a function
    python profile_memory.py --function "mymodule.myfunction"

    # Profile a script
    python profile_memory.py --script "path/to/script.py"

    # Detect memory leaks
    python profile_memory.py --script "script.py" --leak-detection

    # Compare heap snapshots
    python profile_memory.py --compare snapshot1.json snapshot2.json

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import sys
import gc
import time
import argparse
import importlib
import tracemalloc
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import linecache


class MemoryProfiler:
    """Memory profiling with leak detection."""

    def __init__(self, output_dir: Path = None):
        """
        Initialize memory profiler.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir or Path("./profiles")
        self.output_dir.mkdir(exist_ok=True)

        self.snapshots = []
        self.timeline = []
        self.tracking = False

    def start(self, trace_malloc: bool = True):
        """
        Start memory tracking.

        Args:
            trace_malloc: Use tracemalloc for detailed tracking
        """
        if trace_malloc:
            tracemalloc.start()

        self.tracking = True
        self.snapshots = []
        self.timeline = []

        # Initial snapshot
        self._take_snapshot("start")

    def stop(self):
        """Stop memory tracking."""
        self.tracking = False

        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def _take_snapshot(self, label: str = None):
        """
        Take a memory snapshot.

        Args:
            label: Snapshot label
        """
        if not tracemalloc.is_tracing():
            return

        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            "label": label or f"snapshot_{len(self.snapshots)}",
            "timestamp": time.time(),
            "snapshot": snapshot
        })

    def record_checkpoint(self, label: str):
        """
        Record a checkpoint in memory timeline.

        Args:
            label: Checkpoint label
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        self.timeline.append({
            "label": label,
            "timestamp": time.time(),
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024)
        })

        # Also take tracemalloc snapshot
        self._take_snapshot(label)

    def profile_function(self, func, *args, **kwargs):
        """
        Profile a function's memory usage.

        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, memory_stats)
        """
        self.start()

        try:
            self.record_checkpoint("before_execution")
            result = func(*args, **kwargs)
            self.record_checkpoint("after_execution")

            stats = self.get_memory_stats()
            return result, stats

        finally:
            self.stop()

    def get_memory_stats(self) -> Dict:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory stats
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        stats = {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            stats.update({
                "current_mb": current / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024)
            })

        return stats

    def detect_leaks(self, threshold_mb: float = 10.0) -> List[Dict]:
        """
        Detect potential memory leaks.

        Args:
            threshold_mb: Memory growth threshold in MB

        Returns:
            List of potential leaks
        """
        if len(self.timeline) < 2:
            return []

        leaks = []

        # Check for monotonic memory growth
        for i in range(1, len(self.timeline)):
            prev = self.timeline[i - 1]
            curr = self.timeline[i]

            growth = curr["rss_mb"] - prev["rss_mb"]

            if growth > threshold_mb:
                leaks.append({
                    "from": prev["label"],
                    "to": curr["label"],
                    "growth_mb": growth,
                    "severity": "high" if growth > 50 else "medium"
                })

        return leaks

    def compare_snapshots(
        self,
        snapshot1_idx: int = 0,
        snapshot2_idx: int = -1,
        top_n: int = 20
    ) -> List[Dict]:
        """
        Compare two memory snapshots.

        Args:
            snapshot1_idx: Index of first snapshot
            snapshot2_idx: Index of second snapshot
            top_n: Number of top differences to return

        Returns:
            List of memory differences
        """
        if len(self.snapshots) < 2:
            return []

        snap1 = self.snapshots[snapshot1_idx]["snapshot"]
        snap2 = self.snapshots[snapshot2_idx]["snapshot"]

        top_stats = snap2.compare_to(snap1, 'lineno')

        differences = []
        for stat in top_stats[:top_n]:
            differences.append({
                "file": stat.traceback.format()[0],
                "size_diff_mb": stat.size_diff / (1024 * 1024),
                "count_diff": stat.count_diff,
                "size_mb": stat.size / (1024 * 1024)
            })

        return differences

    def get_top_allocations(self, limit: int = 20) -> List[Dict]:
        """
        Get top memory allocations.

        Args:
            limit: Number of top allocations to return

        Returns:
            List of top allocations
        """
        if not self.snapshots:
            return []

        snapshot = self.snapshots[-1]["snapshot"]
        top_stats = snapshot.statistics('lineno')

        allocations = []
        for stat in top_stats[:limit]:
            frame = stat.traceback[0]
            allocations.append({
                "file": frame.filename,
                "line": frame.lineno,
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count,
                "code": linecache.getline(frame.filename, frame.lineno).strip()
            })

        return allocations

    def generate_html_report(self, output_file: str = "memory_report.html"):
        """
        Generate comprehensive HTML memory report.

        Args:
            output_file: Output filename
        """
        top_allocations = self.get_top_allocations(limit=20)
        leaks = self.detect_leaks()

        # Calculate timeline stats
        if self.timeline:
            start_mem = self.timeline[0]["rss_mb"]
            end_mem = self.timeline[-1]["rss_mb"]
            peak_mem = max(t["rss_mb"] for t in self.timeline)
            mem_growth = end_mem - start_mem
        else:
            start_mem = end_mem = peak_mem = mem_growth = 0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Profile Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
            border-bottom: 3px solid #e74c3c;
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
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        .danger {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #e74c3c;
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
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        #timeline {{
            margin: 20px 0;
            height: 400px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Memory Profile Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-label">Start Memory</div>
                <div class="summary-value">{start_mem:.1f} MB</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">End Memory</div>
                <div class="summary-value">{end_mem:.1f} MB</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Peak Memory</div>
                <div class="summary-value">{peak_mem:.1f} MB</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Growth</div>
                <div class="summary-value" style="color: {'#e74c3c' if mem_growth > 50 else '#27ae60'}">
                    {mem_growth:+.1f} MB
                </div>
            </div>
        </div>
        """

        # Add leak warnings
        if leaks:
            html += "<h2>Potential Memory Leaks</h2>"
            for leak in leaks:
                severity_class = "danger" if leak["severity"] == "high" else "warning"
                html += f"""
                <div class="{severity_class}">
                    <strong>Memory growth detected:</strong> {leak['from']} → {leak['to']}<br>
                    <strong>Growth:</strong> {leak['growth_mb']:.1f} MB
                    ({leak['severity']} severity)
                </div>
                """

        # Memory timeline chart
        if self.timeline:
            timestamps = [t["label"] for t in self.timeline]
            memory_values = [t["rss_mb"] for t in self.timeline]

            html += f"""
            <h2>Memory Timeline</h2>
            <div id="timeline"></div>
            <script>
                var trace = {{
                    x: {json.dumps(timestamps)},
                    y: {json.dumps(memory_values)},
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{color: '#e74c3c', width: 2}},
                    marker: {{size: 8}}
                }};

                var layout = {{
                    title: 'Memory Usage Over Time',
                    xaxis: {{title: 'Checkpoint'}},
                    yaxis: {{title: 'Memory (MB)'}}
                }};

                Plotly.newPlot('timeline', [trace], layout);
            </script>
            """

        # Top allocations table
        html += """
        <h2>Top 20 Memory Allocations</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Line</th>
                    <th>Size (MB)</th>
                    <th>Count</th>
                    <th>Code</th>
                </tr>
            </thead>
            <tbody>
        """

        for alloc in top_allocations:
            html += f"""
                <tr>
                    <td>{alloc['file']}</td>
                    <td>{alloc['line']}</td>
                    <td>{alloc['size_mb']:.2f}</td>
                    <td>{alloc['count']}</td>
                    <td><code>{alloc['code']}</code></td>
                </tr>
            """

        html += """
            </tbody>
        </table>

        <h2>Recommendations</h2>
        <ul>
        """

        # Add recommendations
        if mem_growth > 50:
            html += f"<li>Memory grew by {mem_growth:.1f} MB during execution. Investigate potential leaks.</li>"

        if leaks:
            html += f"<li>Detected {len(leaks)} potential memory leak(s). Review the highlighted sections.</li>"

        for alloc in top_allocations[:3]:
            if alloc['size_mb'] > 10:
                html += f"""
                <li>Large allocation ({alloc['size_mb']:.1f} MB) at
                    <code>{alloc['file']}:{alloc['line']}</code>. Consider optimization.</li>
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

        print(f"Memory report saved to: {output_path}")

    def save_snapshot(self, filename: str = None) -> Path:
        """
        Save current snapshot to JSON.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_snapshot_{timestamp}.json"

        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "timeline": self.timeline,
            "stats": self.get_memory_stats()
        }

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(snapshot_data, f, indent=2)

        return output_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Memory Profiling Tool")

    parser.add_argument("--script", help="Python script to profile")
    parser.add_argument("--function", help="Function to profile (module.function)")
    parser.add_argument("--args", help="Function arguments (comma-separated)")
    parser.add_argument("--output-dir", default="./profiles", help="Output directory")
    parser.add_argument("--leak-detection", action="store_true", help="Enable leak detection")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--snapshot", action="store_true", help="Save snapshot to JSON")

    args = parser.parse_args()

    profiler = MemoryProfiler(output_dir=Path(args.output_dir))

    # Profile script or function
    if args.script:
        profiler.start()

        with open(args.script) as f:
            code = compile(f.read(), args.script, 'exec')

        profiler.record_checkpoint("before_script")
        exec(code, {'__name__': '__main__'})
        profiler.record_checkpoint("after_script")

        profiler.stop()

    elif args.function:
        # Import and run function
        module_name, func_name = args.function.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        func_args = []
        if args.args:
            func_args = args.args.split(",")

        result, stats = profiler.profile_function(func, *func_args)

        print("\nMemory Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")

    else:
        parser.print_help()
        return

    # Leak detection
    if args.leak_detection:
        leaks = profiler.detect_leaks()
        if leaks:
            print(f"\n⚠️  Detected {len(leaks)} potential memory leak(s):")
            for leak in leaks:
                print(f"  - {leak['from']} → {leak['to']}: +{leak['growth_mb']:.1f} MB")
        else:
            print("\n✓ No memory leaks detected")

    # Generate report
    if args.report:
        profiler.generate_html_report()

    # Save snapshot
    if args.snapshot:
        snapshot_file = profiler.save_snapshot()
        print(f"Snapshot saved to: {snapshot_file}")


if __name__ == "__main__":
    main()
