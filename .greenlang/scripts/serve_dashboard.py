#!/usr/bin/env python3
# -*- coding: utf-8 -*-

logger = logging.getLogger(__name__)
"""
GreenLang Migration Dashboard Server

Real-time dashboard showing migration progress, infrastructure usage metrics,
team leaderboard, and ADR status.
"""

import logging
import argparse
import os
import sys
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from greenlang.determinism import DeterministicClock

# Try Flask first, fall back to http.server if not available
try:
    from flask import Flask, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    from http.server import HTTPServer, SimpleHTTPRequestHandler

# Import our analysis tools
sys.path.insert(0, os.path.dirname(__file__))
try:
    from generate_usage_report import UsageAnalyzer
    from create_adr import ADRGenerator
except ImportError:
    UsageAnalyzer = None
    ADRGenerator = None


class DashboardData:
    """Manages dashboard data collection and updates."""

    def __init__(self, app_directory: str):
        self.app_directory = app_directory
        self.data = {
            'last_update': None,
            'usage': {},
            'adrs': [],
            'team_stats': {},
            'migration_progress': {}
        }
        self.lock = threading.Lock()

    def update_data(self):
        """Update all dashboard data."""
        with self.lock:
            self.data['last_update'] = DeterministicClock.now().isoformat()

            # Update usage statistics
            if UsageAnalyzer:
                analyzer = UsageAnalyzer()
                self.data['usage'] = analyzer.analyze_directory(self.app_directory)

            # Update ADR status
            if ADRGenerator:
                adr_gen = ADRGenerator()
                self.data['adrs'] = adr_gen.existing_adrs

            # Calculate migration progress
            self._calculate_migration_progress()

            # Update team stats
            self._update_team_stats()

    def _calculate_migration_progress(self):
        """Calculate overall migration progress."""
        if not self.data['usage']:
            return

        summary = self.data['usage'].get('summary', {})

        self.data['migration_progress'] = {
            'overall_ium': summary.get('overall_ium', 0),
            'adoption_rate': summary.get('adoption_rate', 0),
            'files_migrated': summary.get('files_using_greenlang', 0),
            'total_files': summary.get('total_files', 0),
            'greenlang_imports': summary.get('total_greenlang_imports', 0),
            'remaining_imports': summary.get('total_other_imports', 0)
        }

    def _update_team_stats(self):
        """Update team statistics (mock data for now)."""
        # In a real implementation, this would track git commits and file ownership
        self.data['team_stats'] = {
            'top_contributors': [
                {'name': 'Team Member 1', 'ium': 85, 'files': 45},
                {'name': 'Team Member 2', 'ium': 72, 'files': 38},
                {'name': 'Team Member 3', 'ium': 68, 'files': 32},
            ]
        }

    def get_data(self) -> Dict:
        """Get current dashboard data."""
        with self.lock:
            return self.data.copy()


# Flask-based dashboard (preferred)
if FLASK_AVAILABLE:
    app = Flask(__name__)
    dashboard_data = None

    DASHBOARD_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GreenLang Migration Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
            }

            .container {
                max-width: 1600px;
                margin: 0 auto;
            }

            .header {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                text-align: center;
            }

            .header h1 {
                color: #2c5f2d;
                font-size: 2.5em;
                margin-bottom: 10px;
            }

            .last-update {
                color: #666;
                font-size: 0.9em;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }

            .card {
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .card h2 {
                color: #2c5f2d;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #4caf50;
            }

            .metric {
                text-align: center;
                padding: 20px;
            }

            .metric-value {
                font-size: 3em;
                font-weight: bold;
                color: #2c5f2d;
                margin: 10px 0;
            }

            .metric-label {
                color: #666;
                font-size: 1em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .progress-bar {
                background: #e0e0e0;
                height: 30px;
                border-radius: 15px;
                overflow: hidden;
                margin: 10px 0;
            }

            .progress-fill {
                background: linear-gradient(90deg, #4caf50 0%, #2c5f2d 100%);
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                transition: width 0.5s ease;
            }

            .leaderboard {
                list-style: none;
            }

            .leaderboard li {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                margin: 10px 0;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #4caf50;
            }

            .rank {
                font-size: 1.5em;
                font-weight: bold;
                color: #4caf50;
                min-width: 40px;
            }

            .name {
                flex: 1;
                margin: 0 15px;
            }

            .stats {
                text-align: right;
                color: #666;
                font-size: 0.9em;
            }

            .adr-list {
                max-height: 400px;
                overflow-y: auto;
            }

            .adr-item {
                padding: 10px;
                margin: 5px 0;
                background: #f8f9fa;
                border-radius: 5px;
                font-family: monospace;
                font-size: 0.9em;
            }

            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 0.8em;
                font-weight: bold;
                margin-left: 10px;
            }

            .status-proposed { background: #ffc107; color: white; }
            .status-accepted { background: #4caf50; color: white; }
            .status-rejected { background: #f44336; color: white; }

            .auto-refresh {
                text-align: center;
                color: white;
                margin-top: 20px;
                font-size: 0.9em;
            }

            .component-bar {
                display: flex;
                align-items: center;
                margin: 10px 0;
            }

            .component-name {
                min-width: 150px;
                font-weight: 500;
            }

            .component-visual {
                flex: 1;
                background: #e0e0e0;
                height: 25px;
                border-radius: 5px;
                overflow: hidden;
                position: relative;
            }

            .component-fill {
                background: linear-gradient(90deg, #4caf50 0%, #2c5f2d 100%);
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                color: white;
                font-weight: bold;
                font-size: 0.8em;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .updating {
                animation: pulse 1s infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 GreenLang Migration Dashboard</h1>
                <div class="last-update">Last updated: <span id="last-update">-</span></div>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="metric">
                        <div class="metric-label">Overall IUM</div>
                        <div class="metric-value" id="overall-ium">-</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="ium-progress" style="width: 0%">0%</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="metric">
                        <div class="metric-label">Adoption Rate</div>
                        <div class="metric-value" id="adoption-rate">-</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="adoption-progress" style="width: 0%">0%</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="metric">
                        <div class="metric-label">Files Migrated</div>
                        <div class="metric-value" id="files-migrated">-</div>
                        <div id="files-total" style="color: #666; margin-top: 10px;">of - total</div>
                    </div>
                </div>

                <div class="card">
                    <div class="metric">
                        <div class="metric-label">GreenLang Imports</div>
                        <div class="metric-value" id="gl-imports">-</div>
                        <div id="other-imports" style="color: #666; margin-top: 10px;">vs - other</div>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>Component Usage</h2>
                    <div id="component-usage">
                        <p style="text-align: center; color: #666;">Loading...</p>
                    </div>
                </div>

                <div class="card">
                    <h2>Team Leaderboard</h2>
                    <ul class="leaderboard" id="leaderboard">
                        <li style="text-align: center; color: #666;">Loading...</li>
                    </ul>
                </div>
            </div>

            <div class="card">
                <h2>Architecture Decision Records</h2>
                <div class="adr-list" id="adr-list">
                    <p style="text-align: center; color: #666;">Loading...</p>
                </div>
            </div>

            <div class="auto-refresh">
                Auto-refreshing every 5 seconds
            </div>
        </div>

        <script>
            let isUpdating = false;

            async function updateDashboard() {
                if (isUpdating) return;
                isUpdating = true;

                try {
                    const response = await fetch('/api/data');
                    const data = await response.json();

                    // Update last update time
                    const lastUpdate = new Date(data.last_update);
                    document.getElementById('last-update').textContent = lastUpdate.toLocaleString();

                    // Update migration progress
                    const progress = data.migration_progress;
                    if (progress) {
                        const ium = Math.round(progress.overall_ium);
                        const adoption = Math.round(progress.adoption_rate);

                        document.getElementById('overall-ium').textContent = ium + '%';
                        document.getElementById('ium-progress').style.width = ium + '%';
                        document.getElementById('ium-progress').textContent = ium + '%';

                        document.getElementById('adoption-rate').textContent = adoption + '%';
                        document.getElementById('adoption-progress').style.width = adoption + '%';
                        document.getElementById('adoption-progress').textContent = adoption + '%';

                        document.getElementById('files-migrated').textContent = progress.files_migrated;
                        document.getElementById('files-total').textContent = `of ${progress.total_files} total`;

                        document.getElementById('gl-imports').textContent = progress.greenlang_imports;
                        document.getElementById('other-imports').textContent = `vs ${progress.remaining_imports} other`;
                    }

                    // Update component usage
                    const usage = data.usage;
                    if (usage && usage.by_component) {
                        const components = Object.entries(usage.by_component)
                            .sort((a, b) => b[1] - a[1]);

                        const maxCount = components[0] ? components[0][1] : 1;

                        const html = components.map(([name, count]) => {
                            const percentage = (count / maxCount * 100);
                            return `
                                <div class="component-bar">
                                    <div class="component-name">${name}</div>
                                    <div class="component-visual">
                                        <div class="component-fill" style="width: ${percentage}%">${count}</div>
                                    </div>
                                </div>
                            `;
                        }).join('');

                        document.getElementById('component-usage').innerHTML = html || '<p style="text-align: center; color: #666;">No data</p>';
                    }

                    // Update leaderboard
                    const teamStats = data.team_stats;
                    if (teamStats && teamStats.top_contributors) {
                        const html = teamStats.top_contributors.map((member, index) => `
                            <li>
                                <span class="rank">#${index + 1}</span>
                                <span class="name">${member.name}</span>
                                <span class="stats">
                                    IUM: ${member.ium}% | Files: ${member.files}
                                </span>
                            </li>
                        `).join('');

                        document.getElementById('leaderboard').innerHTML = html || '<li style="text-align: center; color: #666;">No data</li>';
                    }

                    // Update ADRs
                    const adrs = data.adrs;
                    if (adrs && adrs.length > 0) {
                        const html = adrs.map(adr => `
                            <div class="adr-item">
                                ${adr.file}
                            </div>
                        `).join('');

                        document.getElementById('adr-list').innerHTML = html;
                    } else {
                        document.getElementById('adr-list').innerHTML = '<p style="text-align: center; color: #666;">No ADRs found</p>';
                    }

                } catch (error) {
                    console.error('Error updating dashboard:', error);
                } finally {
                    isUpdating = false;
                }
            }

            // Initial update
            updateDashboard();

            // Auto-refresh every 5 seconds
            setInterval(updateDashboard, 5000);
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        """Serve dashboard HTML."""
        return render_template_string(DASHBOARD_HTML)

    @app.route('/api/data')
    def api_data():
        """Return dashboard data as JSON."""
        return jsonify(dashboard_data.get_data())

    def run_flask_server(app_directory: str, port: int = 8080):
        """Run Flask-based dashboard server."""
        global dashboard_data
        dashboard_data = DashboardData(app_directory)

        # Initial data collection
        print("Collecting initial data...")
        dashboard_data.update_data()

        # Start background thread for periodic updates
        def update_loop():
            while True:
                time.sleep(30)  # Update every 30 seconds
                print("Updating dashboard data...")
                dashboard_data.update_data()

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

        # Start Flask server
        print(f"\n{'=' * 80}")
        print(f"GreenLang Migration Dashboard")
        print(f"{'=' * 80}")
        print(f"\nServer running at: http://localhost:{port}")
        print(f"Monitoring directory: {app_directory}")
        print(f"\nPress Ctrl+C to stop\n")

        app.run(host='0.0.0.0', port=port, debug=False)

else:
    # Fallback to simple HTTP server
    def run_simple_server(app_directory: str, port: int = 8080):
        """Run simple HTTP server with static HTML."""
        logger.warning(f"Flask not available. Using simple HTTP server.")
        print("Install Flask for full dashboard functionality: pip install flask")
        print(f"\nServer running at: http://localhost:{port}")

        # Create simple HTML file
        html_path = os.path.join(os.getcwd(), 'dashboard.html')
        with open(html_path, 'w') as f:
            f.write(DASHBOARD_HTML if 'DASHBOARD_HTML' in globals() else "<h1>Dashboard</h1>")

        os.chdir(os.path.dirname(html_path))

        server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Migration Dashboard - Real-time migration progress dashboard"
    )

    parser.add_argument(
        '--directory',
        default='.',
        help='Directory to monitor (default: current directory)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to run server on (default: 8080)'
    )

    args = parser.parse_args()

    app_directory = os.path.abspath(args.directory)

    if not os.path.isdir(app_directory):
        logger.error(f"{app_directory} is not a valid directory")
        sys.exit(1)

    try:
        if FLASK_AVAILABLE:
            run_flask_server(app_directory, args.port)
        else:
            run_simple_server(app_directory, args.port)
    except KeyboardInterrupt:
        print("\n\nShutting down dashboard server...")
        sys.exit(0)


if __name__ == '__main__':
    main()
