# -*- coding: utf-8 -*-
# GL-VCCI CLI Module
# Command-line interface for VCCI Scope 3 Platform

"""
VCCI CLI
========

Command-line interface for the VCCI Scope 3 Platform.

Available Commands:
------------------
1. vcci intake      - Ingest and validate value chain data
2. vcci calculate   - Calculate Scope 3 emissions
3. vcci analyze     - Perform hotspot analysis
4. vcci engage      - Manage supplier engagement campaigns
5. vcci report      - Generate compliance reports
6. vcci pipeline    - Run complete end-to-end pipeline
7. vcci status      - Check platform status
8. vcci config      - Manage configuration

Usage Examples:
--------------
```bash
# Ingest procurement data
vcci intake --file examples/procurement_sample.csv --format csv

# Calculate Scope 3 emissions
vcci calculate --data validated_data.json --categories all

# Analyze hotspots
vcci analyze --emissions scope3_results.json --pareto

# Generate GHG Protocol report
vcci report --emissions scope3_results.json --format ghg-protocol

# Run complete pipeline
vcci pipeline --input data/ --output results/
```

Implementation:
--------------
CLI is built using Typer (modern CLI framework) with Rich for beautiful output.
"""

__version__ = "1.0.0"

# CLI commands will be implemented in vcci_commands.py
__all__ = [
    # "main",  # To be implemented
]
