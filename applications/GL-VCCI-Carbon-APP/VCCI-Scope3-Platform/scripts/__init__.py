# -*- coding: utf-8 -*-
# GL-VCCI Scripts Module
# Utility scripts for setup, maintenance, and operations

"""
VCCI Scripts
============

Utility scripts for database setup, data migration, and maintenance.

Available Scripts:
-----------------
1. init_database.py
   - Initialize PostgreSQL database
   - Create tables, indexes, partitions
   - Set up TimescaleDB for time-series data

2. seed_emission_factors.py
   - Load emission factor databases
   - Import DEFRA, EPA, Ecoinvent data
   - Update emission factors

3. migrate_data.py
   - Data migration utilities
   - Import historical data
   - Data transformation

4. backup_database.py
   - Database backup scripts
   - S3 upload
   - Automated backup scheduling

5. health_check.py
   - System health checks
   - Database connectivity
   - API availability
   - ERP connector status

Usage:
------
```bash
# Initialize database
python scripts/init_database.py

# Seed emission factors
python scripts/seed_emission_factors.py --sources defra,epa

# Health check
python scripts/health_check.py
```
"""

__version__ = "1.0.0"
