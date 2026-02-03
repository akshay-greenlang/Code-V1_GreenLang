#!/usr/bin/env python3
"""
PostgreSQL/TimescaleDB Settings Calculator
==========================================
Calculates optimal PostgreSQL configuration based on system resources.

Usage:
    python calculate-settings.py --ram 32 --cpu 8 --storage ssd
    python calculate-settings.py --ram 64 --cpu 16 --storage nvme --workload oltp
    python calculate-settings.py --interactive
    python calculate-settings.py --preset production-large

Author: GreenLang DevOps Team
Last Updated: 2026-02-03
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any


class StorageType(Enum):
    """Storage device types with their performance characteristics."""
    HDD = "hdd"
    SSD = "ssd"
    NVME = "nvme"
    CLOUD_SSD = "cloud_ssd"  # AWS gp3, Azure Premium SSD
    CLOUD_NVME = "cloud_nvme"  # AWS io2, Azure Ultra


class WorkloadType(Enum):
    """Database workload profiles."""
    OLTP = "oltp"  # Online Transaction Processing
    OLAP = "olap"  # Online Analytical Processing
    MIXED = "mixed"  # Mixed workload
    DW = "dw"  # Data Warehouse
    WEB = "web"  # Web application
    TIMESERIES = "timeseries"  # TimescaleDB time-series


@dataclass
class SystemResources:
    """System resource specifications."""
    ram_gb: int
    cpu_cores: int
    storage_type: StorageType
    workload: WorkloadType = WorkloadType.MIXED
    max_connections: int = 200
    replication_enabled: bool = True
    timescaledb_enabled: bool = True


@dataclass
class PostgreSQLSettings:
    """Calculated PostgreSQL settings."""
    # Connection Settings
    max_connections: int = 200
    superuser_reserved_connections: int = 3

    # Memory Settings
    shared_buffers: str = "8GB"
    effective_cache_size: str = "24GB"
    work_mem: str = "256MB"
    maintenance_work_mem: str = "2GB"
    huge_pages: str = "try"
    temp_buffers: str = "64MB"

    # WAL Settings
    wal_level: str = "replica"
    wal_buffers: str = "64MB"
    checkpoint_timeout: str = "15min"
    checkpoint_completion_target: float = 0.9
    max_wal_size: str = "8GB"
    min_wal_size: str = "2GB"
    wal_compression: str = "on"

    # Query Planner
    random_page_cost: float = 1.1
    effective_io_concurrency: int = 200
    default_statistics_target: int = 100

    # Parallel Query
    max_parallel_workers_per_gather: int = 4
    max_parallel_workers: int = 8
    max_parallel_maintenance_workers: int = 4

    # Autovacuum
    autovacuum_max_workers: int = 4
    autovacuum_naptime: str = "30s"
    autovacuum_vacuum_threshold: int = 50
    autovacuum_analyze_threshold: int = 50
    autovacuum_vacuum_scale_factor: float = 0.02
    autovacuum_analyze_scale_factor: float = 0.01
    autovacuum_vacuum_cost_delay: str = "2ms"
    autovacuum_vacuum_cost_limit: int = 1000

    # Replication
    max_wal_senders: int = 10
    max_replication_slots: int = 10
    wal_keep_size: str = "4GB"

    # TimescaleDB
    timescaledb_max_background_workers: int = 8

    # Additional settings storage
    additional: Dict[str, Any] = field(default_factory=dict)


def format_size(size_bytes: int, unit: str = "auto") -> str:
    """Format bytes to human-readable size."""
    if unit == "auto":
        if size_bytes >= 1024**3:
            return f"{size_bytes // (1024**3)}GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes // (1024**2)}MB"
        elif size_bytes >= 1024:
            return f"{size_bytes // 1024}kB"
        return f"{size_bytes}B"
    elif unit == "GB":
        return f"{size_bytes // (1024**3)}GB"
    elif unit == "MB":
        return f"{size_bytes // (1024**2)}MB"
    elif unit == "kB":
        return f"{size_bytes // 1024}kB"
    return f"{size_bytes}B"


def parse_size(size_str: str) -> int:
    """Parse size string to bytes."""
    size_str = size_str.strip().upper()
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4,
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            num_str = size_str[:-len(suffix)].strip()
            return int(float(num_str) * multiplier)

    return int(size_str)


class PostgreSQLCalculator:
    """Calculator for PostgreSQL configuration settings."""

    # Storage type characteristics
    STORAGE_PROFILES = {
        StorageType.HDD: {
            'random_page_cost': 4.0,
            'effective_io_concurrency': 2,
            'wal_compression': 'on',
        },
        StorageType.SSD: {
            'random_page_cost': 1.1,
            'effective_io_concurrency': 200,
            'wal_compression': 'on',
        },
        StorageType.NVME: {
            'random_page_cost': 1.0,
            'effective_io_concurrency': 256,
            'wal_compression': 'lz4',
        },
        StorageType.CLOUD_SSD: {
            'random_page_cost': 1.1,
            'effective_io_concurrency': 200,
            'wal_compression': 'on',
        },
        StorageType.CLOUD_NVME: {
            'random_page_cost': 1.0,
            'effective_io_concurrency': 256,
            'wal_compression': 'lz4',
        },
    }

    # Workload type profiles
    WORKLOAD_PROFILES = {
        WorkloadType.OLTP: {
            'shared_buffers_pct': 0.25,
            'effective_cache_pct': 0.75,
            'work_mem_divisor': 4,
            'checkpoint_timeout': '10min',
            'default_statistics_target': 100,
            'autovacuum_vacuum_scale_factor': 0.02,
        },
        WorkloadType.OLAP: {
            'shared_buffers_pct': 0.25,
            'effective_cache_pct': 0.75,
            'work_mem_divisor': 1,
            'checkpoint_timeout': '30min',
            'default_statistics_target': 500,
            'autovacuum_vacuum_scale_factor': 0.1,
        },
        WorkloadType.MIXED: {
            'shared_buffers_pct': 0.25,
            'effective_cache_pct': 0.75,
            'work_mem_divisor': 2,
            'checkpoint_timeout': '15min',
            'default_statistics_target': 100,
            'autovacuum_vacuum_scale_factor': 0.02,
        },
        WorkloadType.DW: {
            'shared_buffers_pct': 0.25,
            'effective_cache_pct': 0.75,
            'work_mem_divisor': 1,
            'checkpoint_timeout': '60min',
            'default_statistics_target': 1000,
            'autovacuum_vacuum_scale_factor': 0.2,
        },
        WorkloadType.WEB: {
            'shared_buffers_pct': 0.25,
            'effective_cache_pct': 0.75,
            'work_mem_divisor': 8,
            'checkpoint_timeout': '10min',
            'default_statistics_target': 100,
            'autovacuum_vacuum_scale_factor': 0.01,
        },
        WorkloadType.TIMESERIES: {
            'shared_buffers_pct': 0.25,
            'effective_cache_pct': 0.75,
            'work_mem_divisor': 2,
            'checkpoint_timeout': '15min',
            'default_statistics_target': 100,
            'autovacuum_vacuum_scale_factor': 0.05,
        },
    }

    def __init__(self, resources: SystemResources):
        self.resources = resources
        self.ram_bytes = resources.ram_gb * 1024**3

    def calculate(self) -> PostgreSQLSettings:
        """Calculate all PostgreSQL settings."""
        settings = PostgreSQLSettings()

        # Get profiles
        storage_profile = self.STORAGE_PROFILES[self.resources.storage_type]
        workload_profile = self.WORKLOAD_PROFILES[self.resources.workload]

        # Connection Settings
        settings.max_connections = self.resources.max_connections
        settings.superuser_reserved_connections = min(3, self.resources.max_connections // 50)

        # Memory Settings
        settings.shared_buffers = self._calculate_shared_buffers(workload_profile)
        settings.effective_cache_size = self._calculate_effective_cache(workload_profile)
        settings.work_mem = self._calculate_work_mem(workload_profile)
        settings.maintenance_work_mem = self._calculate_maintenance_work_mem()
        settings.huge_pages = self._calculate_huge_pages()
        settings.temp_buffers = self._calculate_temp_buffers()

        # WAL Settings
        settings.wal_buffers = self._calculate_wal_buffers()
        settings.max_wal_size = self._calculate_max_wal_size()
        settings.min_wal_size = self._calculate_min_wal_size()
        settings.checkpoint_timeout = workload_profile['checkpoint_timeout']
        settings.checkpoint_completion_target = 0.9
        settings.wal_compression = storage_profile['wal_compression']

        if self.resources.replication_enabled:
            settings.wal_level = "replica"
        else:
            settings.wal_level = "minimal"

        # Query Planner
        settings.random_page_cost = storage_profile['random_page_cost']
        settings.effective_io_concurrency = storage_profile['effective_io_concurrency']
        settings.default_statistics_target = workload_profile['default_statistics_target']

        # Parallel Query
        settings.max_parallel_workers = min(self.resources.cpu_cores, 32)
        settings.max_parallel_workers_per_gather = min(
            self.resources.cpu_cores // 2,
            settings.max_parallel_workers // 2,
            4
        )
        settings.max_parallel_maintenance_workers = min(
            self.resources.cpu_cores // 2,
            4
        )

        # Autovacuum
        settings.autovacuum_max_workers = self._calculate_autovacuum_workers()
        settings.autovacuum_vacuum_scale_factor = workload_profile['autovacuum_vacuum_scale_factor']
        settings.autovacuum_analyze_scale_factor = settings.autovacuum_vacuum_scale_factor / 2
        settings.autovacuum_vacuum_cost_limit = self._calculate_autovacuum_cost_limit()

        # Replication
        if self.resources.replication_enabled:
            settings.max_wal_senders = 10
            settings.max_replication_slots = 10
            settings.wal_keep_size = self._calculate_wal_keep_size()
        else:
            settings.max_wal_senders = 0
            settings.max_replication_slots = 0
            settings.wal_keep_size = "0"

        # TimescaleDB
        if self.resources.timescaledb_enabled:
            settings.timescaledb_max_background_workers = min(
                self.resources.cpu_cores,
                16
            )

        return settings

    def _calculate_shared_buffers(self, profile: dict) -> str:
        """Calculate shared_buffers (25% of RAM, max 8GB for typical workloads)."""
        shared = int(self.ram_bytes * profile['shared_buffers_pct'])
        # Cap at 40% for very large RAM systems
        max_shared = min(int(self.ram_bytes * 0.4), 128 * 1024**3)
        shared = min(shared, max_shared)
        return format_size(shared)

    def _calculate_effective_cache(self, profile: dict) -> str:
        """Calculate effective_cache_size (75% of RAM)."""
        cache = int(self.ram_bytes * profile['effective_cache_pct'])
        return format_size(cache)

    def _calculate_work_mem(self, profile: dict) -> str:
        """Calculate work_mem based on available memory and connections."""
        # Available memory after shared_buffers
        shared_buffers = int(self.ram_bytes * profile['shared_buffers_pct'])
        available = self.ram_bytes - shared_buffers

        # Divide by connections and work_mem divisor
        # Each connection can use multiple work_mem allocations
        work_mem = available // (self.resources.max_connections * profile['work_mem_divisor'] * 2)

        # Apply reasonable bounds
        min_work_mem = 4 * 1024**2  # 4MB minimum
        max_work_mem = 2 * 1024**3  # 2GB maximum
        work_mem = max(min_work_mem, min(work_mem, max_work_mem))

        return format_size(work_mem)

    def _calculate_maintenance_work_mem(self) -> str:
        """Calculate maintenance_work_mem (5% of RAM, max 2GB)."""
        maint = min(
            int(self.ram_bytes * 0.05),
            2 * 1024**3  # 2GB max
        )
        maint = max(maint, 64 * 1024**2)  # 64MB minimum
        return format_size(maint)

    def _calculate_huge_pages(self) -> str:
        """Determine huge_pages setting."""
        if self.resources.ram_gb >= 16:
            return "try"
        return "off"

    def _calculate_temp_buffers(self) -> str:
        """Calculate temp_buffers."""
        temp = min(
            self.ram_bytes // 256,
            128 * 1024**2  # 128MB max
        )
        temp = max(temp, 8 * 1024**2)  # 8MB minimum
        return format_size(temp)

    def _calculate_wal_buffers(self) -> str:
        """Calculate wal_buffers (3% of shared_buffers, max 64MB)."""
        shared = parse_size(self._calculate_shared_buffers(
            self.WORKLOAD_PROFILES[self.resources.workload]
        ))
        wal_buf = min(int(shared * 0.03), 64 * 1024**2)
        wal_buf = max(wal_buf, 16 * 1024**2)  # 16MB minimum
        return format_size(wal_buf)

    def _calculate_max_wal_size(self) -> str:
        """Calculate max_wal_size."""
        # Scale with RAM, typical range 2GB-16GB
        max_wal = min(
            self.resources.ram_gb // 4 * 1024**3,
            16 * 1024**3
        )
        max_wal = max(max_wal, 2 * 1024**3)
        return format_size(max_wal)

    def _calculate_min_wal_size(self) -> str:
        """Calculate min_wal_size."""
        max_wal = parse_size(self._calculate_max_wal_size())
        min_wal = max_wal // 4
        return format_size(min_wal)

    def _calculate_wal_keep_size(self) -> str:
        """Calculate wal_keep_size for replication."""
        # Keep enough WAL for typical replica lag
        wal_keep = min(
            self.resources.ram_gb // 8 * 1024**3,
            8 * 1024**3
        )
        wal_keep = max(wal_keep, 1024**3)  # 1GB minimum
        return format_size(wal_keep)

    def _calculate_autovacuum_workers(self) -> int:
        """Calculate autovacuum_max_workers."""
        workers = max(3, min(self.resources.cpu_cores // 4, 8))
        return workers

    def _calculate_autovacuum_cost_limit(self) -> int:
        """Calculate autovacuum_vacuum_cost_limit."""
        if self.resources.storage_type in [StorageType.NVME, StorageType.CLOUD_NVME]:
            return 2000
        elif self.resources.storage_type in [StorageType.SSD, StorageType.CLOUD_SSD]:
            return 1000
        return 200


def generate_config_file(settings: PostgreSQLSettings, resources: SystemResources) -> str:
    """Generate PostgreSQL configuration file content."""
    config = f"""# =============================================================================
# PostgreSQL Auto-Generated Configuration
# =============================================================================
# Generated by: calculate-settings.py
# System: {resources.ram_gb}GB RAM, {resources.cpu_cores} CPU cores, {resources.storage_type.value} storage
# Workload: {resources.workload.value}
# =============================================================================

# -----------------------------------------------------------------------------
# CONNECTION SETTINGS
# -----------------------------------------------------------------------------
max_connections = {settings.max_connections}
superuser_reserved_connections = {settings.superuser_reserved_connections}

# -----------------------------------------------------------------------------
# MEMORY CONFIGURATION
# -----------------------------------------------------------------------------
shared_buffers = {settings.shared_buffers}
effective_cache_size = {settings.effective_cache_size}
work_mem = {settings.work_mem}
maintenance_work_mem = {settings.maintenance_work_mem}
huge_pages = {settings.huge_pages}
temp_buffers = {settings.temp_buffers}

# -----------------------------------------------------------------------------
# WAL CONFIGURATION
# -----------------------------------------------------------------------------
wal_level = {settings.wal_level}
wal_buffers = {settings.wal_buffers}
checkpoint_timeout = {settings.checkpoint_timeout}
checkpoint_completion_target = {settings.checkpoint_completion_target}
max_wal_size = {settings.max_wal_size}
min_wal_size = {settings.min_wal_size}
wal_compression = {settings.wal_compression}
archive_mode = on

# -----------------------------------------------------------------------------
# QUERY PLANNER
# -----------------------------------------------------------------------------
random_page_cost = {settings.random_page_cost}
effective_io_concurrency = {settings.effective_io_concurrency}
default_statistics_target = {settings.default_statistics_target}

# -----------------------------------------------------------------------------
# PARALLEL QUERY
# -----------------------------------------------------------------------------
max_parallel_workers_per_gather = {settings.max_parallel_workers_per_gather}
max_parallel_workers = {settings.max_parallel_workers}
max_parallel_maintenance_workers = {settings.max_parallel_maintenance_workers}
parallel_leader_participation = on

# -----------------------------------------------------------------------------
# AUTOVACUUM
# -----------------------------------------------------------------------------
autovacuum = on
autovacuum_max_workers = {settings.autovacuum_max_workers}
autovacuum_naptime = {settings.autovacuum_naptime}
autovacuum_vacuum_threshold = {settings.autovacuum_vacuum_threshold}
autovacuum_analyze_threshold = {settings.autovacuum_analyze_threshold}
autovacuum_vacuum_scale_factor = {settings.autovacuum_vacuum_scale_factor}
autovacuum_analyze_scale_factor = {settings.autovacuum_analyze_scale_factor}
autovacuum_vacuum_cost_delay = {settings.autovacuum_vacuum_cost_delay}
autovacuum_vacuum_cost_limit = {settings.autovacuum_vacuum_cost_limit}
"""

    if resources.replication_enabled:
        config += f"""
# -----------------------------------------------------------------------------
# REPLICATION
# -----------------------------------------------------------------------------
max_wal_senders = {settings.max_wal_senders}
max_replication_slots = {settings.max_replication_slots}
wal_keep_size = {settings.wal_keep_size}
hot_standby = on
hot_standby_feedback = on
synchronous_commit = on
"""

    if resources.timescaledb_enabled:
        config += f"""
# -----------------------------------------------------------------------------
# TIMESCALEDB
# -----------------------------------------------------------------------------
shared_preload_libraries = 'timescaledb,pg_stat_statements'
timescaledb.max_background_workers = {settings.timescaledb_max_background_workers}
timescaledb.telemetry_level = off
"""

    return config


def generate_json_output(settings: PostgreSQLSettings, resources: SystemResources) -> dict:
    """Generate JSON output of settings."""
    return {
        "system": {
            "ram_gb": resources.ram_gb,
            "cpu_cores": resources.cpu_cores,
            "storage_type": resources.storage_type.value,
            "workload": resources.workload.value,
        },
        "settings": {
            "connection": {
                "max_connections": settings.max_connections,
                "superuser_reserved_connections": settings.superuser_reserved_connections,
            },
            "memory": {
                "shared_buffers": settings.shared_buffers,
                "effective_cache_size": settings.effective_cache_size,
                "work_mem": settings.work_mem,
                "maintenance_work_mem": settings.maintenance_work_mem,
                "huge_pages": settings.huge_pages,
                "temp_buffers": settings.temp_buffers,
            },
            "wal": {
                "wal_level": settings.wal_level,
                "wal_buffers": settings.wal_buffers,
                "checkpoint_timeout": settings.checkpoint_timeout,
                "checkpoint_completion_target": settings.checkpoint_completion_target,
                "max_wal_size": settings.max_wal_size,
                "min_wal_size": settings.min_wal_size,
                "wal_compression": settings.wal_compression,
            },
            "query_planner": {
                "random_page_cost": settings.random_page_cost,
                "effective_io_concurrency": settings.effective_io_concurrency,
                "default_statistics_target": settings.default_statistics_target,
            },
            "parallel_query": {
                "max_parallel_workers_per_gather": settings.max_parallel_workers_per_gather,
                "max_parallel_workers": settings.max_parallel_workers,
                "max_parallel_maintenance_workers": settings.max_parallel_maintenance_workers,
            },
            "autovacuum": {
                "autovacuum_max_workers": settings.autovacuum_max_workers,
                "autovacuum_naptime": settings.autovacuum_naptime,
                "autovacuum_vacuum_scale_factor": settings.autovacuum_vacuum_scale_factor,
                "autovacuum_analyze_scale_factor": settings.autovacuum_analyze_scale_factor,
                "autovacuum_vacuum_cost_limit": settings.autovacuum_vacuum_cost_limit,
            },
            "replication": {
                "max_wal_senders": settings.max_wal_senders,
                "max_replication_slots": settings.max_replication_slots,
                "wal_keep_size": settings.wal_keep_size,
            },
            "timescaledb": {
                "max_background_workers": settings.timescaledb_max_background_workers,
            },
        }
    }


def get_preset(preset_name: str) -> SystemResources:
    """Get predefined system presets."""
    presets = {
        "development": SystemResources(
            ram_gb=4, cpu_cores=2, storage_type=StorageType.SSD,
            workload=WorkloadType.MIXED, max_connections=50,
            replication_enabled=False, timescaledb_enabled=True
        ),
        "staging": SystemResources(
            ram_gb=16, cpu_cores=4, storage_type=StorageType.SSD,
            workload=WorkloadType.MIXED, max_connections=100,
            replication_enabled=True, timescaledb_enabled=True
        ),
        "production-small": SystemResources(
            ram_gb=32, cpu_cores=8, storage_type=StorageType.SSD,
            workload=WorkloadType.MIXED, max_connections=200,
            replication_enabled=True, timescaledb_enabled=True
        ),
        "production-medium": SystemResources(
            ram_gb=64, cpu_cores=16, storage_type=StorageType.NVME,
            workload=WorkloadType.MIXED, max_connections=400,
            replication_enabled=True, timescaledb_enabled=True
        ),
        "production-large": SystemResources(
            ram_gb=128, cpu_cores=32, storage_type=StorageType.NVME,
            workload=WorkloadType.MIXED, max_connections=500,
            replication_enabled=True, timescaledb_enabled=True
        ),
        "timeseries-small": SystemResources(
            ram_gb=32, cpu_cores=8, storage_type=StorageType.NVME,
            workload=WorkloadType.TIMESERIES, max_connections=100,
            replication_enabled=True, timescaledb_enabled=True
        ),
        "timeseries-large": SystemResources(
            ram_gb=128, cpu_cores=32, storage_type=StorageType.NVME,
            workload=WorkloadType.TIMESERIES, max_connections=200,
            replication_enabled=True, timescaledb_enabled=True
        ),
        "analytics": SystemResources(
            ram_gb=256, cpu_cores=64, storage_type=StorageType.NVME,
            workload=WorkloadType.OLAP, max_connections=100,
            replication_enabled=True, timescaledb_enabled=True
        ),
    }

    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return presets[preset_name]


def interactive_mode() -> SystemResources:
    """Interactive mode for collecting system specifications."""
    print("\n" + "=" * 60)
    print("PostgreSQL/TimescaleDB Settings Calculator - Interactive Mode")
    print("=" * 60 + "\n")

    # RAM
    while True:
        try:
            ram = int(input("Enter total RAM in GB (e.g., 32): "))
            if ram < 1:
                raise ValueError("RAM must be at least 1 GB")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # CPU
    while True:
        try:
            cpu = int(input("Enter number of CPU cores (e.g., 8): "))
            if cpu < 1:
                raise ValueError("CPU cores must be at least 1")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Storage type
    print("\nStorage types:")
    for i, st in enumerate(StorageType, 1):
        print(f"  {i}. {st.value}")
    while True:
        try:
            st_choice = int(input("Select storage type (1-5): "))
            storage = list(StorageType)[st_choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid selection")

    # Workload type
    print("\nWorkload types:")
    for i, wl in enumerate(WorkloadType, 1):
        print(f"  {i}. {wl.value}")
    while True:
        try:
            wl_choice = int(input("Select workload type (1-6): "))
            workload = list(WorkloadType)[wl_choice - 1]
            break
        except (ValueError, IndexError):
            print("Invalid selection")

    # Max connections
    while True:
        try:
            max_conn = int(input("Enter max connections (default 200): ") or "200")
            if max_conn < 10:
                raise ValueError("Max connections must be at least 10")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Replication
    repl = input("Enable replication? (y/n, default y): ").lower()
    replication = repl != 'n'

    # TimescaleDB
    tsdb = input("Enable TimescaleDB? (y/n, default y): ").lower()
    timescaledb = tsdb != 'n'

    return SystemResources(
        ram_gb=ram,
        cpu_cores=cpu,
        storage_type=storage,
        workload=workload,
        max_connections=max_conn,
        replication_enabled=replication,
        timescaledb_enabled=timescaledb
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PostgreSQL/TimescaleDB Settings Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ram 32 --cpu 8 --storage ssd
  %(prog)s --ram 64 --cpu 16 --storage nvme --workload olap
  %(prog)s --preset production-large
  %(prog)s --interactive
  %(prog)s --ram 32 --cpu 8 --storage ssd --output json
        """
    )

    parser.add_argument("--ram", type=int, help="Total RAM in GB")
    parser.add_argument("--cpu", type=int, help="Number of CPU cores")
    parser.add_argument(
        "--storage",
        choices=[s.value for s in StorageType],
        help="Storage type"
    )
    parser.add_argument(
        "--workload",
        choices=[w.value for w in WorkloadType],
        default="mixed",
        help="Workload type (default: mixed)"
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=200,
        help="Maximum connections (default: 200)"
    )
    parser.add_argument(
        "--no-replication",
        action="store_true",
        help="Disable replication settings"
    )
    parser.add_argument(
        "--no-timescaledb",
        action="store_true",
        help="Disable TimescaleDB settings"
    )
    parser.add_argument(
        "--preset",
        help="Use a predefined preset (development, staging, production-small, production-medium, production-large, timeseries-small, timeseries-large, analytics)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["config", "json", "both"],
        default="config",
        help="Output format (default: config)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Output file path (default: stdout)"
    )

    args = parser.parse_args()

    # Determine resources
    if args.interactive:
        resources = interactive_mode()
    elif args.preset:
        resources = get_preset(args.preset)
    elif args.ram and args.cpu and args.storage:
        resources = SystemResources(
            ram_gb=args.ram,
            cpu_cores=args.cpu,
            storage_type=StorageType(args.storage),
            workload=WorkloadType(args.workload),
            max_connections=args.max_connections,
            replication_enabled=not args.no_replication,
            timescaledb_enabled=not args.no_timescaledb
        )
    else:
        parser.error("Must specify --ram, --cpu, and --storage, or use --preset or --interactive")

    # Calculate settings
    calculator = PostgreSQLCalculator(resources)
    settings = calculator.calculate()

    # Generate output
    output_parts = []

    if args.output in ["config", "both"]:
        output_parts.append(generate_config_file(settings, resources))

    if args.output in ["json", "both"]:
        json_output = generate_json_output(settings, resources)
        output_parts.append(json.dumps(json_output, indent=2))

    output = "\n".join(output_parts)

    # Write output
    if args.file:
        with open(args.file, 'w') as f:
            f.write(output)
        print(f"Configuration written to: {args.file}")
    else:
        print(output)

    # Print summary
    if args.output != "json":
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"System: {resources.ram_gb}GB RAM, {resources.cpu_cores} CPU cores")
        print(f"Storage: {resources.storage_type.value}")
        print(f"Workload: {resources.workload.value}")
        print(f"Max Connections: {resources.max_connections}")
        print(f"Replication: {'Enabled' if resources.replication_enabled else 'Disabled'}")
        print(f"TimescaleDB: {'Enabled' if resources.timescaledb_enabled else 'Disabled'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
