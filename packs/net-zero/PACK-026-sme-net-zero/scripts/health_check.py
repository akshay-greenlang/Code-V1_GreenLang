#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-026: Health Check Script
==============================

Comprehensive health check for PACK-026 SME Net Zero Pack.
Verifies all components are functional before production deployment.

Usage:
    python scripts/health_check.py [--api-url http://localhost:8000]

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add pack to path
PACK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK_DIR))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checker for PACK-026."""

    def __init__(self, api_url: str):
        """Initialize health checker."""
        self.api_url = api_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warning = 0

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def check_health_endpoint(self) -> bool:
        """Check /health endpoint."""
        logger.info("\n1. Checking /health endpoint...")

        try:
            response = await self.client.get(f"{self.api_url}/health")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"  ✓ Status: {data.get('status')}")
                logger.info(f"  ✓ Pack: {data.get('pack_name')} v{data.get('version')}")
                logger.info(f"  ✓ Uptime: {data.get('uptime_seconds', 0):.1f}s")

                # Check component health
                components = data.get("components", {})
                if components:
                    logger.info("  Components:")
                    for comp, status in components.items():
                        if isinstance(status, dict):
                            comp_status = status.get("status", "unknown")
                            logger.info(f"    - {comp}: {comp_status}")
                        else:
                            logger.info(f"    - {comp}: {status}")

                self.checks_passed += 1
                return True
            else:
                logger.error(f"  ✗ Health endpoint returned {response.status_code}")
                self.checks_failed += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ Health check failed: {e}")
            self.checks_failed += 1
            return False

    async def check_readiness_endpoint(self) -> bool:
        """Check /ready endpoint."""
        logger.info("\n2. Checking /ready endpoint...")

        try:
            response = await self.client.get(f"{self.api_url}/ready")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"  ✓ Status: {data.get('status')}")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"  ✗ Readiness endpoint returned {response.status_code}")
                self.checks_failed += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ Readiness check failed: {e}")
            self.checks_failed += 1
            return False

    async def check_metrics_endpoint(self) -> bool:
        """Check /metrics endpoint."""
        logger.info("\n3. Checking /metrics endpoint...")

        try:
            response = await self.client.get(f"{self.api_url}/metrics")

            if response.status_code == 200:
                metrics = response.text
                # Check for expected metrics
                expected = [
                    "pack026_http_requests_total",
                    "pack026_http_request_duration_seconds",
                    "pack026_baseline_calculations_total",
                ]

                found = sum(1 for metric in expected if metric in metrics)
                logger.info(f"  ✓ Metrics endpoint responding")
                logger.info(f"  ✓ Found {found}/{len(expected)} expected metrics")

                if found == len(expected):
                    self.checks_passed += 1
                else:
                    logger.warning(f"  ⚠ Only {found}/{len(expected)} metrics found")
                    self.checks_warning += 1

                return True
            else:
                logger.error(f"  ✗ Metrics endpoint returned {response.status_code}")
                self.checks_failed += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ Metrics check failed: {e}")
            self.checks_failed += 1
            return False

    async def check_baseline_engine(self) -> bool:
        """Check baseline engine endpoint."""
        logger.info("\n4. Checking baseline engine...")

        payload = {
            "headcount": 25,
            "revenue_usd": 2500000,
            "sector": "information_technology",
            "reporting_year": 2025,
            "data_tier": "BRONZE",
        }

        try:
            response = await self.client.post(
                f"{self.api_url}/engines/baseline",
                json=payload,
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"  ✓ Baseline engine responding")
                logger.info(f"  ✓ Total emissions: {data.get('total_tco2e', 'N/A')} tCO2e")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"  ✗ Baseline engine returned {response.status_code}")
                logger.error(f"  Response: {response.text[:200]}")
                self.checks_failed += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ Baseline engine check failed: {e}")
            self.checks_failed += 1
            return False

    async def check_quick_wins_engine(self) -> bool:
        """Check quick wins engine endpoint."""
        logger.info("\n5. Checking quick wins engine...")

        payload = {
            "baseline_tco2e": 500,
            "sector": "information_technology",
            "annual_budget_usd": 10000,
        }

        try:
            response = await self.client.post(
                f"{self.api_url}/engines/quick-wins",
                json=payload,
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                actions_count = len(data.get("actions", []))
                logger.info(f"  ✓ Quick wins engine responding")
                logger.info(f"  ✓ Identified {actions_count} actions")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"  ✗ Quick wins engine returned {response.status_code}")
                logger.error(f"  Response: {response.text[:200]}")
                self.checks_failed += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ Quick wins engine check failed: {e}")
            self.checks_failed += 1
            return False

    async def check_templates(self) -> bool:
        """Check templates endpoint."""
        logger.info("\n6. Checking templates...")

        try:
            response = await self.client.get(f"{self.api_url}/templates")

            if response.status_code == 200:
                data = response.json()
                template_count = data.get("count", 0)
                logger.info(f"  ✓ Templates endpoint responding")
                logger.info(f"  ✓ Available templates: {template_count}")

                if template_count >= 8:
                    self.checks_passed += 1
                else:
                    logger.warning(f"  ⚠ Only {template_count} templates (expected 8)")
                    self.checks_warning += 1

                return True
            else:
                logger.error(f"  ✗ Templates endpoint returned {response.status_code}")
                self.checks_failed += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ Templates check failed: {e}")
            self.checks_failed += 1
            return False

    async def check_api_docs(self) -> bool:
        """Check API documentation endpoints."""
        logger.info("\n7. Checking API documentation...")

        try:
            # Check Swagger UI
            response = await self.client.get(f"{self.api_url}/docs")
            docs_ok = response.status_code == 200

            # Check ReDoc
            response = await self.client.get(f"{self.api_url}/redoc")
            redoc_ok = response.status_code == 200

            if docs_ok and redoc_ok:
                logger.info(f"  ✓ Swagger UI: available at /docs")
                logger.info(f"  ✓ ReDoc: available at /redoc")
                self.checks_passed += 1
                return True
            else:
                logger.warning(f"  ⚠ API docs partially available")
                logger.warning(f"    - Swagger UI (/docs): {'OK' if docs_ok else 'FAILED'}")
                logger.warning(f"    - ReDoc (/redoc): {'OK' if redoc_ok else 'FAILED'}")
                self.checks_warning += 1
                return False

        except Exception as e:
            logger.error(f"  ✗ API docs check failed: {e}")
            self.checks_failed += 1
            return False

    async def run_all_checks(self) -> bool:
        """Run all health checks."""
        logger.info("=" * 70)
        logger.info(f"PACK-026 Health Check - {datetime.now().isoformat()}")
        logger.info(f"API URL: {self.api_url}")
        logger.info("=" * 70)

        checks = [
            self.check_health_endpoint(),
            self.check_readiness_endpoint(),
            self.check_metrics_endpoint(),
            self.check_baseline_engine(),
            self.check_quick_wins_engine(),
            self.check_templates(),
            self.check_api_docs(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 70)

        total_checks = len(checks)
        logger.info(f"\n  Total checks:    {total_checks}")
        logger.info(f"  ✓ Passed:        {self.checks_passed}")
        logger.info(f"  ⚠ Warnings:      {self.checks_warning}")
        logger.info(f"  ✗ Failed:        {self.checks_failed}")

        success_rate = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0

        logger.info(f"\n  Success rate:    {success_rate:.1f}%")

        if self.checks_failed == 0:
            logger.info("\n✅ All critical checks passed!")
            logger.info("PACK-026 is healthy and ready for production.")
            return True
        else:
            logger.error("\n❌ Some checks failed!")
            logger.error("Please review errors above before deploying to production.")
            return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PACK-026 Health Check")
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API base URL",
    )
    args = parser.parse_args()

    checker = HealthChecker(args.api_url)

    try:
        success = await checker.run_all_checks()
        exit_code = 0 if success else 1
    finally:
        await checker.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
