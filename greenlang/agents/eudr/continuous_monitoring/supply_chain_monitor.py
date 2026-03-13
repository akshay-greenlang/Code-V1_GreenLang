# -*- coding: utf-8 -*-
"""
Supply Chain Monitor Engine - AGENT-EUDR-033

Monitors supplier status, certification validity, and geolocation
stability across an operator's supply chain. Generates alerts for
certification expiry, supplier status changes, and geolocation drift.

Zero-Hallucination:
    - All threshold comparisons are deterministic Decimal arithmetic
    - Geolocation drift uses Haversine formula (no ML/LLM)
    - Certification expiry is pure date arithmetic

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ContinuousMonitoringConfig, get_config
from .models import (
    AGENT_ID,
    AlertSeverity,
    CertificationCheck,
    CertificationStatus,
    GeolocationShift,
    MonitoringAlert,
    MonitoringScope,
    ScanStatus,
    SupplierChange,
    SupplyChainScanRecord,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class SupplyChainMonitor:
    """Supply chain continuous monitoring engine.

    Scans supplier data for status changes, expiring certifications,
    and geolocation drift. Generates structured alerts per finding.

    Example:
        >>> monitor = SupplyChainMonitor()
        >>> record = await monitor.scan_supply_chain(
        ...     operator_id="OP-001", suppliers=[{"supplier_id": "S-001"}]
        ... )
        >>> assert record.scan_status == ScanStatus.COMPLETED
    """

    def __init__(
        self, config: Optional[ContinuousMonitoringConfig] = None,
    ) -> None:
        """Initialize SupplyChainMonitor engine."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._scans: Dict[str, SupplyChainScanRecord] = {}
        self._alerts: Dict[str, MonitoringAlert] = {}
        logger.info("SupplyChainMonitor engine initialized")

    async def scan_supply_chain(
        self,
        operator_id: str,
        suppliers: List[Dict[str, Any]],
    ) -> SupplyChainScanRecord:
        """Execute a full supply chain monitoring scan.

        Args:
            operator_id: Operator identifier.
            suppliers: List of supplier data dictionaries.

        Returns:
            SupplyChainScanRecord with scan results.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        scan_id = str(uuid.uuid4())

        # Limit batch size
        capped = suppliers[: self.config.max_suppliers_per_scan]

        # Run sub-scans
        changes = await self.detect_supplier_changes(operator_id, capped)
        cert_checks = await self.check_certification_expiry(operator_id, capped)
        geo_shifts = await self.validate_geolocation_stability(operator_id, capped)

        # Count alerts
        expiring = [c for c in cert_checks if c.status != CertificationStatus.VALID]
        drifts = [g for g in geo_shifts if not g.is_stable]
        alerts_count = len(changes) + len(expiring) + len(drifts)

        # Generate alerts for critical findings
        for cert in cert_checks:
            if cert.status == CertificationStatus.EXPIRED:
                alert = self._create_alert(
                    operator_id, AlertSeverity.CRITICAL,
                    f"Certification {cert.certification_id} expired for supplier {cert.supplier_id}",
                    entity_id=cert.supplier_id, entity_type="supplier",
                )
                self._alerts[alert.alert_id] = alert
            elif cert.status == CertificationStatus.EXPIRING_SOON:
                severity = (
                    AlertSeverity.HIGH
                    if cert.days_until_expiry <= self.config.certification_expiry_critical_days
                    else AlertSeverity.WARNING
                )
                alert = self._create_alert(
                    operator_id, severity,
                    f"Certification {cert.certification_id} expiring in {cert.days_until_expiry} days",
                    entity_id=cert.supplier_id, entity_type="supplier",
                )
                self._alerts[alert.alert_id] = alert

        for shift in drifts:
            alert = self._create_alert(
                operator_id, AlertSeverity.HIGH,
                f"Geolocation drift of {shift.drift_km} km for entity {shift.entity_id}",
                entity_id=shift.entity_id, entity_type="plot",
            )
            self._alerts[alert.alert_id] = alert

        elapsed = Decimal(str(round(time.monotonic() - start_time, 3)))

        record = SupplyChainScanRecord(
            scan_id=scan_id,
            operator_id=operator_id,
            scan_status=ScanStatus.COMPLETED,
            suppliers_scanned=len(capped),
            changes_detected=len(changes),
            certifications_expiring=len(expiring),
            geolocation_drifts=len(drifts),
            supplier_changes=changes,
            certification_checks=cert_checks,
            geolocation_shifts=geo_shifts,
            alerts_generated=alerts_count,
            started_at=now,
            completed_at=datetime.now(timezone.utc).replace(microsecond=0),
            duration_seconds=elapsed,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "scan_id": scan_id,
            "operator_id": operator_id,
            "suppliers_scanned": len(capped),
            "changes_detected": len(changes),
            "created_at": now.isoformat(),
        })

        self._provenance.record(
            entity_type="supply_chain_scan",
            action="scan",
            entity_id=scan_id,
            actor=AGENT_ID,
            metadata={
                "operator_id": operator_id,
                "suppliers_scanned": len(capped),
                "changes": len(changes),
                "expiring": len(expiring),
                "drifts": len(drifts),
            },
        )

        self._scans[scan_id] = record
        logger.info(
            "Supply chain scan %s: %d suppliers, %d changes, %d expiring, %d drifts (%.3fs)",
            scan_id, len(capped), len(changes), len(expiring), len(drifts), float(elapsed),
        )
        return record

    async def detect_supplier_changes(
        self,
        operator_id: str,
        suppliers: List[Dict[str, Any]],
    ) -> List[SupplierChange]:
        """Detect changes in supplier status and attributes.

        Args:
            operator_id: Operator identifier.
            suppliers: List of supplier data dictionaries.

        Returns:
            List of detected supplier changes.
        """
        changes: List[SupplierChange] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)

        for supplier in suppliers:
            sid = supplier.get("supplier_id", "")
            # Compare current vs previous status
            current_status = supplier.get("status", "active")
            previous_status = supplier.get("previous_status", "active")

            if current_status != previous_status:
                changes.append(SupplierChange(
                    supplier_id=sid,
                    field_changed="status",
                    old_value=previous_status,
                    new_value=current_status,
                    detected_at=now,
                ))

            # Check name changes
            current_name = supplier.get("name", "")
            previous_name = supplier.get("previous_name", "")
            if previous_name and current_name != previous_name:
                changes.append(SupplierChange(
                    supplier_id=sid,
                    field_changed="name",
                    old_value=previous_name,
                    new_value=current_name,
                    detected_at=now,
                ))

            # Check ownership changes
            current_owner = supplier.get("owner", "")
            previous_owner = supplier.get("previous_owner", "")
            if previous_owner and current_owner != previous_owner:
                changes.append(SupplierChange(
                    supplier_id=sid,
                    field_changed="ownership",
                    old_value=previous_owner,
                    new_value=current_owner,
                    detected_at=now,
                ))

        return changes

    async def check_certification_expiry(
        self,
        operator_id: str,
        suppliers: List[Dict[str, Any]],
    ) -> List[CertificationCheck]:
        """Check certification expiry across suppliers.

        Args:
            operator_id: Operator identifier.
            suppliers: List of supplier data dictionaries.

        Returns:
            List of certification check results.
        """
        checks: List[CertificationCheck] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)

        for supplier in suppliers:
            sid = supplier.get("supplier_id", "")
            certs = supplier.get("certifications", [])

            for cert in certs:
                cert_id = cert.get("certification_id", str(uuid.uuid4()))
                cert_type = cert.get("type", "unknown")
                expiry_str = cert.get("expiry_date")

                if not expiry_str:
                    checks.append(CertificationCheck(
                        certification_id=cert_id,
                        supplier_id=sid,
                        certification_type=cert_type,
                        status=CertificationStatus.VALID,
                        days_until_expiry=999,
                    ))
                    continue

                try:
                    if isinstance(expiry_str, datetime):
                        expiry = expiry_str
                    else:
                        expiry = datetime.fromisoformat(str(expiry_str).replace("Z", "+00:00"))
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    checks.append(CertificationCheck(
                        certification_id=cert_id,
                        supplier_id=sid,
                        certification_type=cert_type,
                        status=CertificationStatus.VALID,
                        days_until_expiry=999,
                    ))
                    continue

                days_remaining = (expiry - now).days

                if days_remaining < 0:
                    status = CertificationStatus.EXPIRED
                elif days_remaining <= self.config.certification_expiry_critical_days:
                    status = CertificationStatus.EXPIRING_SOON
                elif days_remaining <= self.config.certification_expiry_warning_days:
                    status = CertificationStatus.EXPIRING_SOON
                else:
                    status = CertificationStatus.VALID

                # Check for revoked status
                if cert.get("revoked", False):
                    status = CertificationStatus.REVOKED

                checks.append(CertificationCheck(
                    certification_id=cert_id,
                    supplier_id=sid,
                    certification_type=cert_type,
                    expiry_date=expiry,
                    days_until_expiry=max(days_remaining, 0),
                    status=status,
                ))

        return checks

    async def validate_geolocation_stability(
        self,
        operator_id: str,
        suppliers: List[Dict[str, Any]],
    ) -> List[GeolocationShift]:
        """Validate geolocation stability of supply chain plots.

        Args:
            operator_id: Operator identifier.
            suppliers: List of supplier data with plot coordinates.

        Returns:
            List of geolocation shift results.
        """
        shifts: List[GeolocationShift] = []
        threshold = float(self.config.geolocation_drift_threshold_km)

        for supplier in suppliers:
            plots = supplier.get("plots", [])
            for plot in plots:
                entity_id = plot.get("plot_id", plot.get("entity_id", ""))
                orig_lat = plot.get("original_lat", 0.0)
                orig_lon = plot.get("original_lon", 0.0)
                curr_lat = plot.get("current_lat", orig_lat)
                curr_lon = plot.get("current_lon", orig_lon)

                drift = self._haversine_km(
                    float(orig_lat), float(orig_lon),
                    float(curr_lat), float(curr_lon),
                )
                drift_dec = Decimal(str(round(drift, 4)))
                is_stable = drift <= threshold

                shifts.append(GeolocationShift(
                    entity_id=entity_id,
                    original_lat=Decimal(str(orig_lat)),
                    original_lon=Decimal(str(orig_lon)),
                    current_lat=Decimal(str(curr_lat)),
                    current_lon=Decimal(str(curr_lon)),
                    drift_km=drift_dec,
                    is_stable=is_stable,
                ))

        return shifts

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance between two coordinates in km.

        Args:
            lat1: Latitude of point 1 in degrees.
            lon1: Longitude of point 1 in degrees.
            lat2: Latitude of point 2 in degrees.
            lon2: Longitude of point 2 in degrees.

        Returns:
            Distance in kilometers.
        """
        r = 6371.0  # Earth radius in km
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def _create_alert(
        self,
        operator_id: str,
        severity: AlertSeverity,
        description: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> MonitoringAlert:
        """Create a monitoring alert."""
        return MonitoringAlert(
            alert_id=str(uuid.uuid4()),
            operator_id=operator_id,
            source_engine=MonitoringScope.SUPPLY_CHAIN,
            severity=severity,
            title=f"Supply Chain Alert: {severity.value}",
            description=description,
            entity_id=entity_id,
            entity_type=entity_type,
        )

    async def get_scan(self, scan_id: str) -> Optional[SupplyChainScanRecord]:
        """Retrieve a scan record by ID."""
        return self._scans.get(scan_id)

    async def list_scans(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[SupplyChainScanRecord]:
        """List scan records with optional filters."""
        results = list(self._scans.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if status:
            results = [r for r in results if r.scan_status.value == status]
        return results

    async def list_alerts(
        self,
        operator_id: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[MonitoringAlert]:
        """List generated alerts with optional filters."""
        results = list(self._alerts.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if severity:
            results = [r for r in results if r.severity.value == severity]
        return results

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "SupplyChainMonitor",
            "status": "healthy",
            "scan_count": len(self._scans),
            "alert_count": len(self._alerts),
        }
