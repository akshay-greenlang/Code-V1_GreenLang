"""
Audit logging for GreenLang multi-tenancy
"""

import json
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import hashlib
import uuid
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_CREATED = "auth.token.created"
    TOKEN_REVOKED = "auth.token.revoked"
    API_KEY_CREATED = "auth.apikey.created"
    API_KEY_REVOKED = "auth.apikey.revoked"
    
    # Authorization events
    PERMISSION_GRANTED = "authz.permission.granted"
    PERMISSION_DENIED = "authz.permission.denied"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REVOKED = "authz.role.revoked"
    
    # Resource events
    RESOURCE_CREATED = "resource.created"
    RESOURCE_READ = "resource.read"
    RESOURCE_UPDATED = "resource.updated"
    RESOURCE_DELETED = "resource.deleted"
    
    # Tenant events
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DELETED = "tenant.deleted"
    TENANT_QUOTA_EXCEEDED = "tenant.quota.exceeded"
    
    # Pipeline events
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    
    # Security events
    SECURITY_ALERT = "security.alert"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    ACCESS_VIOLATION = "security.violation"
    
    # System events
    CONFIG_CHANGED = "system.config.changed"
    SERVICE_STARTED = "system.service.started"
    SERVICE_STOPPED = "system.service.stopped"
    ERROR = "system.error"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event record"""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.RESOURCE_READ
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Context
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Event details
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = AuditEventType(data['event_type'])
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = AuditSeverity(data['severity'])
        return cls(**data)


class AuditTrail:
    """Audit trail for tracking events"""
    
    def __init__(self, tenant_id: str, max_events: int = 10000):
        """
        Initialize audit trail
        
        Args:
            tenant_id: Tenant ID
            max_events: Maximum events to keep in memory
        """
        self.tenant_id = tenant_id
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.last_event_time: Dict[str, datetime] = {}
    
    def add_event(self, event: AuditEvent):
        """Add event to trail"""
        event.tenant_id = self.tenant_id
        self.events.append(event)
        self.event_counts[event.event_type.value] += 1
        self.last_event_time[event.event_type.value] = event.timestamp
    
    def get_events(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_types: Optional[List[AuditEventType]] = None,
                   user_id: Optional[str] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """
        Get filtered events
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            event_types: Event type filter
            user_id: User ID filter
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        results = []
        
        for event in reversed(self.events):  # Most recent first
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if user_id and event.user_id != user_id:
                continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        return {
            "tenant_id": self.tenant_id,
            "total_events": len(self.events),
            "event_counts": dict(self.event_counts),
            "last_event_times": {
                k: v.isoformat() for k, v in self.last_event_time.items()
            }
        }


class AuditLogger:
    """Centralized audit logging system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize audit logger
        
        Args:
            storage_path: Path to store audit logs
        """
        self.storage_path = storage_path or Path("./audit_logs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.trails: Dict[str, AuditTrail] = {}
        self.handlers: List[Any] = []
        self.filters: List[Any] = []
        self.retention_days = 90
        self.async_queue: Optional[asyncio.Queue] = None
        
        # Alert configurations
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: List[Any] = []
        
        logger.info(f"AuditLogger initialized with storage at {self.storage_path}")
    
    def log(self, event: AuditEvent):
        """
        Log an audit event
        
        Args:
            event: Audit event to log
        """
        # Apply filters
        for filter_func in self.filters:
            if not filter_func(event):
                return
        
        # Add to tenant trail
        if event.tenant_id:
            if event.tenant_id not in self.trails:
                self.trails[event.tenant_id] = AuditTrail(event.tenant_id)
            self.trails[event.tenant_id].add_event(event)
        
        # Process handlers
        for handler in self.handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in audit handler: {e}")
        
        # Check alert rules
        self._check_alerts(event)
        
        # Write to storage
        self._write_to_storage(event)
        
        # Log to system logger
        log_level = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL
        }.get(event.severity, logging.INFO)
        
        logger.log(log_level, f"Audit: {event.event_type.value} - {event.to_json()}")
    
    def log_login(self, user_id: str, success: bool, 
                  ip_address: Optional[str] = None,
                  metadata: Optional[Dict] = None):
        """Log login attempt"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            action="login",
            result="success" if success else "failure",
            metadata=metadata or {}
        )
        self.log(event)
    
    def log_permission_check(self, user_id: str, resource: str, 
                            action: str, granted: bool,
                            tenant_id: Optional[str] = None):
        """Log permission check"""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_DENIED,
            severity=AuditSeverity.INFO if granted else AuditSeverity.WARNING,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource.split(":")[0] if ":" in resource else resource,
            resource_id=resource,
            action=action,
            result="granted" if granted else "denied"
        )
        self.log(event)
    
    def log_resource_access(self, user_id: str, resource_type: str,
                           resource_id: str, action: str,
                           tenant_id: Optional[str] = None):
        """Log resource access"""
        event_type_map = {
            "create": AuditEventType.RESOURCE_CREATED,
            "read": AuditEventType.RESOURCE_READ,
            "update": AuditEventType.RESOURCE_UPDATED,
            "delete": AuditEventType.RESOURCE_DELETED
        }
        
        event = AuditEvent(
            event_type=event_type_map.get(action, AuditEventType.RESOURCE_READ),
            severity=AuditSeverity.INFO,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action
        )
        self.log(event)
    
    def log_security_alert(self, alert_type: str, message: str,
                          user_id: Optional[str] = None,
                          tenant_id: Optional[str] = None,
                          metadata: Optional[Dict] = None):
        """Log security alert"""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.CRITICAL,
            tenant_id=tenant_id,
            user_id=user_id,
            action=alert_type,
            error_message=message,
            metadata=metadata or {}
        )
        self.log(event)
    
    def add_handler(self, handler):
        """Add custom event handler"""
        self.handlers.append(handler)
    
    def add_filter(self, filter_func):
        """Add event filter"""
        self.filters.append(filter_func)
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """
        Add alert rule
        
        Args:
            rule: Alert rule configuration
                - event_types: List of event types to match
                - threshold: Number of events
                - window: Time window in seconds
                - action: Alert action
        """
        self.alert_rules.append(rule)
    
    def query(self, tenant_id: str, **kwargs) -> List[AuditEvent]:
        """
        Query audit events
        
        Args:
            tenant_id: Tenant ID
            **kwargs: Query parameters
            
        Returns:
            List of matching events
        """
        trail = self.trails.get(tenant_id)
        if not trail:
            return []
        
        return trail.get_events(**kwargs)
    
    def get_report(self, tenant_id: str, 
                  start_time: datetime,
                  end_time: datetime) -> Dict[str, Any]:
        """
        Generate audit report
        
        Args:
            tenant_id: Tenant ID
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Audit report
        """
        events = self.query(
            tenant_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Analyze events
        report = {
            "tenant_id": tenant_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events": len(events),
            "events_by_type": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "events_by_user": defaultdict(int),
            "failed_logins": 0,
            "permission_denials": 0,
            "security_alerts": 0,
            "top_users": [],
            "top_resources": defaultdict(int)
        }
        
        for event in events:
            report["events_by_type"][event.event_type.value] += 1
            report["events_by_severity"][event.severity.value] += 1
            
            if event.user_id:
                report["events_by_user"][event.user_id] += 1
            
            if event.event_type == AuditEventType.LOGIN_FAILURE:
                report["failed_logins"] += 1
            elif event.event_type == AuditEventType.PERMISSION_DENIED:
                report["permission_denials"] += 1
            elif event.event_type == AuditEventType.SECURITY_ALERT:
                report["security_alerts"] += 1
            
            if event.resource_id:
                report["top_resources"][event.resource_id] += 1
        
        # Get top users
        report["top_users"] = sorted(
            report["events_by_user"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Convert defaultdicts to regular dicts
        report["events_by_type"] = dict(report["events_by_type"])
        report["events_by_severity"] = dict(report["events_by_severity"])
        report["events_by_user"] = dict(report["events_by_user"])
        report["top_resources"] = dict(sorted(
            report["top_resources"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return report
    
    def _write_to_storage(self, event: AuditEvent):
        """Write event to storage"""
        try:
            # Create daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.storage_path / f"audit_{date_str}.jsonl"
            
            with open(log_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event to storage: {e}")
    
    def _check_alerts(self, event: AuditEvent):
        """Check if event triggers any alerts"""
        for rule in self.alert_rules:
            if self._evaluate_alert_rule(event, rule):
                self._trigger_alert(event, rule)
    
    def _evaluate_alert_rule(self, event: AuditEvent, rule: Dict) -> bool:
        """Evaluate if event matches alert rule"""
        # Check event type
        if "event_types" in rule:
            if event.event_type not in rule["event_types"]:
                return False
        
        # Check threshold (would need to track event counts over time window)
        # This is simplified - in production would use time-series data
        if "threshold" in rule and "window" in rule:
            # Check if threshold exceeded in time window
            pass
        
        return True
    
    def _trigger_alert(self, event: AuditEvent, rule: Dict):
        """Trigger alert action"""
        for handler in self.alert_handlers:
            try:
                handler(event, rule)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def cleanup_old_logs(self):
        """Remove old audit logs based on retention policy"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        for log_file in self.storage_path.glob("audit_*.jsonl"):
            try:
                # Parse date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Removed old audit log: {log_file}")
            except Exception as e:
                logger.error(f"Error cleaning up audit log {log_file}: {e}")
    
    async def async_log(self, event: AuditEvent):
        """Asynchronously log event"""
        if not self.async_queue:
            self.async_queue = asyncio.Queue()
        
        await self.async_queue.put(event)
    
    async def process_async_queue(self):
        """Process async logging queue"""
        if not self.async_queue:
            return
        
        while True:
            try:
                event = await self.async_queue.get()
                self.log(event)
            except Exception as e:
                logger.error(f"Error processing async audit event: {e}")


class ComplianceReporter:
    """Generate compliance reports from audit data"""
    
    def __init__(self, audit_logger: AuditLogger):
        """
        Initialize compliance reporter
        
        Args:
            audit_logger: Audit logger instance
        """
        self.audit_logger = audit_logger
    
    def generate_sox_report(self, tenant_id: str,
                           start_time: datetime,
                           end_time: datetime) -> Dict[str, Any]:
        """Generate SOX compliance report"""
        events = self.audit_logger.query(
            tenant_id,
            start_time=start_time,
            end_time=end_time
        )
        
        report = {
            "report_type": "SOX Compliance",
            "tenant_id": tenant_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "privileged_access": [],
            "configuration_changes": [],
            "access_violations": [],
            "user_provisioning": []
        }
        
        for event in events:
            # Track privileged access
            if event.event_type == AuditEventType.PERMISSION_GRANTED:
                if event.metadata.get("privileged"):
                    report["privileged_access"].append(event.to_dict())
            
            # Track configuration changes
            elif event.event_type == AuditEventType.CONFIG_CHANGED:
                report["configuration_changes"].append(event.to_dict())
            
            # Track access violations
            elif event.event_type in [
                AuditEventType.ACCESS_VIOLATION,
                AuditEventType.PERMISSION_DENIED
            ]:
                report["access_violations"].append(event.to_dict())
            
            # Track user provisioning
            elif event.event_type in [
                AuditEventType.ROLE_ASSIGNED,
                AuditEventType.ROLE_REVOKED
            ]:
                report["user_provisioning"].append(event.to_dict())
        
        return report
    
    def generate_gdpr_report(self, tenant_id: str,
                            start_time: datetime,
                            end_time: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        events = self.audit_logger.query(
            tenant_id,
            start_time=start_time,
            end_time=end_time
        )
        
        report = {
            "report_type": "GDPR Compliance",
            "tenant_id": tenant_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "data_access": [],
            "data_modifications": [],
            "data_deletions": [],
            "consent_tracking": []
        }
        
        for event in events:
            # Track data access
            if event.event_type == AuditEventType.RESOURCE_READ:
                if event.metadata.get("personal_data"):
                    report["data_access"].append(event.to_dict())
            
            # Track data modifications
            elif event.event_type == AuditEventType.RESOURCE_UPDATED:
                if event.metadata.get("personal_data"):
                    report["data_modifications"].append(event.to_dict())
            
            # Track data deletions
            elif event.event_type == AuditEventType.RESOURCE_DELETED:
                if event.metadata.get("personal_data"):
                    report["data_deletions"].append(event.to_dict())
        
        return report


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if not _audit_logger:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(event_type: AuditEventType, **kwargs):
    """Convenience function for audit logging"""
    event = AuditEvent(event_type=event_type, **kwargs)
    get_audit_logger().log(event)