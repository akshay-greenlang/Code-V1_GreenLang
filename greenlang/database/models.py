"""
Database Models
===============

ORM models for GreenLang database.

Author: Data Team
Created: 2025-11-21
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class BaseModel:
    """Base class for all database models."""
    __tablename__ = "base"

    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        data = asdict(self)
        # Convert datetime objects to strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_row(cls, row: tuple) -> 'BaseModel':
        """Create model instance from database row."""
        # This should be overridden by subclasses
        raise NotImplementedError

    def validate(self) -> bool:
        """Validate model data."""
        return True


@dataclass
class EmissionFactorModel(BaseModel):
    """Model for emission factors."""
    __tablename__ = "emission_factors"

    material_id: str = ""
    factor: float = 0.0
    unit: str = ""
    source: Optional[str] = None
    region: Optional[str] = None
    year: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: tuple) -> 'EmissionFactorModel':
        """Create from database row."""
        return cls(
            id=row[0],
            material_id=row[1],
            factor=row[2],
            unit=row[3],
            source=row[4],
            region=row[5],
            year=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            updated_at=datetime.fromisoformat(row[8]) if row[8] else None
        )

    def validate(self) -> bool:
        """Validate emission factor data."""
        if not self.material_id:
            raise ValueError("material_id is required")
        if self.factor < 0:
            raise ValueError("factor must be non-negative")
        if not self.unit:
            raise ValueError("unit is required")
        return True


@dataclass
class ActivityDataModel(BaseModel):
    """Model for activity data."""
    __tablename__ = "activity_data"

    activity_id: str = ""
    activity_type: str = ""
    value: float = 0.0
    unit: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: tuple) -> 'ActivityDataModel':
        """Create from database row."""
        return cls(
            id=row[0],
            activity_id=row[1],
            activity_type=row[2],
            value=row[3],
            unit=row[4],
            timestamp=datetime.fromisoformat(row[5]) if row[5] else None,
            metadata=json.loads(row[6]) if row[6] else {},
            created_at=datetime.fromisoformat(row[7]) if row[7] else None
        )

    def validate(self) -> bool:
        """Validate activity data."""
        if not self.activity_id:
            raise ValueError("activity_id is required")
        if not self.activity_type:
            raise ValueError("activity_type is required")
        if self.value < 0:
            raise ValueError("value must be non-negative")
        if not self.unit:
            raise ValueError("unit is required")
        return True

    def calculate_emissions(self, emission_factor: float) -> float:
        """Calculate emissions for this activity."""
        return self.value * emission_factor


@dataclass
class SupplierModel(BaseModel):
    """Model for supplier data."""
    __tablename__ = "suppliers"

    supplier_id: str = ""
    name: str = ""
    country: Optional[str] = None
    industry: Optional[str] = None
    emissions_data: Optional[Dict[str, Any]] = None
    certification: Optional[str] = None
    risk_score: Optional[float] = None
    engagement_status: str = "not_started"

    @classmethod
    def from_row(cls, row: tuple) -> 'SupplierModel':
        """Create from database row."""
        return cls(
            id=row[0],
            supplier_id=row[1],
            name=row[2],
            country=row[3],
            industry=row[4],
            emissions_data=json.loads(row[5]) if row[5] else None,
            certification=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            updated_at=datetime.fromisoformat(row[8]) if row[8] else None
        )

    def validate(self) -> bool:
        """Validate supplier data."""
        if not self.supplier_id:
            raise ValueError("supplier_id is required")
        if not self.name:
            raise ValueError("name is required")
        if self.risk_score is not None and (self.risk_score < 0 or self.risk_score > 100):
            raise ValueError("risk_score must be between 0 and 100")
        return True

    def get_emissions_intensity(self) -> Optional[float]:
        """Get supplier's emissions intensity."""
        if self.emissions_data:
            return self.emissions_data.get('intensity')
        return None


@dataclass
class AuditLogModel(BaseModel):
    """Model for audit log entries."""
    __tablename__ = "audit_log"

    event_type: str = ""
    event_data: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple) -> 'AuditLogModel':
        """Create from database row."""
        return cls(
            id=row[0],
            event_type=row[1],
            event_data=json.loads(row[2]) if row[2] else None,
            user_id=row[3],
            timestamp=datetime.fromisoformat(row[4]) if row[4] else None,
            ip_address=row[5],
            session_id=row[6]
        )

    def validate(self) -> bool:
        """Validate audit log entry."""
        if not self.event_type:
            raise ValueError("event_type is required")
        return True

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        return json.dumps(data, default=str)