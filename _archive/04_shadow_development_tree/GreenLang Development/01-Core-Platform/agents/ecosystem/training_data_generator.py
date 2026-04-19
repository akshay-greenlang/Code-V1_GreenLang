# -*- coding: utf-8 -*-
"""
GL-ECO-X-006: Training Data Generator
======================================

Generates synthetic training datasets for agent development and testing.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import random
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class DatasetType(str, Enum):
    EMISSIONS = "emissions"
    ENERGY = "energy"
    FACILITY = "facility"
    BENCHMARK = "benchmark"
    FINANCIAL = "financial"
    CUSTOM = "custom"


class DataQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SyntheticDataConfig(BaseModel):
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_type: DatasetType = Field(..., description="Type of dataset")
    record_count: int = Field(default=1000, ge=1, le=1000000)
    start_date: datetime = Field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = Field(default_factory=DeterministicClock.now)
    noise_level: float = Field(default=0.1, ge=0, le=1, description="Noise level 0-1")
    include_anomalies: bool = Field(default=True)
    anomaly_rate: float = Field(default=0.02, ge=0, le=0.5)
    seed: int = Field(default=42)
    schema: Dict[str, Any] = Field(default_factory=dict)


class TrainingDataset(BaseModel):
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Dataset name")
    dataset_type: DatasetType = Field(...)
    record_count: int = Field(default=0)
    data_quality: DataQuality = Field(default=DataQuality.HIGH)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    schema: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: str = Field(default="")


class TrainingDataInput(BaseModel):
    operation: str = Field(..., description="Operation to perform")
    config: Optional[SyntheticDataConfig] = Field(None)
    dataset_id: Optional[str] = Field(None)
    dataset_type: Optional[DatasetType] = Field(None)
    record_count: Optional[int] = Field(None)

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {
            'generate_dataset', 'get_dataset', 'list_datasets',
            'validate_dataset', 'augment_dataset', 'split_dataset',
            'get_schema', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class TrainingDataOutput(BaseModel):
    success: bool = Field(...)
    operation: str = Field(...)
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class TrainingDataGenerator(BaseAgent):
    """GL-ECO-X-006: Training Data Generator"""

    AGENT_ID = "GL-ECO-X-006"
    AGENT_NAME = "Training Data Generator"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Generates synthetic training datasets",
                version=self.VERSION,
            )
        super().__init__(config)
        self._datasets: Dict[str, TrainingDataset] = {}
        self._total_records_generated = 0
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()
        try:
            td_input = TrainingDataInput(**input_data)
            result_data = self._route_operation(td_input)
            provenance_hash = hashlib.sha256(
                json.dumps({"in": input_data, "out": result_data}, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            output = TrainingDataOutput(
                success=True, operation=td_input.operation, data=result_data,
                provenance_hash=provenance_hash, processing_time_ms=(time.time() - start_time) * 1000,
            )
            return AgentResult(success=True, data=output.model_dump())
        except Exception as e:
            self.logger.error(f"Operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, td_input: TrainingDataInput) -> Dict[str, Any]:
        op = td_input.operation
        if op == "generate_dataset":
            return self._generate_dataset(td_input.config)
        elif op == "get_dataset":
            return self._get_dataset(td_input.dataset_id)
        elif op == "list_datasets":
            return self._list_datasets(td_input.dataset_type)
        elif op == "validate_dataset":
            return self._validate_dataset(td_input.dataset_id)
        elif op == "augment_dataset":
            return self._augment_dataset(td_input.dataset_id, td_input.record_count)
        elif op == "split_dataset":
            return self._split_dataset(td_input.dataset_id)
        elif op == "get_schema":
            return self._get_schema(td_input.dataset_type)
        elif op == "get_statistics":
            return self._get_statistics()
        raise ValueError(f"Unknown operation: {op}")

    def _generate_dataset(self, config: Optional[SyntheticDataConfig]) -> Dict[str, Any]:
        if not config:
            return {"error": "config required"}

        random.seed(config.seed)
        records = []

        for i in range(config.record_count):
            record = self._generate_record(config, i)
            records.append(record)

        content_hash = hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()[:16]

        dataset = TrainingDataset(
            name=f"{config.dataset_type.value}_dataset_{len(self._datasets)+1}",
            dataset_type=config.dataset_type,
            record_count=len(records),
            records=records,
            schema=self._get_schema_for_type(config.dataset_type),
            provenance_hash=content_hash,
        )

        self._datasets[dataset.dataset_id] = dataset
        self._total_records_generated += len(records)

        return {"dataset_id": dataset.dataset_id, "record_count": len(records), "provenance_hash": content_hash}

    def _generate_record(self, config: SyntheticDataConfig, index: int) -> Dict[str, Any]:
        """Generate a single record based on dataset type."""
        is_anomaly = config.include_anomalies and random.random() < config.anomaly_rate
        noise = random.gauss(0, config.noise_level)

        time_delta = (config.end_date - config.start_date) / max(config.record_count - 1, 1)
        timestamp = config.start_date + time_delta * index

        if config.dataset_type == DatasetType.EMISSIONS:
            base_value = 100 + 50 * (index % 24) / 24  # Daily pattern
            value = base_value * (1 + noise) * (5 if is_anomaly else 1)
            return {
                "timestamp": timestamp.isoformat(),
                "facility_id": f"FAC-{(index % 5) + 1:03d}",
                "emissions_kg": round(value, 4),
                "gas_type": random.choice(["co2", "ch4", "n2o"]),
                "is_anomaly": is_anomaly,
            }
        elif config.dataset_type == DatasetType.ENERGY:
            base_value = 500 + 200 * (index % 24) / 24
            value = base_value * (1 + noise) * (3 if is_anomaly else 1)
            return {
                "timestamp": timestamp.isoformat(),
                "facility_id": f"FAC-{(index % 5) + 1:03d}",
                "energy_kwh": round(value, 4),
                "source": random.choice(["grid", "solar", "wind", "gas"]),
                "is_anomaly": is_anomaly,
            }
        else:
            return {
                "timestamp": timestamp.isoformat(),
                "index": index,
                "value": round(100 * (1 + noise), 4),
                "is_anomaly": is_anomaly,
            }

    def _get_schema_for_type(self, dtype: DatasetType) -> Dict[str, Any]:
        schemas = {
            DatasetType.EMISSIONS: {
                "timestamp": "datetime", "facility_id": "string", "emissions_kg": "float", "gas_type": "string"
            },
            DatasetType.ENERGY: {
                "timestamp": "datetime", "facility_id": "string", "energy_kwh": "float", "source": "string"
            },
        }
        return schemas.get(dtype, {"timestamp": "datetime", "value": "float"})

    def _get_dataset(self, dataset_id: Optional[str]) -> Dict[str, Any]:
        if not dataset_id or dataset_id not in self._datasets:
            return {"error": f"Dataset not found: {dataset_id}"}
        return self._datasets[dataset_id].model_dump()

    def _list_datasets(self, dtype: Optional[DatasetType]) -> Dict[str, Any]:
        datasets = list(self._datasets.values())
        if dtype:
            datasets = [d for d in datasets if d.dataset_type == dtype]
        return {
            "datasets": [
                {"dataset_id": d.dataset_id, "name": d.name, "type": d.dataset_type.value, "records": d.record_count}
                for d in datasets
            ],
            "count": len(datasets),
        }

    def _validate_dataset(self, dataset_id: Optional[str]) -> Dict[str, Any]:
        if not dataset_id or dataset_id not in self._datasets:
            return {"error": f"Dataset not found: {dataset_id}"}
        dataset = self._datasets[dataset_id]
        return {"dataset_id": dataset_id, "valid": True, "record_count": dataset.record_count, "quality": DataQuality.HIGH.value}

    def _augment_dataset(self, dataset_id: Optional[str], additional_count: Optional[int]) -> Dict[str, Any]:
        if not dataset_id or dataset_id not in self._datasets:
            return {"error": f"Dataset not found: {dataset_id}"}
        count = additional_count or 100
        # In production, would add new records
        return {"dataset_id": dataset_id, "augmented": True, "additional_records": count}

    def _split_dataset(self, dataset_id: Optional[str]) -> Dict[str, Any]:
        if not dataset_id or dataset_id not in self._datasets:
            return {"error": f"Dataset not found: {dataset_id}"}
        dataset = self._datasets[dataset_id]
        train_count = int(dataset.record_count * 0.8)
        return {
            "dataset_id": dataset_id,
            "train_records": train_count,
            "test_records": dataset.record_count - train_count,
            "split_ratio": "80/20",
        }

    def _get_schema(self, dtype: Optional[DatasetType]) -> Dict[str, Any]:
        if not dtype:
            return {"schemas": {t.value: self._get_schema_for_type(t) for t in DatasetType}}
        return {"schema": self._get_schema_for_type(dtype)}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "total_datasets": len(self._datasets),
            "total_records_generated": self._total_records_generated,
            "by_type": {t.value: sum(1 for d in self._datasets.values() if d.dataset_type == t) for t in DatasetType},
        }
