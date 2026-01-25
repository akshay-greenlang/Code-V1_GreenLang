# -*- coding: utf-8 -*-
"""
Multi-Modal Data Pipeline for GreenLang
========================================

This module provides comprehensive support for processing multiple data modalities
including P&ID diagrams, thermal images, equipment photos, time-series data,
and technical documents.

Components:
- multimodal_models: Pydantic models for all asset types
- pid_processor: P&ID diagram analysis and entity extraction
- thermal_processor: Thermal image analysis and hotspot detection
- document_processor: PDF and document extraction
- time_series_processor: Time-series handling and analysis
- multimodal_service: Unified orchestration service
"""

from .multimodal_models import (
    AssetType,
    ProcessingStatus,
    AssetMetadata,
    ImageAsset,
    DiagramAsset,
    ThermalImage,
    TimeSeriesAsset,
    DocumentAsset,
    BoundingBox,
    EmbeddingVector,
    EquipmentEntity,
    ThermalHotspot,
)

__all__ = [
    "AssetType",
    "ProcessingStatus",
    "AssetMetadata",
    "ImageAsset",
    "DiagramAsset",
    "ThermalImage",
    "TimeSeriesAsset",
    "DocumentAsset",
    "BoundingBox",
    "EmbeddingVector",
    "EquipmentEntity",
    "ThermalHotspot",
]

__version__ = "1.0.0"
