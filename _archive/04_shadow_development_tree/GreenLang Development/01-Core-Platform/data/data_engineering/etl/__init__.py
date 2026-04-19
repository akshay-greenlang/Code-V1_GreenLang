"""
ETL Pipeline Components
=======================

Extract-Transform-Load pipeline infrastructure for emission factor data sources.

Supported Sources:
- DEFRA (UK Government Conversion Factors)
- EPA eGRID (US Grid Factors)
- EPA Emission Factor Hub (Stationary, Mobile, Fugitive)
- Ecoinvent (LCA Database - requires license)
- IEA (International Energy Agency)
- IPCC AR6/AR7 (Global Emission Factors)

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from greenlang.data_engineering.etl.base_pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    LoadMode,
    PipelineMetrics,
    PipelineCheckpoint,
    DeadLetterQueue,
)
from greenlang.data_engineering.etl.defra_pipeline import (
    DEFRAPipeline,
    DEFRAPipelineConfig,
    DEFRARecord,
)
from greenlang.data_engineering.etl.epa_egrid_pipeline import (
    EPAeGRIDPipeline,
    eGRIDPipelineConfig,
    eGRIDRecord,
)
from greenlang.data_engineering.etl.epa_hub_pipeline import (
    EPAHubPipeline,
    EPAHubPipelineConfig,
    EPAHubRecord,
    EPAHubCategory,
)
from greenlang.data_engineering.etl.ecoinvent_pipeline import (
    EcoinventPipeline,
    EcoinventPipelineConfig,
    EcoinventRecord,
    EcoinventSystemModel,
    EcoinventExportFormat,
)
from greenlang.data_engineering.etl.iea_pipeline import (
    IEAPipeline,
    IEAPipelineConfig,
    IEARecord,
    IEADataset,
)
from greenlang.data_engineering.etl.transformers import (
    BaseTransformer,
    CleaningTransformer,
    EnrichmentTransformer,
    NormalizationTransformer,
    AggregationTransformer,
)

__all__ = [
    # Base pipeline
    "BasePipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "LoadMode",
    "PipelineMetrics",
    "PipelineCheckpoint",
    "DeadLetterQueue",
    # DEFRA
    "DEFRAPipeline",
    "DEFRAPipelineConfig",
    "DEFRARecord",
    # EPA eGRID
    "EPAeGRIDPipeline",
    "eGRIDPipelineConfig",
    "eGRIDRecord",
    # EPA Hub
    "EPAHubPipeline",
    "EPAHubPipelineConfig",
    "EPAHubRecord",
    "EPAHubCategory",
    # Ecoinvent
    "EcoinventPipeline",
    "EcoinventPipelineConfig",
    "EcoinventRecord",
    "EcoinventSystemModel",
    "EcoinventExportFormat",
    # IEA
    "IEAPipeline",
    "IEAPipelineConfig",
    "IEARecord",
    "IEADataset",
    # Transformers
    "BaseTransformer",
    "CleaningTransformer",
    "EnrichmentTransformer",
    "NormalizationTransformer",
    "AggregationTransformer",
]
