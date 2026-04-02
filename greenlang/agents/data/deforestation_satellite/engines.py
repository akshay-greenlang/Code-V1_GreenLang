# -*- coding: utf-8 -*-
"""
Barrel re-exports for deforestation satellite engine classes.

Maps the canonical engine names used in __init__.py to the actual
implementation classes spread across individual module files.
"""

from greenlang.agents.data.deforestation_satellite.satellite_data import (
    SatelliteDataEngine as SatelliteEngine,
)

# IndexEngine: vegetation index computation lives inside SatelliteDataEngine;
# re-export under the IndexEngine alias for backward compatibility.
IndexEngine = SatelliteEngine

from greenlang.agents.data.deforestation_satellite.forest_change import (
    ForestChangeEngine as ChangeEngine,
)
from greenlang.agents.data.deforestation_satellite.deforestation_classifier import (
    DeforestationClassifierEngine as ClassificationEngine,
)
from greenlang.agents.data.deforestation_satellite.alert_aggregation import (
    AlertAggregationEngine as AlertEngine,
)
from greenlang.agents.data.deforestation_satellite.baseline_assessment import (
    BaselineAssessmentEngine as BaselineEngine,
)
from greenlang.agents.data.deforestation_satellite.monitoring_pipeline import (
    MonitoringPipelineEngine as MonitoringEngine,
)
from greenlang.agents.data.deforestation_satellite.provenance import (
    ProvenanceTracker,
)

__all__ = [
    "SatelliteEngine",
    "IndexEngine",
    "ChangeEngine",
    "ClassificationEngine",
    "AlertEngine",
    "BaselineEngine",
    "MonitoringEngine",
    "ProvenanceTracker",
]
