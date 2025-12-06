"""
HAZOPAnalyzer - Hazard and Operability Study Framework

This module implements HAZOP study framework per IEC 61882
for systematic identification of hazards and operability problems.

Key concepts:
- Nodes: Process sections to analyze
- Guide Words: Deviation descriptors (NO, MORE, LESS, etc.)
- Deviations: Combination of parameter + guide word
- Consequences: Potential outcomes of deviations
- Safeguards: Existing protections

Reference: IEC 61882:2016

Example:
    >>> from greenlang.safety.risk.hazop_analyzer import HAZOPAnalyzer
    >>> analyzer = HAZOPAnalyzer()
    >>> study = analyzer.create_study("Plant-001")
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class GuideWord(str, Enum):
    """HAZOP guide words per IEC 61882."""
    NO = "no"  # Complete negation
    MORE = "more"  # Quantitative increase
    LESS = "less"  # Quantitative decrease
    AS_WELL_AS = "as_well_as"  # Qualitative modification
    PART_OF = "part_of"  # Qualitative modification
    REVERSE = "reverse"  # Logical opposite
    OTHER_THAN = "other_than"  # Complete substitution
    EARLY = "early"  # Time
    LATE = "late"  # Time
    BEFORE = "before"  # Order/sequence
    AFTER = "after"  # Order/sequence


class ProcessParameter(str, Enum):
    """Process parameters for HAZOP."""
    FLOW = "flow"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    LEVEL = "level"
    COMPOSITION = "composition"
    PHASE = "phase"
    REACTION = "reaction"
    MIXING = "mixing"
    TIME = "time"
    SEQUENCE = "sequence"


class HAZOPDeviation(BaseModel):
    """HAZOP deviation record."""
    deviation_id: str = Field(default_factory=lambda: f"DEV-{uuid.uuid4().hex[:6].upper()}")
    node_id: str = Field(...)
    parameter: ProcessParameter = Field(...)
    guide_word: GuideWord = Field(...)
    deviation_description: str = Field(...)
    causes: List[str] = Field(default_factory=list)
    consequences: List[str] = Field(default_factory=list)
    existing_safeguards: List[str] = Field(default_factory=list)
    severity: int = Field(default=1, ge=1, le=5)
    likelihood: int = Field(default=1, ge=1, le=5)
    risk_ranking: int = Field(default=1)
    recommendations: List[str] = Field(default_factory=list)
    action_required: bool = Field(default=False)
    action_party: Optional[str] = Field(None)
    target_date: Optional[datetime] = Field(None)


class HAZOPNode(BaseModel):
    """HAZOP study node (process section)."""
    node_id: str = Field(...)
    node_name: str = Field(...)
    description: str = Field(default="")
    design_intent: str = Field(...)
    pid_reference: str = Field(default="")
    equipment_list: List[str] = Field(default_factory=list)
    deviations: List[HAZOPDeviation] = Field(default_factory=list)


class HAZOPStudy(BaseModel):
    """Complete HAZOP study record."""
    study_id: str = Field(default_factory=lambda: f"HAZOP-{uuid.uuid4().hex[:8].upper()}")
    study_title: str = Field(...)
    facility: str = Field(...)
    study_date: datetime = Field(default_factory=datetime.utcnow)
    team_leader: str = Field(...)
    team_members: List[str] = Field(default_factory=list)
    scribe: str = Field(default="")
    scope: str = Field(...)
    nodes: List[HAZOPNode] = Field(default_factory=list)
    total_deviations: int = Field(default=0)
    total_recommendations: int = Field(default=0)
    high_risk_items: int = Field(default=0)
    status: str = Field(default="in_progress")
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HAZOPAnalyzer:
    """
    HAZOP Study Analyzer.

    Implements HAZOP methodology per IEC 61882 for systematic
    hazard identification.

    Example:
        >>> analyzer = HAZOPAnalyzer()
        >>> study = analyzer.create_study("Reactor System", facility="Plant-001")
        >>> analyzer.add_node(study.study_id, node_config)
    """

    # Standard deviation combinations
    STANDARD_DEVIATIONS = {
        ProcessParameter.FLOW: [GuideWord.NO, GuideWord.MORE, GuideWord.LESS, GuideWord.REVERSE],
        ProcessParameter.PRESSURE: [GuideWord.MORE, GuideWord.LESS],
        ProcessParameter.TEMPERATURE: [GuideWord.MORE, GuideWord.LESS],
        ProcessParameter.LEVEL: [GuideWord.MORE, GuideWord.LESS],
        ProcessParameter.COMPOSITION: [GuideWord.AS_WELL_AS, GuideWord.PART_OF, GuideWord.OTHER_THAN],
    }

    def __init__(self):
        """Initialize HAZOPAnalyzer."""
        self.studies: Dict[str, HAZOPStudy] = {}
        logger.info("HAZOPAnalyzer initialized")

    def create_study(
        self,
        title: str,
        facility: str,
        team_leader: str,
        scope: str,
        team_members: Optional[List[str]] = None
    ) -> HAZOPStudy:
        """Create a new HAZOP study."""
        study = HAZOPStudy(
            study_title=title,
            facility=facility,
            team_leader=team_leader,
            scope=scope,
            team_members=team_members or [],
        )

        self.studies[study.study_id] = study
        logger.info(f"HAZOP study created: {study.study_id}")
        return study

    def add_node(
        self,
        study_id: str,
        node_id: str,
        node_name: str,
        design_intent: str,
        description: str = "",
        pid_reference: str = ""
    ) -> HAZOPNode:
        """Add a node to HAZOP study."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        node = HAZOPNode(
            node_id=node_id,
            node_name=node_name,
            description=description,
            design_intent=design_intent,
            pid_reference=pid_reference,
        )

        self.studies[study_id].nodes.append(node)
        logger.info(f"Node added to study {study_id}: {node_id}")
        return node

    def add_deviation(
        self,
        study_id: str,
        node_id: str,
        deviation: HAZOPDeviation
    ) -> HAZOPDeviation:
        """Add a deviation to a node."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        study = self.studies[study_id]
        for node in study.nodes:
            if node.node_id == node_id:
                # Calculate risk ranking
                deviation.risk_ranking = deviation.severity * deviation.likelihood
                deviation.action_required = deviation.risk_ranking >= 12

                node.deviations.append(deviation)
                study.total_deviations += 1
                study.total_recommendations += len(deviation.recommendations)

                if deviation.risk_ranking >= 15:
                    study.high_risk_items += 1

                logger.info(f"Deviation added: {deviation.deviation_id}")
                return deviation

        raise ValueError(f"Node not found: {node_id}")

    def generate_standard_deviations(
        self,
        node_id: str,
        parameters: Optional[List[ProcessParameter]] = None
    ) -> List[Dict[str, Any]]:
        """Generate standard deviations for a node."""
        parameters = parameters or list(ProcessParameter)
        deviations = []

        for param in parameters:
            guide_words = self.STANDARD_DEVIATIONS.get(param, [GuideWord.NO, GuideWord.MORE, GuideWord.LESS])
            for gw in guide_words:
                deviations.append({
                    "parameter": param.value,
                    "guide_word": gw.value,
                    "description": f"{gw.value.upper()} {param.value.upper()}",
                })

        return deviations

    def complete_study(self, study_id: str) -> HAZOPStudy:
        """Mark study as complete and calculate provenance."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        study = self.studies[study_id]
        study.status = "completed"
        study.provenance_hash = hashlib.sha256(
            f"{study.study_id}|{study.total_deviations}|{study.total_recommendations}".encode()
        ).hexdigest()

        return study
