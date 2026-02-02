# -*- coding: utf-8 -*-
# LIME Fuel Property Explainer for GL-011 FuelCraft
# Author: GreenLang AI Team, Version: 1.0.0

from __future__ import annotations
import hashlib, json, logging, time, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
DEFAULT_RANDOM_SEED, DEFAULT_NUM_SAMPLES, DEFAULT_NUM_FEATURES = 42, 5000, 10
MIN_LOCAL_FIDELITY, PRECISION = 0.7, 6

class FuelPropertyType(str, Enum):
    OCTANE_RATING = "octane_rating"
    CETANE_NUMBER = "cetane_number"
    FLASH_POINT = "flash_point"
    EMISSIONS_FACTOR = "emissions_factor"

class PerturbationType(str, Enum):
    CHEMISTRY_CONSTRAINED = "chemistry_constrained"
    GAUSSIAN = "gaussian"

FUEL_LABELS = {"aromatics_pct": "Aromatics (%)", "ethanol_pct": "Ethanol (%)"}

@dataclass
class ChemistryConstraint:
    feature_name: str
    min_value: float
    max_value: float
    typical_range: Tuple[float, float]

class FuelLocalSurrogateModel(BaseModel):
    coefficients: Dict[str, float] = Field(default_factory=dict)
    r_squared: float = Field(0.0, ge=0.0, le=1.0)

class FuelFeatureContribution(BaseModel):
    feature_name: str
    lime_weight: float
    contribution_pct: float
    direction: str
    rank: int

class LIMEFuelExplanation(BaseModel):
    explanation_id: str
    property_type: FuelPropertyType
    prediction_value: float
    contributions: List[FuelFeatureContribution] = Field(default_factory=list)
    local_fidelity_r2: float
    provenance_hash: str = ""
    def model_post_init(self, ctx): 
        if not self.provenance_hash:
            self.provenance_hash = hashlib.sha256(json.dumps({"id": self.explanation_id}).encode()).hexdigest()

class LIMEFuelExplainer:
    def __init__(self, training_data, feature_names, config=None):
        self.training_data, self.feature_names = training_data, feature_names
        self._explainer = None
        try:
            import lime.lime_tabular
            self._explainer = lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=feature_names, mode="regression")
        except: pass
    def explain(self, features, predict_fn, property_type):
        fv = np.asarray(features).flatten()
        pred = float(predict_fn(fv.reshape(1,-1))[0])
        if not self._explainer: raise RuntimeError("LIME unavailable")
        exp = self._explainer.explain_instance(fv, predict_fn, num_features=10)
        weights = {}
        for d, w in exp.as_list():
            for n in self.feature_names:
                if n in d: weights[n] = weights.get(n,0)+w; break
        total = sum(abs(v) for v in weights.values()) or 1
        contribs = [FuelFeatureContribution(feature_name=n, lime_weight=round(w,6), contribution_pct=round(abs(w)/total*100,2), direction="positive" if w>=0 else "negative", rank=i+1) for i,(n,w) in enumerate(sorted(weights.items(), key=lambda x:abs(x[1]), reverse=True)[:10])]
        return LIMEFuelExplanation(explanation_id=str(uuid.uuid4()), property_type=property_type, prediction_value=round(pred,6), contributions=contribs, local_fidelity_r2=round(getattr(exp,"score",0.0),6))

__all__ = ["FuelPropertyType", "LIMEFuelExplainer", "LIMEFuelExplanation", "FuelFeatureContribution"]
