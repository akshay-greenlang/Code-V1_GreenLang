# -*- coding: utf-8 -*-
"""
GreenLang Robustness Framework

Provides robustness testing and fail-safe capabilities for ML models,
including adversarial testing, distribution shift detection, and safe bounds.

Classes:
    AdversarialTester: Adversarial testing for model robustness
    DistributionShift: Distribution shift detection and handling
    SafeBounds: Safe exploration boundaries
    FailsafeML: Fail-safe behaviors for production

Example:
    >>> from greenlang.ml.robustness import AdversarialTester, FailsafeML
    >>> tester = AdversarialTester(model)
    >>> vulnerabilities = tester.test_robustness(X_test)
    >>> failsafe = FailsafeML(model, fallback_strategy="conservative")
    >>> safe_prediction = failsafe.predict_safe(X)
"""

from greenlang.ml.robustness.adversarial_tester import AdversarialTester
from greenlang.ml.robustness.distribution_shift import DistributionShift
from greenlang.ml.robustness.safe_bounds import SafeBounds
from greenlang.ml.robustness.failsafe_ml import FailsafeML

__all__ = [
    "AdversarialTester",
    "DistributionShift",
    "SafeBounds",
    "FailsafeML",
]
