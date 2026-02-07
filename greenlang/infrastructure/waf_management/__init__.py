# -*- coding: utf-8 -*-
"""
GreenLang WAF Management - SEC-010

DDoS protection and Web Application Firewall management for the
GreenLang Climate OS platform. Provides WAF rule building, testing,
deployment, anomaly detection, and AWS Shield Advanced integration.

Public API:
    - WAFRuleBuilder: Factory for creating and deploying WAF rules
    - WAFRuleTester: Test rules against synthetic traffic
    - AnomalyDetector: Real-time traffic analysis and attack detection
    - ShieldManager: AWS Shield Advanced management

Models:
    - WAFRule: WAF rule definition
    - RuleType: Supported rule types (rate_limit, geo_block, etc)
    - RuleAction: Rule actions (allow, block, count, captcha)
    - Attack: Detected attack representation
    - AttackType: Attack types (volumetric, slowloris, etc)
    - MitigationResult: Attack mitigation outcome
    - TrafficMetrics: Real-time traffic statistics

Configuration:
    - WAFConfig: Configuration model with environment profiles
    - get_config: Factory function for configuration singleton

Example:
    >>> from greenlang.infrastructure.waf_management import (
    ...     WAFRuleBuilder, WAFRuleTester, AnomalyDetector, ShieldManager,
    ...     WAFRule, RuleType, RuleAction, get_config,
    ... )
    >>> config = get_config()
    >>> builder = WAFRuleBuilder(config)
    >>> rule = builder.create_rule(
    ...     rule_type="rate_limit",
    ...     name="RateLimitPerIP",
    ...     config={"threshold": 2000, "window_seconds": 300}
    ... )
    >>> tester = WAFRuleTester(config)
    >>> results = await tester.test_rule(rule, tester.generate_test_requests("rate_limit"))
    >>> await builder.deploy_rule(rule, web_acl_id)
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.waf_management.config import (
    WAFConfig,
    EnvironmentConfig,
    EnvironmentProfile,
    get_config,
    reset_config,
)
from greenlang.infrastructure.waf_management.models import (
    # Enums
    RuleType,
    RuleAction,
    RuleConditionOperator,
    RuleStatus,
    AttackType,
    AttackSeverity,
    MitigationStatus,
    # Core Models
    RuleCondition,
    WAFRule,
    WAFRuleMetrics,
    Attack,
    MitigationAction,
    MitigationResult,
    TrafficMetrics,
    ShieldProtection,
    ProtectionGroup,
)
from greenlang.infrastructure.waf_management.rule_builder import (
    WAFRuleBuilder,
    RuleValidationResult,
    BaseRuleBuilder,
    RateLimitRuleBuilder,
    GeoBlockRuleBuilder,
    IPReputationRuleBuilder,
    SQLInjectionRuleBuilder,
    XSSRuleBuilder,
    CustomRegexRuleBuilder,
    BotControlRuleBuilder,
)
from greenlang.infrastructure.waf_management.rule_tester import (
    WAFRuleTester,
    TestRequest,
    TestResult,
    TestReport,
)
from greenlang.infrastructure.waf_management.anomaly_detector import (
    AnomalyDetector,
    TrafficBaseline,
    DetectionResult,
)
from greenlang.infrastructure.waf_management.shield_manager import (
    ShieldManager,
    ShieldSubscriptionStatus,
    AttackStatistics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration
    "WAFConfig",
    "EnvironmentConfig",
    "EnvironmentProfile",
    "get_config",
    "reset_config",
    # Enums
    "RuleType",
    "RuleAction",
    "RuleConditionOperator",
    "RuleStatus",
    "AttackType",
    "AttackSeverity",
    "MitigationStatus",
    # Core Models
    "RuleCondition",
    "WAFRule",
    "WAFRuleMetrics",
    "Attack",
    "MitigationAction",
    "MitigationResult",
    "TrafficMetrics",
    "ShieldProtection",
    "ProtectionGroup",
    # Rule Builder
    "WAFRuleBuilder",
    "RuleValidationResult",
    "BaseRuleBuilder",
    "RateLimitRuleBuilder",
    "GeoBlockRuleBuilder",
    "IPReputationRuleBuilder",
    "SQLInjectionRuleBuilder",
    "XSSRuleBuilder",
    "CustomRegexRuleBuilder",
    "BotControlRuleBuilder",
    # Rule Tester
    "WAFRuleTester",
    "TestRequest",
    "TestResult",
    "TestReport",
    # Anomaly Detector
    "AnomalyDetector",
    "TrafficBaseline",
    "DetectionResult",
    # Shield Manager
    "ShieldManager",
    "ShieldSubscriptionStatus",
    "AttackStatistics",
]

__version__ = "1.0.0"
