# -*- coding: utf-8 -*-
"""
WAF Rule Builder - SEC-010

Provides a fluent interface for building AWS WAF v2 rules programmatically.
Supports all rule types defined in the WAF management system and handles
deployment to AWS WAF via boto3.

Classes:
    - BaseRuleBuilder: Abstract base class for rule builders
    - RateLimitRuleBuilder: Build rate-based blocking rules
    - GeoBlockRuleBuilder: Build geographic blocking rules
    - IPReputationRuleBuilder: Build IP set reference rules
    - SQLInjectionRuleBuilder: Build SQL injection detection rules
    - XSSRuleBuilder: Build XSS detection rules
    - CustomRegexRuleBuilder: Build custom regex pattern rules
    - BotControlRuleBuilder: Build AWS managed bot control rules
    - WAFRuleBuilder: Factory for creating and deploying WAF rules

Example:
    >>> from greenlang.infrastructure.waf_management.rule_builder import WAFRuleBuilder
    >>> builder = WAFRuleBuilder(config)
    >>> rule = await builder.create_rule(
    ...     rule_type="rate_limit",
    ...     name="RateLimitPerIP",
    ...     config={"threshold": 2000, "window_seconds": 300}
    ... )
    >>> await builder.deploy_rule(rule, web_acl_id)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Import boto3 with graceful fallback
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore
    BotoCoreError = Exception  # type: ignore
    ClientError = Exception  # type: ignore

from greenlang.infrastructure.waf_management.config import WAFConfig, get_config
from greenlang.infrastructure.waf_management.models import (
    RuleAction,
    RuleCondition,
    RuleStatus,
    RuleType,
    WAFRule,
)


# ---------------------------------------------------------------------------
# Validation Results
# ---------------------------------------------------------------------------


class RuleValidationResult:
    """Result of rule validation.

    Attributes:
        is_valid: Whether the rule passed validation.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
    """

    def __init__(
        self,
        is_valid: bool = True,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Base Rule Builder
# ---------------------------------------------------------------------------


class BaseRuleBuilder(ABC):
    """Abstract base class for WAF rule builders.

    Each rule type has a specific builder that generates the appropriate
    AWS WAF statement structure.
    """

    def __init__(self, config: WAFConfig):
        """Initialize the rule builder.

        Args:
            config: WAF configuration instance.
        """
        self.config = config

    @abstractmethod
    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build the AWS WAF statement for this rule type.

        Args:
            rule: The WAF rule to build a statement for.

        Returns:
            AWS WAF statement dictionary.
        """
        pass

    @abstractmethod
    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate the rule configuration.

        Args:
            rule: The WAF rule to validate.

        Returns:
            Validation result with any errors or warnings.
        """
        pass

    def _build_visibility_config(self, rule: WAFRule) -> Dict[str, Any]:
        """Build the visibility configuration for metrics.

        Args:
            rule: The WAF rule.

        Returns:
            Visibility config dictionary.
        """
        # Create a CloudWatch metric name from the rule name
        metric_name = re.sub(r"[^a-zA-Z0-9]", "", rule.name)[:128]
        return {
            "SampledRequestsEnabled": True,
            "CloudWatchMetricsEnabled": True,
            "MetricName": f"GL-WAF-{metric_name}",
        }

    def _map_action_to_aws(self, action: RuleAction) -> Dict[str, Any]:
        """Map internal action to AWS WAF action.

        Args:
            action: Internal rule action.

        Returns:
            AWS WAF action dictionary.
        """
        action_mapping = {
            RuleAction.ALLOW: {"Allow": {}},
            RuleAction.BLOCK: {"Block": {}},
            RuleAction.COUNT: {"Count": {}},
            RuleAction.CAPTCHA: {"Captcha": {}},
            RuleAction.CHALLENGE: {"Challenge": {}},
        }
        return action_mapping.get(action, {"Block": {}})


# ---------------------------------------------------------------------------
# Rate Limit Rule Builder
# ---------------------------------------------------------------------------


class RateLimitRuleBuilder(BaseRuleBuilder):
    """Builder for rate-based WAF rules.

    Creates AWS WAF rate-based statements that block or count
    requests exceeding a threshold per IP address.
    """

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build rate-based statement.

        Args:
            rule: WAF rule with rate_limit_threshold and rate_limit_window_seconds.

        Returns:
            AWS WAF RateBasedStatement.
        """
        statement = {
            "RateBasedStatement": {
                "Limit": rule.rate_limit_threshold,
                "AggregateKeyType": "IP",
            }
        }

        # Add scope-down statement if conditions are specified
        if rule.conditions:
            scope_down = self._build_scope_down_statement(rule.conditions)
            if scope_down:
                statement["RateBasedStatement"]["ScopeDownStatement"] = scope_down

        return statement

    def _build_scope_down_statement(
        self, conditions: List[RuleCondition]
    ) -> Optional[Dict[str, Any]]:
        """Build scope-down statement from conditions.

        Args:
            conditions: List of rule conditions.

        Returns:
            AWS WAF statement or None if no conditions.
        """
        if not conditions:
            return None

        # For now, support simple URI path conditions
        statements = []
        for cond in conditions:
            if cond.field == "uri_path" and cond.values:
                statements.append({
                    "ByteMatchStatement": {
                        "SearchString": cond.values[0],
                        "FieldToMatch": {"UriPath": {}},
                        "TextTransformations": [
                            {"Priority": 0, "Type": "LOWERCASE"}
                        ],
                        "PositionalConstraint": "STARTS_WITH"
                        if cond.operator.value == "starts_with"
                        else "CONTAINS",
                    }
                })

        if len(statements) == 1:
            return statements[0]
        elif len(statements) > 1:
            return {"AndStatement": {"Statements": statements}}
        return None

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate rate limit rule configuration.

        Args:
            rule: The rate limit rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.RATE_LIMIT:
            result.add_error(f"Expected rate_limit rule type, got {rule.rule_type}")

        # AWS WAF minimum rate limit is 100
        if rule.rate_limit_threshold < 100:
            result.add_error(
                f"Rate limit threshold must be at least 100, got {rule.rate_limit_threshold}"
            )

        # AWS WAF maximum rate limit is 2,000,000,000
        if rule.rate_limit_threshold > 2000000000:
            result.add_error(
                f"Rate limit threshold must be at most 2,000,000,000, got {rule.rate_limit_threshold}"
            )

        # Warn if threshold is very high
        if rule.rate_limit_threshold > 10000:
            result.add_warning(
                f"Rate limit threshold of {rule.rate_limit_threshold} may be too permissive"
            )

        return result


# ---------------------------------------------------------------------------
# Geo Block Rule Builder
# ---------------------------------------------------------------------------


class GeoBlockRuleBuilder(BaseRuleBuilder):
    """Builder for geographic blocking WAF rules.

    Creates AWS WAF geo-match statements that block or count
    requests from specified countries.
    """

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build geo-match statement.

        Args:
            rule: WAF rule with blocked_countries list.

        Returns:
            AWS WAF GeoMatchStatement.
        """
        return {
            "GeoMatchStatement": {
                "CountryCodes": rule.blocked_countries,
            }
        }

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate geo-block rule configuration.

        Args:
            rule: The geo-block rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.GEO_BLOCK:
            result.add_error(f"Expected geo_block rule type, got {rule.rule_type}")

        if not rule.blocked_countries:
            result.add_error("Geo-block rule requires at least one country code")

        # Validate country codes are ISO 3166-1 alpha-2
        valid_country_pattern = re.compile(r"^[A-Z]{2}$")
        for code in rule.blocked_countries:
            if not valid_country_pattern.match(code):
                result.add_error(
                    f"Invalid country code '{code}'. Must be ISO 3166-1 alpha-2."
                )

        # Warn if blocking major markets
        major_markets = {"US", "GB", "DE", "FR", "JP", "CA", "AU"}
        blocked_major = set(rule.blocked_countries) & major_markets
        if blocked_major:
            result.add_warning(
                f"Blocking major markets: {', '.join(blocked_major)}. "
                f"This may impact legitimate traffic."
            )

        return result


# ---------------------------------------------------------------------------
# IP Reputation Rule Builder
# ---------------------------------------------------------------------------


class IPReputationRuleBuilder(BaseRuleBuilder):
    """Builder for IP reputation-based WAF rules.

    Creates AWS WAF IP set reference statements for blocking
    known malicious IPs.
    """

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build IP set reference statement.

        Args:
            rule: WAF rule with ip_set_arn.

        Returns:
            AWS WAF IPSetReferenceStatement.
        """
        if not rule.ip_set_arn:
            raise ValueError("IP reputation rule requires ip_set_arn")

        return {
            "IPSetReferenceStatement": {
                "ARN": rule.ip_set_arn,
            }
        }

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate IP reputation rule configuration.

        Args:
            rule: The IP reputation rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.IP_REPUTATION:
            result.add_error(f"Expected ip_reputation rule type, got {rule.rule_type}")

        if not rule.ip_set_arn:
            result.add_error("IP reputation rule requires ip_set_arn")
        elif not rule.ip_set_arn.startswith("arn:aws:wafv2:"):
            result.add_error(
                f"Invalid IP set ARN format: {rule.ip_set_arn}"
            )

        return result


# ---------------------------------------------------------------------------
# SQL Injection Rule Builder
# ---------------------------------------------------------------------------


class SQLInjectionRuleBuilder(BaseRuleBuilder):
    """Builder for SQL injection detection WAF rules.

    Creates AWS WAF SQLi match statements for detecting
    SQL injection attempts in request components.
    """

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build SQLi match statement.

        Args:
            rule: WAF rule for SQL injection detection.

        Returns:
            AWS WAF SqliMatchStatement.
        """
        # Check multiple request components for SQLi
        statements = []

        # Check query string
        statements.append({
            "SqliMatchStatement": {
                "FieldToMatch": {"QueryString": {}},
                "TextTransformations": [
                    {"Priority": 0, "Type": "URL_DECODE"},
                    {"Priority": 1, "Type": "HTML_ENTITY_DECODE"},
                ],
            }
        })

        # Check body
        statements.append({
            "SqliMatchStatement": {
                "FieldToMatch": {
                    "Body": {
                        "OversizeHandling": "CONTINUE",
                    }
                },
                "TextTransformations": [
                    {"Priority": 0, "Type": "URL_DECODE"},
                    {"Priority": 1, "Type": "HTML_ENTITY_DECODE"},
                ],
            }
        })

        # Check URI path
        statements.append({
            "SqliMatchStatement": {
                "FieldToMatch": {"UriPath": {}},
                "TextTransformations": [
                    {"Priority": 0, "Type": "URL_DECODE"},
                ],
            }
        })

        # Combine with OR logic
        return {"OrStatement": {"Statements": statements}}

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate SQL injection rule configuration.

        Args:
            rule: The SQL injection rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.SQL_INJECTION:
            result.add_error(f"Expected sql_injection rule type, got {rule.rule_type}")

        # Warn if using count instead of block
        if rule.action == RuleAction.COUNT:
            result.add_warning(
                "SQL injection rule is set to COUNT mode. Consider BLOCK for production."
            )

        return result


# ---------------------------------------------------------------------------
# XSS Rule Builder
# ---------------------------------------------------------------------------


class XSSRuleBuilder(BaseRuleBuilder):
    """Builder for XSS detection WAF rules.

    Creates AWS WAF XSS match statements for detecting
    cross-site scripting attempts.
    """

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build XSS match statement.

        Args:
            rule: WAF rule for XSS detection.

        Returns:
            AWS WAF XssMatchStatement.
        """
        statements = []

        # Check query string
        statements.append({
            "XssMatchStatement": {
                "FieldToMatch": {"QueryString": {}},
                "TextTransformations": [
                    {"Priority": 0, "Type": "URL_DECODE"},
                    {"Priority": 1, "Type": "HTML_ENTITY_DECODE"},
                ],
            }
        })

        # Check body
        statements.append({
            "XssMatchStatement": {
                "FieldToMatch": {
                    "Body": {
                        "OversizeHandling": "CONTINUE",
                    }
                },
                "TextTransformations": [
                    {"Priority": 0, "Type": "URL_DECODE"},
                    {"Priority": 1, "Type": "HTML_ENTITY_DECODE"},
                ],
            }
        })

        # Check URI path
        statements.append({
            "XssMatchStatement": {
                "FieldToMatch": {"UriPath": {}},
                "TextTransformations": [
                    {"Priority": 0, "Type": "URL_DECODE"},
                ],
            }
        })

        # Combine with OR logic
        return {"OrStatement": {"Statements": statements}}

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate XSS rule configuration.

        Args:
            rule: The XSS rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.XSS:
            result.add_error(f"Expected xss rule type, got {rule.rule_type}")

        if rule.action == RuleAction.COUNT:
            result.add_warning(
                "XSS rule is set to COUNT mode. Consider BLOCK for production."
            )

        return result


# ---------------------------------------------------------------------------
# Custom Regex Rule Builder
# ---------------------------------------------------------------------------


class CustomRegexRuleBuilder(BaseRuleBuilder):
    """Builder for custom regex pattern WAF rules.

    Creates AWS WAF regex pattern set statements for matching
    custom patterns in request components.
    """

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build regex pattern statement.

        Args:
            rule: WAF rule with regex_pattern.

        Returns:
            AWS WAF RegexMatchStatement.
        """
        if not rule.regex_pattern:
            raise ValueError("Custom regex rule requires regex_pattern")

        # Determine which field to match based on conditions
        field_to_match = {"UriPath": {}}
        if rule.conditions:
            for cond in rule.conditions:
                if cond.field == "query_string":
                    field_to_match = {"QueryString": {}}
                elif cond.field == "body":
                    field_to_match = {"Body": {"OversizeHandling": "CONTINUE"}}
                elif cond.field == "headers":
                    field_to_match = {
                        "SingleHeader": {"Name": cond.values[0] if cond.values else "user-agent"}
                    }

        return {
            "RegexMatchStatement": {
                "RegexString": rule.regex_pattern,
                "FieldToMatch": field_to_match,
                "TextTransformations": [
                    {"Priority": 0, "Type": "LOWERCASE"},
                ],
            }
        }

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate custom regex rule configuration.

        Args:
            rule: The custom regex rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.CUSTOM_REGEX:
            result.add_error(f"Expected custom_regex rule type, got {rule.rule_type}")

        if not rule.regex_pattern:
            result.add_error("Custom regex rule requires regex_pattern")
        else:
            # Validate regex is syntactically correct
            try:
                re.compile(rule.regex_pattern)
            except re.error as e:
                result.add_error(f"Invalid regex pattern: {e}")

            # Warn about potentially expensive patterns
            if len(rule.regex_pattern) > 100:
                result.add_warning(
                    "Regex pattern is very long, which may impact performance"
                )

            # Check for catastrophic backtracking patterns
            dangerous_patterns = [r"(a+)+", r"(a|a)+", r"(.*)+"]
            for pattern in dangerous_patterns:
                if pattern in rule.regex_pattern:
                    result.add_warning(
                        f"Regex may be vulnerable to catastrophic backtracking: {pattern}"
                    )

        return result


# ---------------------------------------------------------------------------
# Bot Control Rule Builder
# ---------------------------------------------------------------------------


class BotControlRuleBuilder(BaseRuleBuilder):
    """Builder for AWS managed bot control rules.

    Creates AWS WAF managed rule group statements for bot detection
    and management.
    """

    MANAGED_RULE_GROUPS = {
        "AWSManagedRulesCommonRuleSet": {
            "vendor": "AWS",
            "name": "AWSManagedRulesCommonRuleSet",
            "description": "Common web exploits protection",
        },
        "AWSManagedRulesKnownBadInputsRuleSet": {
            "vendor": "AWS",
            "name": "AWSManagedRulesKnownBadInputsRuleSet",
            "description": "Known bad inputs protection",
        },
        "AWSManagedRulesSQLiRuleSet": {
            "vendor": "AWS",
            "name": "AWSManagedRulesSQLiRuleSet",
            "description": "SQL injection protection",
        },
        "AWSManagedRulesLinuxRuleSet": {
            "vendor": "AWS",
            "name": "AWSManagedRulesLinuxRuleSet",
            "description": "Linux-specific vulnerabilities",
        },
        "AWSManagedRulesBotControlRuleSet": {
            "vendor": "AWS",
            "name": "AWSManagedRulesBotControlRuleSet",
            "description": "Bot control and management",
        },
        "AWSManagedRulesATPRuleSet": {
            "vendor": "AWS",
            "name": "AWSManagedRulesATPRuleSet",
            "description": "Account takeover prevention",
        },
    }

    def build_statement(self, rule: WAFRule) -> Dict[str, Any]:
        """Build managed rule group statement.

        Args:
            rule: WAF rule with managed_rule_group.

        Returns:
            AWS WAF ManagedRuleGroupStatement.
        """
        rule_group = rule.managed_rule_group or "AWSManagedRulesBotControlRuleSet"
        group_info = self.MANAGED_RULE_GROUPS.get(
            rule_group,
            {"vendor": "AWS", "name": rule_group},
        )

        statement = {
            "ManagedRuleGroupStatement": {
                "VendorName": group_info["vendor"],
                "Name": group_info["name"],
            }
        }

        # Add rule action overrides if needed
        if rule.metadata.get("excluded_rules"):
            statement["ManagedRuleGroupStatement"]["ExcludedRules"] = [
                {"Name": name} for name in rule.metadata["excluded_rules"]
            ]

        return statement

    def validate(self, rule: WAFRule) -> RuleValidationResult:
        """Validate bot control rule configuration.

        Args:
            rule: The bot control rule to validate.

        Returns:
            Validation result.
        """
        result = RuleValidationResult()

        if rule.rule_type != RuleType.BOT_CONTROL:
            result.add_error(f"Expected bot_control rule type, got {rule.rule_type}")

        rule_group = rule.managed_rule_group or "AWSManagedRulesBotControlRuleSet"
        if rule_group not in self.MANAGED_RULE_GROUPS:
            result.add_warning(
                f"Unrecognized managed rule group: {rule_group}. "
                f"Proceeding with custom group name."
            )

        return result


# ---------------------------------------------------------------------------
# WAF Rule Builder (Factory)
# ---------------------------------------------------------------------------


class WAFRuleBuilder:
    """Factory for creating and deploying WAF rules.

    Provides a unified interface for building rules of any type
    and deploying them to AWS WAF.

    Example:
        >>> builder = WAFRuleBuilder(config)
        >>> rule = await builder.create_rule(
        ...     rule_type="rate_limit",
        ...     name="RateLimitPerIP",
        ...     config={"threshold": 2000}
        ... )
        >>> validation = builder.validate_rule(rule)
        >>> if validation.is_valid:
        ...     await builder.deploy_rule(rule, web_acl_id)
    """

    # Registry of rule builders by type
    RULE_BUILDERS: Dict[RuleType, Type[BaseRuleBuilder]] = {
        RuleType.RATE_LIMIT: RateLimitRuleBuilder,
        RuleType.GEO_BLOCK: GeoBlockRuleBuilder,
        RuleType.IP_REPUTATION: IPReputationRuleBuilder,
        RuleType.SQL_INJECTION: SQLInjectionRuleBuilder,
        RuleType.XSS: XSSRuleBuilder,
        RuleType.CUSTOM_REGEX: CustomRegexRuleBuilder,
        RuleType.BOT_CONTROL: BotControlRuleBuilder,
    }

    def __init__(self, config: Optional[WAFConfig] = None):
        """Initialize the WAF rule builder.

        Args:
            config: WAF configuration. If None, loads from environment.
        """
        self.config = config or get_config()
        self._waf_client = None
        self._builders: Dict[RuleType, BaseRuleBuilder] = {}

        # Initialize builders for each rule type
        for rule_type, builder_class in self.RULE_BUILDERS.items():
            self._builders[rule_type] = builder_class(self.config)

    @property
    def waf_client(self):
        """Get or create the boto3 WAF client.

        Returns:
            boto3 WAFv2 client.
        """
        if self._waf_client is None:
            if not BOTO3_AVAILABLE:
                raise ImportError(
                    "boto3 is required for WAF operations. "
                    "Install it with: pip install boto3"
                )
            self._waf_client = boto3.client(
                "wafv2",
                region_name=self.config.aws_region,
            )
        return self._waf_client

    def create_rule(
        self,
        rule_type: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> WAFRule:
        """Create a WAF rule from configuration.

        Args:
            rule_type: Type of rule (rate_limit, geo_block, etc.).
            name: Human-readable rule name.
            config: Rule-specific configuration.
            **kwargs: Additional rule attributes.

        Returns:
            Configured WAFRule instance.

        Raises:
            ValueError: If rule_type is invalid.
        """
        try:
            rule_type_enum = RuleType(rule_type)
        except ValueError:
            raise ValueError(
                f"Invalid rule type '{rule_type}'. "
                f"Valid types: {[t.value for t in RuleType]}"
            )

        config = config or {}

        # Build rule from configuration
        rule = WAFRule(
            name=name,
            rule_type=rule_type_enum,
            action=RuleAction(config.get("action", "block").lower()),
            priority=config.get("priority", 100),
            enabled=config.get("enabled", True),
            description=config.get("description", ""),
            rate_limit_threshold=config.get("threshold", 2000),
            rate_limit_window_seconds=config.get("window_seconds", 300),
            blocked_countries=config.get("blocked_countries", []),
            ip_set_arn=config.get("ip_set_arn"),
            regex_pattern=config.get("regex_pattern"),
            managed_rule_group=config.get("managed_rule_group"),
            created_by=config.get("created_by", ""),
            metadata=config.get("metadata", {}),
            **kwargs,
        )

        # Parse conditions if provided
        if "conditions" in config:
            rule.conditions = [
                RuleCondition(**cond) if isinstance(cond, dict) else cond
                for cond in config["conditions"]
            ]

        logger.info(
            "Created WAF rule: name=%s, type=%s, priority=%d",
            rule.name,
            rule.rule_type.value,
            rule.priority,
        )

        return rule

    def validate_rule(self, rule: WAFRule) -> RuleValidationResult:
        """Validate a WAF rule configuration.

        Args:
            rule: The WAF rule to validate.

        Returns:
            Validation result with errors and warnings.
        """
        if rule.rule_type not in self._builders:
            result = RuleValidationResult()
            result.add_error(f"No builder available for rule type: {rule.rule_type}")
            return result

        builder = self._builders[rule.rule_type]
        result = builder.validate(rule)

        # Add common validations
        if not rule.name:
            result.add_error("Rule name is required")

        if rule.priority < 0 or rule.priority > 10000:
            result.add_error("Rule priority must be between 0 and 10000")

        if not rule.enabled:
            result.add_warning("Rule is disabled and will not be evaluated")

        return result

    def build_aws_rule(self, rule: WAFRule) -> Dict[str, Any]:
        """Build the AWS WAF rule structure from a WAFRule.

        Args:
            rule: The WAF rule to convert.

        Returns:
            AWS WAF rule dictionary ready for API call.

        Raises:
            ValueError: If no builder exists for the rule type.
        """
        if rule.rule_type not in self._builders:
            raise ValueError(f"No builder available for rule type: {rule.rule_type}")

        builder = self._builders[rule.rule_type]
        statement = builder.build_statement(rule)
        visibility_config = builder._build_visibility_config(rule)

        # For managed rule groups, use OverrideAction instead of Action
        if rule.rule_type == RuleType.BOT_CONTROL:
            return {
                "Name": rule.name,
                "Priority": rule.priority,
                "Statement": statement,
                "OverrideAction": {"None": {}},
                "VisibilityConfig": visibility_config,
            }

        return {
            "Name": rule.name,
            "Priority": rule.priority,
            "Statement": statement,
            "Action": builder._map_action_to_aws(rule.action),
            "VisibilityConfig": visibility_config,
        }

    async def deploy_rule(
        self,
        rule: WAFRule,
        web_acl_id: Optional[str] = None,
    ) -> WAFRule:
        """Deploy a WAF rule to AWS WAF.

        Args:
            rule: The WAF rule to deploy.
            web_acl_id: Web ACL ID. If None, uses config.web_acl_id.

        Returns:
            Updated WAFRule with deployment information.

        Raises:
            ValueError: If validation fails or deployment fails.
        """
        # Validate first
        validation = self.validate_rule(rule)
        if not validation.is_valid:
            raise ValueError(
                f"Rule validation failed: {', '.join(validation.errors)}"
            )

        web_acl_id = web_acl_id or self.config.web_acl_id
        if not web_acl_id:
            raise ValueError("web_acl_id is required for deployment")

        try:
            # Get current Web ACL configuration
            response = self.waf_client.get_web_acl(
                Name=self._get_web_acl_name_from_id(web_acl_id),
                Scope=self.config.web_acl_scope,
                Id=web_acl_id,
            )

            web_acl = response["WebACL"]
            lock_token = response["LockToken"]

            # Build the new rule
            aws_rule = self.build_aws_rule(rule)

            # Add or update the rule in the Web ACL
            existing_rules = list(web_acl.get("Rules", []))
            rule_updated = False

            for i, existing in enumerate(existing_rules):
                if existing["Name"] == rule.name:
                    existing_rules[i] = aws_rule
                    rule_updated = True
                    break

            if not rule_updated:
                existing_rules.append(aws_rule)

            # Sort by priority
            existing_rules.sort(key=lambda r: r["Priority"])

            # Update the Web ACL
            self.waf_client.update_web_acl(
                Name=web_acl["Name"],
                Scope=self.config.web_acl_scope,
                Id=web_acl_id,
                DefaultAction=web_acl["DefaultAction"],
                Rules=existing_rules,
                VisibilityConfig=web_acl["VisibilityConfig"],
                LockToken=lock_token,
            )

            # Update rule metadata
            rule.status = RuleStatus.ACTIVE
            rule.deployed_at = datetime.now(timezone.utc)
            rule.aws_rule_id = f"{web_acl_id}:{rule.name}"

            logger.info(
                "Deployed WAF rule: name=%s, web_acl=%s",
                rule.name,
                web_acl_id,
            )

            return rule

        except (BotoCoreError, ClientError) as e:
            logger.error("Failed to deploy WAF rule: %s", str(e))
            raise ValueError(f"WAF deployment failed: {str(e)}")

    async def delete_rule(
        self,
        rule_id: str,
        web_acl_id: Optional[str] = None,
    ) -> bool:
        """Delete a WAF rule from AWS WAF.

        Args:
            rule_id: The rule ID or name to delete.
            web_acl_id: Web ACL ID. If None, uses config.web_acl_id.

        Returns:
            True if deleted successfully, False otherwise.
        """
        web_acl_id = web_acl_id or self.config.web_acl_id
        if not web_acl_id:
            raise ValueError("web_acl_id is required for deletion")

        try:
            # Get current Web ACL configuration
            response = self.waf_client.get_web_acl(
                Name=self._get_web_acl_name_from_id(web_acl_id),
                Scope=self.config.web_acl_scope,
                Id=web_acl_id,
            )

            web_acl = response["WebACL"]
            lock_token = response["LockToken"]

            # Remove the rule
            existing_rules = [
                r for r in web_acl.get("Rules", [])
                if r["Name"] != rule_id
            ]

            if len(existing_rules) == len(web_acl.get("Rules", [])):
                logger.warning("Rule not found for deletion: %s", rule_id)
                return False

            # Update the Web ACL without the rule
            self.waf_client.update_web_acl(
                Name=web_acl["Name"],
                Scope=self.config.web_acl_scope,
                Id=web_acl_id,
                DefaultAction=web_acl["DefaultAction"],
                Rules=existing_rules,
                VisibilityConfig=web_acl["VisibilityConfig"],
                LockToken=lock_token,
            )

            logger.info(
                "Deleted WAF rule: name=%s, web_acl=%s",
                rule_id,
                web_acl_id,
            )

            return True

        except (BotoCoreError, ClientError) as e:
            logger.error("Failed to delete WAF rule: %s", str(e))
            return False

    def _get_web_acl_name_from_id(self, web_acl_id: str) -> str:
        """Extract Web ACL name from ID or ARN.

        Args:
            web_acl_id: Web ACL ID or ARN.

        Returns:
            Web ACL name.
        """
        # If it's an ARN, extract the name
        if web_acl_id.startswith("arn:aws:wafv2:"):
            # ARN format: arn:aws:wafv2:region:account:scope/webacl/name/id
            parts = web_acl_id.split("/")
            if len(parts) >= 3:
                return parts[-2]

        # Otherwise, try to get it from the API
        try:
            response = self.waf_client.list_web_acls(
                Scope=self.config.web_acl_scope,
            )
            for acl in response.get("WebACLs", []):
                if acl["Id"] == web_acl_id:
                    return acl["Name"]
        except (BotoCoreError, ClientError):
            pass

        # Fallback: assume the ID is the name
        return web_acl_id


__all__ = [
    "BaseRuleBuilder",
    "RateLimitRuleBuilder",
    "GeoBlockRuleBuilder",
    "IPReputationRuleBuilder",
    "SQLInjectionRuleBuilder",
    "XSSRuleBuilder",
    "CustomRegexRuleBuilder",
    "BotControlRuleBuilder",
    "WAFRuleBuilder",
    "RuleValidationResult",
]
