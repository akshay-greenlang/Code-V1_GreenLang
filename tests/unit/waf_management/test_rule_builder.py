"""
Unit tests for WAFRuleBuilder.

Tests rule building for rate limiting, geo-blocking, IP reputation,
SQL injection, XSS, custom regex, and bot control rules.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.waf_management.rule_builder import (
    WAFRuleBuilder,
    BaseRuleBuilder,
    RateLimitRuleBuilder,
    GeoBlockRuleBuilder,
    IPReputationRuleBuilder,
    SQLInjectionRuleBuilder,
    XSSRuleBuilder,
    CustomRegexRuleBuilder,
    BotControlRuleBuilder,
)
from greenlang.infrastructure.waf_management.models import (
    WAFRule,
    RuleType,
    RuleAction,
    RulePriority,
)


class TestWAFRuleBuilderFactory:
    """Test WAFRuleBuilder factory pattern."""

    def test_factory_creates_rate_limit_builder(self, rule_builder_config):
        """Test factory creates rate limit rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.RATE_LIMIT,
            config=rule_builder_config,
        )

        assert isinstance(builder, RateLimitRuleBuilder)

    def test_factory_creates_geo_block_builder(self, rule_builder_config):
        """Test factory creates geo block rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.GEO_BLOCK,
            config=rule_builder_config,
        )

        assert isinstance(builder, GeoBlockRuleBuilder)

    def test_factory_creates_ip_reputation_builder(self, rule_builder_config):
        """Test factory creates IP reputation rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.IP_REPUTATION,
            config=rule_builder_config,
        )

        assert isinstance(builder, IPReputationRuleBuilder)

    def test_factory_creates_sqli_builder(self, rule_builder_config):
        """Test factory creates SQL injection rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.SQL_INJECTION,
            config=rule_builder_config,
        )

        assert isinstance(builder, SQLInjectionRuleBuilder)

    def test_factory_creates_xss_builder(self, rule_builder_config):
        """Test factory creates XSS rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.XSS,
            config=rule_builder_config,
        )

        assert isinstance(builder, XSSRuleBuilder)

    def test_factory_creates_custom_regex_builder(self, rule_builder_config):
        """Test factory creates custom regex rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.CUSTOM_REGEX,
            config=rule_builder_config,
        )

        assert isinstance(builder, CustomRegexRuleBuilder)

    def test_factory_creates_bot_control_builder(self, rule_builder_config):
        """Test factory creates bot control rule builder."""
        builder = WAFRuleBuilder.create(
            RuleType.BOT_CONTROL,
            config=rule_builder_config,
        )

        assert isinstance(builder, BotControlRuleBuilder)

    def test_factory_raises_for_unknown_type(self, rule_builder_config):
        """Test factory raises error for unknown rule type."""
        with pytest.raises(ValueError):
            WAFRuleBuilder.create("unknown_type", config=rule_builder_config)


class TestRateLimitRuleBuilder:
    """Test RateLimitRuleBuilder."""

    def test_build_basic_rate_limit_rule(self, rule_builder_config):
        """Test building basic rate limit rule."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="API Rate Limit",
            description="Limit API requests",
            limit=100,
            window_seconds=60,
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.RATE_LIMIT
        assert rule.parameters["limit"] == 100
        assert rule.parameters["window_seconds"] == 60

    def test_build_rate_limit_with_path_pattern(self, rule_builder_config):
        """Test building rate limit rule with path pattern."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Login Rate Limit",
            description="Limit login attempts",
            limit=5,
            window_seconds=300,
            path_pattern="/api/v1/auth/login",
        )

        assert rule.conditions["path_pattern"] == "/api/v1/auth/login"

    def test_build_rate_limit_with_key_by_ip(self, rule_builder_config):
        """Test building rate limit rule keyed by IP."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="IP Rate Limit",
            description="Limit by IP",
            limit=1000,
            window_seconds=60,
            key="ip_address",
        )

        assert rule.parameters["key"] == "ip_address"

    def test_build_rate_limit_with_key_by_user(self, rule_builder_config):
        """Test building rate limit rule keyed by user."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="User Rate Limit",
            description="Limit by user",
            limit=100,
            window_seconds=60,
            key="user_id",
        )

        assert rule.parameters["key"] == "user_id"

    def test_build_rate_limit_with_custom_action(self, rule_builder_config):
        """Test building rate limit rule with custom action."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Soft Rate Limit",
            description="Log rate limit violations",
            limit=100,
            window_seconds=60,
            action=RuleAction.LOG,
        )

        assert rule.action == RuleAction.LOG

    def test_rate_limit_uses_config_defaults(self, rule_builder_config):
        """Test rate limit builder uses config defaults."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Default Rate Limit",
            description="Uses defaults",
        )

        assert rule.parameters["limit"] == rule_builder_config.rate_limit_defaults["limit"]
        assert rule.parameters["window_seconds"] == rule_builder_config.rate_limit_defaults["window_seconds"]


class TestGeoBlockRuleBuilder:
    """Test GeoBlockRuleBuilder."""

    def test_build_geo_block_rule(self, rule_builder_config):
        """Test building geo block rule."""
        builder = GeoBlockRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Block High Risk Countries",
            description="Block traffic from high-risk countries",
            country_codes=["CN", "RU", "KP"],
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.GEO_BLOCK
        assert "CN" in rule.conditions["country_codes"]
        assert "RU" in rule.conditions["country_codes"]

    def test_build_geo_block_with_allow_list(self, rule_builder_config):
        """Test building geo block rule with allow list."""
        builder = GeoBlockRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Block with Exceptions",
            description="Block with allow list",
            country_codes=["CN"],
            allow_list=["partner.cn.example.com", "10.0.0.0/8"],
        )

        assert "partner.cn.example.com" in rule.parameters["allow_list"]

    def test_build_geo_block_validates_country_codes(self, rule_builder_config):
        """Test geo block builder validates country codes."""
        builder = GeoBlockRuleBuilder(config=rule_builder_config)

        with pytest.raises(ValueError):
            builder.build(
                name="Invalid Codes",
                description="Invalid country codes",
                country_codes=["INVALID", "XX"],
            )

    def test_build_geo_allow_rule(self, rule_builder_config):
        """Test building geo allow rule (inverse)."""
        builder = GeoBlockRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Allow Only US/CA",
            description="Allow only US and Canada",
            country_codes=["US", "CA"],
            mode="allow",
        )

        # In allow mode, action should allow these countries
        assert rule.conditions["country_codes"] == ["US", "CA"]


class TestIPReputationRuleBuilder:
    """Test IPReputationRuleBuilder."""

    def test_build_ip_reputation_rule(self, rule_builder_config):
        """Test building IP reputation rule."""
        builder = IPReputationRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Block Bad IPs",
            description="Block IPs with bad reputation",
            reputation_threshold=30,
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.IP_REPUTATION
        assert rule.conditions["reputation_threshold"] == 30

    def test_build_ip_reputation_with_feeds(self, rule_builder_config):
        """Test building IP reputation rule with specific feeds."""
        builder = IPReputationRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Use Custom Feeds",
            description="Use specific threat feeds",
            reputation_threshold=50,
            feeds=["spamhaus", "abuseipdb"],
        )

        assert "spamhaus" in rule.conditions["feeds"]
        assert "abuseipdb" in rule.conditions["feeds"]

    def test_build_ip_reputation_with_whitelist(self, rule_builder_config):
        """Test building IP reputation rule with whitelist."""
        builder = IPReputationRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="With Whitelist",
            description="Whitelist internal IPs",
            reputation_threshold=30,
            whitelist=["10.0.0.0/8", "192.168.0.0/16"],
        )

        assert "10.0.0.0/8" in rule.parameters["whitelist"]


class TestSQLInjectionRuleBuilder:
    """Test SQLInjectionRuleBuilder."""

    def test_build_sqli_rule_high_sensitivity(self, rule_builder_config):
        """Test building SQL injection rule with high sensitivity."""
        builder = SQLInjectionRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="High Sensitivity SQLi",
            description="Detect SQL injection (high)",
            sensitivity="high",
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.SQL_INJECTION
        assert rule.parameters["sensitivity"] == "high"

    def test_build_sqli_rule_low_sensitivity(self, rule_builder_config):
        """Test building SQL injection rule with low sensitivity."""
        builder = SQLInjectionRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Low Sensitivity SQLi",
            description="Detect SQL injection (low)",
            sensitivity="low",
        )

        assert rule.parameters["sensitivity"] == "low"

    def test_build_sqli_rule_has_standard_patterns(self, rule_builder_config):
        """Test SQL injection rule includes standard patterns."""
        builder = SQLInjectionRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Standard SQLi",
            description="Standard SQL injection detection",
        )

        # Should have patterns for common SQL injection techniques
        patterns = rule.conditions.get("patterns", [])
        assert len(patterns) > 0

    def test_build_sqli_rule_custom_patterns(self, rule_builder_config):
        """Test SQL injection rule with custom patterns."""
        builder = SQLInjectionRuleBuilder(config=rule_builder_config)

        custom_patterns = [r"CUSTOM_PATTERN_1", r"CUSTOM_PATTERN_2"]

        rule = builder.build(
            name="Custom SQLi",
            description="Custom patterns",
            additional_patterns=custom_patterns,
        )

        patterns = rule.conditions.get("patterns", [])
        assert "CUSTOM_PATTERN_1" in patterns or any("CUSTOM" in p for p in patterns)

    def test_build_sqli_rule_specifies_targets(self, rule_builder_config):
        """Test SQL injection rule specifies inspection targets."""
        builder = SQLInjectionRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Target Body Only",
            description="Only inspect body",
            targets=["body"],
        )

        assert rule.conditions["targets"] == ["body"]


class TestXSSRuleBuilder:
    """Test XSSRuleBuilder."""

    def test_build_xss_rule_high_sensitivity(self, rule_builder_config):
        """Test building XSS rule with high sensitivity."""
        builder = XSSRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="High Sensitivity XSS",
            description="Detect XSS (high)",
            sensitivity="high",
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.XSS
        assert rule.parameters["sensitivity"] == "high"

    def test_build_xss_rule_has_standard_patterns(self, rule_builder_config):
        """Test XSS rule includes standard patterns."""
        builder = XSSRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Standard XSS",
            description="Standard XSS detection",
        )

        patterns = rule.conditions.get("patterns", [])
        assert len(patterns) > 0

    def test_build_xss_rule_includes_script_detection(self, rule_builder_config):
        """Test XSS rule includes script tag detection."""
        builder = XSSRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Script Detection",
            description="Detect script tags",
        )

        patterns = rule.conditions.get("patterns", [])
        # Should detect script tags
        assert any("script" in p.lower() for p in patterns)

    def test_build_xss_rule_with_encoding_output(self, rule_builder_config):
        """Test XSS rule with output encoding option."""
        builder = XSSRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Encode Output",
            description="Enable output encoding",
            encode_output=True,
        )

        assert rule.parameters.get("encode_output") is True


class TestCustomRegexRuleBuilder:
    """Test CustomRegexRuleBuilder."""

    def test_build_custom_regex_rule(self, rule_builder_config):
        """Test building custom regex rule."""
        builder = CustomRegexRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Block .env Files",
            description="Block access to .env files",
            pattern=r"\.env$",
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.CUSTOM_REGEX
        assert rule.conditions["pattern"] == r"\.env$"

    def test_build_custom_regex_with_targets(self, rule_builder_config):
        """Test custom regex rule with specific targets."""
        builder = CustomRegexRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Path Pattern",
            description="Match path only",
            pattern=r"/admin/.*",
            targets=["path"],
        )

        assert rule.conditions["targets"] == ["path"]

    def test_build_custom_regex_case_insensitive(self, rule_builder_config):
        """Test custom regex rule with case insensitivity."""
        builder = CustomRegexRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Case Insensitive",
            description="Case insensitive match",
            pattern=r"\.bak$",
            case_insensitive=True,
        )

        assert rule.parameters["case_insensitive"] is True

    def test_build_custom_regex_validates_pattern(self, rule_builder_config):
        """Test custom regex rule validates pattern."""
        builder = CustomRegexRuleBuilder(config=rule_builder_config)

        with pytest.raises(ValueError):
            builder.build(
                name="Invalid Regex",
                description="Invalid regex pattern",
                pattern=r"[invalid(regex",  # Invalid regex
            )


class TestBotControlRuleBuilder:
    """Test BotControlRuleBuilder."""

    def test_build_bot_control_rule(self, rule_builder_config):
        """Test building bot control rule."""
        builder = BotControlRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Bot Control",
            description="Control bot traffic",
            bot_categories=["scraper", "crawler"],
        )

        assert isinstance(rule, WAFRule)
        assert rule.rule_type == RuleType.BOT_CONTROL
        assert "scraper" in rule.conditions["bot_categories"]

    def test_build_bot_control_with_exempt_bots(self, rule_builder_config):
        """Test bot control rule with exempt good bots."""
        builder = BotControlRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Exempt Good Bots",
            description="Allow search engine bots",
            bot_categories=["scraper"],
            exempt_bots=["googlebot", "bingbot"],
        )

        assert "googlebot" in rule.conditions["exempt_bots"]

    def test_build_bot_control_with_challenge(self, rule_builder_config):
        """Test bot control rule with challenge action."""
        builder = BotControlRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="Challenge Bots",
            description="Challenge suspected bots",
            bot_categories=["automation"],
            action=RuleAction.CHALLENGE,
            challenge_type="captcha",
        )

        assert rule.action == RuleAction.CHALLENGE
        assert rule.parameters["challenge_type"] == "captcha"


class TestBaseRuleBuilder:
    """Test BaseRuleBuilder abstract class."""

    def test_base_builder_is_abstract(self):
        """Test BaseRuleBuilder cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseRuleBuilder()

    def test_all_builders_inherit_from_base(self, rule_builder_config):
        """Test all builders inherit from BaseRuleBuilder."""
        builder_classes = [
            RateLimitRuleBuilder,
            GeoBlockRuleBuilder,
            IPReputationRuleBuilder,
            SQLInjectionRuleBuilder,
            XSSRuleBuilder,
            CustomRegexRuleBuilder,
            BotControlRuleBuilder,
        ]

        for builder_class in builder_classes:
            assert issubclass(builder_class, BaseRuleBuilder)


class TestRuleValidation:
    """Test rule validation across builders."""

    @pytest.mark.parametrize("rule_type", [
        RuleType.RATE_LIMIT,
        RuleType.GEO_BLOCK,
        RuleType.SQL_INJECTION,
        RuleType.XSS,
        RuleType.CUSTOM_REGEX,
        RuleType.BOT_CONTROL,
    ])
    def test_built_rules_have_required_fields(self, rule_builder_config, rule_type):
        """Test all built rules have required fields."""
        builder = WAFRuleBuilder.create(rule_type, config=rule_builder_config)

        # Build with minimal required params
        if rule_type == RuleType.GEO_BLOCK:
            rule = builder.build(
                name="Test Rule",
                description="Test",
                country_codes=["US"],
            )
        elif rule_type == RuleType.CUSTOM_REGEX:
            rule = builder.build(
                name="Test Rule",
                description="Test",
                pattern=r"test",
            )
        else:
            rule = builder.build(
                name="Test Rule",
                description="Test",
            )

        # All rules must have these fields
        assert rule.rule_id is not None
        assert rule.name is not None
        assert rule.rule_type == rule_type
        assert rule.action is not None
        assert rule.priority is not None
        assert rule.enabled is True
        assert rule.created_at is not None

    def test_rule_ids_are_unique(self, rule_builder_config):
        """Test generated rule IDs are unique."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rules = [
            builder.build(name=f"Rule {i}", description="Test")
            for i in range(100)
        ]

        rule_ids = [r.rule_id for r in rules]
        assert len(rule_ids) == len(set(rule_ids))  # All unique


class TestRulePriority:
    """Test rule priority handling."""

    def test_sqli_rule_default_critical_priority(self, rule_builder_config):
        """Test SQL injection rules default to critical priority."""
        builder = SQLInjectionRuleBuilder(config=rule_builder_config)

        rule = builder.build(name="SQLi", description="Test")

        assert rule.priority == RulePriority.CRITICAL

    def test_xss_rule_default_critical_priority(self, rule_builder_config):
        """Test XSS rules default to critical priority."""
        builder = XSSRuleBuilder(config=rule_builder_config)

        rule = builder.build(name="XSS", description="Test")

        assert rule.priority == RulePriority.CRITICAL

    def test_rate_limit_rule_default_medium_priority(self, rule_builder_config):
        """Test rate limit rules default to medium priority."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(name="Rate Limit", description="Test")

        assert rule.priority == RulePriority.MEDIUM

    def test_custom_priority_override(self, rule_builder_config):
        """Test priority can be overridden."""
        builder = RateLimitRuleBuilder(config=rule_builder_config)

        rule = builder.build(
            name="High Priority Rate Limit",
            description="Test",
            priority=RulePriority.HIGH,
        )

        assert rule.priority == RulePriority.HIGH
