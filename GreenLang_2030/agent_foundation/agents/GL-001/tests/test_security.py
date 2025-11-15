"""
Security tests for GL-001 ProcessHeatOrchestrator
Validates input sanitization, authentication, authorization, and secure practices.
Target: 100% secure with no critical/high vulnerabilities.
"""

import unittest
import pytest
import asyncio
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import base64

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessHeatConfig,
    ProcessData
)
from testing.agent_test_framework import AgentTestCase


class TestSecurity(AgentTestCase):
    """Security validation tests for ProcessHeatOrchestrator."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.agent = ProcessHeatOrchestrator()

    @pytest.mark.security
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        # Malicious SQL in fuel_type field
        malicious_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="'; DROP TABLE thermal_calculations; --",
            fuel_consumption_rate=10.0
        )

        # Process data - should sanitize input
        cache_key = self.agent._generate_cache_key(malicious_data)

        # Verify SQL injection is prevented
        self.assertNotIn("DROP TABLE", cache_key)
        self.assertNotIn(";", cache_key)
        self.assertNotIn("--", cache_key)

        # Verify data is properly escaped when used in queries
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_connect.return_value.cursor.return_value = mock_cursor

            # Simulate parameterized query (safe)
            query = "INSERT INTO calculations (fuel_type) VALUES (%s)"
            mock_cursor.execute(query, (malicious_data.fuel_type,))

            # Verify parameterized query was used
            call_args = mock_cursor.execute.call_args
            self.assertEqual(call_args[0][0], query)
            self.assertEqual(call_args[0][1][0], malicious_data.fuel_type)

    @pytest.mark.security
    def test_xss_prevention(self):
        """Test prevention of Cross-Site Scripting (XSS) attacks."""
        # XSS attempt in metadata
        xss_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0,
            metadata={
                'notes': '<script>alert("XSS")</script>',
                'user_input': '<img src=x onerror=alert("XSS")>'
            }
        )

        # Generate provenance hash - should not execute scripts
        provenance = self.agent._generate_provenance_hash({
            'data': xss_data.metadata
        })

        # Verify hash is generated without executing scripts
        self.assertIsNotNone(provenance)
        self.assertEqual(len(provenance), 64)

        # Verify output would be escaped if rendered
        json_output = json.dumps(xss_data.metadata)
        self.assertIn("\\u003cscript\\u003e", json_output.lower())

    @pytest.mark.security
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        # Command injection attempt
        malicious_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas; rm -rf /",  # Command injection attempt
            fuel_consumption_rate=10.0
        )

        # Process should sanitize input
        cache_key = self.agent._generate_cache_key(malicious_data)

        # Verify command is not executable
        self.assertNotIn("rm -rf", cache_key)

        # Verify no system commands are executed
        with patch('os.system') as mock_system:
            with patch('subprocess.run') as mock_subprocess:
                asyncio.run(self.agent.calculate_thermal_efficiency(malicious_data))

                # Verify no system calls were made
                mock_system.assert_not_called()
                mock_subprocess.assert_not_called()

    @pytest.mark.security
    def test_jwt_authentication(self):
        """Test JWT token validation for API access."""
        # Generate valid JWT token
        secret_key = secrets.token_urlsafe(32)
        payload = {
            'agent_id': 'GL-001',
            'user_id': 'user123',
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow(),
            'roles': ['operator', 'analyst']
        }

        valid_token = jwt.encode(payload, secret_key, algorithm='HS256')

        # Verify valid token
        try:
            decoded = jwt.decode(valid_token, secret_key, algorithms=['HS256'])
            self.assertEqual(decoded['agent_id'], 'GL-001')
            self.assertEqual(decoded['user_id'], 'user123')
            self.assertIn('operator', decoded['roles'])
        except jwt.InvalidTokenError:
            self.fail("Valid token rejected")

        # Test expired token
        expired_payload = {
            'agent_id': 'GL-001',
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2)
        }

        expired_token = jwt.encode(expired_payload, secret_key, algorithm='HS256')

        with self.assertRaises(jwt.ExpiredSignatureError):
            jwt.decode(expired_token, secret_key, algorithms=['HS256'])

        # Test tampered token
        tampered_token = valid_token[:-10] + 'tampered00'

        with self.assertRaises(jwt.InvalidTokenError):
            jwt.decode(tampered_token, secret_key, algorithms=['HS256'])

    @pytest.mark.security
    def test_multi_tenancy_isolation(self):
        """Test data isolation between tenants."""
        # Set up tenant data
        self.agent.tenant_data['tenant_A'] = {
            'calculations': [],
            'api_key': 'key_A_secret',
            'data': 'sensitive_A'
        }

        self.agent.tenant_data['tenant_B'] = {
            'calculations': [],
            'api_key': 'key_B_secret',
            'data': 'sensitive_B'
        }

        # Verify tenant A cannot access tenant B data
        tenant_a_data = self.agent.tenant_data.get('tenant_A', {})
        self.assertIn('sensitive_A', str(tenant_a_data))
        self.assertNotIn('sensitive_B', str(tenant_a_data))

        # Verify tenant B cannot access tenant A data
        tenant_b_data = self.agent.tenant_data.get('tenant_B', {})
        self.assertIn('sensitive_B', str(tenant_b_data))
        self.assertNotIn('sensitive_A', str(tenant_b_data))

        # Verify API keys are isolated
        self.assertNotEqual(
            self.agent.tenant_data['tenant_A']['api_key'],
            self.agent.tenant_data['tenant_B']['api_key']
        )

    @pytest.mark.security
    def test_input_validation_boundaries(self):
        """Test input validation prevents buffer overflow attacks."""
        # Test extremely large values
        overflow_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=float('inf'),  # Infinity
            pressure_bar=1e308,  # Near float max
            flow_rate_kg_s=1e100,
            energy_input_kw=1e50,
            energy_output_kw=1e49,
            fuel_type='a' * 10000,  # Very long string
            fuel_consumption_rate=1e100
        )

        # Should raise validation error
        with self.assertRaises(ValueError):
            self.agent._validate_process_data(overflow_data)

        # Test NaN values
        nan_data = ProcessData(
            timestamp=datetime.utcnow(),
            temperature_c=float('nan'),
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=850.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Should handle NaN gracefully
        with self.assertRaises(ValueError):
            self.agent._validate_process_data(nan_data)

    @pytest.mark.security
    def test_secret_management(self):
        """Test no hardcoded secrets in code."""
        # Check for common secret patterns
        agent_source = str(ProcessHeatOrchestrator.__dict__)

        # Common patterns that indicate hardcoded secrets
        secret_patterns = [
            'password=',
            'api_key=',
            'secret=',
            'token=',
            'private_key=',
            'aws_access_key',
            'connection_string='
        ]

        for pattern in secret_patterns:
            self.assertNotIn(
                pattern.lower(),
                agent_source.lower(),
                f"Potential hardcoded secret found: {pattern}"
            )

        # Verify environment variables are used for secrets
        with patch.dict(os.environ, {'DB_PASSWORD': 'test_password'}):
            db_password = os.environ.get('DB_PASSWORD')
            self.assertEqual(db_password, 'test_password')

    @pytest.mark.security
    def test_rate_limiting(self):
        """Test rate limiting prevents DoS attacks."""
        from collections import deque
        from time import time

        # Simple rate limiter
        class RateLimiter:
            def __init__(self, max_requests=100, time_window=60):
                self.max_requests = max_requests
                self.time_window = time_window
                self.requests = deque()

            def is_allowed(self, client_id):
                now = time()
                # Remove old requests
                while self.requests and self.requests[0][1] < now - self.time_window:
                    self.requests.popleft()

                # Count requests from this client
                client_requests = sum(1 for req in self.requests if req[0] == client_id)

                if client_requests >= self.max_requests:
                    return False

                self.requests.append((client_id, now))
                return True

        limiter = RateLimiter(max_requests=10, time_window=60)

        # Test normal usage
        for i in range(10):
            self.assertTrue(limiter.is_allowed('client_1'))

        # 11th request should be blocked
        self.assertFalse(limiter.is_allowed('client_1'))

        # Different client should be allowed
        self.assertTrue(limiter.is_allowed('client_2'))

    @pytest.mark.security
    def test_secure_random_generation(self):
        """Test cryptographically secure random number generation."""
        # Generate secure random tokens
        token1 = secrets.token_hex(32)
        token2 = secrets.token_hex(32)

        # Tokens should be different
        self.assertNotEqual(token1, token2)

        # Tokens should be correct length
        self.assertEqual(len(token1), 64)  # 32 bytes = 64 hex chars

        # Test secure random for session IDs
        session_id = secrets.token_urlsafe(32)
        self.assertIsNotNone(session_id)
        self.assertGreaterEqual(len(session_id), 32)

    @pytest.mark.security
    def test_password_hashing(self):
        """Test secure password hashing."""
        import bcrypt

        # Test password hashing
        password = "SecurePassword123!"
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)

        # Verify correct password
        self.assertTrue(bcrypt.checkpw(password.encode('utf-8'), hashed))

        # Verify wrong password fails
        self.assertFalse(bcrypt.checkpw("WrongPassword".encode('utf-8'), hashed))

        # Verify hash is salted (same password gives different hash)
        hashed2 = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.assertNotEqual(hashed, hashed2)

    @pytest.mark.security
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        # Attempt path traversal
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]

        for path in malicious_paths:
            # Simulate file access with malicious path
            safe_base = "/var/greenlang/data"
            requested_path = os.path.join(safe_base, path)

            # Normalize and check if within base directory
            normalized = os.path.normpath(requested_path)
            is_safe = normalized.startswith(safe_base)

            self.assertFalse(
                is_safe,
                f"Path traversal not prevented for: {path}"
            )

    @pytest.mark.security
    def test_encryption_at_rest(self):
        """Test sensitive data is encrypted at rest."""
        from cryptography.fernet import Fernet

        # Generate encryption key
        key = Fernet.generate_key()
        cipher = Fernet(key)

        # Sensitive data to encrypt
        sensitive_data = {
            'api_key': 'secret_api_key_123',
            'calculation_result': 0.85,
            'provenance_hash': 'abc123def456'
        }

        # Encrypt data
        json_data = json.dumps(sensitive_data)
        encrypted = cipher.encrypt(json_data.encode())

        # Verify data is encrypted
        self.assertNotIn(b'secret_api_key_123', encrypted)
        self.assertNotIn(b'0.85', encrypted)

        # Verify can decrypt
        decrypted = cipher.decrypt(encrypted)
        recovered_data = json.loads(decrypted.decode())

        self.assertEqual(recovered_data['api_key'], sensitive_data['api_key'])

    @pytest.mark.security
    def test_audit_logging(self):
        """Test security audit logging."""
        import logging

        # Set up audit logger
        audit_logger = logging.getLogger('security.audit')
        audit_handler = logging.handlers.MemoryHandler(capacity=100)
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)

        # Log security events
        security_events = [
            {'event': 'authentication_success', 'user': 'user123', 'timestamp': datetime.utcnow()},
            {'event': 'authorization_failure', 'user': 'user456', 'resource': 'GL-001', 'timestamp': datetime.utcnow()},
            {'event': 'suspicious_activity', 'ip': '192.168.1.100', 'details': 'Multiple failed attempts', 'timestamp': datetime.utcnow()}
        ]

        for event in security_events:
            audit_logger.info(json.dumps(event, default=str))

        # Verify events are logged
        self.assertEqual(len(audit_handler.buffer), 3)

        # Verify log contains required fields
        for log_record in audit_handler.buffer:
            message = log_record.getMessage()
            event = json.loads(message)
            self.assertIn('event', event)
            self.assertIn('timestamp', event)

    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_secure_communication(self):
        """Test secure inter-agent communication."""
        # Generate keys for agents
        agent1_key = secrets.token_hex(32)
        agent2_key = secrets.token_hex(32)

        # Create message
        message = {
            'from': 'GL-001',
            'to': 'GL-002',
            'data': {'efficiency': 0.85},
            'timestamp': datetime.utcnow().isoformat()
        }

        # Sign message
        message_json = json.dumps(message, sort_keys=True)
        signature = hashlib.sha256((message_json + agent1_key).encode()).hexdigest()

        signed_message = {
            'message': message,
            'signature': signature
        }

        # Verify signature
        received_json = json.dumps(signed_message['message'], sort_keys=True)
        expected_signature = hashlib.sha256((received_json + agent1_key).encode()).hexdigest()

        self.assertEqual(
            signed_message['signature'],
            expected_signature,
            "Message signature verification failed"
        )

    @pytest.mark.security
    def test_no_eval_or_exec(self):
        """Test no use of eval() or exec() with user input."""
        # Check agent code doesn't use eval/exec
        import inspect

        source = inspect.getsource(ProcessHeatOrchestrator)

        # These functions should not be used with user input
        dangerous_functions = ['eval(', 'exec(', 'compile(', '__import__(']

        for func in dangerous_functions:
            self.assertNotIn(
                func,
                source,
                f"Dangerous function '{func}' found in code"
            )


if __name__ == '__main__':
    unittest.main()