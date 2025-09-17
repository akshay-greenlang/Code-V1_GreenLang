"""
Security Unit Tests
===================

This module contains unit tests that verify the security hardening
implemented for the default-deny gate:

1. Policy Default-Deny Tests (test_default_deny_policy.py)
   - Test A: No policies loaded ⇒ deny
   - Test B: Policy returns allow=false ⇒ deny
   - Test C: Policy returns allow=true ⇒ allow
   - Test D: OPA error/timeout ⇒ deny

2. Signature Verification Tests (test_signature_verification.py)
   - Test E: Installing pack without .sig ⇒ fails
   - Test F: Installing pack with invalid .sig ⇒ fails
   - Test G: Installing pack with valid signature ⇒ succeeds
   - Test H: Installing unsigned with --allow-unsigned ⇒ succeeds with WARNING

3. Network Security Tests (test_network_security.py)
   - Test I: HTTP URL install ⇒ fails by default
   - Test J: HTTPS with bad cert ⇒ fails
   - Test K: HTTPS good cert ⇒ succeeds
   - Test L: HTTP with --insecure-transport and GL_DEBUG_INSECURE=1 ⇒ warn + allow

4. Capability Tests (test_capabilities.py)
   - Test M: Pack tries network while net:false ⇒ denied
   - Test N: Pack tries file read while fs:false ⇒ denied
   - Test O: Same actions when capability is true ⇒ allowed

SECURITY GATE: ✅ Unit tests for "deny by default"
"""