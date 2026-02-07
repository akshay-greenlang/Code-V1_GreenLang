# -*- coding: utf-8 -*-
"""
Security Training API - SEC-010

FastAPI router, schemas, and endpoints for the security training REST API.

Provides:
    - REST API endpoints for training management and assessment
    - Phishing campaign management endpoints
    - Security score and leaderboard endpoints
    - Request/response schemas with full validation

Endpoints:
    Training:
        GET  /api/v1/secops/training/courses          - List courses
        GET  /api/v1/secops/training/courses/{id}     - Get course content
        GET  /api/v1/secops/training/my-progress      - User's progress
        GET  /api/v1/secops/training/my-curriculum    - User's required training
        POST /api/v1/secops/training/courses/{id}/start     - Start course
        POST /api/v1/secops/training/courses/{id}/complete  - Mark complete
        POST /api/v1/secops/training/courses/{id}/assessment - Submit assessment
        GET  /api/v1/secops/training/certificates     - User's certificates
        GET  /api/v1/secops/training/certificates/{code}/verify - Verify certificate
        GET  /api/v1/secops/training/team-compliance  - Team stats (manager)

    Phishing:
        POST /api/v1/secops/phishing/campaigns           - Create campaign
        GET  /api/v1/secops/phishing/campaigns           - List campaigns
        GET  /api/v1/secops/phishing/campaigns/{id}      - Get campaign
        PUT  /api/v1/secops/phishing/campaigns/{id}      - Update campaign
        POST /api/v1/secops/phishing/campaigns/{id}/send - Send emails
        GET  /api/v1/secops/phishing/campaigns/{id}/metrics - Campaign metrics
        POST /api/v1/secops/phishing/track/{campaign_id}/{user_id}/open  - Track open
        POST /api/v1/secops/phishing/track/{campaign_id}/{user_id}/click - Track click

    Security Score:
        GET  /api/v1/secops/security-score            - User's security score
        GET  /api/v1/secops/security-score/leaderboard - Team leaderboard
"""

from __future__ import annotations

__all__: list[str] = []
