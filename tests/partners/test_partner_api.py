# -*- coding: utf-8 -*-
"""
Tests for Partner API

Comprehensive tests for partner registration, authentication,
API key management, and rate limiting.
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import secrets

from greenlang.partners.api import (
from greenlang.determinism import DeterministicClock
    app,
    Base,
    PartnerModel,
    APIKeyModel,
    PartnerTier,
    PartnerStatus,
    APIKeyStatus,
    hash_password,
    verify_password,
    hash_api_key,
    generate_api_key,
    get_db,
)


# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_partners.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(setup_database):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def db_session():
    """Create database session for tests"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def sample_partner(db_session):
    """Create sample partner for testing"""
    partner = PartnerModel(
        id="partner_test_123",
        name="Test Partner",
        email="test@example.com",
        website="https://example.com",
        tier=PartnerTier.FREE,
        status=PartnerStatus.ACTIVE,
        api_quota=100,
        password_hash=hash_password("testpassword123"),
        webhook_secret=secrets.token_hex(32)
    )
    db_session.add(partner)
    db_session.commit()
    db_session.refresh(partner)
    return partner


@pytest.fixture
def auth_token(client, sample_partner):
    """Get authentication token for testing"""
    response = client.post(
        "/api/partners/login",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    return response.json()["access_token"]


class TestPartnerRegistration:
    """Tests for partner registration"""

    def test_register_partner_success(self, client):
        """Test successful partner registration"""
        response = client.post(
            "/api/partners/register",
            json={
                "name": "New Partner",
                "email": "newpartner@example.com",
                "password": "SecurePass123!",
                "website": "https://newpartner.com",
                "tier": "FREE"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Partner"
        assert data["email"] == "newpartner@example.com"
        assert data["tier"] == "FREE"
        assert data["status"] == "PENDING"
        assert "id" in data

    def test_register_partner_duplicate_email(self, client, sample_partner):
        """Test registration with duplicate email"""
        response = client.post(
            "/api/partners/register",
            json={
                "name": "Duplicate",
                "email": "test@example.com",  # Same as sample_partner
                "password": "password123",
                "tier": "FREE"
            }
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_register_partner_weak_password(self, client):
        """Test registration with weak password"""
        response = client.post(
            "/api/partners/register",
            json={
                "name": "Test",
                "email": "weak@example.com",
                "password": "123",  # Too short
                "tier": "FREE"
            }
        )

        assert response.status_code == 422

    def test_register_partner_invalid_email(self, client):
        """Test registration with invalid email"""
        response = client.post(
            "/api/partners/register",
            json={
                "name": "Test",
                "email": "notanemail",
                "password": "password123",
                "tier": "FREE"
            }
        )

        assert response.status_code == 422


class TestAuthentication:
    """Tests for authentication"""

    def test_login_success(self, client, sample_partner):
        """Test successful login"""
        response = client.post(
            "/api/partners/login",
            json={
                "email": "test@example.com",
                "password": "testpassword123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_wrong_password(self, client, sample_partner):
        """Test login with wrong password"""
        response = client.post(
            "/api/partners/login",
            json={
                "email": "test@example.com",
                "password": "wrongpassword"
            }
        )

        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user"""
        response = client.post(
            "/api/partners/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123"
            }
        )

        assert response.status_code == 401

    def test_get_current_partner(self, client, auth_token):
        """Test getting current partner details"""
        response = client.get(
            "/api/partners/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test Partner"

    def test_get_current_partner_no_token(self, client):
        """Test getting current partner without token"""
        response = client.get("/api/partners/me")

        assert response.status_code == 401


class TestPartnerManagement:
    """Tests for partner management"""

    def test_get_partner_by_id(self, client, auth_token, sample_partner):
        """Test getting partner by ID"""
        response = client.get(
            f"/api/partners/{sample_partner.id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_partner.id
        assert data["name"] == sample_partner.name

    def test_get_partner_unauthorized(self, client, auth_token):
        """Test getting another partner's details"""
        response = client.get(
            "/api/partners/other_partner_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 403

    def test_update_partner(self, client, auth_token, sample_partner):
        """Test updating partner details"""
        response = client.put(
            f"/api/partners/{sample_partner.id}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "name": "Updated Partner Name",
                "webhook_url": "https://webhook.example.com/hook"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Partner Name"
        assert data["webhook_url"] == "https://webhook.example.com/hook"

    def test_update_partner_duplicate_email(self, client, auth_token, sample_partner, db_session):
        """Test updating partner with duplicate email"""
        # Create another partner
        other_partner = PartnerModel(
            id="partner_other",
            name="Other Partner",
            email="other@example.com",
            tier=PartnerTier.FREE,
            status=PartnerStatus.ACTIVE,
            api_quota=100,
            password_hash=hash_password("password")
        )
        db_session.add(other_partner)
        db_session.commit()

        response = client.put(
            f"/api/partners/{sample_partner.id}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"email": "other@example.com"}
        )

        assert response.status_code == 400

    def test_delete_partner(self, client, auth_token, sample_partner):
        """Test deleting partner account"""
        response = client.delete(
            f"/api/partners/{sample_partner.id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 204


class TestAPIKeyManagement:
    """Tests for API key management"""

    def test_create_api_key(self, client, auth_token, sample_partner):
        """Test creating API key"""
        response = client.post(
            f"/api/partners/{sample_partner.id}/api-keys",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "name": "Test API Key",
                "scopes": ["read", "write"]
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test API Key"
        assert "key" in data  # Full key only shown on creation
        assert data["key"].startswith("gl_")
        assert data["scopes"] == ["read", "write"]
        assert data["status"] == "ACTIVE"

    def test_create_api_key_with_expiration(self, client, auth_token, sample_partner):
        """Test creating API key with expiration"""
        response = client.post(
            f"/api/partners/{sample_partner.id}/api-keys",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "name": "Expiring Key",
                "scopes": ["read"],
                "expires_in_days": 30
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["expires_at"] is not None

    def test_create_api_key_exceeds_limit(self, client, auth_token, sample_partner, db_session):
        """Test creating API key when limit exceeded"""
        # Create maximum allowed keys (FREE tier = 1)
        api_key = APIKeyModel(
            id="key_1",
            partner_id=sample_partner.id,
            key_hash=hash_api_key("test_key"),
            key_prefix="gl_test",
            name="Existing Key",
            status=APIKeyStatus.ACTIVE,
            scopes=["read"]
        )
        db_session.add(api_key)
        db_session.commit()

        response = client.post(
            f"/api/partners/{sample_partner.id}/api-keys",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"name": "Too Many Keys", "scopes": ["read"]}
        )

        assert response.status_code == 400
        assert "Maximum API keys" in response.json()["detail"]

    def test_list_api_keys(self, client, auth_token, sample_partner, db_session):
        """Test listing API keys"""
        # Create some API keys
        for i in range(3):
            key = APIKeyModel(
                id=f"key_{i}",
                partner_id=sample_partner.id,
                key_hash=hash_api_key(f"key_{i}"),
                key_prefix=f"gl_key_{i}",
                name=f"Key {i}",
                status=APIKeyStatus.ACTIVE,
                scopes=["read"]
            )
            db_session.add(key)
        db_session.commit()

        response = client.get(
            f"/api/partners/{sample_partner.id}/api-keys",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        # Keys should not include full key value
        for key in data:
            assert "key" not in key
            assert "key_prefix" in key

    def test_revoke_api_key(self, client, auth_token, sample_partner, db_session):
        """Test revoking API key"""
        # Create API key
        api_key = APIKeyModel(
            id="key_revoke",
            partner_id=sample_partner.id,
            key_hash=hash_api_key("test_key"),
            key_prefix="gl_test",
            name="Key to Revoke",
            status=APIKeyStatus.ACTIVE,
            scopes=["read"]
        )
        db_session.add(api_key)
        db_session.commit()

        response = client.delete(
            f"/api/partners/{sample_partner.id}/api-keys/key_revoke",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 204

        # Verify key is revoked
        db_session.refresh(api_key)
        assert api_key.status == APIKeyStatus.REVOKED


class TestRateLimiting:
    """Tests for rate limiting"""

    def test_rate_limit_enforcement(self, client, db_session):
        """Test rate limit enforcement"""
        # Create partner with low quota
        partner = PartnerModel(
            id="partner_ratelimit",
            name="Rate Limited Partner",
            email="ratelimit@example.com",
            tier=PartnerTier.FREE,
            status=PartnerStatus.ACTIVE,
            api_quota=2,  # Very low quota for testing
            password_hash=hash_password("password")
        )
        db_session.add(partner)
        db_session.commit()

        # Login
        login_response = client.post(
            "/api/partners/login",
            json={
                "email": "ratelimit@example.com",
                "password": "password"
            }
        )
        token = login_response.json()["access_token"]

        # Make requests up to quota
        # Note: In real implementation, you'd need to implement rate limiting
        # middleware and test actual API endpoints


class TestUsageStatistics:
    """Tests for usage statistics"""

    def test_get_usage_statistics(self, client, auth_token, sample_partner):
        """Test getting usage statistics"""
        response = client.get(
            f"/api/partners/{sample_partner.id}/usage",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "partner_id" in data
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data

    def test_get_usage_statistics_custom_range(self, client, auth_token, sample_partner):
        """Test getting usage statistics with custom date range"""
        start_date = (DeterministicClock.utcnow() - timedelta(days=7)).isoformat()
        end_date = DeterministicClock.utcnow().isoformat()

        response = client.get(
            f"/api/partners/{sample_partner.id}/usage",
            headers={"Authorization": f"Bearer {auth_token}"},
            params={
                "start_date": start_date,
                "end_date": end_date
            }
        )

        assert response.status_code == 200


class TestBilling:
    """Tests for billing information"""

    def test_get_billing_information(self, client, auth_token, sample_partner):
        """Test getting billing information"""
        response = client.get(
            f"/api/partners/{sample_partner.id}/billing",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "partner_id" in data
        assert "current_usage" in data
        assert "quota_limit" in data
        assert "invoices" in data


class TestHealthCheck:
    """Tests for health check endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "SecurePassword123!"
        hashed = hash_password(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("WrongPassword", hashed)

    def test_api_key_generation(self):
        """Test API key generation"""
        key = generate_api_key()

        assert key.startswith("gl_")
        assert len(key) > 20

    def test_api_key_hashing(self):
        """Test API key hashing"""
        key = generate_api_key()
        hashed = hash_api_key(key)

        assert hashed != key
        assert len(hashed) == 64  # SHA256 hex digest length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
