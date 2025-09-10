"""
Authentication and API Key Management for GreenLang
"""

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logging.warning("bcrypt not available. Install with: pip install bcrypt")

logger = logging.getLogger(__name__)


@dataclass
class AuthToken:
    """Authentication token"""
    
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_value: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    tenant_id: str = ""
    user_id: str = ""
    
    # Token metadata
    name: str = ""
    description: str = ""
    token_type: str = "bearer"  # bearer, api_key, service_account
    
    # Permissions
    scopes: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    
    # Validity
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    
    # Status
    active: bool = True
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None
    
    # Usage limits
    max_uses: Optional[int] = None
    use_count: int = 0
    rate_limit: Optional[int] = None  # Requests per hour
    
    # Security
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if token is valid"""
        if not self.active or self.revoked:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        if self.max_uses and self.use_count >= self.max_uses:
            return False
        
        return True
    
    def use(self):
        """Record token usage"""
        self.use_count += 1
        self.last_used_at = datetime.utcnow()
    
    def revoke(self, by: str = None, reason: str = None):
        """Revoke token"""
        self.revoked = True
        self.revoked_at = datetime.utcnow()
        self.revoked_by = by
        self.revoke_reason = reason
        self.active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "token_id": self.token_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "name": self.name,
            "token_type": self.token_type,
            "scopes": self.scopes,
            "roles": self.roles,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "active": self.active,
            "use_count": self.use_count
        }


@dataclass
class APIKey:
    """API Key for programmatic access"""
    
    key_id: str = field(default_factory=lambda: f"glk_{uuid.uuid4().hex[:16]}")
    key_secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    tenant_id: str = ""
    
    # Metadata
    name: str = ""
    description: str = ""
    
    # Permissions
    scopes: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    allowed_resources: List[str] = field(default_factory=list)
    
    # Validity
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_rotated_at: Optional[datetime] = None
    
    # Status
    active: bool = True
    
    # Usage tracking
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    
    # Restrictions
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    
    def rotate(self) -> str:
        """Rotate API key secret"""
        self.key_secret = secrets.token_urlsafe(32)
        self.last_rotated_at = datetime.utcnow()
        return self.key_secret
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if not self.active:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def get_display_key(self) -> str:
        """Get display version of key (masked)"""
        if len(self.key_secret) > 8:
            return f"{self.key_id}.{self.key_secret[:4]}...{self.key_secret[-4:]}"
        return f"{self.key_id}.****"


@dataclass
class ServiceAccount:
    """Service account for automated access"""
    
    account_id: str = field(default_factory=lambda: f"sa_{uuid.uuid4().hex[:16]}")
    tenant_id: str = ""
    
    # Metadata
    name: str = ""
    description: str = ""
    email: str = ""
    
    # Authentication
    api_keys: List[APIKey] = field(default_factory=list)
    tokens: List[AuthToken] = field(default_factory=list)
    
    # Permissions
    roles: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    
    # Status
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    # Kubernetes integration
    k8s_namespace: Optional[str] = None
    k8s_service_account: Optional[str] = None
    
    def create_api_key(self, name: str = "", **kwargs) -> APIKey:
        """Create new API key for service account"""
        key = APIKey(
            tenant_id=self.tenant_id,
            name=name or f"{self.name}-key-{len(self.api_keys)+1}",
            **kwargs
        )
        self.api_keys.append(key)
        self.updated_at = datetime.utcnow()
        return key
    
    def revoke_all_keys(self):
        """Revoke all API keys"""
        for key in self.api_keys:
            key.active = False
        self.updated_at = datetime.utcnow()


class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize auth manager
        
        Args:
            config: Configuration including:
                - secret_key: Secret for token generation
                - token_expiry: Default token expiry in seconds
                - password_min_length: Minimum password length
                - require_mfa: Whether to require MFA
        """
        self.config = config or {}
        self.secret_key = config.get("secret_key", secrets.token_urlsafe(32))
        self.token_expiry = config.get("token_expiry", 3600 * 24)  # 24 hours
        self.password_min_length = config.get("password_min_length", 8)
        self.require_mfa = config.get("require_mfa", False)
        
        # Storage (in production, use database)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, AuthToken] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.service_accounts: Dict[str, ServiceAccount] = {}
        
        # Session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Failed attempt tracking (for rate limiting)
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        logger.info("AuthManager initialized")
    
    def create_user(self, tenant_id: str, username: str, email: str,
                   password: str, **kwargs) -> str:
        """
        Create new user
        
        Args:
            tenant_id: Tenant ID
            username: Username
            email: Email address
            password: Password
            **kwargs: Additional user properties
            
        Returns:
            User ID
        """
        # Validate password
        if len(password) < self.password_min_length:
            raise ValueError(f"Password must be at least {self.password_min_length} characters")
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Create user
        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
            "active": True,
            "mfa_enabled": False,
            "mfa_secret": None,
            **kwargs
        }
        
        logger.info(f"Created user: {user_id} ({username})")
        return user_id
    
    def authenticate(self, username: str, password: str,
                    tenant_id: Optional[str] = None,
                    mfa_code: Optional[str] = None) -> Optional[AuthToken]:
        """
        Authenticate user
        
        Args:
            username: Username or email
            password: Password
            tenant_id: Optional tenant ID
            mfa_code: Optional MFA code
            
        Returns:
            AuthToken if successful, None otherwise
        """
        # Find user
        user = None
        for u in self.users.values():
            if (u["username"] == username or u["email"] == username):
                if not tenant_id or u["tenant_id"] == tenant_id:
                    user = u
                    break
        
        if not user:
            self._record_failed_attempt(username)
            return None
        
        # Check if account is active
        if not user.get("active", True):
            logger.warning(f"Login attempt for inactive account: {username}")
            return None
        
        # Verify password
        if not self._verify_password(password, user["password_hash"]):
            self._record_failed_attempt(username)
            return None
        
        # Check MFA if enabled
        if user.get("mfa_enabled") and self.require_mfa:
            if not mfa_code or not self._verify_mfa(user["mfa_secret"], mfa_code):
                logger.warning(f"Invalid MFA code for user: {username}")
                return None
        
        # Create token
        token = self.create_token(
            tenant_id=user["tenant_id"],
            user_id=user["user_id"],
            name=f"Session for {username}",
            token_type="bearer"
        )
        
        # Create session
        self.sessions[token.token_id] = {
            "user_id": user["user_id"],
            "tenant_id": user["tenant_id"],
            "username": username,
            "login_time": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        logger.info(f"User authenticated: {username}")
        return token
    
    def create_token(self, tenant_id: str, user_id: str = None,
                    name: str = "", token_type: str = "bearer",
                    expires_in: Optional[int] = None,
                    **kwargs) -> AuthToken:
        """
        Create authentication token
        
        Args:
            tenant_id: Tenant ID
            user_id: Optional user ID
            name: Token name
            token_type: Token type
            expires_in: Expiry in seconds
            **kwargs: Additional token properties
            
        Returns:
            AuthToken
        """
        token = AuthToken(
            tenant_id=tenant_id,
            user_id=user_id or "",
            name=name,
            token_type=token_type,
            **kwargs
        )
        
        if expires_in:
            token.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        elif self.token_expiry:
            token.expires_at = datetime.utcnow() + timedelta(seconds=self.token_expiry)
        
        # Store token
        self.tokens[token.token_value] = token
        
        logger.info(f"Created token: {token.token_id}")
        return token
    
    def validate_token(self, token_value: str) -> Optional[AuthToken]:
        """
        Validate token
        
        Args:
            token_value: Token value
            
        Returns:
            AuthToken if valid, None otherwise
        """
        token = self.tokens.get(token_value)
        
        if not token:
            return None
        
        if not token.is_valid():
            return None
        
        # Update usage
        token.use()
        
        # Update session activity
        if token.token_id in self.sessions:
            self.sessions[token.token_id]["last_activity"] = datetime.utcnow()
        
        return token
    
    def revoke_token(self, token_value: str, by: str = None, reason: str = None) -> bool:
        """
        Revoke token
        
        Args:
            token_value: Token value
            by: Who revoked it
            reason: Revocation reason
            
        Returns:
            True if revoked
        """
        token = self.tokens.get(token_value)
        
        if token:
            token.revoke(by, reason)
            
            # Remove session
            if token.token_id in self.sessions:
                del self.sessions[token.token_id]
            
            logger.info(f"Revoked token: {token.token_id}")
            return True
        
        return False
    
    def create_api_key(self, tenant_id: str, name: str,
                      scopes: List[str] = None,
                      expires_in: Optional[int] = None,
                      **kwargs) -> APIKey:
        """
        Create API key
        
        Args:
            tenant_id: Tenant ID
            name: Key name
            scopes: Permission scopes
            expires_in: Expiry in seconds
            **kwargs: Additional properties
            
        Returns:
            APIKey
        """
        key = APIKey(
            tenant_id=tenant_id,
            name=name,
            scopes=scopes or [],
            **kwargs
        )
        
        if expires_in:
            key.expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        # Store key
        self.api_keys[f"{key.key_id}.{key.key_secret}"] = key
        
        logger.info(f"Created API key: {key.key_id}")
        return key
    
    def validate_api_key(self, key_string: str) -> Optional[APIKey]:
        """
        Validate API key
        
        Args:
            key_string: API key string (id.secret)
            
        Returns:
            APIKey if valid, None otherwise
        """
        key = self.api_keys.get(key_string)
        
        if not key:
            return None
        
        if not key.is_valid():
            return None
        
        # Update usage
        key.use_count += 1
        key.last_used_at = datetime.utcnow()
        
        return key
    
    def create_service_account(self, tenant_id: str, name: str,
                             email: str = None, **kwargs) -> ServiceAccount:
        """
        Create service account
        
        Args:
            tenant_id: Tenant ID
            name: Account name
            email: Service account email
            **kwargs: Additional properties
            
        Returns:
            ServiceAccount
        """
        account = ServiceAccount(
            tenant_id=tenant_id,
            name=name,
            email=email or f"{name}@greenlang.service",
            **kwargs
        )
        
        # Store account
        self.service_accounts[account.account_id] = account
        
        logger.info(f"Created service account: {account.account_id}")
        return account
    
    def _hash_password(self, password: str) -> str:
        """Hash password"""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        else:
            # Fallback to SHA256 (less secure)
            salt = secrets.token_hex(32)
            hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
            return f"{salt}${hash_obj.hexdigest()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password"""
        if BCRYPT_AVAILABLE:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        else:
            # Fallback verification
            if '$' not in password_hash:
                return False
            salt, hash_value = password_hash.split('$', 1)
            hash_obj = hashlib.sha256((salt + password).encode('utf-8'))
            return hash_obj.hexdigest() == hash_value
    
    def _verify_mfa(self, secret: str, code: str) -> bool:
        """Verify MFA code"""
        # Simplified MFA verification (in production, use pyotp)
        expected = hmac.new(
            secret.encode('utf-8'),
            str(int(datetime.utcnow().timestamp() // 30)).encode('utf-8'),
            hashlib.sha256
        ).hexdigest()[:6]
        
        return code == expected
    
    def _record_failed_attempt(self, identifier: str):
        """Record failed login attempt"""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(datetime.utcnow())
        
        # Clean old attempts (keep last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.failed_attempts[identifier] = [
            dt for dt in self.failed_attempts[identifier]
            if dt > cutoff
        ]
        
        # Log if too many attempts
        if len(self.failed_attempts[identifier]) > 5:
            logger.warning(f"Multiple failed login attempts for: {identifier}")