"""
Network utilities with policy enforcement
"""

import os
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# Global egress allowlist (can be overridden by policy)
DEFAULT_ALLOWED_DOMAINS = [
    "greenlang.io",
    "hub.greenlang.io",
    "api.greenlang.io",
    "github.com",
    "githubusercontent.com",
    "pypi.org",
    "files.pythonhosted.org",
]


class NetworkPolicy:
    """Network policy enforcer"""

    def __init__(self):
        self.allowed_domains = self._load_allowed_domains()
        self.blocked_domains = self._load_blocked_domains()
        self.audit_log = []

    def _load_allowed_domains(self) -> List[str]:
        """Load allowed domains from config or environment"""
        domains = DEFAULT_ALLOWED_DOMAINS.copy()

        # Add from environment
        env_domains = os.getenv("GL_ALLOWED_DOMAINS", "")
        if env_domains:
            domains.extend(env_domains.split(","))

        # Add from config file
        config_file = Path.home() / ".greenlang" / "network_allowlist.txt"
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        domains.append(line)

        return list(set(domains))

    def _load_blocked_domains(self) -> List[str]:
        """Load blocked domains from config"""
        domains = []

        # Add from environment
        env_domains = os.getenv("GL_BLOCKED_DOMAINS", "")
        if env_domains:
            domains.extend(env_domains.split(","))

        # Add from config file
        config_file = Path.home() / ".greenlang" / "network_blocklist.txt"
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        domains.append(line)

        return list(set(domains))

    def check_url(self, url: str, tag: str = "unknown") -> bool:
        """
        Check if URL is allowed by policy

        Args:
            url: URL to check
            tag: Tag describing the purpose of the request

        Returns:
            True if allowed, False otherwise
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]

        # Check blocklist first
        for blocked in self.blocked_domains:
            if domain == blocked or domain.endswith(f".{blocked}"):
                logger.warning(f"Network access blocked: {domain} (tag: {tag})")
                self.audit_log.append(
                    {
                        "action": "blocked",
                        "url": url,
                        "domain": domain,
                        "tag": tag,
                        "reason": "domain in blocklist",
                    }
                )
                return False

        # Check allowlist
        for allowed in self.allowed_domains:
            if domain == allowed or domain.endswith(f".{allowed}"):
                logger.debug(f"Network access allowed: {domain} (tag: {tag})")
                self.audit_log.append(
                    {
                        "action": "allowed",
                        "url": url,
                        "domain": domain,
                        "tag": tag,
                        "reason": "domain in allowlist",
                    }
                )
                return True

        # Not in allowlist
        logger.warning(f"Network access denied: {domain} not in allowlist (tag: {tag})")
        self.audit_log.append(
            {
                "action": "denied",
                "url": url,
                "domain": domain,
                "tag": tag,
                "reason": "domain not in allowlist",
            }
        )
        return False

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log of network access attempts"""
        return self.audit_log.copy()


# Global policy instance
_network_policy = NetworkPolicy()


def policy_allow(url: str, tag: str = "unknown") -> None:
    """
    Check if URL access is allowed by policy

    Args:
        url: URL to access
        tag: Tag describing the purpose

    Raises:
        RuntimeError: If access is denied by policy
    """
    if not _network_policy.check_url(url, tag):
        raise RuntimeError(f"Network access denied by policy: {url}")


def http_get(
    url: str, *, tag: str = "unknown", timeout: int = 30, **kwargs
) -> requests.Response:
    """
    HTTP GET with policy enforcement

    Args:
        url: URL to fetch
        tag: Tag describing the purpose
        timeout: Request timeout in seconds
        **kwargs: Additional arguments for requests.get

    Returns:
        Response object

    Raises:
        RuntimeError: If denied by policy
        requests.RequestException: If request fails
    """
    # Check policy
    policy_allow(url, tag)

    # Make request
    logger.info(f"HTTP GET: {url} (tag: {tag})")
    response = requests.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()

    return response


def http_post(
    url: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    tag: str = "unknown",
    timeout: int = 30,
    **kwargs,
) -> requests.Response:
    """
    HTTP POST with policy enforcement

    Args:
        url: URL to post to
        data: Data to send
        tag: Tag describing the purpose
        timeout: Request timeout
        **kwargs: Additional arguments for requests.post

    Returns:
        Response object

    Raises:
        RuntimeError: If denied by policy
        requests.RequestException: If request fails
    """
    # Check policy
    policy_allow(url, tag)

    # Make request
    logger.info(f"HTTP POST: {url} (tag: {tag})")
    response = requests.post(url, data=data, timeout=timeout, **kwargs)
    response.raise_for_status()

    return response


def download_file(
    url: str, dest: Path, *, tag: str = "download", chunk_size: int = 8192
) -> Path:
    """
    Download file with policy enforcement

    Args:
        url: URL to download from
        dest: Destination path
        tag: Tag for the download
        chunk_size: Download chunk size

    Returns:
        Path to downloaded file

    Raises:
        RuntimeError: If denied by policy
        requests.RequestException: If download fails
    """
    # Check policy
    policy_allow(url, tag)

    # Download file
    logger.info(f"Downloading: {url} -> {dest} (tag: {tag})")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    # Ensure destination directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Write file in chunks
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    logger.info(f"Downloaded {dest.stat().st_size} bytes to {dest}")
    return dest


def add_allowed_domain(domain: str) -> None:
    """Add a domain to the allowlist"""
    _network_policy.allowed_domains.append(domain)
    logger.info(f"Added domain to allowlist: {domain}")


def add_blocked_domain(domain: str) -> None:
    """Add a domain to the blocklist"""
    _network_policy.blocked_domains.append(domain)
    logger.info(f"Added domain to blocklist: {domain}")


def get_network_audit_log() -> List[Dict[str, Any]]:
    """Get the network access audit log"""
    return _network_policy.get_audit_log()


def reset_network_policy() -> None:
    """Reset network policy to defaults"""
    global _network_policy
    _network_policy = NetworkPolicy()
    logger.info("Network policy reset to defaults")
