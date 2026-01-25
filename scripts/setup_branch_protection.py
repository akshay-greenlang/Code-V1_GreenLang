#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup GitHub Branch Protection Rules via API
This script configures branch protection for the master branch.

Security Note:
This script uses GreenLang's secure HTTP wrapper when available, which provides:
- Policy enforcement for network access
- Audit logging of HTTP requests
- Security compliance checks
When running outside the GreenLang environment, it falls back to direct requests.

Prerequisites:
1. Create a Personal Access Token at: https://github.com/settings/tokens
2. Select scopes: repo (full control)
3. Set environment variable: GITHUB_TOKEN=your_token_here
4. If using GreenLang security framework, ensure api.github.com is in allowed domains

Usage:
    python scripts/setup_branch_protection.py
"""

import os
import sys
import json
from typing import Optional

# Use GreenLang's secure HTTP wrapper instead of direct requests
try:
    from greenlang.security import http as secure_http
    USE_SECURE_HTTP = True
except ImportError:
    # Fallback to direct requests if running outside GreenLang environment
    import requests
    USE_SECURE_HTTP = False
    print("⚠️  Running outside GreenLang environment - using direct HTTP calls")
    print("   For security compliance, consider running within GreenLang framework")

# Configuration
REPO_OWNER = "your-username"
REPO_NAME = "GreenLang"
BRANCH = "master"

def make_http_request(method: str, url: str, headers: dict, json_data=None):
    """
    Make HTTP request using either secure wrapper or direct requests.

    Args:
        method: HTTP method (GET, PUT, etc.)
        url: Request URL
        headers: Request headers
        json_data: JSON payload for PUT/POST requests

    Returns:
        Response object
    """
    if USE_SECURE_HTTP:
        # Add GitHub domain to allowed domains for this session
        # Note: This is for administrative GitHub API access
        try:
            from greenlang.utils.net import add_allowed_domain
            add_allowed_domain("api.github.com")
        except ImportError:
            pass

        # Use secure HTTP wrapper
        if method.upper() == 'GET':
            return secure_http.get(url, headers=headers)
        elif method.upper() == 'PUT':
            return secure_http.put(url, headers=headers, json=json_data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    else:
        # Fallback to direct requests
        if method.upper() == 'GET':
            return requests.get(url, headers=headers)
        elif method.upper() == 'PUT':
            return requests.put(url, headers=headers, json=json_data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

def setup_branch_protection(token: str) -> None:
    """Configure branch protection rules via GitHub API."""

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # API endpoint for branch protection
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/branches/{BRANCH}/protection"

    # Protection rules configuration
    protection_rules = {
        "required_status_checks": {
            "strict": True,  # Require branches to be up to date
            "contexts": [
                # CI checks for all OS and Python combinations
                "CI / Unit (ubuntu-latest | py3.10)",
                "CI / Unit (ubuntu-latest | py3.11)",
                "CI / Unit (ubuntu-latest | py3.12)",
                "CI / Unit (macos-latest | py3.10)",
                "CI / Unit (macos-latest | py3.11)",
                "CI / Unit (macos-latest | py3.12)",
                "CI / Unit (windows-latest | py3.10)",
                "CI / Unit (windows-latest | py3.11)",
                "CI / Unit (windows-latest | py3.12)",
                # Optional: Add other checks if they exist
                "GreenLang Guards / governance",
                "acceptance / scaffolding"
            ]
        },
        "enforce_admins": True,  # Apply rules to administrators too
        "required_pull_request_reviews": {
            "dismissal_restrictions": {},
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "required_approving_review_count": 1,  # Require 1 approval
            "bypass_pull_request_allowances": {}
        },
        "restrictions": None,  # No user/team restrictions
        "allow_force_pushes": False,
        "allow_deletions": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": False
    }

    print(f"🔐 Setting up branch protection for {REPO_OWNER}/{REPO_NAME}:{BRANCH}")
    print("📋 Configuration:")
    print("  - Require PR before merging: Yes")
    print("  - Required approving reviews: 1")
    print("  - Dismiss stale PR approvals: Yes")
    print("  - Require status checks: Yes")
    print("  - Require up-to-date branches: Yes")
    print("  - Include administrators: Yes")
    print(f"  - Total status checks: {len(protection_rules['required_status_checks']['contexts'])}")

    try:
        response = make_http_request('PUT', url, headers, protection_rules)

        if response.status_code == 200:
            print("✅ Branch protection successfully configured!")
            print("\n📊 Protected branch details:")
            result = response.json()

            if 'required_status_checks' in result:
                print(f"  - Status checks: {len(result['required_status_checks']['contexts'])} required")
            if 'required_pull_request_reviews' in result:
                print(f"  - PR reviews: {result['required_pull_request_reviews']['required_approving_review_count']} required")
            if 'enforce_admins' in result:
                print(f"  - Enforce for admins: {result['enforce_admins']['enabled']}")

        elif response.status_code == 404:
            print("❌ Error: Repository or branch not found")
            print(f"   Please check: REPO_OWNER={REPO_OWNER}, REPO_NAME={REPO_NAME}")
        elif response.status_code == 401:
            print("❌ Error: Authentication failed")
            print("   Please check your GitHub token has 'repo' scope")
        elif response.status_code == 403:
            print("❌ Error: Insufficient permissions")
            print("   You need admin access to the repository")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")

    except Exception as e:
        print(f"❌ Error setting up protection: {e}")
        sys.exit(1)

def check_current_protection(token: str) -> bool:
    """Check current protection status."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/branches/{BRANCH}/protection"

    try:
        response = make_http_request('GET', url, headers)
        if response.status_code == 200:
            print("📌 Current protection rules are active")
            return True
        elif response.status_code == 404:
            print("⚠️  No protection rules currently configured")
            return False
    except:
        return False

def main():
    """Main execution."""
    print("🚀 GitHub Branch Protection Setup Script")
    print("=" * 50)

    # Get token from environment
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        print("\n⚠️  No GitHub token found!")
        print("\nTo set up branch protection, follow these steps:")
        print("\n1. Create a Personal Access Token:")
        print("   https://github.com/settings/tokens/new")
        print("   - Give it a name (e.g., 'GreenLang Branch Protection')")
        print("   - Select scope: ✅ repo (Full control)")
        print("   - Click 'Generate token'")
        print("   - Copy the token (you won't see it again!)")
        print("\n2. Set the token as environment variable:")
        print("   Windows PowerShell:")
        print('     $env:GITHUB_TOKEN="your_token_here"')
        print("   Windows CMD:")
        print('     set GITHUB_TOKEN=your_token_here')
        print("   Linux/Mac:")
        print('     export GITHUB_TOKEN="your_token_here"')
        print("\n3. Update this script:")
        print(f"   - Change REPO_OWNER from '{REPO_OWNER}' to your GitHub username")
        print("\n4. Run this script again:")
        print("     python scripts/setup_branch_protection.py")
        sys.exit(1)

    # Check current status
    print(f"\n🔍 Checking {REPO_OWNER}/{REPO_NAME}:{BRANCH}...")
    has_protection = check_current_protection(token)

    if has_protection:
        print("\n⚠️  Branch already has protection rules.")
        response = input("Do you want to update them? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without changes.")
            sys.exit(0)

    # Apply protection
    print("\n🔧 Applying branch protection rules...")
    setup_branch_protection(token)

    print("\n✨ Setup complete!")
    print("\nNext steps:")
    print("1. Visit: https://github.com/{}/{}/settings/branches".format(REPO_OWNER, REPO_NAME))
    print("2. Verify the protection rules are active")
    print("3. Create a test PR to confirm everything works")

if __name__ == "__main__":
    main()