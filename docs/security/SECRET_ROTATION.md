# Secret Rotation Guide

## Overview
This document provides procedures for rotating secrets when they are discovered in the repository.

## Immediate Response Protocol

### 1. Assessment (First 15 minutes)
1. **Identify the secret type**: API key, password, token, certificate, etc.
2. **Determine exposure scope**:
   - How long has it been exposed?
   - Which branches/tags contain it?
   - Has it been used in production?
3. **Check access logs** at the provider for any unauthorized usage

### 2. Containment (Within 1 hour)
1. **DO NOT** delete the commit immediately (this alerts attackers)
2. **Rotate the secret FIRST** at the provider
3. **Update all systems** using the old secret
4. **Monitor** for failed authentication attempts

### 3. Remediation (Within 24 hours)
1. **Remove from history** using git filter-repo
2. **Update documentation** below
3. **Notify stakeholders** if required

## Provider-Specific Rotation Steps

### AWS Credentials
```bash
# 1. Create new access key
aws iam create-access-key --user-name <username>

# 2. Update applications with new key
# 3. Test new key works
# 4. Deactivate old key
aws iam update-access-key --access-key-id <old-key-id> --status Inactive

# 5. After 24 hours, delete old key
aws iam delete-access-key --access-key-id <old-key-id>
```

### GitHub Tokens
1. Go to Settings → Developer settings → Personal access tokens
2. Generate new token with same permissions
3. Update all CI/CD systems
4. Revoke old token immediately

### API Keys (Generic)
1. Log into provider dashboard
2. Generate new API key
3. Update environment variables:
   ```bash
   # Local
   export API_KEY="new-key-value"

   # GitHub Actions
   gh secret set API_KEY -b "new-key-value"

   # Docker
   docker build --build-arg API_KEY="new-key-value"
   ```
4. Revoke old key at provider

### Database Passwords
```sql
-- PostgreSQL
ALTER USER username WITH PASSWORD 'new_password';

-- MySQL
ALTER USER 'username'@'host' IDENTIFIED BY 'new_password';

-- MongoDB
db.changeUserPassword("username", "new_password")
```

## Removing Secrets from Git History

### Using git filter-repo (Recommended)
```bash
# Install
pip install git-filter-repo

# Remove specific file
git filter-repo --path path/to/secret_file --invert-paths

# Replace secret strings in all files
echo "old_secret==>REDACTED" > replacements.txt
git filter-repo --replace-text replacements.txt

# Force push (coordinate with team!)
git push --force --all
git push --force --tags
```

### Using BFG Repo-Cleaner (Alternative)
```bash
# Download BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Remove file
java -jar bfg.jar --delete-files secret_file.txt

# Replace text
echo "old_secret" > passwords.txt
java -jar bfg.jar --replace-text passwords.txt

# Clean and push
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags
```

## Prevention Best Practices

### 1. Use Environment Variables
```python
import os
api_key = os.environ.get('API_KEY')  # Never hardcode!
```

### 2. Use Secret Managers
```python
# AWS Secrets Manager
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='prod/api/key')

# HashiCorp Vault
import hvac
client = hvac.Client(url='https://vault.example.com')
secret = client.read('secret/data/api-key')
```

### 3. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/trufflesecurity/trufflehog
    rev: v3.0.0
    hooks:
      - id: trufflehog
        args: ['--only-verified', '--fail']
```

### 4. .gitignore Templates
```gitignore
# Environment files
.env
.env.*
!.env.example

# Key files
*.key
*.pem
*.p12
*.pfx

# Credential files
credentials.json
service-account.json
```

## Rotation Log

Document all secret rotations here:

| Date | Secret Type | Affected Systems | Rotated By | Ticket # | Notes |
|------|-------------|------------------|------------|----------|-------|
| Example: 2025-01-15 | AWS Access Key | CI/CD Pipeline | DevOps Team | SEC-001 | Exposed in commit abc123, rotated within 1 hour |

## Contact Information

- **Security Team**: security@greenlang.com
- **On-call Engineer**: Use PagerDuty
- **CTO**: For CRITICAL exposures only

## References
- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [AWS: Best practices for managing AWS access keys](https://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html)
- [OWASP: Key Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html)