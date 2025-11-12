# Security Scanning Framework

## 1. Static Application Security Testing (SAST)

### SonarQube Configuration

```yaml
# sonarqube-config.yaml
sonarqube:
  server:
    url: "https://sonar.greenlang.io"
    version: "10.3"

  quality_gates:
    security_gate:
      conditions:
        - metric: "security_rating"
          operator: "GREATER_THAN"
          value: "A"
        - metric: "vulnerabilities"
          operator: "LESS_THAN"
          value: "1"
        - metric: "security_hotspots_reviewed"
          operator: "GREATER_THAN"
          value: "100"

    code_quality_gate:
      conditions:
        - metric: "coverage"
          operator: "GREATER_THAN"
          value: "80"
        - metric: "duplicated_lines_density"
          operator: "LESS_THAN"
          value: "3"
        - metric: "code_smells"
          operator: "LESS_THAN"
          value: "10"

  security_rules:
    profile: "OWASP Top 10 + CWE Top 25"
    custom_rules:
      - id: "GREENLANG001"
        description: "Detect hardcoded API keys"
        severity: "BLOCKER"
        pattern: "api_key\\s*=\\s*[\"'][\\w]+[\"']"

      - id: "GREENLANG002"
        description: "Detect direct HTTP calls without wrapper"
        severity: "BLOCKER"
        pattern: "http\\.(Get|Post|Put|Delete)\\([^)]*\\)"

      - id: "GREENLANG003"
        description: "Detect SQL injection vulnerabilities"
        severity: "CRITICAL"
        pattern: "fmt\\.Sprintf.*SELECT.*%s"
```

### Checkmarx Integration

```javascript
// checkmarx-scanner.js
const CheckmarxScanner = {
  config: {
    server: 'https://checkmarx.greenlang.io',
    projectName: 'GreenLang-Platform',
    preset: 'OWASP+SANS+PCI',
    engineConfiguration: 'Improved Data Flow',

    thresholds: {
      high: 0,
      medium: 5,
      low: 10,
      info: null
    },

    exclusions: [
      'node_modules',
      'vendor',
      'test',
      '*.min.js',
      '*.test.ts'
    ],

    incrementalScan: true,
    generatePDFReport: true
  },

  scanPolicies: {
    preCommit: {
      enabled: true,
      incremental: true,
      breakBuild: false
    },

    pullRequest: {
      enabled: true,
      incremental: true,
      breakBuild: true,
      commentOnPR: true
    },

    nightly: {
      enabled: true,
      fullScan: true,
      breakBuild: false,
      notifyTeam: true
    }
  },

  customQueries: [
    {
      name: 'DetectSensitiveDataExposure',
      language: 'JavaScript',
      severity: 'High',
      query: `
        // CxQL query to detect sensitive data exposure
        result = All.FindByType("MethodInvocation")
          .FindByName("console.log", "logger.debug")
          .FindByMemberAccess("password", "token", "apiKey", "secret");
      `
    }
  ]
};
```

### Semgrep Rules

```yaml
# semgrep-rules.yaml
rules:
  - id: direct-http-calls
    pattern-either:
      - pattern: requests.$METHOD(...)
      - pattern: urllib.request.$METHOD(...)
      - pattern: http.$METHOD(...)
    message: "Direct HTTP calls must use security wrapper"
    severity: ERROR
    languages: [python, javascript, go]

  - id: hardcoded-secrets
    pattern-either:
      - pattern: |
          $KEY = "..."
      - pattern: |
          $KEY = '...'
    metavariable-regex:
      metavariable: $KEY
      regex: '.*(password|pwd|token|api_key|apikey|secret|private_key).*'
    message: "Potential hardcoded secret detected"
    severity: ERROR

  - id: sql-injection
    patterns:
      - pattern-either:
          - pattern: |
              $QUERY = "SELECT * FROM " + $INPUT
          - pattern: |
              $QUERY = f"SELECT * FROM {$INPUT}"
          - pattern: |
              cursor.execute("..." % $INPUT)
    message: "SQL injection vulnerability"
    severity: ERROR

  - id: weak-crypto
    pattern-either:
      - pattern: hashlib.md5(...)
      - pattern: hashlib.sha1(...)
      - pattern: DES.new(...)
      - pattern: Random().random()
    message: "Weak cryptographic algorithm"
    severity: WARNING
```

## 2. Dynamic Application Security Testing (DAST)

### OWASP ZAP Configuration

```yaml
# zap-config.yaml
zap:
  context:
    name: "GreenLang Platform"
    urls:
      - "https://app.greenlang.io"
      - "https://api.greenlang.io"

  authentication:
    method: "oauth2"
    parameters:
      token_endpoint: "https://auth.greenlang.io/oauth2/token"
      client_id: "${ZAP_CLIENT_ID}"
      client_secret: "${ZAP_CLIENT_SECRET}"

  scan_policies:
    active_scan:
      strength: "HIGH"
      threshold: "MEDIUM"
      categories:
        - "Injection"
        - "Broken Authentication"
        - "Sensitive Data Exposure"
        - "XXE"
        - "Broken Access Control"
        - "Security Misconfiguration"
        - "XSS"
        - "Insecure Deserialization"
        - "Using Components with Known Vulnerabilities"
        - "Insufficient Logging"

    passive_scan:
      enabled: true
      rules:
        - id: 10015  # Application Error Disclosure
          threshold: "HIGH"
        - id: 10017  # Cross-Domain JavaScript Source File Inclusion
          threshold: "MEDIUM"
        - id: 10019  # Content-Type Header Missing
          threshold: "LOW"

  automation:
    environments:
      - name: "staging"
        context: "staging-context"
        spider:
          maxDuration: 60
          maxDepth: 10
          maxChildren: 50

      - name: "production"
        context: "production-context"
        spider:
          maxDuration: 120
          maxDepth: 5
          maxChildren: 20

  reporting:
    formats:
      - "json"
      - "html"
      - "sarif"
    risk_levels:
      - "High"
      - "Medium"
    confidence_levels:
      - "High"
      - "Medium"
```

### Burp Suite Enterprise Configuration

```json
{
  "burp_suite_enterprise": {
    "scan_configurations": {
      "comprehensive": {
        "scan_type": "Full",
        "crawl_config": {
          "max_depth": 10,
          "max_unique_locations": 5000,
          "crawl_optimization": "fastest"
        },
        "audit_config": {
          "issues_to_detect": "all",
          "detection_methods": "active_and_passive",
          "thoroughness": "thorough"
        }
      },
      "api_scan": {
        "scan_type": "API",
        "api_definition": "openapi.json",
        "authentication": {
          "type": "bearer_token",
          "token_refresh": true
        }
      },
      "ci_cd_scan": {
        "scan_type": "Lightweight",
        "max_duration_minutes": 30,
        "severity_threshold": "medium"
      }
    },
    "schedule": {
      "daily_scans": [
        {
          "name": "API Security Scan",
          "time": "02:00",
          "configuration": "api_scan",
          "targets": ["https://api.greenlang.io"]
        }
      ],
      "weekly_scans": [
        {
          "name": "Comprehensive Platform Scan",
          "day": "Saturday",
          "time": "00:00",
          "configuration": "comprehensive",
          "targets": ["https://app.greenlang.io"]
        }
      ]
    }
  }
}
```

## 3. Software Composition Analysis (SCA)

### Snyk Configuration

```yaml
# .snyk
version: v1.0.0
language-settings:
  python:
    enableLicensesScan: true
    enableVulnerabilitiesScan: true

  javascript:
    enableLicensesScan: true
    enableVulnerabilitiesScan: true
    packageManager: npm

  go:
    enableLicensesScan: true
    enableVulnerabilitiesScan: true

patches: {}

ignore:
  # Example: Ignore a specific vulnerability
  - SNYK-JS-LODASH-567746:
      reason: "No upgrade path available"
      expires: "2025-12-31T23:59:59.999Z"

policies:
  severity-threshold: high
  license-policy:
    prohibited:
      - "GPL-3.0"
      - "AGPL-3.0"
    warning:
      - "GPL-2.0"
      - "LGPL-3.0"
    allowed:
      - "MIT"
      - "Apache-2.0"
      - "BSD-3-Clause"
      - "ISC"

  fail-on:
    - vulnerabilities:
        severity: critical
        cvss-score: ">= 9.0"
    - license-issues:
        severity: high

monitoring:
  projects:
    - name: "greenlang-api"
      path: "./api"
      branch: "main"

    - name: "greenlang-frontend"
      path: "./frontend"
      branch: "main"

    - name: "greenlang-services"
      path: "./services"
      branch: "main"
```

### Dependabot Configuration

```yaml
# .github/dependabot.yml
version: 2
registries:
  npm-github:
    type: npm-registry
    url: https://npm.pkg.github.com
    token: ${{secrets.GITHUB_TOKEN}}

updates:
  - package-ecosystem: "npm"
    directory: "/frontend"
    registries:
      - npm-github
    schedule:
      interval: "daily"
      time: "04:00"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    commit-message:
      prefix: "deps"
      prefix-development: "deps-dev"
    ignore:
      - dependency-name: "react"
        versions: ["18.x"]

  - package-ecosystem: "pip"
    directory: "/api"
    schedule:
      interval: "weekly"
    reviewers:
      - "backend-team"
    labels:
      - "python"
      - "dependencies"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "devops-team"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"

security-updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "daily"
    priority: "high"

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
    priority: "high"
```

## 4. Container Security Scanning

### Trivy Configuration

```yaml
# trivy-config.yaml
scan:
  security-checks:
    - vuln
    - config
    - secret
    - license

  severity:
    - CRITICAL
    - HIGH
    - MEDIUM

  ignore-unfixed: false

  scanners:
    - os
    - library
    - config

vulnerability:
  type:
    - os
    - library

  ignore-file: .trivyignore

  skip-dirs:
    - /usr/local/lib/node_modules
    - /root/.cache

secret:
  config: trivy-secret.yaml

license:
  forbidden:
    - GPL-3.0
    - AGPL-3.0
  restricted:
    - GPL-2.0
    - LGPL-3.0

config:
  policy:
    - policies/docker-cis-1.2.0.rego
    - policies/kubernetes-cis-1.6.1.rego

  skip-policy-update: false

report:
  format: table
  dependency-tree: true
  list-all-pkgs: false
  exit-code: 1
  output: trivy-report.json

cache:
  backend: fs
  cache-dir: /tmp/trivy-cache

db:
  repository: ghcr.io/aquasecurity/trivy-db
  skip-update: false
  download-db-only: false

misconfiguration:
  scanners:
    - dockerfile
    - kubernetes
    - terraform
    - cloudformation
```

### Anchore Engine Configuration

```yaml
# anchore-config.yaml
services:
  analyzer:
    enabled: true
    max_compressed_image_size_mb: 5000
    layer_cache_max_gigabytes: 20

  policy_engine:
    enabled: true
    policies:
      - id: "production"
        name: "Production Policy"
        rules:
          - gate: "vulnerabilities"
            trigger: "package"
            action: "stop"
            parameters:
              package_type: "all"
              severity: "critical"

          - gate: "secret_scans"
            trigger: "content_regex_check"
            action: "stop"
            parameters:
              regex: "(?i)(api_key|apikey|password|passwd|pwd|secret|token)"

          - gate: "licenses"
            trigger: "blacklist"
            action: "stop"
            parameters:
              licenses:
                - "GPL-3.0"
                - "AGPL-3.0"

  catalog:
    enabled: true
    image_ttl_days: 30
    runtime_inventory:
      enabled: true
      inventory_ttl_days: 7

  simplequeue:
    enabled: true

feeds:
  grypedb:
    enabled: true
    url: "https://toolbox-data.anchore.io/grype/databases/listing.json"
    poll_interval_seconds: 21600

  nvdv2:
    enabled: true
    poll_interval_seconds: 3600

  github:
    enabled: true
    token: "${GITHUB_TOKEN}"

webhooks:
  - endpoint: "https://notifications.greenlang.io/anchore"
    enabled: true
    events:
      - "analysis_update"
      - "vuln_update"
      - "policy_eval"
```

## 5. Infrastructure as Code (IaC) Scanning

### Terraform Sentinel Policies

```hcl
# sentinel/policies/security.sentinel
import "tfplan/v2" as tfplan
import "tfconfig/v2" as tfconfig

# Ensure all S3 buckets are encrypted
s3_encryption = rule {
    all tfplan.resource_changes as _, rc {
        rc.type is not "aws_s3_bucket" or
        rc.change.after.server_side_encryption_configuration[0].rule[0].apply_server_side_encryption_by_default[0].sse_algorithm is "AES256"
    }
}

# Ensure all RDS instances are encrypted
rds_encryption = rule {
    all tfplan.resource_changes as _, rc {
        rc.type is not "aws_db_instance" or
        rc.change.after.storage_encrypted is true
    }
}

# Ensure security groups don't allow 0.0.0.0/0 ingress
no_public_ingress = rule {
    all tfplan.resource_changes as _, rc {
        rc.type is not "aws_security_group_rule" or
        rc.change.after.type is not "ingress" or
        rc.change.after.cidr_blocks not contains "0.0.0.0/0"
    }
}

# Ensure IAM policies follow least privilege
iam_least_privilege = rule {
    all tfplan.resource_changes as _, rc {
        rc.type is not "aws_iam_policy" or
        not rc.change.after.policy contains "*:*"
    }
}

# Main policy
main = rule {
    s3_encryption and
    rds_encryption and
    no_public_ingress and
    iam_least_privilege
}
```

### Checkov Custom Policies

```python
# checkov_custom_policies.py
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck
from checkov.common.models.enums import CheckResult, CheckCategories

class EnsureEKSLoggingEnabled(BaseResourceCheck):
    def __init__(self):
        name = "Ensure EKS cluster has logging enabled"
        id = "GREENLANG_AWS_1"
        supported_resources = ['aws_eks_cluster']
        categories = [CheckCategories.LOGGING]
        super().__init__(name=name, id=id, categories=categories, supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        if 'enabled_cluster_log_types' in conf:
            log_types = conf['enabled_cluster_log_types'][0]
            required_logs = ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler']
            if all(log in log_types for log in required_logs):
                return CheckResult.PASSED
        return CheckResult.FAILED

class EnsureKMSKeyRotation(BaseResourceCheck):
    def __init__(self):
        name = "Ensure KMS key rotation is enabled"
        id = "GREENLANG_AWS_2"
        supported_resources = ['aws_kms_key']
        categories = [CheckCategories.ENCRYPTION]
        super().__init__(name=name, id=id, categories=categories, supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        if 'enable_key_rotation' in conf:
            if conf['enable_key_rotation'][0]:
                return CheckResult.PASSED
        return CheckResult.FAILED

# Register checks
check1 = EnsureEKSLoggingEnabled()
check2 = EnsureKMSKeyRotation()
```

## 6. License Compliance Scanning

### License Scanner Configuration

```yaml
# license-scanner.yaml
license_finder:
  permitted_licenses:
    - MIT
    - Apache-2.0
    - BSD-3-Clause
    - BSD-2-Clause
    - ISC
    - CC0-1.0
    - Unlicense

  restricted_licenses:
    - GPL-2.0:
        reason: "Copyleft - requires approval"
        approval_required: true
    - LGPL-2.1:
        reason: "Weak copyleft - requires approval"
        approval_required: true
    - LGPL-3.0:
        reason: "Weak copyleft - requires approval"
        approval_required: true

  prohibited_licenses:
    - GPL-3.0:
        reason: "Strong copyleft - not compatible"
    - AGPL-3.0:
        reason: "Network copyleft - not compatible"
    - SSPL:
        reason: "Source-available - not open source"
    - Commons-Clause:
        reason: "Commercial restriction"

  package_managers:
    - npm:
        command: "npm list --json --depth=0"
    - pip:
        command: "pip-licenses --format=json"
    - go:
        command: "go-licenses report"
    - maven:
        command: "mvn license:aggregate-add-third-party"

  reporting:
    format: "json"
    output_file: "licenses-report.json"
    include_dev_dependencies: false
    group_by_license: true

  whitelist:
    packages:
      - name: "problematic-package"
        version: "1.0.0"
        license: "GPL-3.0"
        reason: "Approved by legal - isolated use"
        expiry: "2025-12-31"

  ci_integration:
    fail_on_prohibited: true
    fail_on_restricted_without_approval: true
    generate_sbom: true
    update_frequency: "daily"
```

### FOSSA Configuration

```yaml
# .fossa.yml
version: 3

project:
  id: greenlang-platform
  name: GreenLang Platform
  team: engineering
  policy: strict-compliance

analyze:
  modules:
    - path: ./frontend
      type: npm
      target: package.json

    - path: ./api
      type: pip
      target: requirements.txt

    - path: ./services
      type: go
      target: go.mod

    - path: ./infrastructure
      type: terraform

filters:
  - "vendor/**"
  - "node_modules/**"
  - "*.test.*"
  - "test/**"

license:
  policy:
    - deny:
        - GPL-3.0-or-later
        - AGPL-3.0-or-later
    - flag:
        - GPL-2.0-or-later
        - LGPL-3.0-or-later
    - allow:
        - MIT
        - Apache-2.0
        - BSD-3-Clause

experimental:
  enable_vendored_dependencies: true
  enable_dynamic_analysis: true
  enable_container_scanning: true
```