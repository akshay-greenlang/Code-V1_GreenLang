# Software Bill of Materials (SBOM) Management

## 1. SBOM Generation Process

### SPDX Format Generation

```yaml
# sbom-generation-config.yaml
sbom:
  format: "SPDX-2.3"
  output_formats:
    - json
    - xml
    - yaml
    - cyclonedx

  metadata:
    spdx_version: "SPDX-2.3"
    data_license: "CC0-1.0"
    creator: "Tool: GreenLang-SBOM-Generator-1.0"
    created: "ISO8601_TIMESTAMP"
    document_namespace: "https://sbom.greenlang.io/spdxdocs/"

  package_information:
    name: "${PACKAGE_NAME}"
    version: "${PACKAGE_VERSION}"
    supplier: "Organization: GreenLang Inc."
    download_location: "${REPO_URL}"
    files_analyzed: true
    verification_code: "${SHA256_HASH}"
    license_concluded: "${DETECTED_LICENSE}"
    license_declared: "${DECLARED_LICENSE}"
    copyright_text: "Copyright (c) 2025 GreenLang Inc."

  generation_pipeline:
    stages:
      - source_analysis
      - dependency_resolution
      - license_detection
      - vulnerability_scanning
      - cryptographic_signing
      - validation
      - storage
```

### CycloneDX Generation

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:${UUID}",
  "version": 1,
  "metadata": {
    "timestamp": "${ISO8601_TIMESTAMP}",
    "tools": [
      {
        "vendor": "GreenLang",
        "name": "SBOM-Generator",
        "version": "1.0.0"
      }
    ],
    "authors": [
      {
        "name": "GreenLang Security Team",
        "email": "security@greenlang.io"
      }
    ],
    "component": {
      "bom-ref": "${COMPONENT_REF}",
      "type": "application",
      "name": "${APPLICATION_NAME}",
      "version": "${APPLICATION_VERSION}",
      "purl": "pkg:generic/${APPLICATION_NAME}@${APPLICATION_VERSION}"
    },
    "properties": [
      {
        "name": "build.timestamp",
        "value": "${BUILD_TIMESTAMP}"
      },
      {
        "name": "git.commit",
        "value": "${GIT_COMMIT_HASH}"
      }
    ]
  },
  "components": [],
  "dependencies": [],
  "vulnerabilities": []
}
```

### Automated SBOM Pipeline

```python
# sbom_generator.py
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import subprocess

class SBOMGenerator:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.sbom_data = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": "",
            "documentNamespace": "",
            "creationInfo": {},
            "packages": [],
            "relationships": [],
            "externalDocumentRefs": []
        }

    def generate_sbom(self) -> Dict[str, Any]:
        """Generate complete SBOM for the project"""
        self.collect_metadata()
        self.scan_dependencies()
        self.detect_licenses()
        self.scan_vulnerabilities()
        self.calculate_hashes()
        self.sign_sbom()
        return self.sbom_data

    def collect_metadata(self):
        """Collect project metadata"""
        self.sbom_data["name"] = self.get_project_name()
        self.sbom_data["documentNamespace"] = f"https://sbom.greenlang.io/{self.get_project_name()}/{datetime.now().isoformat()}"
        self.sbom_data["creationInfo"] = {
            "created": datetime.now().isoformat(),
            "creators": ["Tool: GreenLang-SBOM-Generator-1.0"],
            "licenseListVersion": "3.20"
        }

    def scan_dependencies(self):
        """Scan all project dependencies"""
        dependencies = []

        # NPM dependencies
        if self.has_package_json():
            npm_deps = self.get_npm_dependencies()
            dependencies.extend(npm_deps)

        # Python dependencies
        if self.has_requirements():
            py_deps = self.get_python_dependencies()
            dependencies.extend(py_deps)

        # Go dependencies
        if self.has_go_mod():
            go_deps = self.get_go_dependencies()
            dependencies.extend(go_deps)

        # Container dependencies
        if self.has_dockerfile():
            container_deps = self.get_container_dependencies()
            dependencies.extend(container_deps)

        self.sbom_data["packages"] = dependencies

    def detect_licenses(self):
        """Detect licenses for all components"""
        for package in self.sbom_data["packages"]:
            package["licenseConcluded"] = self.detect_license(package)
            package["licenseInfoFromFiles"] = self.scan_license_files(package)

    def scan_vulnerabilities(self):
        """Scan for known vulnerabilities"""
        vulnerabilities = []

        for package in self.sbom_data["packages"]:
            vulns = self.check_vulnerabilities(package)
            if vulns:
                package["vulnerabilities"] = vulns
                vulnerabilities.extend(vulns)

        self.sbom_data["vulnerabilities"] = vulnerabilities

    def calculate_hashes(self):
        """Calculate cryptographic hashes"""
        for package in self.sbom_data["packages"]:
            if "downloadLocation" in package:
                package["checksums"] = [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": self.calculate_sha256(package["downloadLocation"])
                    },
                    {
                        "algorithm": "SHA512",
                        "checksumValue": self.calculate_sha512(package["downloadLocation"])
                    }
                ]

    def sign_sbom(self):
        """Digitally sign the SBOM"""
        sbom_json = json.dumps(self.sbom_data, sort_keys=True)
        signature = self.create_signature(sbom_json)

        self.sbom_data["signature"] = {
            "algorithm": "RSA-SHA256",
            "value": signature,
            "publicKey": self.get_public_key()
        }

    def validate_sbom(self) -> bool:
        """Validate SBOM against schema"""
        # Implement SPDX/CycloneDX schema validation
        return True

    def export_sbom(self, format: str = "json") -> str:
        """Export SBOM in specified format"""
        if format == "json":
            return json.dumps(self.sbom_data, indent=2)
        elif format == "xml":
            return self.convert_to_xml(self.sbom_data)
        elif format == "yaml":
            return self.convert_to_yaml(self.sbom_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
```

## 2. Component Tracking

### Component Registry

```yaml
# component-registry.yaml
component_registry:
  database:
    type: "PostgreSQL"
    schema:
      components:
        - id: "UUID PRIMARY KEY"
        - name: "VARCHAR(255) NOT NULL"
        - version: "VARCHAR(50) NOT NULL"
        - type: "ENUM('library', 'framework', 'tool', 'service')"
        - language: "VARCHAR(50)"
        - license: "VARCHAR(100)"
        - purl: "VARCHAR(500) UNIQUE"
        - source_repo: "VARCHAR(500)"
        - vendor: "VARCHAR(255)"
        - first_seen: "TIMESTAMP"
        - last_seen: "TIMESTAMP"
        - active: "BOOLEAN DEFAULT true"

      component_dependencies:
        - id: "UUID PRIMARY KEY"
        - parent_id: "UUID REFERENCES components(id)"
        - child_id: "UUID REFERENCES components(id)"
        - relationship_type: "VARCHAR(50)"
        - scope: "ENUM('compile', 'runtime', 'test', 'provided')"

      component_metadata:
        - component_id: "UUID REFERENCES components(id)"
        - metadata_type: "VARCHAR(100)"
        - metadata_value: "JSONB"
        - created_at: "TIMESTAMP"
        - updated_at: "TIMESTAMP"

  tracking:
    automated_discovery:
      - package_managers:
          - npm
          - pip
          - maven
          - go_modules
          - cargo
      - container_scanning:
          - docker_images
          - kubernetes_deployments
      - binary_analysis:
          - executable_scanning
          - library_detection

    manual_registration:
      approval_required: true
      fields:
        - name: "required"
        - version: "required"
        - license: "required"
        - source: "required"
        - justification: "required"

    update_frequency:
      automated: "hourly"
      manual_review: "weekly"
```

### Dependency Graph

```python
# dependency_graph.py
import networkx as nx
from typing import Dict, List, Set, Tuple

class DependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.component_data = {}

    def add_component(self, component_id: str, metadata: Dict):
        """Add a component to the dependency graph"""
        self.graph.add_node(component_id)
        self.component_data[component_id] = metadata

    def add_dependency(self, parent: str, child: str, relationship: str):
        """Add a dependency relationship"""
        self.graph.add_edge(parent, child, relationship=relationship)

    def find_critical_path(self) -> List[str]:
        """Find critical dependencies that affect many components"""
        centrality = nx.betweenness_centrality(self.graph)
        critical = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in critical[:10]]

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies"""
        return list(nx.simple_cycles(self.graph))

    def find_outdated(self, max_age_days: int = 180) -> List[str]:
        """Find outdated components"""
        outdated = []
        for component_id, metadata in self.component_data.items():
            if self.is_outdated(metadata, max_age_days):
                outdated.append(component_id)
        return outdated

    def calculate_risk_score(self, component_id: str) -> float:
        """Calculate risk score for a component"""
        score = 0.0

        # Check vulnerabilities
        if self.has_vulnerabilities(component_id):
            score += 5.0

        # Check license risk
        license_risk = self.get_license_risk(component_id)
        score += license_risk

        # Check dependency depth
        depth = self.get_dependency_depth(component_id)
        score += min(depth * 0.5, 3.0)

        # Check update frequency
        if self.is_abandoned(component_id):
            score += 2.0

        return min(score, 10.0)

    def generate_impact_analysis(self, component_id: str) -> Dict:
        """Analyze impact of updating/removing a component"""
        affected = list(nx.ancestors(self.graph, component_id))

        return {
            "directly_affected": list(self.graph.predecessors(component_id)),
            "total_affected": affected,
            "affected_count": len(affected),
            "critical_services": self.find_critical_services(affected),
            "estimated_effort": self.estimate_update_effort(component_id)
        }

    def export_sbom_graph(self) -> Dict:
        """Export dependency graph in SBOM format"""
        return {
            "components": self.component_data,
            "dependencies": [
                {
                    "parent": edge[0],
                    "child": edge[1],
                    "relationship": self.graph.edges[edge].get("relationship")
                }
                for edge in self.graph.edges()
            ],
            "graph_metrics": {
                "total_components": self.graph.number_of_nodes(),
                "total_dependencies": self.graph.number_of_edges(),
                "max_depth": self.get_max_depth(),
                "cycles_detected": len(self.detect_cycles())
            }
        }
```

## 3. Vulnerability Management

### Vulnerability Database Integration

```yaml
# vulnerability-management.yaml
vulnerability_management:
  data_sources:
    - source: "NVD"
      api_endpoint: "https://services.nvd.nist.gov/rest/json/cves/2.0"
      update_frequency: "hourly"
      priority: 1

    - source: "GitHub Advisory Database"
      api_endpoint: "https://api.github.com/advisories"
      update_frequency: "hourly"
      priority: 2

    - source: "OSV"
      api_endpoint: "https://osv.dev/api/v1"
      update_frequency: "daily"
      priority: 3

    - source: "Snyk"
      api_endpoint: "https://api.snyk.io/v1"
      api_key: "${SNYK_API_KEY}"
      update_frequency: "real-time"
      priority: 1

  scoring:
    cvss_v3:
      critical: ">= 9.0"
      high: "7.0 - 8.9"
      medium: "4.0 - 6.9"
      low: "0.1 - 3.9"

    epss_threshold: 0.1  # Exploit Prediction Scoring System

  remediation_slas:
    critical:
      detection_to_triage: "1 hour"
      triage_to_patch: "24 hours"
      patch_to_deployment: "48 hours"

    high:
      detection_to_triage: "4 hours"
      triage_to_patch: "72 hours"
      patch_to_deployment: "1 week"

    medium:
      detection_to_triage: "24 hours"
      triage_to_patch: "2 weeks"
      patch_to_deployment: "30 days"

    low:
      detection_to_triage: "1 week"
      triage_to_patch: "30 days"
      patch_to_deployment: "90 days"

  automated_patching:
    enabled: true
    auto_patch_categories:
      - "patch"
      - "minor"
    require_approval:
      - "major"
      - "breaking"
    test_before_merge: true
    rollback_on_failure: true

  alerting:
    channels:
      - type: "slack"
        webhook: "${SLACK_WEBHOOK}"
        severity: ["critical", "high"]

      - type: "email"
        recipients: ["security@greenlang.io"]
        severity: ["critical"]

      - type: "pagerduty"
        integration_key: "${PAGERDUTY_KEY}"
        severity: ["critical"]

      - type: "jira"
        project: "VULN"
        severity: ["all"]
```

### Vulnerability Scanner

```python
# vulnerability_scanner.py
import asyncio
import aiohttp
from typing import List, Dict, Any
from datetime import datetime

class VulnerabilityScanner:
    def __init__(self, sbom_data: Dict):
        self.sbom_data = sbom_data
        self.vulnerabilities = []
        self.api_endpoints = {
            "nvd": "https://services.nvd.nist.gov/rest/json/cves/2.0",
            "osv": "https://api.osv.dev/v1/query",
            "github": "https://api.github.com/advisories"
        }

    async def scan_all_components(self) -> List[Dict]:
        """Scan all SBOM components for vulnerabilities"""
        tasks = []

        async with aiohttp.ClientSession() as session:
            for component in self.sbom_data.get("packages", []):
                task = self.scan_component(session, component)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        self.vulnerabilities = [v for result in results for v in result if v]
        return self.vulnerabilities

    async def scan_component(self, session: aiohttp.ClientSession, component: Dict) -> List[Dict]:
        """Scan individual component for vulnerabilities"""
        vulnerabilities = []

        # Check NVD
        nvd_vulns = await self.check_nvd(session, component)
        vulnerabilities.extend(nvd_vulns)

        # Check OSV
        osv_vulns = await self.check_osv(session, component)
        vulnerabilities.extend(osv_vulns)

        # Check GitHub Advisory
        github_vulns = await self.check_github(session, component)
        vulnerabilities.extend(github_vulns)

        return self.deduplicate_vulnerabilities(vulnerabilities)

    async def check_nvd(self, session: aiohttp.ClientSession, component: Dict) -> List[Dict]:
        """Check NVD for vulnerabilities"""
        cpe = self.generate_cpe(component)
        params = {
            "cpeMatchString": cpe,
            "resultsPerPage": 100
        }

        try:
            async with session.get(self.api_endpoints["nvd"], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.parse_nvd_response(data, component)
        except Exception as e:
            print(f"NVD check failed: {e}")

        return []

    async def check_osv(self, session: aiohttp.ClientSession, component: Dict) -> List[Dict]:
        """Check OSV database for vulnerabilities"""
        payload = {
            "package": {
                "name": component.get("name"),
                "ecosystem": self.get_ecosystem(component)
            },
            "version": component.get("version")
        }

        try:
            async with session.post(self.api_endpoints["osv"], json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.parse_osv_response(data, component)
        except Exception as e:
            print(f"OSV check failed: {e}")

        return []

    def calculate_risk_score(self, vulnerability: Dict) -> float:
        """Calculate risk score for vulnerability"""
        base_score = vulnerability.get("cvss_score", 0)

        # Adjust based on exploitability
        if vulnerability.get("exploit_available"):
            base_score *= 1.5

        # Adjust based on age
        age_days = (datetime.now() - vulnerability.get("published_date")).days
        if age_days < 30:
            base_score *= 1.2

        # Adjust based on patch availability
        if not vulnerability.get("patch_available"):
            base_score *= 1.3

        return min(base_score, 10.0)

    def prioritize_vulnerabilities(self) -> List[Dict]:
        """Prioritize vulnerabilities for remediation"""
        for vuln in self.vulnerabilities:
            vuln["risk_score"] = self.calculate_risk_score(vuln)
            vuln["priority"] = self.get_priority(vuln["risk_score"])

        return sorted(self.vulnerabilities, key=lambda x: x["risk_score"], reverse=True)

    def generate_remediation_plan(self) -> Dict:
        """Generate automated remediation plan"""
        plan = {
            "immediate_actions": [],
            "scheduled_updates": [],
            "manual_review": [],
            "risk_acceptance": []
        }

        for vuln in self.prioritize_vulnerabilities():
            if vuln["priority"] == "critical":
                plan["immediate_actions"].append(self.create_patch_action(vuln))
            elif vuln["priority"] == "high":
                plan["scheduled_updates"].append(self.create_update_action(vuln))
            elif vuln.get("requires_code_change"):
                plan["manual_review"].append(vuln)
            else:
                plan["risk_acceptance"].append(vuln)

        return plan
```

## 4. Update Procedures

### Automated Update Pipeline

```yaml
# update-pipeline.yaml
update_pipeline:
  stages:
    - name: "detection"
      steps:
        - scan_for_updates
        - check_compatibility
        - assess_risk
        - create_update_branch

    - name: "testing"
      steps:
        - run_unit_tests
        - run_integration_tests
        - run_security_scans
        - run_performance_tests

    - name: "validation"
      steps:
        - validate_functionality
        - validate_security
        - validate_compliance
        - generate_impact_report

    - name: "approval"
      steps:
        - automated_approval_check
        - manual_review_if_required
        - sign_off_collection

    - name: "deployment"
      steps:
        - staged_rollout
        - monitor_metrics
        - rollback_if_needed
        - update_sbom

  update_policies:
    automatic_updates:
      patch_versions: true
      minor_versions: true
      major_versions: false
      security_patches: true

    approval_matrix:
      patch:
        dev: "automatic"
        staging: "automatic"
        production: "automatic after 24h"

      minor:
        dev: "automatic"
        staging: "automatic after 24h"
        production: "manual approval"

      major:
        dev: "manual approval"
        staging: "manual approval"
        production: "change board approval"

  rollback_criteria:
    - error_rate_increase: "> 5%"
    - response_time_increase: "> 20%"
    - test_failure_rate: "> 10%"
    - security_scan_failures: "any critical"
```

### Update Automation Script

```python
# update_automation.py
import subprocess
import json
from typing import Dict, List, Optional
import semver

class UpdateAutomation:
    def __init__(self, sbom_path: str):
        self.sbom_path = sbom_path
        self.sbom_data = self.load_sbom()
        self.update_queue = []

    def check_for_updates(self) -> List[Dict]:
        """Check all components for available updates"""
        updates = []

        for component in self.sbom_data["packages"]:
            latest_version = self.get_latest_version(component)

            if latest_version and self.is_update_available(component, latest_version):
                update = {
                    "component": component["name"],
                    "current_version": component["version"],
                    "latest_version": latest_version,
                    "update_type": self.classify_update(
                        component["version"],
                        latest_version
                    ),
                    "changelog": self.get_changelog(component, latest_version),
                    "breaking_changes": self.check_breaking_changes(
                        component,
                        latest_version
                    )
                }
                updates.append(update)

        return updates

    def classify_update(self, current: str, latest: str) -> str:
        """Classify update type using semantic versioning"""
        try:
            current_ver = semver.VersionInfo.parse(current)
            latest_ver = semver.VersionInfo.parse(latest)

            if latest_ver.major > current_ver.major:
                return "major"
            elif latest_ver.minor > current_ver.minor:
                return "minor"
            elif latest_ver.patch > current_ver.patch:
                return "patch"
            else:
                return "none"
        except:
            return "unknown"

    def create_update_pr(self, update: Dict) -> str:
        """Create pull request for component update"""
        branch_name = f"update-{update['component']}-{update['latest_version']}"

        # Create branch
        subprocess.run(["git", "checkout", "-b", branch_name])

        # Apply update
        self.apply_update(update)

        # Run tests
        test_results = self.run_tests()

        # Create PR if tests pass
        if test_results["passed"]:
            pr_body = self.generate_pr_description(update, test_results)
            pr_url = self.create_github_pr(branch_name, pr_body)
            return pr_url
        else:
            # Rollback if tests fail
            subprocess.run(["git", "checkout", "main"])
            subprocess.run(["git", "branch", "-D", branch_name])
            raise Exception(f"Tests failed for update: {update}")

    def apply_update(self, update: Dict):
        """Apply component update based on package manager"""
        component = update["component"]
        version = update["latest_version"]

        # Detect package manager and apply update
        if self.is_npm_package(component):
            subprocess.run(["npm", "install", f"{component}@{version}"])
        elif self.is_pip_package(component):
            subprocess.run(["pip", "install", f"{component}=={version}"])
        elif self.is_go_module(component):
            subprocess.run(["go", "get", f"{component}@{version}"])

        # Update SBOM
        self.update_sbom_component(component, version)

    def validate_update(self, update: Dict) -> Dict:
        """Validate update doesn't break functionality"""
        validation_results = {
            "tests_passed": False,
            "security_scan_passed": False,
            "performance_impact": None,
            "compatibility_check": False
        }

        # Run test suite
        validation_results["tests_passed"] = self.run_tests()["passed"]

        # Run security scan
        validation_results["security_scan_passed"] = self.run_security_scan()

        # Check performance impact
        validation_results["performance_impact"] = self.measure_performance_impact()

        # Check compatibility
        validation_results["compatibility_check"] = self.check_compatibility(update)

        return validation_results

    def generate_update_report(self) -> Dict:
        """Generate comprehensive update report"""
        return {
            "scan_date": datetime.now().isoformat(),
            "total_components": len(self.sbom_data["packages"]),
            "updates_available": len(self.update_queue),
            "critical_updates": self.count_critical_updates(),
            "estimated_effort": self.estimate_update_effort(),
            "risk_assessment": self.assess_update_risk(),
            "recommended_schedule": self.generate_update_schedule()
        }
```

## 5. Customer SBOM Reporting

### Customer Portal Integration

```yaml
# customer-sbom-portal.yaml
customer_portal:
  features:
    sbom_access:
      - real_time_view: true
      - download_formats:
          - SPDX
          - CycloneDX
          - JSON
          - PDF
      - api_access: true
      - webhook_notifications: true

    vulnerability_reporting:
      - real_time_alerts: true
      - severity_filtering: true
      - remediation_status: true
      - patch_timeline: true

    compliance_reporting:
      - license_summary: true
      - compliance_status: true
      - audit_trail: true
      - certification_documents: true

    customization:
      - custom_reports: true
      - alert_thresholds: true
      - notification_preferences: true
      - api_rate_limits: true

  api_endpoints:
    - path: "/api/v1/sbom/{product_id}"
      method: "GET"
      description: "Get current SBOM for product"
      authentication: "Bearer token"

    - path: "/api/v1/sbom/{product_id}/vulnerabilities"
      method: "GET"
      description: "Get vulnerability report"
      authentication: "Bearer token"

    - path: "/api/v1/sbom/{product_id}/licenses"
      method: "GET"
      description: "Get license report"
      authentication: "Bearer token"

    - path: "/api/v1/sbom/{product_id}/updates"
      method: "GET"
      description: "Get update history"
      authentication: "Bearer token"

    - path: "/api/v1/sbom/webhook"
      method: "POST"
      description: "Register webhook for updates"
      authentication: "Bearer token"

  reporting_templates:
    executive_summary:
      sections:
        - component_overview
        - vulnerability_summary
        - license_compliance
        - update_recommendations

    technical_report:
      sections:
        - detailed_component_list
        - dependency_graph
        - vulnerability_details
        - remediation_plan

    compliance_report:
      sections:
        - license_analysis
        - regulatory_compliance
        - audit_trail
        - certification_status
```

### Customer SBOM API

```python
# customer_sbom_api.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List, Dict
import json

app = FastAPI(title="GreenLang SBOM API", version="1.0.0")
security = HTTPBearer()

class CustomerSBOMAPI:
    @app.get("/api/v1/sbom/{product_id}")
    async def get_sbom(
        product_id: str,
        format: Optional[str] = "json",
        version: Optional[str] = "latest",
        credentials: HTTPAuthorizationCredentials = Security(security)
    ):
        """Get SBOM for customer product"""
        # Verify customer authorization
        customer = verify_customer_token(credentials.credentials)

        # Check customer has access to product
        if not has_product_access(customer, product_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Retrieve SBOM
        sbom = get_product_sbom(product_id, version)

        # Format SBOM
        if format == "spdx":
            return format_as_spdx(sbom)
        elif format == "cyclonedx":
            return format_as_cyclonedx(sbom)
        else:
            return sbom

    @app.get("/api/v1/sbom/{product_id}/vulnerabilities")
    async def get_vulnerabilities(
        product_id: str,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ):
        """Get vulnerability report for product"""
        customer = verify_customer_token(credentials.credentials)

        if not has_product_access(customer, product_id):
            raise HTTPException(status_code=403, detail="Access denied")

        vulnerabilities = get_product_vulnerabilities(product_id)

        # Filter by severity
        if severity:
            vulnerabilities = filter_by_severity(vulnerabilities, severity)

        # Filter by status
        if status:
            vulnerabilities = filter_by_status(vulnerabilities, status)

        return {
            "product_id": product_id,
            "scan_date": get_last_scan_date(product_id),
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "remediation_plan": generate_remediation_plan(vulnerabilities)
        }

    @app.post("/api/v1/sbom/webhook")
    async def register_webhook(
        webhook_config: Dict,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ):
        """Register webhook for SBOM updates"""
        customer = verify_customer_token(credentials.credentials)

        webhook_id = register_customer_webhook(
            customer_id=customer["id"],
            url=webhook_config["url"],
            events=webhook_config["events"],
            product_ids=webhook_config.get("product_ids", [])
        )

        return {
            "webhook_id": webhook_id,
            "status": "registered",
            "events": webhook_config["events"]
        }

    @app.get("/api/v1/sbom/{product_id}/compliance")
    async def get_compliance_report(
        product_id: str,
        framework: Optional[str] = None,
        credentials: HTTPAuthorizationCredentials = Security(security)
    ):
        """Get compliance report for product"""
        customer = verify_customer_token(credentials.credentials)

        if not has_product_access(customer, product_id):
            raise HTTPException(status_code=403, detail="Access denied")

        compliance_data = {
            "product_id": product_id,
            "report_date": datetime.now().isoformat(),
            "license_compliance": get_license_compliance(product_id),
            "regulatory_compliance": get_regulatory_compliance(product_id, framework),
            "security_compliance": get_security_compliance(product_id),
            "audit_trail": get_audit_trail(product_id)
        }

        return compliance_data

def generate_customer_report(customer_id: str, product_id: str, report_type: str) -> Dict:
    """Generate custom SBOM report for customer"""
    sbom_data = get_product_sbom(product_id)

    if report_type == "executive":
        return generate_executive_report(sbom_data)
    elif report_type == "technical":
        return generate_technical_report(sbom_data)
    elif report_type == "compliance":
        return generate_compliance_report(sbom_data)
    else:
        return sbom_data
```