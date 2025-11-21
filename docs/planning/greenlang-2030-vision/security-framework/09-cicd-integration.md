# CI/CD Security Integration

## 1. Security Pipeline Architecture

### Multi-Stage Security Pipeline

```yaml
# cicd-security-pipeline.yaml
security_pipeline:
  stages:
    pre_commit:
      hooks:
        - name: "secret_detection"
          tool: "git-secrets"
          action: "block_commit"
          timeout: 30

        - name: "code_formatting"
          tool: "pre-commit"
          action: "auto_fix"
          timeout: 60

        - name: "dependency_check"
          tool: "npm-audit / pip-check"
          action: "warn"
          timeout: 120

    commit:
      triggers:
        - "git push"
      actions:
        - "trigger_pipeline"
        - "notify_team"

    build:
      parallel_jobs:
        unit_tests:
          tool: "jest / pytest"
          coverage_required: 80
          timeout: 600

        sast_scan:
          tools:
            - "SonarQube"
            - "Semgrep"
          fail_on: ["critical", "high"]
          timeout: 1200

        dependency_scan:
          tools:
            - "Snyk"
            - "OWASP Dependency Check"
          fail_on: "critical_vulns"
          timeout: 600

        container_scan:
          tool: "Trivy"
          scan_targets:
            - "docker_image"
            - "Dockerfile"
          fail_on: ["critical", "high"]
          timeout: 300

        secrets_scan:
          tool: "TruffleHog"
          scan_depth: "full_history"
          fail_on: "any_secret"
          timeout: 600

    security_testing:
      api_security:
        tool: "OWASP ZAP"
        scan_type: "active"
        timeout: 1800

      integration_tests:
        security_tests:
          - "authentication_tests"
          - "authorization_tests"
          - "input_validation_tests"
          - "session_management_tests"
        timeout: 1200

    compliance_checks:
      license_compliance:
        tool: "FOSSA"
        fail_on: ["prohibited_licenses"]
        timeout: 300

      policy_validation:
        tool: "OPA (Open Policy Agent)"
        policies:
          - "infrastructure_policies"
          - "security_policies"
          - "compliance_policies"
        timeout: 180

      sbom_generation:
        tools:
          - "Syft"
          - "CycloneDX"
        output_formats: ["json", "xml"]
        storage: "artifact_repository"

    staging_deployment:
      pre_deployment:
        - "infrastructure_scan"
        - "configuration_validation"
        - "secrets_management_check"

      deployment:
        strategy: "blue_green"
        rollback_on_failure: true
        health_checks: true

      post_deployment:
        - "runtime_security_check"
        - "penetration_test_light"
        - "monitoring_validation"

    production_deployment:
      approval_required: true
      approvers:
        - "security_team"
        - "engineering_manager"

      deployment_window:
        days: ["Tuesday", "Wednesday", "Thursday"]
        hours: "10:00-16:00"
        exclude: "holidays"

      canary_deployment:
        enabled: true
        traffic_percentage: [10, 25, 50, 100]
        promotion_time: 3600  # seconds
        rollback_on_error: true

      post_deployment_validation:
        - "security_smoke_tests"
        - "compliance_verification"
        - "vulnerability_scan"
        - "log_monitoring"

  security_gates:
    quality_gate:
      conditions:
        - "security_rating >= A"
        - "critical_vulnerabilities == 0"
        - "high_vulnerabilities <= 3"
        - "code_coverage >= 80"
        - "license_compliance == true"

    approval_gate:
      required_for:
        - "production_deployments"
        - "infrastructure_changes"
        - "security_policy_changes"

      approvers:
        automatic:
          - "All checks passed"
          - "No security findings"
          - "Minor version bump"

        manual:
          - "Security findings exist"
          - "Major version changes"
          - "Infrastructure modifications"

  failure_handling:
    on_security_failure:
      actions:
        - "stop_pipeline"
        - "notify_security_team"
        - "create_security_ticket"
        - "block_deployment"

    on_critical_vulnerability:
      actions:
        - "emergency_notification"
        - "escalate_to_security_lead"
        - "mandatory_fix_required"
        - "audit_log_entry"

    rollback_triggers:
      - "critical_security_finding"
      - "failed_health_checks"
      - "elevated_error_rates"
      - "security_incident_detected"
```

### GitHub Actions Security Workflow

```yaml
# .github/workflows/security-pipeline.yml
name: Security Pipeline

on:
  push:
    branches: [main, develop, 'release/**']
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  secret-detection:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: TruffleHog Scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD

      - name: GitLeaks Scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  sast-analysis:
    runs-on: ubuntu-latest
    needs: secret-detection
    steps:
      - uses: actions/checkout@v3

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      - name: Semgrep Security Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/owasp-top-ten
            p/ci

      - name: CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  dependency-scan:
    runs-on: ubuntu-latest
    needs: secret-detection
    steps:
      - uses: actions/checkout@v3

      - name: Snyk Security Scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high --fail-on=all

      - name: OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'greenlang'
          path: '.'
          format: 'ALL'
          args: >
            --failOnCVSS 7
            --enableRetired

      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: dependency-check-report.json

  container-security:
    runs-on: ubuntu-latest
    needs: [sast-analysis, dependency-scan]
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker Image
        run: |
          docker build -t greenlang:${{ github.sha }} .

      - name: Trivy Container Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: greenlang:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Dockle Container Linting
        uses: hands-lab/dockle-action@v1
        with:
          image: greenlang:${{ github.sha }}
          exit-code: '1'
          exit-level: 'warn'

      - name: Anchore Container Scan
        uses: anchore/scan-action@v3
        with:
          image: greenlang:${{ github.sha }}
          fail-build: true
          severity-cutoff: high

  license-compliance:
    runs-on: ubuntu-latest
    needs: dependency-scan
    steps:
      - uses: actions/checkout@v3

      - name: FOSSA Analysis
        uses: fossas/fossa-action@main
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}

      - name: License Check
        run: |
          npm install -g license-checker
          license-checker --production --failOn 'GPL-3.0;AGPL-3.0'

  iac-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Terraform Security Scan
        uses: aquasecurity/tfsec-action@v1.0.0
        with:
          soft_fail: false

      - name: Checkov IaC Scan
        uses: bridgecrewio/checkov-action@master
        with:
          directory: infrastructure/
          framework: terraform
          output_format: sarif
          output_file_path: checkov-results.sarif

      - name: Terrascan
        uses: tenable/terrascan-action@main
        with:
          iac_type: 'terraform'
          iac_dir: 'infrastructure/'
          policy_type: 'all'
          fail_on_violation: true

  dast-scan:
    runs-on: ubuntu-latest
    needs: [container-security]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Test Environment
        run: |
          # Deploy application to test environment
          ./scripts/deploy-test.sh

      - name: OWASP ZAP Scan
        uses: zaproxy/action-baseline@v0.7.0
        with:
          target: 'https://test.greenlang.io'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

      - name: Nuclei Security Scan
        uses: projectdiscovery/nuclei-action@main
        with:
          target: 'https://test.greenlang.io'
          templates: 'cves,vulnerabilities'

  security-report:
    runs-on: ubuntu-latest
    needs: [sast-analysis, dependency-scan, container-security, license-compliance]
    if: always()
    steps:
      - name: Download All Artifacts
        uses: actions/download-artifact@v3

      - name: Generate Security Report
        run: |
          python scripts/generate-security-report.py

      - name: Upload to Security Dashboard
        run: |
          curl -X POST ${{ secrets.SECURITY_DASHBOARD_URL }}/reports \
            -H "Authorization: Bearer ${{ secrets.SECURITY_TOKEN }}" \
            -F "report=@security-report.json"

      - name: Notify Security Team
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Security scan failed for ${{ github.repository }}'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Jenkins Security Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        SONAR_TOKEN = credentials('sonarqube-token')
        SNYK_TOKEN = credentials('snyk-token')
        SECURITY_DASHBOARD = 'https://security.greenlang.io'
    }

    stages {
        stage('Pre-Scan Checks') {
            parallel {
                stage('Secret Detection') {
                    steps {
                        script {
                            sh '''
                                docker run --rm -v "$(pwd)":/src \
                                    trufflesecurity/trufflehog:latest \
                                    filesystem /src --json > trufflehog-results.json
                            '''

                            def results = readJSON file: 'trufflehog-results.json'
                            if (results.size() > 0) {
                                error("Secrets detected in repository!")
                            }
                        }
                    }
                }

                stage('Dependency Audit') {
                    steps {
                        sh 'npm audit --audit-level=high || true'
                        sh 'npm audit --json > npm-audit.json'
                    }
                }
            }
        }

        stage('Static Analysis') {
            parallel {
                stage('SonarQube') {
                    steps {
                        withSonarQubeEnv('SonarQube') {
                            sh '''
                                sonar-scanner \
                                    -Dsonar.projectKey=greenlang \
                                    -Dsonar.sources=. \
                                    -Dsonar.host.url=${SONAR_HOST_URL} \
                                    -Dsonar.login=${SONAR_TOKEN}
                            '''
                        }
                    }
                }

                stage('Semgrep') {
                    steps {
                        sh '''
                            docker run --rm -v "$(pwd):/src" \
                                returntocorp/semgrep \
                                --config=p/security-audit \
                                --config=p/owasp-top-ten \
                                --json --output=semgrep-results.json /src
                        '''
                    }
                }

                stage('Bandit (Python)') {
                    when {
                        expression { fileExists('requirements.txt') }
                    }
                    steps {
                        sh 'bandit -r . -f json -o bandit-results.json || true'
                    }
                }
            }
        }

        stage('Quality Gate') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Dependency Scanning') {
            parallel {
                stage('Snyk Test') {
                    steps {
                        sh '''
                            snyk test \
                                --severity-threshold=high \
                                --json > snyk-results.json || true
                        '''
                    }
                }

                stage('OWASP Dependency Check') {
                    steps {
                        dependencyCheck \
                            additionalArguments: '--format JSON --format HTML --failOnCVSS 7',
                            odcInstallation: 'OWASP-DC'

                        dependencyCheckPublisher \
                            pattern: '**/dependency-check-report.json'
                    }
                }
            }
        }

        stage('Container Security') {
            when {
                expression { fileExists('Dockerfile') }
            }
            steps {
                script {
                    docker.build("greenlang:${env.BUILD_ID}")

                    sh '''
                        trivy image \
                            --exit-code 1 \
                            --severity CRITICAL,HIGH \
                            --format json \
                            --output trivy-results.json \
                            greenlang:${BUILD_ID}
                    '''
                }
            }
        }

        stage('Infrastructure Security') {
            when {
                expression { fileExists('infrastructure/') }
            }
            steps {
                sh '''
                    docker run --rm -v "$(pwd):/src" \
                        bridgecrew/checkov \
                        -d /src/infrastructure \
                        --framework terraform \
                        --output json > checkov-results.json
                '''
            }
        }

        stage('License Compliance') {
            steps {
                sh '''
                    fossa analyze
                    fossa test --timeout 600
                '''
            }
        }

        stage('SBOM Generation') {
            steps {
                sh '''
                    syft packages . \
                        -o cyclonedx-json > sbom.json
                '''

                archiveArtifacts artifacts: 'sbom.json', fingerprint: true
            }
        }

        stage('Security Approval Gate') {
            when {
                branch 'main'
            }
            steps {
                script {
                    def securityReport = generateSecurityReport()

                    if (securityReport.critical > 0) {
                        emailext (
                            subject: "Critical Security Issues - Build ${env.BUILD_NUMBER}",
                            body: securityReport.summary,
                            to: "${env.SECURITY_TEAM_EMAIL}"
                        )

                        input message: 'Critical security issues found. Approve deployment?',
                              submitter: 'security-team'
                    }
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                script {
                    deploy(environment: 'staging', approval: false)

                    // Run DAST after deployment
                    sh '''
                        docker run --rm \
                            -v "$(pwd):/zap/wrk" \
                            owasp/zap2docker-stable \
                            zap-baseline.py \
                            -t https://staging.greenlang.io \
                            -r zap-report.html
                    '''
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?',
                      submitter: 'release-managers,security-team'

                script {
                    deploy(environment: 'production', approval: true)
                }
            }
        }

        stage('Post-Deployment Validation') {
            when {
                branch 'main'
            }
            steps {
                sh './scripts/security-smoke-tests.sh'

                sh '''
                    trivy image \
                        --severity CRITICAL \
                        greenlang:production
                '''
            }
        }
    }

    post {
        always {
            script {
                // Collect all security reports
                def reports = [
                    'trufflehog-results.json',
                    'semgrep-results.json',
                    'snyk-results.json',
                    'trivy-results.json',
                    'checkov-results.json'
                ]

                // Upload to security dashboard
                reports.each { report ->
                    if (fileExists(report)) {
                        sh """
                            curl -X POST ${SECURITY_DASHBOARD}/api/reports \
                                -H 'Content-Type: application/json' \
                                -H 'Authorization: Bearer \${SECURITY_TOKEN}' \
                                -d @${report}
                        """
                    }
                }

                // Generate unified report
                sh 'python scripts/consolidate-security-reports.py'

                // Archive reports
                archiveArtifacts artifacts: '*-results.json,security-report.pdf',
                                fingerprint: true
            }
        }

        failure {
            emailext (
                subject: "Security Pipeline Failed - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: """
                    Security pipeline failed for ${env.JOB_NAME} build #${env.BUILD_NUMBER}

                    Check console output: ${env.BUILD_URL}console

                    Security reports: ${env.BUILD_URL}artifact/
                """,
                to: "${env.SECURITY_TEAM_EMAIL}"
            )

            slackSend (
                color: 'danger',
                message: "Security pipeline failed: ${env.JOB_NAME} #${env.BUILD_NUMBER} - ${env.BUILD_URL}"
            )
        }

        success {
            script {
                if (env.BRANCH_NAME == 'main') {
                    slackSend (
                        color: 'good',
                        message: "Deployment successful: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
                    )
                }
            }
        }
    }
}

def generateSecurityReport() {
    def report = [
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
        summary: ''
    ]

    // Parse all security scan results
    // Aggregate findings
    // Generate summary

    return report
}

def deploy(Map config) {
    echo "Deploying to ${config.environment}"

    if (config.approval) {
        // Production deployment with canary
        sh """
            kubectl apply -f k8s/${config.environment}/
            kubectl rollout status deployment/greenlang -n ${config.environment}
        """
    } else {
        // Staging deployment
        sh """
            kubectl apply -f k8s/${config.environment}/
        """
    }
}
```

## 2. Security Tool Integration

### Tool Configuration Management

```python
# security_tools_orchestrator.py
from typing import Dict, List, Optional
from datetime import datetime
import subprocess
import json

class SecurityToolsOrchestrator:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.tools = self.initialize_tools()
        self.results = []

    def initialize_tools(self) -> Dict:
        """Initialize all security scanning tools"""
        return {
            "sast": {
                "sonarqube": SonarQubeScanner(self.config["sonarqube"]),
                "semgrep": SemgrepScanner(self.config["semgrep"]),
                "bandit": BanditScanner(self.config["bandit"])
            },
            "sca": {
                "snyk": SnykScanner(self.config["snyk"]),
                "dependency_check": DependencyCheckScanner(self.config["owasp_dc"])
            },
            "container": {
                "trivy": TrivyScanner(self.config["trivy"]),
                "anchore": AnchoreScanner(self.config["anchore"])
            },
            "dast": {
                "zap": ZAPScanner(self.config["zap"]),
                "nuclei": NucleiScanner(self.config["nuclei"])
            },
            "secrets": {
                "trufflehog": TruffleHogScanner(self.config["trufflehog"]),
                "gitleaks": GitLeaksScanner(self.config["gitleaks"])
            },
            "iac": {
                "checkov": CheckovScanner(self.config["checkov"]),
                "tfsec": TFSecScanner(self.config["tfsec"])
            }
        }

    async def run_security_scans(self, scan_config: Dict) -> Dict:
        """Run configured security scans"""
        results = {
            "scan_id": self.generate_scan_id(),
            "started_at": datetime.now(),
            "scan_type": scan_config["type"],
            "target": scan_config["target"],
            "findings": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": [],
                "info": []
            },
            "summary": {}
        }

        # Run scans based on configuration
        if "sast" in scan_config["enabled_scans"]:
            sast_results = await self.run_sast_scans(scan_config)
            self.merge_results(results, sast_results)

        if "sca" in scan_config["enabled_scans"]:
            sca_results = await self.run_sca_scans(scan_config)
            self.merge_results(results, sca_results)

        if "container" in scan_config["enabled_scans"]:
            container_results = await self.run_container_scans(scan_config)
            self.merge_results(results, container_results)

        if "secrets" in scan_config["enabled_scans"]:
            secrets_results = await self.run_secrets_scans(scan_config)
            self.merge_results(results, secrets_results)

        # Generate summary
        results["summary"] = self.generate_scan_summary(results)
        results["completed_at"] = datetime.now()

        # Store results
        await self.store_results(results)

        # Check if scan should pass/fail
        results["pass"] = self.evaluate_scan_results(results)

        return results

    async def run_sast_scans(self, config: Dict) -> Dict:
        """Run SAST scans"""
        results = {"sast": {}}

        # SonarQube
        if "sonarqube" in config.get("sast_tools", []):
            sonar_results = await self.tools["sast"]["sonarqube"].scan(config["target"])
            results["sast"]["sonarqube"] = sonar_results

        # Semgrep
        if "semgrep" in config.get("sast_tools", []):
            semgrep_results = await self.tools["sast"]["semgrep"].scan(config["target"])
            results["sast"]["semgrep"] = semgrep_results

        return results

    def evaluate_scan_results(self, results: Dict) -> bool:
        """Evaluate if scan results pass quality gates"""
        gates = self.config["quality_gates"]

        # Check critical vulnerabilities
        if len(results["findings"]["critical"]) > gates["max_critical"]:
            return False

        # Check high vulnerabilities
        if len(results["findings"]["high"]) > gates["max_high"]:
            return False

        # Check security rating
        if results["summary"].get("security_rating", "E") < gates["min_security_rating"]:
            return False

        # Check code coverage
        if results["summary"].get("coverage", 0) < gates["min_coverage"]:
            return False

        return True

    def generate_security_report(self, results: Dict) -> str:
        """Generate comprehensive security report"""
        report = {
            "executive_summary": self.create_executive_summary(results),
            "findings_overview": self.summarize_findings(results),
            "detailed_findings": results["findings"],
            "tool_reports": self.aggregate_tool_reports(results),
            "recommendations": self.generate_recommendations(results),
            "compliance_status": self.check_compliance_status(results),
            "trend_analysis": self.analyze_trends(results)
        }

        return json.dumps(report, indent=2)

    async def create_security_ticket(self, finding: Dict) -> str:
        """Create security ticket for finding"""
        ticket = {
            "title": f"[{finding['severity']}] {finding['title']}",
            "description": finding["description"],
            "severity": finding["severity"],
            "component": finding["component"],
            "remediation": finding["remediation"],
            "references": finding.get("references", []),
            "labels": ["security", finding["severity"].lower(), finding["category"]]
        }

        # Create ticket in issue tracker (Jira, GitHub Issues, etc.)
        ticket_id = await self.issue_tracker.create_ticket(ticket)

        return ticket_id

class SonarQubeScanner:
    def __init__(self, config: Dict):
        self.config = config
        self.server_url = config["server_url"]
        self.token = config["token"]

    async def scan(self, target: str) -> Dict:
        """Run SonarQube scan"""
        cmd = [
            "sonar-scanner",
            f"-Dsonar.projectKey={self.config['project_key']}",
            f"-Dsonar.sources={target}",
            f"-Dsonar.host.url={self.server_url}",
            f"-Dsonar.login={self.token}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Get scan results from API
        results = await self.get_scan_results()

        return results

    async def get_scan_results(self) -> Dict:
        """Fetch scan results from SonarQube API"""
        # Implementation to fetch results
        pass

class TrivyScanner:
    def __init__(self, config: Dict):
        self.config = config

    async def scan(self, image: str) -> Dict:
        """Run Trivy container scan"""
        cmd = [
            "trivy",
            "image",
            "--format", "json",
            "--severity", "CRITICAL,HIGH,MEDIUM",
            "--exit-code", "0",  # Don't fail, just report
            image
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            raise Exception(f"Trivy scan failed: {result.stderr}")

class SnykScanner:
    def __init__(self, config: Dict):
        self.config = config
        self.token = config["token"]

    async def scan(self, target: str) -> Dict:
        """Run Snyk dependency scan"""
        cmd = [
            "snyk",
            "test",
            "--json",
            "--severity-threshold=high",
            f"--org={self.config['org']}",
            target
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={"SNYK_TOKEN": self.token}
        )

        return json.loads(result.stdout) if result.stdout else {}
```

## 3. Automated Security Reporting

### Security Dashboard

```python
# security_dashboard.py
from flask import Flask, jsonify, request
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go

app = Flask(__name__)

class SecurityDashboard:
    def __init__(self, database):
        self.db = database

    @app.route('/api/security/overview')
    def get_security_overview(self):
        """Get security overview dashboard data"""
        overview = {
            "current_status": {
                "security_score": self.calculate_security_score(),
                "open_vulnerabilities": self.count_open_vulnerabilities(),
                "critical_issues": self.count_critical_issues(),
                "sla_compliance": self.calculate_sla_compliance()
            },
            "trends": {
                "vulnerabilities_over_time": self.get_vulnerability_trend(30),
                "mttr_trend": self.get_mttr_trend(30),
                "security_score_trend": self.get_security_score_trend(30)
            },
            "breakdown": {
                "by_severity": self.group_by_severity(),
                "by_component": self.group_by_component(),
                "by_category": self.group_by_category()
            },
            "recent_scans": self.get_recent_scans(10),
            "alerts": self.get_active_alerts()
        }

        return jsonify(overview)

    @app.route('/api/security/pipeline/<pipeline_id>')
    def get_pipeline_report(self, pipeline_id: str):
        """Get security report for specific pipeline run"""
        report = {
            "pipeline_id": pipeline_id,
            "scan_results": self.get_pipeline_scans(pipeline_id),
            "findings": self.get_pipeline_findings(pipeline_id),
            "quality_gate": self.get_quality_gate_status(pipeline_id),
            "artifacts": self.get_pipeline_artifacts(pipeline_id)
        }

        return jsonify(report)

    @app.route('/api/security/metrics')
    def get_security_metrics(self):
        """Get comprehensive security metrics"""
        metrics = {
            "kpis": {
                "mttd": self.calculate_mttd(),
                "mttr": self.calculate_mttr(),
                "vulnerability_density": self.calculate_vuln_density(),
                "fix_rate": self.calculate_fix_rate()
            },
            "compliance": {
                "soc2_controls": self.check_soc2_controls(),
                "iso27001_controls": self.check_iso27001_controls(),
                "policy_compliance": self.check_policy_compliance()
            },
            "coverage": {
                "sast_coverage": self.calculate_sast_coverage(),
                "sca_coverage": self.calculate_sca_coverage(),
                "dast_coverage": self.calculate_dast_coverage()
            }
        }

        return jsonify(metrics)

    def generate_executive_report(self, period: str) -> Dict:
        """Generate executive security report"""
        data = self.db.get_data_for_period(period)

        report = {
            "period": period,
            "executive_summary": {
                "security_posture": self.assess_security_posture(data),
                "key_metrics": self.extract_key_metrics(data),
                "major_incidents": self.summarize_incidents(data),
                "trend": self.determine_trend(data)
            },
            "achievements": {
                "vulnerabilities_fixed": self.count_fixed_vulnerabilities(data),
                "improvements_implemented": self.list_improvements(data),
                "certifications_achieved": self.list_certifications(data)
            },
            "concerns": {
                "open_critical_issues": self.list_critical_issues(data),
                "overdue_remediations": self.list_overdue_items(data),
                "emerging_threats": self.identify_emerging_threats(data)
            },
            "recommendations": self.generate_executive_recommendations(data)
        }

        return report

    def create_vulnerability_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create vulnerability trend chart"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['critical'],
            name='Critical',
            line=dict(color='red', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['high'],
            name='High',
            line=dict(color='orange', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['medium'],
            name='Medium',
            line=dict(color='yellow', width=2)
        ))

        fig.update_layout(
            title='Vulnerability Trend',
            xaxis_title='Date',
            yaxis_title='Count',
            hovermode='x unified'
        )

        return fig
```

This comprehensive CI/CD security integration framework ensures that security is embedded at every stage of the software development lifecycle, from pre-commit hooks to production deployment validation, with automated reporting and continuous monitoring.