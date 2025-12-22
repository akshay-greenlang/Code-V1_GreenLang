"""
GreenLang Framework - Agent Scorer

Automated scoring engine for GreenLang AI agents.
Analyzes agent code and structure against quality standards.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import json
import os
import re


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension_name: str
    score: float
    max_score: float
    weight: float
    weighted_score: float
    findings: List[str]
    recommendations: List[str]
    sub_scores: List["DimensionScore"] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        """Score as percentage."""
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0


@dataclass
class ScoreReport:
    """Complete scoring report for an agent."""
    agent_id: str
    agent_name: str
    agent_path: str
    total_score: float
    max_possible_score: float
    percentage_score: float
    grade: str
    certification_level: str
    dimension_scores: List[DimensionScore]
    critical_issues: List[str]
    recommendations: List[str]
    strengths: List[str]
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""

    def __post_init__(self):
        """Compute hash after initialization."""
        if not self.computation_hash:
            self.computation_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of scoring."""
        data = {
            "agent_id": self.agent_id,
            "total_score": self.total_score,
            "percentage_score": self.percentage_score,
            "scored_at": self.scored_at.isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class AgentScorer:
    """
    Automated agent quality scorer.

    Analyzes agent code structure, documentation, tests,
    and implementation against GreenLang quality standards.

    Usage:
        >>> scorer = AgentScorer()
        >>> report = scorer.score_agent("/path/to/GL-001_AGENT")
        >>> print(f"Score: {report.percentage_score:.1f}% ({report.grade})")
    """

    def __init__(self, standard=None):
        """Initialize scorer with quality standard."""
        from ..standards.quality_standards import GREENLANG_STANDARD
        self.standard = standard or GREENLANG_STANDARD

    def score_agent(self, agent_path: str) -> ScoreReport:
        """
        Score an agent against quality standards.

        Args:
            agent_path: Path to agent directory

        Returns:
            ScoreReport with detailed scores
        """
        agent_path = Path(agent_path)

        if not agent_path.exists():
            raise ValueError(f"Agent path does not exist: {agent_path}")

        # Extract agent info
        agent_id, agent_name = self._extract_agent_info(agent_path)

        # Score each dimension
        dimension_scores = []
        for dimension in self.standard.dimensions:
            score = self._score_dimension(agent_path, dimension)
            dimension_scores.append(score)

        # Calculate totals
        total_score = sum(ds.weighted_score for ds in dimension_scores)
        max_possible = sum(ds.weight for ds in dimension_scores)
        percentage = (total_score / max_possible * 100) if max_possible > 0 else 0

        # Get grade and certification
        grade = self.standard.get_grade(percentage)
        certification = self.standard.get_certification_level(percentage)

        # Extract findings
        critical_issues = self._extract_critical_issues(dimension_scores)
        recommendations = self._extract_recommendations(dimension_scores)
        strengths = self._extract_strengths(dimension_scores)

        return ScoreReport(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_path=str(agent_path),
            total_score=round(total_score, 2),
            max_possible_score=max_possible,
            percentage_score=round(percentage, 2),
            grade=grade,
            certification_level=certification,
            dimension_scores=dimension_scores,
            critical_issues=critical_issues,
            recommendations=recommendations,
            strengths=strengths,
        )

    def _extract_agent_info(self, agent_path: Path) -> Tuple[str, str]:
        """Extract agent ID and name from path."""
        dir_name = agent_path.name

        # Try to extract GL-XXX pattern
        match = re.match(r"(GL-\d+)[_-](.+)", dir_name)
        if match:
            return match.group(1), match.group(2)

        # Check pack.yaml
        pack_yaml = agent_path / "pack.yaml"
        if pack_yaml.exists():
            import yaml
            with open(pack_yaml) as f:
                pack = yaml.safe_load(f)
                return pack.get("agent_id", dir_name), pack.get("name", dir_name)

        return dir_name, dir_name

    def _score_dimension(
        self,
        agent_path: Path,
        dimension,
    ) -> DimensionScore:
        """Score a single quality dimension."""
        findings = []
        recommendations = []
        sub_scores = []

        # Score sub-dimensions if present
        if dimension.sub_dimensions:
            total_sub_score = 0
            total_sub_weight = 0

            for sub_dim in dimension.sub_dimensions:
                sub_score = self._evaluate_sub_dimension(agent_path, sub_dim)
                sub_scores.append(sub_score)
                total_sub_score += sub_score.weighted_score
                total_sub_weight += sub_score.weight
                findings.extend(sub_score.findings)
                recommendations.extend(sub_score.recommendations)

            # Calculate weighted average
            if total_sub_weight > 0:
                score = (total_sub_score / total_sub_weight) * 100
            else:
                score = 0
        else:
            # Direct evaluation
            score, findings, recommendations = self._evaluate_dimension_directly(
                agent_path, dimension
            )

        weighted_score = (score / 100) * dimension.weight

        return DimensionScore(
            dimension_name=dimension.name,
            score=round(score, 2),
            max_score=100.0,
            weight=dimension.weight,
            weighted_score=round(weighted_score, 2),
            findings=findings,
            recommendations=recommendations,
            sub_scores=sub_scores,
        )

    def _evaluate_sub_dimension(
        self,
        agent_path: Path,
        sub_dimension,
    ) -> DimensionScore:
        """Evaluate a sub-dimension."""
        score, findings, recommendations = self._evaluate_dimension_directly(
            agent_path, sub_dimension
        )

        return DimensionScore(
            dimension_name=sub_dimension.name,
            score=round(score, 2),
            max_score=100.0,
            weight=sub_dimension.weight,
            weighted_score=round((score / 100) * sub_dimension.weight, 2),
            findings=findings,
            recommendations=recommendations,
        )

    def _evaluate_dimension_directly(
        self,
        agent_path: Path,
        dimension,
    ) -> Tuple[float, List[str], List[str]]:
        """Directly evaluate dimension based on criteria."""
        findings = []
        recommendations = []
        criteria_met = 0
        total_criteria = len(dimension.evaluation_criteria) or 1

        dim_name = dimension.name.lower()

        # Mathematical Rigor checks
        if "formula" in dim_name or "mathematical" in dim_name:
            score, f, r = self._check_mathematical_rigor(agent_path)
            return score, f, r

        # Provenance checks
        if "provenance" in dim_name or "sha" in dim_name:
            score, f, r = self._check_provenance_tracking(agent_path)
            return score, f, r

        # Schema checks
        if "schema" in dim_name or "validation" in dim_name:
            score, f, r = self._check_data_models(agent_path)
            return score, f, r

        # Test coverage checks
        if "test" in dim_name or "coverage" in dim_name:
            score, f, r = self._check_test_coverage(agent_path)
            return score, f, r

        # API checks
        if "api" in dim_name or "rest" in dim_name:
            score, f, r = self._check_api_quality(agent_path)
            return score, f, r

        # Docker/deployment checks
        if "docker" in dim_name or "container" in dim_name:
            score, f, r = self._check_containerization(agent_path)
            return score, f, r

        # Kubernetes checks
        if "kubernetes" in dim_name or "k8s" in dim_name:
            score, f, r = self._check_kubernetes(agent_path)
            return score, f, r

        # Documentation checks
        if "readme" in dim_name or "documentation" in dim_name:
            score, f, r = self._check_documentation(agent_path)
            return score, f, r

        # Explainability checks
        if "explainability" in dim_name or "shap" in dim_name or "lime" in dim_name:
            score, f, r = self._check_explainability(agent_path)
            return score, f, r

        # Default: check file presence
        return self._default_evaluation(agent_path, dimension)

    def _check_mathematical_rigor(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check mathematical implementation quality."""
        findings = []
        recommendations = []
        score = 0

        calculators_dir = agent_path / "calculators"
        if calculators_dir.exists():
            calc_files = list(calculators_dir.glob("*.py"))
            if calc_files:
                findings.append(f"Found {len(calc_files)} calculator modules")
                score += 30

                # Check for docstrings with formulas
                formula_refs = 0
                for calc_file in calc_files:
                    content = calc_file.read_text(errors='ignore')
                    if "Reference:" in content or "Eq." in content or "equation" in content.lower():
                        formula_refs += 1
                    if "def " in content:
                        score += 5

                if formula_refs > 0:
                    findings.append(f"Formula references found in {formula_refs} files")
                    score += 20
                else:
                    recommendations.append("Add formula references to calculator docstrings")
            else:
                recommendations.append("Add calculator modules with mathematical implementations")
        else:
            recommendations.append("Create calculators/ directory with domain calculations")

        return min(score, 100), findings, recommendations

    def _check_provenance_tracking(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check SHA-256 provenance tracking."""
        findings = []
        recommendations = []
        score = 0

        # Search for hash/provenance patterns
        py_files = list(agent_path.rglob("*.py"))
        sha_files = 0
        hash_patterns = ["sha256", "computation_hash", "provenance", "hashlib"]

        for py_file in py_files:
            try:
                content = py_file.read_text(errors='ignore').lower()
                if any(p in content for p in hash_patterns):
                    sha_files += 1
            except Exception:
                pass

        if sha_files > 0:
            findings.append(f"Provenance tracking found in {sha_files} files")
            score += min(sha_files * 15, 70)
        else:
            recommendations.append("Add SHA-256 provenance tracking to calculations")

        # Check for result schemas with hashes
        schemas_file = agent_path / "core" / "schemas.py"
        if schemas_file.exists():
            content = schemas_file.read_text(errors='ignore')
            if "computation_hash" in content:
                findings.append("Schema includes computation_hash field")
                score += 30
            else:
                recommendations.append("Add computation_hash field to result schemas")

        return min(score, 100), findings, recommendations

    def _check_data_models(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check data model quality."""
        findings = []
        recommendations = []
        score = 0

        schemas_file = agent_path / "core" / "schemas.py"
        if schemas_file.exists():
            content = schemas_file.read_text(errors='ignore')
            score += 30
            findings.append("schemas.py found")

            # Check for Pydantic
            if "pydantic" in content.lower() or "BaseModel" in content:
                score += 25
                findings.append("Pydantic models used")
            else:
                recommendations.append("Use Pydantic for data validation")

            # Check for validators
            if "validator" in content or "field_validator" in content:
                score += 20
                findings.append("Field validators implemented")
            else:
                recommendations.append("Add field validators for input validation")

            # Count model classes
            model_count = content.count("class ") - content.count("class Meta")
            if model_count > 5:
                score += 15
                findings.append(f"Found {model_count} data models")

            # Check for type hints
            if ": str" in content or ": float" in content or ": int" in content:
                score += 10
                findings.append("Type hints used")
        else:
            recommendations.append("Create core/schemas.py with Pydantic models")

        return min(score, 100), findings, recommendations

    def _check_test_coverage(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check test coverage."""
        findings = []
        recommendations = []
        score = 0

        tests_dir = agent_path / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            score += 20
            findings.append(f"Found {len(test_files)} test files")

            if len(test_files) >= 3:
                score += 30
            elif len(test_files) >= 1:
                score += 15
                recommendations.append("Add more test files for better coverage")
            else:
                recommendations.append("Add test files (test_*.py)")

            # Check for conftest
            if (tests_dir / "conftest.py").exists():
                score += 15
                findings.append("conftest.py with fixtures found")
            else:
                recommendations.append("Add conftest.py with shared fixtures")

            # Check for pytest markers
            for test_file in test_files[:3]:
                content = test_file.read_text(errors='ignore')
                if "@pytest" in content or "def test_" in content:
                    score += 10
                    break
        else:
            recommendations.append("Create tests/ directory with pytest tests")

        return min(score, 100), findings, recommendations

    def _check_api_quality(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check API implementation quality."""
        findings = []
        recommendations = []
        score = 0

        api_dir = agent_path / "api"
        if api_dir.exists():
            score += 20
            findings.append("api/ directory found")

            # Check for REST API
            rest_file = api_dir / "rest_api.py"
            if rest_file.exists():
                content = rest_file.read_text(errors='ignore')
                score += 25
                findings.append("REST API implemented")

                if "FastAPI" in content:
                    score += 15
                    findings.append("FastAPI used")

                if "@router" in content or "@app" in content:
                    score += 10

            # Check for GraphQL
            graphql_file = api_dir / "graphql_schema.py"
            if graphql_file.exists():
                score += 20
                findings.append("GraphQL schema implemented")
            else:
                recommendations.append("Add GraphQL API for flexible queries")

            # Check for middleware
            if (api_dir / "middleware.py").exists():
                score += 10
                findings.append("API middleware implemented")
        else:
            recommendations.append("Create api/ directory with REST endpoints")

        return min(score, 100), findings, recommendations

    def _check_containerization(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check Docker configuration."""
        findings = []
        recommendations = []
        score = 0

        dockerfile = agent_path / "Dockerfile"
        if dockerfile.exists():
            content = dockerfile.read_text(errors='ignore')
            score += 40
            findings.append("Dockerfile found")

            if "FROM" in content and "AS" in content:
                score += 20
                findings.append("Multi-stage build used")
            else:
                recommendations.append("Use multi-stage Docker build")

            if "HEALTHCHECK" in content:
                score += 15
                findings.append("Health check configured")
            else:
                recommendations.append("Add HEALTHCHECK instruction")

            if "USER" in content and "root" not in content.split("USER")[-1][:20]:
                score += 15
                findings.append("Non-root user configured")
            else:
                recommendations.append("Run as non-root user")

            if "EXPOSE" in content:
                score += 10
        else:
            recommendations.append("Add Dockerfile for containerization")

        # Check docker-compose
        compose_file = agent_path / "docker-compose.yaml"
        if not compose_file.exists():
            compose_file = agent_path / "docker-compose.yml"

        if compose_file.exists():
            findings.append("docker-compose.yaml found")
            score = min(score + 10, 100)

        return min(score, 100), findings, recommendations

    def _check_kubernetes(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check Kubernetes configuration."""
        findings = []
        recommendations = []
        score = 0

        k8s_dir = agent_path / "deploy" / "kubernetes"
        if not k8s_dir.exists():
            k8s_dir = agent_path / "k8s"

        if k8s_dir.exists():
            yaml_files = list(k8s_dir.glob("*.yaml")) + list(k8s_dir.glob("*.yml"))
            if yaml_files:
                score += 30
                findings.append(f"Found {len(yaml_files)} K8s manifests")

                for yaml_file in yaml_files:
                    content = yaml_file.read_text(errors='ignore')
                    if "kind: Deployment" in content:
                        score += 20
                        findings.append("Deployment manifest found")
                    if "kind: Service" in content:
                        score += 15
                        findings.append("Service manifest found")
                    if "HorizontalPodAutoscaler" in content:
                        score += 15
                        findings.append("HPA configured")
                    if "PodDisruptionBudget" in content:
                        score += 10
                        findings.append("PDB configured")
            else:
                recommendations.append("Add Kubernetes YAML manifests")
        else:
            recommendations.append("Create deploy/kubernetes/ directory with manifests")

        return min(score, 100), findings, recommendations

    def _check_documentation(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check documentation quality."""
        findings = []
        recommendations = []
        score = 0

        readme = agent_path / "README.md"
        if readme.exists():
            content = readme.read_text(errors='ignore')
            score += 30
            findings.append("README.md found")

            # Check for sections
            if "## " in content:
                sections = content.count("## ")
                if sections >= 5:
                    score += 25
                    findings.append(f"Well-structured with {sections} sections")
                else:
                    score += 10
                    recommendations.append("Add more documentation sections")

            if "```" in content:
                score += 15
                findings.append("Code examples included")
            else:
                recommendations.append("Add code examples to README")

            if "install" in content.lower():
                score += 10
                findings.append("Installation instructions present")

            if "api" in content.lower() or "endpoint" in content.lower():
                score += 10
                findings.append("API documentation present")
        else:
            recommendations.append("Create README.md with documentation")

        # Check for pack.yaml
        if (agent_path / "pack.yaml").exists():
            score += 10
            findings.append("pack.yaml manifest found")

        return min(score, 100), findings, recommendations

    def _check_explainability(
        self,
        agent_path: Path,
    ) -> Tuple[float, List[str], List[str]]:
        """Check explainability implementation."""
        findings = []
        recommendations = []
        score = 0

        explain_dir = agent_path / "explainability"
        if explain_dir.exists():
            score += 30
            findings.append("explainability/ directory found")

            py_files = list(explain_dir.glob("*.py"))
            for py_file in py_files:
                name = py_file.name.lower()
                if "shap" in name:
                    score += 20
                    findings.append("SHAP explainer implemented")
                if "lime" in name:
                    score += 15
                    findings.append("LIME explainer implemented")
                if "causal" in name:
                    score += 15
                    findings.append("Causal analysis implemented")
                if "rationale" in name:
                    score += 10
                    findings.append("Engineering rationale generator found")
        else:
            recommendations.append("Add explainability/ module with SHAP/LIME")

        return min(score, 100), findings, recommendations

    def _default_evaluation(
        self,
        agent_path: Path,
        dimension,
    ) -> Tuple[float, List[str], List[str]]:
        """Default evaluation for unhandled dimensions."""
        findings = []
        recommendations = []

        # Check general structure
        score = 50  # Base score for existence

        if (agent_path / "core").exists():
            score += 10
        if (agent_path / "__init__.py").exists():
            score += 10
        if (agent_path / "requirements.txt").exists():
            score += 10

        return min(score, 100), findings, recommendations

    def _extract_critical_issues(
        self,
        dimension_scores: List[DimensionScore],
    ) -> List[str]:
        """Extract critical issues from scores."""
        issues = []
        for ds in dimension_scores:
            if ds.percentage < 50:
                issues.append(f"{ds.dimension_name}: Score {ds.percentage:.0f}% is below minimum")
            for rec in ds.recommendations[:1]:
                if "must" in rec.lower() or "required" in rec.lower():
                    issues.append(rec)
        return issues[:10]

    def _extract_recommendations(
        self,
        dimension_scores: List[DimensionScore],
    ) -> List[str]:
        """Extract recommendations from scores."""
        all_recs = []
        for ds in dimension_scores:
            all_recs.extend(ds.recommendations)
        return list(set(all_recs))[:15]

    def _extract_strengths(
        self,
        dimension_scores: List[DimensionScore],
    ) -> List[str]:
        """Extract strengths from scores."""
        strengths = []
        for ds in dimension_scores:
            if ds.percentage >= 80:
                strengths.append(f"{ds.dimension_name}: {ds.percentage:.0f}%")
            for finding in ds.findings[:2]:
                if "found" in finding.lower() or "implemented" in finding.lower():
                    strengths.append(finding)
        return list(set(strengths))[:10]
