"""
Card Validator for Documentation
=================================

Validates model and dataset cards for completeness and quality.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CardValidationResult:
    """Result of card validation"""

    valid: bool
    score: float  # 0-100
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    missing_sections: List[str]
    stats: Dict[str, Any]


class CardValidator:
    """Validates documentation cards"""

    # Required sections by card type
    REQUIRED_SECTIONS = {
        "pack": ["Overview", "Usage", "Inputs", "Outputs", "License", "Dependencies"],
        "dataset": [
            "Dataset Summary",
            "Dataset Description",
            "Dataset Structure",
            "Dataset Creation",
            "Licensing Information",
        ],
        "model": [
            "Model Details",
            "Uses",
            "Training Details",
            "Evaluation",
            "Environmental Impact",
            "Citation",
        ],
        "pipeline": [
            "Pipeline Overview",
            "Inputs and Outputs",
            "Dependencies",
            "Performance",
            "Usage Examples",
        ],
    }

    # Important fields in frontmatter
    REQUIRED_FRONTMATTER = {
        "pack": ["name", "version", "license"],
        "dataset": ["name", "version", "format", "license"],
        "model": ["name", "version", "architecture", "license"],
        "pipeline": ["name", "version", "components", "license"],
    }

    def __init__(self):
        self.min_section_length = 20  # Minimum characters per section
        self.min_total_length = 500  # Minimum total card length

    def validate_card(
        self, content: str, card_type: str = "pack"
    ) -> CardValidationResult:
        """
        Validate a card for completeness and quality

        Args:
            content: Card content (markdown)
            card_type: Type of card (pack, dataset, model, pipeline)

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []
        missing_sections = []

        # Parse frontmatter and content
        frontmatter, body = self._parse_card(content)

        # Validate frontmatter
        if frontmatter:
            fm_errors = self._validate_frontmatter(frontmatter, card_type)
            errors.extend(fm_errors)
        else:
            errors.append("Missing frontmatter (YAML header)")

        # Check required sections
        sections = self._extract_sections(body)
        required = self.REQUIRED_SECTIONS.get(card_type, [])

        for req_section in required:
            found = False
            for section in sections:
                if req_section.lower() in section.lower():
                    found = True
                    # Check section content
                    section_content = sections[section]
                    if len(section_content.strip()) < self.min_section_length:
                        warnings.append(
                            f"Section '{section}' is too short (< {self.min_section_length} chars)"
                        )
                    break

            if not found:
                missing_sections.append(req_section)

        # Check total length
        if len(body) < self.min_total_length:
            warnings.append(f"Card is too short (< {self.min_total_length} chars)")

        # Check for code examples
        code_blocks = re.findall(r"```[\s\S]*?```", body)
        if len(code_blocks) == 0:
            suggestions.append("Add code examples to demonstrate usage")

        # Check for links
        links = re.findall(r"\[.*?\]\(.*?\)", body)
        if len(links) == 0:
            suggestions.append("Add links to relevant documentation or resources")

        # Check for contact information
        if "@" not in body and "contact" not in body.lower():
            suggestions.append("Add contact information for support")

        # Check for version information
        if "version" not in body.lower() and not frontmatter.get("version"):
            warnings.append("No version information found")

        # Check for license information
        if "license" not in body.lower() and not frontmatter.get("license"):
            errors.append("No license information found")

        # Check environmental impact (for GreenLang)
        if card_type in ["model", "pipeline"]:
            if "carbon" not in body.lower() and "emission" not in body.lower():
                suggestions.append(
                    "Add environmental impact information (carbon emissions)"
                )

        # Calculate score
        score = self._calculate_score(
            errors, warnings, suggestions, missing_sections, sections, body
        )

        # Gather statistics
        stats = {
            "length": len(body),
            "sections": len(sections),
            "code_blocks": len(code_blocks),
            "links": len(links),
            "images": len(re.findall(r"!\[.*?\]\(.*?\)", body)),
            "tables": len(re.findall(r"\|.*\|.*\|", body)),
            "lists": len(re.findall(r"^[\*\-\+]\s", body, re.MULTILINE)),
        }

        return CardValidationResult(
            valid=len(errors) == 0 and len(missing_sections) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            missing_sections=missing_sections,
            stats=stats,
        )

    def _parse_card(self, content: str) -> Tuple[Dict, str]:
        """Parse frontmatter and body from card content"""
        lines = content.split("\n")

        # Look for frontmatter
        if lines[0].strip() == "---":
            end_idx = -1
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end_idx = i
                    break

            if end_idx > 0:
                frontmatter_text = "\n".join(lines[1:end_idx])
                try:
                    frontmatter = yaml.safe_load(frontmatter_text)
                except:
                    frontmatter = {}

                body = "\n".join(lines[end_idx + 1 :])
                return frontmatter, body

        return {}, content

    def _validate_frontmatter(self, frontmatter: Dict, card_type: str) -> List[str]:
        """Validate frontmatter fields"""
        errors = []
        required_fields = self.REQUIRED_FRONTMATTER.get(card_type, [])

        for field in required_fields:
            if field not in frontmatter:
                errors.append(f"Missing required frontmatter field: {field}")

        return errors

    def _extract_sections(self, body: str) -> Dict[str, str]:
        """Extract sections from markdown body"""
        sections = {}
        current_section = "Introduction"
        current_content = []

        for line in body.split("\n"):
            # Check for headers
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _calculate_score(
        self, errors, warnings, suggestions, missing_sections, sections, body
    ) -> float:
        """Calculate quality score (0-100)"""
        score = 100.0

        # Deduct for errors (10 points each)
        score -= len(errors) * 10

        # Deduct for missing sections (8 points each)
        score -= len(missing_sections) * 8

        # Deduct for warnings (5 points each)
        score -= len(warnings) * 5

        # Deduct for suggestions (2 points each)
        score -= len(suggestions) * 2

        # Bonus for completeness
        if len(body) > 2000:
            score += 5
        if len(sections) > 10:
            score += 5

        # Ensure score is between 0 and 100
        return max(0, min(100, score))


def validate_card(content: str, card_type: str = "pack") -> CardValidationResult:
    """
    Convenience function to validate a card

    Args:
        content: Card content
        card_type: Type of card

    Returns:
        Validation result
    """
    validator = CardValidator()
    return validator.validate_card(content, card_type)


def validate_card_file(
    path: Path, card_type: Optional[str] = None
) -> CardValidationResult:
    """
    Validate a card from file

    Args:
        path: Path to card file
        card_type: Type of card (auto-detected if None)

    Returns:
        Validation result
    """
    with open(path) as f:
        content = f.read()

    # Auto-detect card type if not specified
    if card_type is None:
        if "dataset" in str(path).lower():
            card_type = "dataset"
        elif "model" in str(path).lower():
            card_type = "model"
        elif "pipeline" in str(path).lower():
            card_type = "pipeline"
        else:
            card_type = "pack"

    return validate_card(content, card_type)


def generate_validation_report(result: CardValidationResult) -> str:
    """
    Generate a human-readable validation report

    Args:
        result: Validation result

    Returns:
        Report text
    """
    report = []

    # Header
    report.append("=" * 50)
    report.append("CARD VALIDATION REPORT")
    report.append("=" * 50)
    report.append("")

    # Overall status
    status = "VALID" if result.valid else "INVALID"
    report.append(f"Status: {status}")
    report.append(f"Score: {result.score:.1f}/100")
    report.append("")

    # Errors
    if result.errors:
        report.append("ERRORS:")
        for error in result.errors:
            report.append(f"  - {error}")
        report.append("")

    # Missing sections
    if result.missing_sections:
        report.append("MISSING SECTIONS:")
        for section in result.missing_sections:
            report.append(f"  - {section}")
        report.append("")

    # Warnings
    if result.warnings:
        report.append("WARNINGS:")
        for warning in result.warnings:
            report.append(f"  - {warning}")
        report.append("")

    # Suggestions
    if result.suggestions:
        report.append("SUGGESTIONS:")
        for suggestion in result.suggestions:
            report.append(f"  - {suggestion}")
        report.append("")

    # Statistics
    report.append("STATISTICS:")
    for key, value in result.stats.items():
        report.append(f"  - {key}: {value}")
    report.append("")

    # Summary
    report.append("SUMMARY:")
    if result.valid:
        report.append("  Card meets minimum requirements.")
    else:
        report.append("  Card needs improvement to meet requirements.")

    if result.score >= 80:
        report.append("  Quality: Excellent")
    elif result.score >= 60:
        report.append("  Quality: Good")
    elif result.score >= 40:
        report.append("  Quality: Fair")
    else:
        report.append("  Quality: Poor")

    report.append("=" * 50)

    return "\n".join(report)
