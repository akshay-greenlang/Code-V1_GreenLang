# -*- coding: utf-8 -*-
"""
RaceToZeroCertificateTemplate - Race to Zero verification certificate for PACK-025.

Renders an official Race to Zero verification certificate with organization
name, sector, verification level (QUALIFIED/VERIFIED/EXEMPLARY), credibility
score, valid-through date, and QR code placeholder for verification.

Sections:
    1. Certificate Header & Branding
    2. Organization Details
    3. Verification Level & Score
    4. Pledge Details
    5. Compliance Summary
    6. Verification Body
    7. QR Code & Verification Link
    8. Certificate Footer & Authorization

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "race_to_zero_certificate"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)

def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

# Verification levels
VERIFICATION_LEVELS = {
    "QUALIFIED": {
        "description": "Organization has met all starting line criteria and submitted a valid pledge",
        "min_score": 0,
        "badge_color": "#43a047",
    },
    "VERIFIED": {
        "description": "Organization has demonstrated credible progress with independent verification",
        "min_score": 60,
        "badge_color": "#1b5e20",
    },
    "EXEMPLARY": {
        "description": "Organization demonstrates best-in-class climate leadership and transparency",
        "min_score": 85,
        "badge_color": "#004d40",
    },
}

class RaceToZeroCertificateTemplate:
    """Race to Zero verification certificate template for PACK-025.

    Generates official-style verification certificates with organization
    details, verification level (QUALIFIED/VERIFIED/EXEMPLARY), credibility
    score, and QR code placeholder for digital verification.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    LEVELS = VERIFICATION_LEVELS

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the certificate as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_organization(data),
            self._md_verification_level(data),
            self._md_pledge_details(data),
            self._md_compliance_summary(data),
            self._md_verification_body(data),
            self._md_qr_verification(data),
            self._md_authorization(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the certificate as a styled HTML document."""
        self.generated_at = utcnow()
        css = self._certificate_css()
        body = "\n".join([
            self._html_certificate_body(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Verification Certificate</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'{body}\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the certificate as structured JSON."""
        self.generated_at = utcnow()
        level = data.get("verification_level", "QUALIFIED")
        level_info = VERIFICATION_LEVELS.get(level, VERIFICATION_LEVELS["QUALIFIED"])
        cert_number = data.get("certificate_number", f"R2Z-{_new_uuid()[:8].upper()}")

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "certificate_number": cert_number,
            "org_name": data.get("org_name", ""),
            "sector": data.get("sector", ""),
            "country": data.get("country", ""),
            "verification_level": level,
            "level_description": level_info["description"],
            "credibility_score": data.get("credibility_score", 0),
            "issue_date": data.get("issue_date", self.generated_at.strftime("%Y-%m-%d")),
            "valid_through": data.get("valid_through", ""),
            "pledge": {
                "interim_target": data.get("interim_target", {}),
                "longterm_target": data.get("longterm_target", {}),
                "baseline_tco2e": data.get("baseline_tco2e", 0),
            },
            "verification_body": data.get("verification_body", {}),
            "partner_initiative": data.get("partner_initiative", ""),
            "verification_url": data.get("verification_url", ""),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = utcnow()
        cert_number = data.get("certificate_number", f"R2Z-{_new_uuid()[:8].upper()}")
        level = data.get("verification_level", "QUALIFIED")
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        sheets["Certificate Details"] = [
            {"Field": "Certificate Number", "Value": cert_number},
            {"Field": "Organization", "Value": data.get("org_name", "")},
            {"Field": "Sector", "Value": data.get("sector", "")},
            {"Field": "Country", "Value": data.get("country", "")},
            {"Field": "Verification Level", "Value": level},
            {"Field": "Credibility Score", "Value": data.get("credibility_score", 0)},
            {"Field": "Issue Date", "Value": data.get("issue_date", "")},
            {"Field": "Valid Through", "Value": data.get("valid_through", "")},
            {"Field": "Partner Initiative", "Value": data.get("partner_initiative", "")},
            {"Field": "Baseline Emissions (tCO2e)", "Value": data.get("baseline_tco2e", 0)},
            {"Field": "Interim Target Year", "Value": data.get("interim_target", {}).get("year", 2030)},
            {"Field": "Net-Zero Target Year", "Value": data.get("longterm_target", {}).get("year", 2050)},
        ]

        vb = data.get("verification_body", {})
        sheets["Verification Body"] = [
            {"Field": "Name", "Value": vb.get("name", "")},
            {"Field": "Accreditation", "Value": vb.get("accreditation", "")},
            {"Field": "Lead Verifier", "Value": vb.get("lead_verifier", "")},
            {"Field": "Assurance Level", "Value": vb.get("assurance_level", "")},
            {"Field": "Standard", "Value": vb.get("standard", "")},
        ]

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        return (
            "---\n\n"
            "# RACE TO ZERO\n"
            "## Verification Certificate\n\n"
            "---"
        )

    def _md_organization(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        return (
            f"## Organization\n\n"
            f"**This is to certify that**\n\n"
            f"### {org}\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Sector | {data.get('sector', '')} |\n"
            f"| Country | {data.get('country', '')} |\n"
            f"| Partner Initiative | {data.get('partner_initiative', '')} |"
        )

    def _md_verification_level(self, data: Dict[str, Any]) -> str:
        level = data.get("verification_level", "QUALIFIED")
        level_info = VERIFICATION_LEVELS.get(level, VERIFICATION_LEVELS["QUALIFIED"])
        score = data.get("credibility_score", 0)

        return (
            f"## Verification Level: {level}\n\n"
            f"> {level_info['description']}\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Verification Level | **{level}** |\n"
            f"| Credibility Score | **{_dec(score, 1)}/100** |\n"
            f"| Certificate Number | {data.get('certificate_number', 'N/A')} |\n"
            f"| Issue Date | {data.get('issue_date', '')} |\n"
            f"| Valid Through | {data.get('valid_through', '')} |"
        )

    def _md_pledge_details(self, data: Dict[str, Any]) -> str:
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        return (
            f"## Pledge Details\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Campaign | Race to Zero |\n"
            f"| Baseline Emissions | {_dec_comma(data.get('baseline_tco2e', 0))} tCO2e |\n"
            f"| Interim Target | {_pct(interim.get('reduction_pct', 50))} by {interim.get('year', 2030)} |\n"
            f"| Net-Zero Target | {longterm.get('year', 2050)} |\n"
            f"| Scope Coverage | {longterm.get('scope_coverage', 'S1+S2+S3')} |\n"
            f"| Pathway | {interim.get('pathway', '1.5C aligned')} |"
        )

    def _md_compliance_summary(self, data: Dict[str, Any]) -> str:
        compliance = data.get("compliance", {})
        return (
            f"## Compliance Summary\n\n"
            f"| Criterion | Status |\n|-----------|:------:|\n"
            f"| Starting Line Criteria | {compliance.get('starting_line', 'Met')} |\n"
            f"| Transition Plan | {compliance.get('transition_plan', 'Submitted')} |\n"
            f"| Annual Reporting | {compliance.get('annual_reporting', 'Committed')} |\n"
            f"| Independent Verification | {compliance.get('verification', 'Completed')} |\n"
            f"| Offset Policy | {compliance.get('offset_policy', 'Compliant')} |\n"
            f"| Public Disclosure | {compliance.get('disclosure', 'Published')} |"
        )

    def _md_verification_body(self, data: Dict[str, Any]) -> str:
        vb = data.get("verification_body", {})
        return (
            f"## Verification Body\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Verifier | {vb.get('name', 'N/A')} |\n"
            f"| Accreditation | {vb.get('accreditation', 'N/A')} |\n"
            f"| Lead Verifier | {vb.get('lead_verifier', 'N/A')} |\n"
            f"| Assurance Level | {vb.get('assurance_level', 'Limited')} |\n"
            f"| Standard | {vb.get('standard', 'ISO 14064-3:2019')} |"
        )

    def _md_qr_verification(self, data: Dict[str, Any]) -> str:
        cert_number = data.get("certificate_number", "N/A")
        url = data.get("verification_url", "")
        return (
            f"## Digital Verification\n\n"
            f"**Certificate Number:** {cert_number}\n\n"
            f"Verify this certificate online:\n"
            f"{url if url else 'https://verify.greenlang.io/r2z/' + cert_number}\n\n"
            f"*[QR Code placeholder -- rendered in HTML/PDF output]*"
        )

    def _md_authorization(self, data: Dict[str, Any]) -> str:
        auth = data.get("authorization", {})
        issue_date = data.get("issue_date", "")
        return (
            f"## Authorization\n\n"
            f"This certificate has been issued by the GreenLang verification platform "
            f"upon successful completion of all Race to Zero campaign requirements.\n\n"
            f"**Authorized by:** {auth.get('authorized_by', 'GreenLang Platform')}  \n"
            f"**Date:** {issue_date}  \n"
            f"**Signature:** {auth.get('signature', '_________________')}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Race to Zero is a global campaign led by the UNFCCC High-Level Champions.*  \n"
            f"*This certificate is subject to annual renewal upon continued compliance.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML certificate (styled formal document)                           #
    # ------------------------------------------------------------------ #

    def _certificate_css(self) -> str:
        return (
            "@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');"
            "*{box-sizing:border-box;margin:0;padding:0;}"
            "body{font-family:'Segoe UI',system-ui,sans-serif;background:#e8e8e8;"
            "display:flex;justify-content:center;align-items:center;min-height:100vh;padding:20px;}"
            ".certificate{width:900px;background:#fff;border:3px solid #1b5e20;"
            "border-radius:16px;padding:0;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,0.15);}"
            ".cert-border{border:2px solid #c8e6c9;margin:12px;border-radius:10px;padding:40px 50px;}"
            ".cert-header{text-align:center;margin-bottom:30px;}"
            ".cert-campaign{font-size:0.9em;color:#43a047;text-transform:uppercase;letter-spacing:3px;"
            "margin-bottom:8px;}"
            ".cert-title{font-family:'Playfair Display',serif;font-size:2.4em;color:#1b5e20;"
            "margin-bottom:4px;}"
            ".cert-subtitle{font-size:1.1em;color:#2e7d32;}"
            ".cert-divider{border:none;border-top:2px solid #c8e6c9;margin:20px 0;}"
            ".cert-org{text-align:center;margin:25px 0;}"
            ".cert-org-name{font-family:'Playfair Display',serif;font-size:2em;color:#1b5e20;"
            "margin:10px 0;}"
            ".cert-org-sector{font-size:1em;color:#558b2f;}"
            ".cert-level{text-align:center;margin:30px 0;}"
            ".level-badge{display:inline-block;padding:12px 40px;border-radius:30px;"
            "font-size:1.4em;font-weight:700;letter-spacing:2px;color:#fff;}"
            ".level-qualified{background:linear-gradient(135deg,#43a047,#66bb6a);}"
            ".level-verified{background:linear-gradient(135deg,#1b5e20,#2e7d32);}"
            ".level-exemplary{background:linear-gradient(135deg,#004d40,#00695c);}"
            ".cert-score{text-align:center;margin:20px 0;}"
            ".score-circle{display:inline-block;width:100px;height:100px;border-radius:50%;"
            "border:5px solid #2e7d32;background:#e8f5e9;line-height:90px;"
            "font-size:2em;font-weight:700;color:#1b5e20;text-align:center;}"
            ".score-label{font-size:0.85em;color:#558b2f;margin-top:6px;}"
            ".cert-details{margin:25px 0;}"
            ".cert-details table{width:100%;border-collapse:collapse;}"
            ".cert-details th{text-align:left;padding:8px 12px;color:#1b5e20;font-weight:600;"
            "width:40%;border-bottom:1px solid #e8f5e9;}"
            ".cert-details td{padding:8px 12px;border-bottom:1px solid #e8f5e9;}"
            ".cert-qr{text-align:center;margin:25px 0;}"
            ".qr-placeholder{display:inline-block;width:120px;height:120px;border:2px solid #c8e6c9;"
            "border-radius:8px;background:#f9fbe7;line-height:120px;color:#689f38;"
            "font-size:0.8em;}"
            ".cert-verify-text{font-size:0.85em;color:#689f38;margin-top:8px;}"
            ".cert-auth{text-align:center;margin:30px 0;}"
            ".cert-auth-line{display:inline-block;width:250px;border-bottom:1px solid #333;"
            "margin:0 30px;padding-bottom:4px;}"
            ".cert-auth-label{font-size:0.85em;color:#555;margin-top:4px;}"
            ".cert-footer{text-align:center;font-size:0.8em;color:#689f38;margin-top:20px;}"
            ".cert-number{font-family:monospace;font-size:0.85em;color:#888;margin-top:6px;}"
        )

    def _html_certificate_body(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        sector = data.get("sector", "")
        country = data.get("country", "")
        level = data.get("verification_level", "QUALIFIED")
        level_class = f"level-{level.lower()}"
        score = data.get("credibility_score", 0)
        cert_number = data.get("certificate_number", f"R2Z-{_new_uuid()[:8].upper()}")
        issue_date = data.get("issue_date", self.generated_at.strftime("%Y-%m-%d") if self.generated_at else "")
        valid_through = data.get("valid_through", "")
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        vb = data.get("verification_body", {})
        auth = data.get("authorization", {})
        url = data.get("verification_url", f"https://verify.greenlang.io/r2z/{cert_number}")

        # Generate a simple QR-like visual (text-based placeholder)
        qr_data = _compute_hash({"cert": cert_number, "org": org})[:16]

        return (
            f'<div class="certificate">\n'
            f'  <div class="cert-border">\n'
            f'    <div class="cert-header">\n'
            f'      <div class="cert-campaign">UNFCCC Race to Zero Campaign</div>\n'
            f'      <div class="cert-title">Verification Certificate</div>\n'
            f'      <div class="cert-subtitle">Climate Action Commitment</div>\n'
            f'    </div>\n'
            f'    <hr class="cert-divider">\n'
            f'    <div class="cert-org">\n'
            f'      <div style="font-size:0.9em;color:#558b2f;">This is to certify that</div>\n'
            f'      <div class="cert-org-name">{org}</div>\n'
            f'      <div class="cert-org-sector">{sector}'
            f'{" | " + country if country else ""}</div>\n'
            f'    </div>\n'
            f'    <div class="cert-level">\n'
            f'      <div style="font-size:0.9em;color:#555;margin-bottom:8px;">'
            f'has achieved the verification level of</div>\n'
            f'      <div class="level-badge {level_class}">{level}</div>\n'
            f'    </div>\n'
            f'    <div class="cert-score">\n'
            f'      <div class="score-circle">{_dec(score, 0)}</div>\n'
            f'      <div class="score-label">Credibility Score</div>\n'
            f'    </div>\n'
            f'    <hr class="cert-divider">\n'
            f'    <div class="cert-details">\n'
            f'      <table>\n'
            f'        <tr><th>Campaign</th><td>Race to Zero</td></tr>\n'
            f'        <tr><th>Partner Initiative</th>'
            f'<td>{data.get("partner_initiative", "N/A")}</td></tr>\n'
            f'        <tr><th>Baseline Emissions</th>'
            f'<td>{_dec_comma(data.get("baseline_tco2e", 0))} tCO2e</td></tr>\n'
            f'        <tr><th>Interim Target</th>'
            f'<td>{_pct(interim.get("reduction_pct", 50))} by {interim.get("year", 2030)}</td></tr>\n'
            f'        <tr><th>Net-Zero Target</th>'
            f'<td>{longterm.get("year", 2050)}</td></tr>\n'
            f'        <tr><th>Issue Date</th><td>{issue_date}</td></tr>\n'
            f'        <tr><th>Valid Through</th><td>{valid_through}</td></tr>\n'
            f'        <tr><th>Certificate Number</th>'
            f'<td style="font-family:monospace;">{cert_number}</td></tr>\n'
            f'      </table>\n'
            f'    </div>\n'
            f'    <div class="cert-qr">\n'
            f'      <div class="qr-placeholder">[QR Code]<br>{qr_data[:8]}</div>\n'
            f'      <div class="cert-verify-text">Verify at: {url}</div>\n'
            f'    </div>\n'
            f'    <hr class="cert-divider">\n'
            f'    <div class="cert-auth">\n'
            f'      <div>\n'
            f'        <span class="cert-auth-line">'
            f'{auth.get("authorized_by", "")}</span>\n'
            f'        <span class="cert-auth-line">'
            f'{vb.get("lead_verifier", "")}</span>\n'
            f'      </div>\n'
            f'      <div>\n'
            f'        <span style="display:inline-block;width:250px;margin:0 30px;">'
            f'<div class="cert-auth-label">Authorized Signatory</div></span>\n'
            f'        <span style="display:inline-block;width:250px;margin:0 30px;">'
            f'<div class="cert-auth-label">Lead Verifier</div></span>\n'
            f'      </div>\n'
            f'    </div>\n'
            f'    <div class="cert-footer">\n'
            f'      <p>Race to Zero is a global campaign led by the UNFCCC High-Level Champions.</p>\n'
            f'      <p>This certificate is subject to annual renewal upon continued compliance.</p>\n'
            f'      <div class="cert-number">Certificate: {cert_number} | '
            f'Hash: {_compute_hash(cert_number)[:16]}</div>\n'
            f'    </div>\n'
            f'  </div>\n'
            f'</div>'
        )
