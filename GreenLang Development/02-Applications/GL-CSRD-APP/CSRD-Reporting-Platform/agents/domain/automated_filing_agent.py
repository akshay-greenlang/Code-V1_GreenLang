# -*- coding: utf-8 -*-
"""
CSRD Automated Filing Agent

Automates CSRD report filing to national regulatory authorities.
Validates ESEF packages and manages electronic submissions.

Security Features:
- XXE Attack Protection: All XML parsing uses secure parser configuration
- Input Validation: File size limits and content validation
- Network Isolation: External entity resolution disabled

Author: GreenLang AI Team
Date: 2025-10-18
Version: 1.0.1 (Security Update)
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import zipfile
import logging
import json
from datetime import datetime
from lxml import etree
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY: SECURE XML PARSING
# ============================================================================

def create_secure_xml_parser():
    """
    Create XML parser with XXE protection.

    Security Features:
    - Disables external entity resolution (prevents XXE attacks)
    - Disables DTD processing (prevents entity expansion attacks)
    - Disables network access (prevents SSRF)
    - Prevents billion laughs attack (huge_tree=False)

    Returns:
        etree.XMLParser configured with security settings

    References:
    - OWASP XXE Prevention: https://cheatsheetseries.owasp.org/cheatsheets/XML_External_Entity_Prevention_Cheat_Sheet.html
    - CWE-611: Improper Restriction of XML External Entity Reference
    """
    parser = etree.XMLParser(
        resolve_entities=False,  # Disable external entities (XXE protection)
        no_network=True,         # Disable network access (SSRF protection)
        dtd_validation=False,    # Disable DTD validation
        load_dtd=False,          # Don't load DTD
        huge_tree=False,         # Prevent billion laughs attack
        remove_blank_text=True   # Clean whitespace
    )
    logger.debug("Created secure XML parser with XXE protection")
    return parser


def validate_xml_input(xml_content: Union[str, bytes], max_size_mb: int = 10) -> bool:
    """
    Validate XML input before parsing.

    Args:
        xml_content: XML string or bytes
        max_size_mb: Maximum allowed size in MB (default: 10MB)

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails

    Security Checks:
    - File size limit to prevent DoS
    - DOCTYPE declaration check
    - External entity declaration check
    """
    # Size check
    if isinstance(xml_content, str):
        size_bytes = len(xml_content.encode('utf-8'))
    else:
        size_bytes = len(xml_content)

    if size_bytes > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"XML content too large: {size_bytes} bytes (max {max_size_mb}MB). "
            "This may indicate a DoS attempt."
        )

    # Content validation - check for suspicious patterns
    content_str = xml_content if isinstance(xml_content, str) else xml_content.decode('utf-8', errors='ignore')

    # Check for DOCTYPE declarations (can be used for XXE attacks)
    if '<!DOCTYPE' in content_str:
        raise ValueError(
            "DOCTYPE declarations not allowed. "
            "This is a security restriction to prevent XXE attacks."
        )

    # Check for entity declarations
    if '<!ENTITY' in content_str:
        raise ValueError(
            "Entity declarations not allowed. "
            "This is a security restriction to prevent XXE and entity expansion attacks."
        )

    # Check for SYSTEM keyword (external entity reference)
    if 'SYSTEM' in content_str and ('file://' in content_str or 'http://' in content_str):
        raise ValueError(
            "External entity references not allowed. "
            "This is a security restriction to prevent XXE attacks."
        )

    logger.debug(f"XML input validation passed: {size_bytes} bytes")
    return True


def parse_xml_safely(xml_content: Union[str, bytes], max_size_mb: int = 10) -> etree._Element:
    """
    Parse XML content with security validation.

    Args:
        xml_content: XML string or bytes
        max_size_mb: Maximum allowed size in MB

    Returns:
        Parsed XML Element tree

    Raises:
        ValueError: If validation fails
        etree.XMLSyntaxError: If XML is malformed
    """
    # Validate input first
    validate_xml_input(xml_content, max_size_mb)

    # Parse with secure parser
    parser = create_secure_xml_parser()

    try:
        if isinstance(xml_content, str):
            tree = etree.fromstring(xml_content.encode('utf-8'), parser)
        else:
            tree = etree.fromstring(xml_content, parser)
        return tree
    except etree.XMLSyntaxError as e:
        logger.error(f"XML parsing error: {e}")
        raise ValueError(f"Invalid XML structure: {e}")


class CSRDAutomatedFilingAgent:
    """
    Automate CSRD report filing to national authorities

    Features:
    - ESEF package validation
    - Electronic submission to national registers
    - Filing status tracking
    - Automatic retry on failure
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Automated Filing Agent

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.filing_endpoints = self._load_filing_endpoints()
        self.submission_history_file = Path('data/submission_history.json')

    def _load_filing_endpoints(self) -> Dict[str, str]:
        """Load national filing endpoints"""
        return {
            'DE': 'https://unternehmensregister.de/api/v1/csrd/submit',
            'FR': 'https://infogreffe.fr/api/v1/csrd/submit',
            'NL': 'https://kvk.nl/api/v1/csrd/submit',
            'IT': 'https://registroimprese.it/api/v1/csrd/submit',
            'ES': 'https://rmc.es/api/v1/csrd/submit',
            'BE': 'https://nbb.be/api/v1/csrd/submit',
            'AT': 'https://wko.at/api/v1/csrd/submit',
            'SE': 'https://bolagsverket.se/api/v1/csrd/submit',
            'DK': 'https://erhvervsstyrelsen.dk/api/v1/csrd/submit',
            'FI': 'https://prh.fi/api/v1/csrd/submit'
        }

    def validate_esef_package(self, package_path: Path) -> Dict[str, Any]:
        """
        Validate ESEF package before submission

        Args:
            package_path: Path to ESEF package (ZIP file)

        Returns:
            Validation results
        """
        logger.info(f"Validating ESEF package: {package_path}")

        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'package_path': str(package_path)
        }

        # Check if package exists
        if not package_path.exists():
            validation_results['valid'] = False
            validation_results['errors'].append('Package file not found')
            return validation_results

        # Check if it's a valid ZIP
        if not zipfile.is_zipfile(package_path):
            validation_results['valid'] = False
            validation_results['errors'].append('Invalid ZIP file')
            return validation_results

        # Extract and validate contents
        try:
            with zipfile.ZipFile(package_path, 'r') as zf:
                files = zf.namelist()

                # Check required files
                required_patterns = ['META-INF/', 'reports/', '.xhtml']

                for pattern in required_patterns:
                    if not any(pattern in f for f in files):
                        validation_results['errors'].append(
                            f'Missing required file pattern: {pattern}'
                        )
                        validation_results['valid'] = False

                # Validate XHTML files
                xhtml_files = [f for f in files if f.endswith('.xhtml')]

                if not xhtml_files:
                    validation_results['errors'].append('No XHTML files found')
                    validation_results['valid'] = False
                else:
                    for xhtml_file in xhtml_files:
                        xhtml_content = zf.read(xhtml_file)

                        try:
                            # SECURITY: Use secure parser to prevent XXE attacks
                            tree = parse_xml_safely(xhtml_content, max_size_mb=50)

                            # Check for iXBRL namespace
                            if 'inlineXBRL' not in str(tree.nsmap):
                                validation_results['warnings'].append(
                                    f'{xhtml_file}: Missing iXBRL namespace'
                                )

                        except ValueError as e:
                            # Security validation failed
                            validation_results['errors'].append(
                                f'Security validation failed for {xhtml_file}: {str(e)}'
                            )
                            validation_results['valid'] = False
                            logger.warning(f"SECURITY: Rejected potentially malicious XML in {xhtml_file}: {e}")

                        except etree.XMLSyntaxError as e:
                            validation_results['errors'].append(
                                f'Invalid XHTML in {xhtml_file}: {str(e)}'
                            )
                            validation_results['valid'] = False

                # Check package size
                package_size_mb = package_path.stat().st_size / (1024 * 1024)
                if package_size_mb > 100:  # 100 MB limit
                    validation_results['warnings'].append(
                        f'Large package size: {package_size_mb:.2f} MB'
                    )

        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f'Error reading package: {str(e)}')

        if validation_results['valid']:
            logger.info("✅ ESEF package validation passed")
        else:
            logger.error(f"❌ ESEF package validation failed: {validation_results['errors']}")

        return validation_results

    def submit_filing(
        self,
        package_path: Path,
        country_code: str,
        company_lei: str,
        reporting_year: int
    ) -> Dict[str, Any]:
        """
        Submit CSRD report to national authority

        Args:
            package_path: Path to ESEF package
            country_code: ISO country code (e.g., 'DE', 'FR')
            company_lei: Company Legal Entity Identifier
            reporting_year: Reporting year

        Returns:
            Submission result
        """
        logger.info(f"Submitting CSRD report for {company_lei} to {country_code}")

        # Validate package first
        validation = self.validate_esef_package(package_path)

        if not validation['valid']:
            return {
                'status': 'failed',
                'reason': 'validation_failed',
                'errors': validation['errors'],
                'submission_id': None
            }

        # Get filing endpoint
        filing_url = self.filing_endpoints.get(country_code)

        if not filing_url:
            return {
                'status': 'failed',
                'reason': 'unsupported_country',
                'message': f'No filing endpoint for country {country_code}',
                'submission_id': None
            }

        # In production, this would make actual API call
        # For now, we simulate successful submission

        submission_id = f"SUB-{country_code}-{company_lei}-{reporting_year}-{DeterministicClock.now().strftime('%Y%m%d%H%M%S')}"

        result = {
            'status': 'success',
            'submission_id': submission_id,
            'confirmation_number': f"CONF-{submission_id}",
            'submission_timestamp': DeterministicClock.now().isoformat(),
            'country_code': country_code,
            'company_lei': company_lei,
            'reporting_year': reporting_year,
            'package_path': str(package_path),
            'filing_url': filing_url
        }

        # Save submission history
        self._save_submission_history(result)

        logger.info(f"✅ Filing submitted successfully: {submission_id}")

        return result

    def _save_submission_history(self, submission: Dict[str, Any]):
        """Save submission to history file"""
        self.submission_history_file.parent.mkdir(parents=True, exist_ok=True)

        history = []
        if self.submission_history_file.exists():
            with open(self.submission_history_file, 'r') as f:
                history = json.load(f)

        history.append(submission)

        with open(self.submission_history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def track_filing_status(
        self,
        submission_id: str,
        country_code: str
    ) -> Dict[str, Any]:
        """
        Track filing status

        Args:
            submission_id: Submission ID
            country_code: Country code

        Returns:
            Status information
        """
        logger.info(f"Tracking status for submission {submission_id}")

        # In production, this would query the API
        # For now, we return mock status

        status = {
            'submission_id': submission_id,
            'status': 'accepted',
            'last_updated': DeterministicClock.now().isoformat(),
            'processing_stage': 'validation',
            'messages': [
                'Package received',
                'Formal validation in progress',
                'Expected completion: 2-5 business days'
            ]
        }

        return status

    def retry_failed_filing(
        self,
        submission_id: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Retry failed filing

        Args:
            submission_id: Failed submission ID
            max_retries: Maximum retry attempts

        Returns:
            Retry result
        """
        logger.info(f"Retrying failed submission {submission_id}")

        # Load submission history
        if not self.submission_history_file.exists():
            return {
                'status': 'error',
                'message': 'Submission not found in history'
            }

        with open(self.submission_history_file, 'r') as f:
            history = json.load(f)

        # Find original submission
        original = None
        for sub in history:
            if sub.get('submission_id') == submission_id:
                original = sub
                break

        if not original:
            return {
                'status': 'error',
                'message': 'Submission not found'
            }

        # Retry submission
        retry_result = self.submit_filing(
            package_path=Path(original['package_path']),
            country_code=original['country_code'],
            company_lei=original['company_lei'],
            reporting_year=original['reporting_year']
        )

        return retry_result

    def generate_submission_report(
        self,
        reporting_year: int,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Generate report of all submissions

        Args:
            reporting_year: Year to report on
            output_path: Path to save report

        Returns:
            Report summary
        """
        logger.info(f"Generating submission report for {reporting_year}")

        if not self.submission_history_file.exists():
            return {
                'total_submissions': 0,
                'successful': 0,
                'failed': 0
            }

        with open(self.submission_history_file, 'r') as f:
            history = json.load(f)

        # Filter by year
        year_submissions = [
            s for s in history
            if s.get('reporting_year') == reporting_year
        ]

        # Count statuses
        successful = len([s for s in year_submissions if s.get('status') == 'success'])
        failed = len([s for s in year_submissions if s.get('status') == 'failed'])

        # Group by country
        by_country = {}
        for sub in year_submissions:
            country = sub.get('country_code')
            if country not in by_country:
                by_country[country] = []
            by_country[country].append(sub)

        report = {
            'reporting_year': reporting_year,
            'total_submissions': len(year_submissions),
            'successful': successful,
            'failed': failed,
            'success_rate': round((successful / len(year_submissions) * 100) if year_submissions else 0, 2),
            'by_country': {
                country: len(subs)
                for country, subs in by_country.items()
            },
            'submissions': year_submissions
        }

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Submission report saved to {output_file}")

        return report


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize agent
    config = {
        'api_key': 'DEMO_KEY'
    }

    agent = CSRDAutomatedFilingAgent(config)

    # Create mock ESEF package for testing
    test_package = Path('output/test_esef_package.zip')
    test_package.parent.mkdir(parents=True, exist_ok=True)

    # Create minimal valid package
    with zipfile.ZipFile(test_package, 'w') as zf:
        # Add META-INF directory
        zf.writestr('META-INF/manifest.xml', '<?xml version="1.0"?><manifest/>')

        # Add reports directory with sample XHTML
        xhtml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
    <head><title>CSRD Report</title></head>
    <body>
        <p>Sample CSRD Report</p>
    </body>
</html>'''
        zf.writestr('reports/csrd_report.xhtml', xhtml_content)

    logger.info(f"Created test package: {test_package}")

    # Validate package
    validation = agent.validate_esef_package(test_package)

    if validation['valid']:
        logger.info("✅ Package validation passed")

        # Submit filing
        result = agent.submit_filing(
            package_path=test_package,
            country_code='DE',
            company_lei='DE123456789012345678',
            reporting_year=2024
        )

        if result['status'] == 'success':
            logger.info(f"✅ Filing submitted: {result['confirmation_number']}")

            # Track status
            status = agent.track_filing_status(
                submission_id=result['submission_id'],
                country_code='DE'
            )

            logger.info(f"Filing status: {status['status']}")
        else:
            logger.error(f"❌ Filing failed: {result['reason']}")
    else:
        logger.error(f"❌ Package validation failed: {validation['errors']}")

    # Generate submission report
    report = agent.generate_submission_report(
        reporting_year=2024,
        output_path='output/submission_report_2024.json'
    )

    logger.info(f"\n✅ Filing process complete")
    logger.info(f"Total submissions: {report['total_submissions']}")
    logger.info(f"Success rate: {report['success_rate']}%")
