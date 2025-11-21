# -*- coding: utf-8 -*-
"""
CSRD Regulatory Intelligence Agent

Monitors regulatory updates from EFRAG, EU Commission, and national authorities.
Auto-generates compliance rules from new regulatory guidance.

Author: GreenLang AI Team
Date: 2025-10-18
"""

from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import hashlib
import json
import yaml
from pathlib import Path
import logging
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class CSRDRegulatoryIntelligenceAgent:
    """
    Monitor and interpret CSRD/ESRS regulatory updates

    Features:
    - Web scraping of regulatory sources
    - Document analysis with RAG
    - Automatic rule generation
    - Alert system for critical updates
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Regulatory Intelligence Agent

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.regulatory_sources = self._load_sources()
        self.processed_docs_file = Path('data/processed_regulatory_docs.json')
        self.processed_docs = self._load_processed_docs()

    def _load_sources(self) -> List[Dict[str, str]]:
        """Load regulatory sources to monitor"""
        return [
            {
                'name': 'EFRAG',
                'url': 'https://www.efrag.org/lab6',
                'type': 'primary',
                'check_frequency': 'daily'
            },
            {
                'name': 'EU Commission - CSRD',
                'url': 'https://finance.ec.europa.eu/capital-markets-union-and-financial-markets/company-reporting-and-auditing/company-reporting/corporate-sustainability-reporting_en',
                'type': 'primary',
                'check_frequency': 'daily'
            },
            {
                'name': 'ESMA',
                'url': 'https://www.esma.europa.eu/policy-activities/corporate-disclosure/sustainability-disclosure',
                'type': 'secondary',
                'check_frequency': 'weekly'
            }
        ]

    def _load_processed_docs(self) -> Dict[str, Any]:
        """Load previously processed documents"""
        if self.processed_docs_file.exists():
            with open(self.processed_docs_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_doc(self, doc_hash: str, doc_info: Dict[str, Any]):
        """Save processed document info"""
        self.processed_docs[doc_hash] = doc_info

        # Ensure data directory exists
        self.processed_docs_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.processed_docs_file, 'w') as f:
            json.dump(self.processed_docs, f, indent=2)

    def monitor_regulatory_updates(self, max_age_days: int = 30) -> List[Dict[str, Any]]:
        """
        Check all sources for regulatory updates

        Args:
            max_age_days: Only check documents from last N days

        Returns:
            List of new regulatory updates found
        """
        updates = []

        for source in self.regulatory_sources:
            logger.info(f"Checking {source['name']} for updates...")

            try:
                new_documents = self._check_source(source, max_age_days)

                for doc in new_documents:
                    update = self._analyze_document(doc, source)
                    updates.append(update)

            except Exception as e:
                logger.error(f"Error checking {source['name']}: {e}")
                continue

        logger.info(f"Found {len(updates)} new regulatory updates")
        return updates

    def _check_source(self, source: Dict[str, str], max_age_days: int) -> List[Dict[str, Any]]:
        """
        Check a regulatory source for new documents

        Args:
            source: Source configuration
            max_age_days: Maximum age of documents to consider

        Returns:
            List of new documents found
        """
        new_documents = []

        try:
            # Fetch webpage with timeout
            response = requests.get(source['url'], timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find document links (PDFs and document pages)
            document_links = soup.find_all('a', href=lambda x: x and (
                '.pdf' in x.lower() or
                'document' in x.lower() or
                'esrs' in x.lower()
            ))

            for link in document_links[:20]:  # Limit to first 20 links
                doc_url = link.get('href')
                if not doc_url:
                    continue

                # Make absolute URL
                if not doc_url.startswith('http'):
                    from urllib.parse import urljoin
                    doc_url = urljoin(source['url'], doc_url)

                doc_title = link.text.strip()

                # Generate hash
                doc_hash = hashlib.sha256(doc_url.encode()).hexdigest()

                # Check if already processed
                if doc_hash not in self.processed_docs:
                    new_documents.append({
                        'url': doc_url,
                        'title': doc_title,
                        'hash': doc_hash,
                        'found_date': DeterministicClock.now().isoformat()
                    })

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {source['url']}: {e}")

        return new_documents

    def _analyze_document(self, document: Dict[str, Any], source: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze regulatory document

        For now, this is a simplified version. In production, this would:
        - Download the document
        - Extract text
        - Use LLM to analyze content
        - Store in vector DB for RAG

        Args:
            document: Document metadata
            source: Source configuration

        Returns:
            Analysis results
        """
        # Simplified analysis - just structure the metadata
        analysis = {
            'document_type': self._infer_document_type(document['title']),
            'impact_level': self._assess_impact(document['title']),
            'summary': f"New document from {source['name']}: {document['title']}",
            'requires_action': self._requires_action(document['title']),
            'affected_standards': self._identify_affected_standards(document['title'])
        }

        # Mark as processed
        self._save_processed_doc(document['hash'], {
            'url': document['url'],
            'title': document['title'],
            'source': source['name'],
            'processed_date': DeterministicClock.now().isoformat(),
            'analysis': analysis
        })

        return {
            'document': document,
            'source': source,
            'analysis': analysis,
            'requires_action': analysis['requires_action']
        }

    def _infer_document_type(self, title: str) -> str:
        """Infer document type from title"""
        title_lower = title.lower()

        if 'amendment' in title_lower:
            return 'amendment'
        elif 'q&a' in title_lower or 'question' in title_lower:
            return 'qa'
        elif 'guidance' in title_lower:
            return 'guidance'
        elif 'standard' in title_lower:
            return 'standard'
        else:
            return 'other'

    def _assess_impact(self, title: str) -> str:
        """Assess impact level from title"""
        title_lower = title.lower()

        high_keywords = ['mandatory', 'required', 'must', 'amendment', 'deadline']
        medium_keywords = ['guidance', 'recommendation', 'should']

        if any(keyword in title_lower for keyword in high_keywords):
            return 'high'
        elif any(keyword in title_lower for keyword in medium_keywords):
            return 'medium'
        else:
            return 'low'

    def _requires_action(self, title: str) -> bool:
        """Determine if update requires immediate action"""
        title_lower = title.lower()

        action_keywords = [
            'mandatory', 'required', 'must', 'shall',
            'deadline', 'effective immediately', 'urgent'
        ]

        return any(keyword in title_lower for keyword in action_keywords)

    def _identify_affected_standards(self, title: str) -> List[str]:
        """Identify which ESRS standards are affected"""
        standards = []
        title_upper = title.upper()

        # Check for specific ESRS standards
        standard_codes = ['E1', 'E2', 'E3', 'E4', 'E5', 'S1', 'S2', 'S3', 'S4', 'G1']

        for code in standard_codes:
            if f'ESRS {code}' in title_upper or f'ESRS{code}' in title_upper:
                standards.append(code)

        # If no specific standards, mark as general
        if not standards and 'ESRS' in title_upper:
            standards.append('GENERAL')

        return standards

    def generate_compliance_rules(self, regulatory_update: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Auto-generate compliance rules from regulatory update

        This is a simplified version. In production, this would use LLM
        to extract specific requirements and generate YAML rules.

        Args:
            regulatory_update: Update analysis

        Returns:
            List of generated compliance rules
        """
        analysis = regulatory_update['analysis']

        # Generate basic rule template
        rules = []

        if analysis['requires_action']:
            rule_id = f"ESRS-AUTO-{DeterministicClock.now().strftime('%Y%m%d')}"

            rule = {
                'rule_id': rule_id,
                'rule_name': regulatory_update['document']['title'],
                'severity': analysis['impact_level'],
                'source': regulatory_update['source']['name'],
                'affected_standards': analysis['affected_standards'],
                'requires_manual_review': True,
                'created_date': DeterministicClock.now().isoformat(),
                'url': regulatory_update['document']['url']
            }

            rules.append(rule)

        return rules

    def send_alerts(self, updates: List[Dict[str, Any]]) -> None:
        """
        Send alerts for high-impact regulatory changes

        Args:
            updates: List of regulatory updates
        """
        high_impact_updates = [
            u for u in updates
            if u['analysis']['impact_level'] == 'high'
        ]

        if not high_impact_updates:
            logger.info("No high-impact updates to alert on")
            return

        # In production, this would send email/Slack notifications
        for update in high_impact_updates:
            logger.warning(
                f"🚨 HIGH-IMPACT UPDATE: {update['document']['title']} "
                f"from {update['source']['name']}"
            )

            # Save alert to file
            self._save_alert(update)

    def _save_alert(self, update: Dict[str, Any]):
        """Save alert to file for review"""
        alerts_dir = Path('data/regulatory_alerts')
        alerts_dir.mkdir(parents=True, exist_ok=True)

        alert_file = alerts_dir / f"alert_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(alert_file, 'w') as f:
            json.dump(update, f, indent=2)

        logger.info(f"Alert saved to {alert_file}")

    def export_rules_to_yaml(self, rules: List[Dict[str, Any]], output_path: str):
        """
        Export generated rules to YAML format

        Args:
            rules: List of compliance rules
            output_path: Path to save YAML file
        """
        with open(output_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported {len(rules)} rules to {output_path}")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize agent
    agent = CSRDRegulatoryIntelligenceAgent()

    # Monitor for updates
    logger.info("Starting regulatory monitoring...")
    updates = agent.monitor_regulatory_updates(max_age_days=30)

    logger.info(f"Found {len(updates)} regulatory updates")

    # Generate rules
    all_rules = []
    for update in updates:
        if update['requires_action']:
            rules = agent.generate_compliance_rules(update)
            all_rules.extend(rules)

    if all_rules:
        logger.info(f"Generated {len(all_rules)} new compliance rules")
        agent.export_rules_to_yaml(all_rules, 'data/auto_generated_rules.yaml')

    # Send alerts
    agent.send_alerts(updates)

    logger.info("✅ Regulatory monitoring complete")
