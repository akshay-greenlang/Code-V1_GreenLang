# -*- coding: utf-8 -*-
"""
CSRD Supply Chain Agent

Manages Scope 3 emissions data collection from supply chain partners.
Automates supplier data requests and aggregation.

Author: GreenLang AI Team
Date: 2025-10-18
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class CSRDSupplyChainAgent:
    """
    Manage supply chain ESG data collection and Scope 3 emissions

    Features:
    - Automated supplier data requests
    - Scope 3 emissions calculation
    - Supplier ESG scoring
    - Supply chain risk assessment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Supply Chain Agent

        Args:
            config: Configuration dictionary with supplier portal settings
        """
        self.config = config or {}
        self.supplier_portal_url = self.config.get('supplier_portal_url')
        self.api_key = self.config.get('api_key')
        self.supplier_data_file = Path('data/supplier_data.json')

    def request_supplier_data(
        self,
        supplier_ids: List[str],
        reporting_year: int,
        deadline_days: int = 30
    ) -> Dict[str, Any]:
        """
        Send data request to suppliers

        Args:
            supplier_ids: List of supplier IDs
            reporting_year: Year for which data is requested
            deadline_days: Days until deadline

        Returns:
            Summary of requests sent
        """
        logger.info(f"Sending data requests to {len(supplier_ids)} suppliers")

        requests_sent = []
        deadline = DeterministicClock.now() + timedelta(days=deadline_days)

        for supplier_id in supplier_ids:
            request = {
                'supplier_id': supplier_id,
                'reporting_year': reporting_year,
                'requested_data': [
                    'scope1_emissions',
                    'scope2_emissions',
                    'renewable_energy_percentage',
                    'waste_generated',
                    'water_consumption',
                    'employee_count',
                    'safety_incidents'
                ],
                'deadline': deadline.isoformat(),
                'request_date': DeterministicClock.now().isoformat(),
                'status': 'pending'
            }

            # In production, this would make API call to supplier portal
            # For now, we simulate the request

            requests_sent.append({
                'supplier_id': supplier_id,
                'request_id': f"REQ-{supplier_id}-{reporting_year}",
                'status': 'sent',
                'deadline': deadline.isoformat()
            })

        # Save requests
        self._save_requests(requests_sent)

        logger.info(f"Sent {len(requests_sent)} data requests")

        return {
            'total_requests': len(requests_sent),
            'requests': requests_sent,
            'deadline': deadline.isoformat()
        }

    def _save_requests(self, requests: List[Dict[str, Any]]):
        """Save data requests to file"""
        requests_file = Path('data/supplier_requests.json')
        requests_file.parent.mkdir(parents=True, exist_ok=True)

        existing_requests = []
        if requests_file.exists():
            with open(requests_file, 'r') as f:
                existing_requests = json.load(f)

        existing_requests.extend(requests)

        with open(requests_file, 'w') as f:
            json.dump(existing_requests, f, indent=2)

    def collect_supplier_responses(self) -> pd.DataFrame:
        """
        Collect completed supplier data submissions

        Returns:
            DataFrame with supplier responses
        """
        logger.info("Collecting supplier responses...")

        # In production, this would query supplier portal API
        # For now, we generate mock data

        mock_responses = [
            {
                'supplier_id': 'SUP-001',
                'supplier_name': 'ABC Manufacturing',
                'reporting_year': 2024,
                'scope1_emissions': 5000.0,
                'scope2_emissions': 2500.0,
                'renewable_energy_pct': 35.0,
                'waste_generated': 120.0,
                'water_consumption': 8000.0,
                'employee_count': 150,
                'revenue_million_eur': 25.0,
                'emissions_intensity': 300.0,  # tCO2e per M€
                'data_completeness': 95.0,
                'submission_date': DeterministicClock.now().isoformat(),
                'submission_status': 'completed'
            },
            {
                'supplier_id': 'SUP-002',
                'supplier_name': 'XYZ Services',
                'reporting_year': 2024,
                'scope1_emissions': 1200.0,
                'scope2_emissions': 800.0,
                'renewable_energy_pct': 60.0,
                'waste_generated': 45.0,
                'water_consumption': 3000.0,
                'employee_count': 75,
                'revenue_million_eur': 10.0,
                'emissions_intensity': 200.0,
                'data_completeness': 100.0,
                'submission_date': DeterministicClock.now().isoformat(),
                'submission_status': 'completed'
            },
            {
                'supplier_id': 'SUP-003',
                'supplier_name': 'Green Logistics',
                'reporting_year': 2024,
                'scope1_emissions': 8000.0,
                'scope2_emissions': 1500.0,
                'renewable_energy_pct': 45.0,
                'waste_generated': 80.0,
                'water_consumption': 5000.0,
                'employee_count': 200,
                'revenue_million_eur': 30.0,
                'emissions_intensity': 316.67,
                'data_completeness': 90.0,
                'submission_date': DeterministicClock.now().isoformat(),
                'submission_status': 'completed'
            }
        ]

        df = pd.DataFrame(mock_responses)

        logger.info(f"Received responses from {len(df)} suppliers")

        return df

    def calculate_scope3_emissions(
        self,
        supplier_data: pd.DataFrame,
        purchase_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate Scope 3 Category 1 emissions (Purchased goods and services)

        Args:
            supplier_data: DataFrame with supplier emissions data
            purchase_data: DataFrame with purchase amounts

        Returns:
            Scope 3 emissions calculation
        """
        logger.info("Calculating Scope 3 emissions...")

        # Merge supplier emissions with purchase amounts
        merged = pd.merge(
            supplier_data,
            purchase_data,
            on='supplier_id',
            how='inner'
        )

        # Calculate Scope 3 emissions based on supplier intensity
        merged['scope3_cat1_emissions'] = (
            merged['emissions_intensity'] * merged['purchase_amount_million_eur']
        )

        # Sum total
        total_scope3 = merged['scope3_cat1_emissions'].sum()

        # Calculate data quality score
        avg_completeness = merged['data_completeness'].mean()

        result = {
            'category': 'Scope 3 - Category 1: Purchased Goods and Services',
            'metric_code': 'E1-1-S3-Cat1',
            'value': round(total_scope3, 2),
            'unit': 'tCO2e',
            'number_of_suppliers': len(merged),
            'data_quality_score': round(avg_completeness, 2),
            'calculation_date': DeterministicClock.now().isoformat(),
            'supplier_breakdown': merged[[
                'supplier_id',
                'supplier_name',
                'emissions_intensity',
                'purchase_amount_million_eur',
                'scope3_cat1_emissions'
            ]].to_dict('records')
        }

        logger.info(f"Scope 3 Cat 1 Emissions: {result['value']} {result['unit']}")

        return result

    def score_suppliers(self, supplier_data: pd.DataFrame) -> pd.DataFrame:
        """
        Score suppliers based on ESG performance

        Args:
            supplier_data: DataFrame with supplier data

        Returns:
            DataFrame with ESG scores and rankings
        """
        logger.info("Scoring suppliers based on ESG performance...")

        # Calculate ESG score
        # Lower emissions intensity is better
        max_intensity = supplier_data['emissions_intensity'].max()

        supplier_data['esg_score'] = (
            # Renewable energy (40% weight)
            supplier_data['renewable_energy_pct'] * 0.4 +
            # Emissions intensity (40% weight, inverted)
            (100 - (supplier_data['emissions_intensity'] / max_intensity * 100)) * 0.4 +
            # Data completeness (20% weight)
            supplier_data['data_completeness'] * 0.2
        )

        # Rank suppliers
        supplier_data['rank'] = supplier_data['esg_score'].rank(ascending=False)

        # Categorize performance
        supplier_data['performance_category'] = pd.cut(
            supplier_data['esg_score'],
            bins=[0, 40, 60, 80, 100],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )

        # Sort by rank
        supplier_data = supplier_data.sort_values('rank')

        logger.info("Supplier scoring complete")

        return supplier_data

    def identify_high_risk_suppliers(
        self,
        supplier_data: pd.DataFrame,
        threshold_score: float = 50.0
    ) -> pd.DataFrame:
        """
        Identify high-risk suppliers based on ESG scores

        Args:
            supplier_data: DataFrame with scored suppliers
            threshold_score: ESG score threshold for high risk

        Returns:
            DataFrame with high-risk suppliers
        """
        # Ensure suppliers are scored
        if 'esg_score' not in supplier_data.columns:
            supplier_data = self.score_suppliers(supplier_data)

        high_risk = supplier_data[supplier_data['esg_score'] < threshold_score].copy()

        logger.warning(f"Identified {len(high_risk)} high-risk suppliers")

        return high_risk

    def generate_supplier_report(
        self,
        supplier_data: pd.DataFrame,
        scope3_result: Dict[str, Any],
        output_path: str
    ):
        """
        Generate comprehensive supplier ESG report

        Args:
            supplier_data: DataFrame with supplier data
            scope3_result: Scope 3 calculation results
            output_path: Path to save report
        """
        # Score suppliers if not already done
        if 'esg_score' not in supplier_data.columns:
            supplier_data = self.score_suppliers(supplier_data)

        report = {
            'report_date': DeterministicClock.now().isoformat(),
            'summary': {
                'total_suppliers': len(supplier_data),
                'scope3_emissions': scope3_result,
                'average_esg_score': round(supplier_data['esg_score'].mean(), 2),
                'suppliers_by_category': supplier_data['performance_category'].value_counts().to_dict()
            },
            'top_performers': supplier_data.head(10)[[
                'supplier_id', 'supplier_name', 'esg_score', 'rank', 'performance_category'
            ]].to_dict('records'),
            'high_risk_suppliers': self.identify_high_risk_suppliers(supplier_data)[[
                'supplier_id', 'supplier_name', 'esg_score', 'emissions_intensity'
            ]].to_dict('records'),
            'emissions_intensity_stats': {
                'mean': round(supplier_data['emissions_intensity'].mean(), 2),
                'median': round(supplier_data['emissions_intensity'].median(), 2),
                'min': round(supplier_data['emissions_intensity'].min(), 2),
                'max': round(supplier_data['emissions_intensity'].max(), 2)
            }
        }

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Supplier report saved to {output_file}")

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
        'supplier_portal_url': 'https://suppliers.example.com',
        'api_key': 'DEMO_KEY'
    }

    agent = CSRDSupplyChainAgent(config)

    # Request data from suppliers
    supplier_ids = [f"SUP-{i:03d}" for i in range(1, 51)]  # 50 suppliers
    requests = agent.request_supplier_data(supplier_ids, reporting_year=2024)
    logger.info(f"Sent {requests['total_requests']} data requests")

    # Collect responses (mock data)
    supplier_data = agent.collect_supplier_responses()
    logger.info(f"Received data from {len(supplier_data)} suppliers")

    # Score suppliers
    scored_suppliers = agent.score_suppliers(supplier_data)

    logger.info("\nTop 3 Suppliers:")
    for _, row in scored_suppliers.head(3).iterrows():
        logger.info(f"  {row['supplier_name']}: {row['esg_score']:.1f} ({row['performance_category']})")

    # Prepare purchase data
    purchase_data = pd.DataFrame({
        'supplier_id': supplier_data['supplier_id'],
        'purchase_amount_million_eur': [15.0, 8.0, 20.0]  # Purchase amounts
    })

    # Calculate Scope 3
    scope3 = agent.calculate_scope3_emissions(supplier_data, purchase_data)
    logger.info(f"\nScope 3 Cat 1 Emissions: {scope3['value']} {scope3['unit']}")

    # Generate report
    report = agent.generate_supplier_report(
        supplier_data,
        scope3,
        'output/supplier_esg_report_2024.json'
    )

    logger.info(f"\n✅ Supply chain analysis complete")
    logger.info(f"Average ESG Score: {report['summary']['average_esg_score']}")
    logger.info(f"High-risk suppliers: {len(report['high_risk_suppliers'])}")
