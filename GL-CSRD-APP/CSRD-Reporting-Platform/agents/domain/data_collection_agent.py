"""
CSRD Data Collection Agent

Automates ESG data collection from enterprise systems including:
- ERP systems (SAP, Oracle, Microsoft Dynamics)
- Energy management systems
- HR systems
- IoT sensors and meters

Author: GreenLang AI Team
Date: 2025-10-18
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CSRDDataCollectionAgent:
    """
    Automate ESG data collection from internal enterprise systems

    Features:
    - Multi-system integration (ERP, Energy, HR, IoT)
    - Automated data mapping to ESRS metrics
    - Real-time data quality monitoring
    - Scheduled collection support
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Data Collection Agent

        Args:
            config: Configuration dictionary with system connection details
        """
        self.config = config or {}
        self.connectors = self._initialize_connectors()
        self.metric_mappings = self._load_metric_mappings()

    def _initialize_connectors(self) -> Dict[str, Any]:
        """Initialize data source connectors"""
        connectors = {}

        # ERP connector
        if self.config.get('erp', {}).get('enabled'):
            connectors['erp'] = self._init_erp_connector()

        # Energy system connector
        if self.config.get('energy_system', {}).get('enabled'):
            connectors['energy'] = self._init_energy_connector()

        # HR system connector
        if self.config.get('hr_system', {}).get('enabled'):
            connectors['hr'] = self._init_hr_connector()

        # IoT connector
        if self.config.get('iot_platform', {}).get('enabled'):
            connectors['iot'] = self._init_iot_connector()

        logger.info(f"Initialized {len(connectors)} data connectors")
        return connectors

    def _init_erp_connector(self) -> Dict[str, Any]:
        """Initialize ERP system connector"""
        erp_config = self.config.get('erp', {})

        return {
            'type': erp_config.get('type', 'generic'),
            'enabled': True,
            'queries': self._load_erp_queries(erp_config.get('type', 'generic'))
        }

    def _load_erp_queries(self, erp_type: str) -> Dict[str, str]:
        """Load pre-configured SQL queries for ERP system"""
        # In production, these would be actual SQL queries
        # Here we provide templates

        if erp_type == 'sap':
            return {
                'energy_consumption': "SELECT material_code, SUM(quantity) as total, unit, fiscal_year FROM energy_table WHERE fiscal_year = {year}",
                'emissions': "SELECT emission_type, SUM(co2_equivalent) as total, period FROM emissions_table WHERE period = {year}",
                'waste': "SELECT category, SUM(weight) as total, disposal_method FROM waste_table WHERE fiscal_year = {year}"
            }
        elif erp_type == 'oracle':
            return {
                'energy_consumption': "SELECT material_code, SUM(quantity) as total, unit, fiscal_year FROM energy_consumption WHERE fiscal_year = :year",
                'emissions': "SELECT emission_type, SUM(co2_equiv) as total FROM emissions WHERE fiscal_year = :year"
            }
        else:
            return {
                'energy_consumption': "Generic energy query",
                'emissions': "Generic emissions query"
            }

    def _init_energy_connector(self) -> Dict[str, Any]:
        """Initialize energy management system connector"""
        energy_config = self.config.get('energy_system', {})

        return {
            'api_url': energy_config.get('api_url'),
            'api_key': energy_config.get('api_key'),
            'enabled': True,
            'endpoints': {
                'electricity': '/api/v1/electricity/consumption',
                'gas': '/api/v1/gas/consumption',
                'renewable': '/api/v1/renewable/generation'
            }
        }

    def _init_hr_connector(self) -> Dict[str, Any]:
        """Initialize HR system connector"""
        hr_config = self.config.get('hr_system', {})

        return {
            'api_url': hr_config.get('api_url'),
            'api_key': hr_config.get('api_key'),
            'enabled': True,
            'endpoints': {
                'employees': '/api/v1/employees/count',
                'diversity': '/api/v1/diversity/metrics',
                'training': '/api/v1/training/hours',
                'safety': '/api/v1/safety/incidents'
            }
        }

    def _init_iot_connector(self) -> Dict[str, Any]:
        """Initialize IoT sensor connector"""
        iot_config = self.config.get('iot_platform', {})

        return {
            'api_url': iot_config.get('api_url'),
            'api_key': iot_config.get('api_key'),
            'enabled': True,
            'sensor_types': ['energy_meter', 'water_meter', 'air_quality']
        }

    def _load_metric_mappings(self) -> Dict[str, str]:
        """Load mappings from data sources to ESRS metrics"""
        return {
            # Energy mappings
            'energy_consumption': 'E1-4',
            'electricity': 'E1-4',
            'gas_consumption': 'E1-4',
            'renewable_energy': 'E1-5',

            # Emissions mappings
            'scope1_emissions': 'E1-1',
            'scope2_emissions': 'E1-2',
            'scope3_emissions': 'E1-3',

            # Water mappings
            'water_consumption': 'E3-1',
            'water_discharge': 'E3-2',

            # Waste mappings
            'waste_generated': 'E5-1',
            'waste_recycled': 'E5-2',

            # Social mappings
            'employees': 'S1-1',
            'diversity_metrics': 'S1-9',
            'training_hours': 'S1-13',
            'safety_incidents': 'S1-14',

            # Governance
            'board_diversity': 'G1-1',
        }

    def collect_all_data(self, reporting_year: int) -> pd.DataFrame:
        """
        Collect ESG data from all configured sources

        Args:
            reporting_year: Year for which to collect data

        Returns:
            DataFrame with collected ESG data mapped to ESRS metrics
        """
        logger.info(f"Starting data collection for year {reporting_year}")

        all_data = []

        # Collect from each source
        if 'erp' in self.connectors:
            erp_data = self._collect_from_erp(reporting_year)
            all_data.extend(erp_data)

        if 'energy' in self.connectors:
            energy_data = self._collect_from_energy_system(reporting_year)
            all_data.extend(energy_data)

        if 'hr' in self.connectors:
            hr_data = self._collect_from_hr_system(reporting_year)
            all_data.extend(hr_data)

        if 'iot' in self.connectors:
            iot_data = self._collect_from_iot(reporting_year)
            all_data.extend(iot_data)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Map to ESRS metrics
        df = self._map_to_esrs_metrics(df)

        logger.info(f"Collected {len(df)} data points")

        return df

    def _collect_from_erp(self, year: int) -> List[Dict[str, Any]]:
        """
        Collect data from ERP system

        In production, this would execute actual SQL queries.
        Here we provide mock data for demonstration.

        Args:
            year: Reporting year

        Returns:
            List of data records
        """
        logger.info("Collecting data from ERP system...")

        # Mock data for demonstration
        data = [
            {
                'source': 'erp',
                'metric_name': 'energy_consumption',
                'value': 1500000,
                'unit': 'kWh',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            },
            {
                'source': 'erp',
                'metric_name': 'scope1_emissions',
                'value': 9072.0,
                'unit': 'tCO2e',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            },
            {
                'source': 'erp',
                'metric_name': 'waste_generated',
                'value': 450,
                'unit': 'tonnes',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'medium'
            }
        ]

        logger.info(f"Collected {len(data)} records from ERP")
        return data

    def _collect_from_energy_system(self, year: int) -> List[Dict[str, Any]]:
        """
        Collect data from energy management system

        Args:
            year: Reporting year

        Returns:
            List of data records
        """
        logger.info("Collecting data from Energy Management System...")

        # Mock data
        data = [
            {
                'source': 'energy_system',
                'metric_name': 'electricity',
                'value': 1200000,
                'unit': 'kWh',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            },
            {
                'source': 'energy_system',
                'metric_name': 'gas_consumption',
                'value': 45000,
                'unit': 'm3',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            },
            {
                'source': 'energy_system',
                'metric_name': 'renewable_energy',
                'value': 480000,
                'unit': 'kWh',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            }
        ]

        logger.info(f"Collected {len(data)} records from Energy System")
        return data

    def _collect_from_hr_system(self, year: int) -> List[Dict[str, Any]]:
        """
        Collect data from HR system

        Args:
            year: Reporting year

        Returns:
            List of data records
        """
        logger.info("Collecting data from HR System...")

        # Mock data
        data = [
            {
                'source': 'hr_system',
                'metric_name': 'employees',
                'value': 325,
                'unit': 'FTE',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            },
            {
                'source': 'hr_system',
                'metric_name': 'training_hours',
                'value': 12500,
                'unit': 'hours',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'medium'
            },
            {
                'source': 'hr_system',
                'metric_name': 'safety_incidents',
                'value': 3,
                'unit': 'count',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            }
        ]

        logger.info(f"Collected {len(data)} records from HR System")
        return data

    def _collect_from_iot(self, year: int) -> List[Dict[str, Any]]:
        """
        Collect data from IoT sensors

        Args:
            year: Reporting year

        Returns:
            List of data records
        """
        logger.info("Collecting data from IoT sensors...")

        # Mock data
        data = [
            {
                'source': 'iot_sensors',
                'metric_name': 'water_consumption',
                'value': 25000,
                'unit': 'm3',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            },
            {
                'source': 'iot_sensors',
                'metric_name': 'air_quality',
                'value': 42,
                'unit': 'AQI',
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat(),
                'data_quality': 'medium'
            }
        ]

        logger.info(f"Collected {len(data)} records from IoT sensors")
        return data

    def _map_to_esrs_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map collected data to ESRS metric codes

        Args:
            df: DataFrame with collected data

        Returns:
            DataFrame with ESRS metric codes added
        """
        df['metric_code'] = df['metric_name'].map(self.metric_mappings)

        # Filter out unmapped metrics
        unmapped = df[df['metric_code'].isna()]
        if len(unmapped) > 0:
            logger.warning(f"Found {len(unmapped)} unmapped metrics: {unmapped['metric_name'].unique().tolist()}")

        df = df[df['metric_code'].notna()]

        return df

    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality

        Args:
            df: DataFrame with collected data

        Returns:
            Data quality assessment
        """
        total_records = len(df)

        if total_records == 0:
            return {
                'overall_score': 0.0,
                'total_records': 0,
                'completeness': 0.0,
                'timeliness': 0.0
            }

        # Completeness: % of non-null values
        completeness = (df.notna().sum().sum() / df.size) * 100

        # Quality scores from data
        quality_scores = df['data_quality'].value_counts(normalize=True).to_dict()

        # Calculate weighted quality score
        quality_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        weighted_quality = sum(
            quality_scores.get(level, 0) * quality_map.get(level, 0)
            for level in quality_map.keys()
        ) * 100

        overall_score = (completeness * 0.5 + weighted_quality * 0.5)

        return {
            'overall_score': round(overall_score, 2),
            'total_records': total_records,
            'completeness': round(completeness, 2),
            'quality_distribution': quality_scores,
            'weighted_quality': round(weighted_quality, 2)
        }

    def export_to_csv(self, df: pd.DataFrame, output_path: str):
        """
        Export collected data to CSV

        Args:
            df: DataFrame to export
            output_path: Path to save CSV file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)
        logger.info(f"Exported data to {output_file}")

    def schedule_collection(self, frequency: str = 'monthly'):
        """
        Schedule automated data collection

        Args:
            frequency: Collection frequency ('daily', 'weekly', 'monthly')
        """
        logger.info(f"Scheduling {frequency} data collection")

        # In production, this would use a scheduler like APScheduler or Celery
        # For now, we just log the intent

        schedule_config = {
            'frequency': frequency,
            'enabled': True,
            'last_run': None,
            'next_run': datetime.now().isoformat()
        }

        config_file = Path('data/collection_schedule.json')
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(schedule_config, f, indent=2)

        logger.info(f"Schedule configuration saved to {config_file}")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize agent with sample config
    config = {
        'erp': {
            'enabled': True,
            'type': 'sap'
        },
        'energy_system': {
            'enabled': True,
            'api_url': 'https://energy.example.com',
            'api_key': 'DEMO_KEY'
        },
        'hr_system': {
            'enabled': True,
            'api_url': 'https://hr.example.com',
            'api_key': 'DEMO_KEY'
        },
        'iot_platform': {
            'enabled': True,
            'api_url': 'https://iot.example.com',
            'api_key': 'DEMO_KEY'
        }
    }

    agent = CSRDDataCollectionAgent(config)

    # Collect data for 2024
    data = agent.collect_all_data(reporting_year=2024)

    logger.info(f"\nCollected Data Summary:")
    logger.info(f"Total records: {len(data)}")
    logger.info(f"Unique metrics: {data['metric_code'].nunique()}")
    logger.info(f"Data sources: {data['source'].unique().tolist()}")

    # Assess data quality
    quality = agent.assess_data_quality(data)
    logger.info(f"\nData Quality Assessment:")
    logger.info(f"Overall Score: {quality['overall_score']}%")
    logger.info(f"Completeness: {quality['completeness']}%")

    # Export to CSV
    agent.export_to_csv(data, 'output/collected_esg_data_2024.csv')

    # Schedule future collections
    agent.schedule_collection(frequency='monthly')

    logger.info("\n✅ Data collection complete")
