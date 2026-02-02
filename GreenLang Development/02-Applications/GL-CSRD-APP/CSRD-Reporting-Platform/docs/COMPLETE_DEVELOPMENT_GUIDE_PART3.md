# GL-CSRD-APP Complete Development Guide
## Part 3: Domain Specialization, Operations & Business Strategy

**Document Version:** 1.0
**Last Updated:** 2025-10-18
**Continuation of:** COMPLETE_DEVELOPMENT_GUIDE_PART2.md

---

## 3.4 Week 3: Domain Specialization (Days 11-15)

### **Day 11-15: Phase 10 - CSRD-Specific Domain Agents (4 Agents)**

**Objective:** Create 4 specialized domain agents for CSRD intelligence and automation

#### **Domain Agent Architecture**

The 4 CSRD-specific domain agents complement the 6 core pipeline agents and 14 GreenLang platform agents:

```
18-Agent Ecosystem:
├── Core Pipeline (6 agents)
│   ├── IntakeAgent
│   ├── MaterialityAgent
│   ├── CalculatorAgent
│   ├── AggregatorAgent
│   ├── ReportingAgent
│   └── AuditAgent
│
├── GreenLang Platform (14 agents)
│   ├── GL-CodeSentinel
│   ├── GL-SecScan
│   ├── GL-SpecGuardian
│   ├── GL-SupplyChainSentinel
│   ├── GL-PolicyLinter
│   ├── GL-PackQC
│   ├── GL-HubRegistrar
│   ├── GL-ExitBarAuditor
│   ├── GL-DeterminismAuditor
│   ├── GL-DataFlowGuardian
│   ├── GL-ConnectorValidator
│   ├── GL-TaskChecker
│   ├── GL-ProductDevelopmentTracker
│   └── GL-ProjectStatusReporter
│
└── CSRD Domain (4 agents) ← NEW
    ├── CSRD-RegulatoryIntelligenceAgent
    ├── CSRD-DataCollectionAgent
    ├── CSRD-SupplyChainAgent
    └── CSRD-AutomatedFilingAgent
```

---

### **Agent 1: CSRD-RegulatoryIntelligenceAgent**

**Purpose:** Monitor regulatory updates, interpret new guidance, and auto-update compliance rules

**Key Features:**
- Web scraping of EFRAG, EU Commission, national regulator websites
- RAG-based regulatory document analysis
- Automatic rule generation for new requirements
- Alert system for regulatory changes
- Impact analysis for updates

**Implementation:**

```python
# agents/domain/csrd_regulatory_intelligence_agent.py

"""
CSRD Regulatory Intelligence Agent

Monitors regulatory changes and auto-updates compliance framework
"""

from typing import Dict, List, Any
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import hashlib
from anthropic import Anthropic
from chromadb import Client as ChromaClient

class CSRDRegulatoryIntelligenceAgent:
    """Monitor and interpret CSRD/ESRS regulatory updates"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = Anthropic(api_key=config['anthropic_api_key'])
        self.vector_db = ChromaClient()
        self.regulatory_sources = self._load_sources()

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
                'name': 'EU Commission',
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

    async def monitor_regulatory_updates(self) -> List[Dict[str, Any]]:
        """Check all sources for regulatory updates"""
        updates = []

        for source in self.regulatory_sources:
            print(f"Checking {source['name']}...")

            try:
                new_documents = await self._check_source(source)

                for doc in new_documents:
                    update = await self._analyze_document(doc, source)
                    updates.append(update)

            except Exception as e:
                print(f"Error checking {source['name']}: {e}")
                continue

        return updates

    async def _check_source(self, source: Dict[str, str]) -> List[Dict[str, Any]]:
        """Check a regulatory source for new documents"""
        # Fetch webpage
        response = requests.get(source['url'])
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find document links
        document_links = soup.find_all('a', href=lambda x: x and ('.pdf' in x or 'document' in x))

        new_documents = []

        for link in document_links:
            doc_url = link.get('href')
            doc_title = link.text.strip()

            # Check if already processed
            doc_hash = hashlib.sha256(doc_url.encode()).hexdigest()

            if not self._is_processed(doc_hash):
                new_documents.append({
                    'url': doc_url,
                    'title': doc_title,
                    'hash': doc_hash,
                    'found_date': datetime.now().isoformat()
                })

        return new_documents

    def _is_processed(self, doc_hash: str) -> bool:
        """Check if document already processed"""
        # Query vector DB
        results = self.vector_db.query(
            collection_name='processed_documents',
            query_texts=[doc_hash],
            n_results=1
        )

        return len(results['ids'][0]) > 0

    async def _analyze_document(self, document: Dict[str, Any], source: Dict[str, str]) -> Dict[str, Any]:
        """Analyze regulatory document using LLM + RAG"""
        # Download document
        doc_content = self._download_document(document['url'])

        # Extract text
        doc_text = self._extract_text(doc_content)

        # Analyze with LLM
        analysis_prompt = f"""
You are a CSRD/ESRS regulatory expert. Analyze this regulatory document and extract:

1. Document type (e.g., new standard, amendment, guidance, Q&A)
2. Key changes or new requirements
3. Effective date
4. Impact assessment (high/medium/low)
5. Affected ESRS standards (E1-E5, S1-S4, G1)
6. Required actions for companies
7. Recommended system updates

Document:
{doc_text[:10000]}  # First 10k characters

Provide a structured JSON response.
"""

        response = await self.llm.messages.create(
            model="claude-sonnet-4",
            max_tokens=4096,
            messages=[{"role": "user", "content": analysis_prompt}]
        )

        analysis = response.content[0].text

        # Store in vector DB for RAG
        self.vector_db.add(
            collection_name='regulatory_documents',
            documents=[doc_text],
            metadatas=[{
                'source': source['name'],
                'url': document['url'],
                'title': document['title'],
                'analysis': analysis,
                'date': datetime.now().isoformat()
            }],
            ids=[document['hash']]
        )

        # Mark as processed
        self.vector_db.add(
            collection_name='processed_documents',
            documents=[document['hash']],
            ids=[document['hash']]
        )

        return {
            'document': document,
            'source': source,
            'analysis': analysis,
            'requires_action': self._requires_action(analysis)
        }

    def _download_document(self, url: str) -> bytes:
        """Download document content"""
        response = requests.get(url)
        return response.content

    def _extract_text(self, content: bytes) -> str:
        """Extract text from PDF or HTML"""
        # Implementation depends on document type
        # Use libraries like pdfplumber, PyPDF2, etc.
        pass

    def _requires_action(self, analysis: str) -> bool:
        """Determine if regulatory update requires immediate action"""
        # Check for keywords indicating mandatory changes
        action_keywords = [
            'must', 'shall', 'required', 'mandatory',
            'effective immediately', 'compliance deadline'
        ]

        return any(keyword in analysis.lower() for keyword in action_keywords)

    async def generate_compliance_rules(self, regulatory_update: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Auto-generate compliance rules from regulatory update"""
        analysis = regulatory_update['analysis']

        rule_generation_prompt = f"""
Based on this regulatory update analysis, generate compliance rules in YAML format.

Analysis:
{analysis}

Generate rules following this structure:

```yaml
rule_id: ESRS-XX-RXX
rule_name: "Rule description"
severity: high|medium|low
validation_logic:
  - condition: "..."
    error_message: "..."
affected_metrics:
  - metric_code
authoritative_source: "..."
effective_date: "YYYY-MM-DD"
```

Provide 1-5 rules based on the update.
"""

        response = await self.llm.messages.create(
            model="claude-sonnet-4",
            max_tokens=4096,
            messages=[{"role": "user", "content": rule_generation_prompt}]
        )

        rules_yaml = response.content[0].text

        # Parse YAML
        import yaml
        rules = yaml.safe_load(rules_yaml)

        return rules

    async def send_alerts(self, updates: List[Dict[str, Any]]) -> None:
        """Send alerts for high-impact regulatory changes"""
        high_impact_updates = [
            u for u in updates
            if 'high' in u['analysis'].lower()
        ]

        if not high_impact_updates:
            return

        # Send email/Slack notifications
        for update in high_impact_updates:
            self._send_notification(
                title=f"🚨 High-Impact CSRD Update: {update['document']['title']}",
                message=f"Source: {update['source']['name']}\n\n{update['analysis']}",
                urgency='high'
            )

    def _send_notification(self, title: str, message: str, urgency: str):
        """Send notification via email/Slack"""
        # Implementation depends on notification system
        print(f"[{urgency.upper()}] {title}\n{message}")

# Usage example

async def main():
    config = {
        'anthropic_api_key': 'YOUR_API_KEY',
        'notification_webhook': 'https://hooks.slack.com/...'
    }

    agent = CSRDRegulatoryIntelligenceAgent(config)

    # Monitor for updates
    updates = await agent.monitor_regulatory_updates()

    print(f"Found {len(updates)} regulatory updates")

    # Generate compliance rules
    for update in updates:
        if update['requires_action']:
            rules = await agent.generate_compliance_rules(update)
            print(f"Generated {len(rules)} new compliance rules")

    # Send alerts
    await agent.send_alerts(updates)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Testing:**

```python
# tests/unit/test_csrd_regulatory_intelligence_agent.py

import pytest
from agents.domain.csrd_regulatory_intelligence_agent import CSRDRegulatoryIntelligenceAgent
from unittest.mock import patch, Mock

@pytest.fixture
def agent():
    config = {
        'anthropic_api_key': 'test_key',
        'notification_webhook': 'test_webhook'
    }
    return CSRDRegulatoryIntelligenceAgent(config)

class TestRegulatoryMonitoring:
    """Test regulatory source monitoring"""

    @patch('requests.get')
    async def test_check_source(self, mock_get, agent):
        """Test checking a regulatory source"""
        # Mock HTML response
        mock_get.return_value.content = b'''
        <html>
            <a href="/documents/new-esrs-guidance.pdf">New ESRS E1 Guidance</a>
            <a href="/documents/amendment-2024.pdf">ESRS Amendment 2024</a>
        </html>
        '''

        source = {
            'name': 'EFRAG',
            'url': 'https://www.efrag.org/lab6',
            'type': 'primary'
        }

        documents = await agent._check_source(source)

        assert len(documents) == 2
        assert 'new-esrs-guidance.pdf' in documents[0]['url']

    @patch.object(CSRDRegulatoryIntelligenceAgent, '_download_document')
    @patch.object(CSRDRegulatoryIntelligenceAgent, '_extract_text')
    @patch('anthropic.Anthropic')
    async def test_analyze_document(self, mock_llm, mock_extract, mock_download, agent):
        """Test document analysis with LLM"""
        mock_download.return_value = b'PDF content'
        mock_extract.return_value = 'ESRS E1 amendment...'

        mock_llm.return_value.messages.create.return_value = Mock(
            content=[Mock(text='{"type": "amendment", "impact": "high"}')]
        )

        document = {
            'url': 'https://example.com/doc.pdf',
            'title': 'ESRS E1 Amendment',
            'hash': 'abc123'
        }

        source = {'name': 'EFRAG', 'url': 'https://efrag.org'}

        analysis = await agent._analyze_document(document, source)

        assert analysis['requires_action'] is not None

class TestRuleGeneration:
    """Test automatic rule generation"""

    @patch('anthropic.Anthropic')
    async def test_generate_compliance_rules(self, mock_llm, agent):
        """Test generating compliance rules from regulatory update"""
        mock_llm.return_value.messages.create.return_value = Mock(
            content=[Mock(text='''
rule_id: ESRS-E1-R100
rule_name: "New emissions disclosure requirement"
severity: high
validation_logic:
  - condition: "metric E1-1 must be present"
    error_message: "Missing E1-1"
affected_metrics:
  - E1-1
authoritative_source: "ESRS E1 Amendment 2024"
effective_date: "2025-01-01"
            ''')]
        )

        regulatory_update = {
            'analysis': 'New requirement for Scope 1 emissions disclosure...'
        }

        rules = await agent.generate_compliance_rules(regulatory_update)

        assert len(rules) > 0
        assert 'rule_id' in rules[0]
        assert 'severity' in rules[0]
```

---

### **Agent 2: CSRD-DataCollectionAgent**

**Purpose:** Automate ESG data collection from internal systems, APIs, and integrations

**Key Features:**
- ERP integration (SAP, Oracle, Microsoft Dynamics)
- Energy management system integration
- HR system integration
- IoT sensor data ingestion
- Automated data mapping and transformation
- Real-time data quality monitoring

**Implementation:**

```python
# agents/domain/csrd_data_collection_agent.py

"""
CSRD Data Collection Agent

Automates ESG data collection from enterprise systems
"""

from typing import Dict, List, Any
import pandas as pd
from sqlalchemy import create_engine
import requests
from datetime import datetime

class CSRDDataCollectionAgent:
    """Automate ESG data collection from internal systems"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors = self._initialize_connectors()

    def _initialize_connectors(self) -> Dict[str, Any]:
        """Initialize data source connectors"""
        return {
            'erp': self._init_erp_connector(),
            'energy': self._init_energy_connector(),
            'hr': self._init_hr_connector(),
            'iot': self._init_iot_connector()
        }

    def _init_erp_connector(self):
        """Initialize ERP system connector (SAP, Oracle, etc.)"""
        erp_config = self.config.get('erp')

        if not erp_config:
            return None

        # Connect to ERP database
        connection_string = f"{erp_config['type']}://{erp_config['host']}:{erp_config['port']}/{erp_config['database']}"
        engine = create_engine(connection_string)

        return {
            'type': erp_config['type'],
            'engine': engine,
            'queries': self._load_erp_queries(erp_config['type'])
        }

    def _load_erp_queries(self, erp_type: str) -> Dict[str, str]:
        """Load pre-configured SQL queries for ERP system"""
        # SAP queries for ESG data
        if erp_type == 'sap':
            return {
                'energy_consumption': '''
                    SELECT
                        material_code,
                        SUM(quantity) as total_consumption,
                        unit,
                        fiscal_year
                    FROM energy_consumption_table
                    WHERE fiscal_year = :year
                    GROUP BY material_code, unit, fiscal_year
                ''',
                'emissions': '''
                    SELECT
                        emission_type,
                        SUM(co2_equivalent) as total_emissions,
                        reporting_period
                    FROM emissions_table
                    WHERE reporting_period = :year
                    GROUP BY emission_type, reporting_period
                ''',
                'waste': '''
                    SELECT
                        waste_category,
                        SUM(weight_tonnes) as total_waste,
                        disposal_method,
                        fiscal_year
                    FROM waste_management_table
                    WHERE fiscal_year = :year
                    GROUP BY waste_category, disposal_method, fiscal_year
                '''
            }

        # Oracle queries
        elif erp_type == 'oracle':
            # Similar queries adapted for Oracle syntax
            pass

        return {}

    def _init_energy_connector(self):
        """Initialize energy management system connector"""
        energy_config = self.config.get('energy_system')

        if not energy_config:
            return None

        return {
            'api_url': energy_config['api_url'],
            'api_key': energy_config['api_key'],
            'endpoints': {
                'electricity': '/api/v1/electricity/consumption',
                'gas': '/api/v1/gas/consumption',
                'renewable': '/api/v1/renewable/generation'
            }
        }

    def _init_hr_connector(self):
        """Initialize HR system connector (Workday, SAP SuccessFactors, etc.)"""
        hr_config = self.config.get('hr_system')

        if not hr_config:
            return None

        return {
            'api_url': hr_config['api_url'],
            'api_key': hr_config['api_key'],
            'endpoints': {
                'employees': '/api/v1/employees',
                'diversity': '/api/v1/diversity_metrics',
                'training': '/api/v1/training_hours'
            }
        }

    def _init_iot_connector(self):
        """Initialize IoT sensor connector"""
        iot_config = self.config.get('iot_platform')

        if not iot_config:
            return None

        return {
            'api_url': iot_config['api_url'],
            'api_key': iot_config['api_key'],
            'sensor_types': ['energy_meter', 'water_meter', 'air_quality']
        }

    async def collect_all_data(self, reporting_year: int) -> pd.DataFrame:
        """Collect ESG data from all sources"""
        all_data = []

        # Collect from ERP
        if self.connectors['erp']:
            erp_data = await self._collect_from_erp(reporting_year)
            all_data.extend(erp_data)

        # Collect from energy system
        if self.connectors['energy']:
            energy_data = await self._collect_from_energy_system(reporting_year)
            all_data.extend(energy_data)

        # Collect from HR system
        if self.connectors['hr']:
            hr_data = await self._collect_from_hr_system(reporting_year)
            all_data.extend(hr_data)

        # Collect from IoT sensors
        if self.connectors['iot']:
            iot_data = await self._collect_from_iot(reporting_year)
            all_data.extend(iot_data)

        # Combine into DataFrame
        df = pd.DataFrame(all_data)

        # Map to ESRS metrics
        df = self._map_to_esrs_metrics(df)

        return df

    async def _collect_from_erp(self, year: int) -> List[Dict[str, Any]]:
        """Collect data from ERP system"""
        erp = self.connectors['erp']
        engine = erp['engine']
        queries = erp['queries']

        data = []

        for metric_name, query in queries.items():
            df = pd.read_sql(query, engine, params={'year': year})

            for _, row in df.iterrows():
                data.append({
                    'source': 'erp',
                    'metric_name': metric_name,
                    'value': row['total_consumption'] if 'total_consumption' in row else row.iloc[1],
                    'unit': row['unit'] if 'unit' in row else 'unknown',
                    'reporting_year': year,
                    'collection_timestamp': datetime.now().isoformat()
                })

        return data

    async def _collect_from_energy_system(self, year: int) -> List[Dict[str, Any]]:
        """Collect data from energy management system"""
        energy = self.connectors['energy']

        data = []

        for energy_type, endpoint in energy['endpoints'].items():
            url = f"{energy['api_url']}{endpoint}"
            headers = {'Authorization': f"Bearer {energy['api_key']}"}
            params = {'year': year}

            response = requests.get(url, headers=headers, params=params)
            response_data = response.json()

            data.append({
                'source': 'energy_system',
                'metric_name': energy_type,
                'value': response_data['total_consumption'],
                'unit': response_data['unit'],
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat()
            })

        return data

    async def _collect_from_hr_system(self, year: int) -> List[Dict[str, Any]]:
        """Collect data from HR system"""
        hr = self.connectors['hr']

        data = []

        for metric_type, endpoint in hr['endpoints'].items():
            url = f"{hr['api_url']}{endpoint}"
            headers = {'Authorization': f"Bearer {hr['api_key']}"}
            params = {'year': year}

            response = requests.get(url, headers=headers, params=params)
            response_data = response.json()

            data.append({
                'source': 'hr_system',
                'metric_name': metric_type,
                'value': response_data['value'],
                'unit': response_data['unit'],
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat()
            })

        return data

    async def _collect_from_iot(self, year: int) -> List[Dict[str, Any]]:
        """Collect data from IoT sensors"""
        iot = self.connectors['iot']

        data = []

        for sensor_type in iot['sensor_types']:
            url = f"{iot['api_url']}/sensors/{sensor_type}/readings"
            headers = {'Authorization': f"Bearer {iot['api_key']}"}
            params = {'year': year, 'aggregation': 'sum'}

            response = requests.get(url, headers=headers, params=params)
            response_data = response.json()

            data.append({
                'source': 'iot_sensors',
                'metric_name': sensor_type,
                'value': response_data['total'],
                'unit': response_data['unit'],
                'reporting_year': year,
                'collection_timestamp': datetime.now().isoformat()
            })

        return data

    def _map_to_esrs_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map collected data to ESRS metric codes"""
        # Mapping dictionary
        mappings = {
            'energy_consumption': 'E1-4',
            'electricity': 'E1-4',
            'emissions': 'E1-1',
            'employees': 'S1-1',
            'diversity': 'S1-9',
            'training': 'S1-13'
        }

        df['metric_code'] = df['metric_name'].map(mappings)

        # Remove unmapped metrics
        df = df[df['metric_code'].notna()]

        return df

    async def schedule_collection(self, frequency: str = 'monthly'):
        """Schedule automated data collection"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler()

        if frequency == 'daily':
            scheduler.add_job(
                self.collect_all_data,
                'cron',
                hour=2,  # Run at 2 AM
                args=[datetime.now().year]
            )
        elif frequency == 'weekly':
            scheduler.add_job(
                self.collect_all_data,
                'cron',
                day_of_week='mon',
                hour=2,
                args=[datetime.now().year]
            )
        elif frequency == 'monthly':
            scheduler.add_job(
                self.collect_all_data,
                'cron',
                day=1,
                hour=2,
                args=[datetime.now().year]
            )

        scheduler.start()

# Usage example

async def main():
    config = {
        'erp': {
            'type': 'sap',
            'host': 'erp.company.com',
            'port': 5432,
            'database': 'sap_prod'
        },
        'energy_system': {
            'api_url': 'https://energy.company.com',
            'api_key': 'YOUR_API_KEY'
        },
        'hr_system': {
            'api_url': 'https://workday.company.com',
            'api_key': 'YOUR_API_KEY'
        },
        'iot_platform': {
            'api_url': 'https://iot.company.com',
            'api_key': 'YOUR_API_KEY'
        }
    }

    agent = CSRDDataCollectionAgent(config)

    # Collect data for 2024
    data = await agent.collect_all_data(reporting_year=2024)

    print(f"Collected {len(data)} ESG data points")
    print(data.head())

    # Save to CSV
    data.to_csv('collected_esg_data_2024.csv', index=False)

    # Schedule automated collection
    await agent.schedule_collection(frequency='monthly')

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

### **Agent 3: CSRD-SupplyChainAgent**

**Purpose:** Collect and aggregate Scope 3 emissions from supply chain partners

**Key Features:**
- Supplier portal integration
- Automated data requests to suppliers
- Supply chain emissions calculation
- Supplier scoring and ranking
- Supply chain risk assessment

**Implementation:**

```python
# agents/domain/csrd_supply_chain_agent.py

"""
CSRD Supply Chain Agent

Manages Scope 3 emissions data collection from supply chain
"""

from typing import Dict, List, Any
import pandas as pd
import requests
from datetime import datetime, timedelta

class CSRDSupplyChainAgent:
    """Manage supply chain ESG data collection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supplier_portal_url = config.get('supplier_portal_url')
        self.api_key = config.get('api_key')

    async def request_supplier_data(self, supplier_ids: List[str], reporting_year: int) -> Dict[str, Any]:
        """Send data request to suppliers"""
        requests_sent = []

        for supplier_id in supplier_ids:
            request = {
                'supplier_id': supplier_id,
                'reporting_year': reporting_year,
                'requested_data': [
                    'scope1_emissions',
                    'scope2_emissions',
                    'renewable_energy_percentage',
                    'waste_generated',
                    'water_consumption'
                ],
                'deadline': (datetime.now() + timedelta(days=30)).isoformat(),
                'request_date': datetime.now().isoformat()
            }

            # Send request via supplier portal API
            response = await self._send_request_to_supplier(request)

            requests_sent.append({
                'supplier_id': supplier_id,
                'request_id': response['request_id'],
                'status': response['status']
            })

        return {
            'total_requests': len(requests_sent),
            'requests': requests_sent
        }

    async def _send_request_to_supplier(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send data request to supplier via portal"""
        url = f"{self.supplier_portal_url}/api/v1/data_requests"
        headers = {'Authorization': f"Bearer {self.api_key}"}

        response = requests.post(url, json=request, headers=headers)

        return response.json()

    async def collect_supplier_responses(self) -> pd.DataFrame:
        """Collect completed supplier data submissions"""
        url = f"{self.supplier_portal_url}/api/v1/data_submissions"
        headers = {'Authorization': f"Bearer {self.api_key}"}

        response = requests.get(url, headers=headers)
        submissions = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(submissions)

        return df

    async def calculate_scope3_emissions(self, supplier_data: pd.DataFrame, purchase_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Scope 3 emissions from supplier data"""
        # Merge supplier emissions with purchase amounts
        merged = pd.merge(
            supplier_data,
            purchase_data,
            on='supplier_id',
            how='inner'
        )

        # Calculate Scope 3 Category 1: Purchased goods and services
        merged['scope3_cat1_emissions'] = (
            merged['supplier_emissions_intensity'] * merged['purchase_amount_eur']
        )

        # Sum by category
        scope3_total = merged['scope3_cat1_emissions'].sum()

        return {
            'category': 'Scope 3 - Category 1',
            'metric_code': 'E1-1-S3-Cat1',
            'value': scope3_total,
            'unit': 'tCO2e',
            'number_of_suppliers': len(merged),
            'data_quality_score': self._calculate_data_quality(merged)
        }

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score for supplier data"""
        # Factors: completeness, timeliness, accuracy indicators
        completeness = (data.notna().sum().sum() / data.size)
        timeliness = (data['submission_date'] <= data['deadline']).mean()

        quality_score = (completeness * 0.6 + timeliness * 0.4) * 100

        return round(quality_score, 2)

    async def score_suppliers(self, supplier_data: pd.DataFrame) -> pd.DataFrame:
        """Score suppliers based on ESG performance"""
        # Calculate scores
        supplier_data['esg_score'] = (
            supplier_data['renewable_energy_pct'] * 0.4 +
            (100 - supplier_data['emissions_intensity']) * 0.4 +
            supplier_data['data_completeness'] * 0.2
        )

        # Rank suppliers
        supplier_data['rank'] = supplier_data['esg_score'].rank(ascending=False)

        return supplier_data.sort_values('rank')

# Usage example

async def main():
    config = {
        'supplier_portal_url': 'https://suppliers.company.com',
        'api_key': 'YOUR_API_KEY'
    }

    agent = CSRDSupplyChainAgent(config)

    # Request data from top 100 suppliers
    supplier_ids = [f"SUP-{i:04d}" for i in range(1, 101)]

    requests = await agent.request_supplier_data(supplier_ids, reporting_year=2024)
    print(f"Sent {requests['total_requests']} data requests")

    # Collect responses
    supplier_data = await agent.collect_supplier_responses()
    print(f"Received {len(supplier_data)} supplier responses")

    # Calculate Scope 3 emissions
    purchase_data = pd.DataFrame({
        'supplier_id': supplier_ids,
        'purchase_amount_eur': [100000] * len(supplier_ids)
    })

    scope3 = await agent.calculate_scope3_emissions(supplier_data, purchase_data)
    print(f"Scope 3 emissions: {scope3['value']} {scope3['unit']}")

    # Score suppliers
    scored_suppliers = await agent.score_suppliers(supplier_data)
    print("Top 10 suppliers by ESG score:")
    print(scored_suppliers.head(10)[['supplier_id', 'esg_score', 'rank']])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

### **Agent 4: CSRD-AutomatedFilingAgent**

**Purpose:** Automate regulatory filing submissions to national authorities

**Key Features:**
- ESEF package validation
- Electronic submission to national registers
- Filing status tracking
- Automatic retry on failure
- Compliance verification

**Implementation:**

```python
# agents/domain/csrd_automated_filing_agent.py

"""
CSRD Automated Filing Agent

Automates CSRD report filing to national authorities
"""

from typing import Dict, List, Any
import requests
from pathlib import Path
import zipfile
from lxml import etree

class CSRDAutomatedFilingAgent:
    """Automate CSRD report filing to national authorities"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filing_endpoints = self._load_filing_endpoints()

    def _load_filing_endpoints(self) -> Dict[str, str]:
        """Load national filing endpoints"""
        return {
            'DE': 'https://unternehmensregister.de/api/v1/csrd/submit',
            'FR': 'https://infogreffe.fr/api/v1/csrd/submit',
            'NL': 'https://kvk.nl/api/v1/csrd/submit',
            'IT': 'https://registroimprese.it/api/v1/csrd/submit',
            'ES': 'https://rmc.es/api/v1/csrd/submit'
        }

    async def validate_esef_package(self, package_path: Path) -> Dict[str, Any]:
        """Validate ESEF package before submission"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
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
        with zipfile.ZipFile(package_path, 'r') as zf:
            files = zf.namelist()

            # Check required files
            required_files = ['META-INF/manifest.xml', 'reports/']

            for required in required_files:
                if not any(required in f for f in files):
                    validation_results['valid'] = False
                    validation_results['errors'].append(f'Missing required file: {required}')

            # Validate XHTML
            xhtml_files = [f for f in files if f.endswith('.xhtml')]

            for xhtml_file in xhtml_files:
                xhtml_content = zf.read(xhtml_file)

                try:
                    tree = etree.fromstring(xhtml_content)
                    # Additional XHTML validation
                except Exception as e:
                    validation_results['valid'] = False
                    validation_results['errors'].append(f'Invalid XHTML in {xhtml_file}: {e}')

        return validation_results

    async def submit_filing(self, package_path: Path, country_code: str, company_lei: str) -> Dict[str, Any]:
        """Submit CSRD report to national authority"""
        # Validate package first
        validation = await self.validate_esef_package(package_path)

        if not validation['valid']:
            return {
                'status': 'failed',
                'reason': 'validation_failed',
                'errors': validation['errors']
            }

        # Get filing endpoint
        filing_url = self.filing_endpoints.get(country_code)

        if not filing_url:
            return {
                'status': 'failed',
                'reason': 'unsupported_country',
                'message': f'No filing endpoint for country {country_code}'
            }

        # Prepare submission
        with open(package_path, 'rb') as f:
            files = {'esef_package': f}

            data = {
                'company_lei': company_lei,
                'reporting_year': self.config.get('reporting_year'),
                'submission_date': datetime.now().isoformat()
            }

            headers = {
                'Authorization': f"Bearer {self.config['api_key']}",
                'X-Country-Code': country_code
            }

            # Submit
            response = requests.post(
                filing_url,
                files=files,
                data=data,
                headers=headers
            )

        if response.status_code == 200:
            return {
                'status': 'success',
                'submission_id': response.json()['submission_id'],
                'confirmation_number': response.json()['confirmation_number'],
                'submission_timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'failed',
                'reason': 'submission_error',
                'http_status': response.status_code,
                'message': response.text
            }

    async def track_filing_status(self, submission_id: str, country_code: str) -> Dict[str, Any]:
        """Track filing status"""
        filing_url = self.filing_endpoints.get(country_code)
        status_url = f"{filing_url}/{submission_id}/status"

        headers = {'Authorization': f"Bearer {self.config['api_key']}"}

        response = requests.get(status_url, headers=headers)

        return response.json()

    async def retry_failed_filing(self, submission_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """Retry failed filing"""
        for attempt in range(max_retries):
            print(f"Retry attempt {attempt + 1}/{max_retries}")

            # Retrieve original submission details
            # Re-submit
            # ...

            # If successful, break
            # If failed, continue

        return {'status': 'failed_after_retries', 'attempts': max_retries}

# Usage example

async def main():
    config = {
        'api_key': 'YOUR_API_KEY',
        'reporting_year': 2024
    }

    agent = CSRDAutomatedFilingAgent(config)

    # Validate ESEF package
    package_path = Path('output/company_csrd_2024.zip')

    validation = await agent.validate_esef_package(package_path)

    if validation['valid']:
        print("✅ ESEF package valid")

        # Submit filing
        result = await agent.submit_filing(
            package_path=package_path,
            country_code='DE',
            company_lei='DE123456789012345678'
        )

        if result['status'] == 'success':
            print(f"✅ Filing submitted: {result['confirmation_number']}")

            # Track status
            status = await agent.track_filing_status(
                submission_id=result['submission_id'],
                country_code='DE'
            )

            print(f"Filing status: {status}")
        else:
            print(f"❌ Filing failed: {result['reason']}")
    else:
        print(f"❌ ESEF validation failed: {validation['errors']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Success Criteria for Week 3:**
- ✅ All 4 CSRD domain agents implemented
- ✅ Regulatory intelligence monitoring working
- ✅ Automated data collection from enterprise systems
- ✅ Supply chain Scope 3 calculation functional
- ✅ Automated filing system tested

**Time Estimate:** 30-40 hours (Week 3)

---

## 3.5 Week 4: Integration & Deployment (Days 16-20)

### **Day 16-17: Phase 11 - Full System Integration**

**Objective:** Integrate all 18 agents into cohesive ecosystem

**Tasks:**
1. Integration testing with all agents
2. End-to-end workflow testing
3. Performance optimization
4. Error handling and resilience
5. Documentation updates

**Full System Integration Test:**

```python
# tests/integration/test_full_system_integration.py

import pytest
from csrd_pipeline import CSRDPipeline
from utils.agent_orchestrator import GreenLangAgentOrchestrator
from agents.domain.csrd_regulatory_intelligence_agent import CSRDRegulatoryIntelligenceAgent
from agents.domain.csrd_data_collection_agent import CSRDDataCollectionAgent
from agents.domain.csrd_supply_chain_agent import CSRDSupplyChainAgent
from agents.domain.csrd_automated_filing_agent import CSRDAutomatedFilingAgent

@pytest.fixture
async def full_system():
    """Initialize full 18-agent system"""
    # Core pipeline (6 agents)
    pipeline = CSRDPipeline()

    # GreenLang platform agents (14 agents)
    orchestrator = GreenLangAgentOrchestrator('config/greenlang_agents_config.yaml')

    # CSRD domain agents (4 agents)
    regulatory_agent = CSRDRegulatoryIntelligenceAgent(config={})
    data_collection_agent = CSRDDataCollectionAgent(config={})
    supply_chain_agent = CSRDSupplyChainAgent(config={})
    filing_agent = CSRDAutomatedFilingAgent(config={})

    return {
        'pipeline': pipeline,
        'orchestrator': orchestrator,
        'regulatory': regulatory_agent,
        'data_collection': data_collection_agent,
        'supply_chain': supply_chain_agent,
        'filing': filing_agent
    }

class TestFullSystemIntegration:
    """Test complete 18-agent ecosystem"""

    async def test_end_to_end_csrd_workflow(self, full_system):
        """Test complete CSRD workflow from data collection to filing"""
        # Step 1: Regulatory intelligence - check for updates
        regulatory_updates = await full_system['regulatory'].monitor_regulatory_updates()
        print(f"Step 1: Found {len(regulatory_updates)} regulatory updates")

        # Step 2: Data collection - gather ESG data
        esg_data = await full_system['data_collection'].collect_all_data(reporting_year=2024)
        print(f"Step 2: Collected {len(esg_data)} ESG data points")

        # Step 3: Supply chain - collect Scope 3 data
        supplier_data = await full_system['supply_chain'].collect_supplier_responses()
        print(f"Step 3: Received data from {len(supplier_data)} suppliers")

        # Step 4: Run GreenLang quality gates
        quality_results = await full_system['orchestrator'].run_workflow(
            'development_quality',
            context={'data': esg_data}
        )
        print(f"Step 4: Quality gates - {quality_results['gl_codesentinel']['status']}")

        # Step 5: Execute CSRD pipeline
        pipeline_result = full_system['pipeline'].run(
            input_data=esg_data,
            output_dir='output/'
        )
        print(f"Step 5: Pipeline complete - {pipeline_result['status']}")

        # Step 6: Validate with GreenLang exit bar
        exit_bar_results = await full_system['orchestrator'].run_workflow(
            'release_readiness',
            context={'report_path': pipeline_result['output_path']}
        )
        print(f"Step 6: Exit bar - {exit_bar_results['gl_exitbar_auditor']['status']}")

        # Step 7: Automated filing
        if exit_bar_results['gl_exitbar_auditor']['status'] == 'PASS':
            filing_result = await full_system['filing'].submit_filing(
                package_path=pipeline_result['esef_package_path'],
                country_code='DE',
                company_lei='DE123456789012345678'
            )
            print(f"Step 7: Filing - {filing_result['status']}")

            assert filing_result['status'] == 'success'
        else:
            pytest.skip("Exit bar failed, skipping filing")

        # Verify all steps completed successfully
        assert len(esg_data) > 0
        assert pipeline_result['status'] == 'success'
        assert exit_bar_results['gl_exitbar_auditor']['status'] == 'PASS'

    async def test_agent_fault_tolerance(self, full_system):
        """Test system resilience when agents fail"""
        # Simulate agent failure
        # Verify system continues with degraded functionality
        pass

    async def test_performance_at_scale(self, full_system):
        """Test system performance with large datasets"""
        # Generate 10,000 data points
        # Run through full pipeline
        # Verify <30 minute completion time
        pass
```

---

### **Day 18-20: Phase 12 - Production Deployment**

**Objective:** Deploy to production environment

**Deployment Checklist:**

```markdown
# CSRD Production Deployment Checklist

## Pre-Deployment
- [ ] All tests passing (unit, integration, E2E)
- [ ] Code review completed
- [ ] Security scan passed (GL-SecScan)
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Backup plan prepared

## Infrastructure Setup
- [ ] Production servers provisioned
- [ ] Database configured (with replication)
- [ ] Load balancer configured
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring tools set up (Prometheus, Grafana)
- [ ] Logging aggregation configured (ELK stack)

## Application Deployment
- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Database migrations run
- [ ] Static files deployed
- [ ] Worker processes configured
- [ ] Cron jobs scheduled

## Testing in Production
- [ ] Smoke tests passed
- [ ] Health checks responding
- [ ] API endpoints accessible
- [ ] File uploads working
- [ ] Database connections stable
- [ ] External integrations working

## Monitoring Setup
- [ ] Application metrics tracked
- [ ] Error alerting configured
- [ ] Performance dashboards created
- [ ] Log monitoring active
- [ ] Uptime monitoring configured

## Go-Live
- [ ] Blue-green deployment executed
- [ ] DNS updated (if applicable)
- [ ] Load balancer traffic shifted
- [ ] Old version kept on standby
- [ ] Rollback plan ready

## Post-Deployment
- [ ] Monitor for 24 hours
- [ ] Review error logs
- [ ] Verify performance metrics
- [ ] Customer communication sent
- [ ] Documentation updated
- [ ] Lessons learned documented
```

**Docker Deployment:**

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKERS=4

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "300", "app:app"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/csrd
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    restart: always

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=csrd
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always

  redis:
    image: redis:7-alpine
    restart: always

  worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    depends_on:
      - redis
      - db
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: always

volumes:
  postgres_data:
```

**Kubernetes Deployment:**

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: csrd-app
  labels:
    app: csrd
spec:
  replicas: 3
  selector:
    matchLabels:
      app: csrd
  template:
    metadata:
      labels:
        app: csrd
    spec:
      containers:
      - name: csrd
        image: greenlang/csrd-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: csrd-secrets
              key: database-url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: csrd-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: csrd-service
spec:
  selector:
    app: csrd
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: csrd-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: csrd-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Success Criteria for Week 4:**
- ✅ Full system integration tested
- ✅ All 18 agents working together
- ✅ Production deployment complete
- ✅ Monitoring and alerting active
- ✅ Documentation finalized

---

# PART IV: AGENT ECOSYSTEM INTEGRATION

## 4.1 18-Agent Architecture Overview

### **Agent Categories**

**1. Core Pipeline Agents (6 agents)**
- Purpose: Execute CSRD reporting workflow
- Execution: Sequential pipeline
- Ownership: CSRD application

**2. GreenLang Platform Agents (14 agents)**
- Purpose: Development quality, security, compliance
- Execution: Parallel workflows
- Ownership: GreenLang platform

**3. CSRD Domain Agents (4 agents)**
- Purpose: Domain-specific intelligence and automation
- Execution: On-demand/scheduled
- Ownership: CSRD application

### **Agent Interaction Patterns**

```
Pattern 1: Development Quality Workflow
GL-CodeSentinel → GL-SecScan → GL-SpecGuardian
↓
Quality Report → IntakeAgent (validates code quality before processing)

Pattern 2: Data Pipeline Workflow
CSRD-DataCollectionAgent → IntakeAgent → GL-DataFlowGuardian → CalculatorAgent
↓
GL-DeterminismAuditor (verifies reproducibility)

Pattern 3: Release Workflow
ReportingAgent → GL-PackQC → GL-ExitBarAuditor → CSRD-AutomatedFilingAgent
↓
GL-ProjectStatusReporter (generates release report)

Pattern 4: Regulatory Compliance Workflow
CSRD-RegulatoryIntelligenceAgent → AuditAgent (updates compliance rules)
↓
GL-PolicyLinter (validates new OPA policies)
```

---

## 4.2 Configuration Management

### **Unified Configuration**

```yaml
# config/agents_config.yaml

# Core Pipeline Agents
pipeline:
  intake_agent:
    enabled: true
    throughput_target: 1000  # records/sec
    quality_threshold: 0.85

  materiality_agent:
    enabled: true
    llm_model: "claude-sonnet-4"
    require_expert_review: true

  calculator_agent:
    enabled: true
    formula_path: "data/esrs_formulas.yaml"
    emission_factors_path: "data/emission_factors.json"
    zero_hallucination_mode: true

  aggregator_agent:
    enabled: true
    frameworks: ["TCFD", "GRI", "SASB"]
    mappings_path: "data/framework_mappings.json"

  reporting_agent:
    enabled: true
    output_formats: ["xbrl", "ixbrl", "esef"]
    generate_narratives: true

  audit_agent:
    enabled: true
    rules_path: "rules/esrs_compliance_rules.yaml"
    compliance_threshold: 90

# GreenLang Platform Agents
greenlang:
  development_quality:
    gl_codesentinel:
      enabled: true
      max_complexity: 10
      fail_on_error: true

    gl_secscan:
      enabled: true
      scan_dependencies: true
      fail_on_high: true

    gl_spec_guardian:
      enabled: true
      validate_schemas: true

  data_pipeline:
    gl_dataflow_guardian:
      enabled: true
      trace_lineage: true

    gl_determinism_auditor:
      enabled: true
      tolerance: 0.0

  release_readiness:
    gl_packqc:
      enabled: true
      validate_all: true

    gl_exitbar_auditor:
      enabled: true
      quality_threshold: 90
      security_threshold: 95
      performance_threshold: 85

    gl_hub_registrar:
      enabled: false  # Manual publish only

# CSRD Domain Agents
domain:
  regulatory_intelligence:
    enabled: true
    check_frequency: "daily"
    sources: ["EFRAG", "EU Commission", "ESMA"]
    auto_update_rules: false  # Require manual approval

  data_collection:
    enabled: true
    schedule: "monthly"
    systems:
      erp:
        enabled: true
        type: "sap"
      energy:
        enabled: true
      hr:
        enabled: true
      iot:
        enabled: true

  supply_chain:
    enabled: true
    auto_request_data: true
    deadline_days: 30

  automated_filing:
    enabled: true
    auto_submit: false  # Require manual approval
    countries: ["DE", "FR", "NL"]
```

---

## 4.3 Orchestration Patterns

### **Sequential Execution**

```python
# Example: Core pipeline sequential execution

async def run_sequential_pipeline(data):
    """Execute core pipeline agents sequentially"""
    # Stage 1: Intake
    intake_result = await intake_agent.process(data)

    # Stage 2: Calculation
    calc_result = await calculator_agent.calculate(intake_result)

    # Stage 3: Aggregation
    agg_result = await aggregator_agent.aggregate(calc_result)

    # Stage 4: Reporting
    report_result = await reporting_agent.generate(agg_result)

    # Stage 5: Audit
    audit_result = await audit_agent.validate(report_result)

    return audit_result
```

### **Parallel Execution**

```python
# Example: GreenLang quality gates in parallel

async def run_parallel_quality_gates(context):
    """Execute quality gate agents in parallel"""
    tasks = [
        gl_codesentinel.run(context),
        gl_secscan.run(context),
        gl_spec_guardian.run(context)
    ]

    results = await asyncio.gather(*tasks)

    # Aggregate results
    return {
        'codesentinel': results[0],
        'secscan': results[1],
        'spec_guardian': results[2]
    }
```

### **Event-Driven Execution**

```python
# Example: Trigger agents on events

class AgentEventBus:
    """Event bus for agent coordination"""

    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, agent):
        """Subscribe agent to event"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        self.subscribers[event_type].append(agent)

    async def publish(self, event_type, data):
        """Publish event to subscribers"""
        if event_type in self.subscribers:
            tasks = [
                agent.handle_event(data)
                for agent in self.subscribers[event_type]
            ]

            await asyncio.gather(*tasks)

# Usage
event_bus = AgentEventBus()

# Subscribe agents to events
event_bus.subscribe('data_collected', calculator_agent)
event_bus.subscribe('calculation_complete', audit_agent)
event_bus.subscribe('audit_complete', reporting_agent)

# Trigger workflow
await event_bus.publish('data_collected', esg_data)
```

---

*Document continues in COMPLETE_DEVELOPMENT_GUIDE_PART4.md...*
