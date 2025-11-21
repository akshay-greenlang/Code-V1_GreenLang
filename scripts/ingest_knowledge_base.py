# -*- coding: utf-8 -*-
"""
Knowledge Base Ingestion Script for GreenLang

Ingests curated knowledge documents into RAG system:
- GHG Protocol documentation
- Technology specifications
- Case studies
- Regulatory standards

Usage:
    python scripts/ingest_knowledge_base.py --collection ghg_protocol_corp --kb-dir knowledge_base/
    python scripts/ingest_knowledge_base.py --all  # Ingest all collections
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.intelligence.rag.engine import RAGEngine
from greenlang.intelligence.rag.config import RAGConfig
from greenlang.intelligence.rag.models import DocMeta
from greenlang.intelligence.rag.hashing import file_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseIngester:
    """Manages knowledge base ingestion into RAG system."""

    def __init__(self, config: RAGConfig):
        """
        Initialize ingester.

        Args:
            config: RAG configuration
        """
        self.config = config
        self.engine = RAGEngine(config=config)
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_errors": 0,
            "collections": {},
        }

    async def ingest_document(
        self,
        file_path: Path,
        collection: str,
        doc_meta: DocMeta,
    ) -> bool:
        """
        Ingest a single document.

        Args:
            file_path: Path to document
            collection: Collection name
            doc_meta: Document metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Ingesting: {doc_meta.title}")
            logger.info(f"  Collection: {collection}")
            logger.info(f"  File: {file_path}")

            manifest = await self.engine.ingest_document(
                file_path=file_path,
                collection=collection,
                doc_meta=doc_meta,
            )

            logger.info(f"  ✓ Ingested {manifest.total_embeddings} chunks")
            logger.info(f"  ✓ Duration: {manifest.ingestion_duration_seconds:.2f}s")

            # Update stats
            self.stats["total_documents"] += 1
            self.stats["total_chunks"] += manifest.total_embeddings

            if collection not in self.stats["collections"]:
                self.stats["collections"][collection] = {
                    "documents": 0,
                    "chunks": 0,
                }

            self.stats["collections"][collection]["documents"] += 1
            self.stats["collections"][collection]["chunks"] += manifest.total_embeddings

            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to ingest {file_path}: {e}")
            self.stats["total_errors"] += 1
            return False

    async def ingest_collection(
        self,
        collection: str,
        documents: List[Dict],
    ) -> None:
        """
        Ingest all documents in a collection.

        Args:
            collection: Collection name
            documents: List of document info dicts
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"INGESTING COLLECTION: {collection}")
        logger.info(f"{'='*60}")

        for doc_info in documents:
            await self.ingest_document(
                file_path=doc_info["path"],
                collection=collection,
                doc_meta=doc_info["meta"],
            )

    def print_stats(self):
        """Print ingestion statistics."""
        logger.info(f"\n{'='*60}")
        logger.info("INGESTION STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total documents: {self.stats['total_documents']}")
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"Total errors: {self.stats['total_errors']}")

        logger.info(f"\nBy Collection:")
        for collection, stats in self.stats["collections"].items():
            logger.info(f"  {collection}:")
            logger.info(f"    - Documents: {stats['documents']}")
            logger.info(f"    - Chunks: {stats['chunks']}")

    async def test_retrieval(self, queries: List[str]) -> None:
        """
        Test retrieval quality with sample queries.

        Args:
            queries: List of test queries
        """
        logger.info(f"\n{'='*60}")
        logger.info("TESTING RETRIEVAL QUALITY")
        logger.info(f"{'='*60}")

        for query in queries:
            logger.info(f"\nQuery: {query}")

            result = await self.engine.query(
                query=query,
                top_k=3,
                collections=self.config.allowlist,
            )

            logger.info(f"  Retrieved {len(result.chunks)} chunks ({result.search_time_ms}ms)")

            for i, (chunk, score) in enumerate(zip(result.chunks, result.relevance_scores)):
                logger.info(f"  [{i+1}] Score: {score:.3f}")
                logger.info(f"      Doc: {chunk.doc_id}")
                logger.info(f"      Text: {chunk.text[:150]}...")


def create_ghg_protocol_documents(kb_dir: Path) -> List[Dict]:
    """Create GHG Protocol knowledge base documents."""

    ghg_dir = kb_dir / "ghg_protocol"
    ghg_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: GHG Protocol Overview
    doc1_content = """
GHG Protocol Corporate Accounting and Reporting Standard

Overview:
The GHG Protocol Corporate Accounting and Reporting Standard provides requirements
and guidance for companies and other organizations preparing a corporate-level GHG
emissions inventory. It is the most widely used international accounting tool for
government and business leaders to understand, quantify, and manage greenhouse gas emissions.

Published by: World Resources Institute (WRI) and World Business Council for
Sustainable Development (WBCSD)

The GHG Protocol establishes comprehensive global standardized frameworks to measure
and manage greenhouse gas (GHG) emissions from private and public sector operations,
value chains and mitigation actions.

Key Principles:
1. Relevance: Ensure the GHG inventory appropriately reflects the GHG emissions
   of the company and serves the decision-making needs of users

2. Completeness: Account for and report on all GHG emission sources and activities
   within the chosen inventory boundary

3. Consistency: Use consistent methodologies to allow for meaningful comparisons
   of emissions over time

4. Transparency: Address all relevant issues in a factual and coherent manner,
   based on a clear audit trail

5. Accuracy: Ensure that the quantification of GHG emissions is systematically
   neither over nor under actual emissions, as far as can be judged, and that
   uncertainties are reduced as far as practicable
"""

    doc1_path = ghg_dir / "01_overview.txt"
    doc1_path.write_text(doc1_content, encoding="utf-8")

    # Document 2: Emission Scopes
    doc2_content = """
GHG Protocol Emission Scopes

The GHG Protocol divides organizational emissions into three scopes:

SCOPE 1: DIRECT GHG EMISSIONS
Direct GHG emissions occur from sources that are owned or controlled by the company,
for example, emissions from combustion in owned or controlled boilers, furnaces,
vehicles, emissions from chemical production in owned or controlled process equipment.

Examples of Scope 1 emissions:
- Fuel combustion in company-owned vehicles
- Fuel combustion in company-owned boilers, furnaces, and power generators
- Process emissions from physical or chemical processing (e.g., cement production,
  aluminum production, ammonia processing)
- Fugitive emissions from intentional or unintentional releases (e.g., refrigerant
  leaks from air conditioning systems, methane leaks from gas transport)

SCOPE 2: ELECTRICITY INDIRECT GHG EMISSIONS
Scope 2 accounts for GHG emissions from the generation of purchased electricity
consumed by the company. Purchased electricity is defined as electricity that is
purchased or otherwise brought into the organizational boundary of the company.

Scope 2 emissions physically occur at the facility where electricity is generated.
Companies have two methods for calculating Scope 2 emissions:
- Location-based method: Uses average emission factors for the grid
- Market-based method: Uses emissions from contractual instruments (e.g., RECs, PPAs)

SCOPE 3: OTHER INDIRECT GHG EMISSIONS (VALUE CHAIN)
Scope 3 is an optional reporting category that allows for the treatment of all
other indirect emissions. Scope 3 emissions are a consequence of the activities
of the company, but occur from sources not owned or controlled by the company.

Examples of Scope 3 emissions:
- Purchased goods and services (Category 1)
- Capital goods (Category 2)
- Fuel and energy-related activities not included in Scope 1 or 2 (Category 3)
- Upstream transportation and distribution (Category 4)
- Waste generated in operations (Category 5)
- Business travel (Category 6)
- Employee commuting (Category 7)
- Upstream leased assets (Category 8)
- Downstream transportation and distribution (Category 9)
- Processing of sold products (Category 10)
- Use of sold products (Category 11)
- End-of-life treatment of sold products (Category 12)
- Downstream leased assets (Category 13)
- Franchises (Category 14)
- Investments (Category 15)
"""

    doc2_path = ghg_dir / "02_scopes.txt"
    doc2_path.write_text(doc2_content, encoding="utf-8")

    # Document 3: Emission Factors
    doc3_content = """
GHG Protocol Emission Factors

Emission factors convert activity data into GHG emissions. Activity data represents
a measure of the level of activity that results in GHG emissions (e.g., kWh of
electricity consumed, liters of fuel consumed).

COMMON EMISSION FACTORS:

Stationary Combustion (Scope 1):
- Natural Gas: 0.0531 kg CO2e/kWh (or 53.06 kg CO2e/MMBtu)
- Diesel: 0.2647 kg CO2e/liter (or 10.21 kg CO2e/gallon)
- Gasoline: 0.2329 kg CO2e/liter (or 8.81 kg CO2e/gallon)
- Coal (bituminous): 0.341 kg CO2e/kWh (or 95.3 kg CO2e/MMBtu)
- Propane: 0.216 kg CO2e/liter (or 5.76 kg CO2e/gallon)
- Heating Oil: 0.2754 kg CO2e/liter (or 10.43 kg CO2e/gallon)

Electricity (Scope 2):
- US National Average: 0.417 kg CO2e/kWh
- US Northeast (NEWE): 0.237 kg CO2e/kWh
- US Midwest (MROE): 0.687 kg CO2e/kWh
- US California (CAMX): 0.205 kg CO2e/kWh
- US Texas (ERCT): 0.390 kg CO2e/kWh
- UK Grid: 0.233 kg CO2e/kWh
- Germany Grid: 0.420 kg CO2e/kWh
- China Grid: 0.581 kg CO2e/kWh

Refrigerants (Scope 1 - Fugitive):
- R-134a: Global Warming Potential (GWP) = 1,430
- R-410A: GWP = 2,088
- R-404A: GWP = 3,922
- R-22: GWP = 1,810

Transportation (Scope 1 or 3):
- Passenger Car (average): 0.192 kg CO2e/km (or 0.309 kg CO2e/mile)
- Light Commercial Vehicle: 0.258 kg CO2e/km (or 0.415 kg CO2e/mile)
- Heavy Goods Vehicle: 0.968 kg CO2e/km (or 1.558 kg CO2e/mile)
- Motorcycle: 0.113 kg CO2e/km (or 0.182 kg CO2e/mile)

Air Travel (Scope 3):
- Domestic flight (short-haul): 0.255 kg CO2e/passenger-km
- International flight (long-haul): 0.195 kg CO2e/passenger-km
- Economy class multiplier: 1.0
- Business class multiplier: 1.5
- First class multiplier: 2.0

Calculation Method:
GHG Emissions = Activity Data × Emission Factor

Example:
A company consumes 100,000 kWh of natural gas annually.
Emissions = 100,000 kWh × 0.0531 kg CO2e/kWh = 5,310 kg CO2e = 5.31 metric tons CO2e

Note: Emission factors vary by year, region, and methodology. Always use the most
recent factors from authoritative sources (EPA, DEFRA, IEA, etc.) and document
the source and vintage of factors used.
"""

    doc3_path = ghg_dir / "03_emission_factors.txt"
    doc3_path.write_text(doc3_content, encoding="utf-8")

    # Create document metadata
    documents = [
        {
            "path": doc1_path,
            "meta": DocMeta(
                doc_id="ghg_protocol_overview_v1",
                title="GHG Protocol Corporate Standard - Overview",
                collection="ghg_protocol_corp",
                publisher="WRI/WBCSD",
                content_hash=file_hash(str(doc1_path)),
                version="1.0",
                extra={"category": "methodology", "standard": "GHG Protocol"},
            ),
        },
        {
            "path": doc2_path,
            "meta": DocMeta(
                doc_id="ghg_protocol_scopes_v1",
                title="GHG Protocol Corporate Standard - Emission Scopes",
                collection="ghg_protocol_corp",
                publisher="WRI/WBCSD",
                content_hash=file_hash(str(doc2_path)),
                version="1.0",
                extra={"category": "methodology", "standard": "GHG Protocol"},
            ),
        },
        {
            "path": doc3_path,
            "meta": DocMeta(
                doc_id="ghg_protocol_factors_v1",
                title="GHG Protocol Emission Factors Reference",
                collection="ghg_protocol_corp",
                publisher="WRI/WBCSD",
                content_hash=file_hash(str(doc3_path)),
                version="1.0",
                extra={"category": "reference_data", "standard": "GHG Protocol"},
            ),
        },
    ]

    return documents


def create_technology_documents(kb_dir: Path) -> List[Dict]:
    """Create technology database documents."""

    tech_dir = kb_dir / "technologies"
    tech_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: Heat Pumps
    doc1_content = """
Industrial Heat Pumps for Decarbonization

Overview:
Industrial heat pumps are electric devices that transfer heat from a low-temperature
source to a higher-temperature sink, providing efficient heating for industrial
processes. They can achieve temperatures up to 160°C with modern high-temperature
heat pump technology.

Technology Types:
1. Air-source heat pumps: Use ambient air as heat source (COP: 2.5-3.5)
2. Water-source heat pumps: Use water bodies or wastewater (COP: 3.0-4.0)
3. Ground-source heat pumps: Use ground loops (COP: 3.5-4.5)
4. Waste heat recovery heat pumps: Utilize industrial waste heat (COP: 3.0-5.0)

Key Performance Metrics:
- Coefficient of Performance (COP): Ratio of heat output to electricity input
  * Low-temperature (<60°C): COP 3.5-4.5
  * Medium-temperature (60-90°C): COP 2.5-3.5
  * High-temperature (90-160°C): COP 2.0-3.0

- Temperature lift: Difference between source and sink temperatures
  * Higher lifts reduce COP
  * Optimal lifts: 30-50°C for best efficiency

Economic Considerations:
- Capital cost: $500-$1,500 per kW of heating capacity
- Operating cost: Depends on electricity price and COP
- Payback period: Typically 3-7 years
- Lifetime: 15-20 years with proper maintenance

Best Applications:
1. Food and beverage processing (pasteurization, cleaning, drying)
2. Chemical manufacturing (process heating, distillation)
3. Pharmaceutical production (sterilization, drying)
4. Textile manufacturing (dyeing, drying)
5. District heating networks

Prerequisites for Success:
- Consistent heat demand profile
- Available cold source (air, water, ground)
- Adequate electrical infrastructure
- Space for equipment installation
- Favorable electricity-to-gas price ratio

Emission Reduction Potential:
- Replacing natural gas boilers: 50-70% reduction in CO2 emissions
- Grid electricity carbon intensity: Critical factor
- With renewable electricity: Near-zero operational emissions
- Example: 100 kW heat pump (COP 3.0) replacing gas boiler
  * Gas boiler: 100 kW × 8760 h/yr × 0.05 kg CO2/kWh = 43.8 tons CO2/yr
  * Heat pump: 33.3 kW × 8760 h/yr × 0.417 kg CO2/kWh = 12.2 tons CO2/yr
  * Reduction: 31.6 tons CO2/yr (72%)
"""

    doc1_path = tech_dir / "01_heat_pumps.txt"
    doc1_path.write_text(doc1_content, encoding="utf-8")

    # Document 2: Solar Thermal
    doc2_content = """
Solar Thermal Systems for Industrial Process Heat

Overview:
Solar thermal systems convert sunlight into heat energy for industrial processes.
They can provide heat up to 400°C using concentrated solar power (CSP) technology,
making them suitable for many industrial applications.

Technology Types:
1. Flat-plate collectors: Up to 80°C, simple and cost-effective
2. Evacuated tube collectors: Up to 150°C, good for medium temperatures
3. Parabolic trough concentrators: Up to 400°C, for high-temperature processes
4. Linear Fresnel reflectors: Up to 300°C, lower cost than parabolic troughs

Performance Characteristics:
- Solar irradiance requirement: >1,800 kWh/m²/year for economic viability
- System efficiency: 30-70% depending on technology and temperature
- Capacity factor: 20-40% (varies with location and season)
- Thermal storage: Extends operating hours beyond daylight

Economic Analysis:
- Capital cost: $200-$600 per m² of collector area
- Levelized cost of heat: $30-$80 per MWh thermal
- Payback period: 5-10 years (depends on fuel prices and solar resource)
- Lifetime: 20-25 years

Best Applications:
- Food processing (washing, blanching, pasteurization)
- Textile industry (dyeing, washing, drying)
- Dairy industry (cleaning, pasteurization)
- Brewing industry (mashing, wort heating)
- Chemical industry (preheating, distillation)

Location Considerations:
- Optimal locations: High solar irradiance (>2,000 kWh/m²/year)
- Sun belt regions: Southern US, Mediterranean, Middle East, Australia
- Roof or ground mounting: Requires significant space
- Shading analysis: Must avoid obstructions

Thermal Storage Options:
- Sensible heat storage: Water tanks, rock beds
- Latent heat storage: Phase change materials (PCMs)
- Storage capacity: Typically 2-8 hours of full-load operation
- Benefits: Extends operating hours, smooths output variability

Environmental Benefits:
- Zero direct emissions during operation
- Renewable energy source
- Reduces fossil fuel dependency
- Long-term energy cost stability

Integration with Existing Systems:
- Hybrid systems: Solar + conventional boilers
- Preheating: Solar preheats feedwater for boilers
- Seasonal storage: Long-duration thermal storage for year-round operation
"""

    doc2_path = tech_dir / "02_solar_thermal.txt"
    doc2_path.write_text(doc2_content, encoding="utf-8")

    # Document 3: Combined Heat and Power (CHP)
    doc3_content = """
Combined Heat and Power (Cogeneration) Systems

Overview:
Combined Heat and Power (CHP), also known as cogeneration, is the simultaneous
production of electricity and useful heat from a single fuel source. CHP systems
achieve overall efficiencies of 65-85%, compared to 30-35% for conventional
electricity generation.

Technology Types:
1. Gas turbines: 5-250 MW, efficiency 65-80%
2. Reciprocating engines: 0.1-5 MW, efficiency 70-85%
3. Steam turbines: 0.5-250 MW, efficiency 60-75%
4. Microturbines: 30-300 kW, efficiency 65-75%
5. Fuel cells: 100 kW-2 MW, efficiency 70-85%

Operating Principles:
- Primary energy input: Natural gas, biogas, biomass, or waste fuels
- Electricity generation: Via turbine or engine
- Heat recovery: From exhaust gases and cooling systems
- Simultaneous production: Electricity and thermal energy

Performance Metrics:
- Electrical efficiency: 25-45%
- Thermal efficiency: 35-55%
- Overall efficiency: 65-85%
- Power-to-heat ratio: Varies by technology (0.5-2.0)

Economic Analysis:
- Capital cost: $1,000-$3,000 per kW electrical capacity
- Operating cost: Fuel, maintenance, emissions compliance
- Payback period: 4-8 years
- Lifetime: 15-25 years
- Incentives: May qualify for tax credits, grants

Ideal Applications:
- Manufacturing facilities with consistent heat and power loads
- Hospitals and healthcare facilities (24/7 operation)
- Universities and district energy systems
- Food processing plants
- Chemical manufacturing
- Data centers with heat recovery

Prerequisites for Economic Viability:
1. High and consistent thermal demand (>4,000 operating hours/year)
2. Favorable electricity-to-gas price ratio (spark spread)
3. Coincident heat and power demand
4. On-site space for equipment
5. Gas supply infrastructure

Emission Considerations:
- Lower total emissions than separate heat and power generation
- Displaced grid emissions: Depends on grid carbon intensity
- On-site emissions: From fuel combustion (Scope 1)
- Emission controls: NOx reduction, exhaust treatment
- Natural gas CHP vs coal grid: Can reduce emissions by 40-60%

Operational Modes:
1. Base-load operation: Run continuously at rated capacity
2. Load-following: Adjust output to match facility demand
3. Grid-connected: Export excess electricity to grid
4. Island mode: Operate independently during grid outages

Case Example:
A manufacturing facility with 1 MW electrical demand and 2 MW thermal demand:
- Gas engine CHP: 1 MW electrical, 2 MW thermal
- Fuel input: 3.5 MW (natural gas)
- Overall efficiency: 85%
- Annual operating hours: 8,000 hours
- Emission reduction vs separate generation: 3,500 tons CO2/year
- Simple payback: 5.2 years
"""

    doc3_path = tech_dir / "03_cogeneration_chp.txt"
    doc3_path.write_text(doc3_content, encoding="utf-8")

    documents = [
        {
            "path": doc1_path,
            "meta": DocMeta(
                doc_id="tech_heat_pumps_v1",
                title="Industrial Heat Pumps Technology Guide",
                collection="technology_database",
                publisher="GreenLang Research",
                content_hash=file_hash(str(doc1_path)),
                version="1.0",
                extra={"category": "electrification", "technology_type": "heat_pump"},
            ),
        },
        {
            "path": doc2_path,
            "meta": DocMeta(
                doc_id="tech_solar_thermal_v1",
                title="Solar Thermal Systems Technology Guide",
                collection="technology_database",
                publisher="GreenLang Research",
                content_hash=file_hash(str(doc2_path)),
                version="1.0",
                extra={"category": "renewable_energy", "technology_type": "solar"},
            ),
        },
        {
            "path": doc3_path,
            "meta": DocMeta(
                doc_id="tech_chp_v1",
                title="Combined Heat and Power (Cogeneration) Technology Guide",
                collection="technology_database",
                publisher="GreenLang Research",
                content_hash=file_hash(str(doc3_path)),
                version="1.0",
                extra={"category": "efficiency", "technology_type": "cogeneration"},
            ),
        },
    ]

    return documents


def create_case_study_documents(kb_dir: Path) -> List[Dict]:
    """Create case study documents."""

    case_dir = kb_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)

    doc_content = """
Industrial Decarbonization Case Studies

CASE STUDY 1: Food Processing Plant - Heat Pump Installation
Client: Regional dairy processing facility
Location: Wisconsin, USA
Annual Production: 50 million liters of milk products

Challenge:
- High natural gas consumption for pasteurization (150°C)
- Aging boiler infrastructure requiring replacement
- Corporate carbon reduction target: 50% by 2030

Solution Implemented:
- 300 kW high-temperature heat pump system
- Waste heat recovery from refrigeration systems
- Thermal storage tank for load shifting
- Hybrid system with existing gas boilers for peak demand

Technical Details:
- Heat pump COP: 3.2 average
- Operating temperature: 90°C (upgraded process to lower temperature requirement)
- Annual operating hours: 7,200 hours
- Cold source: Cooling water from refrigeration (10°C)

Results:
- Energy savings: 65% reduction in natural gas consumption
- CO2 emission reduction: 520 tons/year (45% of facility total)
- Annual cost savings: $85,000
- Capital investment: $420,000
- Simple payback: 4.9 years
- Additional benefits: Improved process control, reduced maintenance

Key Success Factors:
- Process modification to lower temperature requirement
- Available waste cold from refrigeration systems
- Supportive state energy efficiency incentive ($50,000 grant)
- Internal energy champion driving project

Lessons Learned:
- Process temperature optimization is critical for heat pump viability
- Thermal storage enables smaller equipment sizing
- Hybrid approach provides reliability during equipment maintenance

---

CASE STUDY 2: Chemical Manufacturing - Solar Thermal Installation
Client: Specialty chemicals manufacturer
Location: Arizona, USA
Annual Production: 10,000 tons of chemical products

Challenge:
- High process heat demand at 180°C for distillation
- Volatile natural gas prices impacting profitability
- Limited grid reliability requiring energy independence

Solution Implemented:
- 2,500 m² parabolic trough solar collectors
- 8-hour thermal storage using molten salt
- Integration with existing natural gas boilers (hybrid system)
- Automated control system for seamless transition

Technical Details:
- Solar resource: 2,400 kWh/m²/year (excellent location)
- Annual solar heat production: 3,800 MWh thermal
- System efficiency: 55%
- Solar fraction: 40% of total process heat demand

Results:
- Natural gas displacement: 380,000 m³/year
- CO2 emission reduction: 760 tons/year
- Annual cost savings: $120,000
- Capital investment: $1.2 million
- Simple payback: 10 years (7 years with incentives)
- Federal tax credit: 30% of project cost

Key Success Factors:
- Excellent solar resource (>2,000 kWh/m²/year)
- Consistent heat demand during daylight hours
- Available land for collector installation
- Thermal storage for process continuity

Challenges Overcome:
- Land preparation and grading costs higher than expected
- Specialized commissioning required for molten salt system
- Integration controls required custom development

---

CASE STUDY 3: Steel Rolling Mill - Waste Heat Recovery
Client: Steel service center with rolling mill
Location: Pennsylvania, USA
Annual Production: 200,000 tons of rolled steel products

Challenge:
- High-temperature exhaust gases (600°C) from reheating furnaces
- High electricity costs for facility operations
- Limited capital budget for decarbonization projects

Solution Implemented:
- Waste heat recovery with Organic Rankine Cycle (ORC) system
- 750 kW electrical generation capacity
- Heat exchanger network for exhaust gas cooling
- Grid-tied system for net metering

Technical Details:
- Waste heat source: Furnace exhaust at 600°C
- ORC working fluid: Siloxane (thermal stability)
- Electrical generation: 5,200 MWh/year
- System efficiency: 18% (thermal to electrical)
- Payback of waste heat: 85% of available thermal energy

Results:
- Electricity cost reduction: $520,000/year
- Demand charge reduction: $80,000/year (peak shaving)
- CO2 emission reduction: 2,170 tons/year (from displaced grid electricity)
- Capital investment: $2.1 million
- Simple payback: 3.5 years
- Additional benefit: Improved furnace efficiency due to better exhaust management

Key Success Factors:
- Consistent high-grade waste heat availability
- High electricity prices ($0.12/kWh average)
- State incentive for combined heat and power systems
- Experienced ORC equipment supplier

Technical Innovations:
- Custom heat exchanger design for particulate-laden exhaust
- Automated bypass system for furnace startup/shutdown
- Predictive maintenance system for ORC turbine

Return on Investment:
- Annual savings: $600,000
- 15-year project NPV (7% discount): $4.2 million
- Internal rate of return: 32%
- Carbon abatement cost: -$95/ton CO2 (saves money while reducing emissions)

---

CROSS-CUTTING LESSONS FROM ALL CASE STUDIES:

1. Process Integration is Key:
   - Understand existing thermal and electrical loads
   - Identify waste heat sources and cold sinks
   - Optimize process temperatures where possible

2. Economic Drivers:
   - Energy price differential (electricity vs gas) is critical
   - Incentives and tax credits significantly improve payback
   - Long-term energy price stability is valuable

3. Technical Requirements:
   - Adequate space for equipment installation
   - Reliable utility infrastructure (gas, electricity, water)
   - Internal technical expertise or strong O&M contract

4. Organizational Factors:
   - Executive sponsorship accelerates project approval
   - Cross-functional team (operations, engineering, finance) ensures success
   - Clear carbon reduction targets drive investment decisions

5. Risk Mitigation:
   - Hybrid systems provide reliability and flexibility
   - Performance guarantees from equipment suppliers
   - Phased implementation reduces upfront capital risk
"""

    doc_path = case_dir / "01_industrial_case_studies.txt"
    doc_path.write_text(doc_content, encoding="utf-8")

    documents = [
        {
            "path": doc_path,
            "meta": DocMeta(
                doc_id="case_studies_industrial_v1",
                title="Industrial Decarbonization Case Studies - Heat Pumps, Solar, WHR",
                collection="case_studies",
                publisher="GreenLang Research",
                content_hash=file_hash(str(doc_path)),
                version="1.0",
                extra={"category": "case_studies", "industries": "food,chemical,steel"},
            ),
        },
    ]

    return documents


async def main():
    """Main ingestion workflow."""
    parser = argparse.ArgumentParser(
        description="Ingest knowledge base documents into GreenLang RAG system"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Specific collection to ingest (ghg_protocol_corp, technology_database, case_studies)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all collections",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("knowledge_base"),
        help="Knowledge base directory (default: knowledge_base/)",
    )
    parser.add_argument(
        "--test-retrieval",
        action="store_true",
        help="Test retrieval quality after ingestion",
    )

    args = parser.parse_args()

    # Create knowledge base directory
    kb_dir = args.kb_dir
    kb_dir.mkdir(exist_ok=True)

    # Create RAG configuration
    config = RAGConfig(
        mode="live",
        allowlist=[
            "ghg_protocol_corp",
            "technology_database",
            "case_studies",
        ],
        embedding_provider="minilm",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_store_provider="faiss",
        vector_store_path=str(kb_dir / "vector_store"),
        retrieval_method="mmr",
        default_top_k=6,
        default_fetch_k=20,
        mmr_lambda=0.5,
        chunk_size=512,  # Larger chunks for knowledge docs
        chunk_overlap=64,
        enable_sanitization=True,
        verify_checksums=True,
    )

    # Create ingester
    ingester = KnowledgeBaseIngester(config=config)

    # Determine which collections to ingest
    collections_to_ingest = {}

    if args.all or args.collection == "ghg_protocol_corp" or args.collection is None:
        logger.info("Creating GHG Protocol documents...")
        collections_to_ingest["ghg_protocol_corp"] = create_ghg_protocol_documents(kb_dir)

    if args.all or args.collection == "technology_database" or args.collection is None:
        logger.info("Creating technology database documents...")
        collections_to_ingest["technology_database"] = create_technology_documents(kb_dir)

    if args.all or args.collection == "case_studies" or args.collection is None:
        logger.info("Creating case study documents...")
        collections_to_ingest["case_studies"] = create_case_study_documents(kb_dir)

    # Ingest all collections
    for collection, documents in collections_to_ingest.items():
        await ingester.ingest_collection(collection, documents)

    # Print statistics
    ingester.print_stats()

    # Test retrieval if requested
    if args.test_retrieval:
        test_queries = [
            "What are the emission factors for natural gas and coal?",
            "How do industrial heat pumps work and what is their COP?",
            "Show me case studies of waste heat recovery implementations",
            "What are the three scopes of GHG emissions?",
            "What is the payback period for solar thermal systems?",
        ]
        await ingester.test_retrieval(test_queries)


if __name__ == "__main__":
    asyncio.run(main())
