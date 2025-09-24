"""
GreenLang AI Assistant with RAG (Retrieval-Augmented Generation)
Provides evidence-based answers using vector search and LLM
"""

import re
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
from dotenv import load_dotenv

# Core imports
from greenlang.core.orchestrator import Orchestrator
from greenlang.agents import (
    FuelAgent,
    CarbonAgent,
    InputValidatorAgent,
    ReportAgent,
    BenchmarkAgent,
    BoilerAgent,
    GridFactorAgent,
    BuildingProfileAgent,
    IntensityAgent,
    RecommendationAgent,
)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Try importing RAG components
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    # Create a dummy Document class for type hints when langchain is not available
    if TYPE_CHECKING:
        from langchain.schema import Document
    else:

        class Document:
            pass

    print(
        "Warning: RAG components not available. Install with: pip install langchain langchain-community faiss-cpu sentence-transformers"
    )

# Try importing OpenAI
try:
    from openai import OpenAI
    from langchain_community.chat_models import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAssistant:
    """Enhanced AI Assistant with RAG capabilities for evidence-based climate intelligence"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        vector_store_path: str = "knowledge_base/vector_store",
        use_chroma: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize RAG-enhanced assistant

        Args:
            api_key: OpenAI API key
            vector_store_path: Path to vector store
            use_chroma: Use ChromaDB instead of FAISS
            embedding_model: Embedding model name
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.vector_store_path = Path(vector_store_path)
        self.use_chroma = use_chroma
        self.embedding_model_name = embedding_model

        # Initialize components
        self.orchestrator = self._setup_orchestrator()
        self.vector_store = None
        self.embeddings = None
        self.qa_chain = None
        self.llm = None

        # Setup OpenAI
        self.llm_available = False
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.llm_available = True
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Could not initialize OpenAI client: {e}")

        # Setup RAG if available
        self.rag_available = False
        if RAG_AVAILABLE:
            self._setup_rag()

    def _setup_orchestrator(self) -> Orchestrator:
        """Setup orchestrator with all agents"""
        orchestrator = Orchestrator()
        orchestrator.register_agent("validator", InputValidatorAgent())
        orchestrator.register_agent("fuel", FuelAgent())
        orchestrator.register_agent("boiler", BoilerAgent())
        orchestrator.register_agent("carbon", CarbonAgent())
        orchestrator.register_agent("report", ReportAgent())
        orchestrator.register_agent("benchmark", BenchmarkAgent())
        orchestrator.register_agent("grid_factor", GridFactorAgent())
        orchestrator.register_agent("building_profile", BuildingProfileAgent())
        orchestrator.register_agent("intensity", IntensityAgent())
        orchestrator.register_agent("recommendation", RecommendationAgent())
        return orchestrator

    def _setup_rag(self) -> None:
        """Setup RAG components"""
        try:
            # Initialize embeddings
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Load vector store
            if self._load_vector_store():
                self.rag_available = True
                logger.info("RAG components initialized successfully")

                # Setup QA chain if LLM is available
                if self.llm_available:
                    self._setup_qa_chain()
            else:
                logger.warning("Could not load vector store")

        except Exception as e:
            logger.error(f"Error setting up RAG: {e}")
            self.rag_available = False

    def _load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            if self.use_chroma:
                persist_dir = str(self.vector_store_path / "chroma")
                if Path(persist_dir).exists():
                    self.vector_store = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=self.embeddings,
                    )
                    logger.info("ChromaDB vector store loaded")
                    return True
            else:
                faiss_path = str(self.vector_store_path / "faiss")
                if Path(faiss_path).exists():
                    self.vector_store = FAISS.load_local(
                        faiss_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info("FAISS vector store loaded")
                    return True

            logger.warning("No existing vector store found")
            return False

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

    def _setup_qa_chain(self) -> None:
        """Setup question-answering chain"""
        try:
            # Create LangChain LLM wrapper
            self.llm = ChatOpenAI(
                temperature=0.7, model_name="gpt-4", openai_api_key=self.api_key
            )

            # Create custom prompt
            prompt_template = """You are GreenLang AI Assistant, an expert on carbon emissions and climate intelligence.
            
Use the following context to answer the question. If the context doesn't contain enough information, 
use your knowledge to provide a comprehensive answer, but indicate what is from the context vs general knowledge.

Context from knowledge base:
{context}

Question: {question}

Provide a detailed, accurate answer with specific emission factors, methodologies, and calculations where applicable.
If the question asks for calculations, show your work step-by-step.

Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # Create QA chain
            self.qa_chain = load_qa_chain(
                llm=self.llm, chain_type="stuff", prompt=PROMPT, verbose=False
            )

            logger.info("QA chain initialized successfully")

        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")

    def search_knowledge_base(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the knowledge base for relevant documents

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if not self.rag_available or not self.vector_store:
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

    def get_rag_response(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """
        Get RAG-enhanced response with sources

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            Tuple of (answer, list of sources)
        """
        if not self.rag_available:
            return "RAG not available. Please run build_vector_store.py first.", []

        try:
            # Search for relevant documents
            docs = self.search_knowledge_base(query, k=k)

            if not docs:
                return "No relevant documents found in knowledge base.", []

            # Extract sources
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

            # If we have QA chain, use it
            if self.qa_chain and self.llm_available:
                answer = self.qa_chain.run(input_documents=docs, question=query)
                return answer, sources

            # Otherwise, return formatted context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"[{i}] {doc.page_content[:500]}...")

            answer = "Based on the knowledge base:\n\n" + "\n\n".join(context_parts)
            return answer, sources

        except Exception as e:
            logger.error(f"Error getting RAG response: {e}")
            return f"Error processing query: {str(e)}", []

    def process_query(
        self, query: str, verbose: bool = False, use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Process user query with RAG enhancement

        Args:
            query: User query
            verbose: Show detailed output
            use_rag: Use RAG for response

        Returns:
            Response dictionary
        """
        # Check if this is a calculation request
        if self._is_calculation_request(query):
            # For calculations, extract data and execute workflow
            if self.llm_available:
                return self._process_calculation_with_llm(query, verbose)
            else:
                return self._process_with_rules(query, verbose)
        else:
            # For general questions, use RAG if available
            if use_rag and self.rag_available:
                answer, sources = self.get_rag_response(query)

                response_parts = [answer]

                if sources:
                    response_parts.append("\n\nSources:")
                    for source in sources:
                        response_parts.append(f"  - {source}")

                return {
                    "success": True,
                    "response": "\n".join(response_parts),
                    "sources": sources,
                    "method": "RAG",
                }
            elif self.llm_available:
                # Use standard LLM without RAG
                return self._answer_general_question(query)
            else:
                # Fallback to knowledge base
                return self._answer_with_knowledge_base(query)

    def _is_calculation_request(self, query: str) -> bool:
        """Check if the query is asking for a calculation"""
        calculation_keywords = [
            "calculate",
            "compute",
            "what is the carbon footprint",
            "emissions for",
            "how much co2",
            "carbon emissions",
            "kwh",
            "therms",
            "gallons",
            "liters",
            "consumption",
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in calculation_keywords)

    def _answer_general_question(self, query: str) -> Dict[str, Any]:
        """Answer general questions using OpenAI (with RAG context if available)"""
        try:
            # Get RAG context if available
            context = ""
            sources = []
            if self.rag_available:
                docs = self.search_knowledge_base(query, k=3)
                if docs:
                    context_parts = []
                    for doc in docs:
                        context_parts.append(doc.page_content)
                        source = doc.metadata.get("source", "Unknown")
                        if source not in sources:
                            sources.append(source)
                    context = "\n\n".join(context_parts)

            system_prompt = """You are GreenLang AI Assistant, an expert on carbon emissions, climate change, and environmental sustainability.
            
You have access to emission factors and can answer questions about:
- Emission factors for different fuels and regions
- Carbon footprint calculations and methodologies
- Climate change and environmental impact
- Best practices for reducing emissions
- Industry benchmarks and standards"""

            if context:
                system_prompt += f"\n\nRelevant context from knowledge base:\n{context}"

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            answer = response.choices[0].message.content

            result = {
                "success": True,
                "response": answer,
                "method": "LLM with RAG" if context else "LLM",
            }

            if sources:
                result["sources"] = sources

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing with OpenAI: {str(e)}",
                "response": self._answer_with_knowledge_base(query)["response"],
            }

    def _answer_with_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Answer questions using built-in knowledge base when OpenAI is not available"""
        query_lower = query.lower()

        # Try RAG first if available
        if self.rag_available:
            docs = self.search_knowledge_base(query, k=3)
            if docs:
                response_parts = ["Based on the knowledge base:"]
                sources = []
                for doc in docs:
                    response_parts.append(f"\n{doc.page_content[:300]}...")
                    source = doc.metadata.get("source", "Unknown")
                    if source not in sources:
                        sources.append(source)

                if sources:
                    response_parts.append("\n\nSources:")
                    for source in sources:
                        response_parts.append(f"  - {source}")

                return {
                    "success": True,
                    "response": "\n".join(response_parts),
                    "sources": sources,
                    "method": "RAG fallback",
                }

        # Fallback to static knowledge base
        knowledge_base = {
            "electricity": {
                "us": "0.385 kg CO2e/kWh (US average)",
                "india": "0.82 kg CO2e/kWh (India grid average)",
                "eu": "0.295 kg CO2e/kWh (EU average)",
                "china": "0.555 kg CO2e/kWh (China grid average)",
                "uk": "0.233 kg CO2e/kWh (UK grid average)",
                "default": "0.385 kg CO2e/kWh (varies by region)",
            },
            "natural_gas": "5.3 kg CO2e/therm or 53.06 kg CO2e/MMBtu",
            "diesel": "10.21 kg CO2e/gallon",
            "gasoline": "8.887 kg CO2e/gallon",
            "propane": "5.76 kg CO2e/gallon",
            "coal": "2.86 kg CO2e/kg (anthracite)",
            "jet_fuel": "9.75 kg CO2e/gallon",
        }

        response_parts = []

        # Check for emission factor questions
        if "emission factor" in query_lower or "factor" in query_lower:
            response_parts.append("Common emission factors:")
            response_parts.append(
                f"  - Electricity: {knowledge_base['electricity']['default']}"
            )
            response_parts.append(f"  - Natural Gas: {knowledge_base['natural_gas']}")
            response_parts.append(f"  - Diesel: {knowledge_base['diesel']}")
            response_parts.append(f"  - Gasoline: {knowledge_base['gasoline']}")
        else:
            response_parts.append("I can help you with:")
            response_parts.append("  - Emission factors for various fuels")
            response_parts.append("  - Carbon footprint calculations")
            response_parts.append("  - Industry benchmarks")
            response_parts.append("  - Tips for reducing emissions")

        return {
            "success": True,
            "response": "\n".join(response_parts),
            "method": "Static knowledge base",
        }

    def _process_calculation_with_llm(
        self, query: str, verbose: bool
    ) -> Dict[str, Any]:
        """Process calculation requests with LLM"""
        try:
            prompt = self._create_llm_prompt(query)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant for GreenLang. Extract structured data from user queries about carbon emissions calculations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            extracted_data = self._parse_llm_response(
                response.choices[0].message.content
            )

            if not extracted_data or not extracted_data.get("fuels"):
                # If no calculation data found, treat as general question
                return self._answer_general_question(query)

            return self._execute_workflow(extracted_data, verbose)

        except Exception as e:
            return {
                "success": False,
                "error": f"LLM processing failed: {str(e)}",
                "response": self._process_with_rules(query, verbose)["response"],
            }

    def _process_with_rules(self, query: str, verbose: bool) -> Dict[str, Any]:
        """Process query with rule-based extraction"""
        try:
            extracted_data = self._extract_data_from_query(query)

            if not extracted_data.get("fuels"):
                return {
                    "success": False,
                    "error": "Could not extract fuel consumption data from query",
                    "response": "Please provide fuel consumption data. Example: 'Calculate emissions for 1000 kWh electricity in India'",
                }

            result = self._execute_workflow(extracted_data, verbose)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing request: {str(e)}",
            }

    def _extract_data_from_query(self, query: str) -> Dict[str, Any]:
        """Extract data from query using regex patterns"""
        query_lower = query.lower()
        data = {"fuels": [], "building_info": {}, "country": None}

        # Extract country/region
        countries = [
            ("united states", "US"),
            ("united kingdom", "UK"),
            ("india", "IN"),
            ("china", "CN"),
            ("europe", "EU"),
            ("japan", "JP"),
            ("brazil", "BR"),
            ("germany", "DE"),
            ("canada", "CA"),
            ("australia", "AU"),
        ]

        for country_name, country_code in countries:
            if country_name in query_lower:
                data["country"] = country_code
                break

        # Extract electricity
        electricity_pattern = r"(\d+(?:\.\d+)?)\s*(kwh|mwh|gwh)"
        electricity_match = re.search(electricity_pattern, query_lower)
        if electricity_match:
            value = float(electricity_match.group(1))
            unit = electricity_match.group(2)
            unit_map = {"kwh": "kWh", "mwh": "MWh", "gwh": "GWh"}
            unit = unit_map.get(unit, unit)
            data["fuels"].append(
                {"fuel_type": "electricity", "consumption": value, "unit": unit}
            )

        # Extract natural gas
        gas_patterns = [
            (r"(\d+(?:\.\d+)?)\s*therms?\s*(?:of\s*)?(?:natural\s*)?gas", "therms"),
            (r"(\d+(?:\.\d+)?)\s*ccf\s*(?:of\s*)?(?:natural\s*)?gas", "ccf"),
            (
                r"(\d+(?:\.\d+)?)\s*(?:cubic\s*meters?|m3)\s*(?:of\s*)?(?:natural\s*)?gas",
                "m3",
            ),
        ]

        for pattern, unit in gas_patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))
                data["fuels"].append(
                    {"fuel_type": "natural_gas", "consumption": value, "unit": unit}
                )
                break

        return data

    def _execute_workflow(self, data: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """Execute calculation workflow"""
        try:
            from greenlang.sdk import GreenLangClient

            country = data.get("country", "US")
            # Ensure country is not None
            if country is None:
                country = "US"
            client = GreenLangClient(region=country)

            emissions_list = []
            for fuel in data.get("fuels", []):
                calc_result = client.calculate_emissions(
                    fuel["fuel_type"], fuel["consumption"], fuel["unit"]
                )
                if calc_result["success"]:
                    emissions_list.append(calc_result["data"])

            if not emissions_list:
                return {
                    "success": False,
                    "response": "No emissions could be calculated",
                }

            # Aggregate emissions
            agg_result = client.aggregate_emissions(emissions_list)

            # Format response
            response_parts = []
            if agg_result["success"]:
                carbon_data = agg_result["data"]
                total_emissions = carbon_data.get("total_co2e_tons", 0)

                response_parts.append(f"Carbon Footprint Analysis")
                if country != "US":
                    response_parts.append(f"Country/Region: {country}")
                response_parts.append(f"{'=' * 40}")
                response_parts.append(
                    f"Total Emissions: {total_emissions:.3f} metric tons CO2e"
                )
                response_parts.append(
                    f"Total Emissions: {carbon_data.get('total_co2e_kg', 0):.2f} kg CO2e"
                )

                if "emissions_breakdown" in carbon_data:
                    response_parts.append("\nEmissions Breakdown:")
                    for item in carbon_data["emissions_breakdown"]:
                        response_parts.append(
                            f"  - {item['source']}: {item['co2e_tons']:.3f} tons ({item['percentage']}%)"
                        )

            response = (
                "\n".join(response_parts)
                if response_parts
                else "No emissions data calculated"
            )

            return {"success": True, "response": response, "method": "Calculation"}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing request: {str(e)}",
            }

    def _create_llm_prompt(self, query: str) -> str:
        """Create prompt for LLM extraction"""
        return f"""Extract carbon emission data from this query: "{query}"

Return a JSON object with:
- fuels: array of {{fuel_type, consumption, unit}}
- building_info: {{type, area, area_unit, occupancy}}
- country: country code (US, IN, CN, EU, UK, etc.)

Fuel types: electricity, natural_gas, diesel, gasoline, propane
Electricity units: kWh, MWh, GWh
Natural gas units: therms, ccf, m3
Liquid fuel units: gallons, liters

If the query is not about calculating emissions, return an empty JSON object.

Example:
{{
  "fuels": [
    {{"fuel_type": "electricity", "consumption": 1000, "unit": "kWh"}},
    {{"fuel_type": "natural_gas", "consumption": 500, "unit": "therms"}}
  ],
  "building_info": {{
    "type": "commercial_office",
    "area": 10000,
    "area_unit": "sqft",
    "occupancy": 50
  }},
  "country": "US"
}}"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON"""
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
            return {}


# Keep original AIAssistant for backward compatibility
AIAssistant = RAGAssistant
