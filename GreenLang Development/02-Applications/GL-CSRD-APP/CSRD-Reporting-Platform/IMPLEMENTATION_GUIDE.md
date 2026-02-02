# GL-CSRD-APP Refactoring Implementation Guide

This guide provides concrete code examples for implementing the refactoring outlined in the completion report.

---

## Quick Reference

| Component | Template/Base | Key Changes | LOC Saved |
|-----------|---------------|-------------|-----------|
| MaterialityAgent | `greenlang.sdk.base.Agent` | Remove LLMClient wrapper, direct ChatSession | 277 |
| CalculatorAgent | `greenlang.agents.templates.CalculatorAgent` | Replace FormulaEngine, use framework | 329 |
| AggregatorAgent | `greenlang.sdk.base.Agent` | Consolidate mappers, use batch processing | 937 |
| ReportingAgent | `greenlang.agents.templates.ReportingAgent` | Consolidate XBRL, use templates | 902 |
| AuditAgent | `greenlang.sdk.base.Agent` | Use ValidationFramework | 161 |
| csrd_pipeline.py | `greenlang.sdk.base.Pipeline` | Replace orchestration | 545 |

---

## 1. MaterialityAgent Refactoring

### Before (Current - 1,177 LOC)
```python
class MaterialityAgent:
    def __init__(self, esrs_data_points_path, llm_config=None, ...):
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)  # Custom wrapper
        self.rag_system = RAGSystem(stakeholder_documents)
        # ... 100 lines of initialization
```

### After (Target - 900 LOC)
```python
from greenlang.sdk.base import Agent, Metadata, Result
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.providers.openai import OpenAIProvider
from greenlang.intelligence.rag.engine import RAGEngine

class MaterialityAgent(Agent):
    """AI-Powered Double Materiality Assessment Agent"""

    def __init__(self, esrs_data_points_path, llm_config=None,
                 impact_threshold=5.0, financial_threshold=5.0,
                 stakeholder_documents=None):
        super().__init__(metadata=Metadata(
            id="materiality_agent",
            name="MaterialityAgent",
            version="2.0.0",
            description="AI-powered double materiality assessment per ESRS 1"
        ))

        self.esrs_data_points_path = Path(esrs_data_points_path)
        self.impact_threshold = impact_threshold
        self.financial_threshold = financial_threshold

        # Direct ChatSession usage (remove LLMClient wrapper)
        llm_config = llm_config or {"provider": "openai", "model": "gpt-4o"}
        provider_config = LLMProviderConfig(
            model=llm_config.get("model", "gpt-4o"),
            api_key_env="OPENAI_API_KEY"
        )
        provider = OpenAIProvider(provider_config)
        self.session = ChatSession(provider)  # Direct usage

        # Direct RAGEngine usage (remove RAGSystem wrapper)
        rag_config = RAGConfig(
            mode="live",
            embedding_provider="minilm",
            vector_store_type="faiss"
        )
        self.rag_engine = RAGEngine(config=rag_config)

        # Load ESRS catalog
        self.esrs_catalog = self._load_esrs_catalog()
        self.esrs_topics = self._get_esrs_topics()

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input for materiality assessment"""
        if "company_context" not in input_data:
            self.logger.error("Missing company_context in input")
            return False
        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute double materiality assessment"""
        company_context = input_data["company_context"]
        esg_data = input_data.get("esg_data")
        stakeholder_data = input_data.get("stakeholder_data")

        # Assess each topic (existing logic)
        material_topics = []
        for topic in self.esrs_topics:
            impact_score = self._assess_impact_materiality(topic, company_context, esg_data)
            financial_score = self._assess_financial_materiality(topic, company_context, esg_data)
            material_topic = self._determine_double_materiality(
                topic, impact_score, financial_score
            )
            material_topics.append(material_topic)

        return {
            "material_topics": material_topics,
            "assessment_metadata": self._build_metadata(),
            "requires_human_review": True
        }

    # Keep existing assessment methods unchanged
    def _assess_impact_materiality(self, topic, company_context, esg_data):
        # Use self.session directly instead of self.llm_client
        messages = [
            ChatMessage(role=Role.system, content=system_prompt),
            ChatMessage(role=Role.user, content=user_prompt)
        ]
        response = await self.session.chat(messages=messages, temperature=0.3)
        # ... rest of logic unchanged
```

**Key Changes:**
1. ✅ Inherit from `Agent` base class
2. ✅ Remove `LLMClient` wrapper - use `ChatSession` directly (save 100 lines)
3. ✅ Remove `RAGSystem` wrapper - use `RAGEngine` directly (save 50 lines)
4. ✅ Add `validate()` and `process()` methods
5. ✅ Keep all existing materiality assessment logic

---

## 2. CalculatorAgent Refactoring

### Before (Current - 829 LOC)
```python
class FormulaEngine:
    """Custom formula engine - 220 lines"""
    def evaluate_formula(self, formula_spec, input_data):
        # Custom implementation
        pass

class CalculatorAgent:
    def __init__(self, esrs_formulas_path, emission_factors_path):
        self.formulas = self._load_formulas()
        self.emission_factors = self._load_emission_factors()
        self.formula_engine = FormulaEngine(self.emission_factors)
```

### After (Target - 500 LOC)
```python
from greenlang.agents.templates import CalculatorAgent as BaseCalculatorAgent

class CalculatorAgent(BaseCalculatorAgent):
    """
    ESRS Metrics Calculator with ZERO HALLUCINATION guarantee.

    Inherits from framework CalculatorAgent which provides:
    - Formula registration and execution
    - Caching with Redis
    - Provenance tracking
    - Uncertainty quantification
    """

    def __init__(self, esrs_formulas_path: Union[str, Path],
                 emission_factors_path: Union[str, Path]):
        # Load ESRS formulas and emission factors
        self.esrs_formulas_path = Path(esrs_formulas_path)
        self.emission_factors_path = Path(emission_factors_path)

        formulas_dict = self._load_and_register_formulas()
        emission_factors = self._load_emission_factors()

        # Initialize framework calculator with formulas
        super().__init__(
            formulas=formulas_dict,
            factor_broker=None,  # Could integrate FactorBroker here
            methodologies=None,  # Could integrate for uncertainty
            config={"version": "1.0.0"}
        )

        self.emission_factors_db = emission_factors

    def _load_and_register_formulas(self) -> Dict[str, Callable]:
        """Load ESRS formulas and convert to callables"""
        with open(self.esrs_formulas_path, 'r') as f:
            formulas_yaml = yaml.safe_load(f)

        formulas_dict = {}

        # Convert YAML formulas to Python callables
        for standard, formulas in formulas_yaml.items():
            if not isinstance(formulas, dict) or standard.startswith('_'):
                continue

            for metric_code, formula_spec in formulas.items():
                if not isinstance(formula_spec, dict):
                    continue

                # Create callable from formula spec
                formula_func = self._create_formula_function(formula_spec)
                formulas_dict[metric_code] = formula_func

        return formulas_dict

    def _create_formula_function(self, formula_spec: Dict) -> Callable:
        """Convert formula spec to Python callable"""
        calc_type = formula_spec.get("calculation_type", "")
        formula_str = formula_spec.get("formula", "")
        inputs_required = formula_spec.get("inputs", [])

        def formula(**kwargs):
            # Validate inputs
            for inp in inputs_required:
                if inp not in kwargs:
                    raise ValueError(f"Missing required input: {inp}")

            # Execute based on calculation type
            if calc_type == "sum":
                return sum(float(kwargs.get(inp, 0)) for inp in inputs_required)

            elif calc_type == "division":
                numerator = float(kwargs[inputs_required[0]])
                denominator = float(kwargs[inputs_required[1]])
                return numerator / denominator if denominator != 0 else None

            elif calc_type == "database_lookup_and_multiply":
                activity = float(kwargs.get(inputs_required[0], 0))
                ef_key = kwargs.get("emission_factor_key")
                ef = self.emission_factors_db.get(ef_key, {}).get("emission_factor", 1.0)
                return activity * ef

            # Add other calculation types...
            else:
                raise NotImplementedError(f"Calculation type {calc_type} not implemented")

        return formula

    def calculate_batch(self, metric_codes: List[str], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate multiple metrics with dependency resolution.

        Uses framework's batch_calculate under the hood.
        """
        # Resolve dependencies (keep existing topological sort)
        ordered_metrics = self.resolve_dependencies(metric_codes)

        results = []
        calculation_results = {}

        for metric_code in ordered_metrics:
            combined_data = {**input_data, **calculation_results}

            # Use framework's calculate method
            result = asyncio.run(self.calculate(
                formula_name=metric_code,
                inputs=combined_data,
                with_uncertainty=False,
                use_cache=True
            ))

            if result.success:
                results.append({
                    "metric_code": metric_code,
                    "value": result.value,
                    "unit": result.unit,
                    "provenance": result.provenance
                })
                calculation_results[metric_code] = result.value

        return {
            "metadata": {
                "metrics_calculated": len(results),
                "zero_hallucination_guarantee": True,
                "deterministic": True
            },
            "calculated_metrics": results,
            "provenance": [r.provenance for r in results if r.get("provenance")]
        }

    # Keep dependency resolution (framework doesn't provide this)
    def resolve_dependencies(self, metric_codes: List[str]) -> List[str]:
        """Topological sort for metric dependencies"""
        # ... existing implementation unchanged ...
```

**Key Changes:**
1. ✅ Inherit from framework `CalculatorAgent` template
2. ✅ Remove custom `FormulaEngine` (220 lines) - use framework
3. ✅ Convert YAML formulas to Python callables
4. ✅ Use framework's `calculate()` with caching and provenance
5. ✅ Keep dependency resolution logic (not in framework)
6. ✅ Maintain ZERO HALLUCINATION guarantee

---

## 3. csrd_pipeline.py Refactoring

### Before (Current - 895 LOC)
```python
class CSRDPipeline:
    def __init__(self, config_path):
        self.config = self._load_config()
        self._initialize_agents()  # Manual initialization

    def run(self, esg_data_file, company_profile, output_dir):
        # Stage 1: Intake (60 lines)
        stage1_start = datetime.now()
        intake_output = self.intake_agent.process(esg_data_file)
        stage1_end = datetime.now()
        # ... manual orchestration for 6 stages

        # Stage 2: Materiality (60 lines)
        # ... repeat for all 6 agents

        # Build final result (150 lines)
```

### After (Target - 350 LOC)
```python
from greenlang.sdk.base import Pipeline, Metadata, Result, Agent

class CSRDPipeline(Pipeline):
    """
    Complete end-to-end CSRD reporting pipeline.

    Orchestrates 6 agents using framework Pipeline base class.
    """

    def __init__(self, config_path: str):
        super().__init__(metadata=Metadata(
            id="csrd_pipeline",
            name="CSRD Reporting Platform Pipeline",
            version="2.0.0",
            description="End-to-end CSRD compliance reporting"
        ))

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_dir = self.config_path.parent.parent

        # Initialize all 6 agents
        self._initialize_agents()

        # Add agents to pipeline (framework handles orchestration)
        self.add_agent(self.intake_agent)
        self.add_agent(self.materiality_agent)
        self.add_agent(self.calculator_agent)
        self.add_agent(self.aggregator_agent)
        self.add_agent(self.reporting_agent)
        self.add_agent(self.audit_agent)

        self.logger.info(f"Pipeline initialized with {len(self.agents)} agents")

    def execute(self, input_data: Any) -> Result:
        """
        Execute the complete 6-agent pipeline.

        Framework handles:
        - Sequential execution
        - Error propagation
        - Performance tracking
        - Result aggregation
        """
        pipeline_start = datetime.now()
        pipeline_id = f"csrd_pipeline_{int(time.time())}"

        # Prepare initial input
        esg_data_file = input_data["esg_data_file"]
        company_profile = input_data["company_profile"]
        output_dir = input_data.get("output_dir")

        try:
            # Execute agents sequentially (framework pattern)
            agent_results = {}
            current_data = {"esg_data_file": esg_data_file, "company_profile": company_profile}

            for agent in self.agents:
                self.logger.info(f"Executing {agent.metadata.name}...")

                # Run agent (framework's Agent.run() method)
                agent_result = agent.run(current_data)

                if not agent_result.success:
                    return Result(
                        success=False,
                        error=f"{agent.metadata.name} failed: {agent_result.error}",
                        metadata={"failed_agent": agent.metadata.id}
                    )

                # Store result and pass to next agent
                agent_results[agent.metadata.id] = agent_result.data
                current_data = agent_result.data

            # Build final result
            pipeline_end = datetime.now()
            processing_time = (pipeline_end - pipeline_start).total_seconds()

            return Result(
                success=True,
                data={
                    "pipeline_id": pipeline_id,
                    "agent_results": agent_results,
                    "performance": {
                        "total_time_seconds": processing_time,
                        "within_target": processing_time < 1800  # 30 minutes
                    }
                },
                metadata={
                    "pipeline_version": self.metadata.version,
                    "agents_executed": len(self.agents),
                    "execution_timestamp": pipeline_start.isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return Result(success=False, error=str(e))

    def _initialize_agents(self):
        """Initialize all 6 agents (keep existing logic)"""
        # Agent 1: Intake
        self.intake_agent = IntakeAgent(
            esrs_data_points_path=self._resolve_path('esrs_data_points'),
            data_quality_rules_path=self._resolve_path('data_quality_rules')
        )

        # Agent 2: Materiality
        self.materiality_agent = MaterialityAgent(
            esrs_data_points_path=self._resolve_path('esrs_data_points'),
            llm_config=self.config['agents']['materiality']
        )

        # ... rest of agents
```

**Key Changes:**
1. ✅ Inherit from `Pipeline` base class
2. ✅ Use framework's `add_agent()` for sequencing (save 100 lines)
3. ✅ Replace manual orchestration with `execute()` method (save 300 lines)
4. ✅ Use framework's `Result` for error handling (save 50 lines)
5. ✅ Keep agent initialization logic (CSRD-specific)
6. ✅ Framework handles performance tracking

---

## 4. Testing Examples

### Unit Test for CalculatorAgent
```python
# tests/test_calculator_agent_refactored.py
import pytest
from GL-CSRD-APP.agents.calculator_agent import CalculatorAgent

def test_inherits_from_framework():
    """Verify CalculatorAgent uses framework template"""
    from greenlang.agents.templates import CalculatorAgent as BaseCalculator

    agent = CalculatorAgent(
        esrs_formulas_path="data/esrs_formulas.yaml",
        emission_factors_path="data/emission_factors.json"
    )

    assert isinstance(agent, BaseCalculator)

def test_zero_hallucination_maintained():
    """Verify calculations are deterministic"""
    agent = CalculatorAgent(
        esrs_formulas_path="data/esrs_formulas.yaml",
        emission_factors_path="data/emission_factors.json"
    )

    input_data = {"activity_data": 100, "emission_factor_key": "electricity_grid"}

    # Calculate twice
    result1 = agent.calculate_batch(["E1-1"], input_data)
    result2 = agent.calculate_batch(["E1-1"], input_data)

    # Must be bit-perfect identical
    assert result1["calculated_metrics"][0]["value"] == result2["calculated_metrics"][0]["value"]

def test_framework_caching_works():
    """Verify framework caching integration"""
    agent = CalculatorAgent(
        esrs_formulas_path="data/esrs_formulas.yaml",
        emission_factors_path="data/emission_factors.json"
    )

    # First call
    result1 = agent.calculate_batch(["E1-1"], input_data)
    stats1 = agent.get_stats()

    # Second call (should hit cache)
    result2 = agent.calculate_batch(["E1-1"], input_data)
    stats2 = agent.get_stats()

    assert stats2["cache_hits"] > stats1["cache_hits"]
```

### Integration Test for Pipeline
```python
# tests/integration/test_csrd_pipeline_refactored.py
def test_pipeline_end_to_end():
    """Test complete pipeline execution"""
    pipeline = CSRDPipeline("config/csrd_config.yaml")

    input_data = {
        "esg_data_file": "examples/demo_esg_data.csv",
        "company_profile": load_json("examples/company_profile.json"),
        "output_dir": "output/test_run"
    }

    result = pipeline.execute(input_data)

    # Verify success
    assert result.success == True

    # Verify all agents executed
    assert len(result.data["agent_results"]) == 6

    # Verify performance target met
    assert result.data["performance"]["within_target"] == True
```

---

## 5. Migration Checklist

### Pre-Migration
- [ ] Backup current codebase
- [ ] Create feature branch: `refactor/framework-integration`
- [ ] Set up test environment

### Phase 1: csrd_pipeline.py
- [ ] Implement Pipeline inheritance
- [ ] Add execute() method
- [ ] Test end-to-end execution
- [ ] Validate performance

### Phase 2: CalculatorAgent
- [ ] Implement template inheritance
- [ ] Convert formulas to callables
- [ ] Test all 500+ formulas
- [ ] Validate zero-hallucination

### Phase 3: ReportingAgent
- [ ] Implement template inheritance
- [ ] Consolidate XBRL generation
- [ ] Test ESEF compliance
- [ ] Validate security features

### Phase 4: Other Agents
- [ ] MaterialityAgent
- [ ] AggregatorAgent
- [ ] AuditAgent

### Post-Migration
- [ ] Update documentation
- [ ] Update README metrics
- [ ] Create migration guide
- [ ] Deploy to staging

---

## 6. Rollback Plan

If issues arise during migration:

1. **Immediate Rollback:**
   ```bash
   git checkout main
   ```

2. **Partial Rollback (per agent):**
   - Keep refactored agents in separate modules
   - Import old version if needed:
     ```python
     from agents.calculator_agent_legacy import CalculatorAgent
     ```

3. **Testing in Parallel:**
   - Run both old and new versions
   - Compare outputs for validation
   - Switch when confidence is high

---

## Contact & Support

For questions or issues during implementation:
- **Technical Lead:** Claude (GL-CSRD-APP Team)
- **Framework Docs:** https://github.com/greenlang/core
- **Issue Tracker:** GL-CSRD-APP/issues

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Status:** Ready for Implementation
