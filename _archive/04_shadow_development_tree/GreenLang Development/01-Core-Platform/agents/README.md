# GreenLang Agent Templates

**Note:** The actual agent implementations are located in `greenlang/agents/`. This directory contains documentation and templates for creating new agents.

## Structure

```
agents/
├── README.md
├── templates/           # Agent templates for different use cases
│   ├── basic/          # Basic agent templates
│   ├── industry/       # Industry-specific agents
│   ├── advanced/       # Advanced/composite agents
│   └── custom/         # User-contributed agents
├── examples/           # Example implementations
└── tests/             # Agent template tests
```

## Available Agent Templates

### Core Agents (in `greenlang/agents/`)
- **FuelAgent**: Calculate emissions from fuel consumption
- **BoilerAgent**: Thermal systems and boiler operations
- **CarbonAgent**: Aggregate carbon footprint calculations
- **GridFactorAgent**: Regional emission factors
- **BenchmarkAgent**: Industry benchmark comparisons
- **IntensityAgent**: Emission intensity metrics
- **ReportAgent**: Generate reports in multiple formats

### Industry Templates (in `templates/industry/`)
- Power generation
- Cement production
- Steel manufacturing
- Dairy & F&B
- Chemical processing
- Logistics & transportation
- Commercial buildings
- Data centers

## Usage

Agent templates can be used in three ways:

1. **Direct Import** (for core agents):
```python
from greenlang.agents import FuelAgent
agent = FuelAgent()
```

2. **Template Loading**:
```python
from greenlang.sdk import load_agent_template
agent = load_agent_template("industry/cement_kiln")
```

3. **Custom Extension**:
```python
from greenlang.agents.base import BaseAgent
class MyCustomAgent(BaseAgent):
    # Your implementation
```

## Contributing

To add a new agent template:
1. Create your agent in the appropriate subdirectory
2. Include documentation and tests
3. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.