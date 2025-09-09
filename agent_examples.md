# How to Use the `gl agent` Command

## Syntax
```
greenlang agent [AGENT_ID]
```

## Available Agent IDs

First, list all agents to see available IDs:
```bash
gl agents
```

This will show you these agent IDs:
- `validator` - Input validation
- `fuel` - Fuel emissions calculator
- `carbon` - Carbon aggregator
- `report` - Report generator
- `benchmark` - Benchmarking
- `grid_factor` - Country-specific emission factors
- `building_profile` - Building categorization
- `intensity` - Intensity metrics
- `recommendation` - Optimization recommendations

## Usage Examples

### Example 1: View Fuel Agent Details
```bash
greenlang agent fuel
```
**Output:**
```
+-------------+
| Agent: fuel |
+-------------+
Class: FuelAgent
Description: Calculates emissions based on fuel consumption
Version: 0.0.1
Enabled: True
```

### Example 2: View Grid Factor Agent Details
```bash
greenlang agent grid_factor
```
**Output:**
```
+----------------------+
| Agent: grid_factor   |
+----------------------+
Class: GridFactorAgent
Description: Retrieves country-specific emission factors
Version: 0.0.1
Enabled: True
```

### Example 3: View Building Profile Agent Details
```bash
greenlang agent building_profile
```
**Output:**
```
+--------------------------+
| Agent: building_profile  |
+--------------------------+
Class: BuildingProfileAgent
Description: Categorizes buildings and provides expected performance
Version: 0.0.1
Enabled: True
```

### Example 4: View Intensity Agent Details
```bash
greenlang agent intensity
```
**Output:**
```
+------------------+
| Agent: intensity |
+------------------+
Class: IntensityAgent
Description: Calculates emission intensity metrics
Version: 0.0.1
Enabled: True
```

### Example 5: View Recommendation Agent Details
```bash
greenlang agent recommendation
```
**Output:**
```
+------------------------+
| Agent: recommendation  |
+------------------------+
Class: RecommendationAgent
Description: Provides optimization recommendations
Version: 0.0.1
Enabled: True
```

## All Agent Commands

Here are all the valid agent commands you can run:

```bash
# Core Agents (Original)
greenlang agent validator
greenlang agent fuel
greenlang agent carbon
greenlang agent report
greenlang agent benchmark

# Enhanced Agents (v0.0.1)
greenlang agent grid_factor
greenlang agent building_profile
greenlang agent intensity
greenlang agent recommendation
```

## What Information Does It Show?

For each agent, the command displays:
- **Agent ID**: The identifier used in workflows
- **Class Name**: The Python class implementing the agent
- **Description**: What the agent does
- **Version**: Agent version number
- **Enabled**: Whether the agent is active

## Common Mistakes

❌ **Wrong**: `agent fuel` (missing greenlang prefix)
❌ **Wrong**: `gl agent [fuel]` (don't use brackets)
❌ **Wrong**: `gl agent-fuel` (use space, not hyphen)
❌ **Wrong**: `gl agent "fuel"` (don't use quotes)

✅ **Correct**: `gl agent fuel`

## Get Help

To see help for the agent command:
```bash
greenlang agent --help
```

## Quick Test

Run these commands to test:
```bash
# List all agents
gl agents

# View specific agent
greenlang agent fuel
greenlang agent grid_factor
greenlang agent recommendation
```