# -*- coding: utf-8 -*-
from greenlang.agents import FuelAgent, CarbonAgent, GridFactorAgent

# Test each agent
agents_tests = [
    (FuelAgent(), {'fuel_type': 'electricity', 'consumption': 100, 'unit': 'kWh'}),
    (CarbonAgent(), {'emissions': []}),
    (GridFactorAgent(), {'country': 'US', 'fuel_type': 'electricity', 'unit': 'kWh'})
]

for agent, test_input in agents_tests:
    result = agent.run(test_input)
    print(f"{agent.__class__.__name__}: type={type(result)}, is_dict={isinstance(result, dict)}")
    if hasattr(result, 'success'):
        print(f"  Has .success attribute")
    elif isinstance(result, dict) and 'success' in result:
        print(f"  Has 'success' key in dict")