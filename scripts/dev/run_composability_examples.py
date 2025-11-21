# -*- coding: utf-8 -*-
"""
Simple runner for composability examples to demonstrate the framework.
This can be executed directly without pytest.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run examples
try:
    from examples.composability_examples import main
    import asyncio

    print("="*80)
    print("Running GreenLang Composability Framework Examples")
    print("="*80)

    # Run the examples
    asyncio.run(main())

    print("\n" + "="*80)
    print("SUCCESS: All examples completed successfully!")
    print("="*80)

except ImportError as e:
    print(f"Import Error: {e}")
    print("\nNote: The composability framework requires the following:")
    print("1. Python 3.8 or higher with asyncio support")
    print("2. Pydantic for data validation")
    print("3. NumPy for calculations (optional)")
    print("\nInstall dependencies with:")
    print("pip install pydantic numpy")

except Exception as e:
    print(f"Error running examples: {e}")
    import traceback
    traceback.print_exc()