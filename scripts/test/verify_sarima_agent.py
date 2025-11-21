# -*- coding: utf-8 -*-
"""Quick verification script for SARIMA Forecast Agent.

This script performs a simple smoke test to verify the agent is working correctly.

Run:
    python verify_sarima_agent.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from greenlang.agents.forecast_agent_sarima import SARIMAForecastAgent
    print("[OK] Successfully imported SARIMAForecastAgent")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)


def run_smoke_test():
    """Run a basic smoke test."""
    print("\n" + "="*60)
    print("SARIMA Forecast Agent - Smoke Test")
    print("="*60 + "\n")

    # 1. Create agent
    print("1. Creating agent...")
    try:
        agent = SARIMAForecastAgent(
            budget_usd=0.5,
            enable_explanations=False,  # Disable to avoid API calls
            enable_recommendations=False,
        )
        print("   [OK] Agent created successfully")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 2. Generate test data
    print("\n2. Generating test data...")
    try:
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=36, freq='M')
        t = np.arange(36)
        values = 100 + 2*t + 10*np.sin(2*np.pi*t/12) + np.random.normal(0, 3, 36)

        df = pd.DataFrame({'value': values}, index=dates)
        print(f"   [OK] Generated {len(df)} data points")
        print(f"   [OK] Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 3. Validate input
    print("\n3. Validating input...")
    try:
        input_data = {
            "data": df,
            "target_column": "value",
            "forecast_horizon": 6,
            "seasonal_period": 12,
        }

        is_valid = agent.validate(input_data)
        if is_valid:
            print("   [OK] Input validation passed")
        else:
            print("   [FAIL] Input validation failed")
            return False
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 4. Test preprocessing
    print("\n4. Testing preprocessing tool...")
    try:
        result = agent._preprocess_data_impl(input_data)
        print(f"   [OK] Preprocessing successful")
        print(f"   - Missing values filled: {result['missing_values_filled']}")
        print(f"   - Outliers detected: {result['outliers_detected']}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 5. Test seasonality detection
    print("\n5. Testing seasonality detection...")
    try:
        result = agent._detect_seasonality_impl(input_data)
        print(f"   [OK] Seasonality detection successful")
        print(f"   - Seasonal period: {result['seasonal_period']}")
        print(f"   - Has seasonality: {result['has_seasonality']}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 6. Test stationarity validation
    print("\n6. Testing stationarity validation...")
    try:
        result = agent._validate_stationarity_impl(input_data)
        print(f"   [OK] Stationarity test successful")
        print(f"   - Is stationary: {result['is_stationary']}")
        print(f"   - P-value: {result['p_value']:.4f}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 7. Test model fitting
    print("\n7. Testing SARIMA model fitting...")
    try:
        result = agent._fit_sarima_impl(
            input_data,
            auto_tune=False,  # Fast mode
            seasonal_period=12,
        )
        print(f"   [OK] Model fitting successful")
        print(f"   - Order: {result['order']}")
        print(f"   - Seasonal order: {result['seasonal_order']}")
        print(f"   - AIC: {result['aic']:.2f}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 8. Test forecasting
    print("\n8. Testing forecast generation...")
    try:
        result = agent._forecast_future_impl(input_data, horizon=6)
        print(f"   [OK] Forecast generation successful")
        print(f"   - Forecast length: {len(result['forecast'])}")
        print(f"   - First forecast: {result['forecast'][0]:.2f}")
        print(f"   - Confidence level: {result['confidence_level']}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 9. Test evaluation
    print("\n9. Testing model evaluation...")
    try:
        result = agent._evaluate_model_impl(input_data)
        print(f"   [OK] Model evaluation successful")
        print(f"   - RMSE: {result['rmse']:.2f}")
        print(f"   - MAE: {result['mae']:.2f}")
        print(f"   - MAPE: {result['mape']:.2f}%")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    # 10. Performance summary
    print("\n10. Performance summary...")
    try:
        perf = agent.get_performance_summary()
        print(f"   [OK] Performance tracking working")
        print(f"   - Tool calls: {perf['ai_metrics']['tool_call_count']}")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
        return False

    return True


def main():
    """Main entry point."""
    success = run_smoke_test()

    print("\n" + "="*60)
    if success:
        print("[SUCCESS] ALL SMOKE TESTS PASSED")
        print("\nThe SARIMA Forecast Agent is working correctly!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest tests/agents/test_forecast_agent_sarima.py -v")
        print("  2. Try interactive demo: python examples/forecast_sarima_demo.py")
        print("  3. Read documentation: docs/FORECAST_AGENT_SARIMA_IMPLEMENTATION.md")
    else:
        print("[FAILED] SMOKE TESTS FAILED")
        print("\nPlease check the error messages above.")
        sys.exit(1)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
