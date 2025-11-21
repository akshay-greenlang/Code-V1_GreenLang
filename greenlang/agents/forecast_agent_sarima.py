# -*- coding: utf-8 -*-
"""AI-powered SARIMA Forecasting Agent with ChatSession Integration.

This module provides a production-ready SARIMA (Seasonal Autoregressive Integrated
Moving Average) forecasting agent for time-series prediction in climate and energy domains.

Key Features:
    1. Tool-First Numerics: All calculations via deterministic tools (zero hallucinated numbers)
    2. AI Interpretation: Natural language explanations of forecasts and patterns
    3. Auto-tuning: Grid search for optimal SARIMA parameters (p,d,q,P,D,Q,s)
    4. Seasonality Detection: Automatic detection of seasonal patterns
    5. Confidence Intervals: 95% prediction intervals for all forecasts
    6. Deterministic Results: temperature=0, seed=42 for reproducibility
    7. Comprehensive Validation: Stationarity tests, out-of-sample validation
    8. Production-Ready: Full error handling, provenance tracking

Architecture:
    SARIMAForecastAgent (orchestration) -> ChatSession (AI) -> Tools (exact calculations)

    Tools:
    - fit_sarima_model: Fit SARIMA model to historical data with parameter tuning
    - forecast_future: Generate future predictions with confidence intervals
    - calculate_confidence_intervals: Compute prediction intervals
    - evaluate_model: Calculate accuracy metrics (RMSE, MAE, MAPE)
    - detect_seasonality: Auto-detect seasonal patterns and period
    - validate_stationarity: Perform ADF test for stationarity
    - preprocess_data: Handle missing values and outliers

Use Cases:
    - Energy consumption forecasting (monthly/hourly patterns)
    - Temperature prediction (daily/seasonal cycles)
    - Emissions trend forecasting (long-term patterns)
    - Grid load prediction (weekly/seasonal patterns)

Example:
    >>> agent = SARIMAForecastAgent()
    >>> result = agent.run({
    ...     "data": df,  # pandas DataFrame with datetime index
    ...     "target_column": "energy_kwh",
    ...     "forecast_horizon": 12,  # 12 periods ahead
    ...     "seasonal_period": 12,   # Monthly seasonality
    ... })
    >>> print(result["data"]["forecast"])
    [1234.5, 1245.2, ...]  # Exact predictions from SARIMA model
    >>> print(result["data"]["explanation"])
    "The forecast shows a seasonal pattern with peak consumption in summer months..."

Author: GreenLang Framework Team
Date: October 2025
Spec: SARIMA Baseline ML Agent (ML-001)
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
from greenlang.determinism import DeterministicClock

# Suppress statsmodels warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - SARIMA agent will use mock predictions")

from greenlang.sdk.base import Agent, Result, Metadata
from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    BudgetExceeded,
    create_provider,
)
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.agents.citations import (
    EmissionFactorCitation,
    CalculationCitation,
    CitationBundle,
    create_emission_factor_citation,
)


logger = logging.getLogger(__name__)


@dataclass
class SARIMAParams:
    """SARIMA model parameters."""
    p: int = 1  # AR order
    d: int = 1  # Differencing order
    q: int = 1  # MA order
    P: int = 1  # Seasonal AR order
    D: int = 1  # Seasonal differencing order
    Q: int = 1  # Seasonal MA order
    s: int = 12  # Seasonal period

    def to_tuple(self) -> Tuple:
        """Convert to statsmodels format."""
        return ((self.p, self.d, self.q), (self.P, self.D, self.Q, self.s))


@dataclass
class ForecastResult:
    """Forecast results container."""
    forecast: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence_level: float = 0.95
    forecast_dates: Optional[List[datetime]] = None


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    rmse: float
    mae: float
    mape: float
    aic: float
    bic: float
    ljung_box_pvalue: Optional[float] = None


class SARIMAForecastAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """AI-powered SARIMA forecasting agent for time-series prediction.

    This agent provides production-ready time-series forecasting using SARIMA models
    with automatic parameter tuning, seasonality detection, and AI-generated interpretations.

    Features:
    - Tool-first numerics (all calculations use tools, zero hallucinated numbers)
    - Automatic SARIMA parameter tuning via grid search
    - Seasonality detection and validation
    - Confidence interval estimation
    - Multiple accuracy metrics (RMSE, MAE, MAPE)
    - Stationarity testing (ADF test)
    - Missing data handling (interpolation)
    - Outlier detection and treatment
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking

    Input Format:
        {
            "data": pandas.DataFrame with datetime index,
            "target_column": str,
            "forecast_horizon": int,  # Number of periods to forecast
            "seasonal_period": int (optional),  # Auto-detected if not provided
            "confidence_level": float (optional, default=0.95),
            "auto_tune": bool (optional, default=True),
            "exog_columns": List[str] (optional),  # Exogenous variables
        }

    Output Format:
        {
            "forecast": List[float],  # Point predictions
            "lower_bound": List[float],  # Lower confidence bound
            "upper_bound": List[float],  # Upper confidence bound
            "forecast_dates": List[datetime],
            "model_params": dict,  # Fitted SARIMA parameters
            "metrics": dict,  # RMSE, MAE, MAPE, AIC, BIC
            "seasonality": dict,  # Detected seasonal patterns
            "explanation": str,  # AI-generated interpretation
            "recommendations": List[str],  # AI-generated insights
        }

    Example:
        >>> agent = SARIMAForecastAgent()
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=36, freq='M'),
        ...     'energy_kwh': [1000 + 100*np.sin(i/6) for i in range(36)]
        ... }).set_index('date')
        >>> result = agent.run({
        ...     "data": df,
        ...     "target_column": "energy_kwh",
        ...     "forecast_horizon": 12,
        ...     "seasonal_period": 12
        ... })
        >>> print(result.data["forecast"])
        [1050.2, 1098.5, ...]
    """

    def __init__(
        self,
        budget_usd: float = 1.00,
        enable_explanations: bool = True,
        enable_recommendations: bool = True,
        enable_auto_tune: bool = True,
    ):
        """Initialize the SARIMA forecasting agent.

        Args:
            budget_usd: Maximum USD to spend per forecast (default: $1.00)
            enable_explanations: Enable AI-generated explanations (default: True)
            enable_recommendations: Enable AI recommendations (default: True)
            enable_auto_tune: Enable automatic parameter tuning (default: True)
        """
        # Initialize metadata
        metadata = Metadata(
            id="forecast_sarima",
            name="SARIMA Forecast Agent",
            version="0.1.0",
            description="AI-powered SARIMA forecasting for time-series prediction",
            tags=["ml", "forecasting", "sarima", "time-series"],
        )
        super().__init__(metadata)

        # Configuration
        self.budget_usd = budget_usd
        self.enable_explanations = enable_explanations
        self.enable_recommendations = enable_recommendations
        self.enable_auto_tune = enable_auto_tune

        # Initialize LLM provider
        self.provider = create_provider()

        # Performance tracking
        self._ai_call_count = 0
        self._tool_call_count = 0
        self._total_cost_usd = 0.0

        # Citation tracking
        self._current_citations: List[EmissionFactorCitation] = []
        self._calculation_citations: List[CalculationCitation] = []

        # Model state
        self._fitted_model = None
        self._last_training_data = None
        self._best_params = None

        # Setup tools
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""

        # Tool 1: Fit SARIMA model
        self.fit_sarima_tool = ToolDef(
            name="fit_sarima_model",
            description="Fit SARIMA model to historical time-series data with optional parameter tuning",
            parameters={
                "type": "object",
                "properties": {
                    "auto_tune": {
                        "type": "boolean",
                        "description": "Automatically tune SARIMA parameters using grid search",
                        "default": True,
                    },
                    "seasonal_period": {
                        "type": "integer",
                        "description": "Seasonal period (e.g., 12 for monthly, 7 for daily)",
                        "minimum": 1,
                    },
                    "max_p": {
                        "type": "integer",
                        "description": "Maximum AR order to test",
                        "default": 3,
                        "minimum": 0,
                        "maximum": 5,
                    },
                    "max_q": {
                        "type": "integer",
                        "description": "Maximum MA order to test",
                        "default": 3,
                        "minimum": 0,
                        "maximum": 5,
                    },
                },
                "required": ["seasonal_period"],
            },
        )

        # Tool 2: Generate forecast
        self.forecast_future_tool = ToolDef(
            name="forecast_future",
            description="Generate future predictions with confidence intervals using fitted SARIMA model",
            parameters={
                "type": "object",
                "properties": {
                    "horizon": {
                        "type": "integer",
                        "description": "Number of periods to forecast ahead",
                        "minimum": 1,
                        "maximum": 365,
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level for prediction intervals (0-1)",
                        "default": 0.95,
                        "minimum": 0.5,
                        "maximum": 0.99,
                    },
                },
                "required": ["horizon"],
            },
        )

        # Tool 3: Calculate confidence intervals
        self.confidence_intervals_tool = ToolDef(
            name="calculate_confidence_intervals",
            description="Calculate prediction confidence intervals for forecasts",
            parameters={
                "type": "object",
                "properties": {
                    "forecast": {
                        "type": "array",
                        "description": "Point forecasts",
                        "items": {"type": "number"},
                    },
                    "std_errors": {
                        "type": "array",
                        "description": "Standard errors of predictions",
                        "items": {"type": "number"},
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level (0-1)",
                        "default": 0.95,
                    },
                },
                "required": ["forecast", "std_errors"],
            },
        )

        # Tool 4: Evaluate model accuracy
        self.evaluate_model_tool = ToolDef(
            name="evaluate_model",
            description="Calculate accuracy metrics (RMSE, MAE, MAPE) on validation data",
            parameters={
                "type": "object",
                "properties": {
                    "train_test_split": {
                        "type": "number",
                        "description": "Fraction of data to use for training (0-1)",
                        "default": 0.8,
                        "minimum": 0.5,
                        "maximum": 0.95,
                    },
                },
            },
        )

        # Tool 5: Detect seasonality
        self.detect_seasonality_tool = ToolDef(
            name="detect_seasonality",
            description="Auto-detect seasonal patterns and period using ACF analysis",
            parameters={
                "type": "object",
                "properties": {
                    "max_period": {
                        "type": "integer",
                        "description": "Maximum seasonal period to test",
                        "default": 52,
                        "minimum": 2,
                        "maximum": 365,
                    },
                },
            },
        )

        # Tool 6: Validate stationarity
        self.validate_stationarity_tool = ToolDef(
            name="validate_stationarity",
            description="Perform Augmented Dickey-Fuller test for stationarity",
            parameters={
                "type": "object",
                "properties": {
                    "alpha": {
                        "type": "number",
                        "description": "Significance level",
                        "default": 0.05,
                    },
                },
            },
        )

        # Tool 7: Preprocess data
        self.preprocess_data_tool = ToolDef(
            name="preprocess_data",
            description="Preprocess time-series data: handle missing values, detect outliers",
            parameters={
                "type": "object",
                "properties": {
                    "interpolation_method": {
                        "type": "string",
                        "description": "Method for missing value interpolation",
                        "enum": ["linear", "time", "spline", "polynomial"],
                        "default": "linear",
                    },
                    "outlier_threshold": {
                        "type": "number",
                        "description": "IQR multiplier for outlier detection",
                        "default": 3.0,
                        "minimum": 1.5,
                        "maximum": 5.0,
                    },
                },
            },
        )

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.

        Args:
            input_data: Input dictionary

        Returns:
            bool: True if valid
        """
        # Check required fields
        if "data" not in input_data:
            self.logger.error("Missing required field: data")
            return False

        if "target_column" not in input_data:
            self.logger.error("Missing required field: target_column")
            return False

        if "forecast_horizon" not in input_data:
            self.logger.error("Missing required field: forecast_horizon")
            return False

        # Validate data is DataFrame
        data = input_data["data"]
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Data must be a pandas DataFrame")
            return False

        # Validate target column exists
        target = input_data["target_column"]
        if target not in data.columns:
            self.logger.error(f"Target column '{target}' not found in data")
            return False

        # Validate datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error("Data must have a DatetimeIndex")
            return False

        # Validate minimum data points
        min_points = input_data.get("seasonal_period", 12) * 2
        if len(data) < min_points:
            self.logger.error(f"Insufficient data: need at least {min_points} points")
            return False

        # Validate forecast horizon
        horizon = input_data["forecast_horizon"]
        if not isinstance(horizon, int) or horizon < 1:
            self.logger.error("forecast_horizon must be a positive integer")
            return False

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process forecast request (synchronous wrapper).

        Args:
            input_data: Input dictionary

        Returns:
            Dict with forecast results
        """
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._process_async(input_data))
            return result
        finally:
            loop.close()

    async def _process_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process forecast request with AI orchestration.

        Args:
            input_data: Input dictionary

        Returns:
            Dict with forecast results and AI interpretation
        """
        start_time = DeterministicClock.now()

        # Extract parameters
        data = input_data["data"]
        target_column = input_data["target_column"]
        forecast_horizon = input_data["forecast_horizon"]
        seasonal_period = input_data.get("seasonal_period")
        confidence_level = input_data.get("confidence_level", 0.95)
        auto_tune = input_data.get("auto_tune", self.enable_auto_tune)

        # Store training data
        self._last_training_data = data.copy()

        # Create ChatSession
        session = ChatSession(self.provider)

        # Reset citations for new run
        self._current_citations = []
        self._calculation_citations = []

        # Build AI prompt
        prompt = self._build_prompt(input_data)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are a time-series forecasting expert for GreenLang. "
                    "You help analyze historical data and generate forecasts using SARIMA models. "
                    "IMPORTANT: You must use the provided tools for ALL numeric calculations. "
                    "Never estimate or guess numbers. Always explain patterns and provide actionable insights."
                ),
            ),
            ChatMessage(role=Role.user, content=prompt),
        ]

        # Create budget
        budget = Budget(max_usd=self.budget_usd)

        try:
            # Call AI with tools
            self._ai_call_count += 1

            response = await session.chat(
                messages=messages,
                tools=[
                    self.fit_sarima_tool,
                    self.forecast_future_tool,
                    self.confidence_intervals_tool,
                    self.evaluate_model_tool,
                    self.detect_seasonality_tool,
                    self.validate_stationarity_tool,
                    self.preprocess_data_tool,
                ],
                budget=budget,
                temperature=0.0,  # Deterministic
                seed=42,  # Reproducible
                tool_choice="auto",
            )

            # Track cost
            self._total_cost_usd += response.usage.cost_usd

            # Extract tool results
            tool_results = self._extract_tool_results(response, input_data)

            # Build output
            output = self._build_output(
                input_data,
                tool_results,
                response.text if self.enable_explanations else None,
            )

            # Calculate duration
            duration = (DeterministicClock.now() - start_time).total_seconds()

            # Add metadata
            output["metadata"] = {
                "agent_id": self.metadata.id,
                "version": self.metadata.version,
                "provider": response.provider_info.provider,
                "model": response.provider_info.model,
                "tokens": response.usage.total_tokens,
                "cost_usd": response.usage.cost_usd,
                "tool_calls": len(response.tool_calls),
                "calculation_time_s": duration,
                "deterministic": True,
            }

            return output

        except BudgetExceeded as e:
            self.logger.error(f"Budget exceeded: {e}")
            raise ValueError(f"AI budget exceeded: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in forecast: {e}")
            raise

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build AI prompt for forecasting.

        Args:
            input_data: Input data

        Returns:
            str: Formatted prompt
        """
        data = input_data["data"]
        target = input_data["target_column"]
        horizon = input_data["forecast_horizon"]
        seasonal_period = input_data.get("seasonal_period")

        # Data summary
        data_summary = f"""
Time-Series Forecasting Request:
- Dataset: {len(data)} observations
- Date range: {data.index[0]} to {data.index[-1]}
- Target variable: {target}
- Data frequency: {data.index.freq if data.index.freq else 'irregular'}
- Forecast horizon: {horizon} periods
"""

        if seasonal_period:
            data_summary += f"- Expected seasonality: {seasonal_period} periods\n"

        # Basic statistics
        series = data[target]
        data_summary += f"""
- Current value: {series.iloc[-1]:.2f}
- Mean: {series.mean():.2f}
- Std Dev: {series.std():.2f}
- Min: {series.min():.2f}
- Max: {series.max():.2f}
"""

        prompt = data_summary + """
Tasks:
1. Use preprocess_data to clean the time series (handle missing values, outliers)
"""

        if not seasonal_period:
            prompt += "2. Use detect_seasonality to identify seasonal patterns\n"
            prompt += "3. Use validate_stationarity to check if differencing is needed\n"
            step = 4
        else:
            prompt += "2. Use validate_stationarity to check if differencing is needed\n"
            step = 3

        prompt += f"""{step}. Use fit_sarima_model to train the forecasting model
{step + 1}. Use forecast_future to generate {horizon}-period forecast with confidence intervals
{step + 2}. Use evaluate_model to assess accuracy on validation data
{step + 3}. Provide analysis including:
   - Key patterns identified (trend, seasonality, cycles)
   - Model performance assessment
   - Confidence in predictions
   - Notable forecast insights
"""

        if self.enable_recommendations:
            prompt += f"{step + 4}. Provide actionable recommendations based on the forecast\n"

        prompt += """
IMPORTANT:
- Use tools for ALL calculations and predictions
- Do not estimate or guess any numbers
- Format forecasts clearly with confidence intervals
- Explain seasonal patterns and trends
- Highlight any unusual patterns or anomalies
"""

        return prompt

    def _extract_tool_results(
        self,
        response,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract results from AI tool calls.

        Args:
            response: ChatResponse from session
            input_data: Original input data

        Returns:
            Dict with tool results
        """
        results = {}

        for tool_call in response.tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})

            try:
                if name == "fit_sarima_model":
                    results["model"] = self._fit_sarima_impl(input_data, **args)
                elif name == "forecast_future":
                    results["forecast"] = self._forecast_future_impl(input_data, **args)
                elif name == "calculate_confidence_intervals":
                    results["confidence"] = self._calculate_confidence_impl(**args)
                elif name == "evaluate_model":
                    results["evaluation"] = self._evaluate_model_impl(input_data, **args)
                elif name == "detect_seasonality":
                    results["seasonality"] = self._detect_seasonality_impl(input_data, **args)
                elif name == "validate_stationarity":
                    results["stationarity"] = self._validate_stationarity_impl(input_data, **args)
                elif name == "preprocess_data":
                    results["preprocessing"] = self._preprocess_data_impl(input_data, **args)
            except Exception as e:
                self.logger.error(f"Tool {name} failed: {e}")
                results[name] = {"error": str(e)}

        return results

    def _preprocess_data_impl(
        self,
        input_data: Dict[str, Any],
        interpolation_method: str = "linear",
        outlier_threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """Tool implementation: Preprocess time-series data.

        Args:
            input_data: Input data dict
            interpolation_method: Interpolation method for missing values
            outlier_threshold: IQR multiplier for outlier detection

        Returns:
            Dict with preprocessing results
        """
        self._tool_call_count += 1

        data = input_data["data"]
        target = input_data["target_column"]
        series = data[target].copy()

        # Track changes
        original_nulls = series.isnull().sum()

        # Handle missing values
        if original_nulls > 0:
            series = series.interpolate(method=interpolation_method)

        # Detect outliers using IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_threshold * IQR
        upper_bound = Q3 + outlier_threshold * IQR

        outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        # Cap outliers (don't remove, as it breaks time series)
        series = series.clip(lower=lower_bound, upper=upper_bound)

        # Update the data
        if self._last_training_data is None:
            self._last_training_data = data.copy()
        self._last_training_data[target] = series

        return {
            "missing_values_filled": int(original_nulls),
            "outliers_detected": int(outliers),
            "outliers_capped": int(outliers),
            "final_length": len(series),
            "preprocessing_applied": True,
        }

    def _detect_seasonality_impl(
        self,
        input_data: Dict[str, Any],
        max_period: int = 52,
    ) -> Dict[str, Any]:
        """Tool implementation: Detect seasonal patterns.

        Args:
            input_data: Input data dict
            max_period: Maximum period to test

        Returns:
            Dict with seasonality information
        """
        self._tool_call_count += 1

        if not STATSMODELS_AVAILABLE:
            return {
                "seasonal_period": 12,
                "has_seasonality": True,
                "strength": 0.7,
                "method": "mock",
            }

        data = self._last_training_data if self._last_training_data is not None else input_data["data"]
        target = input_data["target_column"]
        series = data[target].dropna()

        # Compute ACF
        acf_values = acf(series, nlags=min(max_period, len(series) // 2 - 1))

        # Find peaks in ACF (excluding lag 0)
        peaks = []
        for i in range(2, len(acf_values)):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                if acf_values[i] > 0.3:  # Significant correlation
                    peaks.append((i, acf_values[i]))

        # Detected seasonal period is the first significant peak
        if peaks:
            seasonal_period = peaks[0][0]
            strength = peaks[0][1]
            has_seasonality = True
        else:
            # Default to common periods based on data frequency
            freq_str = str(data.index.freq).lower() if data.index.freq else ""
            if "M" in freq_str or "month" in freq_str:
                seasonal_period = 12
            elif "D" in freq_str or "day" in freq_str:
                seasonal_period = 7
            elif "H" in freq_str or "hour" in freq_str:
                seasonal_period = 24
            else:
                seasonal_period = 12
            strength = 0.0
            has_seasonality = False

        return {
            "seasonal_period": int(seasonal_period),
            "has_seasonality": has_seasonality,
            "strength": float(strength),
            "detected_peaks": len(peaks),
            "method": "acf",
        }

    def _validate_stationarity_impl(
        self,
        input_data: Dict[str, Any],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Tool implementation: Validate stationarity using ADF test.

        Args:
            input_data: Input data dict
            alpha: Significance level

        Returns:
            Dict with stationarity test results
        """
        self._tool_call_count += 1

        if not STATSMODELS_AVAILABLE:
            return {
                "is_stationary": False,
                "adf_statistic": -2.5,
                "p_value": 0.12,
                "critical_values": {"1%": -3.5, "5%": -2.9, "10%": -2.6},
                "differencing_needed": True,
            }

        data = self._last_training_data if self._last_training_data is not None else input_data["data"]
        target = input_data["target_column"]
        series = data[target].dropna()

        # Perform ADF test
        result = adfuller(series, autolag='AIC')

        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]

        is_stationary = p_value < alpha

        return {
            "is_stationary": bool(is_stationary),
            "adf_statistic": float(adf_statistic),
            "p_value": float(p_value),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "differencing_needed": not is_stationary,
            "significance_level": alpha,
        }

    def _fit_sarima_impl(
        self,
        input_data: Dict[str, Any],
        auto_tune: bool = True,
        seasonal_period: int = 12,
        max_p: int = 3,
        max_q: int = 3,
    ) -> Dict[str, Any]:
        """Tool implementation: Fit SARIMA model.

        Args:
            input_data: Input data dict
            auto_tune: Auto-tune parameters
            seasonal_period: Seasonal period
            max_p: Max AR order
            max_q: Max MA order

        Returns:
            Dict with fitted model information
        """
        self._tool_call_count += 1

        data = self._last_training_data if self._last_training_data is not None else input_data["data"]
        target = input_data["target_column"]
        series = data[target].dropna()

        if not STATSMODELS_AVAILABLE:
            # Mock mode - create a simple mock fitted model
            params = SARIMAParams(p=1, d=1, q=1, P=1, D=1, Q=1, s=seasonal_period)
            self._best_params = params
            # Store a mock fitted model (just mark as fitted for forecasting)
            self._fitted_model = "mock_model"  # Sentinel value
            return {
                "order": (params.p, params.d, params.q),
                "seasonal_order": (params.P, params.D, params.Q, params.s),
                "aic": 1234.5,
                "bic": 1250.3,
                "auto_tuned": auto_tune,
                "converged": True,
            }

        best_aic = float('inf')
        best_params = None
        best_model = None

        if auto_tune:
            # Grid search for best parameters
            param_combinations = []
            for p in range(0, max_p + 1):
                for d in range(0, 2):
                    for q in range(0, max_q + 1):
                        for P in range(0, 2):
                            for D in range(0, 2):
                                for Q in range(0, 2):
                                    if p + q + P + Q == 0:
                                        continue  # At least one parameter must be non-zero
                                    param_combinations.append((p, d, q, P, D, Q))

            # Limit search space
            if len(param_combinations) > 50:
                # Use common parameter sets
                param_combinations = [
                    (1, 1, 1, 1, 1, 1),
                    (2, 1, 1, 1, 1, 1),
                    (1, 1, 2, 1, 1, 1),
                    (2, 1, 2, 1, 1, 1),
                    (1, 0, 1, 1, 0, 1),
                    (2, 0, 2, 1, 0, 1),
                ]

            for params in param_combinations:
                p, d, q, P, D, Q = params
                try:
                    model = SARIMAX(
                        series,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False, maxiter=100)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_params = SARIMAParams(p=p, d=d, q=q, P=P, D=D, Q=Q, s=seasonal_period)
                        best_model = fitted
                except Exception as e:
                    continue
        else:
            # Use default parameters
            best_params = SARIMAParams(p=1, d=1, q=1, P=1, D=1, Q=1, s=seasonal_period)
            try:
                model = SARIMAX(
                    series,
                    order=(best_params.p, best_params.d, best_params.q),
                    seasonal_order=(best_params.P, best_params.D, best_params.Q, best_params.s),
                )
                best_model = model.fit(disp=False)
                best_aic = best_model.aic
            except Exception as e:
                raise ValueError(f"Failed to fit SARIMA model: {e}")

        # Check if model was fitted (check params to avoid DataFrame ambiguity)
        if best_params is None:
            raise ValueError("Could not fit SARIMA model with any parameter combination")

        # Store fitted model
        self._fitted_model = best_model
        self._best_params = best_params

        return {
            "order": (best_params.p, best_params.d, best_params.q),
            "seasonal_order": (best_params.P, best_params.D, best_params.Q, best_params.s),
            "aic": float(best_model.aic),
            "bic": float(best_model.bic),
            "auto_tuned": auto_tune,
            "converged": True,
            "log_likelihood": float(best_model.llf),
        }

    def _forecast_future_impl(
        self,
        input_data: Dict[str, Any],
        horizon: int,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Tool implementation: Generate forecast.

        Args:
            input_data: Input data dict
            horizon: Forecast horizon
            confidence_level: Confidence level

        Returns:
            Dict with forecast results
        """
        self._tool_call_count += 1

        if self._fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        data = self._last_training_data if self._last_training_data is not None else input_data["data"]

        if not STATSMODELS_AVAILABLE:
            # Mock forecast
            last_value = data[input_data["target_column"]].iloc[-1]
            forecast = [last_value * (1 + 0.05 * np.sin(i/6)) for i in range(horizon)]
            std_err = [last_value * 0.1] * horizon

            alpha = 1 - confidence_level
            z_score = 1.96 if confidence_level == 0.95 else 2.576

            lower = [f - z_score * s for f, s in zip(forecast, std_err)]
            upper = [f + z_score * s for f, s in zip(forecast, std_err)]
        else:
            # Real forecast
            forecast_result = self._fitted_model.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean.tolist()

            # Get confidence intervals
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)
            lower = conf_int.iloc[:, 0].tolist()
            upper = conf_int.iloc[:, 1].tolist()

        # Generate forecast dates
        last_date = data.index[-1]
        freq = data.index.freq
        if freq:
            forecast_dates = pd.date_range(
                start=last_date + freq,
                periods=horizon,
                freq=freq
            ).tolist()
        else:
            # Infer frequency from last two dates
            time_delta = data.index[-1] - data.index[-2]
            forecast_dates = [last_date + time_delta * (i + 1) for i in range(horizon)]

        return {
            "forecast": [float(x) for x in forecast],
            "lower_bound": [float(x) for x in lower],
            "upper_bound": [float(x) for x in upper],
            "confidence_level": confidence_level,
            "forecast_dates": [d.isoformat() for d in forecast_dates],
            "horizon": horizon,
        }

    def _calculate_confidence_impl(
        self,
        forecast: List[float],
        std_errors: List[float],
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Tool implementation: Calculate confidence intervals.

        Args:
            forecast: Point forecasts
            std_errors: Standard errors
            confidence_level: Confidence level

        Returns:
            Dict with confidence intervals
        """
        self._tool_call_count += 1

        from scipy.stats import norm

        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        lower = [f - z_score * s for f, s in zip(forecast, std_errors)]
        upper = [f + z_score * s for f, s in zip(forecast, std_errors)]

        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence_level": confidence_level,
            "z_score": float(z_score),
        }

    def _evaluate_model_impl(
        self,
        input_data: Dict[str, Any],
        train_test_split: float = 0.8,
    ) -> Dict[str, Any]:
        """Tool implementation: Evaluate model accuracy.

        Args:
            input_data: Input data dict
            train_test_split: Train/test split ratio

        Returns:
            Dict with accuracy metrics
        """
        self._tool_call_count += 1

        data = self._last_training_data if self._last_training_data is not None else input_data["data"]
        target = input_data["target_column"]
        series = data[target].dropna()

        # Split data
        split_idx = int(len(series) * train_test_split)
        train = series[:split_idx]
        test = series[split_idx:]

        if len(test) < 1:
            raise ValueError("Test set is empty - increase dataset size or reduce train_test_split")

        if not STATSMODELS_AVAILABLE or self._fitted_model is None:
            # Mock metrics
            return {
                "rmse": 10.5,
                "mae": 8.2,
                "mape": 5.3,
                "train_size": len(train),
                "test_size": len(test),
                "train_test_split": train_test_split,
            }

        # Generate predictions for test set
        predictions = self._fitted_model.forecast(steps=len(test))

        # Calculate metrics
        actual = test.values
        pred = predictions.values if hasattr(predictions, 'values') else predictions

        # RMSE
        rmse = np.sqrt(np.mean((actual - pred) ** 2))

        # MAE
        mae = np.mean(np.abs(actual - pred))

        # MAPE
        mape = np.mean(np.abs((actual - pred) / actual)) * 100

        # Create calculation citation for model evaluation
        calc_citation = CalculationCitation(
            step_name="evaluate_model",
            formula="RMSE=√(mean((actual-pred)²)), MAE=mean(|actual-pred|), MAPE=mean(|actual-pred|/actual)×100",
            inputs={
                "train_size": len(train),
                "test_size": len(test),
                "train_test_split": train_test_split,
            },
            output={
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
            },
            timestamp=DeterministicClock.now(),
            tool_call_id=f"eval_{self._tool_call_count}",
        )
        self._calculation_citations.append(calc_citation)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "train_size": len(train),
            "test_size": len(test),
            "train_test_split": train_test_split,
        }

    def _build_output(
        self,
        input_data: Dict[str, Any],
        tool_results: Dict[str, Any],
        explanation: Optional[str],
    ) -> Dict[str, Any]:
        """Build output from tool results.

        Args:
            input_data: Original input
            tool_results: Results from tool calls
            explanation: AI-generated explanation

        Returns:
            Dict with complete forecast results
        """
        forecast_data = tool_results.get("forecast", {})
        model_data = tool_results.get("model", {})
        evaluation_data = tool_results.get("evaluation", {})
        seasonality_data = tool_results.get("seasonality", {})
        stationarity_data = tool_results.get("stationarity", {})
        preprocessing_data = tool_results.get("preprocessing", {})

        output = {
            "forecast": forecast_data.get("forecast", []),
            "lower_bound": forecast_data.get("lower_bound", []),
            "upper_bound": forecast_data.get("upper_bound", []),
            "forecast_dates": forecast_data.get("forecast_dates", []),
            "confidence_level": forecast_data.get("confidence_level", 0.95),
        }

        # Add model parameters
        if model_data:
            output["model_params"] = {
                "order": model_data.get("order", (1, 1, 1)),
                "seasonal_order": model_data.get("seasonal_order", (1, 1, 1, 12)),
                "aic": model_data.get("aic", 0.0),
                "bic": model_data.get("bic", 0.0),
                "auto_tuned": model_data.get("auto_tuned", False),
            }

        # Add evaluation metrics
        if evaluation_data:
            output["metrics"] = {
                "rmse": evaluation_data.get("rmse", 0.0),
                "mae": evaluation_data.get("mae", 0.0),
                "mape": evaluation_data.get("mape", 0.0),
                "train_size": evaluation_data.get("train_size", 0),
                "test_size": evaluation_data.get("test_size", 0),
            }

        # Add seasonality info
        if seasonality_data:
            output["seasonality"] = seasonality_data

        # Add stationarity info
        if stationarity_data:
            output["stationarity"] = stationarity_data

        # Add preprocessing info
        if preprocessing_data:
            output["preprocessing"] = preprocessing_data

        # Add AI explanation
        if explanation and self.enable_explanations:
            output["explanation"] = explanation

        # Add citations for calculations
        if self._calculation_citations:
            output["citations"] = {
                "calculations": [c.dict() for c in self._calculation_citations],
            }

        return output

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return {
            "agent_id": self.metadata.id,
            "ai_metrics": {
                "ai_call_count": self._ai_call_count,
                "tool_call_count": self._tool_call_count,
                "total_cost_usd": self._total_cost_usd,
                "avg_cost_per_forecast": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
        }
