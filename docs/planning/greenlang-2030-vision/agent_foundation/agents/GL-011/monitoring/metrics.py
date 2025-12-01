# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Prometheus Metrics Module.

This module defines comprehensive Prometheus metrics for the FuelManagementAgent,
covering fuel management, optimization, integration, calculator performance,
business KPIs, compliance, health, alerts, and system performance.

The metrics follow Prometheus naming conventions:
- Counter: Monotonically increasing values (totals)
- Gauge: Values that can go up and down (current state)
- Histogram: Observations bucketed by value (latencies, durations)
- Summary: Quantile observations (similar to histogram but client-side)
- Info: Static metadata about the service

Total Metrics: 178 (GL-010 standard compliance)

Usage:
    >>> from gl011_monitoring.metrics import (
    ...     fuel_price_current,
    ...     fuel_inventory_level,
    ...     optimization_execution_time_seconds
    ... )
    >>> fuel_price_current.labels(fuel_type='coal', supplier='supplier_a').set(85.50)
    >>> fuel_inventory_level.labels(fuel_type='natural_gas', storage_location='tank_1').set(45000)
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# AGENT INFORMATION (3 metrics)
# =============================================================================

AGENT_INFO = Info(
    'gl011_fuelcraft',
    'GL-011 FUELCRAFT FuelManagementAgent information'
)

AGENT_VERSION = Info(
    'gl011_version',
    'GL-011 agent version and build information'
)

AGENT_CONFIG = Info(
    'gl011_config',
    'GL-011 agent configuration metadata'
)

# =============================================================================
# FUEL MANAGEMENT METRICS - Fuel Prices (10 metrics)
# =============================================================================

fuel_price_current = Gauge(
    'gl011_fuel_price_current_usd',
    'Current fuel price in USD per unit',
    ['fuel_type', 'supplier', 'unit']
)

fuel_price_spot = Gauge(
    'gl011_fuel_price_spot_usd',
    'Current spot market fuel price in USD',
    ['fuel_type', 'market']
)

fuel_price_contract = Gauge(
    'gl011_fuel_price_contract_usd',
    'Contract fuel price in USD per unit',
    ['fuel_type', 'supplier', 'contract_id']
)

fuel_price_forecast = Gauge(
    'gl011_fuel_price_forecast_usd',
    'Forecasted fuel price in USD',
    ['fuel_type', 'horizon_days']
)

fuel_price_volatility = Gauge(
    'gl011_fuel_price_volatility_percent',
    'Fuel price volatility as percentage (30-day rolling)',
    ['fuel_type']
)

fuel_price_change_percent = Gauge(
    'gl011_fuel_price_change_percent',
    'Fuel price change percentage from previous period',
    ['fuel_type', 'period']
)

fuel_price_updates_total = Counter(
    'gl011_fuel_price_updates_total',
    'Total fuel price updates received',
    ['fuel_type', 'source']
)

fuel_price_alert_threshold_usd = Gauge(
    'gl011_fuel_price_alert_threshold_usd',
    'Price alert threshold in USD',
    ['fuel_type', 'direction']
)

fuel_price_index = Gauge(
    'gl011_fuel_price_index',
    'Fuel price index value (base = 100)',
    ['fuel_type', 'index_name']
)

fuel_price_arbitrage_usd = Gauge(
    'gl011_fuel_price_arbitrage_usd',
    'Potential arbitrage opportunity in USD',
    ['fuel_type', 'from_supplier', 'to_supplier']
)

# =============================================================================
# FUEL MANAGEMENT METRICS - Inventory (10 metrics)
# =============================================================================

fuel_inventory_level = Gauge(
    'gl011_fuel_inventory_level_kg',
    'Current fuel inventory level in kilograms',
    ['fuel_type', 'storage_location']
)

fuel_inventory_level_mwh = Gauge(
    'gl011_fuel_inventory_level_mwh',
    'Current fuel inventory energy content in MWh',
    ['fuel_type', 'storage_location']
)

fuel_inventory_days = Gauge(
    'gl011_fuel_inventory_days',
    'Days of fuel inventory remaining at current consumption',
    ['fuel_type']
)

fuel_inventory_min_level_kg = Gauge(
    'gl011_fuel_inventory_min_level_kg',
    'Minimum required inventory level in kilograms',
    ['fuel_type', 'storage_location']
)

fuel_inventory_max_level_kg = Gauge(
    'gl011_fuel_inventory_max_level_kg',
    'Maximum storage capacity in kilograms',
    ['fuel_type', 'storage_location']
)

fuel_inventory_utilization_percent = Gauge(
    'gl011_fuel_inventory_utilization_percent',
    'Storage utilization as percentage of capacity',
    ['fuel_type', 'storage_location']
)

fuel_inventory_reorder_point_kg = Gauge(
    'gl011_fuel_inventory_reorder_point_kg',
    'Reorder point in kilograms',
    ['fuel_type']
)

fuel_inventory_safety_stock_kg = Gauge(
    'gl011_fuel_inventory_safety_stock_kg',
    'Safety stock level in kilograms',
    ['fuel_type']
)

fuel_inventory_value_usd = Gauge(
    'gl011_fuel_inventory_value_usd',
    'Total inventory value in USD',
    ['fuel_type', 'storage_location']
)

fuel_inventory_adjustments_total = Counter(
    'gl011_fuel_inventory_adjustments_total',
    'Total inventory adjustments',
    ['fuel_type', 'adjustment_type', 'reason']
)

# =============================================================================
# FUEL MANAGEMENT METRICS - Consumption (10 metrics)
# =============================================================================

fuel_consumption_rate_kg_hr = Gauge(
    'gl011_fuel_consumption_rate_kg_hr',
    'Current fuel consumption rate in kg per hour',
    ['fuel_type', 'unit_id']
)

fuel_consumption_rate_mwh = Gauge(
    'gl011_fuel_consumption_rate_mwh',
    'Current fuel consumption rate in MWh equivalent',
    ['fuel_type']
)

fuel_consumption_daily_kg = Gauge(
    'gl011_fuel_consumption_daily_kg',
    'Daily fuel consumption in kilograms',
    ['fuel_type', 'date']
)

fuel_consumption_monthly_kg = Gauge(
    'gl011_fuel_consumption_monthly_kg',
    'Monthly fuel consumption in kilograms',
    ['fuel_type', 'year_month']
)

fuel_consumption_ytd_kg = Gauge(
    'gl011_fuel_consumption_ytd_kg',
    'Year-to-date fuel consumption in kilograms',
    ['fuel_type', 'year']
)

fuel_consumption_forecast_kg = Gauge(
    'gl011_fuel_consumption_forecast_kg',
    'Forecasted fuel consumption in kilograms',
    ['fuel_type', 'horizon_days']
)

fuel_consumption_variance_percent = Gauge(
    'gl011_fuel_consumption_variance_percent',
    'Variance from forecasted consumption as percentage',
    ['fuel_type']
)

fuel_consumption_per_mwh = Gauge(
    'gl011_fuel_consumption_per_mwh_kg',
    'Fuel consumption per MWh generated in kilograms',
    ['fuel_type', 'unit_id']
)

fuel_consumption_total = Counter(
    'gl011_fuel_consumption_total_kg',
    'Total fuel consumed in kilograms',
    ['fuel_type', 'unit_id']
)

fuel_consumption_efficiency_percent = Gauge(
    'gl011_fuel_consumption_efficiency_percent',
    'Fuel consumption efficiency percentage',
    ['fuel_type', 'unit_id']
)

# =============================================================================
# FUEL MANAGEMENT METRICS - Quality (10 metrics)
# =============================================================================

fuel_quality_score = Gauge(
    'gl011_fuel_quality_score',
    'Overall fuel quality score (0-100)',
    ['fuel_type', 'batch_id']
)

fuel_quality_sulfur_percent = Gauge(
    'gl011_fuel_quality_sulfur_percent',
    'Fuel sulfur content as percentage',
    ['fuel_type', 'batch_id']
)

fuel_quality_ash_percent = Gauge(
    'gl011_fuel_quality_ash_percent',
    'Fuel ash content as percentage',
    ['fuel_type', 'batch_id']
)

fuel_quality_moisture_percent = Gauge(
    'gl011_fuel_quality_moisture_percent',
    'Fuel moisture content as percentage',
    ['fuel_type', 'batch_id']
)

fuel_quality_calorific_mj_kg = Gauge(
    'gl011_fuel_quality_calorific_mj_kg',
    'Fuel calorific value in MJ/kg',
    ['fuel_type', 'batch_id']
)

fuel_quality_volatiles_percent = Gauge(
    'gl011_fuel_quality_volatiles_percent',
    'Volatile matter content as percentage',
    ['fuel_type', 'batch_id']
)

fuel_quality_nitrogen_percent = Gauge(
    'gl011_fuel_quality_nitrogen_percent',
    'Nitrogen content as percentage',
    ['fuel_type', 'batch_id']
)

fuel_quality_hardgrove_index = Gauge(
    'gl011_fuel_quality_hardgrove_index',
    'Hardgrove Grindability Index (HGI) for coal',
    ['fuel_type', 'batch_id']
)

fuel_quality_tests_total = Counter(
    'gl011_fuel_quality_tests_total',
    'Total fuel quality tests performed',
    ['fuel_type', 'test_type', 'result']
)

fuel_quality_violations_total = Counter(
    'gl011_fuel_quality_violations_total',
    'Total fuel quality violations detected',
    ['fuel_type', 'parameter', 'severity']
)

# =============================================================================
# FUEL BLEND METRICS (5 metrics)
# =============================================================================

fuel_blend_ratio = Gauge(
    'gl011_fuel_blend_ratio',
    'Current fuel blend ratio (0-1)',
    ['fuel_type', 'blend_name']
)

fuel_blend_target_ratio = Gauge(
    'gl011_fuel_blend_target_ratio',
    'Target fuel blend ratio (0-1)',
    ['fuel_type', 'blend_name']
)

fuel_blend_deviation_percent = Gauge(
    'gl011_fuel_blend_deviation_percent',
    'Deviation from target blend ratio as percentage',
    ['blend_name']
)

fuel_blend_changes_total = Counter(
    'gl011_fuel_blend_changes_total',
    'Total blend ratio changes',
    ['blend_name', 'reason']
)

fuel_blend_optimization_score = Gauge(
    'gl011_fuel_blend_optimization_score',
    'Blend optimization score (0-100)',
    ['blend_name']
)

# =============================================================================
# FUEL SUPPLIER METRICS (5 metrics)
# =============================================================================

fuel_supplier_reliability_score = Gauge(
    'gl011_fuel_supplier_reliability_score',
    'Supplier reliability score (0-100)',
    ['supplier', 'fuel_type']
)

fuel_supplier_lead_time_days = Gauge(
    'gl011_fuel_supplier_lead_time_days',
    'Average supplier lead time in days',
    ['supplier', 'fuel_type']
)

fuel_delivery_delays_total = Counter(
    'gl011_fuel_delivery_delays_total',
    'Total delivery delays',
    ['supplier', 'fuel_type', 'severity']
)

fuel_delivery_on_time_percent = Gauge(
    'gl011_fuel_delivery_on_time_percent',
    'On-time delivery percentage',
    ['supplier']
)

fuel_contract_utilization_percent = Gauge(
    'gl011_fuel_contract_utilization_percent',
    'Contract utilization percentage',
    ['contract_id', 'supplier', 'fuel_type']
)

# =============================================================================
# OPTIMIZATION METRICS (30 metrics)
# =============================================================================

optimization_execution_time_seconds = Histogram(
    'gl011_optimization_execution_time_seconds',
    'Optimization runtime in seconds',
    ['objective', 'solver'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

optimization_iterations = Histogram(
    'gl011_optimization_iterations',
    'Number of solver iterations per optimization',
    ['objective'],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
)

optimization_objective_value = Gauge(
    'gl011_optimization_objective_value',
    'Optimal objective function value',
    ['objective', 'solver']
)

optimization_constraints_violated = Gauge(
    'gl011_optimization_constraints_violated',
    'Number of constraint violations in solution',
    ['constraint_type']
)

optimization_constraints_active = Gauge(
    'gl011_optimization_constraints_active',
    'Number of active constraints in solution',
    ['constraint_type']
)

optimization_gap_percent = Gauge(
    'gl011_optimization_gap_percent',
    'Optimality gap as percentage',
    ['objective', 'solver']
)

optimization_fuel_switches_total = Counter(
    'gl011_optimization_fuel_switches_total',
    'Total fuel switch recommendations',
    ['from_fuel', 'to_fuel', 'reason']
)

optimization_cost_savings_usd = Gauge(
    'gl011_optimization_cost_savings_usd',
    'Predicted cost savings in USD',
    ['scenario', 'horizon_days']
)

optimization_cost_savings_cumulative_usd = Counter(
    'gl011_optimization_cost_savings_cumulative_usd_total',
    'Cumulative cost savings in USD'
)

optimization_carbon_reduction_kg = Gauge(
    'gl011_optimization_carbon_reduction_kg',
    'Predicted CO2 reduction in kilograms',
    ['scenario']
)

optimization_reliability_score = Gauge(
    'gl011_optimization_reliability_score',
    'Solution reliability score (0-100)',
    ['objective']
)

optimization_requests_total = Counter(
    'gl011_optimization_requests_total',
    'Total optimization requests',
    ['objective', 'status']
)

optimization_in_progress = Gauge(
    'gl011_optimization_in_progress',
    'Number of optimizations currently in progress'
)

optimization_errors_total = Counter(
    'gl011_optimization_errors_total',
    'Total optimization errors',
    ['objective', 'error_type']
)

optimization_timeouts_total = Counter(
    'gl011_optimization_timeouts_total',
    'Total optimization timeouts',
    ['objective']
)

optimization_retries_total = Counter(
    'gl011_optimization_retries_total',
    'Total optimization retries',
    ['objective', 'retry_reason']
)

optimization_queue_depth = Gauge(
    'gl011_optimization_queue_depth',
    'Number of pending optimization requests'
)

optimization_score = Gauge(
    'gl011_optimization_score',
    'Latest optimization score (0-100)'
)

optimization_fuel_mix_ratio = Gauge(
    'gl011_optimization_fuel_mix_ratio',
    'Optimized fuel mix ratio by fuel type',
    ['fuel_type']
)

optimization_cost_usd = Gauge(
    'gl011_optimization_cost_usd',
    'Optimized total cost in USD'
)

optimization_emissions_kg = Gauge(
    'gl011_optimization_emissions_kg',
    'Optimized emissions in kg CO2e'
)

optimization_renewable_share = Gauge(
    'gl011_optimization_renewable_share',
    'Renewable fuel share in solution (0-1)'
)

optimization_efficiency_percent = Gauge(
    'gl011_optimization_efficiency_percent',
    'Optimization efficiency percentage'
)

optimization_solution_time_seconds = Gauge(
    'gl011_optimization_solution_time_seconds',
    'Time to find optimal solution in seconds',
    ['solver']
)

optimization_model_variables = Gauge(
    'gl011_optimization_model_variables',
    'Number of decision variables in model',
    ['model_type']
)

optimization_model_constraints = Gauge(
    'gl011_optimization_model_constraints',
    'Number of constraints in model',
    ['model_type']
)

optimization_sensitivity_analysis_total = Counter(
    'gl011_optimization_sensitivity_analysis_total',
    'Total sensitivity analyses performed',
    ['parameter_type']
)

optimization_scenario_comparisons_total = Counter(
    'gl011_optimization_scenario_comparisons_total',
    'Total scenario comparisons performed'
)

optimization_recommendations_total = Counter(
    'gl011_optimization_recommendations_total',
    'Total optimization recommendations generated',
    ['recommendation_type']
)

optimization_recommendations_accepted = Counter(
    'gl011_optimization_recommendations_accepted_total',
    'Total recommendations accepted',
    ['recommendation_type']
)

# =============================================================================
# INTEGRATION METRICS (25 metrics)
# =============================================================================

integration_api_calls_total = Counter(
    'gl011_integration_api_calls_total',
    'Total API calls',
    ['endpoint', 'method', 'status']
)

integration_api_latency_seconds = Histogram(
    'gl011_integration_api_latency_seconds',
    'API response time in seconds',
    ['endpoint', 'method'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

integration_api_errors_total = Counter(
    'gl011_integration_api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type', 'status_code']
)

integration_erp_sync_lag_seconds = Gauge(
    'gl011_integration_erp_sync_lag_seconds',
    'ERP data synchronization lag in seconds'
)

integration_erp_sync_duration_seconds = Histogram(
    'gl011_integration_erp_sync_duration_seconds',
    'ERP synchronization duration in seconds',
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
)

integration_erp_sync_status = Gauge(
    'gl011_integration_erp_sync_status',
    'ERP sync status (1=synced, 0=out of sync)',
    ['erp_system', 'data_type']
)

integration_market_data_age_seconds = Gauge(
    'gl011_integration_market_data_age_seconds',
    'Market data staleness in seconds',
    ['data_type', 'market']
)

integration_market_data_updates_total = Counter(
    'gl011_integration_market_data_updates_total',
    'Total market data updates received',
    ['data_type', 'market']
)

integration_storage_connector_health = Gauge(
    'gl011_integration_storage_connector_health',
    'Storage system connector health (1=healthy, 0=unhealthy)',
    ['storage_system', 'connector_type']
)

integration_procurement_sync_success = Gauge(
    'gl011_integration_procurement_sync_success',
    'Procurement integration sync status (1=success, 0=failed)',
    ['system']
)

integration_connection_status = Gauge(
    'gl011_integration_connection_status',
    'Integration connection status (1=connected, 0=disconnected)',
    ['integration_type']
)

integration_retry_total = Counter(
    'gl011_integration_retry_total',
    'Total integration retries',
    ['integration_type', 'retry_reason']
)

integration_circuit_breaker_state = Gauge(
    'gl011_integration_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 0.5=half-open)',
    ['integration_type']
)

integration_requests_total = Counter(
    'gl011_integration_requests_total',
    'Total integration requests',
    ['integration_type', 'status']
)

integration_latency_seconds = Histogram(
    'gl011_integration_latency_seconds',
    'Integration request latency in seconds',
    ['integration_type'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

integration_errors_total = Counter(
    'gl011_integration_errors_total',
    'Total integration errors',
    ['integration_type', 'error_type']
)

integration_scada_connection_status = Gauge(
    'gl011_integration_scada_connection_status',
    'SCADA system connection status',
    ['scada_system']
)

integration_scada_data_points_total = Counter(
    'gl011_integration_scada_data_points_total',
    'Total SCADA data points received',
    ['scada_system', 'data_type']
)

integration_historian_query_duration_seconds = Histogram(
    'gl011_integration_historian_query_duration_seconds',
    'Process historian query duration',
    ['historian_system', 'query_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30)
)

integration_historian_queries_total = Counter(
    'gl011_integration_historian_queries_total',
    'Total queries to process historian',
    ['historian_system', 'query_type']
)

integration_weather_api_calls_total = Counter(
    'gl011_integration_weather_api_calls_total',
    'Total weather API calls',
    ['provider', 'endpoint']
)

integration_weather_data_age_seconds = Gauge(
    'gl011_integration_weather_data_age_seconds',
    'Weather data age in seconds',
    ['location']
)

integration_supplier_portal_status = Gauge(
    'gl011_integration_supplier_portal_status',
    'Supplier portal connection status',
    ['supplier', 'portal_type']
)

integration_logistics_api_status = Gauge(
    'gl011_integration_logistics_api_status',
    'Logistics API connection status',
    ['provider']
)

integration_trading_platform_status = Gauge(
    'gl011_integration_trading_platform_status',
    'Trading platform connection status',
    ['platform']
)

# =============================================================================
# CALCULATOR METRICS (25 metrics)
# =============================================================================

calculator_execution_time_seconds = Histogram(
    'gl011_calculator_execution_time_seconds',
    'Calculator performance in seconds',
    ['calculator_name'],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
)

calculator_cache_hit_rate = Gauge(
    'gl011_calculator_cache_hit_rate',
    'Calculator cache hit rate (0-1)',
    ['calculator_name']
)

calculator_errors_total = Counter(
    'gl011_calculator_errors_total',
    'Total calculator errors',
    ['calculator_name', 'error_type']
)

calculator_provenance_hash = Info(
    'gl011_calculator_provenance_hash',
    'Provenance hash for determinism tracking'
)

calculator_input_validation_failures_total = Counter(
    'gl011_calculator_input_validation_failures_total',
    'Total input validation failures',
    ['calculator_name', 'validation_rule']
)

calorific_value_lookups_total = Counter(
    'gl011_calorific_value_lookups_total',
    'Total calorific value database lookups',
    ['fuel_type']
)

emissions_factor_lookups_total = Counter(
    'gl011_emissions_factor_lookups_total',
    'Total emission factor lookups',
    ['pollutant', 'fuel_type']
)

cost_optimization_solutions_total = Counter(
    'gl011_cost_optimization_solutions_total',
    'Total cost optimization solutions generated',
    ['scenario', 'status']
)

blending_calculations_total = Counter(
    'gl011_blending_calculations_total',
    'Total blending calculations performed',
    ['blend_type']
)

calculator_invocations_total = Counter(
    'gl011_calculator_invocations_total',
    'Total calculator invocations',
    ['calculator_type']
)

calculator_duration_seconds = Histogram(
    'gl011_calculator_duration_seconds',
    'Calculator execution duration',
    ['calculator_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
)

carbon_footprint_kg = Gauge(
    'gl011_carbon_footprint_kg',
    'Latest carbon footprint calculation in kg'
)

carbon_intensity_kg_mwh = Gauge(
    'gl011_carbon_intensity_kg_mwh',
    'Latest carbon intensity in kg/MWh',
    ['fuel_type']
)

blending_quality_score = Gauge(
    'gl011_blending_quality_score',
    'Latest blending quality score'
)

cost_per_mwh_usd = Gauge(
    'gl011_cost_per_mwh_usd',
    'Latest cost per MWh in USD',
    ['fuel_type']
)

eoq_quantity_kg = Gauge(
    'gl011_eoq_quantity_kg',
    'Economic order quantity in kilograms',
    ['fuel_type']
)

emission_factor_co2 = Gauge(
    'gl011_emission_factor_co2_kg_mj',
    'Emission factor for CO2 in kg/MJ',
    ['fuel_type']
)

heating_value_mj_kg = Gauge(
    'gl011_heating_value_mj_kg',
    'Heating value in MJ/kg',
    ['fuel_type']
)

calculator_retries_total = Counter(
    'gl011_calculator_retries_total',
    'Total calculator retries',
    ['calculator_name', 'retry_reason']
)

provenance_hashes_generated_total = Counter(
    'gl011_provenance_hashes_generated_total',
    'Total SHA-256 provenance hashes generated',
    ['data_type']
)

heat_rate_btu_kwh = Gauge(
    'gl011_heat_rate_btu_kwh',
    'Heat rate in BTU/kWh',
    ['unit_id', 'fuel_type']
)

thermal_efficiency_percent = Gauge(
    'gl011_thermal_efficiency_percent',
    'Thermal efficiency percentage',
    ['unit_id']
)

combustion_efficiency_percent = Gauge(
    'gl011_combustion_efficiency_percent',
    'Combustion efficiency percentage',
    ['unit_id', 'fuel_type']
)

mass_balance_calculations_total = Counter(
    'gl011_mass_balance_calculations_total',
    'Total mass balance calculations',
    ['material_type']
)

stoichiometric_calculations_total = Counter(
    'gl011_stoichiometric_calculations_total',
    'Total stoichiometric calculations',
    ['reaction_type']
)

# =============================================================================
# BUSINESS METRICS (20 metrics)
# =============================================================================

fuel_cost_total_usd = Gauge(
    'gl011_fuel_cost_total_usd',
    'Total fuel expenditure in USD',
    ['period']
)

fuel_cost_avoided_usd = Counter(
    'gl011_fuel_cost_avoided_usd_total',
    'Cumulative cost savings vs baseline in USD'
)

carbon_emissions_total_kg = Gauge(
    'gl011_carbon_emissions_total_kg',
    'Total CO2 emissions in kilograms',
    ['scope', 'period']
)

carbon_emissions_avoided_kg = Counter(
    'gl011_carbon_emissions_avoided_kg_total',
    'Cumulative CO2 reduction in kilograms'
)

energy_delivered_mwh = Gauge(
    'gl011_energy_delivered_mwh',
    'Total energy output in MWh',
    ['period']
)

fuel_efficiency_percent = Gauge(
    'gl011_fuel_efficiency_percent',
    'Overall fuel efficiency percentage'
)

procurement_orders_total = Counter(
    'gl011_procurement_orders_total',
    'Total purchase orders processed',
    ['fuel_type', 'supplier', 'status']
)

procurement_value_usd = Counter(
    'gl011_procurement_value_usd_total',
    'Total procurement value in USD',
    ['fuel_type', 'supplier']
)

inventory_turnover_ratio = Gauge(
    'gl011_inventory_turnover_ratio',
    'Inventory turnover ratio',
    ['fuel_type']
)

contract_compliance_score = Gauge(
    'gl011_contract_compliance_score',
    'Contract adherence score (0-100)',
    ['contract_id', 'supplier']
)

fuel_cost_savings_usd = Counter(
    'gl011_fuel_cost_savings_usd_total',
    'Cumulative fuel cost savings in USD'
)

emissions_reduction_kg = Counter(
    'gl011_emissions_reduction_kg_total',
    'Cumulative emissions reduction in kg CO2e'
)

renewable_energy_mwh = Counter(
    'gl011_renewable_energy_mwh_total',
    'Cumulative renewable energy used in MWh'
)

fuel_inventory_kg = Gauge(
    'gl011_fuel_inventory_kg',
    'Current fuel inventory in kg',
    ['fuel_type']
)

budget_utilization_percent = Gauge(
    'gl011_budget_utilization_percent',
    'Fuel budget utilization percentage',
    ['budget_category', 'period']
)

cost_variance_percent = Gauge(
    'gl011_cost_variance_percent',
    'Cost variance from budget as percentage',
    ['category']
)

roi_percent = Gauge(
    'gl011_roi_percent',
    'Return on investment percentage',
    ['initiative']
)

payback_period_months = Gauge(
    'gl011_payback_period_months',
    'Payback period in months',
    ['initiative']
)

energy_cost_per_unit = Gauge(
    'gl011_energy_cost_per_unit_usd',
    'Energy cost per production unit in USD',
    ['production_type']
)

carbon_cost_usd = Gauge(
    'gl011_carbon_cost_usd',
    'Carbon cost in USD (shadow price or market)',
    ['pricing_mechanism']
)

# =============================================================================
# COMPLIANCE METRICS (15 metrics)
# =============================================================================

compliance_fuel_quality_status = Gauge(
    'gl011_compliance_fuel_quality_status',
    'Fuel quality compliance status (1=compliant, 0=violation)',
    ['standard', 'fuel_type', 'parameter']
)

compliance_emissions_status = Gauge(
    'gl011_compliance_emissions_status',
    'Emissions compliance status (1=compliant, 0=violation)',
    ['pollutant', 'limit_type', 'jurisdiction']
)

compliance_violations_total = Counter(
    'gl011_compliance_violations_total',
    'Total compliance violations',
    ['violation_type', 'severity', 'jurisdiction']
)

compliance_reporting_lag_seconds = Gauge(
    'gl011_compliance_reporting_lag_seconds',
    'Reporting timeliness in seconds',
    ['report_type']
)

compliance_audit_score = Gauge(
    'gl011_compliance_audit_score',
    'Audit results score (0-100)',
    ['audit_type', 'auditor']
)

compliance_margin_percent = Gauge(
    'gl011_compliance_margin_percent',
    'Margin to compliance limit as percentage',
    ['parameter', 'jurisdiction']
)

compliance_deadline_days = Gauge(
    'gl011_compliance_deadline_days',
    'Days until compliance deadline',
    ['deadline_type', 'jurisdiction']
)

compliance_permit_expiration_days = Gauge(
    'gl011_compliance_permit_expiration_days',
    'Days until permit expiration',
    ['permit_id', 'permit_type']
)

compliance_inspections_total = Counter(
    'gl011_compliance_inspections_total',
    'Total compliance inspections',
    ['inspection_type', 'result']
)

compliance_corrective_actions_total = Counter(
    'gl011_compliance_corrective_actions_total',
    'Total corrective actions required',
    ['action_type', 'status']
)

compliance_certificate_status = Gauge(
    'gl011_compliance_certificate_status',
    'Certificate validity status (1=valid, 0=expired)',
    ['certificate_type', 'issuer']
)

compliance_sulfur_limit_percent = Gauge(
    'gl011_compliance_sulfur_limit_percent',
    'Sulfur content as percentage of limit',
    ['fuel_type', 'standard']
)

compliance_ash_limit_percent = Gauge(
    'gl011_compliance_ash_limit_percent',
    'Ash content as percentage of limit',
    ['fuel_type', 'standard']
)

compliance_nox_limit_percent = Gauge(
    'gl011_compliance_nox_limit_percent',
    'NOx emissions as percentage of limit',
    ['source', 'jurisdiction']
)

compliance_documentation_status = Gauge(
    'gl011_compliance_documentation_status',
    'Documentation completeness status (1=complete, 0=incomplete)',
    ['document_type']
)

# =============================================================================
# HEALTH METRICS (10 metrics)
# =============================================================================

agent_health_status = Gauge(
    'gl011_agent_health_status',
    'Overall health status (1=healthy, 0=unhealthy)'
)

agent_uptime_seconds = Gauge(
    'gl011_agent_uptime_seconds',
    'Agent uptime in seconds'
)

agent_version_info = Info(
    'gl011_agent_version',
    'Agent version tracking'
)

agent_restarts_total = Counter(
    'gl011_agent_restarts_total',
    'Total agent restarts',
    ['restart_reason']
)

agent_memory_usage_bytes = Gauge(
    'gl011_agent_memory_usage_bytes',
    'Memory consumption in bytes',
    ['memory_type']
)

agent_cpu_usage_percent = Gauge(
    'gl011_agent_cpu_usage_percent',
    'CPU usage percentage'
)

agent_thread_count = Gauge(
    'gl011_agent_thread_count',
    'Number of active threads',
    ['thread_type']
)

agent_cache_size_bytes = Gauge(
    'gl011_agent_cache_size_bytes',
    'Cache size in bytes',
    ['cache_name']
)

agent_start_time_seconds = Gauge(
    'gl011_agent_start_time_seconds',
    'Agent start time as Unix timestamp'
)

agent_last_heartbeat_timestamp = Gauge(
    'gl011_agent_last_heartbeat_timestamp',
    'Last heartbeat timestamp'
)

# =============================================================================
# ALERT METRICS (8 metrics)
# =============================================================================

alerts_active = Gauge(
    'gl011_alerts_active',
    'Number of active alerts',
    ['severity', 'alert_name', 'category']
)

alerts_fired_total = Counter(
    'gl011_alerts_fired_total',
    'Total alerts triggered',
    ['severity', 'alert_type']
)

alerts_resolved_total = Counter(
    'gl011_alerts_resolved_total',
    'Total alerts resolved',
    ['severity', 'resolution_type']
)

alerts_mean_time_to_resolution_seconds = Gauge(
    'gl011_alerts_mean_time_to_resolution_seconds',
    'Mean time to resolution (MTTR) in seconds',
    ['severity', 'alert_type']
)

alerts_acknowledged_total = Counter(
    'gl011_alerts_acknowledged_total',
    'Total alerts acknowledged',
    ['severity', 'acknowledged_by']
)

alerts_escalated_total = Counter(
    'gl011_alerts_escalated_total',
    'Total alerts escalated',
    ['original_severity', 'escalated_severity']
)

alert_response_time_seconds = Histogram(
    'gl011_alert_response_time_seconds',
    'Time to acknowledge alerts in seconds',
    ['severity', 'alert_type'],
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)

alert_notification_failures_total = Counter(
    'gl011_alert_notification_failures_total',
    'Total alert notification failures',
    ['channel', 'failure_reason']
)

# =============================================================================
# PERFORMANCE METRICS (5 metrics)
# =============================================================================

request_duration_seconds = Histogram(
    'gl011_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint', 'status'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

request_size_bytes = Histogram(
    'gl011_request_size_bytes',
    'Request size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)

response_size_bytes = Histogram(
    'gl011_response_size_bytes',
    'Response size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)

concurrent_requests = Gauge(
    'gl011_concurrent_requests',
    'Number of active concurrent requests',
    ['endpoint']
)

queue_depth = Gauge(
    'gl011_queue_depth',
    'Request queue depth',
    ['queue_name', 'priority']
)

# =============================================================================
# CACHE METRICS (8 metrics)
# =============================================================================

cache_hits_total = Counter(
    'gl011_cache_hits_total',
    'Total cache hits',
    ['cache_name']
)

cache_misses_total = Counter(
    'gl011_cache_misses_total',
    'Total cache misses',
    ['cache_name']
)

cache_size = Gauge(
    'gl011_cache_size',
    'Current cache size (entries)',
    ['cache_name']
)

cache_memory_bytes = Gauge(
    'gl011_cache_memory_bytes',
    'Cache memory usage in bytes',
    ['cache_name']
)

cache_evictions_total = Counter(
    'gl011_cache_evictions_total',
    'Total cache evictions',
    ['cache_name', 'eviction_reason']
)

cache_hit_ratio = Gauge(
    'gl011_cache_hit_ratio',
    'Cache hit ratio (0-1)',
    ['cache_name']
)

cache_ttl_expirations_total = Counter(
    'gl011_cache_ttl_expirations_total',
    'Total TTL expirations',
    ['cache_name']
)

cache_update_duration_seconds = Histogram(
    'gl011_cache_update_duration_seconds',
    'Cache update duration in seconds',
    ['cache_name'],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
)

# =============================================================================
# DATABASE METRICS (7 metrics)
# =============================================================================

db_connection_pool_size = Gauge(
    'gl011_db_connection_pool_size',
    'Database connection pool size',
    ['database', 'pool_type']
)

db_connections_active = Gauge(
    'gl011_db_connections_active',
    'Number of active database connections',
    ['database']
)

db_query_duration_seconds = Histogram(
    'gl011_db_query_duration_seconds',
    'Database query duration in seconds',
    ['database', 'query_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)
)

db_queries_total = Counter(
    'gl011_db_queries_total',
    'Total database queries',
    ['database', 'query_type', 'status']
)

db_errors_total = Counter(
    'gl011_db_errors_total',
    'Total database errors',
    ['database', 'error_type']
)

db_transactions_total = Counter(
    'gl011_db_transactions_total',
    'Total database transactions',
    ['database', 'status']
)

db_connection_errors_total = Counter(
    'gl011_db_connection_errors_total',
    'Total database connection errors',
    ['database', 'error_type']
)

# =============================================================================
# BATCH PROCESSING METRICS (5 metrics)
# =============================================================================

batch_jobs_total = Counter(
    'gl011_batch_jobs_total',
    'Total batch jobs executed',
    ['job_type', 'status']
)

batch_job_duration_seconds = Histogram(
    'gl011_batch_job_duration_seconds',
    'Batch job duration in seconds',
    ['job_type'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200)
)

batch_records_processed_total = Counter(
    'gl011_batch_records_processed_total',
    'Total records processed in batch jobs',
    ['job_type', 'status']
)

batch_job_errors_total = Counter(
    'gl011_batch_job_errors_total',
    'Total batch job errors',
    ['job_type', 'error_type']
)

batch_job_queue_size = Gauge(
    'gl011_batch_job_queue_size',
    'Number of batch jobs waiting to run',
    ['job_type']
)

# =============================================================================
# AUDIT METRICS (4 metrics)
# =============================================================================

audit_events_total = Counter(
    'gl011_audit_events_total',
    'Total audit events recorded',
    ['event_type', 'entity_type']
)

audit_trail_size_bytes = Gauge(
    'gl011_audit_trail_size_bytes',
    'Current audit trail storage size in bytes'
)

data_lineage_records_total = Counter(
    'gl011_data_lineage_records_total',
    'Total data lineage records created',
    ['data_type', 'source_system']
)

provenance_verification_total = Counter(
    'gl011_provenance_verification_total',
    'Total provenance hash verifications',
    ['verification_result']
)

# =============================================================================
# REPORTING METRICS (8 metrics)
# =============================================================================

reports_generated_total = Counter(
    'gl011_reports_generated_total',
    'Total reports generated',
    ['report_type', 'format']
)

reports_submitted_total = Counter(
    'gl011_reports_submitted_total',
    'Total reports submitted',
    ['report_type', 'destination']
)

report_generation_duration_seconds = Histogram(
    'gl011_report_generation_duration_seconds',
    'Report generation time in seconds',
    ['report_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

report_errors_total = Counter(
    'gl011_report_errors_total',
    'Total report generation errors',
    ['report_type', 'error_category']
)

report_queue_size = Gauge(
    'gl011_report_queue_size',
    'Number of reports pending generation',
    ['report_type', 'priority']
)

report_deadline_days = Gauge(
    'gl011_report_deadline_days',
    'Days until report submission deadline',
    ['report_type', 'period']
)

report_validation_errors_total = Counter(
    'gl011_report_validation_errors_total',
    'Total report validation errors',
    ['report_type', 'validation_rule']
)

report_distribution_total = Counter(
    'gl011_report_distribution_total',
    'Total report distributions',
    ['report_type', 'channel', 'status']
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def initialize_agent_info(
    version: str,
    build_date: str,
    facility_id: str,
    facility_name: str,
    fuel_types: str
) -> None:
    """
    Initialize agent information metrics.

    Args:
        version: Agent version string
        build_date: Build date in ISO format
        facility_id: Facility identifier
        facility_name: Human-readable facility name
        fuel_types: Comma-separated list of supported fuel types
    """
    AGENT_INFO.info({
        'version': version,
        'build_date': build_date,
        'facility_id': facility_id,
        'facility_name': facility_name,
        'fuel_types': fuel_types
    })
    logger.info(f"Initialized GL-011 agent info: version={version}, facility={facility_id}")


def record_fuel_price(
    fuel_type: str,
    price_usd: float,
    supplier: str,
    unit: str = 'ton'
) -> None:
    """
    Record a fuel price update.

    Args:
        fuel_type: Type of fuel (coal, gas, biomass, etc.)
        price_usd: Price in USD
        supplier: Supplier identifier
        unit: Unit of measurement
    """
    fuel_price_current.labels(
        fuel_type=fuel_type,
        supplier=supplier,
        unit=unit
    ).set(price_usd)

    fuel_price_updates_total.labels(
        fuel_type=fuel_type,
        source=supplier
    ).inc()

    logger.debug(f"Recorded fuel price: {fuel_type} = ${price_usd}/{unit} from {supplier}")


def record_fuel_inventory(
    fuel_type: str,
    quantity_kg: float,
    storage_location: str
) -> None:
    """
    Record fuel inventory level.

    Args:
        fuel_type: Type of fuel
        quantity_kg: Quantity in kilograms
        storage_location: Storage location identifier
    """
    fuel_inventory_level.labels(
        fuel_type=fuel_type,
        storage_location=storage_location
    ).set(quantity_kg)

    fuel_inventory_kg.labels(fuel_type=fuel_type).set(quantity_kg)

    logger.debug(f"Recorded inventory: {fuel_type} = {quantity_kg} kg at {storage_location}")


def record_fuel_consumption(
    fuel_type: str,
    consumption_kg: float,
    unit_id: str = 'default'
) -> None:
    """
    Record fuel consumption.

    Args:
        fuel_type: Type of fuel consumed
        consumption_kg: Consumption in kilograms
        unit_id: Unit identifier
    """
    fuel_consumption_total.labels(
        fuel_type=fuel_type,
        unit_id=unit_id
    ).inc(consumption_kg)

    fuel_consumption_rate_kg_hr.labels(
        fuel_type=fuel_type,
        unit_id=unit_id
    ).set(consumption_kg)


def record_fuel_quality(
    fuel_type: str,
    batch_id: str,
    sulfur_percent: float,
    ash_percent: float,
    moisture_percent: float,
    calorific_mj_kg: float
) -> None:
    """
    Record fuel quality parameters.

    Args:
        fuel_type: Type of fuel
        batch_id: Batch identifier
        sulfur_percent: Sulfur content percentage
        ash_percent: Ash content percentage
        moisture_percent: Moisture content percentage
        calorific_mj_kg: Calorific value in MJ/kg
    """
    fuel_quality_sulfur_percent.labels(
        fuel_type=fuel_type,
        batch_id=batch_id
    ).set(sulfur_percent)

    fuel_quality_ash_percent.labels(
        fuel_type=fuel_type,
        batch_id=batch_id
    ).set(ash_percent)

    fuel_quality_moisture_percent.labels(
        fuel_type=fuel_type,
        batch_id=batch_id
    ).set(moisture_percent)

    fuel_quality_calorific_mj_kg.labels(
        fuel_type=fuel_type,
        batch_id=batch_id
    ).set(calorific_mj_kg)

    fuel_quality_tests_total.labels(
        fuel_type=fuel_type,
        test_type='proximate_analysis',
        result='pass'
    ).inc()


def record_optimization_result(
    objective: str,
    execution_time: float,
    cost_usd: float,
    emissions_kg: float,
    renewable_share: float,
    efficiency_percent: float,
    status: str = 'success'
) -> None:
    """
    Record optimization execution metrics.

    Args:
        objective: Optimization objective (minimize_cost, minimize_emissions, balanced)
        execution_time: Execution time in seconds
        cost_usd: Optimized cost in USD
        emissions_kg: Optimized emissions in kg CO2e
        renewable_share: Renewable fuel share (0-1)
        efficiency_percent: Optimization efficiency
        status: Optimization status
    """
    optimization_execution_time_seconds.labels(
        objective=objective,
        solver='default'
    ).observe(execution_time)

    optimization_requests_total.labels(
        objective=objective,
        status=status
    ).inc()

    if status == 'success':
        optimization_cost_usd.set(cost_usd)
        optimization_emissions_kg.set(emissions_kg)
        optimization_renewable_share.set(renewable_share)
        optimization_efficiency_percent.set(efficiency_percent)

    logger.info(
        f"Optimization completed: objective={objective}, time={execution_time:.2f}s, "
        f"cost=${cost_usd:.2f}, emissions={emissions_kg:.1f}kg"
    )


def record_calculator_invocation(
    calculator_name: str,
    duration_seconds: float,
    success: bool = True,
    error_type: str = None
) -> None:
    """
    Record calculator invocation metrics.

    Args:
        calculator_name: Name of the calculator
        duration_seconds: Execution duration in seconds
        success: Whether calculation succeeded
        error_type: Error type if failed
    """
    calculator_invocations_total.labels(
        calculator_type=calculator_name
    ).inc()

    calculator_execution_time_seconds.labels(
        calculator_name=calculator_name
    ).observe(duration_seconds)

    if not success and error_type:
        calculator_errors_total.labels(
            calculator_name=calculator_name,
            error_type=error_type
        ).inc()


def record_integration_call(
    endpoint: str,
    method: str,
    latency_seconds: float,
    status: str,
    error_type: str = None
) -> None:
    """
    Record integration API call metrics.

    Args:
        endpoint: API endpoint
        method: HTTP method
        latency_seconds: Response time in seconds
        status: Response status (success, error)
        error_type: Error type if failed
    """
    integration_api_calls_total.labels(
        endpoint=endpoint,
        method=method,
        status=status
    ).inc()

    integration_api_latency_seconds.labels(
        endpoint=endpoint,
        method=method
    ).observe(latency_seconds)

    if status == 'error' and error_type:
        integration_api_errors_total.labels(
            endpoint=endpoint,
            error_type=error_type,
            status_code='unknown'
        ).inc()


def record_compliance_status(
    parameter: str,
    is_compliant: bool,
    margin_percent: float,
    jurisdiction: str = 'default'
) -> None:
    """
    Record compliance status.

    Args:
        parameter: Compliance parameter
        is_compliant: Whether in compliance
        margin_percent: Margin to limit as percentage
        jurisdiction: Regulatory jurisdiction
    """
    compliance_fuel_quality_status.labels(
        standard='regulatory',
        fuel_type='general',
        parameter=parameter
    ).set(1 if is_compliant else 0)

    compliance_margin_percent.labels(
        parameter=parameter,
        jurisdiction=jurisdiction
    ).set(margin_percent)

    if not is_compliant:
        severity = 'critical' if margin_percent < -10 else 'warning'
        compliance_violations_total.labels(
            violation_type=parameter,
            severity=severity,
            jurisdiction=jurisdiction
        ).inc()
        logger.warning(f"Compliance violation: {parameter}, margin={margin_percent:.1f}%")


def record_alert(
    severity: str,
    alert_name: str,
    alert_type: str,
    category: str = 'operational'
) -> None:
    """
    Record an alert being triggered.

    Args:
        severity: Alert severity (critical, warning, info)
        alert_name: Name of the alert
        alert_type: Type of alert
        category: Alert category
    """
    alerts_fired_total.labels(
        severity=severity,
        alert_type=alert_type
    ).inc()

    alerts_active.labels(
        severity=severity,
        alert_name=alert_name,
        category=category
    ).inc()

    logger.info(f"Alert triggered: {alert_name} ({severity}) - {alert_type}")


def record_cache_operation(
    cache_name: str,
    hit: bool
) -> None:
    """
    Record cache operation.

    Args:
        cache_name: Name of the cache
        hit: Whether it was a cache hit
    """
    if hit:
        cache_hits_total.labels(cache_name=cache_name).inc()
    else:
        cache_misses_total.labels(cache_name=cache_name).inc()


def update_cache_stats(
    cache_name: str,
    size: int,
    memory_bytes: int,
    hit_ratio: float
) -> None:
    """
    Update cache statistics.

    Args:
        cache_name: Name of the cache
        size: Number of entries
        memory_bytes: Memory usage in bytes
        hit_ratio: Cache hit ratio (0-1)
    """
    cache_size.labels(cache_name=cache_name).set(size)
    cache_memory_bytes.labels(cache_name=cache_name).set(memory_bytes)
    cache_hit_ratio.labels(cache_name=cache_name).set(hit_ratio)


def record_report_generation(
    report_type: str,
    duration_seconds: float,
    format_type: str,
    success: bool
) -> None:
    """
    Record report generation metrics.

    Args:
        report_type: Type of report
        duration_seconds: Generation time in seconds
        format_type: Report format (PDF, Excel, etc.)
        success: Whether generation succeeded
    """
    if success:
        reports_generated_total.labels(
            report_type=report_type,
            format=format_type
        ).inc()
    else:
        report_errors_total.labels(
            report_type=report_type,
            error_category='generation_failed'
        ).inc()

    report_generation_duration_seconds.labels(
        report_type=report_type
    ).observe(duration_seconds)


# =============================================================================
# EXPORTED METRICS REGISTRY
# =============================================================================

# List of all metric objects for registration
ALL_METRICS = [
    # Agent Info (3)
    AGENT_INFO,
    AGENT_VERSION,
    AGENT_CONFIG,

    # Fuel Prices (10)
    fuel_price_current,
    fuel_price_spot,
    fuel_price_contract,
    fuel_price_forecast,
    fuel_price_volatility,
    fuel_price_change_percent,
    fuel_price_updates_total,
    fuel_price_alert_threshold_usd,
    fuel_price_index,
    fuel_price_arbitrage_usd,

    # Fuel Inventory (10)
    fuel_inventory_level,
    fuel_inventory_level_mwh,
    fuel_inventory_days,
    fuel_inventory_min_level_kg,
    fuel_inventory_max_level_kg,
    fuel_inventory_utilization_percent,
    fuel_inventory_reorder_point_kg,
    fuel_inventory_safety_stock_kg,
    fuel_inventory_value_usd,
    fuel_inventory_adjustments_total,

    # Fuel Consumption (10)
    fuel_consumption_rate_kg_hr,
    fuel_consumption_rate_mwh,
    fuel_consumption_daily_kg,
    fuel_consumption_monthly_kg,
    fuel_consumption_ytd_kg,
    fuel_consumption_forecast_kg,
    fuel_consumption_variance_percent,
    fuel_consumption_per_mwh,
    fuel_consumption_total,
    fuel_consumption_efficiency_percent,

    # Fuel Quality (10)
    fuel_quality_score,
    fuel_quality_sulfur_percent,
    fuel_quality_ash_percent,
    fuel_quality_moisture_percent,
    fuel_quality_calorific_mj_kg,
    fuel_quality_volatiles_percent,
    fuel_quality_nitrogen_percent,
    fuel_quality_hardgrove_index,
    fuel_quality_tests_total,
    fuel_quality_violations_total,

    # Fuel Blend (5)
    fuel_blend_ratio,
    fuel_blend_target_ratio,
    fuel_blend_deviation_percent,
    fuel_blend_changes_total,
    fuel_blend_optimization_score,

    # Fuel Supplier (5)
    fuel_supplier_reliability_score,
    fuel_supplier_lead_time_days,
    fuel_delivery_delays_total,
    fuel_delivery_on_time_percent,
    fuel_contract_utilization_percent,

    # Optimization (30)
    optimization_execution_time_seconds,
    optimization_iterations,
    optimization_objective_value,
    optimization_constraints_violated,
    optimization_constraints_active,
    optimization_gap_percent,
    optimization_fuel_switches_total,
    optimization_cost_savings_usd,
    optimization_cost_savings_cumulative_usd,
    optimization_carbon_reduction_kg,
    optimization_reliability_score,
    optimization_requests_total,
    optimization_in_progress,
    optimization_errors_total,
    optimization_timeouts_total,
    optimization_retries_total,
    optimization_queue_depth,
    optimization_score,
    optimization_fuel_mix_ratio,
    optimization_cost_usd,
    optimization_emissions_kg,
    optimization_renewable_share,
    optimization_efficiency_percent,
    optimization_solution_time_seconds,
    optimization_model_variables,
    optimization_model_constraints,
    optimization_sensitivity_analysis_total,
    optimization_scenario_comparisons_total,
    optimization_recommendations_total,
    optimization_recommendations_accepted,

    # Integration (25)
    integration_api_calls_total,
    integration_api_latency_seconds,
    integration_api_errors_total,
    integration_erp_sync_lag_seconds,
    integration_erp_sync_duration_seconds,
    integration_erp_sync_status,
    integration_market_data_age_seconds,
    integration_market_data_updates_total,
    integration_storage_connector_health,
    integration_procurement_sync_success,
    integration_connection_status,
    integration_retry_total,
    integration_circuit_breaker_state,
    integration_requests_total,
    integration_latency_seconds,
    integration_errors_total,
    integration_scada_connection_status,
    integration_scada_data_points_total,
    integration_historian_query_duration_seconds,
    integration_historian_queries_total,
    integration_weather_api_calls_total,
    integration_weather_data_age_seconds,
    integration_supplier_portal_status,
    integration_logistics_api_status,
    integration_trading_platform_status,

    # Calculator (25)
    calculator_execution_time_seconds,
    calculator_cache_hit_rate,
    calculator_errors_total,
    calculator_provenance_hash,
    calculator_input_validation_failures_total,
    calorific_value_lookups_total,
    emissions_factor_lookups_total,
    cost_optimization_solutions_total,
    blending_calculations_total,
    calculator_invocations_total,
    calculator_duration_seconds,
    carbon_footprint_kg,
    carbon_intensity_kg_mwh,
    blending_quality_score,
    cost_per_mwh_usd,
    eoq_quantity_kg,
    emission_factor_co2,
    heating_value_mj_kg,
    calculator_retries_total,
    provenance_hashes_generated_total,
    heat_rate_btu_kwh,
    thermal_efficiency_percent,
    combustion_efficiency_percent,
    mass_balance_calculations_total,
    stoichiometric_calculations_total,

    # Business (20)
    fuel_cost_total_usd,
    fuel_cost_avoided_usd,
    carbon_emissions_total_kg,
    carbon_emissions_avoided_kg,
    energy_delivered_mwh,
    fuel_efficiency_percent,
    procurement_orders_total,
    procurement_value_usd,
    inventory_turnover_ratio,
    contract_compliance_score,
    fuel_cost_savings_usd,
    emissions_reduction_kg,
    renewable_energy_mwh,
    fuel_inventory_kg,
    budget_utilization_percent,
    cost_variance_percent,
    roi_percent,
    payback_period_months,
    energy_cost_per_unit,
    carbon_cost_usd,

    # Compliance (15)
    compliance_fuel_quality_status,
    compliance_emissions_status,
    compliance_violations_total,
    compliance_reporting_lag_seconds,
    compliance_audit_score,
    compliance_margin_percent,
    compliance_deadline_days,
    compliance_permit_expiration_days,
    compliance_inspections_total,
    compliance_corrective_actions_total,
    compliance_certificate_status,
    compliance_sulfur_limit_percent,
    compliance_ash_limit_percent,
    compliance_nox_limit_percent,
    compliance_documentation_status,

    # Health (10)
    agent_health_status,
    agent_uptime_seconds,
    agent_version_info,
    agent_restarts_total,
    agent_memory_usage_bytes,
    agent_cpu_usage_percent,
    agent_thread_count,
    agent_cache_size_bytes,
    agent_start_time_seconds,
    agent_last_heartbeat_timestamp,

    # Alerts (8)
    alerts_active,
    alerts_fired_total,
    alerts_resolved_total,
    alerts_mean_time_to_resolution_seconds,
    alerts_acknowledged_total,
    alerts_escalated_total,
    alert_response_time_seconds,
    alert_notification_failures_total,

    # Performance (5)
    request_duration_seconds,
    request_size_bytes,
    response_size_bytes,
    concurrent_requests,
    queue_depth,

    # Cache (8)
    cache_hits_total,
    cache_misses_total,
    cache_size,
    cache_memory_bytes,
    cache_evictions_total,
    cache_hit_ratio,
    cache_ttl_expirations_total,
    cache_update_duration_seconds,

    # Database (7)
    db_connection_pool_size,
    db_connections_active,
    db_query_duration_seconds,
    db_queries_total,
    db_errors_total,
    db_transactions_total,
    db_connection_errors_total,

    # Batch (5)
    batch_jobs_total,
    batch_job_duration_seconds,
    batch_records_processed_total,
    batch_job_errors_total,
    batch_job_queue_size,

    # Audit (4)
    audit_events_total,
    audit_trail_size_bytes,
    data_lineage_records_total,
    provenance_verification_total,

    # Reporting (8)
    reports_generated_total,
    reports_submitted_total,
    report_generation_duration_seconds,
    report_errors_total,
    report_queue_size,
    report_deadline_days,
    report_validation_errors_total,
    report_distribution_total,
]

# Count of metrics for verification
METRICS_COUNT = len(ALL_METRICS)

# Verify we have 178 metrics as required
assert METRICS_COUNT == 178, f"Expected 178 metrics, got {METRICS_COUNT}"

logger.info(f"GL-011 FUELCRAFT metrics module loaded: {METRICS_COUNT} metrics defined")
