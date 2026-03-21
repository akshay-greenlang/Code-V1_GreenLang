-- =============================================================================
-- V141 Rollback: Drop Portfolio Optimization Results
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_rebal_updated_at ON pack024_carbon_neutral.pack024_rebalancing_actions;
DROP TRIGGER IF EXISTS trg_pack024_rec_updated_at ON pack024_carbon_neutral.pack024_optimization_recommendations;
DROP TRIGGER IF EXISTS trg_pack024_scen_updated_at ON pack024_carbon_neutral.pack024_optimization_scenarios;
DROP TRIGGER IF EXISTS trg_pack024_opt_updated_at ON pack024_carbon_neutral.pack024_portfolio_optimization;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_rebalancing_actions CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_optimization_recommendations CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_optimization_scenarios CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_portfolio_optimization CASCADE;
