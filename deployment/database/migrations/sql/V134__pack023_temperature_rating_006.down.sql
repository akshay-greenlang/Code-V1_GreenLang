-- =============================================================================
-- V134 Down: Rollback Temperature Rating Results and Portfolio Scores
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_scen_updated_at ON pack023_sbti_alignment.pack023_temperature_scenario_comparison;
DROP TRIGGER IF EXISTS trg_pk_hold_updated_at ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings;
DROP TRIGGER IF EXISTS trg_pk_port_updated_at ON pack023_sbti_alignment.pack023_temperature_portfolio_scores;
DROP TRIGGER IF EXISTS trg_pk_temp_updated_at ON pack023_sbti_alignment.pack023_temperature_company_scores;

-- Drop hypertables
SELECT drop_chunks('pack023_sbti_alignment.pack023_temperature_company_scores', INTERVAL '1 day');
SELECT detach_hypertable('pack023_sbti_alignment.pack023_temperature_company_scores', if_exists => TRUE);

SELECT drop_chunks('pack023_sbti_alignment.pack023_temperature_portfolio_scores', INTERVAL '1 day');
SELECT detach_hypertable('pack023_sbti_alignment.pack023_temperature_portfolio_scores', if_exists => TRUE);

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_temperature_scenario_comparison;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_temperature_portfolio_holdings;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_temperature_portfolio_scores;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_temperature_company_scores;
