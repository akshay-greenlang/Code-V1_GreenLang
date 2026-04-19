/**
 * UncertaintyAnalysis - Full page for Monte Carlo uncertainty analysis
 *
 * Provides an integrated view of Monte Carlo simulation results for Scope 3
 * emissions. Layout includes a top row of uncertainty metric cards, a full-width
 * confidence interval chart, and a bottom section with the tabbed MC results
 * panel alongside a scenario comparison chart. Supports category filtering,
 * triggering new simulations, and loading/error states.
 */

import React, { useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Grid,
  Alert,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Paper,
  Divider,
} from '@mui/material';
import {
  CloudQueue as EmissionsIcon,
  Speed as Scope1Icon,
  Bolt as Scope2Icon,
  LocalShipping as Scope3Icon,
  PlayArrow,
  Refresh,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchUncertaintyAnalysis,
  runMonteCarloSimulation,
  fetchSensitivityAnalysis,
  setSelectedCategory,
  setConfidenceLevel,
} from '../store/slices/uncertaintySlice';
import {
  UncertaintyMetricsCard,
  ConfidenceIntervalChart,
  MonteCarloResultsPanel,
  ScenarioComparisonChart,
} from '../components/uncertainty';
import LoadingSpinner from '../components/LoadingSpinner';
import { GHG_CATEGORIES } from '../types';

// ==============================================================================
// Component
// ==============================================================================

const UncertaintyAnalysis: React.FC = () => {
  const dispatch = useAppDispatch();

  const {
    analysisResult,
    scenarioResults,
    sensitivityData,
    confidenceIntervalData,
    loading,
    error,
    selectedCategory,
    confidenceLevel,
    lastUpdated,
  } = useAppSelector((state) => state.uncertainty);

  // Fetch initial data on mount
  useEffect(() => {
    dispatch(fetchUncertaintyAnalysis({ category: selectedCategory ?? undefined }));
    dispatch(fetchSensitivityAnalysis({ category: selectedCategory ?? undefined }));
  }, [dispatch, selectedCategory]);

  // Handle "Run Analysis" button
  const handleRunAnalysis = useCallback(() => {
    dispatch(
      runMonteCarloSimulation({
        category: selectedCategory ?? undefined,
        iterations: 10000,
        confidenceLevel,
      })
    );
    dispatch(fetchSensitivityAnalysis({ category: selectedCategory ?? undefined }));
  }, [dispatch, selectedCategory, confidenceLevel]);

  // Handle category change
  const handleCategoryChange = useCallback(
    (value: string) => {
      const category = value === 'all' ? null : parseInt(value, 10);
      dispatch(setSelectedCategory(category));
    },
    [dispatch]
  );

  // Handle refresh
  const handleRefresh = useCallback(() => {
    dispatch(fetchUncertaintyAnalysis({ category: selectedCategory ?? undefined }));
    dispatch(fetchSensitivityAnalysis({ category: selectedCategory ?? undefined }));
  }, [dispatch, selectedCategory]);

  // Loading state
  if (loading && !analysisResult) {
    return <LoadingSpinner message="Running uncertainty analysis..." />;
  }

  return (
    <Box>
      {/* Page Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Uncertainty Analysis
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Monte Carlo simulation results and sensitivity analysis for Scope 3 emissions
          </Typography>
          {lastUpdated && (
            <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5, display: 'block' }}>
              Last updated: {new Date(lastUpdated).toLocaleString()}
            </Typography>
          )}
        </Box>

        <Stack direction="row" spacing={2} alignItems="center">
          {/* Category selector */}
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>GHG Category</InputLabel>
            <Select
              value={selectedCategory != null ? String(selectedCategory) : 'all'}
              label="GHG Category"
              onChange={(e) => handleCategoryChange(e.target.value)}
            >
              <MenuItem value="all">All Categories (Aggregate)</MenuItem>
              {Object.entries(GHG_CATEGORIES).map(([key, name]) => (
                <MenuItem key={key} value={key}>
                  Cat {key}: {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Action buttons */}
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayArrow />}
            onClick={handleRunAnalysis}
            disabled={loading}
          >
            {loading ? 'Running...' : 'Run Analysis'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Stack>
      </Box>

      {/* Error state */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => {}}>
          {error}
        </Alert>
      )}

      {/* No data state */}
      {!analysisResult && !loading && !error && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No uncertainty analysis data available. Click "Run Analysis" to start a Monte Carlo
          simulation, or upload emissions data first.
        </Alert>
      )}

      {analysisResult && (
        <>
          {/* Top Row: Uncertainty Metric Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <UncertaintyMetricsCard
                title="Total Emissions"
                value={analysisResult.mean}
                uncertainty={analysisResult.stdDev}
                unit={analysisResult.unit}
                tier={analysisResult.dataTier}
                icon={EmissionsIcon}
                color="secondary"
                distributionSamples={analysisResult.distributionSamples.slice(0, 500)}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <UncertaintyMetricsCard
                title="Scope 1 Uncertainty"
                value={analysisResult.percentiles.p50}
                uncertainty={analysisResult.percentiles.p95 - analysisResult.percentiles.p5}
                unit={analysisResult.unit}
                tier={analysisResult.dataTier}
                icon={Scope1Icon}
                color="primary"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <UncertaintyMetricsCard
                title="Scope 2 Uncertainty"
                value={analysisResult.median}
                uncertainty={analysisResult.percentiles.p75 - analysisResult.percentiles.p25}
                unit={analysisResult.unit}
                tier={analysisResult.dataTier}
                icon={Scope2Icon}
                color="info"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <UncertaintyMetricsCard
                title="Scope 3 Uncertainty"
                value={analysisResult.mean}
                uncertainty={analysisResult.stdDev * 2}
                unit={analysisResult.unit}
                tier={analysisResult.dataTier}
                icon={Scope3Icon}
                color="warning"
              />
            </Grid>
          </Grid>

          {/* Middle Row: Confidence Interval Chart (full width) */}
          {confidenceIntervalData.length > 0 && (
            <Box sx={{ mb: 3 }}>
              <ConfidenceIntervalChart
                data={confidenceIntervalData}
                unit={analysisResult.unit}
                title={
                  selectedCategory != null
                    ? `Emissions Trend - Category ${selectedCategory}: ${GHG_CATEGORIES[selectedCategory as keyof typeof GHG_CATEGORIES] || 'Unknown'}`
                    : 'Aggregate Emissions Trend with Confidence Intervals'
                }
              />
            </Box>
          )}

          {/* Bottom Row: MC Results Panel + Scenario Comparison */}
          <Grid container spacing={3}>
            <Grid item xs={12} lg={7}>
              <MonteCarloResultsPanel
                result={analysisResult}
                sensitivityData={sensitivityData}
              />
            </Grid>
            <Grid item xs={12} lg={5}>
              {scenarioResults.length >= 2 ? (
                <ScenarioComparisonChart
                  scenarios={scenarioResults}
                  unit={analysisResult.unit}
                  title="Scenario Comparison"
                />
              ) : (
                <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                  <Typography variant="h6" color="textSecondary" gutterBottom>
                    Scenario Comparison
                  </Typography>
                  <Divider sx={{ width: '100%', mb: 2 }} />
                  <Typography color="textSecondary" textAlign="center">
                    Configure and run multiple scenarios to compare their uncertainty
                    distributions side by side. At least 2 scenarios are needed.
                  </Typography>
                  <Button
                    variant="outlined"
                    sx={{ mt: 2 }}
                    disabled
                  >
                    Coming Soon
                  </Button>
                </Paper>
              )}
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
};

export default UncertaintyAnalysis;
