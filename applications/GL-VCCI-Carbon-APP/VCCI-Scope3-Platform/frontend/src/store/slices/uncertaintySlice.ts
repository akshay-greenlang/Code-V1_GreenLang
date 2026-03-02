/**
 * GL-VCCI Frontend - Uncertainty Analysis Redux Slice
 *
 * Manages state for Monte Carlo simulation results, sensitivity analysis,
 * scenario comparisons, and distribution data for the uncertainty analysis
 * workflow within the Scope 3 platform.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';

// ==============================================================================
// Type Definitions
// ==============================================================================

export interface DistributionPercentiles {
  p5: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
}

export interface MonteCarloResult {
  id: string;
  mean: number;
  median: number;
  stdDev: number;
  cv: number;
  skewness: number;
  kurtosis: number;
  percentiles: DistributionPercentiles;
  iterations: number;
  convergenceAchieved: boolean;
  convergenceThreshold: number;
  computationTimeMs: number;
  dataTier: 1 | 2 | 3;
  unit: string;
  category?: string;
  distributionSamples: number[];
  createdAt: string;
}

export interface SensitivityParameter {
  name: string;
  sobolIndex: number;
  lowValue: number;
  highValue: number;
  baseValue: number;
  category?: string;
}

export interface ScenarioResult {
  id: string;
  name: string;
  mean: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  color?: string;
  description?: string;
}

export interface ConfidenceIntervalDataPoint {
  period: string;
  mean: number;
  p5: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
}

// ==============================================================================
// State Interface
// ==============================================================================

interface UncertaintyState {
  analysisResult: MonteCarloResult | null;
  scenarioResults: ScenarioResult[];
  sensitivityData: SensitivityParameter[];
  distributionData: number[];
  confidenceIntervalData: ConfidenceIntervalDataPoint[];
  loading: boolean;
  error: string | null;
  selectedCategory: number | null;
  confidenceLevel: 90 | 95 | 99;
  lastUpdated: string | null;
}

const initialState: UncertaintyState = {
  analysisResult: null,
  scenarioResults: [],
  sensitivityData: [],
  distributionData: [],
  confidenceIntervalData: [],
  loading: false,
  error: null,
  selectedCategory: null,
  confidenceLevel: 95,
  lastUpdated: null,
};

// ==============================================================================
// Async Thunks
// ==============================================================================

export const fetchUncertaintyAnalysis = createAsyncThunk(
  'uncertainty/fetchAnalysis',
  async ({ category, period }: { category?: number; period?: string } = {}) => {
    const response = await api.getUncertaintyAnalysis({ category, period });
    return response;
  }
);

export const runMonteCarloSimulation = createAsyncThunk(
  'uncertainty/runMonteCarlo',
  async ({
    category,
    iterations,
    confidenceLevel,
  }: {
    category?: number;
    iterations?: number;
    confidenceLevel?: number;
  }) => {
    const response = await api.runMonteCarloSimulation({
      category,
      iterations: iterations || 10000,
      confidence_level: confidenceLevel || 95,
    });
    return response;
  }
);

export const fetchSensitivityAnalysis = createAsyncThunk(
  'uncertainty/fetchSensitivity',
  async ({ category, topN }: { category?: number; topN?: number } = {}) => {
    const response = await api.getSensitivityAnalysis({ category, top_n: topN || 10 });
    return response;
  }
);

export const compareScenarios = createAsyncThunk(
  'uncertainty/compareScenarios',
  async ({ scenarioIds }: { scenarioIds: string[] }) => {
    const response = await api.compareScenarios({ scenario_ids: scenarioIds });
    return response;
  }
);

// ==============================================================================
// Slice
// ==============================================================================

const uncertaintySlice = createSlice({
  name: 'uncertainty',
  initialState,
  reducers: {
    setSelectedCategory: (state, action: PayloadAction<number | null>) => {
      state.selectedCategory = action.payload;
    },
    setConfidenceLevel: (state, action: PayloadAction<90 | 95 | 99>) => {
      state.confidenceLevel = action.payload;
    },
    clearUncertaintyError: (state) => {
      state.error = null;
    },
    clearUncertaintyState: (state) => {
      state.analysisResult = null;
      state.scenarioResults = [];
      state.sensitivityData = [];
      state.distributionData = [];
      state.confidenceIntervalData = [];
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch uncertainty analysis
    builder
      .addCase(fetchUncertaintyAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUncertaintyAnalysis.fulfilled, (state, action) => {
        state.loading = false;
        state.analysisResult = action.payload.result;
        state.distributionData = action.payload.result?.distributionSamples || [];
        state.confidenceIntervalData = action.payload.confidenceIntervalData || [];
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchUncertaintyAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch uncertainty analysis';
      });

    // Run Monte Carlo simulation
    builder
      .addCase(runMonteCarloSimulation.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(runMonteCarloSimulation.fulfilled, (state, action) => {
        state.loading = false;
        state.analysisResult = action.payload;
        state.distributionData = action.payload.distributionSamples || [];
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(runMonteCarloSimulation.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Monte Carlo simulation failed';
      });

    // Fetch sensitivity analysis
    builder
      .addCase(fetchSensitivityAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSensitivityAnalysis.fulfilled, (state, action) => {
        state.loading = false;
        state.sensitivityData = action.payload.parameters || [];
      })
      .addCase(fetchSensitivityAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch sensitivity analysis';
      });

    // Compare scenarios
    builder
      .addCase(compareScenarios.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(compareScenarios.fulfilled, (state, action) => {
        state.loading = false;
        state.scenarioResults = action.payload.scenarios || [];
      })
      .addCase(compareScenarios.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Scenario comparison failed';
      });
  },
});

export const {
  setSelectedCategory,
  setConfidenceLevel,
  clearUncertaintyError,
  clearUncertaintyState,
} = uncertaintySlice.actions;

export default uncertaintySlice.reducer;
