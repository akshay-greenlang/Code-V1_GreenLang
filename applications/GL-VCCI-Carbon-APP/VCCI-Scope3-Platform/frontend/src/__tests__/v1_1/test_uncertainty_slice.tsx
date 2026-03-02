/**
 * APP-003 GL-VCCI-APP v1.1 -- Uncertainty Redux Slice Tests
 *
 * Tests the uncertaintySlice initial state, all 4 async thunks
 * (pending / fulfilled / rejected paths), synchronous reducers,
 * and state shape after mutations.
 *
 * Target: 15+ tests, ~200 lines
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { configureStore } from '@reduxjs/toolkit';

import uncertaintyReducer, {
  fetchUncertaintyAnalysis,
  runMonteCarloSimulation,
  fetchSensitivityAnalysis,
  compareScenarios,
  setSelectedCategory,
  setConfidenceLevel,
  clearUncertaintyError,
  clearUncertaintyState,
} from '../../store/slices/uncertaintySlice';

// ============================================================================
// Mock the API service
// ============================================================================

const mockApi = {
  getUncertaintyAnalysis: vi.fn(),
  runMonteCarloSimulation: vi.fn(),
  getSensitivityAnalysis: vi.fn(),
  compareScenarios: vi.fn(),
};

vi.mock('../../services/api', () => ({
  default: {
    getUncertaintyAnalysis: (...args: any[]) => mockApi.getUncertaintyAnalysis(...args),
    runMonteCarloSimulation: (...args: any[]) => mockApi.runMonteCarloSimulation(...args),
    getSensitivityAnalysis: (...args: any[]) => mockApi.getSensitivityAnalysis(...args),
    compareScenarios: (...args: any[]) => mockApi.compareScenarios(...args),
  },
}));

// ============================================================================
// Helpers
// ============================================================================

function createStore() {
  return configureStore({ reducer: { uncertainty: uncertaintyReducer } });
}

const mockMCResult = {
  id: 'mc-001',
  mean: 1250.5,
  median: 1230.2,
  stdDev: 180.3,
  cv: 0.144,
  skewness: 0.32,
  kurtosis: 2.89,
  percentiles: { p5: 960, p10: 1020, p25: 1120, p50: 1230, p75: 1370, p90: 1480, p95: 1590 },
  iterations: 10000,
  convergenceAchieved: true,
  convergenceThreshold: 0.01,
  computationTimeMs: 2340,
  dataTier: 2,
  unit: 'tCO2e',
  distributionSamples: [1100, 1200, 1300, 1400],
  createdAt: '2026-02-28T12:00:00Z',
};

// ============================================================================
// Tests
// ============================================================================

describe('uncertaintySlice', () => {

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---------- Initial state ----------

  it('has the correct initial state', () => {
    const store = createStore();
    const state = store.getState().uncertainty;

    expect(state.analysisResult).toBeNull();
    expect(state.scenarioResults).toEqual([]);
    expect(state.sensitivityData).toEqual([]);
    expect(state.distributionData).toEqual([]);
    expect(state.confidenceIntervalData).toEqual([]);
    expect(state.loading).toBe(false);
    expect(state.error).toBeNull();
    expect(state.selectedCategory).toBeNull();
    expect(state.confidenceLevel).toBe(95);
    expect(state.lastUpdated).toBeNull();
  });

  // ---------- Synchronous reducers ----------

  it('setSelectedCategory updates the selected category', () => {
    const store = createStore();
    store.dispatch(setSelectedCategory(3));
    expect(store.getState().uncertainty.selectedCategory).toBe(3);
  });

  it('setSelectedCategory can reset to null', () => {
    const store = createStore();
    store.dispatch(setSelectedCategory(5));
    store.dispatch(setSelectedCategory(null));
    expect(store.getState().uncertainty.selectedCategory).toBeNull();
  });

  it('setConfidenceLevel updates the confidence level', () => {
    const store = createStore();
    store.dispatch(setConfidenceLevel(90));
    expect(store.getState().uncertainty.confidenceLevel).toBe(90);
  });

  it('clearUncertaintyError clears the error', () => {
    const store = createStore();
    // Force an error via a rejected thunk
    mockApi.getUncertaintyAnalysis.mockRejectedValueOnce(new Error('Network error'));
    store.dispatch(fetchUncertaintyAnalysis({})).then(() => {
      store.dispatch(clearUncertaintyError());
      expect(store.getState().uncertainty.error).toBeNull();
    });
  });

  it('clearUncertaintyState resets analysis data', () => {
    const store = createStore();
    // Manually simulate having data then clearing
    store.dispatch(setSelectedCategory(7));
    store.dispatch(clearUncertaintyState());

    const state = store.getState().uncertainty;
    expect(state.analysisResult).toBeNull();
    expect(state.scenarioResults).toEqual([]);
    expect(state.sensitivityData).toEqual([]);
    expect(state.distributionData).toEqual([]);
    expect(state.confidenceIntervalData).toEqual([]);
    expect(state.error).toBeNull();
  });

  // ---------- fetchUncertaintyAnalysis ----------

  it('fetchUncertaintyAnalysis.pending sets loading to true', async () => {
    mockApi.getUncertaintyAnalysis.mockResolvedValueOnce({ result: null, confidenceIntervalData: [] });
    const store = createStore();

    const promise = store.dispatch(fetchUncertaintyAnalysis({}));
    // Check synchronously right after dispatch
    expect(store.getState().uncertainty.loading).toBe(true);
    await promise;
  });

  it('fetchUncertaintyAnalysis.fulfilled stores the result', async () => {
    mockApi.getUncertaintyAnalysis.mockResolvedValueOnce({
      result: mockMCResult,
      confidenceIntervalData: [{ period: 'Q1', mean: 1100 }],
    });
    const store = createStore();

    await store.dispatch(fetchUncertaintyAnalysis({}));
    const state = store.getState().uncertainty;

    expect(state.loading).toBe(false);
    expect(state.analysisResult).toEqual(mockMCResult);
    expect(state.distributionData).toEqual(mockMCResult.distributionSamples);
    expect(state.confidenceIntervalData).toHaveLength(1);
    expect(state.lastUpdated).not.toBeNull();
  });

  it('fetchUncertaintyAnalysis.rejected sets error', async () => {
    mockApi.getUncertaintyAnalysis.mockRejectedValueOnce(new Error('Server down'));
    const store = createStore();

    await store.dispatch(fetchUncertaintyAnalysis({}));
    const state = store.getState().uncertainty;

    expect(state.loading).toBe(false);
    expect(state.error).toBe('Server down');
  });

  // ---------- runMonteCarloSimulation ----------

  it('runMonteCarloSimulation.fulfilled stores simulation result', async () => {
    mockApi.runMonteCarloSimulation.mockResolvedValueOnce(mockMCResult);
    const store = createStore();

    await store.dispatch(runMonteCarloSimulation({ category: 1, iterations: 5000 }));
    const state = store.getState().uncertainty;

    expect(state.loading).toBe(false);
    expect(state.analysisResult).toEqual(mockMCResult);
    expect(state.distributionData).toEqual(mockMCResult.distributionSamples);
    expect(state.lastUpdated).not.toBeNull();
  });

  it('runMonteCarloSimulation.rejected sets error message', async () => {
    mockApi.runMonteCarloSimulation.mockRejectedValueOnce(new Error('Timeout'));
    const store = createStore();

    await store.dispatch(runMonteCarloSimulation({ category: 1 }));
    expect(store.getState().uncertainty.error).toBe('Timeout');
  });

  // ---------- fetchSensitivityAnalysis ----------

  it('fetchSensitivityAnalysis.fulfilled stores parameters', async () => {
    const mockParams = [
      { name: 'Emission Factor', sobolIndex: 0.45, lowValue: 970, highValue: 1530, baseValue: 1250 },
    ];
    mockApi.getSensitivityAnalysis.mockResolvedValueOnce({ parameters: mockParams });
    const store = createStore();

    await store.dispatch(fetchSensitivityAnalysis({}));
    expect(store.getState().uncertainty.sensitivityData).toEqual(mockParams);
  });

  it('fetchSensitivityAnalysis.rejected sets error', async () => {
    mockApi.getSensitivityAnalysis.mockRejectedValueOnce(new Error('Bad request'));
    const store = createStore();

    await store.dispatch(fetchSensitivityAnalysis({}));
    expect(store.getState().uncertainty.error).toBe('Bad request');
  });

  // ---------- compareScenarios ----------

  it('compareScenarios.fulfilled stores scenario list', async () => {
    const mockScenarios = [
      { id: 's1', name: 'Baseline', mean: 1250, p5: 960, p25: 1120, p50: 1230, p75: 1370, p95: 1590 },
      { id: 's2', name: 'Optimistic', mean: 1050, p5: 800, p25: 940, p50: 1030, p75: 1160, p95: 1340 },
    ];
    mockApi.compareScenarios.mockResolvedValueOnce({ scenarios: mockScenarios });
    const store = createStore();

    await store.dispatch(compareScenarios({ scenarioIds: ['s1', 's2'] }));
    expect(store.getState().uncertainty.scenarioResults).toEqual(mockScenarios);
  });

  it('compareScenarios.rejected sets error', async () => {
    mockApi.compareScenarios.mockRejectedValueOnce(new Error('Scenario not found'));
    const store = createStore();

    await store.dispatch(compareScenarios({ scenarioIds: ['bad-id'] }));
    expect(store.getState().uncertainty.error).toBe('Scenario not found');
  });

  // ---------- Loading flag isolation ----------

  it('each thunk independently sets and clears loading', async () => {
    mockApi.getUncertaintyAnalysis.mockResolvedValueOnce({ result: null, confidenceIntervalData: [] });
    mockApi.getSensitivityAnalysis.mockResolvedValueOnce({ parameters: [] });
    const store = createStore();

    const p1 = store.dispatch(fetchUncertaintyAnalysis({}));
    expect(store.getState().uncertainty.loading).toBe(true);
    await p1;
    expect(store.getState().uncertainty.loading).toBe(false);

    const p2 = store.dispatch(fetchSensitivityAnalysis({}));
    expect(store.getState().uncertainty.loading).toBe(true);
    await p2;
    expect(store.getState().uncertainty.loading).toBe(false);
  });
});
