import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ScenarioDefinition, ScenarioResult, SensitivityResult, StrandingDataPoint, ScenarioParameter } from '../../types';
import { scenarioApi } from '../../services/api';
import type { RootState } from '../index';

interface ScenarioState {
  scenarios: ScenarioDefinition[];
  selectedScenarioIds: string[];
  results: ScenarioResult[];
  comparisonResults: ScenarioResult[];
  sensitivityResults: SensitivityResult[];
  strandingData: StrandingDataPoint[];
  parameters: ScenarioParameter[];
  loading: boolean;
  runningScenario: boolean;
  error: string | null;
}

const initialState: ScenarioState = {
  scenarios: [],
  selectedScenarioIds: [],
  results: [],
  comparisonResults: [],
  sensitivityResults: [],
  strandingData: [],
  parameters: [],
  loading: false,
  runningScenario: false,
  error: null,
};

export const fetchScenarios = createAsyncThunk(
  'scenario/fetchScenarios',
  async (orgId: string) => scenarioApi.getScenarios(orgId)
);

export const runScenario = createAsyncThunk(
  'scenario/runScenario',
  async ({ id, params }: { id: string; params?: Record<string, number> }) => scenarioApi.runScenario(id, params)
);

export const fetchResults = createAsyncThunk(
  'scenario/fetchResults',
  async (scenarioId: string) => scenarioApi.getResults(scenarioId)
);

export const compareScenarios = createAsyncThunk(
  'scenario/compareScenarios',
  async (scenarioIds: string[]) => scenarioApi.compareScenarios(scenarioIds)
);

export const fetchSensitivity = createAsyncThunk(
  'scenario/fetchSensitivity',
  async (scenarioId: string) => scenarioApi.getSensitivity(scenarioId)
);

export const fetchStrandingTimeline = createAsyncThunk(
  'scenario/fetchStrandingTimeline',
  async (orgId: string) => scenarioApi.getStrandingTimeline(orgId)
);

export const fetchParameters = createAsyncThunk(
  'scenario/fetchParameters',
  async (scenarioType: string) => scenarioApi.getParameters(scenarioType)
);

export const createScenario = createAsyncThunk(
  'scenario/createScenario',
  async (data: Partial<ScenarioDefinition>) => scenarioApi.createScenario(data)
);

const scenarioSlice = createSlice({
  name: 'scenario',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    toggleScenarioSelection(state, action: PayloadAction<string>) {
      const id = action.payload;
      const idx = state.selectedScenarioIds.indexOf(id);
      if (idx >= 0) {
        state.selectedScenarioIds.splice(idx, 1);
      } else {
        state.selectedScenarioIds.push(id);
      }
    },
    setSelectedScenarios(state, action: PayloadAction<string[]>) {
      state.selectedScenarioIds = action.payload;
    },
    updateParameterValue(state, action: PayloadAction<{ id: string; value: number }>) {
      const param = state.parameters.find((p) => p.id === action.payload.id);
      if (param) param.current_value = action.payload.value;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchScenarios.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchScenarios.fulfilled, (state, action: PayloadAction<ScenarioDefinition[]>) => {
        state.loading = false;
        state.scenarios = action.payload;
      })
      .addCase(fetchScenarios.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch scenarios';
      })
      .addCase(runScenario.pending, (state) => { state.runningScenario = true; })
      .addCase(runScenario.fulfilled, (state, action: PayloadAction<ScenarioResult[]>) => {
        state.runningScenario = false;
        state.results = action.payload;
      })
      .addCase(runScenario.rejected, (state, action) => {
        state.runningScenario = false;
        state.error = action.error.message || 'Scenario run failed';
      })
      .addCase(fetchResults.fulfilled, (state, action: PayloadAction<ScenarioResult[]>) => {
        state.results = action.payload;
      })
      .addCase(compareScenarios.fulfilled, (state, action: PayloadAction<ScenarioResult[]>) => {
        state.comparisonResults = action.payload;
      })
      .addCase(fetchSensitivity.fulfilled, (state, action: PayloadAction<SensitivityResult[]>) => {
        state.sensitivityResults = action.payload;
      })
      .addCase(fetchStrandingTimeline.fulfilled, (state, action: PayloadAction<StrandingDataPoint[]>) => {
        state.strandingData = action.payload;
      })
      .addCase(fetchParameters.fulfilled, (state, action: PayloadAction<ScenarioParameter[]>) => {
        state.parameters = action.payload;
      })
      .addCase(createScenario.fulfilled, (state, action: PayloadAction<ScenarioDefinition>) => {
        state.scenarios.push(action.payload);
      });
  },
});

export const { clearError, toggleScenarioSelection, setSelectedScenarios, updateParameterValue } = scenarioSlice.actions;
export const selectScenarios = (state: RootState) => state.scenario.scenarios;
export const selectSelectedScenarioIds = (state: RootState) => state.scenario.selectedScenarioIds;
export const selectScenarioResults = (state: RootState) => state.scenario.results;
export const selectComparisonResults = (state: RootState) => state.scenario.comparisonResults;
export const selectSensitivityResults = (state: RootState) => state.scenario.sensitivityResults;
export const selectStrandingData = (state: RootState) => state.scenario.strandingData;
export const selectParameters = (state: RootState) => state.scenario.parameters;
export const selectScenarioLoading = (state: RootState) => state.scenario.loading;
export const selectRunningScenario = (state: RootState) => state.scenario.runningScenario;
export default scenarioSlice.reducer;
