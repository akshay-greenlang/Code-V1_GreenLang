import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { DNSHAssessment, DNSHMatrix, ClimateRiskAssessment, EnvironmentalObjective } from '../../types';
import { dnshApi } from '../../services/api';
import type { RootState } from '../index';

interface DNSHState {
  assessment: DNSHAssessment | null;
  matrix: DNSHMatrix[];
  climateRisk: ClimateRiskAssessment | null;
  loading: boolean;
  error: string | null;
}

const initialState: DNSHState = {
  assessment: null,
  matrix: [],
  climateRisk: null,
  loading: false,
  error: null,
};

export const assessDNSH = createAsyncThunk(
  'dnsh/assess',
  async ({ activityId, scObjective }: { activityId: string; scObjective: EnvironmentalObjective }) =>
    dnshApi.assess(activityId, scObjective)
);

export const assessClimateRisk = createAsyncThunk(
  'dnsh/climateRisk',
  async (activityId: string) => dnshApi.climateRisk(activityId)
);

export const fetchDNSHMatrix = createAsyncThunk(
  'dnsh/matrix',
  async (orgId: string) => dnshApi.getMatrix(orgId)
);

const dnshSlice = createSlice({
  name: 'dnsh',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(assessDNSH.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(assessDNSH.fulfilled, (state, action: PayloadAction<DNSHAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
      })
      .addCase(assessDNSH.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to assess DNSH';
      })
      .addCase(assessClimateRisk.fulfilled, (state, action) => { state.climateRisk = action.payload; })
      .addCase(fetchDNSHMatrix.fulfilled, (state, action) => { state.matrix = action.payload; });
  },
});

export const { clearError } = dnshSlice.actions;
export const selectDNSHAssessment = (state: RootState) => state.dnsh.assessment;
export const selectDNSHMatrix = (state: RootState) => state.dnsh.matrix;
export const selectClimateRisk = (state: RootState) => state.dnsh.climateRisk;
export const selectDNSHLoading = (state: RootState) => state.dnsh.loading;
export default dnshSlice.reducer;
