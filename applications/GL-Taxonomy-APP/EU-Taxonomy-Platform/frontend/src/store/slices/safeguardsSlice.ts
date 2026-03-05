import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { SafeguardAssessment, DueDiligenceRecord, SafeguardTopic, AdverseFinding } from '../../types';
import { safeguardsApi } from '../../services/api';
import type { RootState } from '../index';

interface SafeguardsState {
  assessment: SafeguardAssessment | null;
  dueDiligence: DueDiligenceRecord[];
  loading: boolean;
  error: string | null;
}

const initialState: SafeguardsState = {
  assessment: null,
  dueDiligence: [],
  loading: false,
  error: null,
};

export const assessSafeguards = createAsyncThunk(
  'safeguards/assess',
  async (orgId: string) => safeguardsApi.assess(orgId)
);

export const fetchSafeguardResults = createAsyncThunk(
  'safeguards/results',
  async (orgId: string) => safeguardsApi.getResults(orgId)
);

export const fetchDueDiligence = createAsyncThunk(
  'safeguards/dueDiligence',
  async (orgId: string) => safeguardsApi.getDueDiligence(orgId)
);

export const recordFinding = createAsyncThunk(
  'safeguards/recordFinding',
  async ({ orgId, finding }: { orgId: string; finding: Partial<AdverseFinding> }) =>
    safeguardsApi.recordFinding(orgId, finding)
);

const safeguardsSlice = createSlice({
  name: 'safeguards',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(assessSafeguards.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(assessSafeguards.fulfilled, (state, action: PayloadAction<SafeguardAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
      })
      .addCase(assessSafeguards.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to assess safeguards';
      })
      .addCase(fetchSafeguardResults.fulfilled, (state, action) => { state.assessment = action.payload; })
      .addCase(fetchDueDiligence.fulfilled, (state, action) => { state.dueDiligence = action.payload; });
  },
});

export const { clearError } = safeguardsSlice.actions;
export const selectSafeguardAssessment = (state: RootState) => state.safeguards.assessment;
export const selectDueDiligence = (state: RootState) => state.safeguards.dueDiligence;
export const selectSafeguardsLoading = (state: RootState) => state.safeguards.loading;
export default safeguardsSlice.reducer;
