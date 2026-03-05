import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { GapAssessment, GapAction } from '../../types';
import { gapApi } from '../../services/api';
import type { RootState } from '../index';

interface GapState {
  assessment: GapAssessment | null;
  actions: GapAction[];
  peerComparison: { pillar: string; org_score: number; peer_avg: number; best_in_class: number }[];
  maturityTrend: { date: string; score: number; maturity: string }[];
  loading: boolean;
  error: string | null;
}

const initialState: GapState = {
  assessment: null,
  actions: [],
  peerComparison: [],
  maturityTrend: [],
  loading: false,
  error: null,
};

export const fetchGapAssessment = createAsyncThunk(
  'gap/fetchAssessment',
  async (orgId: string) => gapApi.getAssessment(orgId)
);

export const runGapAssessment = createAsyncThunk(
  'gap/runAssessment',
  async ({ orgId, framework }: { orgId: string; framework?: string }) =>
    gapApi.runAssessment(orgId, framework)
);

export const fetchGapActions = createAsyncThunk(
  'gap/fetchActions',
  async (orgId: string) => gapApi.getActions(orgId)
);

export const updateGapAction = createAsyncThunk(
  'gap/updateAction',
  async ({ id, data }: { id: string; data: Partial<GapAction> }) =>
    gapApi.updateAction(id, data)
);

export const fetchPeerComparison = createAsyncThunk(
  'gap/fetchPeerComparison',
  async (orgId: string) => gapApi.getPeerComparison(orgId)
);

export const fetchMaturityTrend = createAsyncThunk(
  'gap/fetchMaturityTrend',
  async (orgId: string) => gapApi.getMaturityTrend(orgId)
);

const gapSlice = createSlice({
  name: 'gap',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchGapAssessment.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchGapAssessment.fulfilled, (state, action: PayloadAction<GapAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
      })
      .addCase(fetchGapAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch gap assessment';
      })
      .addCase(runGapAssessment.fulfilled, (state, action: PayloadAction<GapAssessment>) => {
        state.assessment = action.payload;
      })
      .addCase(fetchGapActions.fulfilled, (state, action: PayloadAction<GapAction[]>) => {
        state.actions = action.payload;
      })
      .addCase(updateGapAction.fulfilled, (state, action: PayloadAction<GapAction>) => {
        const idx = state.actions.findIndex((a) => a.id === action.payload.id);
        if (idx >= 0) state.actions[idx] = action.payload;
      })
      .addCase(fetchPeerComparison.fulfilled, (state, action) => {
        state.peerComparison = action.payload;
      })
      .addCase(fetchMaturityTrend.fulfilled, (state, action) => {
        state.maturityTrend = action.payload;
      });
  },
});

export const { clearError } = gapSlice.actions;
export const selectGapAssessment = (state: RootState) => state.gap.assessment;
export const selectGapActions = (state: RootState) => state.gap.actions;
export const selectPeerComparison = (state: RootState) => state.gap.peerComparison;
export const selectMaturityTrend = (state: RootState) => state.gap.maturityTrend;
export const selectGapLoading = (state: RootState) => state.gap.loading;
export default gapSlice.reducer;
