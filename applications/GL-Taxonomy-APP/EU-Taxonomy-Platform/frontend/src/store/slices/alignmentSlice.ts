import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { AlignmentResult, PortfolioAlignment, AlignmentProgress, AlignmentFunnelData, EnvironmentalObjective, BatchAlignmentRequest } from '../../types';
import { alignmentApi } from '../../services/api';
import type { RootState } from '../index';

interface AlignmentState {
  results: AlignmentResult[];
  currentResult: AlignmentResult | null;
  portfolio: PortfolioAlignment | null;
  progress: AlignmentProgress[];
  funnel: AlignmentFunnelData | null;
  loading: boolean;
  error: string | null;
}

const initialState: AlignmentState = {
  results: [],
  currentResult: null,
  portfolio: null,
  progress: [],
  funnel: null,
  loading: false,
  error: null,
};

export const runFullAlignment = createAsyncThunk(
  'alignment/full',
  async ({ activityId, objective }: { activityId: string; objective: EnvironmentalObjective }) =>
    alignmentApi.full(activityId, objective)
);

export const fetchPortfolioAlignment = createAsyncThunk(
  'alignment/portfolio',
  async (orgId: string) => alignmentApi.portfolio(orgId)
);

export const batchAlignment = createAsyncThunk(
  'alignment/batch',
  async (request: BatchAlignmentRequest) => alignmentApi.batch(request)
);

export const fetchAlignmentProgress = createAsyncThunk(
  'alignment/progress',
  async (orgId: string) => alignmentApi.progress(orgId)
);

export const fetchAlignmentFunnel = createAsyncThunk(
  'alignment/funnel',
  async (orgId: string) => alignmentApi.eligibleVsAligned(orgId)
);

const alignmentSlice = createSlice({
  name: 'alignment',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(runFullAlignment.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(runFullAlignment.fulfilled, (state, action: PayloadAction<AlignmentResult>) => {
        state.loading = false;
        state.currentResult = action.payload;
      })
      .addCase(runFullAlignment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to run alignment';
      })
      .addCase(fetchPortfolioAlignment.fulfilled, (state, action) => { state.portfolio = action.payload; })
      .addCase(batchAlignment.fulfilled, (state, action) => { state.results = action.payload; state.loading = false; })
      .addCase(fetchAlignmentProgress.fulfilled, (state, action) => { state.progress = action.payload; })
      .addCase(fetchAlignmentFunnel.fulfilled, (state, action) => { state.funnel = action.payload; });
  },
});

export const { clearError } = alignmentSlice.actions;
export const selectAlignmentResults = (state: RootState) => state.alignment.results;
export const selectCurrentAlignmentResult = (state: RootState) => state.alignment.currentResult;
export const selectPortfolioAlignment = (state: RootState) => state.alignment.portfolio;
export const selectAlignmentProgress = (state: RootState) => state.alignment.progress;
export const selectAlignmentFunnel = (state: RootState) => state.alignment.funnel;
export const selectAlignmentLoading = (state: RootState) => state.alignment.loading;
export default alignmentSlice.reducer;
