import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ProgressSummary, ProgressRecord, VarianceAnalysis } from '../../types';
import { progressApi } from '../../services/api';
import type { RootState } from '../index';

interface ProgressState {
  summaries: ProgressSummary[];
  currentSummary: ProgressSummary | null;
  records: ProgressRecord[];
  variance: VarianceAnalysis | null;
  projection: { year: number; projected_emissions: number; target_emissions: number }[];
  loading: boolean;
  error: string | null;
}

const initialState: ProgressState = {
  summaries: [],
  currentSummary: null,
  records: [],
  variance: null,
  projection: [],
  loading: false,
  error: null,
};

export const fetchProgressDashboard = createAsyncThunk(
  'progress/fetchDashboard',
  async (orgId: string) => progressApi.getDashboard(orgId)
);

export const fetchProgressSummary = createAsyncThunk(
  'progress/fetchSummary',
  async (targetId: string) => progressApi.getSummary(targetId)
);

export const fetchProgressHistory = createAsyncThunk(
  'progress/fetchHistory',
  async (targetId: string) => progressApi.getHistory(targetId)
);

export const fetchVarianceAnalysis = createAsyncThunk(
  'progress/fetchVariance',
  async ({ targetId, year }: { targetId: string; year: number }) =>
    progressApi.getVarianceAnalysis(targetId, year)
);

export const fetchProjection = createAsyncThunk(
  'progress/fetchProjection',
  async (targetId: string) => progressApi.getProjection(targetId)
);

export const recordProgress = createAsyncThunk(
  'progress/record',
  async (data: Partial<ProgressRecord>) => progressApi.recordProgress(data)
);

const progressSlice = createSlice({
  name: 'progress',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchProgressDashboard.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchProgressDashboard.fulfilled, (state, action: PayloadAction<ProgressSummary[]>) => {
        state.loading = false;
        state.summaries = action.payload;
      })
      .addCase(fetchProgressDashboard.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch progress dashboard';
      })
      .addCase(fetchProgressSummary.fulfilled, (state, action: PayloadAction<ProgressSummary>) => {
        state.currentSummary = action.payload;
      })
      .addCase(fetchProgressHistory.fulfilled, (state, action: PayloadAction<ProgressRecord[]>) => {
        state.records = action.payload;
      })
      .addCase(fetchVarianceAnalysis.fulfilled, (state, action: PayloadAction<VarianceAnalysis>) => {
        state.variance = action.payload;
      })
      .addCase(fetchProjection.fulfilled, (state, action) => {
        state.projection = action.payload;
      })
      .addCase(recordProgress.fulfilled, (state, action: PayloadAction<ProgressRecord>) => {
        state.records.push(action.payload);
      });
  },
});

export const { clearError } = progressSlice.actions;
export const selectProgressSummaries = (state: RootState) => state.progress.summaries;
export const selectCurrentProgressSummary = (state: RootState) => state.progress.currentSummary;
export const selectProgressRecords = (state: RootState) => state.progress.records;
export const selectVariance = (state: RootState) => state.progress.variance;
export const selectProjection = (state: RootState) => state.progress.projection;
export const selectProgressLoading = (state: RootState) => state.progress.loading;
export default progressSlice.reducer;
