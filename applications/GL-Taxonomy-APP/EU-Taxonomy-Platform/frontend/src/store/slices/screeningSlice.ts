import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { EligibilityScreening, ScreeningSummary, BatchScreenRequest } from '../../types';
import { screeningApi } from '../../services/api';
import type { RootState } from '../index';

interface ScreeningState {
  screening: EligibilityScreening | null;
  summary: ScreeningSummary | null;
  loading: boolean;
  error: string | null;
}

const initialState: ScreeningState = {
  screening: null,
  summary: null,
  loading: false,
  error: null,
};

export const batchScreen = createAsyncThunk(
  'screening/batch',
  async (request: BatchScreenRequest) => screeningApi.batchScreen(request)
);

export const fetchScreeningResults = createAsyncThunk(
  'screening/results',
  async (orgId: string) => screeningApi.getResults(orgId)
);

export const fetchScreeningSummary = createAsyncThunk(
  'screening/summary',
  async (orgId: string) => screeningApi.getSummary(orgId)
);

export const applyDeMinimis = createAsyncThunk(
  'screening/deMinimis',
  async ({ orgId, threshold }: { orgId: string; threshold: number }) =>
    screeningApi.applyDeMinimis(orgId, threshold)
);

const screeningSlice = createSlice({
  name: 'screening',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(batchScreen.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(batchScreen.fulfilled, (state, action: PayloadAction<EligibilityScreening>) => {
        state.loading = false;
        state.screening = action.payload;
      })
      .addCase(batchScreen.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to run screening';
      })
      .addCase(fetchScreeningResults.fulfilled, (state, action) => { state.screening = action.payload; })
      .addCase(fetchScreeningSummary.fulfilled, (state, action) => { state.summary = action.payload; })
      .addCase(applyDeMinimis.fulfilled, (state, action) => { state.screening = action.payload; });
  },
});

export const { clearError } = screeningSlice.actions;
export const selectScreening = (state: RootState) => state.screening.screening;
export const selectScreeningSummary = (state: RootState) => state.screening.summary;
export const selectScreeningLoading = (state: RootState) => state.screening.loading;
export default screeningSlice.reducer;
