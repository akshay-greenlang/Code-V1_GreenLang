/**
 * Historical Redux Slice
 *
 * Manages historical tracking state: year-over-year scores,
 * comparisons, change logs, and response carry-forward.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { HistoricalState, HistoricalScore, YearComparison } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: HistoricalState = {
  scores: [],
  comparison: null,
  loading: false,
  error: null,
};

export const fetchHistoricalScores = createAsyncThunk<HistoricalScore[], string>(
  'historical/fetchScores',
  async (orgId) => cdpApi.getHistoricalScores(orgId),
);

export const fetchYearComparison = createAsyncThunk<
  YearComparison,
  { orgId: string; yearA: number; yearB: number }
>(
  'historical/fetchComparison',
  async ({ orgId, yearA, yearB }) => cdpApi.getYearComparison(orgId, yearA, yearB),
);

export const carryForward = createAsyncThunk<
  void,
  { questionnaireId: string; sourceYear: number }
>(
  'historical/carryForward',
  async ({ questionnaireId, sourceYear }) =>
    cdpApi.carryForwardResponses(questionnaireId, sourceYear),
);

const historicalSlice = createSlice({
  name: 'historical',
  initialState,
  reducers: {
    clearHistorical: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchHistoricalScores.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchHistoricalScores.fulfilled, (state, action) => {
        state.loading = false;
        state.scores = action.payload;
      })
      .addCase(fetchHistoricalScores.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load historical scores';
      })
      .addCase(fetchYearComparison.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchYearComparison.fulfilled, (state, action) => {
        state.loading = false;
        state.comparison = action.payload;
      })
      .addCase(fetchYearComparison.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load year comparison';
      });
  },
});

export const { clearHistorical } = historicalSlice.actions;
export default historicalSlice.reducer;
