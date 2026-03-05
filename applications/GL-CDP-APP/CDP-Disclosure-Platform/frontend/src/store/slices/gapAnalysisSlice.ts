/**
 * Gap Analysis Redux Slice
 *
 * Manages gap analysis state: gap identification, severity counts,
 * recommendations, uplift predictions, and gap resolution tracking.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { GapAnalysisState, GapAnalysis, GapRecommendation, ScoringLevel } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: GapAnalysisState = {
  analysis: null,
  recommendations: [],
  loading: false,
  error: null,
};

export const runGapAnalysis = createAsyncThunk<
  GapAnalysis,
  { questionnaireId: string; targetLevel?: ScoringLevel }
>(
  'gapAnalysis/run',
  async ({ questionnaireId, targetLevel }) =>
    cdpApi.runGapAnalysis({ questionnaire_id: questionnaireId, target_level: targetLevel }),
);

export const fetchGapAnalysis = createAsyncThunk<GapAnalysis, string>(
  'gapAnalysis/fetch',
  async (questionnaireId) => cdpApi.getGapAnalysis(questionnaireId),
);

export const fetchRecommendations = createAsyncThunk<GapRecommendation[], string>(
  'gapAnalysis/fetchRecommendations',
  async (questionnaireId) => cdpApi.getRecommendations(questionnaireId),
);

export const resolveGap = createAsyncThunk<string, string>(
  'gapAnalysis/resolve',
  async (gapId) => {
    await cdpApi.resolveGap(gapId);
    return gapId;
  },
);

const gapAnalysisSlice = createSlice({
  name: 'gapAnalysis',
  initialState,
  reducers: {
    clearGapAnalysis: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(runGapAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(runGapAnalysis.fulfilled, (state, action) => {
        state.loading = false;
        state.analysis = action.payload;
      })
      .addCase(runGapAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to run gap analysis';
      })
      .addCase(fetchGapAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchGapAnalysis.fulfilled, (state, action) => {
        state.loading = false;
        state.analysis = action.payload;
      })
      .addCase(fetchGapAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load gap analysis';
      })
      .addCase(fetchRecommendations.fulfilled, (state, action) => {
        state.recommendations = action.payload;
      })
      .addCase(resolveGap.fulfilled, (state, action) => {
        if (state.analysis) {
          const gap = state.analysis.gaps.find((g) => g.id === action.payload);
          if (gap) {
            gap.is_resolved = true;
            gap.resolved_at = new Date().toISOString();
          }
        }
      });
  },
});

export const { clearGapAnalysis } = gapAnalysisSlice.actions;
export default gapAnalysisSlice.reducer;
