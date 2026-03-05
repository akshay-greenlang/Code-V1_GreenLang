import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { FiveYearReview } from '../../types';
import { reviewApi } from '../../services/api';
import type { RootState } from '../index';

interface ReviewState {
  currentReview: FiveYearReview | null;
  reviewHistory: FiveYearReview[];
  readiness: { readiness_pct: number; items_completed: number; items_total: number; blockers: string[] } | null;
  loading: boolean;
  error: string | null;
}

const initialState: ReviewState = {
  currentReview: null,
  reviewHistory: [],
  readiness: null,
  loading: false,
  error: null,
};

export const fetchReview = createAsyncThunk(
  'review/fetchReview',
  async (orgId: string) => reviewApi.getReview(orgId)
);

export const fetchReviewReadiness = createAsyncThunk(
  'review/fetchReadiness',
  async (orgId: string) => reviewApi.getReadiness(orgId)
);

export const fetchReviewHistory = createAsyncThunk(
  'review/fetchHistory',
  async (orgId: string) => reviewApi.getHistory(orgId)
);

export const createReview = createAsyncThunk(
  'review/create',
  async (orgId: string) => reviewApi.createReview(orgId)
);

export const submitReviewOutcome = createAsyncThunk(
  'review/submitOutcome',
  async ({ id, outcome }: { id: string; outcome: string }) => reviewApi.submitOutcome(id, outcome)
);

const reviewSlice = createSlice({
  name: 'review',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchReview.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchReview.fulfilled, (state, action: PayloadAction<FiveYearReview>) => {
        state.loading = false;
        state.currentReview = action.payload;
      })
      .addCase(fetchReview.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch review';
      })
      .addCase(fetchReviewReadiness.fulfilled, (state, action) => {
        state.readiness = action.payload;
      })
      .addCase(fetchReviewHistory.fulfilled, (state, action: PayloadAction<FiveYearReview[]>) => {
        state.reviewHistory = action.payload;
      })
      .addCase(createReview.fulfilled, (state, action: PayloadAction<FiveYearReview>) => {
        state.currentReview = action.payload;
      })
      .addCase(submitReviewOutcome.fulfilled, (state, action: PayloadAction<FiveYearReview>) => {
        state.currentReview = action.payload;
      });
  },
});

export const { clearError } = reviewSlice.actions;
export const selectCurrentReview = (state: RootState) => state.review.currentReview;
export const selectReviewHistory = (state: RootState) => state.review.reviewHistory;
export const selectReviewReadiness = (state: RootState) => state.review.readiness;
export const selectReviewLoading = (state: RootState) => state.review.loading;
export default reviewSlice.reducer;
