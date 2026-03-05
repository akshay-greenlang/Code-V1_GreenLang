import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { DashboardSummary } from '../../types';
import { dashboardApi } from '../../services/api';
import type { RootState } from '../index';

interface DashboardState {
  summary: DashboardSummary | null;
  emissionsTrend: { year: number; scope_1: number; scope_2: number; scope_3: number; total: number }[];
  keyMetrics: { name: string; value: number; unit: string; change_pct: number; trend: string }[];
  loading: boolean;
  error: string | null;
}

const initialState: DashboardState = {
  summary: null,
  emissionsTrend: [],
  keyMetrics: [],
  loading: false,
  error: null,
};

export const fetchDashboardSummary = createAsyncThunk(
  'dashboard/fetchSummary',
  async (orgId: string) => dashboardApi.getSummary(orgId)
);

export const fetchEmissionsTrend = createAsyncThunk(
  'dashboard/fetchEmissionsTrend',
  async (orgId: string) => dashboardApi.getEmissionsTrend(orgId)
);

export const fetchKeyMetrics = createAsyncThunk(
  'dashboard/fetchKeyMetrics',
  async (orgId: string) => dashboardApi.getKeyMetrics(orgId)
);

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDashboardSummary.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchDashboardSummary.fulfilled, (state, action: PayloadAction<DashboardSummary>) => {
        state.loading = false;
        state.summary = action.payload;
      })
      .addCase(fetchDashboardSummary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch dashboard summary';
      })
      .addCase(fetchEmissionsTrend.fulfilled, (state, action) => {
        state.emissionsTrend = action.payload;
      })
      .addCase(fetchKeyMetrics.fulfilled, (state, action) => {
        state.keyMetrics = action.payload;
      });
  },
});

export const { clearError } = dashboardSlice.actions;
export const selectDashboardSummary = (state: RootState) => state.dashboard.summary;
export const selectEmissionsTrend = (state: RootState) => state.dashboard.emissionsTrend;
export const selectKeyMetrics = (state: RootState) => state.dashboard.keyMetrics;
export const selectDashboardLoading = (state: RootState) => state.dashboard.loading;
export default dashboardSlice.reducer;
