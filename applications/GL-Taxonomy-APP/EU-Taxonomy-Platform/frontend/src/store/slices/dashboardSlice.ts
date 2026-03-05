import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { DashboardOverview, KPISummary, AlignmentSummaryCard, SectorBreakdownItem, TrendDataPoint } from '../../types';
import { dashboardApi } from '../../services/api';
import type { RootState } from '../index';

interface DashboardState {
  overview: DashboardOverview | null;
  kpiSummary: KPISummary | null;
  alignmentSummary: AlignmentSummaryCard | null;
  sectorBreakdown: SectorBreakdownItem[];
  trendData: TrendDataPoint[];
  loading: boolean;
  error: string | null;
}

const initialState: DashboardState = {
  overview: null,
  kpiSummary: null,
  alignmentSummary: null,
  sectorBreakdown: [],
  trendData: [],
  loading: false,
  error: null,
};

export const fetchDashboardOverview = createAsyncThunk(
  'dashboard/fetchOverview',
  async ({ orgId, period }: { orgId: string; period: string }) =>
    dashboardApi.overview(orgId, period)
);

export const fetchKPISummary = createAsyncThunk(
  'dashboard/fetchKPISummary',
  async ({ orgId, period }: { orgId: string; period: string }) =>
    dashboardApi.kpiCards(orgId, period)
);

export const fetchAlignmentSummary = createAsyncThunk(
  'dashboard/fetchAlignmentSummary',
  async (orgId: string) => dashboardApi.alignmentSummary(orgId)
);

export const fetchSectorBreakdown = createAsyncThunk(
  'dashboard/fetchSectorBreakdown',
  async (orgId: string) => dashboardApi.sectorBreakdown(orgId)
);

export const fetchTrends = createAsyncThunk(
  'dashboard/fetchTrends',
  async ({ orgId, periods }: { orgId: string; periods: number }) =>
    dashboardApi.trends(orgId, periods)
);

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDashboardOverview.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchDashboardOverview.fulfilled, (state, action: PayloadAction<DashboardOverview>) => {
        state.loading = false;
        state.overview = action.payload;
      })
      .addCase(fetchDashboardOverview.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch dashboard overview';
      })
      .addCase(fetchKPISummary.fulfilled, (state, action) => { state.kpiSummary = action.payload; })
      .addCase(fetchAlignmentSummary.fulfilled, (state, action) => { state.alignmentSummary = action.payload; })
      .addCase(fetchSectorBreakdown.fulfilled, (state, action) => { state.sectorBreakdown = action.payload; })
      .addCase(fetchTrends.fulfilled, (state, action) => { state.trendData = action.payload; });
  },
});

export const { clearError } = dashboardSlice.actions;
export const selectDashboardOverview = (state: RootState) => state.dashboard.overview;
export const selectKPISummary = (state: RootState) => state.dashboard.kpiSummary;
export const selectAlignmentSummary = (state: RootState) => state.dashboard.alignmentSummary;
export const selectSectorBreakdown = (state: RootState) => state.dashboard.sectorBreakdown;
export const selectTrendData = (state: RootState) => state.dashboard.trendData;
export const selectDashboardLoading = (state: RootState) => state.dashboard.loading;
export default dashboardSlice.reducer;
