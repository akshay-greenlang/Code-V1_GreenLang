import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';
import type { DashboardMetrics, HotspotAnalysis } from '../../types';

interface DashboardState {
  metrics: DashboardMetrics | null;
  hotspots: HotspotAnalysis | null;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: DashboardState = {
  metrics: null,
  hotspots: null,
  loading: false,
  error: null,
  lastUpdated: null,
};

// Async thunks
export const fetchDashboardMetrics = createAsyncThunk(
  'dashboard/fetchMetrics',
  async ({ startDate, endDate }: { startDate?: string; endDate?: string } = {}) => {
    const metrics = await api.getDashboardMetrics(startDate, endDate);
    return metrics;
  }
);

export const fetchHotspotAnalysis = createAsyncThunk(
  'dashboard/fetchHotspots',
  async (params?: { minEmissions?: number; topN?: number }) => {
    const hotspots = await api.getHotspotAnalysis(params);
    return hotspots;
  }
);

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearDashboard: (state) => {
      state.metrics = null;
      state.hotspots = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch dashboard metrics
    builder
      .addCase(fetchDashboardMetrics.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDashboardMetrics.fulfilled, (state, action: PayloadAction<DashboardMetrics>) => {
        state.loading = false;
        state.metrics = action.payload;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchDashboardMetrics.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch dashboard metrics';
      });

    // Fetch hotspot analysis
    builder
      .addCase(fetchHotspotAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchHotspotAnalysis.fulfilled, (state, action: PayloadAction<HotspotAnalysis>) => {
        state.loading = false;
        state.hotspots = action.payload;
      })
      .addCase(fetchHotspotAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch hotspot analysis';
      });
  },
});

export const { clearDashboard } = dashboardSlice.actions;
export default dashboardSlice.reducer;
