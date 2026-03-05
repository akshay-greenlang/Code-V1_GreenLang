/**
 * Dashboard Redux Slice
 *
 * Manages executive dashboard state: predicted score, module progress,
 * gap summary, timeline countdown, category scores, and alerts.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { CDPDashboardState, DashboardData, DashboardAlert } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: CDPDashboardState = {
  data: null,
  alerts: [],
  loading: false,
  error: null,
};

export const fetchDashboard = createAsyncThunk<
  DashboardData,
  { orgId: string; reportingYear: number }
>(
  'dashboard/fetch',
  async ({ orgId, reportingYear }) => cdpApi.getDashboard(orgId, reportingYear),
);

export const fetchDashboardAlerts = createAsyncThunk<DashboardAlert[], string>(
  'dashboard/fetchAlerts',
  async (orgId) => cdpApi.getDashboardAlerts(orgId),
);

export const markAlertRead = createAsyncThunk<string, string>(
  'dashboard/markAlertRead',
  async (alertId) => {
    await cdpApi.markAlertRead(alertId);
    return alertId;
  },
);

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearDashboard: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDashboard.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDashboard.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchDashboard.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load dashboard';
      })
      .addCase(fetchDashboardAlerts.fulfilled, (state, action) => {
        state.alerts = action.payload;
      })
      .addCase(markAlertRead.fulfilled, (state, action) => {
        const alert = state.alerts.find((a) => a.id === action.payload);
        if (alert) {
          alert.is_read = true;
        }
      });
  },
});

export const { clearDashboard } = dashboardSlice.actions;
export default dashboardSlice.reducer;
