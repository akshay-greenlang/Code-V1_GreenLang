/**
 * Dashboard Redux Slice
 *
 * Manages executive dashboard state: KPI metrics, compliance trends,
 * and alert notifications for the EUDR compliance platform.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  DashboardMetrics,
  ComplianceTrend,
  AlertNotification,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface DashboardState {
  metrics: DashboardMetrics | null;
  trends: ComplianceTrend[];
  alerts: AlertNotification[];
  loading: boolean;
  metricsLoading: boolean;
  trendsLoading: boolean;
  alertsLoading: boolean;
  error: string | null;
}

const initialState: DashboardState = {
  metrics: null,
  trends: [],
  alerts: [],
  loading: false,
  metricsLoading: false,
  trendsLoading: false,
  alertsLoading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const fetchDashboardMetrics = createAsyncThunk(
  'dashboard/fetchMetrics',
  async (_, { rejectWithValue }) => {
    try {
      return await apiClient.getDashboardMetrics();
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch dashboard metrics';
      return rejectWithValue(message);
    }
  }
);

export const fetchComplianceTrends = createAsyncThunk(
  'dashboard/fetchTrends',
  async (period: string | undefined, { rejectWithValue }) => {
    try {
      return await apiClient.getComplianceTrends(period);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch compliance trends';
      return rejectWithValue(message);
    }
  }
);

export const fetchAlerts = createAsyncThunk(
  'dashboard/fetchAlerts',
  async (
    params: { is_read?: boolean; type?: string } | undefined,
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.getAlertNotifications(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch alerts';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearDashboardError(state) {
      state.error = null;
    },
    markAlertRead(state, action: PayloadAction<string>) {
      const alert = state.alerts.find((a) => a.id === action.payload);
      if (alert) alert.is_read = true;
    },
    markAllAlertsRead(state) {
      state.alerts.forEach((a) => {
        a.is_read = true;
      });
    },
  },
  extraReducers: (builder) => {
    // Metrics
    builder
      .addCase(fetchDashboardMetrics.pending, (state) => {
        state.metricsLoading = true;
        state.error = null;
      })
      .addCase(fetchDashboardMetrics.fulfilled, (state, action) => {
        state.metricsLoading = false;
        state.metrics = action.payload;
      })
      .addCase(fetchDashboardMetrics.rejected, (state, action) => {
        state.metricsLoading = false;
        state.error = action.payload as string;
      });

    // Trends
    builder
      .addCase(fetchComplianceTrends.pending, (state) => {
        state.trendsLoading = true;
        state.error = null;
      })
      .addCase(fetchComplianceTrends.fulfilled, (state, action) => {
        state.trendsLoading = false;
        state.trends = action.payload;
      })
      .addCase(fetchComplianceTrends.rejected, (state, action) => {
        state.trendsLoading = false;
        state.error = action.payload as string;
      });

    // Alerts
    builder
      .addCase(fetchAlerts.pending, (state) => {
        state.alertsLoading = true;
        state.error = null;
      })
      .addCase(fetchAlerts.fulfilled, (state, action) => {
        state.alertsLoading = false;
        state.alerts = action.payload;
      })
      .addCase(fetchAlerts.rejected, (state, action) => {
        state.alertsLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearDashboardError, markAlertRead, markAllAlertsRead } =
  dashboardSlice.actions;

export default dashboardSlice.reducer;
