/**
 * Dashboard Redux Slice
 *
 * Manages executive dashboard state: KPI metrics (gross/net emissions,
 * removals, data quality, completeness, verification stage), emissions
 * trends by ISO category, category breakdowns, and alert notifications.
 *
 * Async thunks:
 *   - fetchMetrics: Load executive KPI card data
 *   - fetchTrends: Load historical emissions trend line data
 *   - fetchCategoryBreakdown: Load category-level breakdown
 *   - fetchAlerts: Load dashboard alert feed
 *   - markAlertRead: Dismiss an alert
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  DashboardState,
  DashboardMetrics,
  TrendDataPoint,
  CategoryBreakdownItem,
  DashboardAlert,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: DashboardState = {
  metrics: null,
  trendData: [],
  categoryBreakdown: [],
  alerts: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const fetchMetrics = createAsyncThunk<
  DashboardMetrics,
  { orgId: string; reportingYear: number }
>(
  'dashboard/fetchMetrics',
  async ({ orgId, reportingYear }) => {
    return iso14064Api.getDashboardMetrics(orgId, reportingYear);
  },
);

export const fetchTrends = createAsyncThunk<
  TrendDataPoint[],
  { orgId: string; startYear: number; endYear: number; granularity?: 'monthly' | 'quarterly' | 'yearly' }
>(
  'dashboard/fetchTrends',
  async ({ orgId, startYear, endYear, granularity }) => {
    return iso14064Api.getTrendData(orgId, startYear, endYear, granularity);
  },
);

export const fetchCategoryBreakdown = createAsyncThunk<
  CategoryBreakdownItem[],
  string
>(
  'dashboard/fetchCategoryBreakdown',
  async (inventoryId) => {
    return iso14064Api.getCategoryBreakdown(inventoryId);
  },
);

export const fetchAlerts = createAsyncThunk<
  DashboardAlert[],
  string
>(
  'dashboard/fetchAlerts',
  async (orgId) => {
    return iso14064Api.getDashboardAlerts(orgId);
  },
);

export const markAlertRead = createAsyncThunk<
  string,
  string
>(
  'dashboard/markAlertRead',
  async (alertId) => {
    await iso14064Api.markAlertRead(alertId);
    return alertId;
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    clearDashboard: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      // -- fetchMetrics --
      .addCase(fetchMetrics.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchMetrics.fulfilled, (state, action) => {
        state.loading = false;
        state.metrics = action.payload;
      })
      .addCase(fetchMetrics.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load dashboard metrics';
      })

      // -- fetchTrends --
      .addCase(fetchTrends.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTrends.fulfilled, (state, action) => {
        state.loading = false;
        state.trendData = action.payload;
      })
      .addCase(fetchTrends.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load trend data';
      })

      // -- fetchCategoryBreakdown --
      .addCase(fetchCategoryBreakdown.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCategoryBreakdown.fulfilled, (state, action) => {
        state.loading = false;
        state.categoryBreakdown = action.payload;
      })
      .addCase(fetchCategoryBreakdown.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load category breakdown';
      })

      // -- fetchAlerts --
      .addCase(fetchAlerts.fulfilled, (state, action) => {
        state.alerts = action.payload;
      })

      // -- markAlertRead --
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
