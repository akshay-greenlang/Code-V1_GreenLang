/**
 * Dashboard Redux Slice
 *
 * Manages executive dashboard state: KPI metrics, emissions trends,
 * scope breakdowns, and alert notifications.
 *
 * Async thunks:
 *   - fetchMetrics: Load executive KPI card data
 *   - fetchTrends: Load historical emissions trend line data
 *   - fetchBreakdown: Load category-level scope breakdown
 *   - fetchAlerts: Load dashboard alert feed
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  DashboardState,
  DashboardMetrics,
  TrendDataPoint,
  ScopeBreakdown,
  DashboardAlert,
  Scope,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: DashboardState = {
  metrics: null,
  trendData: [],
  scopeBreakdown: [],
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
    return ghgApi.getDashboardMetrics(orgId, reportingYear);
  },
);

export const fetchTrends = createAsyncThunk<
  TrendDataPoint[],
  { orgId: string; startYear: number; endYear: number; granularity?: 'monthly' | 'quarterly' | 'yearly' }
>(
  'dashboard/fetchTrends',
  async ({ orgId, startYear, endYear, granularity }) => {
    return ghgApi.getTrendData(orgId, startYear, endYear, granularity);
  },
);

export const fetchBreakdown = createAsyncThunk<
  ScopeBreakdown,
  { inventoryId: string; scope: Scope }
>(
  'dashboard/fetchBreakdown',
  async ({ inventoryId, scope }) => {
    return ghgApi.getScopeBreakdown(inventoryId, scope);
  },
);

export const fetchAlerts = createAsyncThunk<
  DashboardAlert[],
  string
>(
  'dashboard/fetchAlerts',
  async (orgId) => {
    return ghgApi.getDashboardAlerts(orgId);
  },
);

export const markAlertRead = createAsyncThunk<
  string,
  string
>(
  'dashboard/markAlertRead',
  async (alertId) => {
    await ghgApi.markAlertRead(alertId);
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
        state.alerts = action.payload.alerts;
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

      // -- fetchBreakdown --
      .addCase(fetchBreakdown.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchBreakdown.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.scopeBreakdown.findIndex(
          (b) => b.scope === action.payload.scope,
        );
        if (idx >= 0) {
          state.scopeBreakdown[idx] = action.payload;
        } else {
          state.scopeBreakdown.push(action.payload);
        }
      })
      .addCase(fetchBreakdown.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load scope breakdown';
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
