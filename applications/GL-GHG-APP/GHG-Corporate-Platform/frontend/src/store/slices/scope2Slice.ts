/**
 * Scope 2 Redux Slice
 *
 * Manages Scope 2 (indirect energy) emissions state: aggregation,
 * summary, location-based, market-based, and reconciliation data.
 *
 * Async thunks:
 *   - aggregateScope2: Trigger server-side aggregation
 *   - fetchSummary: Load dual-reporting summary
 *   - fetchLocationBased: Load location-based emissions detail
 *   - fetchMarketBased: Load market-based emissions detail
 *   - fetchReconciliation: Load location vs. market reconciliation
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  Scope2State,
  Scope2Summary,
  ScopeEmissions,
  ReconciliationData,
  AggregationResult,
  SubmitScope2DataRequest,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: Scope2State = {
  summary: null,
  reconciliation: null,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const aggregateScope2 = createAsyncThunk<
  AggregationResult,
  string
>(
  'scope2/aggregate',
  async (inventoryId) => {
    return ghgApi.aggregateScope2(inventoryId);
  },
);

export const fetchScope2Summary = createAsyncThunk<
  Scope2Summary,
  string
>(
  'scope2/fetchSummary',
  async (inventoryId) => {
    return ghgApi.getScope2Summary(inventoryId);
  },
);

export const fetchScope2LocationBased = createAsyncThunk<
  ScopeEmissions,
  string
>(
  'scope2/fetchLocationBased',
  async (inventoryId) => {
    return ghgApi.getScope2LocationBased(inventoryId);
  },
);

export const fetchScope2MarketBased = createAsyncThunk<
  ScopeEmissions,
  string
>(
  'scope2/fetchMarketBased',
  async (inventoryId) => {
    return ghgApi.getScope2MarketBased(inventoryId);
  },
);

export const fetchScope2Reconciliation = createAsyncThunk<
  ReconciliationData,
  string
>(
  'scope2/fetchReconciliation',
  async (inventoryId) => {
    return ghgApi.getScope2Reconciliation(inventoryId);
  },
);

export const submitScope2Data = createAsyncThunk<
  ScopeEmissions,
  { inventoryId: string; payload: SubmitScope2DataRequest }
>(
  'scope2/submitData',
  async ({ inventoryId, payload }) => {
    return ghgApi.submitScope2Data(inventoryId, payload);
  },
);

export const fetchScope2ByEnergyType = createAsyncThunk<
  Scope2Summary,
  string
>(
  'scope2/fetchByEnergyType',
  async (inventoryId) => {
    return ghgApi.getScope2ByEnergyType(inventoryId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const scope2Slice = createSlice({
  name: 'scope2',
  initialState,
  reducers: {
    clearScope2: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      // -- aggregateScope2 --
      .addCase(aggregateScope2.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(aggregateScope2.fulfilled, (state, action) => {
        state.loading = false;
        if (state.summary) {
          state.summary.location_based_tco2e = action.payload.total_tco2e;
        }
      })
      .addCase(aggregateScope2.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to aggregate Scope 2 emissions';
      })

      // -- fetchScope2Summary --
      .addCase(fetchScope2Summary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchScope2Summary.fulfilled, (state, action) => {
        state.loading = false;
        state.summary = action.payload;
      })
      .addCase(fetchScope2Summary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load Scope 2 summary';
      })

      // -- fetchScope2Reconciliation --
      .addCase(fetchScope2Reconciliation.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchScope2Reconciliation.fulfilled, (state, action) => {
        state.loading = false;
        state.reconciliation = action.payload;
      })
      .addCase(fetchScope2Reconciliation.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load reconciliation';
      })

      // -- submitScope2Data --
      .addCase(submitScope2Data.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(submitScope2Data.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(submitScope2Data.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to submit Scope 2 data';
      })

      // -- fetchScope2ByEnergyType --
      .addCase(fetchScope2ByEnergyType.fulfilled, (state, action) => {
        state.summary = action.payload;
      });
  },
});

export const { clearScope2 } = scope2Slice.actions;
export default scope2Slice.reducer;
