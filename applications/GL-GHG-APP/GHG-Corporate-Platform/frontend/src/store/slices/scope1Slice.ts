/**
 * Scope 1 Redux Slice
 *
 * Manages Scope 1 (direct) emissions state: aggregation, summary,
 * category breakdown, gas breakdown, and data submission.
 *
 * Async thunks:
 *   - aggregateScope1: Trigger server-side aggregation across entities
 *   - fetchSummary: Load summary with category and entity breakdowns
 *   - fetchCategories: Load category-level breakdown
 *   - submitData: Submit activity data for a Scope 1 emission source
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  Scope1State,
  Scope1Summary,
  Scope1CategoryBreakdown,
  ScopeEmissions,
  AggregationResult,
  SubmitScope1DataRequest,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: Scope1State = {
  summary: null,
  categories: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const aggregateScope1 = createAsyncThunk<
  AggregationResult,
  string
>(
  'scope1/aggregate',
  async (inventoryId) => {
    return ghgApi.aggregateScope1(inventoryId);
  },
);

export const fetchScope1Summary = createAsyncThunk<
  Scope1Summary,
  string
>(
  'scope1/fetchSummary',
  async (inventoryId) => {
    return ghgApi.getScope1Summary(inventoryId);
  },
);

export const fetchScope1Categories = createAsyncThunk<
  Scope1CategoryBreakdown[],
  string
>(
  'scope1/fetchCategories',
  async (inventoryId) => {
    return ghgApi.getScope1Categories(inventoryId);
  },
);

export const submitScope1Data = createAsyncThunk<
  ScopeEmissions,
  { inventoryId: string; payload: SubmitScope1DataRequest }
>(
  'scope1/submitData',
  async ({ inventoryId, payload }) => {
    return ghgApi.submitScope1Data(inventoryId, payload);
  },
);

export const fetchScope1GasBreakdown = createAsyncThunk<
  Record<string, number>,
  string
>(
  'scope1/fetchGasBreakdown',
  async (inventoryId) => {
    return ghgApi.getScope1GasBreakdown(inventoryId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const scope1Slice = createSlice({
  name: 'scope1',
  initialState,
  reducers: {
    clearScope1: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      // -- aggregateScope1 --
      .addCase(aggregateScope1.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(aggregateScope1.fulfilled, (state, action) => {
        state.loading = false;
        if (state.summary) {
          state.summary.total_tco2e = action.payload.total_tco2e;
          state.summary.gas_breakdown = action.payload.gas_breakdown;
          state.summary.data_quality_tier = action.payload.data_quality_tier;
        }
      })
      .addCase(aggregateScope1.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to aggregate Scope 1 emissions';
      })

      // -- fetchScope1Summary --
      .addCase(fetchScope1Summary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchScope1Summary.fulfilled, (state, action) => {
        state.loading = false;
        state.summary = action.payload;
        state.categories = action.payload.by_category;
      })
      .addCase(fetchScope1Summary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load Scope 1 summary';
      })

      // -- fetchScope1Categories --
      .addCase(fetchScope1Categories.fulfilled, (state, action) => {
        state.categories = action.payload;
      })

      // -- submitScope1Data --
      .addCase(submitScope1Data.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(submitScope1Data.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(submitScope1Data.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to submit Scope 1 data';
      });
  },
});

export const { clearScope1 } = scope1Slice.actions;
export default scope1Slice.reducer;
