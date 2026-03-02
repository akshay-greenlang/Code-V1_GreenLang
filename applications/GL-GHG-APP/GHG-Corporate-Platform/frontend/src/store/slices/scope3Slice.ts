/**
 * Scope 3 Redux Slice
 *
 * Manages Scope 3 (value chain) emissions state across all 15 categories.
 * Handles aggregation, summary, category detail, and materiality assessment.
 *
 * Async thunks:
 *   - aggregateScope3: Trigger server-side aggregation
 *   - fetchSummary: Load summary with upstream/downstream breakdown
 *   - fetchCategories: Load all 15 category breakdowns
 *   - fetchMateriality: Load materiality assessment results
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  Scope3State,
  Scope3Summary,
  Scope3CategoryBreakdown,
  MaterialityResult,
  ScopeEmissions,
  AggregationResult,
  SubmitScope3DataRequest,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: Scope3State = {
  summary: null,
  categories: [],
  materiality: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const aggregateScope3 = createAsyncThunk<
  AggregationResult,
  string
>(
  'scope3/aggregate',
  async (inventoryId) => {
    return ghgApi.aggregateScope3(inventoryId);
  },
);

export const fetchScope3Summary = createAsyncThunk<
  Scope3Summary,
  string
>(
  'scope3/fetchSummary',
  async (inventoryId) => {
    return ghgApi.getScope3Summary(inventoryId);
  },
);

export const fetchScope3Categories = createAsyncThunk<
  Scope3CategoryBreakdown[],
  string
>(
  'scope3/fetchCategories',
  async (inventoryId) => {
    return ghgApi.getScope3Categories(inventoryId);
  },
);

export const fetchScope3CategoryDetail = createAsyncThunk<
  Scope3CategoryBreakdown,
  { inventoryId: string; categoryKey: string }
>(
  'scope3/fetchCategoryDetail',
  async ({ inventoryId, categoryKey }) => {
    return ghgApi.getScope3CategoryDetail(inventoryId, categoryKey);
  },
);

export const fetchScope3Materiality = createAsyncThunk<
  MaterialityResult[],
  string
>(
  'scope3/fetchMateriality',
  async (inventoryId) => {
    return ghgApi.getScope3Materiality(inventoryId);
  },
);

export const submitScope3Data = createAsyncThunk<
  ScopeEmissions,
  { inventoryId: string; payload: SubmitScope3DataRequest }
>(
  'scope3/submitData',
  async ({ inventoryId, payload }) => {
    return ghgApi.submitScope3Data(inventoryId, payload);
  },
);

export const fetchScope3Upstream = createAsyncThunk<
  Scope3CategoryBreakdown[],
  string
>(
  'scope3/fetchUpstream',
  async (inventoryId) => {
    return ghgApi.getScope3Upstream(inventoryId);
  },
);

export const fetchScope3Downstream = createAsyncThunk<
  Scope3CategoryBreakdown[],
  string
>(
  'scope3/fetchDownstream',
  async (inventoryId) => {
    return ghgApi.getScope3Downstream(inventoryId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const scope3Slice = createSlice({
  name: 'scope3',
  initialState,
  reducers: {
    clearScope3: () => initialState,
    setSelectedCategory: (state, action: PayloadAction<string | null>) => {
      // Find and mark a category as selected in local state if needed
      if (action.payload === null) return;
    },
  },
  extraReducers: (builder) => {
    builder
      // -- aggregateScope3 --
      .addCase(aggregateScope3.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(aggregateScope3.fulfilled, (state, action) => {
        state.loading = false;
        if (state.summary) {
          state.summary.total_tco2e = action.payload.total_tco2e;
        }
      })
      .addCase(aggregateScope3.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to aggregate Scope 3 emissions';
      })

      // -- fetchScope3Summary --
      .addCase(fetchScope3Summary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchScope3Summary.fulfilled, (state, action) => {
        state.loading = false;
        state.summary = action.payload;
        state.categories = action.payload.by_category;
        state.materiality = action.payload.materiality_assessment;
      })
      .addCase(fetchScope3Summary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load Scope 3 summary';
      })

      // -- fetchScope3Categories --
      .addCase(fetchScope3Categories.fulfilled, (state, action) => {
        state.categories = action.payload;
      })

      // -- fetchScope3CategoryDetail --
      .addCase(fetchScope3CategoryDetail.fulfilled, (state, action) => {
        const idx = state.categories.findIndex(
          (c) => c.category === action.payload.category,
        );
        if (idx >= 0) {
          state.categories[idx] = action.payload;
        }
      })

      // -- fetchScope3Materiality --
      .addCase(fetchScope3Materiality.fulfilled, (state, action) => {
        state.materiality = action.payload;
      })

      // -- submitScope3Data --
      .addCase(submitScope3Data.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(submitScope3Data.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(submitScope3Data.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to submit Scope 3 data';
      });
  },
});

export const { clearScope3, setSelectedCategory } = scope3Slice.actions;
export default scope3Slice.reducer;
