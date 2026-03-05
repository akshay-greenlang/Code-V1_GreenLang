/**
 * Inventory Redux Slice
 *
 * Manages ISO 14064-1 inventory state: inventory CRUD, grand totals,
 * category-level results, base year configuration, and recalculation
 * triggers.
 *
 * Async thunks:
 *   - createInventory: Create a new inventory for a reporting year
 *   - fetchInventory: Load an inventory by ID
 *   - fetchInventories: List inventories for an organization
 *   - fetchTotals: Load grand totals for an inventory
 *   - fetchCategoryResults: Load category-level aggregations
 *   - setBaseYear: Set or update the base year
 *   - recalculateBaseYear: Trigger base year recalculation
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  InventoryState,
  ISOInventory,
  InventoryTotals,
  CategoryResult,
  BaseYearRecord,
  BaseYearTrigger,
  CreateInventoryRequest,
  SetBaseYearRequest,
  RecalculateBaseYearRequest,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: InventoryState = {
  currentInventory: null,
  inventories: [],
  totals: null,
  categoryResults: [],
  baseYear: null,
  baseYearTriggers: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const createInventory = createAsyncThunk<
  ISOInventory,
  CreateInventoryRequest
>(
  'inventory/createInventory',
  async (payload) => {
    return iso14064Api.createInventory(payload);
  },
);

export const fetchInventory = createAsyncThunk<
  ISOInventory,
  string
>(
  'inventory/fetchInventory',
  async (inventoryId) => {
    return iso14064Api.getInventory(inventoryId);
  },
);

export const fetchInventories = createAsyncThunk<
  ISOInventory[],
  string
>(
  'inventory/fetchInventories',
  async (orgId) => {
    return iso14064Api.listInventories(orgId);
  },
);

export const fetchTotals = createAsyncThunk<
  InventoryTotals,
  string
>(
  'inventory/fetchTotals',
  async (inventoryId) => {
    return iso14064Api.getInventoryTotals(inventoryId);
  },
);

export const fetchCategoryResults = createAsyncThunk<
  CategoryResult[],
  string
>(
  'inventory/fetchCategoryResults',
  async (inventoryId) => {
    return iso14064Api.getCategoryResults(inventoryId);
  },
);

export const setBaseYear = createAsyncThunk<
  BaseYearRecord,
  { orgId: string; payload: SetBaseYearRequest }
>(
  'inventory/setBaseYear',
  async ({ orgId, payload }) => {
    return iso14064Api.setBaseYear(orgId, payload);
  },
);

export const recalculateBaseYear = createAsyncThunk<
  BaseYearTrigger,
  { orgId: string; payload: RecalculateBaseYearRequest }
>(
  'inventory/recalculateBaseYear',
  async ({ orgId, payload }) => {
    return iso14064Api.recalculateBaseYear(orgId, payload);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const inventorySlice = createSlice({
  name: 'inventory',
  initialState,
  reducers: {
    clearInventory: () => initialState,
    setCurrentInventory: (state, action: PayloadAction<ISOInventory>) => {
      state.currentInventory = action.payload;
    },
    clearInventoryError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // -- createInventory --
      .addCase(createInventory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createInventory.fulfilled, (state, action) => {
        state.loading = false;
        state.currentInventory = action.payload;
        state.inventories.push(action.payload);
      })
      .addCase(createInventory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to create inventory';
      })

      // -- fetchInventory --
      .addCase(fetchInventory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchInventory.fulfilled, (state, action) => {
        state.loading = false;
        state.currentInventory = action.payload;
      })
      .addCase(fetchInventory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch inventory';
      })

      // -- fetchInventories --
      .addCase(fetchInventories.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchInventories.fulfilled, (state, action) => {
        state.loading = false;
        state.inventories = action.payload;
      })
      .addCase(fetchInventories.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch inventories';
      })

      // -- fetchTotals --
      .addCase(fetchTotals.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTotals.fulfilled, (state, action) => {
        state.loading = false;
        state.totals = action.payload;
      })
      .addCase(fetchTotals.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch inventory totals';
      })

      // -- fetchCategoryResults --
      .addCase(fetchCategoryResults.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCategoryResults.fulfilled, (state, action) => {
        state.loading = false;
        state.categoryResults = action.payload;
      })
      .addCase(fetchCategoryResults.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch category results';
      })

      // -- setBaseYear --
      .addCase(setBaseYear.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(setBaseYear.fulfilled, (state, action) => {
        state.loading = false;
        state.baseYear = action.payload;
      })
      .addCase(setBaseYear.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to set base year';
      })

      // -- recalculateBaseYear --
      .addCase(recalculateBaseYear.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(recalculateBaseYear.fulfilled, (state, action) => {
        state.loading = false;
        state.baseYearTriggers.push(action.payload);
      })
      .addCase(recalculateBaseYear.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to recalculate base year';
      });
  },
});

export const { clearInventory, setCurrentInventory, clearInventoryError } =
  inventorySlice.actions;
export default inventorySlice.reducer;
