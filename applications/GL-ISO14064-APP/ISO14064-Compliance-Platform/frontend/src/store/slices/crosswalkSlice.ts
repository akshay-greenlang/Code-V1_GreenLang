/**
 * Crosswalk Redux Slice
 *
 * Manages the ISO 14064-1 to GHG Protocol crosswalk state.
 * The crosswalk maps ISO categories (1-6) to GHG Protocol scopes
 * (Scope 1/2/3) for dual-framework reporting and reconciliation.
 *
 * Async thunks:
 *   - generateCrosswalk: Generate a new crosswalk for an inventory
 *   - fetchCrosswalk: Load existing crosswalk
 *   - exportCrosswalk: Export crosswalk in specified format
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  CrosswalkState,
  CrosswalkResult,
  ExportResult,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: CrosswalkState = {
  crosswalk: null,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const generateCrosswalk = createAsyncThunk<
  CrosswalkResult,
  string
>(
  'crosswalk/generate',
  async (inventoryId) => {
    return iso14064Api.generateCrosswalk(inventoryId);
  },
);

export const fetchCrosswalk = createAsyncThunk<
  CrosswalkResult,
  string
>(
  'crosswalk/fetch',
  async (inventoryId) => {
    return iso14064Api.getCrosswalk(inventoryId);
  },
);

export const exportCrosswalk = createAsyncThunk<
  ExportResult,
  { inventoryId: string; format: string }
>(
  'crosswalk/export',
  async ({ inventoryId, format }) => {
    return iso14064Api.exportCrosswalk(inventoryId, format);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const crosswalkSlice = createSlice({
  name: 'crosswalk',
  initialState,
  reducers: {
    clearCrosswalk: () => initialState,
    clearCrosswalkError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // -- generateCrosswalk --
      .addCase(generateCrosswalk.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(generateCrosswalk.fulfilled, (state, action) => {
        state.loading = false;
        state.crosswalk = action.payload;
      })
      .addCase(generateCrosswalk.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to generate crosswalk';
      })

      // -- fetchCrosswalk --
      .addCase(fetchCrosswalk.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCrosswalk.fulfilled, (state, action) => {
        state.loading = false;
        state.crosswalk = action.payload;
      })
      .addCase(fetchCrosswalk.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch crosswalk';
      })

      // -- exportCrosswalk --
      .addCase(exportCrosswalk.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(exportCrosswalk.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(exportCrosswalk.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to export crosswalk';
      });
  },
});

export const { clearCrosswalk, clearCrosswalkError } = crosswalkSlice.actions;
export default crosswalkSlice.reducer;
