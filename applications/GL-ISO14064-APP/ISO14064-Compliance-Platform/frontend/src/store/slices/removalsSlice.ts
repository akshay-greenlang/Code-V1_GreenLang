/**
 * Removals Redux Slice
 *
 * Manages GHG removal source state per ISO 14064-1 Clause 6.2:
 * forestry, soil carbon, CCS, DAC, BECCS, wetland, and ocean-based
 * removals with permanence adjustments.
 *
 * Async thunks:
 *   - addRemoval: Add a removal source to an inventory
 *   - fetchRemovals: Load all removal sources for an inventory
 *   - updateRemoval: Update a removal source
 *   - deleteRemoval: Remove a removal source
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  RemovalsState,
  RemovalSource,
  AddRemovalSourceRequest,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: RemovalsState = {
  removals: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const addRemoval = createAsyncThunk<
  RemovalSource,
  { inventoryId: string; payload: AddRemovalSourceRequest }
>(
  'removals/addRemoval',
  async ({ inventoryId, payload }) => {
    return iso14064Api.addRemovalSource(inventoryId, payload);
  },
);

export const fetchRemovals = createAsyncThunk<
  RemovalSource[],
  string
>(
  'removals/fetchRemovals',
  async (inventoryId) => {
    return iso14064Api.getRemovalSources(inventoryId);
  },
);

export const updateRemoval = createAsyncThunk<
  RemovalSource,
  { inventoryId: string; removalId: string; payload: Partial<AddRemovalSourceRequest> }
>(
  'removals/updateRemoval',
  async ({ inventoryId, removalId, payload }) => {
    return iso14064Api.updateRemovalSource(inventoryId, removalId, payload);
  },
);

export const deleteRemoval = createAsyncThunk<
  string,
  { inventoryId: string; removalId: string }
>(
  'removals/deleteRemoval',
  async ({ inventoryId, removalId }) => {
    await iso14064Api.deleteRemovalSource(inventoryId, removalId);
    return removalId;
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const removalsSlice = createSlice({
  name: 'removals',
  initialState,
  reducers: {
    clearRemovals: () => initialState,
    clearRemovalsError: (state) => {
      state.error = null;
    },
    updateRemovalLocal: (state, action: PayloadAction<RemovalSource>) => {
      const idx = state.removals.findIndex((r) => r.id === action.payload.id);
      if (idx >= 0) {
        state.removals[idx] = action.payload;
      }
    },
    removeRemovalLocal: (state, action: PayloadAction<string>) => {
      state.removals = state.removals.filter((r) => r.id !== action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      // -- addRemoval --
      .addCase(addRemoval.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(addRemoval.fulfilled, (state, action) => {
        state.loading = false;
        state.removals.push(action.payload);
      })
      .addCase(addRemoval.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to add removal source';
      })

      // -- fetchRemovals --
      .addCase(fetchRemovals.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchRemovals.fulfilled, (state, action) => {
        state.loading = false;
        state.removals = action.payload;
      })
      .addCase(fetchRemovals.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch removal sources';
      })

      // -- updateRemoval --
      .addCase(updateRemoval.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateRemoval.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.removals.findIndex((r) => r.id === action.payload.id);
        if (idx >= 0) {
          state.removals[idx] = action.payload;
        }
      })
      .addCase(updateRemoval.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to update removal source';
      })

      // -- deleteRemoval --
      .addCase(deleteRemoval.fulfilled, (state, action) => {
        state.removals = state.removals.filter((r) => r.id !== action.payload);
      });
  },
});

export const { clearRemovals, clearRemovalsError, updateRemovalLocal, removeRemovalLocal } =
  removalsSlice.actions;
export default removalsSlice.reducer;
