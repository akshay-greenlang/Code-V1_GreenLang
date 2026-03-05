/**
 * Emissions Redux Slice
 *
 * Manages emission source state: source CRUD, quantification results,
 * data quality indicators, and uncertainty analysis for the ISO 14064-1
 * inventory.
 *
 * Async thunks:
 *   - addSource: Add an emission source to an inventory
 *   - fetchSources: Load all emission sources for an inventory
 *   - fetchSourcesByCategory: Load sources filtered by ISO category
 *   - deleteSource: Remove an emission source
 *   - quantifySource: Calculate emissions for a source
 *   - fetchDataQuality: Load data quality indicators
 *   - runUncertainty: Run Monte Carlo uncertainty analysis
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  EmissionsState,
  EmissionSource,
  QuantificationResult,
  DataQualityIndicator,
  UncertaintyResult,
  AddEmissionSourceRequest,
  ISOCategory,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: EmissionsState = {
  sources: [],
  quantificationResults: [],
  dataQuality: null,
  uncertainty: null,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const addSource = createAsyncThunk<
  EmissionSource,
  { inventoryId: string; payload: AddEmissionSourceRequest }
>(
  'emissions/addSource',
  async ({ inventoryId, payload }) => {
    return iso14064Api.addEmissionSource(inventoryId, payload);
  },
);

export const fetchSources = createAsyncThunk<
  EmissionSource[],
  string
>(
  'emissions/fetchSources',
  async (inventoryId) => {
    return iso14064Api.getEmissionSources(inventoryId);
  },
);

export const fetchSourcesByCategory = createAsyncThunk<
  EmissionSource[],
  { inventoryId: string; category: ISOCategory }
>(
  'emissions/fetchSourcesByCategory',
  async ({ inventoryId, category }) => {
    return iso14064Api.getEmissionSourcesByCategory(inventoryId, category);
  },
);

export const deleteSource = createAsyncThunk<
  string,
  { inventoryId: string; sourceId: string }
>(
  'emissions/deleteSource',
  async ({ inventoryId, sourceId }) => {
    await iso14064Api.deleteEmissionSource(inventoryId, sourceId);
    return sourceId;
  },
);

export const quantifySource = createAsyncThunk<
  QuantificationResult,
  { inventoryId: string; sourceId: string }
>(
  'emissions/quantifySource',
  async ({ inventoryId, sourceId }) => {
    return iso14064Api.quantifyEmissions(inventoryId, sourceId);
  },
);

export const fetchDataQuality = createAsyncThunk<
  DataQualityIndicator,
  string
>(
  'emissions/fetchDataQuality',
  async (inventoryId) => {
    return iso14064Api.getDataQuality(inventoryId);
  },
);

export const runUncertainty = createAsyncThunk<
  UncertaintyResult,
  string
>(
  'emissions/runUncertainty',
  async (inventoryId) => {
    return iso14064Api.runUncertaintyAnalysis(inventoryId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const emissionsSlice = createSlice({
  name: 'emissions',
  initialState,
  reducers: {
    clearEmissions: () => initialState,
    clearEmissionsError: (state) => {
      state.error = null;
    },
    updateSourceLocal: (state, action: PayloadAction<EmissionSource>) => {
      const idx = state.sources.findIndex((s) => s.id === action.payload.id);
      if (idx >= 0) {
        state.sources[idx] = action.payload;
      }
    },
    removeSourceLocal: (state, action: PayloadAction<string>) => {
      state.sources = state.sources.filter((s) => s.id !== action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      // -- addSource --
      .addCase(addSource.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(addSource.fulfilled, (state, action) => {
        state.loading = false;
        state.sources.push(action.payload);
      })
      .addCase(addSource.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to add emission source';
      })

      // -- fetchSources --
      .addCase(fetchSources.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSources.fulfilled, (state, action) => {
        state.loading = false;
        state.sources = action.payload;
      })
      .addCase(fetchSources.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch emission sources';
      })

      // -- fetchSourcesByCategory --
      .addCase(fetchSourcesByCategory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSourcesByCategory.fulfilled, (state, action) => {
        state.loading = false;
        state.sources = action.payload;
      })
      .addCase(fetchSourcesByCategory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch sources by category';
      })

      // -- deleteSource --
      .addCase(deleteSource.fulfilled, (state, action) => {
        state.sources = state.sources.filter((s) => s.id !== action.payload);
      })

      // -- quantifySource --
      .addCase(quantifySource.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(quantifySource.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.quantificationResults.findIndex(
          (r) => r.source_id === action.payload.source_id,
        );
        if (idx >= 0) {
          state.quantificationResults[idx] = action.payload;
        } else {
          state.quantificationResults.push(action.payload);
        }
      })
      .addCase(quantifySource.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to quantify emissions';
      })

      // -- fetchDataQuality --
      .addCase(fetchDataQuality.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDataQuality.fulfilled, (state, action) => {
        state.loading = false;
        state.dataQuality = action.payload;
      })
      .addCase(fetchDataQuality.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch data quality';
      })

      // -- runUncertainty --
      .addCase(runUncertainty.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(runUncertainty.fulfilled, (state, action) => {
        state.loading = false;
        state.uncertainty = action.payload;
      })
      .addCase(runUncertainty.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to run uncertainty analysis';
      });
  },
});

export const { clearEmissions, clearEmissionsError, updateSourceLocal, removeSourceLocal } =
  emissionsSlice.actions;
export default emissionsSlice.reducer;
