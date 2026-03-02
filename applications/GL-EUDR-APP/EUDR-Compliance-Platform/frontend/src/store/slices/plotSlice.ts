/**
 * Plot Redux Slice
 *
 * Manages geospatial plot registry state: CRUD, validation against
 * deforestation cutoff, overlap checking, and bulk import.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  Plot,
  PlotCreateRequest,
  PlotFilterParams,
  PlotValidationResult,
  PaginatedResponse,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface PlotState {
  plots: Plot[];
  selectedPlot: Plot | null;
  validationResult: PlotValidationResult | null;
  overlapResult: { overlaps: Array<{ plot_id: string; overlap_area_hectares: number }> } | null;
  pagination: {
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
  };
  filters: PlotFilterParams;
  loading: boolean;
  detailLoading: boolean;
  validating: boolean;
  saving: boolean;
  error: string | null;
}

const initialState: PlotState = {
  plots: [],
  selectedPlot: null,
  validationResult: null,
  overlapResult: null,
  pagination: { total: 0, page: 1, per_page: 25, total_pages: 0 },
  filters: {},
  loading: false,
  detailLoading: false,
  validating: false,
  saving: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const fetchPlots = createAsyncThunk(
  'plots/fetchAll',
  async (params: PlotFilterParams | undefined, { rejectWithValue }) => {
    try {
      return await apiClient.getPlots(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch plots';
      return rejectWithValue(message);
    }
  }
);

export const fetchPlot = createAsyncThunk(
  'plots/fetchOne',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.getPlot(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch plot';
      return rejectWithValue(message);
    }
  }
);

export const createPlot = createAsyncThunk(
  'plots/create',
  async (data: PlotCreateRequest, { rejectWithValue }) => {
    try {
      return await apiClient.createPlot(data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to create plot';
      return rejectWithValue(message);
    }
  }
);

export const updatePlot = createAsyncThunk(
  'plots/update',
  async (
    { id, data }: { id: string; data: Partial<PlotCreateRequest> },
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.updatePlot(id, data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to update plot';
      return rejectWithValue(message);
    }
  }
);

export const validatePlot = createAsyncThunk(
  'plots/validate',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.validatePlot(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Plot validation failed';
      return rejectWithValue(message);
    }
  }
);

export const checkPlotOverlaps = createAsyncThunk(
  'plots/checkOverlaps',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.checkPlotOverlaps(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Overlap check failed';
      return rejectWithValue(message);
    }
  }
);

export const bulkImportPlots = createAsyncThunk(
  'plots/bulkImport',
  async (data: PlotCreateRequest[], { rejectWithValue }) => {
    try {
      return await apiClient.bulkImportPlots(data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Bulk import failed';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const plotSlice = createSlice({
  name: 'plots',
  initialState,
  reducers: {
    clearPlotError(state) {
      state.error = null;
    },
    clearSelectedPlot(state) {
      state.selectedPlot = null;
      state.validationResult = null;
      state.overlapResult = null;
    },
    setPlotFilters(state, action: PayloadAction<PlotFilterParams>) {
      state.filters = action.payload;
    },
    clearValidationResult(state) {
      state.validationResult = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch all
    builder
      .addCase(fetchPlots.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(
        fetchPlots.fulfilled,
        (state, action: PayloadAction<PaginatedResponse<Plot>>) => {
          state.loading = false;
          state.plots = action.payload.items;
          state.pagination = {
            total: action.payload.total,
            page: action.payload.page,
            per_page: action.payload.per_page,
            total_pages: action.payload.total_pages,
          };
        }
      )
      .addCase(fetchPlots.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Fetch one
    builder
      .addCase(fetchPlot.pending, (state) => {
        state.detailLoading = true;
        state.error = null;
      })
      .addCase(fetchPlot.fulfilled, (state, action) => {
        state.detailLoading = false;
        state.selectedPlot = action.payload;
      })
      .addCase(fetchPlot.rejected, (state, action) => {
        state.detailLoading = false;
        state.error = action.payload as string;
      });

    // Create
    builder
      .addCase(createPlot.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(createPlot.fulfilled, (state, action) => {
        state.saving = false;
        state.plots.unshift(action.payload);
        state.pagination.total += 1;
      })
      .addCase(createPlot.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });

    // Update
    builder
      .addCase(updatePlot.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(updatePlot.fulfilled, (state, action) => {
        state.saving = false;
        const idx = state.plots.findIndex((p) => p.id === action.payload.id);
        if (idx !== -1) state.plots[idx] = action.payload;
        if (state.selectedPlot?.id === action.payload.id) {
          state.selectedPlot = action.payload;
        }
      })
      .addCase(updatePlot.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });

    // Validate
    builder
      .addCase(validatePlot.pending, (state) => {
        state.validating = true;
        state.error = null;
      })
      .addCase(validatePlot.fulfilled, (state, action) => {
        state.validating = false;
        state.validationResult = action.payload;
      })
      .addCase(validatePlot.rejected, (state, action) => {
        state.validating = false;
        state.error = action.payload as string;
      });

    // Overlap check
    builder
      .addCase(checkPlotOverlaps.pending, (state) => {
        state.validating = true;
        state.error = null;
      })
      .addCase(checkPlotOverlaps.fulfilled, (state, action) => {
        state.validating = false;
        state.overlapResult = action.payload;
      })
      .addCase(checkPlotOverlaps.rejected, (state, action) => {
        state.validating = false;
        state.error = action.payload as string;
      });

    // Bulk import
    builder
      .addCase(bulkImportPlots.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(bulkImportPlots.fulfilled, (state) => {
        state.saving = false;
      })
      .addCase(bulkImportPlots.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  clearPlotError,
  clearSelectedPlot,
  setPlotFilters,
  clearValidationResult,
} = plotSlice.actions;

export default plotSlice.reducer;
