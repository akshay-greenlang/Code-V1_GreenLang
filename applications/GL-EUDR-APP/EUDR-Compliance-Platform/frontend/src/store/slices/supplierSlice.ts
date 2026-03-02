/**
 * Supplier Redux Slice
 *
 * Manages supplier CRUD operations, pagination, filtering, bulk import,
 * and per-supplier compliance/risk data.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  Supplier,
  SupplierCreateRequest,
  SupplierUpdateRequest,
  SupplierFilterParams,
  PaginatedResponse,
  RiskAssessment,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface SupplierState {
  suppliers: Supplier[];
  selectedSupplier: Supplier | null;
  selectedSupplierRisk: RiskAssessment | null;
  pagination: {
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
  };
  filters: SupplierFilterParams;
  loading: boolean;
  detailLoading: boolean;
  saving: boolean;
  error: string | null;
  bulkImportResult: { imported: number; errors: string[] } | null;
}

const initialState: SupplierState = {
  suppliers: [],
  selectedSupplier: null,
  selectedSupplierRisk: null,
  pagination: { total: 0, page: 1, per_page: 25, total_pages: 0 },
  filters: {},
  loading: false,
  detailLoading: false,
  saving: false,
  error: null,
  bulkImportResult: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const fetchSuppliers = createAsyncThunk(
  'suppliers/fetchAll',
  async (params: SupplierFilterParams | undefined, { rejectWithValue }) => {
    try {
      return await apiClient.getSuppliers(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch suppliers';
      return rejectWithValue(message);
    }
  }
);

export const fetchSupplier = createAsyncThunk(
  'suppliers/fetchOne',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.getSupplier(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch supplier';
      return rejectWithValue(message);
    }
  }
);

export const createSupplier = createAsyncThunk(
  'suppliers/create',
  async (data: SupplierCreateRequest, { rejectWithValue }) => {
    try {
      return await apiClient.createSupplier(data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to create supplier';
      return rejectWithValue(message);
    }
  }
);

export const updateSupplier = createAsyncThunk(
  'suppliers/update',
  async (
    { id, data }: { id: string; data: SupplierUpdateRequest },
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.updateSupplier(id, data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to update supplier';
      return rejectWithValue(message);
    }
  }
);

export const deleteSupplier = createAsyncThunk(
  'suppliers/delete',
  async (id: string, { rejectWithValue }) => {
    try {
      await apiClient.deleteSupplier(id);
      return id;
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to delete supplier';
      return rejectWithValue(message);
    }
  }
);

export const bulkImportSuppliers = createAsyncThunk(
  'suppliers/bulkImport',
  async (data: SupplierCreateRequest[], { rejectWithValue }) => {
    try {
      return await apiClient.bulkImportSuppliers(data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Bulk import failed';
      return rejectWithValue(message);
    }
  }
);

export const fetchSupplierRisk = createAsyncThunk(
  'suppliers/fetchRisk',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.getSupplierRisk(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch supplier risk';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const supplierSlice = createSlice({
  name: 'suppliers',
  initialState,
  reducers: {
    clearSupplierError(state) {
      state.error = null;
    },
    clearSelectedSupplier(state) {
      state.selectedSupplier = null;
      state.selectedSupplierRisk = null;
    },
    setSupplierFilters(state, action: PayloadAction<SupplierFilterParams>) {
      state.filters = action.payload;
    },
    clearBulkImportResult(state) {
      state.bulkImportResult = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch all
    builder
      .addCase(fetchSuppliers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(
        fetchSuppliers.fulfilled,
        (state, action: PayloadAction<PaginatedResponse<Supplier>>) => {
          state.loading = false;
          state.suppliers = action.payload.items;
          state.pagination = {
            total: action.payload.total,
            page: action.payload.page,
            per_page: action.payload.per_page,
            total_pages: action.payload.total_pages,
          };
        }
      )
      .addCase(fetchSuppliers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Fetch one
    builder
      .addCase(fetchSupplier.pending, (state) => {
        state.detailLoading = true;
        state.error = null;
      })
      .addCase(fetchSupplier.fulfilled, (state, action) => {
        state.detailLoading = false;
        state.selectedSupplier = action.payload;
      })
      .addCase(fetchSupplier.rejected, (state, action) => {
        state.detailLoading = false;
        state.error = action.payload as string;
      });

    // Create
    builder
      .addCase(createSupplier.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(createSupplier.fulfilled, (state, action) => {
        state.saving = false;
        state.suppliers.unshift(action.payload);
        state.pagination.total += 1;
      })
      .addCase(createSupplier.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });

    // Update
    builder
      .addCase(updateSupplier.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(updateSupplier.fulfilled, (state, action) => {
        state.saving = false;
        const idx = state.suppliers.findIndex((s) => s.id === action.payload.id);
        if (idx !== -1) state.suppliers[idx] = action.payload;
        if (state.selectedSupplier?.id === action.payload.id) {
          state.selectedSupplier = action.payload;
        }
      })
      .addCase(updateSupplier.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });

    // Delete
    builder
      .addCase(deleteSupplier.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(deleteSupplier.fulfilled, (state, action) => {
        state.saving = false;
        state.suppliers = state.suppliers.filter((s) => s.id !== action.payload);
        state.pagination.total -= 1;
        if (state.selectedSupplier?.id === action.payload) {
          state.selectedSupplier = null;
        }
      })
      .addCase(deleteSupplier.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });

    // Bulk import
    builder
      .addCase(bulkImportSuppliers.pending, (state) => {
        state.saving = true;
        state.error = null;
        state.bulkImportResult = null;
      })
      .addCase(bulkImportSuppliers.fulfilled, (state, action) => {
        state.saving = false;
        state.bulkImportResult = action.payload;
      })
      .addCase(bulkImportSuppliers.rejected, (state, action) => {
        state.saving = false;
        state.error = action.payload as string;
      });

    // Risk
    builder
      .addCase(fetchSupplierRisk.pending, (state) => {
        state.detailLoading = true;
      })
      .addCase(fetchSupplierRisk.fulfilled, (state, action) => {
        state.detailLoading = false;
        state.selectedSupplierRisk = action.payload;
      })
      .addCase(fetchSupplierRisk.rejected, (state, action) => {
        state.detailLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  clearSupplierError,
  clearSelectedSupplier,
  setSupplierFilters,
  clearBulkImportResult,
} = supplierSlice.actions;

export default supplierSlice.reducer;
