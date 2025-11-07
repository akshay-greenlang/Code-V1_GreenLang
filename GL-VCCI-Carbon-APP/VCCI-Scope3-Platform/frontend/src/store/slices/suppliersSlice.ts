import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';
import type { Supplier, PaginatedResponse } from '../../types';

interface SuppliersState {
  suppliers: Supplier[];
  selectedSupplier: Supplier | null;
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
  filters: {
    search?: string;
    status?: string[];
  };
  loading: boolean;
  error: string | null;
}

const initialState: SuppliersState = {
  suppliers: [],
  selectedSupplier: null,
  pagination: {
    page: 1,
    pageSize: 25,
    total: 0,
    totalPages: 0,
  },
  filters: {},
  loading: false,
  error: null,
};

// Async thunks
export const fetchSuppliers = createAsyncThunk(
  'suppliers/fetchAll',
  async (params?: {
    page?: number;
    pageSize?: number;
    search?: string;
    status?: string[];
  }) => {
    const response = await api.getSuppliers(params);
    return response;
  }
);

export const fetchSupplier = createAsyncThunk(
  'suppliers/fetchOne',
  async (id: string) => {
    const supplier = await api.getSupplier(id);
    return supplier;
  }
);

export const createCampaign = createAsyncThunk(
  'suppliers/createCampaign',
  async ({ supplierIds, message }: { supplierIds: string[]; message?: string }) => {
    const response = await api.createEngagementCampaign(supplierIds, message);
    return response;
  }
);

const suppliersSlice = createSlice({
  name: 'suppliers',
  initialState,
  reducers: {
    setPage: (state, action: PayloadAction<number>) => {
      state.pagination.page = action.payload;
    },
    setPageSize: (state, action: PayloadAction<number>) => {
      state.pagination.pageSize = action.payload;
    },
    setFilters: (state, action: PayloadAction<SuppliersState['filters']>) => {
      state.filters = action.payload;
      state.pagination.page = 1;
    },
  },
  extraReducers: (builder) => {
    // Fetch all suppliers
    builder
      .addCase(fetchSuppliers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSuppliers.fulfilled, (state, action: PayloadAction<PaginatedResponse<Supplier>>) => {
        state.loading = false;
        state.suppliers = action.payload.data;
        state.pagination = action.payload.pagination;
      })
      .addCase(fetchSuppliers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch suppliers';
      });

    // Fetch single supplier
    builder
      .addCase(fetchSupplier.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSupplier.fulfilled, (state, action: PayloadAction<Supplier>) => {
        state.loading = false;
        state.selectedSupplier = action.payload;
      })
      .addCase(fetchSupplier.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch supplier';
      });

    // Create campaign
    builder
      .addCase(createCampaign.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createCampaign.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(createCampaign.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to create campaign';
      });
  },
});

export const { setPage, setPageSize, setFilters } = suppliersSlice.actions;
export default suppliersSlice.reducer;
