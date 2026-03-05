/**
 * Supply Chain Redux Slice
 *
 * Manages supply chain module state: supplier list, invitations,
 * response tracking, engagement summary, and emission hotspots.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  SupplyChainState,
  SupplierRequest,
  SupplierResponse,
  SupplyChainSummary,
  InviteSupplierRequest,
} from '../../types';
import { cdpApi } from '../../services/api';

const initialState: SupplyChainState = {
  suppliers: [],
  supplierResponses: [],
  summary: null,
  loading: false,
  error: null,
};

export const fetchSuppliers = createAsyncThunk<SupplierRequest[], string>(
  'supplyChain/fetchSuppliers',
  async (orgId) => cdpApi.getSuppliers(orgId),
);

export const inviteSupplier = createAsyncThunk<
  SupplierRequest,
  { orgId: string; payload: InviteSupplierRequest }
>(
  'supplyChain/invite',
  async ({ orgId, payload }) => cdpApi.inviteSupplier(orgId, payload),
);

export const fetchSupplierResponses = createAsyncThunk<SupplierResponse[], string>(
  'supplyChain/fetchResponses',
  async (orgId) => cdpApi.getSupplierResponses(orgId),
);

export const fetchSupplyChainSummary = createAsyncThunk<SupplyChainSummary, string>(
  'supplyChain/fetchSummary',
  async (orgId) => cdpApi.getSupplyChainSummary(orgId),
);

export const removeSupplier = createAsyncThunk<
  string,
  { orgId: string; supplierId: string }
>(
  'supplyChain/remove',
  async ({ orgId, supplierId }) => {
    await cdpApi.removeSupplier(orgId, supplierId);
    return supplierId;
  },
);

const supplyChainSlice = createSlice({
  name: 'supplyChain',
  initialState,
  reducers: {
    clearSupplyChain: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSuppliers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSuppliers.fulfilled, (state, action) => {
        state.loading = false;
        state.suppliers = action.payload;
      })
      .addCase(fetchSuppliers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load suppliers';
      })
      .addCase(inviteSupplier.fulfilled, (state, action) => {
        state.suppliers.push(action.payload);
      })
      .addCase(fetchSupplierResponses.fulfilled, (state, action) => {
        state.supplierResponses = action.payload;
      })
      .addCase(fetchSupplyChainSummary.fulfilled, (state, action) => {
        state.summary = action.payload;
      })
      .addCase(removeSupplier.fulfilled, (state, action) => {
        state.suppliers = state.suppliers.filter((s) => s.id !== action.payload);
      });
  },
});

export const { clearSupplyChain } = supplyChainSlice.actions;
export default supplyChainSlice.reducer;
