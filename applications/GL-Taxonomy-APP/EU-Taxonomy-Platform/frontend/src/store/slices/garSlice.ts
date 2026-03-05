import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { GARDetail, BTARCalculation, SectorGAR, EBATemplateData } from '../../types';
import { garApi } from '../../services/api';
import type { RootState } from '../index';

interface GARState {
  stockGAR: GARDetail | null;
  flowGAR: GARDetail | null;
  btar: BTARCalculation | null;
  sectorBreakdown: SectorGAR[];
  ebaTemplate: EBATemplateData | null;
  loading: boolean;
  error: string | null;
}

const initialState: GARState = {
  stockGAR: null,
  flowGAR: null,
  btar: null,
  sectorBreakdown: [],
  ebaTemplate: null,
  loading: false,
  error: null,
};

export const fetchStockGAR = createAsyncThunk(
  'gar/stock',
  async (params: { organization_id: string; reporting_date: string }) =>
    garApi.stock(params)
);

export const fetchFlowGAR = createAsyncThunk(
  'gar/flow',
  async (params: { organization_id: string; reporting_date: string }) =>
    garApi.flow(params)
);

export const fetchBTAR = createAsyncThunk(
  'gar/btar',
  async ({ orgId, date }: { orgId: string; date: string }) =>
    garApi.btar(orgId, date)
);

export const fetchGARSectors = createAsyncThunk(
  'gar/sectors',
  async ({ orgId, date }: { orgId: string; date: string }) =>
    garApi.sectorBreakdown(orgId, date)
);

export const fetchEBATemplate = createAsyncThunk(
  'gar/ebaTemplate',
  async ({ orgId, templateId }: { orgId: string; templateId: string }) =>
    garApi.ebaTemplate(orgId, templateId)
);

const garSlice = createSlice({
  name: 'gar',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchStockGAR.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchStockGAR.fulfilled, (state, action: PayloadAction<GARDetail>) => {
        state.loading = false;
        state.stockGAR = action.payload;
      })
      .addCase(fetchStockGAR.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch GAR';
      })
      .addCase(fetchFlowGAR.fulfilled, (state, action) => { state.flowGAR = action.payload; })
      .addCase(fetchBTAR.fulfilled, (state, action) => { state.btar = action.payload; })
      .addCase(fetchGARSectors.fulfilled, (state, action) => { state.sectorBreakdown = action.payload; })
      .addCase(fetchEBATemplate.fulfilled, (state, action) => { state.ebaTemplate = action.payload; });
  },
});

export const { clearError } = garSlice.actions;
export const selectStockGAR = (state: RootState) => state.gar.stockGAR;
export const selectFlowGAR = (state: RootState) => state.gar.flowGAR;
export const selectBTAR = (state: RootState) => state.gar.btar;
export const selectGARSectors = (state: RootState) => state.gar.sectorBreakdown;
export const selectEBATemplate = (state: RootState) => state.gar.ebaTemplate;
export const selectGARLoading = (state: RootState) => state.gar.loading;
export default garSlice.reducer;
