import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { RiskManagementRecord, RiskIndicator, HeatMapCell, PaginatedResponse } from '../../types';
import { riskMgmtApi } from '../../services/api';
import type { RootState } from '../index';

interface RiskMgmtState {
  register: RiskManagementRecord[];
  heatMap: HeatMapCell[];
  indicators: RiskIndicator[];
  ermStatus: { category: string; integrated: boolean; status: string; last_sync: string }[];
  total: number;
  loading: boolean;
  error: string | null;
}

const initialState: RiskMgmtState = {
  register: [],
  heatMap: [],
  indicators: [],
  ermStatus: [],
  total: 0,
  loading: false,
  error: null,
};

export const fetchRiskRegister = createAsyncThunk(
  'riskMgmt/fetchRegister',
  async (orgId: string) => riskMgmtApi.getRegister(orgId)
);

export const fetchHeatMap = createAsyncThunk(
  'riskMgmt/fetchHeatMap',
  async (orgId: string) => riskMgmtApi.getHeatMap(orgId)
);

export const fetchIndicators = createAsyncThunk(
  'riskMgmt/fetchIndicators',
  async (orgId: string) => riskMgmtApi.getIndicators(orgId)
);

export const fetchERMStatus = createAsyncThunk(
  'riskMgmt/fetchERMStatus',
  async (orgId: string) => riskMgmtApi.getERMStatus(orgId)
);

export const updateRecord = createAsyncThunk(
  'riskMgmt/updateRecord',
  async ({ id, data }: { id: string; data: Partial<RiskManagementRecord> }) =>
    riskMgmtApi.updateRecord(id, data)
);

const riskMgmtSlice = createSlice({
  name: 'riskMgmt',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchRiskRegister.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchRiskRegister.fulfilled, (state, action: PayloadAction<PaginatedResponse<RiskManagementRecord>>) => {
        state.loading = false;
        state.register = action.payload.items;
        state.total = action.payload.total;
      })
      .addCase(fetchRiskRegister.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch risk register';
      })
      .addCase(fetchHeatMap.fulfilled, (state, action: PayloadAction<HeatMapCell[]>) => {
        state.heatMap = action.payload;
      })
      .addCase(fetchIndicators.fulfilled, (state, action: PayloadAction<RiskIndicator[]>) => {
        state.indicators = action.payload;
      })
      .addCase(fetchERMStatus.fulfilled, (state, action) => {
        state.ermStatus = action.payload;
      })
      .addCase(updateRecord.fulfilled, (state, action: PayloadAction<RiskManagementRecord>) => {
        const idx = state.register.findIndex((r) => r.id === action.payload.id);
        if (idx >= 0) state.register[idx] = action.payload;
      });
  },
});

export const { clearError } = riskMgmtSlice.actions;
export const selectRiskRegister = (state: RootState) => state.riskMgmt.register;
export const selectHeatMap = (state: RootState) => state.riskMgmt.heatMap;
export const selectIndicators = (state: RootState) => state.riskMgmt.indicators;
export const selectERMStatus = (state: RootState) => state.riskMgmt.ermStatus;
export const selectRiskMgmtLoading = (state: RootState) => state.riskMgmt.loading;
export default riskMgmtSlice.reducer;
