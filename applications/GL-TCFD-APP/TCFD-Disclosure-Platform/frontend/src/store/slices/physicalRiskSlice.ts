import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { PhysicalRiskAssessment, AssetLocation, InsuranceCostProjection, SupplyChainRiskNode } from '../../types';
import { physicalRiskApi } from '../../services/api';
import type { RootState } from '../index';

interface PhysicalRiskState {
  assessment: PhysicalRiskAssessment | null;
  assets: AssetLocation[];
  hazardSummary: { hazard_type: string; asset_count: number; total_exposure: number }[];
  insuranceProjections: InsuranceCostProjection[];
  supplyChainRisks: SupplyChainRiskNode[];
  loading: boolean;
  error: string | null;
}

const initialState: PhysicalRiskState = {
  assessment: null,
  assets: [],
  hazardSummary: [],
  insuranceProjections: [],
  supplyChainRisks: [],
  loading: false,
  error: null,
};

export const fetchPhysicalRiskAssessment = createAsyncThunk(
  'physicalRisk/fetchAssessment',
  async (orgId: string) => physicalRiskApi.getAssessment(orgId)
);

export const fetchAssets = createAsyncThunk(
  'physicalRisk/fetchAssets',
  async (orgId: string) => physicalRiskApi.getAssets(orgId)
);

export const fetchHazardAnalysis = createAsyncThunk(
  'physicalRisk/fetchHazardAnalysis',
  async (orgId: string) => physicalRiskApi.getHazardAnalysis(orgId)
);

export const fetchInsuranceProjections = createAsyncThunk(
  'physicalRisk/fetchInsuranceProjections',
  async (orgId: string) => physicalRiskApi.getInsuranceProjections(orgId)
);

export const fetchSupplyChainRisks = createAsyncThunk(
  'physicalRisk/fetchSupplyChainRisks',
  async (orgId: string) => physicalRiskApi.getSupplyChainRisks(orgId)
);

const physicalRiskSlice = createSlice({
  name: 'physicalRisk',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPhysicalRiskAssessment.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchPhysicalRiskAssessment.fulfilled, (state, action: PayloadAction<PhysicalRiskAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
        state.assets = action.payload.assets;
        state.supplyChainRisks = action.payload.supply_chain_risks;
      })
      .addCase(fetchPhysicalRiskAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch physical risk assessment';
      })
      .addCase(fetchAssets.fulfilled, (state, action: PayloadAction<AssetLocation[]>) => {
        state.assets = action.payload;
      })
      .addCase(fetchHazardAnalysis.fulfilled, (state, action) => {
        state.hazardSummary = action.payload;
      })
      .addCase(fetchInsuranceProjections.fulfilled, (state, action: PayloadAction<InsuranceCostProjection[]>) => {
        state.insuranceProjections = action.payload;
      })
      .addCase(fetchSupplyChainRisks.fulfilled, (state, action: PayloadAction<SupplyChainRiskNode[]>) => {
        state.supplyChainRisks = action.payload;
      });
  },
});

export const { clearError } = physicalRiskSlice.actions;
export const selectPhysicalRiskAssessment = (state: RootState) => state.physicalRisk.assessment;
export const selectAssets = (state: RootState) => state.physicalRisk.assets;
export const selectHazardSummary = (state: RootState) => state.physicalRisk.hazardSummary;
export const selectInsuranceProjections = (state: RootState) => state.physicalRisk.insuranceProjections;
export const selectSupplyChainRisks = (state: RootState) => state.physicalRisk.supplyChainRisks;
export const selectPhysicalRiskLoading = (state: RootState) => state.physicalRisk.loading;
export default physicalRiskSlice.reducer;
