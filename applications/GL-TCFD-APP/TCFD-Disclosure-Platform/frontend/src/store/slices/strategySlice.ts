import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ClimateRisk, ClimateOpportunity, BusinessModelImpact, ValueChainNode, PaginatedResponse } from '../../types';
import { strategyApi } from '../../services/api';
import type { RootState } from '../index';

interface StrategyState {
  risks: ClimateRisk[];
  opportunities: ClimateOpportunity[];
  businessModelImpacts: BusinessModelImpact[];
  valueChain: ValueChainNode[];
  riskTotal: number;
  opportunityTotal: number;
  loading: boolean;
  error: string | null;
}

const initialState: StrategyState = {
  risks: [],
  opportunities: [],
  businessModelImpacts: [],
  valueChain: [],
  riskTotal: 0,
  opportunityTotal: 0,
  loading: false,
  error: null,
};

export const fetchRisks = createAsyncThunk(
  'strategy/fetchRisks',
  async ({ orgId, params }: { orgId: string; params?: { category?: string; level?: string; time_horizon?: string } }) =>
    strategyApi.getRisks(orgId, params)
);

export const fetchOpportunities = createAsyncThunk(
  'strategy/fetchOpportunities',
  async ({ orgId, params }: { orgId: string; params?: { type?: string; status?: string } }) =>
    strategyApi.getOpportunities(orgId, params)
);

export const fetchBusinessModelImpacts = createAsyncThunk(
  'strategy/fetchBusinessModelImpacts',
  async (orgId: string) => strategyApi.getBusinessModelImpacts(orgId)
);

export const fetchValueChain = createAsyncThunk(
  'strategy/fetchValueChain',
  async (orgId: string) => strategyApi.getValueChain(orgId)
);

export const createRisk = createAsyncThunk(
  'strategy/createRisk',
  async (data: Partial<ClimateRisk>) => strategyApi.createRisk(data)
);

export const updateRisk = createAsyncThunk(
  'strategy/updateRisk',
  async ({ id, data }: { id: string; data: Partial<ClimateRisk> }) => strategyApi.updateRisk(id, data)
);

export const createOpportunity = createAsyncThunk(
  'strategy/createOpportunity',
  async (data: Partial<ClimateOpportunity>) => strategyApi.createOpportunity(data)
);

const strategySlice = createSlice({
  name: 'strategy',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchRisks.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchRisks.fulfilled, (state, action: PayloadAction<PaginatedResponse<ClimateRisk>>) => {
        state.loading = false;
        state.risks = action.payload.items;
        state.riskTotal = action.payload.total;
      })
      .addCase(fetchRisks.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch risks';
      })
      .addCase(fetchOpportunities.fulfilled, (state, action: PayloadAction<PaginatedResponse<ClimateOpportunity>>) => {
        state.opportunities = action.payload.items;
        state.opportunityTotal = action.payload.total;
      })
      .addCase(fetchBusinessModelImpacts.fulfilled, (state, action: PayloadAction<BusinessModelImpact[]>) => {
        state.businessModelImpacts = action.payload;
      })
      .addCase(fetchValueChain.fulfilled, (state, action: PayloadAction<ValueChainNode[]>) => {
        state.valueChain = action.payload;
      })
      .addCase(createRisk.fulfilled, (state, action: PayloadAction<ClimateRisk>) => {
        state.risks.push(action.payload);
        state.riskTotal += 1;
      })
      .addCase(updateRisk.fulfilled, (state, action: PayloadAction<ClimateRisk>) => {
        const idx = state.risks.findIndex((r) => r.id === action.payload.id);
        if (idx >= 0) state.risks[idx] = action.payload;
      })
      .addCase(createOpportunity.fulfilled, (state, action: PayloadAction<ClimateOpportunity>) => {
        state.opportunities.push(action.payload);
        state.opportunityTotal += 1;
      });
  },
});

export const { clearError } = strategySlice.actions;
export const selectRisks = (state: RootState) => state.strategy.risks;
export const selectOpportunities = (state: RootState) => state.strategy.opportunities;
export const selectBusinessModelImpacts = (state: RootState) => state.strategy.businessModelImpacts;
export const selectValueChain = (state: RootState) => state.strategy.valueChain;
export const selectStrategyLoading = (state: RootState) => state.strategy.loading;
export default strategySlice.reducer;
