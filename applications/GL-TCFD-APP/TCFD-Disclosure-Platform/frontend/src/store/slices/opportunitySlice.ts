import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ClimateOpportunity, PaginatedResponse } from '../../types';
import { opportunityApi } from '../../services/api';
import type { RootState } from '../index';

interface OpportunityState {
  opportunities: ClimateOpportunity[];
  pipeline: { stage: string; opportunities: ClimateOpportunity[] }[];
  roiAnalysis: { id: string; name: string; investment: number; npv: number; irr: number; payback: number }[];
  revenueSizing: { type: string; low: number; mid: number; high: number }[];
  costSavings: { category: string; current_cost: number; savings: number; investment: number }[];
  priorityMatrix: { id: string; name: string; impact: number; feasibility: number; size: number; type: string }[];
  total: number;
  loading: boolean;
  error: string | null;
}

const initialState: OpportunityState = {
  opportunities: [],
  pipeline: [],
  roiAnalysis: [],
  revenueSizing: [],
  costSavings: [],
  priorityMatrix: [],
  total: 0,
  loading: false,
  error: null,
};

export const fetchOpportunities = createAsyncThunk(
  'opportunity/fetchOpportunities',
  async ({ orgId, params }: { orgId: string; params?: { type?: string; status?: string } }) =>
    opportunityApi.getOpportunities(orgId, params)
);

export const fetchPipeline = createAsyncThunk(
  'opportunity/fetchPipeline',
  async (orgId: string) => opportunityApi.getPipeline(orgId)
);

export const fetchROIAnalysis = createAsyncThunk(
  'opportunity/fetchROIAnalysis',
  async (orgId: string) => opportunityApi.getROIAnalysis(orgId)
);

export const fetchRevenueSizing = createAsyncThunk(
  'opportunity/fetchRevenueSizing',
  async (orgId: string) => opportunityApi.getRevenueSizing(orgId)
);

export const fetchCostSavings = createAsyncThunk(
  'opportunity/fetchCostSavings',
  async (orgId: string) => opportunityApi.getCostSavings(orgId)
);

export const fetchPriorityMatrix = createAsyncThunk(
  'opportunity/fetchPriorityMatrix',
  async (orgId: string) => opportunityApi.getPriorityMatrix(orgId)
);

const opportunitySlice = createSlice({
  name: 'opportunity',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOpportunities.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchOpportunities.fulfilled, (state, action: PayloadAction<PaginatedResponse<ClimateOpportunity>>) => {
        state.loading = false;
        state.opportunities = action.payload.items;
        state.total = action.payload.total;
      })
      .addCase(fetchOpportunities.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch opportunities';
      })
      .addCase(fetchPipeline.fulfilled, (state, action) => { state.pipeline = action.payload; })
      .addCase(fetchROIAnalysis.fulfilled, (state, action) => { state.roiAnalysis = action.payload; })
      .addCase(fetchRevenueSizing.fulfilled, (state, action) => { state.revenueSizing = action.payload; })
      .addCase(fetchCostSavings.fulfilled, (state, action) => { state.costSavings = action.payload; })
      .addCase(fetchPriorityMatrix.fulfilled, (state, action) => { state.priorityMatrix = action.payload; });
  },
});

export const { clearError } = opportunitySlice.actions;
export const selectOpportunities = (state: RootState) => state.opportunity.opportunities;
export const selectPipeline = (state: RootState) => state.opportunity.pipeline;
export const selectROIAnalysis = (state: RootState) => state.opportunity.roiAnalysis;
export const selectRevenueSizing = (state: RootState) => state.opportunity.revenueSizing;
export const selectCostSavings = (state: RootState) => state.opportunity.costSavings;
export const selectPriorityMatrix = (state: RootState) => state.opportunity.priorityMatrix;
export const selectOpportunityLoading = (state: RootState) => state.opportunity.loading;
export default opportunitySlice.reducer;
