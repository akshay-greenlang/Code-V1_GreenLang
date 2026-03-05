import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { TransitionRiskAssessment, PolicyRisk, TechnologyRisk, MarketRisk, ReputationRisk } from '../../types';
import { transitionRiskApi } from '../../services/api';
import type { RootState } from '../index';

interface TransitionRiskState {
  assessment: TransitionRiskAssessment | null;
  policyRisks: PolicyRisk[];
  technologyRisks: TechnologyRisk[];
  marketRisks: MarketRisk[];
  reputationRisks: ReputationRisk[];
  strandedAssets: { asset: string; book_value: number; stranded_value: number; timeline: string }[];
  complianceTimeline: { regulation: string; jurisdiction: string; effective_date: string; status: string }[];
  loading: boolean;
  error: string | null;
}

const initialState: TransitionRiskState = {
  assessment: null,
  policyRisks: [],
  technologyRisks: [],
  marketRisks: [],
  reputationRisks: [],
  strandedAssets: [],
  complianceTimeline: [],
  loading: false,
  error: null,
};

export const fetchTransitionRiskAssessment = createAsyncThunk(
  'transitionRisk/fetchAssessment',
  async (orgId: string) => transitionRiskApi.getAssessment(orgId)
);

export const fetchPolicyRisks = createAsyncThunk(
  'transitionRisk/fetchPolicyRisks',
  async (orgId: string) => transitionRiskApi.getPolicyRisks(orgId)
);

export const fetchTechnologyRisks = createAsyncThunk(
  'transitionRisk/fetchTechnologyRisks',
  async (orgId: string) => transitionRiskApi.getTechnologyRisks(orgId)
);

export const fetchMarketRisks = createAsyncThunk(
  'transitionRisk/fetchMarketRisks',
  async (orgId: string) => transitionRiskApi.getMarketRisks(orgId)
);

export const fetchReputationRisks = createAsyncThunk(
  'transitionRisk/fetchReputationRisks',
  async (orgId: string) => transitionRiskApi.getReputationRisks(orgId)
);

export const fetchStrandedAssets = createAsyncThunk(
  'transitionRisk/fetchStrandedAssets',
  async (orgId: string) => transitionRiskApi.getStrandedAssets(orgId)
);

export const fetchComplianceTimeline = createAsyncThunk(
  'transitionRisk/fetchComplianceTimeline',
  async (orgId: string) => transitionRiskApi.getComplianceTimeline(orgId)
);

const transitionRiskSlice = createSlice({
  name: 'transitionRisk',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTransitionRiskAssessment.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchTransitionRiskAssessment.fulfilled, (state, action: PayloadAction<TransitionRiskAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
        state.policyRisks = action.payload.policy_risks;
        state.technologyRisks = action.payload.technology_risks;
        state.marketRisks = action.payload.market_risks;
        state.reputationRisks = action.payload.reputation_risks;
      })
      .addCase(fetchTransitionRiskAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch transition risk assessment';
      })
      .addCase(fetchPolicyRisks.fulfilled, (state, action: PayloadAction<PolicyRisk[]>) => {
        state.policyRisks = action.payload;
      })
      .addCase(fetchTechnologyRisks.fulfilled, (state, action: PayloadAction<TechnologyRisk[]>) => {
        state.technologyRisks = action.payload;
      })
      .addCase(fetchMarketRisks.fulfilled, (state, action: PayloadAction<MarketRisk[]>) => {
        state.marketRisks = action.payload;
      })
      .addCase(fetchReputationRisks.fulfilled, (state, action: PayloadAction<ReputationRisk[]>) => {
        state.reputationRisks = action.payload;
      })
      .addCase(fetchStrandedAssets.fulfilled, (state, action) => {
        state.strandedAssets = action.payload;
      })
      .addCase(fetchComplianceTimeline.fulfilled, (state, action) => {
        state.complianceTimeline = action.payload;
      });
  },
});

export const { clearError } = transitionRiskSlice.actions;
export const selectTransitionAssessment = (state: RootState) => state.transitionRisk.assessment;
export const selectPolicyRisks = (state: RootState) => state.transitionRisk.policyRisks;
export const selectTechnologyRisks = (state: RootState) => state.transitionRisk.technologyRisks;
export const selectMarketRisks = (state: RootState) => state.transitionRisk.marketRisks;
export const selectReputationRisks = (state: RootState) => state.transitionRisk.reputationRisks;
export const selectStrandedAssets = (state: RootState) => state.transitionRisk.strandedAssets;
export const selectComplianceTimeline = (state: RootState) => state.transitionRisk.complianceTimeline;
export const selectTransitionRiskLoading = (state: RootState) => state.transitionRisk.loading;
export default transitionRiskSlice.reducer;
