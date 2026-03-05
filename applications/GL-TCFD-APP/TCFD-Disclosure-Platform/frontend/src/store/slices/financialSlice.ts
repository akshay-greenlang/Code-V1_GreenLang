import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { FinancialImpact, FinancialLineItem, MACCDataPoint, NPVResult, MonteCarloResult } from '../../types';
import { financialApi } from '../../services/api';
import type { RootState } from '../index';

interface FinancialState {
  impacts: FinancialImpact[];
  incomeStatement: FinancialLineItem[];
  balanceSheet: FinancialLineItem[];
  cashFlow: FinancialLineItem[];
  maccData: MACCDataPoint[];
  npvResult: NPVResult | null;
  carbonSensitivity: { carbon_price: number; financial_impact: number; scenario: string }[];
  monteCarloResult: MonteCarloResult | null;
  financialSummary: { scenario: string; revenue_impact: number; cost_impact: number; net_impact: number }[];
  selectedScenarioId: string;
  loading: boolean;
  error: string | null;
}

const initialState: FinancialState = {
  impacts: [],
  incomeStatement: [],
  balanceSheet: [],
  cashFlow: [],
  maccData: [],
  npvResult: null,
  carbonSensitivity: [],
  monteCarloResult: null,
  financialSummary: [],
  selectedScenarioId: '',
  loading: false,
  error: null,
};

export const fetchFinancialImpacts = createAsyncThunk(
  'financial/fetchImpacts',
  async ({ orgId, scenarioId }: { orgId: string; scenarioId: string }) =>
    financialApi.getImpacts(orgId, scenarioId)
);

export const fetchIncomeStatement = createAsyncThunk(
  'financial/fetchIncomeStatement',
  async ({ orgId, scenarioId }: { orgId: string; scenarioId: string }) =>
    financialApi.getIncomeStatement(orgId, scenarioId)
);

export const fetchBalanceSheet = createAsyncThunk(
  'financial/fetchBalanceSheet',
  async ({ orgId, scenarioId }: { orgId: string; scenarioId: string }) =>
    financialApi.getBalanceSheet(orgId, scenarioId)
);

export const fetchCashFlow = createAsyncThunk(
  'financial/fetchCashFlow',
  async ({ orgId, scenarioId }: { orgId: string; scenarioId: string }) =>
    financialApi.getCashFlow(orgId, scenarioId)
);

export const fetchMACCData = createAsyncThunk(
  'financial/fetchMACCData',
  async (orgId: string) => financialApi.getMACCData(orgId)
);

export const fetchNPVAnalysis = createAsyncThunk(
  'financial/fetchNPVAnalysis',
  async ({ orgId, scenarioId }: { orgId: string; scenarioId: string }) =>
    financialApi.getNPVAnalysis(orgId, scenarioId)
);

export const fetchCarbonSensitivity = createAsyncThunk(
  'financial/fetchCarbonSensitivity',
  async (orgId: string) => financialApi.getCarbonSensitivity(orgId)
);

export const fetchMonteCarloResults = createAsyncThunk(
  'financial/fetchMonteCarloResults',
  async ({ orgId, scenarioId }: { orgId: string; scenarioId: string }) =>
    financialApi.getMonteCarloResults(orgId, scenarioId)
);

export const fetchFinancialSummary = createAsyncThunk(
  'financial/fetchFinancialSummary',
  async (orgId: string) => financialApi.getFinancialSummary(orgId)
);

const financialSlice = createSlice({
  name: 'financial',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    setSelectedScenarioId(state, action: PayloadAction<string>) {
      state.selectedScenarioId = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFinancialImpacts.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchFinancialImpacts.fulfilled, (state, action: PayloadAction<FinancialImpact[]>) => {
        state.loading = false;
        state.impacts = action.payload;
      })
      .addCase(fetchFinancialImpacts.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch financial impacts';
      })
      .addCase(fetchIncomeStatement.fulfilled, (state, action: PayloadAction<FinancialLineItem[]>) => {
        state.incomeStatement = action.payload;
      })
      .addCase(fetchBalanceSheet.fulfilled, (state, action: PayloadAction<FinancialLineItem[]>) => {
        state.balanceSheet = action.payload;
      })
      .addCase(fetchCashFlow.fulfilled, (state, action: PayloadAction<FinancialLineItem[]>) => {
        state.cashFlow = action.payload;
      })
      .addCase(fetchMACCData.fulfilled, (state, action: PayloadAction<MACCDataPoint[]>) => {
        state.maccData = action.payload;
      })
      .addCase(fetchNPVAnalysis.fulfilled, (state, action: PayloadAction<NPVResult>) => {
        state.npvResult = action.payload;
      })
      .addCase(fetchCarbonSensitivity.fulfilled, (state, action) => {
        state.carbonSensitivity = action.payload;
      })
      .addCase(fetchMonteCarloResults.fulfilled, (state, action: PayloadAction<MonteCarloResult>) => {
        state.monteCarloResult = action.payload;
      })
      .addCase(fetchFinancialSummary.fulfilled, (state, action) => {
        state.financialSummary = action.payload;
      });
  },
});

export const { clearError, setSelectedScenarioId } = financialSlice.actions;
export const selectFinancialImpacts = (state: RootState) => state.financial.impacts;
export const selectIncomeStatement = (state: RootState) => state.financial.incomeStatement;
export const selectBalanceSheet = (state: RootState) => state.financial.balanceSheet;
export const selectCashFlow = (state: RootState) => state.financial.cashFlow;
export const selectMACCData = (state: RootState) => state.financial.maccData;
export const selectNPVResult = (state: RootState) => state.financial.npvResult;
export const selectCarbonSensitivity = (state: RootState) => state.financial.carbonSensitivity;
export const selectMonteCarloResult = (state: RootState) => state.financial.monteCarloResult;
export const selectFinancialSummary = (state: RootState) => state.financial.financialSummary;
export const selectFinancialLoading = (state: RootState) => state.financial.loading;
export default financialSlice.reducer;
