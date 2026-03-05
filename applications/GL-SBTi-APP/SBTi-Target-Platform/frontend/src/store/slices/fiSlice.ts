import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { FIPortfolio, PortfolioTemperature } from '../../types';
import { fiApi } from '../../services/api';
import type { RootState } from '../index';

interface FIState {
  portfolios: FIPortfolio[];
  selectedPortfolio: FIPortfolio | null;
  coverage: { overall_pct: number; by_asset_class: { asset_class: string; coverage_pct: number; target_2030: number; target_2040: number }[]; path_to_100: { year: number; coverage_pct: number }[] } | null;
  financedEmissions: { total: number; by_asset_class: { asset_class: string; emissions: number; pct: number }[]; by_sector: { sector: string; emissions: number; pct: number }[] } | null;
  waci: { current_waci: number; trend: { year: number; waci: number }[]; benchmark: number } | null;
  portfolioTemperature: PortfolioTemperature | null;
  loading: boolean;
  error: string | null;
}

const initialState: FIState = {
  portfolios: [],
  selectedPortfolio: null,
  coverage: null,
  financedEmissions: null,
  waci: null,
  portfolioTemperature: null,
  loading: false,
  error: null,
};

export const fetchPortfolios = createAsyncThunk(
  'fi/fetchPortfolios',
  async (orgId: string) => fiApi.getPortfolios(orgId)
);

export const fetchPortfolio = createAsyncThunk(
  'fi/fetchPortfolio',
  async (id: string) => fiApi.getPortfolio(id)
);

export const fetchFICoverage = createAsyncThunk(
  'fi/fetchCoverage',
  async (orgId: string) => fiApi.getCoverage(orgId)
);

export const fetchFinancedEmissions = createAsyncThunk(
  'fi/fetchFinancedEmissions',
  async (orgId: string) => fiApi.getFinancedEmissions(orgId)
);

export const fetchWACI = createAsyncThunk(
  'fi/fetchWACI',
  async (orgId: string) => fiApi.getWACI(orgId)
);

export const fetchPortfolioTemperature = createAsyncThunk(
  'fi/fetchPortfolioTemperature',
  async (portfolioId: string) => fiApi.getPortfolioTemperature(portfolioId)
);

const fiSlice = createSlice({
  name: 'fi',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    clearSelectedPortfolio(state) { state.selectedPortfolio = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPortfolios.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchPortfolios.fulfilled, (state, action: PayloadAction<FIPortfolio[]>) => {
        state.loading = false;
        state.portfolios = action.payload;
      })
      .addCase(fetchPortfolios.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch portfolios';
      })
      .addCase(fetchPortfolio.fulfilled, (state, action: PayloadAction<FIPortfolio>) => {
        state.selectedPortfolio = action.payload;
      })
      .addCase(fetchFICoverage.fulfilled, (state, action) => {
        state.coverage = action.payload;
      })
      .addCase(fetchFinancedEmissions.fulfilled, (state, action) => {
        state.financedEmissions = action.payload;
      })
      .addCase(fetchWACI.fulfilled, (state, action) => {
        state.waci = action.payload;
      })
      .addCase(fetchPortfolioTemperature.fulfilled, (state, action: PayloadAction<PortfolioTemperature>) => {
        state.portfolioTemperature = action.payload;
      });
  },
});

export const { clearError, clearSelectedPortfolio } = fiSlice.actions;
export const selectPortfolios = (state: RootState) => state.fi.portfolios;
export const selectSelectedPortfolio = (state: RootState) => state.fi.selectedPortfolio;
export const selectFICoverage = (state: RootState) => state.fi.coverage;
export const selectFinancedEmissions = (state: RootState) => state.fi.financedEmissions;
export const selectWACI = (state: RootState) => state.fi.waci;
export const selectPortfolioTemperature = (state: RootState) => state.fi.portfolioTemperature;
export const selectFILoading = (state: RootState) => state.fi.loading;
export default fiSlice.reducer;
