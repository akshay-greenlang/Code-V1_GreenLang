import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { Portfolio, Holding, CounterpartySearchResult } from '../../types';
import { portfolioApi } from '../../services/api';
import type { RootState } from '../index';

interface PortfolioState {
  portfolios: Portfolio[];
  currentPortfolio: Portfolio | null;
  holdings: Holding[];
  searchResults: CounterpartySearchResult[];
  loading: boolean;
  error: string | null;
}

const initialState: PortfolioState = {
  portfolios: [],
  currentPortfolio: null,
  holdings: [],
  searchResults: [],
  loading: false,
  error: null,
};

export const fetchPortfolios = createAsyncThunk(
  'portfolio/fetchAll',
  async (orgId: string) => portfolioApi.list(orgId)
);

export const fetchPortfolio = createAsyncThunk(
  'portfolio/fetchOne',
  async (id: string) => portfolioApi.get(id)
);

export const createPortfolio = createAsyncThunk(
  'portfolio/create',
  async (portfolio: Partial<Portfolio>) => portfolioApi.create(portfolio)
);

export const deletePortfolio = createAsyncThunk(
  'portfolio/delete',
  async (id: string) => { await portfolioApi.delete(id); return id; }
);

export const fetchHoldings = createAsyncThunk(
  'portfolio/holdings',
  async (portfolioId: string) => portfolioApi.getHoldings(portfolioId)
);

export const searchCounterparty = createAsyncThunk(
  'portfolio/searchCounterparty',
  async (query: string) => portfolioApi.searchCounterparty(query)
);

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    clearSearchResults(state) { state.searchResults = []; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPortfolios.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchPortfolios.fulfilled, (state, action: PayloadAction<Portfolio[]>) => {
        state.loading = false;
        state.portfolios = action.payload;
      })
      .addCase(fetchPortfolios.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch portfolios';
      })
      .addCase(fetchPortfolio.fulfilled, (state, action) => { state.currentPortfolio = action.payload; })
      .addCase(createPortfolio.fulfilled, (state, action) => { state.portfolios.push(action.payload); })
      .addCase(deletePortfolio.fulfilled, (state, action) => {
        state.portfolios = state.portfolios.filter(p => p.id !== action.payload);
      })
      .addCase(fetchHoldings.fulfilled, (state, action) => { state.holdings = action.payload; })
      .addCase(searchCounterparty.fulfilled, (state, action) => { state.searchResults = action.payload; });
  },
});

export const { clearError, clearSearchResults } = portfolioSlice.actions;
export const selectPortfolios = (state: RootState) => state.portfolio.portfolios;
export const selectCurrentPortfolio = (state: RootState) => state.portfolio.currentPortfolio;
export const selectHoldings = (state: RootState) => state.portfolio.holdings;
export const selectCounterpartySearch = (state: RootState) => state.portfolio.searchResults;
export const selectPortfolioLoading = (state: RootState) => state.portfolio.loading;
export default portfolioSlice.reducer;
