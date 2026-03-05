/**
 * Benchmarking Redux Slice
 *
 * Manages sector/regional benchmarking state: peer comparisons,
 * score distributions, category averages, and A-list rates.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { BenchmarkingState, Benchmark, PeerComparison } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: BenchmarkingState = {
  benchmark: null,
  peerComparison: null,
  loading: false,
  error: null,
};

export const fetchSectorBenchmark = createAsyncThunk<
  Benchmark,
  { sector: string; year?: number }
>(
  'benchmarking/fetchSector',
  async ({ sector, year }) => cdpApi.getSectorBenchmark(sector, year),
);

export const fetchRegionalBenchmark = createAsyncThunk<
  Benchmark,
  { region: string; year?: number }
>(
  'benchmarking/fetchRegional',
  async ({ region, year }) => cdpApi.getRegionalBenchmark(region, year),
);

export const fetchPeerComparison = createAsyncThunk<
  PeerComparison,
  { orgId: string; sector?: string }
>(
  'benchmarking/fetchPeers',
  async ({ orgId, sector }) => cdpApi.getPeerComparison(orgId, sector),
);

const benchmarkingSlice = createSlice({
  name: 'benchmarking',
  initialState,
  reducers: {
    clearBenchmarking: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSectorBenchmark.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSectorBenchmark.fulfilled, (state, action) => {
        state.loading = false;
        state.benchmark = action.payload;
      })
      .addCase(fetchSectorBenchmark.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load sector benchmark';
      })
      .addCase(fetchRegionalBenchmark.fulfilled, (state, action) => {
        state.loading = false;
        state.benchmark = action.payload;
      })
      .addCase(fetchPeerComparison.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPeerComparison.fulfilled, (state, action) => {
        state.loading = false;
        state.peerComparison = action.payload;
      })
      .addCase(fetchPeerComparison.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load peer comparison';
      });
  },
});

export const { clearBenchmarking } = benchmarkingSlice.actions;
export default benchmarkingSlice.reducer;
