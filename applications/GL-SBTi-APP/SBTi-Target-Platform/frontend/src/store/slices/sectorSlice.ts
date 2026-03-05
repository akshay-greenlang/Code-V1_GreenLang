import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { SectorPathway, SectorBenchmark } from '../../types';
import { sectorApi } from '../../services/api';
import type { RootState } from '../index';

interface SectorState {
  sectors: SectorPathway[];
  benchmarks: SectorBenchmark[];
  detectedSector: { detected_sector: string; confidence: number; isic_code: string } | null;
  loading: boolean;
  error: string | null;
}

const initialState: SectorState = {
  sectors: [],
  benchmarks: [],
  detectedSector: null,
  loading: false,
  error: null,
};

export const fetchSectors = createAsyncThunk(
  'sector/fetchSectors',
  async () => sectorApi.getSectors()
);

export const fetchBenchmarks = createAsyncThunk(
  'sector/fetchBenchmarks',
  async (orgId: string) => sectorApi.getBenchmarks(orgId)
);

export const detectSector = createAsyncThunk(
  'sector/detectSector',
  async (orgId: string) => sectorApi.detectSector(orgId)
);

const sectorSlice = createSlice({
  name: 'sector',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSectors.pending, (state) => { state.loading = true; })
      .addCase(fetchSectors.fulfilled, (state, action: PayloadAction<SectorPathway[]>) => {
        state.loading = false;
        state.sectors = action.payload;
      })
      .addCase(fetchSectors.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch sectors';
      })
      .addCase(fetchBenchmarks.fulfilled, (state, action: PayloadAction<SectorBenchmark[]>) => {
        state.benchmarks = action.payload;
      })
      .addCase(detectSector.fulfilled, (state, action) => {
        state.detectedSector = action.payload;
      });
  },
});

export const { clearError } = sectorSlice.actions;
export const selectSectors = (state: RootState) => state.sector.sectors;
export const selectBenchmarks = (state: RootState) => state.sector.benchmarks;
export const selectDetectedSector = (state: RootState) => state.sector.detectedSector;
export const selectSectorLoading = (state: RootState) => state.sector.loading;
export default sectorSlice.reducer;
