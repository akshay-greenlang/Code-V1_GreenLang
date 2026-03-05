import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { TemperatureScore, TemperatureTimeSeries, PeerTemperatureRanking } from '../../types';
import { temperatureApi } from '../../services/api';
import type { RootState } from '../index';

interface TemperatureState {
  score: TemperatureScore | null;
  timeSeries: TemperatureTimeSeries[];
  peerRanking: PeerTemperatureRanking[];
  loading: boolean;
  error: string | null;
}

const initialState: TemperatureState = {
  score: null,
  timeSeries: [],
  peerRanking: [],
  loading: false,
  error: null,
};

export const fetchTemperatureScore = createAsyncThunk(
  'temperature/fetchScore',
  async (orgId: string) => temperatureApi.getScore(orgId)
);

export const fetchTemperatureTimeSeries = createAsyncThunk(
  'temperature/fetchTimeSeries',
  async (orgId: string) => temperatureApi.getTimeSeries(orgId)
);

export const fetchPeerRanking = createAsyncThunk(
  'temperature/fetchPeerRanking',
  async (orgId: string) => temperatureApi.getPeerRanking(orgId)
);

export const recalculateTemperature = createAsyncThunk(
  'temperature/recalculate',
  async (orgId: string) => temperatureApi.recalculate(orgId)
);

const temperatureSlice = createSlice({
  name: 'temperature',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTemperatureScore.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchTemperatureScore.fulfilled, (state, action: PayloadAction<TemperatureScore>) => {
        state.loading = false;
        state.score = action.payload;
      })
      .addCase(fetchTemperatureScore.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch temperature score';
      })
      .addCase(fetchTemperatureTimeSeries.fulfilled, (state, action: PayloadAction<TemperatureTimeSeries[]>) => {
        state.timeSeries = action.payload;
      })
      .addCase(fetchPeerRanking.fulfilled, (state, action: PayloadAction<PeerTemperatureRanking[]>) => {
        state.peerRanking = action.payload;
      })
      .addCase(recalculateTemperature.fulfilled, (state, action: PayloadAction<TemperatureScore>) => {
        state.score = action.payload;
      });
  },
});

export const { clearError } = temperatureSlice.actions;
export const selectTemperatureScore = (state: RootState) => state.temperature.score;
export const selectTemperatureTimeSeries = (state: RootState) => state.temperature.timeSeries;
export const selectPeerRanking = (state: RootState) => state.temperature.peerRanking;
export const selectTemperatureLoading = (state: RootState) => state.temperature.loading;
export default temperatureSlice.reducer;
