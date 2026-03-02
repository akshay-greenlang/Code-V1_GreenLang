/**
 * Pipeline Redux Slice
 *
 * Manages compliance pipeline execution state: start, monitor,
 * retry, cancel, and history tracking.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  PipelineRun,
  PipelineStartRequest,
  PaginatedResponse,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface PipelineState {
  activeRun: PipelineRun | null;
  history: PipelineRun[];
  pagination: {
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
  };
  loading: boolean;
  starting: boolean;
  error: string | null;
}

const initialState: PipelineState = {
  activeRun: null,
  history: [],
  pagination: { total: 0, page: 1, per_page: 25, total_pages: 0 },
  loading: false,
  starting: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const startPipeline = createAsyncThunk(
  'pipeline/start',
  async (data: PipelineStartRequest, { rejectWithValue }) => {
    try {
      return await apiClient.startPipeline(data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to start pipeline';
      return rejectWithValue(message);
    }
  }
);

export const fetchPipelineStatus = createAsyncThunk(
  'pipeline/fetchStatus',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.getPipelineStatus(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch pipeline status';
      return rejectWithValue(message);
    }
  }
);

export const fetchPipelineHistory = createAsyncThunk(
  'pipeline/fetchHistory',
  async (
    params: { supplier_id?: string; page?: number; per_page?: number } | undefined,
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.getPipelineHistory(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch pipeline history';
      return rejectWithValue(message);
    }
  }
);

export const retryPipeline = createAsyncThunk(
  'pipeline/retry',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.retryPipeline(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Pipeline retry failed';
      return rejectWithValue(message);
    }
  }
);

export const cancelPipeline = createAsyncThunk(
  'pipeline/cancel',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.cancelPipeline(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to cancel pipeline';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const pipelineSlice = createSlice({
  name: 'pipeline',
  initialState,
  reducers: {
    clearPipelineError(state) {
      state.error = null;
    },
    clearActiveRun(state) {
      state.activeRun = null;
    },
  },
  extraReducers: (builder) => {
    // Start
    builder
      .addCase(startPipeline.pending, (state) => {
        state.starting = true;
        state.error = null;
      })
      .addCase(startPipeline.fulfilled, (state, action) => {
        state.starting = false;
        state.activeRun = action.payload;
      })
      .addCase(startPipeline.rejected, (state, action) => {
        state.starting = false;
        state.error = action.payload as string;
      });

    // Status
    builder
      .addCase(fetchPipelineStatus.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPipelineStatus.fulfilled, (state, action) => {
        state.loading = false;
        state.activeRun = action.payload;
      })
      .addCase(fetchPipelineStatus.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // History
    builder
      .addCase(fetchPipelineHistory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(
        fetchPipelineHistory.fulfilled,
        (state, action: PayloadAction<PaginatedResponse<PipelineRun>>) => {
          state.loading = false;
          state.history = action.payload.items;
          state.pagination = {
            total: action.payload.total,
            page: action.payload.page,
            per_page: action.payload.per_page,
            total_pages: action.payload.total_pages,
          };
        }
      )
      .addCase(fetchPipelineHistory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Retry
    builder
      .addCase(retryPipeline.pending, (state) => {
        state.starting = true;
        state.error = null;
      })
      .addCase(retryPipeline.fulfilled, (state, action) => {
        state.starting = false;
        state.activeRun = action.payload;
      })
      .addCase(retryPipeline.rejected, (state, action) => {
        state.starting = false;
        state.error = action.payload as string;
      });

    // Cancel
    builder
      .addCase(cancelPipeline.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(cancelPipeline.fulfilled, (state, action) => {
        state.loading = false;
        state.activeRun = action.payload;
      })
      .addCase(cancelPipeline.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearPipelineError, clearActiveRun } = pipelineSlice.actions;

export default pipelineSlice.reducer;
