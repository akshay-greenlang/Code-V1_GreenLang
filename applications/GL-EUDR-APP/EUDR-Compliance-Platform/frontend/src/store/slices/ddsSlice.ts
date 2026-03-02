/**
 * Due Diligence Statement (DDS) Redux Slice
 *
 * Manages the full DDS lifecycle: generation, validation, submission,
 * amendment, bulk operations, and download.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  DueDiligenceStatement,
  DDSGenerateRequest,
  DDSFilterParams,
  DDSValidationResult,
  DDSSubmissionResult,
  PaginatedResponse,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface DDSState {
  ddsList: DueDiligenceStatement[];
  selectedDDS: DueDiligenceStatement | null;
  validationResult: DDSValidationResult | null;
  submissionResult: DDSSubmissionResult | null;
  pagination: {
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
  };
  filters: DDSFilterParams;
  loading: boolean;
  detailLoading: boolean;
  generating: boolean;
  validating: boolean;
  submitting: boolean;
  downloading: boolean;
  error: string | null;
  bulkResult: { generated: number; errors: string[] } | null;
}

const initialState: DDSState = {
  ddsList: [],
  selectedDDS: null,
  validationResult: null,
  submissionResult: null,
  pagination: { total: 0, page: 1, per_page: 25, total_pages: 0 },
  filters: {},
  loading: false,
  detailLoading: false,
  generating: false,
  validating: false,
  submitting: false,
  downloading: false,
  error: null,
  bulkResult: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const fetchDDSList = createAsyncThunk(
  'dds/fetchAll',
  async (params: DDSFilterParams | undefined, { rejectWithValue }) => {
    try {
      return await apiClient.getDDSList(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch DDS list';
      return rejectWithValue(message);
    }
  }
);

export const fetchDDS = createAsyncThunk(
  'dds/fetchOne',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.getDDS(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch DDS';
      return rejectWithValue(message);
    }
  }
);

export const generateDDS = createAsyncThunk(
  'dds/generate',
  async (data: DDSGenerateRequest, { rejectWithValue }) => {
    try {
      return await apiClient.generateDDS(data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to generate DDS';
      return rejectWithValue(message);
    }
  }
);

export const validateDDS = createAsyncThunk(
  'dds/validate',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.validateDDS(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'DDS validation failed';
      return rejectWithValue(message);
    }
  }
);

export const submitDDS = createAsyncThunk(
  'dds/submit',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.submitDDS(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'DDS submission failed';
      return rejectWithValue(message);
    }
  }
);

export const amendDDS = createAsyncThunk(
  'dds/amend',
  async (
    { id, data }: { id: string; data: Partial<DDSGenerateRequest> },
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.amendDDS(id, data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to amend DDS';
      return rejectWithValue(message);
    }
  }
);

export const bulkGenerateDDS = createAsyncThunk(
  'dds/bulkGenerate',
  async (requests: DDSGenerateRequest[], { rejectWithValue }) => {
    try {
      return await apiClient.bulkGenerateDDS(requests);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Bulk DDS generation failed';
      return rejectWithValue(message);
    }
  }
);

export const downloadDDS = createAsyncThunk(
  'dds/download',
  async (
    { id, format }: { id: string; format: 'pdf' | 'xml' | 'json' },
    { rejectWithValue }
  ) => {
    try {
      const blob = await apiClient.downloadDDS(id, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dds-${id}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      return { id, format };
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'DDS download failed';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const ddsSlice = createSlice({
  name: 'dds',
  initialState,
  reducers: {
    clearDDSError(state) {
      state.error = null;
    },
    clearSelectedDDS(state) {
      state.selectedDDS = null;
      state.validationResult = null;
      state.submissionResult = null;
    },
    setDDSFilters(state, action: PayloadAction<DDSFilterParams>) {
      state.filters = action.payload;
    },
    clearValidationResult(state) {
      state.validationResult = null;
    },
    clearSubmissionResult(state) {
      state.submissionResult = null;
    },
    clearBulkResult(state) {
      state.bulkResult = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch all
    builder
      .addCase(fetchDDSList.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(
        fetchDDSList.fulfilled,
        (state, action: PayloadAction<PaginatedResponse<DueDiligenceStatement>>) => {
          state.loading = false;
          state.ddsList = action.payload.items;
          state.pagination = {
            total: action.payload.total,
            page: action.payload.page,
            per_page: action.payload.per_page,
            total_pages: action.payload.total_pages,
          };
        }
      )
      .addCase(fetchDDSList.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Fetch one
    builder
      .addCase(fetchDDS.pending, (state) => {
        state.detailLoading = true;
        state.error = null;
      })
      .addCase(fetchDDS.fulfilled, (state, action) => {
        state.detailLoading = false;
        state.selectedDDS = action.payload;
      })
      .addCase(fetchDDS.rejected, (state, action) => {
        state.detailLoading = false;
        state.error = action.payload as string;
      });

    // Generate
    builder
      .addCase(generateDDS.pending, (state) => {
        state.generating = true;
        state.error = null;
      })
      .addCase(generateDDS.fulfilled, (state, action) => {
        state.generating = false;
        state.ddsList.unshift(action.payload);
        state.selectedDDS = action.payload;
        state.pagination.total += 1;
      })
      .addCase(generateDDS.rejected, (state, action) => {
        state.generating = false;
        state.error = action.payload as string;
      });

    // Validate
    builder
      .addCase(validateDDS.pending, (state) => {
        state.validating = true;
        state.error = null;
      })
      .addCase(validateDDS.fulfilled, (state, action) => {
        state.validating = false;
        state.validationResult = action.payload;
      })
      .addCase(validateDDS.rejected, (state, action) => {
        state.validating = false;
        state.error = action.payload as string;
      });

    // Submit
    builder
      .addCase(submitDDS.pending, (state) => {
        state.submitting = true;
        state.error = null;
      })
      .addCase(submitDDS.fulfilled, (state, action) => {
        state.submitting = false;
        state.submissionResult = action.payload;
        if (state.selectedDDS && action.payload.submitted) {
          state.selectedDDS.status = 'submitted' as const;
          state.selectedDDS.submitted_at = action.payload.submission_date;
        }
        const idx = state.ddsList.findIndex(
          (d) => d.id === action.payload.dds_id
        );
        if (idx !== -1 && action.payload.submitted) {
          state.ddsList[idx].status = 'submitted' as const;
        }
      })
      .addCase(submitDDS.rejected, (state, action) => {
        state.submitting = false;
        state.error = action.payload as string;
      });

    // Amend
    builder
      .addCase(amendDDS.pending, (state) => {
        state.generating = true;
        state.error = null;
      })
      .addCase(amendDDS.fulfilled, (state, action) => {
        state.generating = false;
        state.selectedDDS = action.payload;
        const idx = state.ddsList.findIndex((d) => d.id === action.payload.id);
        if (idx !== -1) state.ddsList[idx] = action.payload;
      })
      .addCase(amendDDS.rejected, (state, action) => {
        state.generating = false;
        state.error = action.payload as string;
      });

    // Bulk generate
    builder
      .addCase(bulkGenerateDDS.pending, (state) => {
        state.generating = true;
        state.error = null;
        state.bulkResult = null;
      })
      .addCase(bulkGenerateDDS.fulfilled, (state, action) => {
        state.generating = false;
        state.bulkResult = action.payload;
      })
      .addCase(bulkGenerateDDS.rejected, (state, action) => {
        state.generating = false;
        state.error = action.payload as string;
      });

    // Download
    builder
      .addCase(downloadDDS.pending, (state) => {
        state.downloading = true;
        state.error = null;
      })
      .addCase(downloadDDS.fulfilled, (state) => {
        state.downloading = false;
      })
      .addCase(downloadDDS.rejected, (state, action) => {
        state.downloading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  clearDDSError,
  clearSelectedDDS,
  setDDSFilters,
  clearValidationResult,
  clearSubmissionResult,
  clearBulkResult,
} = ddsSlice.actions;

export default ddsSlice.reducer;
