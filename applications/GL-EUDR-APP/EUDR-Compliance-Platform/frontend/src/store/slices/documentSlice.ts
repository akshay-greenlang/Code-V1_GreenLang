/**
 * Document Redux Slice
 *
 * Manages document upload, verification, linking, gap analysis,
 * and document browsing for EUDR compliance evidence.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  Document,
  DocumentFilterParams,
  DocumentVerificationResult,
  DocumentGapAnalysis,
  PaginatedResponse,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface DocumentState {
  documents: Document[];
  selectedDocument: Document | null;
  verificationResult: DocumentVerificationResult | null;
  gapAnalysis: DocumentGapAnalysis | null;
  pagination: {
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
  };
  filters: DocumentFilterParams;
  loading: boolean;
  uploading: boolean;
  verifying: boolean;
  error: string | null;
}

const initialState: DocumentState = {
  documents: [],
  selectedDocument: null,
  verificationResult: null,
  gapAnalysis: null,
  pagination: { total: 0, page: 1, per_page: 25, total_pages: 0 },
  filters: {},
  loading: false,
  uploading: false,
  verifying: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const fetchDocuments = createAsyncThunk(
  'documents/fetchAll',
  async (params: DocumentFilterParams | undefined, { rejectWithValue }) => {
    try {
      return await apiClient.getDocuments(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch documents';
      return rejectWithValue(message);
    }
  }
);

export const uploadDocument = createAsyncThunk(
  'documents/upload',
  async (formData: FormData, { rejectWithValue }) => {
    try {
      return await apiClient.uploadDocument(formData);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Document upload failed';
      return rejectWithValue(message);
    }
  }
);

export const verifyDocument = createAsyncThunk(
  'documents/verify',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.verifyDocument(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Document verification failed';
      return rejectWithValue(message);
    }
  }
);

export const linkDocument = createAsyncThunk(
  'documents/link',
  async (
    {
      id,
      data,
    }: { id: string; data: { supplier_id?: string; plot_id?: string; dds_id?: string } },
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.linkDocument(id, data);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to link document';
      return rejectWithValue(message);
    }
  }
);

export const deleteDocument = createAsyncThunk(
  'documents/delete',
  async (id: string, { rejectWithValue }) => {
    try {
      await apiClient.deleteDocument(id);
      return id;
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to delete document';
      return rejectWithValue(message);
    }
  }
);

export const fetchDocumentGapAnalysis = createAsyncThunk(
  'documents/gapAnalysis',
  async (
    { supplierId, ddsId }: { supplierId: string; ddsId?: string },
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.getDocumentGapAnalysis(supplierId, ddsId);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch gap analysis';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const documentSlice = createSlice({
  name: 'documents',
  initialState,
  reducers: {
    clearDocumentError(state) {
      state.error = null;
    },
    clearSelectedDocument(state) {
      state.selectedDocument = null;
      state.verificationResult = null;
    },
    setDocumentFilters(state, action: PayloadAction<DocumentFilterParams>) {
      state.filters = action.payload;
    },
    clearGapAnalysis(state) {
      state.gapAnalysis = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch all
    builder
      .addCase(fetchDocuments.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(
        fetchDocuments.fulfilled,
        (state, action: PayloadAction<PaginatedResponse<Document>>) => {
          state.loading = false;
          state.documents = action.payload.items;
          state.pagination = {
            total: action.payload.total,
            page: action.payload.page,
            per_page: action.payload.per_page,
            total_pages: action.payload.total_pages,
          };
        }
      )
      .addCase(fetchDocuments.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Upload
    builder
      .addCase(uploadDocument.pending, (state) => {
        state.uploading = true;
        state.error = null;
      })
      .addCase(uploadDocument.fulfilled, (state, action) => {
        state.uploading = false;
        state.documents.unshift(action.payload);
        state.pagination.total += 1;
      })
      .addCase(uploadDocument.rejected, (state, action) => {
        state.uploading = false;
        state.error = action.payload as string;
      });

    // Verify
    builder
      .addCase(verifyDocument.pending, (state) => {
        state.verifying = true;
        state.error = null;
      })
      .addCase(verifyDocument.fulfilled, (state, action) => {
        state.verifying = false;
        state.verificationResult = action.payload;
        const idx = state.documents.findIndex(
          (d) => d.id === action.payload.document_id
        );
        if (idx !== -1) {
          state.documents[idx].verification_status = action.payload.is_verified
            ? 'verified'
            : 'rejected';
        }
      })
      .addCase(verifyDocument.rejected, (state, action) => {
        state.verifying = false;
        state.error = action.payload as string;
      });

    // Link
    builder
      .addCase(linkDocument.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(linkDocument.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.documents.findIndex(
          (d) => d.id === action.payload.id
        );
        if (idx !== -1) state.documents[idx] = action.payload;
      })
      .addCase(linkDocument.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Delete
    builder
      .addCase(deleteDocument.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(deleteDocument.fulfilled, (state, action) => {
        state.loading = false;
        state.documents = state.documents.filter(
          (d) => d.id !== action.payload
        );
        state.pagination.total -= 1;
      })
      .addCase(deleteDocument.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Gap analysis
    builder
      .addCase(fetchDocumentGapAnalysis.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDocumentGapAnalysis.fulfilled, (state, action) => {
        state.loading = false;
        state.gapAnalysis = action.payload;
      })
      .addCase(fetchDocumentGapAnalysis.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  clearDocumentError,
  clearSelectedDocument,
  setDocumentFilters,
  clearGapAnalysis,
} = documentSlice.actions;

export default documentSlice.reducer;
