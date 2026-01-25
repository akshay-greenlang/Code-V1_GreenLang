import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';
import type { Transaction, PaginatedResponse, UploadResponse } from '../../types';

interface TransactionsState {
  transactions: Transaction[];
  selectedTransaction: Transaction | null;
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
  filters: {
    search?: string;
    startDate?: string;
    endDate?: string;
    status?: string[];
    categories?: number[];
  };
  uploadStatus: UploadResponse | null;
  loading: boolean;
  uploading: boolean;
  error: string | null;
}

const initialState: TransactionsState = {
  transactions: [],
  selectedTransaction: null,
  pagination: {
    page: 1,
    pageSize: 25,
    total: 0,
    totalPages: 0,
  },
  filters: {},
  uploadStatus: null,
  loading: false,
  uploading: false,
  error: null,
};

// Async thunks
export const fetchTransactions = createAsyncThunk(
  'transactions/fetchAll',
  async (params?: {
    page?: number;
    pageSize?: number;
    search?: string;
    startDate?: string;
    endDate?: string;
    status?: string[];
    categories?: number[];
  }) => {
    const response = await api.getTransactions(params);
    return response;
  }
);

export const fetchTransaction = createAsyncThunk(
  'transactions/fetchOne',
  async (id: string) => {
    const transaction = await api.getTransaction(id);
    return transaction;
  }
);

export const uploadFile = createAsyncThunk(
  'transactions/upload',
  async ({ file, format }: { file: File; format: string }) => {
    const response = await api.uploadTransactions(file, format);
    return response;
  }
);

export const pollUploadStatus = createAsyncThunk(
  'transactions/pollUploadStatus',
  async (jobId: string) => {
    const status = await api.getUploadStatus(jobId);
    return status;
  }
);

export const deleteTransaction = createAsyncThunk(
  'transactions/delete',
  async (id: string) => {
    await api.deleteTransaction(id);
    return id;
  }
);

const transactionsSlice = createSlice({
  name: 'transactions',
  initialState,
  reducers: {
    setPage: (state, action: PayloadAction<number>) => {
      state.pagination.page = action.payload;
    },
    setPageSize: (state, action: PayloadAction<number>) => {
      state.pagination.pageSize = action.payload;
    },
    setFilters: (state, action: PayloadAction<TransactionsState['filters']>) => {
      state.filters = action.payload;
      state.pagination.page = 1; // Reset to first page when filters change
    },
    clearUploadStatus: (state) => {
      state.uploadStatus = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch all transactions
    builder
      .addCase(fetchTransactions.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTransactions.fulfilled, (state, action: PayloadAction<PaginatedResponse<Transaction>>) => {
        state.loading = false;
        state.transactions = action.payload.data;
        state.pagination = action.payload.pagination;
      })
      .addCase(fetchTransactions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch transactions';
      });

    // Fetch single transaction
    builder
      .addCase(fetchTransaction.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTransaction.fulfilled, (state, action: PayloadAction<Transaction>) => {
        state.loading = false;
        state.selectedTransaction = action.payload;
      })
      .addCase(fetchTransaction.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch transaction';
      });

    // Upload file
    builder
      .addCase(uploadFile.pending, (state) => {
        state.uploading = true;
        state.error = null;
      })
      .addCase(uploadFile.fulfilled, (state, action: PayloadAction<UploadResponse>) => {
        state.uploading = false;
        state.uploadStatus = action.payload;
      })
      .addCase(uploadFile.rejected, (state, action) => {
        state.uploading = false;
        state.error = action.error.message || 'Failed to upload file';
      });

    // Poll upload status
    builder
      .addCase(pollUploadStatus.fulfilled, (state, action: PayloadAction<UploadResponse>) => {
        state.uploadStatus = action.payload;
      });

    // Delete transaction
    builder
      .addCase(deleteTransaction.fulfilled, (state, action: PayloadAction<string>) => {
        state.transactions = state.transactions.filter(t => t.id !== action.payload);
        state.pagination.total -= 1;
      });
  },
});

export const { setPage, setPageSize, setFilters, clearUploadStatus } = transactionsSlice.actions;
export default transactionsSlice.reducer;
