import { configureStore } from '@reduxjs/toolkit';
import dashboardReducer from './slices/dashboardSlice';
import transactionsReducer from './slices/transactionsSlice';
import suppliersReducer from './slices/suppliersSlice';
import reportsReducer from './slices/reportsSlice';
import uncertaintyReducer from './slices/uncertaintySlice';
import settingsReducer from './slices/settingsSlice';
import cdpReducer from './slices/cdpSlice';
import complianceReducer from './slices/complianceSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardReducer,
    transactions: transactionsReducer,
    suppliers: suppliersReducer,
    reports: reportsReducer,
    uncertainty: uncertaintyReducer,
    settings: settingsReducer,
    cdp: cdpReducer,
    compliance: complianceReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['transactions/uploadFile/pending'],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['payload.file', 'meta.arg.file'],
        // Ignore these paths in the state
        ignoredPaths: ['transactions.uploadingFile'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
