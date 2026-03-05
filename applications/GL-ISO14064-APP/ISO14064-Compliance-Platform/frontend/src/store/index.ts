/**
 * GL-ISO14064-APP v1.0 - Redux Store Configuration
 *
 * Configures the Redux Toolkit store with 10 domain slices
 * covering organization, inventory, emissions, removals,
 * significance, verification, reports, management plans,
 * crosswalks, and dashboard.
 */

import { configureStore } from '@reduxjs/toolkit';
import organizationReducer from './slices/organizationSlice';
import inventoryReducer from './slices/inventorySlice';
import emissionsReducer from './slices/emissionsSlice';
import removalsReducer from './slices/removalsSlice';
import significanceReducer from './slices/significanceSlice';
import verificationReducer from './slices/verificationSlice';
import reportsReducer from './slices/reportsSlice';
import managementReducer from './slices/managementSlice';
import crosswalkReducer from './slices/crosswalkSlice';
import dashboardReducer from './slices/dashboardSlice';
import qualityReducer from './slices/qualitySlice';

export const store = configureStore({
  reducer: {
    organization: organizationReducer,
    inventory: inventoryReducer,
    emissions: emissionsReducer,
    removals: removalsReducer,
    significance: significanceReducer,
    verification: verificationReducer,
    reports: reportsReducer,
    management: managementReducer,
    crosswalk: crosswalkReducer,
    dashboard: dashboardReducer,
    quality: qualityReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'reports/generateReport/fulfilled',
          'reports/exportData/fulfilled',
          'reports/downloadReport/fulfilled',
        ],
      },
    }),
  devTools: import.meta.env.DEV,
});

export type AppDispatch = typeof store.dispatch;
export type AppRootState = ReturnType<typeof store.getState>;
