/**
 * GL-GHG Corporate Platform - Redux Store Configuration
 *
 * Configures the Redux Toolkit store with 8 domain slices
 * covering dashboard, inventory, scopes, reports, targets,
 * and verification workflows.
 */

import { configureStore } from '@reduxjs/toolkit';
import dashboardReducer from './slices/dashboardSlice';
import inventoryReducer from './slices/inventorySlice';
import scope1Reducer from './slices/scope1Slice';
import scope2Reducer from './slices/scope2Slice';
import scope3Reducer from './slices/scope3Slice';
import reportsReducer from './slices/reportsSlice';
import targetsReducer from './slices/targetsSlice';
import verificationReducer from './slices/verificationSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardReducer,
    inventory: inventoryReducer,
    scope1: scope1Reducer,
    scope2: scope2Reducer,
    scope3: scope3Reducer,
    reports: reportsReducer,
    targets: targetsReducer,
    verification: verificationReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'reports/generateReport/fulfilled',
          'reports/exportData/fulfilled',
        ],
      },
    }),
  devTools: import.meta.env.DEV,
});

export type AppDispatch = typeof store.dispatch;
export type AppRootState = ReturnType<typeof store.getState>;
