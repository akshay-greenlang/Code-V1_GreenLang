/**
 * GL-EUDR-APP Redux Store
 *
 * Central store combining all feature slices for the EUDR compliance platform.
 */

import { configureStore } from '@reduxjs/toolkit';
import dashboardReducer from './slices/dashboardSlice';
import supplierReducer from './slices/supplierSlice';
import plotReducer from './slices/plotSlice';
import ddsReducer from './slices/ddsSlice';
import documentReducer from './slices/documentSlice';
import pipelineReducer from './slices/pipelineSlice';
import riskReducer from './slices/riskSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardReducer,
    suppliers: supplierReducer,
    plots: plotReducer,
    dds: ddsReducer,
    documents: documentReducer,
    pipeline: pipelineReducer,
    risk: riskReducer,
  },
  devTools: import.meta.env.DEV,
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
