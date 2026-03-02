/**
 * Inventory Redux Slice
 *
 * Manages GHG inventory state: organization profile, reporting entities,
 * inventory boundary, base year configuration, and consolidation approach.
 *
 * Async thunks:
 *   - createOrg: Create a new organization
 *   - addEntity: Add a reporting entity under an organization
 *   - setBoundary: Set inventory boundary for a reporting year
 *   - createInventory: Create a new GHG inventory
 *   - getInventory: Fetch an existing inventory by ID
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  InventoryState,
  Organization,
  Entity,
  InventoryBoundary,
  GHGInventory,
  BaseYear,
  CreateOrganizationRequest,
  AddEntityRequest,
  SetBoundaryRequest,
  CreateInventoryRequest,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: InventoryState = {
  currentInventory: null,
  organization: null,
  entities: [],
  boundary: null,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const createOrg = createAsyncThunk<
  Organization,
  CreateOrganizationRequest
>(
  'inventory/createOrg',
  async (payload) => {
    return ghgApi.createOrganization(payload);
  },
);

export const fetchOrganization = createAsyncThunk<
  Organization,
  string
>(
  'inventory/fetchOrganization',
  async (orgId) => {
    return ghgApi.getOrganization(orgId);
  },
);

export const addEntity = createAsyncThunk<
  Entity,
  { orgId: string; payload: AddEntityRequest }
>(
  'inventory/addEntity',
  async ({ orgId, payload }) => {
    return ghgApi.addEntity(orgId, payload);
  },
);

export const fetchEntities = createAsyncThunk<
  Entity[],
  string
>(
  'inventory/fetchEntities',
  async (orgId) => {
    return ghgApi.getEntities(orgId);
  },
);

export const setBoundary = createAsyncThunk<
  InventoryBoundary,
  { orgId: string; payload: SetBoundaryRequest }
>(
  'inventory/setBoundary',
  async ({ orgId, payload }) => {
    return ghgApi.setBoundary(orgId, payload);
  },
);

export const createInventory = createAsyncThunk<
  GHGInventory,
  CreateInventoryRequest
>(
  'inventory/createInventory',
  async (payload) => {
    return ghgApi.createInventory(payload);
  },
);

export const getInventory = createAsyncThunk<
  GHGInventory,
  string
>(
  'inventory/getInventory',
  async (inventoryId) => {
    return ghgApi.getInventory(inventoryId);
  },
);

export const fetchBaseYear = createAsyncThunk<
  BaseYear,
  string
>(
  'inventory/fetchBaseYear',
  async (orgId) => {
    return ghgApi.getBaseYear(orgId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const inventorySlice = createSlice({
  name: 'inventory',
  initialState,
  reducers: {
    clearInventory: () => initialState,
    setOrganization: (state, action: PayloadAction<Organization>) => {
      state.organization = action.payload;
    },
    updateEntityLocal: (state, action: PayloadAction<Entity>) => {
      const idx = state.entities.findIndex((e) => e.id === action.payload.id);
      if (idx >= 0) {
        state.entities[idx] = action.payload;
      }
    },
    removeEntityLocal: (state, action: PayloadAction<string>) => {
      state.entities = state.entities.filter((e) => e.id !== action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      // -- createOrg --
      .addCase(createOrg.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createOrg.fulfilled, (state, action) => {
        state.loading = false;
        state.organization = action.payload;
      })
      .addCase(createOrg.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to create organization';
      })

      // -- fetchOrganization --
      .addCase(fetchOrganization.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchOrganization.fulfilled, (state, action) => {
        state.loading = false;
        state.organization = action.payload;
      })
      .addCase(fetchOrganization.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch organization';
      })

      // -- addEntity --
      .addCase(addEntity.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(addEntity.fulfilled, (state, action) => {
        state.loading = false;
        state.entities.push(action.payload);
      })
      .addCase(addEntity.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to add entity';
      })

      // -- fetchEntities --
      .addCase(fetchEntities.fulfilled, (state, action) => {
        state.entities = action.payload;
      })

      // -- setBoundary --
      .addCase(setBoundary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(setBoundary.fulfilled, (state, action) => {
        state.loading = false;
        state.boundary = action.payload;
      })
      .addCase(setBoundary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to set boundary';
      })

      // -- createInventory --
      .addCase(createInventory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createInventory.fulfilled, (state, action) => {
        state.loading = false;
        state.currentInventory = action.payload;
      })
      .addCase(createInventory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to create inventory';
      })

      // -- getInventory --
      .addCase(getInventory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getInventory.fulfilled, (state, action) => {
        state.loading = false;
        state.currentInventory = action.payload;
        state.boundary = action.payload.boundary;
      })
      .addCase(getInventory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch inventory';
      })

      // -- fetchBaseYear --
      .addCase(fetchBaseYear.fulfilled, (state, action) => {
        if (state.currentInventory) {
          state.currentInventory.base_year = action.payload;
        }
      });
  },
});

export const { clearInventory, setOrganization, updateEntityLocal, removeEntityLocal } =
  inventorySlice.actions;
export default inventorySlice.reducer;
