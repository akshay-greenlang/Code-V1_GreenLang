/**
 * Organization Redux Slice
 *
 * Manages organization state: profile, entity hierarchy, organizational
 * boundary (consolidation approach), and operational boundary (category
 * inclusions with significance decisions).
 *
 * Async thunks:
 *   - createOrg: Create a new organization
 *   - fetchOrganization: Load organization by ID
 *   - addEntity: Add a reporting entity
 *   - fetchEntities: Load all entities
 *   - updateEntity: Update an entity
 *   - deleteEntity: Remove an entity
 *   - setOrgBoundary: Set organizational boundary
 *   - fetchOrgBoundary: Load organizational boundary
 *   - setOpBoundary: Set operational boundary
 *   - fetchOpBoundary: Load operational boundary
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  OrganizationState,
  Organization,
  Entity,
  OrganizationalBoundary,
  OperationalBoundary,
  CreateOrganizationRequest,
  AddEntityRequest,
  UpdateEntityRequest,
  SetOrganizationalBoundaryRequest,
  SetOperationalBoundaryRequest,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: OrganizationState = {
  organization: null,
  entities: [],
  organizationalBoundary: null,
  operationalBoundary: null,
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
  'organization/createOrg',
  async (payload) => {
    return iso14064Api.createOrganization(payload);
  },
);

export const fetchOrganization = createAsyncThunk<
  Organization,
  string
>(
  'organization/fetchOrganization',
  async (orgId) => {
    return iso14064Api.getOrganization(orgId);
  },
);

export const updateOrganization = createAsyncThunk<
  Organization,
  { orgId: string; payload: Partial<CreateOrganizationRequest> }
>(
  'organization/updateOrganization',
  async ({ orgId, payload }) => {
    return iso14064Api.updateOrganization(orgId, payload);
  },
);

export const addEntity = createAsyncThunk<
  Entity,
  { orgId: string; payload: AddEntityRequest }
>(
  'organization/addEntity',
  async ({ orgId, payload }) => {
    return iso14064Api.addEntity(orgId, payload);
  },
);

export const fetchEntities = createAsyncThunk<
  Entity[],
  string
>(
  'organization/fetchEntities',
  async (orgId) => {
    return iso14064Api.getEntities(orgId);
  },
);

export const updateEntity = createAsyncThunk<
  Entity,
  { orgId: string; entityId: string; payload: UpdateEntityRequest }
>(
  'organization/updateEntity',
  async ({ orgId, entityId, payload }) => {
    return iso14064Api.updateEntity(orgId, entityId, payload);
  },
);

export const deleteEntity = createAsyncThunk<
  string,
  { orgId: string; entityId: string }
>(
  'organization/deleteEntity',
  async ({ orgId, entityId }) => {
    await iso14064Api.deleteEntity(orgId, entityId);
    return entityId;
  },
);

export const setOrgBoundary = createAsyncThunk<
  OrganizationalBoundary,
  { orgId: string; payload: SetOrganizationalBoundaryRequest }
>(
  'organization/setOrgBoundary',
  async ({ orgId, payload }) => {
    return iso14064Api.setOrganizationalBoundary(orgId, payload);
  },
);

export const fetchOrgBoundary = createAsyncThunk<
  OrganizationalBoundary,
  string
>(
  'organization/fetchOrgBoundary',
  async (orgId) => {
    return iso14064Api.getOrganizationalBoundary(orgId);
  },
);

export const setOpBoundary = createAsyncThunk<
  OperationalBoundary,
  { orgId: string; payload: SetOperationalBoundaryRequest }
>(
  'organization/setOpBoundary',
  async ({ orgId, payload }) => {
    return iso14064Api.setOperationalBoundary(orgId, payload);
  },
);

export const fetchOpBoundary = createAsyncThunk<
  OperationalBoundary,
  string
>(
  'organization/fetchOpBoundary',
  async (orgId) => {
    return iso14064Api.getOperationalBoundary(orgId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const organizationSlice = createSlice({
  name: 'organization',
  initialState,
  reducers: {
    clearOrganization: () => initialState,
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
        if (action.payload.entities) {
          state.entities = action.payload.entities;
        }
      })
      .addCase(fetchOrganization.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch organization';
      })

      // -- updateOrganization --
      .addCase(updateOrganization.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateOrganization.fulfilled, (state, action) => {
        state.loading = false;
        state.organization = action.payload;
      })
      .addCase(updateOrganization.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to update organization';
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

      // -- updateEntity --
      .addCase(updateEntity.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateEntity.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.entities.findIndex((e) => e.id === action.payload.id);
        if (idx >= 0) {
          state.entities[idx] = action.payload;
        }
      })
      .addCase(updateEntity.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to update entity';
      })

      // -- deleteEntity --
      .addCase(deleteEntity.fulfilled, (state, action) => {
        state.entities = state.entities.filter((e) => e.id !== action.payload);
      })

      // -- setOrgBoundary --
      .addCase(setOrgBoundary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(setOrgBoundary.fulfilled, (state, action) => {
        state.loading = false;
        state.organizationalBoundary = action.payload;
      })
      .addCase(setOrgBoundary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to set organizational boundary';
      })

      // -- fetchOrgBoundary --
      .addCase(fetchOrgBoundary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchOrgBoundary.fulfilled, (state, action) => {
        state.loading = false;
        state.organizationalBoundary = action.payload;
      })
      .addCase(fetchOrgBoundary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch organizational boundary';
      })

      // -- setOpBoundary --
      .addCase(setOpBoundary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(setOpBoundary.fulfilled, (state, action) => {
        state.loading = false;
        state.operationalBoundary = action.payload;
      })
      .addCase(setOpBoundary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to set operational boundary';
      })

      // -- fetchOpBoundary --
      .addCase(fetchOpBoundary.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchOpBoundary.fulfilled, (state, action) => {
        state.loading = false;
        state.operationalBoundary = action.payload;
      })
      .addCase(fetchOpBoundary.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch operational boundary';
      });
  },
});

export const { clearOrganization, setOrganization, updateEntityLocal, removeEntityLocal } =
  organizationSlice.actions;
export default organizationSlice.reducer;
