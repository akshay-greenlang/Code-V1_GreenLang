/**
 * OrganizationSetup Page - Organization creation, entity management,
 * and boundary configuration.
 */

import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import { Box, Typography, Alert } from '@mui/material';
import type { AppDispatch, AppRootState } from '../store';
import {
  createOrg,
  fetchOrganization,
  updateOrganization,
  addEntity,
  fetchEntities,
  updateEntity,
  deleteEntity,
  setOrgBoundary,
  fetchOrgBoundary,
  setOpBoundary,
  fetchOpBoundary,
} from '../store/slices/organizationSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import OrganizationForm from '../components/boundary/OrganizationForm';
import EntityList from '../components/boundary/EntityList';
import BoundaryConfig from '../components/boundary/BoundaryConfig';
import type { CreateOrganizationRequest, AddEntityRequest } from '../types';

const OrganizationSetup: React.FC = () => {
  const { id: orgId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const {
    organization,
    entities,
    organizationalBoundary,
    operationalBoundary,
    loading,
    error,
  } = useSelector((s: AppRootState) => s.organization);

  useEffect(() => {
    if (orgId) {
      dispatch(fetchOrganization(orgId));
      dispatch(fetchEntities(orgId));
      dispatch(fetchOrgBoundary(orgId));
      dispatch(fetchOpBoundary(orgId));
    }
  }, [dispatch, orgId]);

  const handleCreateOrUpdate = (data: CreateOrganizationRequest) => {
    if (organization && orgId) {
      dispatch(updateOrganization({ orgId, payload: data }));
    } else {
      dispatch(createOrg(data));
    }
  };

  const handleAddEntity = (data: AddEntityRequest) => {
    const targetOrgId = orgId || organization?.id;
    if (targetOrgId) {
      dispatch(addEntity({ orgId: targetOrgId, payload: data }));
    }
  };

  const handleEditEntity = (entityId: string, data: Partial<AddEntityRequest>) => {
    const targetOrgId = orgId || organization?.id;
    if (targetOrgId) {
      dispatch(updateEntity({ orgId: targetOrgId, entityId, payload: data }));
    }
  };

  const handleDeleteEntity = (entityId: string) => {
    const targetOrgId = orgId || organization?.id;
    if (targetOrgId) {
      dispatch(deleteEntity({ orgId: targetOrgId, entityId }));
    }
  };

  if (loading && !organization) return <LoadingSpinner message="Loading organization..." />;

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Organization Setup
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Define the reporting organization, entities, and boundaries per ISO 14064-1 Clause 5.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Organization Form */}
      <Box sx={{ mb: 3 }}>
        <OrganizationForm
          organization={organization}
          onSubmit={handleCreateOrUpdate}
          loading={loading}
        />
      </Box>

      {/* Entity List */}
      {(organization || orgId) && (
        <Box sx={{ mb: 3 }}>
          <EntityList
            entities={entities}
            onAdd={handleAddEntity}
            onEdit={handleEditEntity}
            onDelete={handleDeleteEntity}
            loading={loading}
          />
        </Box>
      )}

      {/* Boundary Configuration */}
      {(organization || orgId) && (
        <BoundaryConfig
          entities={entities}
          orgBoundary={organizationalBoundary}
          opBoundary={operationalBoundary}
          onSaveOrgBoundary={(data) => {
            const targetOrgId = orgId || organization?.id;
            if (targetOrgId) {
              dispatch(setOrgBoundary({ orgId: targetOrgId, payload: data }));
            }
          }}
          onSaveOpBoundary={(data) => {
            const targetOrgId = orgId || organization?.id;
            if (targetOrgId) {
              dispatch(setOpBoundary({ orgId: targetOrgId, payload: data }));
            }
          }}
          loading={loading}
        />
      )}
    </Box>
  );
};

export default OrganizationSetup;
