/**
 * InventorySetup Page - GHG inventory boundary configuration
 *
 * Composes BoundaryWizard for new inventory creation, EntityTree
 * showing the organizational structure, BaseYearConfig panel,
 * and a current inventory summary card.
 */

import React, { useEffect, useState } from 'react';
import { Grid, Box, Typography, Card, CardContent, Button, Alert, Chip, Divider } from '@mui/material';
import { Add } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import {
  fetchOrganization,
  fetchEntities,
  addEntity,
  updateEntityLocal,
  removeEntityLocal,
  fetchBaseYear,
} from '../store/slices/inventorySlice';
import BoundaryWizard from '../components/inventory/BoundaryWizard';
import EntityTree from '../components/inventory/EntityTree';
import BaseYearConfig from '../components/inventory/BaseYearConfig';
import LoadingSpinner from '../components/common/LoadingSpinner';
import StatusBadge from '../components/common/StatusBadge';
import { formatNumber } from '../utils/formatters';
import type { Entity } from '../types';

const DEMO_ORG_ID = 'demo-org';

const InventorySetupPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { organization, entities, currentInventory, boundary, loading, error } = useAppSelector(
    (state) => state.inventory
  );
  const [showWizard, setShowWizard] = useState(false);

  useEffect(() => {
    dispatch(fetchOrganization(DEMO_ORG_ID));
    dispatch(fetchEntities(DEMO_ORG_ID));
    dispatch(fetchBaseYear(DEMO_ORG_ID));
  }, [dispatch]);

  const handleWizardComplete = (data: unknown) => {
    setShowWizard(false);
    // In production, dispatch createOrg and createInventory
  };

  const handleEntityUpdate = (entity: Entity) => {
    dispatch(updateEntityLocal(entity));
  };

  const handleEntityAdd = (parentId: string | null, entityData: Partial<Entity>) => {
    dispatch(addEntity({ orgId: DEMO_ORG_ID, payload: entityData as any }));
  };

  const handleEntityRemove = (entityId: string) => {
    dispatch(removeEntityLocal(entityId));
  };

  if (loading && !organization) return <LoadingSpinner message="Loading inventory setup..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      {/* New inventory wizard */}
      {showWizard ? (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <BoundaryWizard onComplete={handleWizardComplete} />
          </CardContent>
        </Card>
      ) : (
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6">{organization?.name || 'Organization'}</Typography>
            <Typography variant="body2" color="text.secondary">
              {organization?.industry} | {organization?.country}
            </Typography>
          </Box>
          <Button variant="contained" startIcon={<Add />} onClick={() => setShowWizard(true)}>
            New Inventory
          </Button>
        </Box>
      )}

      {/* Current inventory summary */}
      {currentInventory && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                {currentInventory.reporting_year} Inventory
              </Typography>
              <StatusBadge status={currentInventory.status} />
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Total (S1+S2)</Typography>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                  {formatNumber(currentInventory.total_scope1_2_tco2e)} tCO2e
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Total (S1+S2+S3)</Typography>
                <Typography variant="body1" sx={{ fontWeight: 600 }}>
                  {formatNumber(currentInventory.total_scope1_2_3_tco2e)} tCO2e
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Verification</Typography>
                <StatusBadge status={currentInventory.verification_status} />
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">Consolidation</Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {boundary?.consolidation_approach?.replace(/_/g, ' ') || 'Not set'}
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Entity tree + Base year config */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={7}>
          <EntityTree
            entities={entities}
            onUpdate={handleEntityUpdate}
            onAdd={handleEntityAdd}
            onRemove={handleEntityRemove}
          />
        </Grid>
        <Grid item xs={12} md={5}>
          {currentInventory?.base_year ? (
            <BaseYearConfig
              baseYear={currentInventory.base_year}
              onRecalculate={() => dispatch(fetchBaseYear(DEMO_ORG_ID))}
              onLock={(locked) => {/* dispatch lockBaseYear */}}
            />
          ) : (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <Typography variant="body2" color="text.secondary">
                  Create an inventory to configure the base year.
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default InventorySetupPage;
