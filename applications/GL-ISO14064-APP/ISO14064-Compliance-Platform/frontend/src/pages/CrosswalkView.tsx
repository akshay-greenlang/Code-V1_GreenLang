/**
 * CrosswalkView Page - Full crosswalk table + scope comparison chart
 */

import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import { Box, Typography, Alert, Button, Grid, Chip } from '@mui/material';
import { Refresh, Download } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchCrosswalk,
  generateCrosswalk,
  exportCrosswalk,
} from '../store/slices/crosswalkSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import CrosswalkTable from '../components/crosswalk/CrosswalkTable';
import ScopeComparisonChart from '../components/crosswalk/ScopeComparisonChart';

const CrosswalkView: React.FC = () => {
  const { id: inventoryId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { crosswalk, loading, error } = useSelector(
    (s: AppRootState) => s.crosswalk,
  );

  useEffect(() => {
    if (inventoryId) {
      dispatch(fetchCrosswalk(inventoryId));
    }
  }, [dispatch, inventoryId]);

  const handleGenerate = () => {
    if (inventoryId) {
      dispatch(generateCrosswalk(inventoryId));
    }
  };

  const handleExport = (format: string) => {
    if (inventoryId) {
      dispatch(exportCrosswalk({ inventoryId, format }));
    }
  };

  if (loading && !crosswalk) return <LoadingSpinner message="Loading crosswalk..." />;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            ISO 14064-1 / GHG Protocol Crosswalk
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Map ISO 14064-1 categories to GHG Protocol scopes for dual-framework reporting
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<Refresh />}
            onClick={handleGenerate}
            disabled={loading}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            {crosswalk ? 'Regenerate' : 'Generate'} Crosswalk
          </Button>
          {crosswalk && (
            <>
              <Button
                variant="outlined"
                startIcon={<Download />}
                onClick={() => handleExport('excel')}
              >
                Export Excel
              </Button>
              <Button
                variant="outlined"
                startIcon={<Download />}
                onClick={() => handleExport('csv')}
              >
                Export CSV
              </Button>
            </>
          )}
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {!crosswalk && !loading && (
        <Alert severity="info">
          No crosswalk generated yet. Click "Generate Crosswalk" to create the ISO-to-GHG Protocol mapping.
        </Alert>
      )}

      {crosswalk && (
        <Grid container spacing={3}>
          {/* Reconciliation status */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 1 }}>
              <Chip
                label={
                  Math.abs(crosswalk.reconciliation_pct) < 1
                    ? 'Reconciled'
                    : 'Reconciliation Gap'
                }
                color={Math.abs(crosswalk.reconciliation_pct) < 1 ? 'success' : 'warning'}
              />
              <Typography variant="body2" color="text.secondary">
                Difference: {crosswalk.reconciliation_difference.toFixed(2)} tCO2e ({crosswalk.reconciliation_pct.toFixed(2)}%)
              </Typography>
            </Box>
          </Grid>

          {/* Crosswalk Table */}
          <Grid item xs={12}>
            <CrosswalkTable crosswalk={crosswalk} />
          </Grid>

          {/* Comparison Chart */}
          <Grid item xs={12}>
            <ScopeComparisonChart crosswalk={crosswalk} />
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default CrosswalkView;
