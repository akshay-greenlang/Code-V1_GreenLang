/**
 * InventoryDetail Page - Single inventory overview
 *
 * Shows category cards, totals summary, and actions (run pipeline,
 * assess significance, run uncertainty analysis).
 */

import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Alert,
  Grid,
  Card,
  CardContent,
  Button,
  Divider,
  Chip,
} from '@mui/material';
import {
  PlayArrow,
  Assessment,
  ScatterPlot,
  CompareArrows,
} from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchInventory,
  fetchTotals,
  fetchCategoryResults,
} from '../store/slices/inventorySlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import StatusChip from '../components/common/StatusChip';
import CategoryOverview from '../components/categories/CategoryOverview';
import { formatTCO2e, formatPercentage, formatDate } from '../utils/formatters';
import { ISOCategory } from '../types';

const InventoryDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const navigate = useNavigate();
  const { currentInventory, totals, categoryResults, loading, error } = useSelector(
    (s: AppRootState) => s.inventory,
  );

  useEffect(() => {
    if (id) {
      dispatch(fetchInventory(id));
      dispatch(fetchTotals(id));
      dispatch(fetchCategoryResults(id));
    }
  }, [dispatch, id]);

  if (loading && !currentInventory) {
    return <LoadingSpinner message="Loading inventory..." />;
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!currentInventory) return <Alert severity="info">Inventory not found.</Alert>;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Inventory: {currentInventory.reporting_year}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, mt: 0.5, flexWrap: 'wrap' }}>
            <StatusChip status={currentInventory.status} />
            <Chip
              label={currentInventory.consolidation_approach.replace(/_/g, ' ')}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`GWP: ${currentInventory.gwp_source.toUpperCase()}`}
              size="small"
              variant="outlined"
            />
          </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<PlayArrow />}
            onClick={() => navigate(`/inventories/${id}/emissions`)}
          >
            Emissions
          </Button>
          <Button
            variant="outlined"
            startIcon={<Assessment />}
            onClick={() => navigate(`/inventories/${id}/significance`)}
          >
            Significance
          </Button>
          <Button
            variant="outlined"
            startIcon={<ScatterPlot />}
            onClick={() => navigate(`/inventories/${id}/uncertainty`)}
          >
            Uncertainty
          </Button>
          <Button
            variant="outlined"
            startIcon={<CompareArrows />}
            onClick={() => navigate(`/inventories/${id}/crosswalk`)}
          >
            Crosswalk
          </Button>
        </Box>
      </Box>

      {/* Totals summary */}
      {totals && (
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Gross Emissions
                </Typography>
                <Typography variant="h5" fontWeight={700} color="error.main">
                  {formatTCO2e(totals.gross_emissions_tco2e)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Total Removals
                </Typography>
                <Typography variant="h5" fontWeight={700} color="success.main">
                  {formatTCO2e(totals.total_removals_tco2e)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Net Emissions
                </Typography>
                <Typography variant="h5" fontWeight={700} color="primary.main">
                  {formatTCO2e(totals.net_emissions_tco2e)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  YoY Change
                </Typography>
                <Typography
                  variant="h5"
                  fontWeight={700}
                  color={
                    totals.yoy_change_pct != null && totals.yoy_change_pct < 0
                      ? 'success.main'
                      : 'error.main'
                  }
                >
                  {totals.yoy_change_pct != null
                    ? formatPercentage(totals.yoy_change_pct)
                    : 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Category Overview */}
      <Typography variant="h6" fontWeight={600} gutterBottom>
        ISO 14064-1 Categories
      </Typography>
      <CategoryOverview
        categories={categoryResults}
        onCategoryClick={(cat) => navigate(`/inventories/${id}/categories`)}
      />

      {/* Quick navigation */}
      <Grid container spacing={2} sx={{ mt: 3 }}>
        <Grid item xs={12} sm={6} md={4}>
          <Button
            fullWidth
            variant="outlined"
            onClick={() => navigate(`/inventories/${id}/removals`)}
            sx={{ py: 2 }}
          >
            Manage Removals
          </Button>
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <Button
            fullWidth
            variant="outlined"
            onClick={() => navigate(`/inventories/${id}/categories`)}
            sx={{ py: 2 }}
          >
            Category Details
          </Button>
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <Button
            fullWidth
            variant="outlined"
            onClick={() => navigate(`/inventories/${id}/crosswalk`)}
            sx={{ py: 2 }}
          >
            GHG Protocol Crosswalk
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
};

export default InventoryDetail;
