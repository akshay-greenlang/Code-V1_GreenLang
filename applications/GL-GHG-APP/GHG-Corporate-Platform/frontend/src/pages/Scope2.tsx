/**
 * Scope 2 Page - Indirect energy emissions
 *
 * Composes dual reporting stat cards (location + market totals),
 * DualReportingComparison, ReconciliationWaterfall, and
 * InstrumentTracker table.
 */

import React, { useEffect } from 'react';
import { Box, Alert, Typography, Grid, Card, CardContent } from '@mui/material';
import { ElectricBolt } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import { fetchScope2Summary, fetchScope2Reconciliation } from '../store/slices/scope2Slice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import DualReportingComparison from '../components/scope2/DualReportingComparison';
import ReconciliationWaterfall from '../components/scope2/ReconciliationWaterfall';
import InstrumentTracker from '../components/scope2/InstrumentTracker';
import StatCard from '../components/common/StatCard';
import { formatEmissions } from '../utils/formatters';

const DEMO_INVENTORY_ID = 'demo-inventory';

const Scope2Page: React.FC = () => {
  const dispatch = useAppDispatch();
  const { summary, reconciliation, loading, error } = useAppSelector((state) => state.scope2);

  useEffect(() => {
    dispatch(fetchScope2Summary(DEMO_INVENTORY_ID));
    dispatch(fetchScope2Reconciliation(DEMO_INVENTORY_ID));
  }, [dispatch]);

  if (loading && !summary) return <LoadingSpinner message="Loading Scope 2 data..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  if (!summary) {
    return (
      <Card>
        <CardContent sx={{ textAlign: 'center', py: 6 }}>
          <Typography variant="body1" color="text.secondary">
            No Scope 2 data available yet. Submit energy consumption data to begin.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Stat cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <StatCard
            title="Location-Based Total"
            value={formatEmissions(summary.location_based_tco2e)}
            icon={<ElectricBolt sx={{ color: '#1e88e5' }} />}
            color="#1e88e5"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard
            title="Market-Based Total"
            value={formatEmissions(summary.market_based_tco2e)}
            icon={<ElectricBolt sx={{ color: '#43a047' }} />}
            color="#43a047"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard
            title="Delta"
            value={formatEmissions(Math.abs(summary.reconciliation_delta_tco2e))}
            subtitle={`${summary.reconciliation_delta_percent > 0 ? '+' : ''}${summary.reconciliation_delta_percent.toFixed(1)}%`}
            color={summary.reconciliation_delta_tco2e < 0 ? '#2e7d32' : '#c62828'}
          />
        </Grid>
      </Grid>

      {/* Dual reporting comparison */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Dual Reporting Comparison
        </Typography>
        <DualReportingComparison summary={summary} />
      </Box>

      {/* Reconciliation waterfall */}
      {reconciliation && (
        <Box sx={{ mb: 3 }}>
          <ReconciliationWaterfall reconciliation={reconciliation} />
        </Box>
      )}

      {/* Instrument tracker */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Contractual Instruments
        </Typography>
        <InstrumentTracker
          instruments={[]}
          totalConsumptionMwh={0}
        />
      </Box>
    </Box>
  );
};

export default Scope2Page;
