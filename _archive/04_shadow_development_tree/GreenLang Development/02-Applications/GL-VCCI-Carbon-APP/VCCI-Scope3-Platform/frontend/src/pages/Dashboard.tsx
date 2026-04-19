import React, { useEffect } from 'react';
import { Grid, Typography, Box, Alert } from '@mui/material';
import {
  CloudQueue as EmissionsIcon,
  AttachMoney as SpendIcon,
  Business as SupplierIcon,
  Description as TransactionIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchDashboardMetrics, fetchHotspotAnalysis } from '../store/slices/dashboardSlice';
import StatCard from '../components/StatCard';
import { CategoryPieChart, MonthlyTrendChart, TopSuppliersChart } from '../components/EmissionsChart';
import LoadingSpinner from '../components/LoadingSpinner';
import { formatCurrency, formatNumber } from '../utils/formatters';

const Dashboard: React.FC = () => {
  const dispatch = useAppDispatch();
  const { metrics, hotspots, loading, error } = useAppSelector((state) => state.dashboard);

  useEffect(() => {
    dispatch(fetchDashboardMetrics());
    dispatch(fetchHotspotAnalysis({ topN: 10 }));
  }, [dispatch]);

  if (loading && !metrics) {
    return <LoadingSpinner message="Loading dashboard metrics..." />;
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!metrics) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        No data available. Please upload transaction data to get started.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Emissions"
            value={`${formatNumber(metrics.totalEmissionsTCO2e, 2)} t CO₂e`}
            icon={EmissionsIcon}
            color="secondary"
            subtitle="Scope 3 Total"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Spend"
            value={formatCurrency(metrics.totalSpendUsd)}
            icon={SpendIcon}
            color="primary"
            subtitle="Analyzed Transactions"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Suppliers"
            value={formatNumber(metrics.supplierCount)}
            icon={SupplierIcon}
            color="info"
            subtitle="In Supply Chain"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Transactions"
            value={formatNumber(metrics.transactionCount)}
            icon={TransactionIcon}
            color="warning"
            subtitle={`Avg DQI: ${metrics.dataQualityAvg.toFixed(2)}`}
          />
        </Grid>
      </Grid>

      {/* Charts Row 1 */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <CategoryPieChart
            data={metrics.categoryBreakdown}
            title="Emissions by GHG Category"
            height={350}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TopSuppliersChart
            data={metrics.topSuppliers.slice(0, 10)}
            title="Top 10 Suppliers by Emissions"
            height={350}
          />
        </Grid>
      </Grid>

      {/* Charts Row 2 */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <MonthlyTrendChart
            data={metrics.monthlyTrend}
            title="Monthly Emissions & Spend Trend"
            height={300}
          />
        </Grid>
      </Grid>

      {/* Hotspots Section */}
      {hotspots && hotspots.hotspots.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h5" gutterBottom>
            Emissions Hotspots
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            Potential reduction: {formatNumber(hotspots.totalPotentialReduction, 2)} t CO₂e
            {' | '}
            Cost savings: {formatCurrency(hotspots.totalCostSavings)}
          </Alert>
          <Grid container spacing={2}>
            {hotspots.hotspots.slice(0, 6).map((hotspot) => (
              <Grid item xs={12} sm={6} md={4} key={hotspot.id}>
                <StatCard
                  title={hotspot.name}
                  value={`${formatNumber(hotspot.emissionsTCO2e, 2)} t CO₂e`}
                  icon={EmissionsIcon}
                  color={hotspot.priority === 'high' ? 'error' : hotspot.priority === 'medium' ? 'warning' : 'success'}
                  subtitle={`${hotspot.reductionPercentage.toFixed(1)}% reduction potential`}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;
