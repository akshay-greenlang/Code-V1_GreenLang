/**
 * Scope 3 Page - Value chain emissions
 *
 * Composes a Scope 3 total stat card, CategoryOverview grid,
 * MaterialityMatrix scatter plot, and a CategoryDetail panel
 * that opens when a category card is clicked.
 */

import React, { useEffect, useState } from 'react';
import { Box, Alert, Typography, Grid, Card, CardContent, Button, Drawer, IconButton } from '@mui/material';
import { AccountTree, Close } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../store/hooks';
import { fetchScope3Summary, fetchScope3Materiality } from '../store/slices/scope3Slice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import StatCard from '../components/common/StatCard';
import CategoryOverview from '../components/scope3/CategoryOverview';
import CategoryDetail from '../components/scope3/CategoryDetail';
import MaterialityMatrix from '../components/scope3/MaterialityMatrix';
import { formatEmissions, formatPercentage } from '../utils/formatters';
import type { Scope3CategoryBreakdown } from '../types';

const DEMO_INVENTORY_ID = 'demo-inventory';

const Scope3Page: React.FC = () => {
  const dispatch = useAppDispatch();
  const { summary, categories, materiality, loading, error } = useAppSelector(
    (state) => state.scope3
  );
  const [selectedCategory, setSelectedCategory] = useState<Scope3CategoryBreakdown | null>(null);

  useEffect(() => {
    dispatch(fetchScope3Summary(DEMO_INVENTORY_ID));
    dispatch(fetchScope3Materiality(DEMO_INVENTORY_ID));
  }, [dispatch]);

  if (loading && !summary) return <LoadingSpinner message="Loading Scope 3 data..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  const materialCount = categories.filter((c) => c.is_material).length;
  const upstreamTotal = summary?.upstream_total_tco2e ?? 0;
  const downstreamTotal = summary?.downstream_total_tco2e ?? 0;

  return (
    <Box>
      {/* Stat cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={3}>
          <StatCard
            title="Scope 3 Total"
            value={summary ? formatEmissions(summary.total_tco2e) : '--'}
            icon={<AccountTree sx={{ color: '#43a047' }} />}
            color="#43a047"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <StatCard
            title="Upstream (Cat 1-8)"
            value={formatEmissions(upstreamTotal)}
            color="#2e7d32"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <StatCard
            title="Downstream (Cat 9-15)"
            value={formatEmissions(downstreamTotal)}
            color="#1e88e5"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <StatCard
            title="Material Categories"
            value={`${materialCount} of ${categories.length}`}
            subtitle={`${formatPercentage(categories.length > 0 ? (materialCount / categories.length) * 100 : 0)} coverage`}
          />
        </Grid>
      </Grid>

      {/* Category overview grid */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          All 15 Categories
        </Typography>
        <CategoryOverview
          categories={categories}
          onCategoryClick={(cat) => setSelectedCategory(cat)}
        />
      </Box>

      {/* Materiality matrix */}
      {materiality.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <MaterialityMatrix materiality={materiality} />
        </Box>
      )}

      {/* Category detail drawer */}
      <Drawer
        anchor="right"
        open={!!selectedCategory}
        onClose={() => setSelectedCategory(null)}
        sx={{ '& .MuiDrawer-paper': { width: { xs: '100%', md: 600 }, p: 3 } }}
      >
        {selectedCategory && (
          <Box>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
              <IconButton onClick={() => setSelectedCategory(null)}>
                <Close />
              </IconButton>
            </Box>
            <CategoryDetail category={selectedCategory} />
          </Box>
        )}
      </Drawer>
    </Box>
  );
};

export default Scope3Page;
