/**
 * EmissionsManagement Page - Emission sources CRUD
 *
 * Full emission source management with add form, table with inline
 * actions, category filtering, and gas breakdown visualization.
 */

import React, { useEffect, useState, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Alert,
  Button,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Add } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchSources,
  addSource,
  deleteSource,
  quantifySource,
} from '../store/slices/emissionsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import EmissionSourceForm from '../components/quantification/EmissionSourceForm';
import EmissionSourceTable from '../components/quantification/EmissionSourceTable';
import GasBreakdownChart from '../components/quantification/GasBreakdownChart';
import { ISOCategory, ISO_CATEGORY_SHORT_NAMES } from '../types';
import type { AddEmissionSourceRequest } from '../types';
import { formatTCO2e } from '../utils/formatters';

const EmissionsManagement: React.FC = () => {
  const { id: inventoryId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { sources, loading, error } = useSelector(
    (s: AppRootState) => s.emissions,
  );
  const [formOpen, setFormOpen] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  useEffect(() => {
    if (inventoryId) {
      dispatch(fetchSources(inventoryId));
    }
  }, [dispatch, inventoryId]);

  const filteredSources = useMemo(() => {
    if (categoryFilter === 'all') return sources;
    return sources.filter((s) => s.category === categoryFilter);
  }, [sources, categoryFilter]);

  const gasBreakdown = useMemo(() => {
    const breakdown: Record<string, number> = {};
    filteredSources.forEach((s) => {
      breakdown[s.gas] = (breakdown[s.gas] || 0) + s.tco2e;
    });
    return breakdown;
  }, [filteredSources]);

  const totalTco2e = filteredSources.reduce((sum, s) => sum + s.tco2e, 0);

  const handleAdd = (data: AddEmissionSourceRequest) => {
    if (inventoryId) {
      dispatch(addSource({ inventoryId, payload: data }));
    }
  };

  const handleQuantify = (sourceId: string) => {
    if (inventoryId) {
      dispatch(quantifySource({ inventoryId, sourceId }));
    }
  };

  const handleDelete = (sourceId: string) => {
    if (inventoryId) {
      dispatch(deleteSource({ inventoryId, sourceId }));
    }
  };

  if (loading && sources.length === 0) {
    return <LoadingSpinner message="Loading emission sources..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Emission Sources
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {filteredSources.length} sources | Total: {formatTCO2e(totalTco2e)}
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setFormOpen(true)}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          Add Source
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Filter + Gas chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Box sx={{ mb: 2 }}>
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel>Category Filter</InputLabel>
              <Select
                value={categoryFilter}
                label="Category Filter"
                onChange={(e) => setCategoryFilter(e.target.value)}
              >
                <MenuItem value="all">All Categories</MenuItem>
                {Object.values(ISOCategory).map((cat) => (
                  <MenuItem key={cat} value={cat}>
                    {ISO_CATEGORY_SHORT_NAMES[cat]}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          <EmissionSourceTable
            sources={filteredSources}
            onQuantify={handleQuantify}
            onDelete={handleDelete}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Gas Breakdown
              </Typography>
              <GasBreakdownChart data={gasBreakdown} height={280} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <EmissionSourceForm
        open={formOpen}
        onClose={() => setFormOpen(false)}
        onSubmit={handleAdd}
        initialCategory={
          categoryFilter !== 'all'
            ? (categoryFilter as ISOCategory)
            : undefined
        }
        loading={loading}
      />
    </Box>
  );
};

export default EmissionsManagement;
