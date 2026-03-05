/**
 * RemovalsManagement Page - Removal sources CRUD with permanence summary
 */

import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import { Box, Typography, Alert, Button } from '@mui/material';
import { Add } from '@mui/icons-material';
import type { AppDispatch, AppRootState } from '../store';
import {
  fetchRemovals,
  addRemoval,
  deleteRemoval,
} from '../store/slices/removalsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import RemovalSourceForm from '../components/removals/RemovalSourceForm';
import RemovalSourceTable from '../components/removals/RemovalSourceTable';
import RemovalsSummaryCard from '../components/removals/RemovalsSummaryCard';
import type { AddRemovalSourceRequest, RemovalSource } from '../types';

const RemovalsManagement: React.FC = () => {
  const { id: inventoryId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { removals, loading, error } = useSelector(
    (s: AppRootState) => s.removals,
  );
  const [formOpen, setFormOpen] = useState(false);
  const [editSource, setEditSource] = useState<RemovalSource | null>(null);

  useEffect(() => {
    if (inventoryId) {
      dispatch(fetchRemovals(inventoryId));
    }
  }, [dispatch, inventoryId]);

  const handleAdd = (data: AddRemovalSourceRequest) => {
    if (inventoryId) {
      dispatch(addRemoval({ inventoryId, payload: data }));
    }
  };

  const handleEdit = (removal: RemovalSource) => {
    setEditSource(removal);
    setFormOpen(true);
  };

  const handleDelete = (removalId: string) => {
    if (inventoryId) {
      dispatch(deleteRemoval({ inventoryId, removalId }));
    }
  };

  if (loading && removals.length === 0) {
    return <LoadingSpinner message="Loading removal sources..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            GHG Removals
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ISO 14064-1 Clause 6.2 - GHG removals and storage
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => {
            setEditSource(null);
            setFormOpen(true);
          }}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          Add Removal
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Summary Card */}
      <Box sx={{ mb: 3 }}>
        <RemovalsSummaryCard removals={removals} />
      </Box>

      {/* Table */}
      <RemovalSourceTable
        removals={removals}
        onEdit={handleEdit}
        onDelete={handleDelete}
      />

      {/* Form Dialog */}
      <RemovalSourceForm
        open={formOpen}
        onClose={() => {
          setFormOpen(false);
          setEditSource(null);
        }}
        onSubmit={handleAdd}
        editSource={editSource}
        loading={loading}
      />
    </Box>
  );
};

export default RemovalsManagement;
