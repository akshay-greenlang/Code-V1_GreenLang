/**
 * EntityList - Entity tree/table with add/edit/delete
 *
 * Displays all reporting entities under an organization in a table
 * with inline add, edit, and delete capabilities.
 */

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Grid,
  MenuItem,
  Tooltip,
  Chip,
} from '@mui/material';
import { Add, Edit, Delete } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import type { Entity, AddEntityRequest } from '../../types';
import { formatNumber } from '../../utils/formatters';

const ENTITY_TYPES = [
  'subsidiary',
  'joint_venture',
  'branch',
  'facility',
  'division',
  'affiliate',
];

interface EntityListProps {
  entities: Entity[];
  onAdd: (data: AddEntityRequest) => void;
  onEdit: (entityId: string, data: Partial<AddEntityRequest>) => void;
  onDelete: (entityId: string) => void;
  loading?: boolean;
}

const EntityList: React.FC<EntityListProps> = ({
  entities,
  onAdd,
  onEdit,
  onDelete,
  loading = false,
}) => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editEntity, setEditEntity] = useState<Entity | null>(null);
  const [form, setForm] = useState<AddEntityRequest>({
    name: '',
    entity_type: 'subsidiary',
    country: '',
    ownership_pct: 100,
    employees: null,
    revenue: null,
    floor_area_m2: null,
  });

  const openAdd = () => {
    setEditEntity(null);
    setForm({
      name: '',
      entity_type: 'subsidiary',
      country: '',
      ownership_pct: 100,
      employees: null,
      revenue: null,
      floor_area_m2: null,
    });
    setDialogOpen(true);
  };

  const openEdit = (entity: Entity) => {
    setEditEntity(entity);
    setForm({
      name: entity.name,
      entity_type: entity.entity_type,
      country: entity.country,
      ownership_pct: entity.ownership_pct,
      employees: entity.employees,
      revenue: entity.revenue,
      floor_area_m2: entity.floor_area_m2,
    });
    setDialogOpen(true);
  };

  const handleSubmit = () => {
    if (editEntity) {
      onEdit(editEntity.id, form);
    } else {
      onAdd(form);
    }
    setDialogOpen(false);
  };

  const columns: Column<Entity>[] = [
    { id: 'name', label: 'Entity Name' },
    {
      id: 'entity_type',
      label: 'Type',
      render: (row) => (
        <Chip
          label={row.entity_type.replace(/_/g, ' ')}
          size="small"
          variant="outlined"
        />
      ),
    },
    { id: 'country', label: 'Country' },
    {
      id: 'ownership_pct',
      label: 'Ownership %',
      align: 'right',
      render: (row) => `${row.ownership_pct}%`,
    },
    {
      id: 'employees',
      label: 'Employees',
      align: 'right',
      render: (row) => (row.employees != null ? formatNumber(row.employees, 0) : '--'),
    },
    {
      id: 'active',
      label: 'Status',
      render: (row) => (
        <Chip
          label={row.active ? 'Active' : 'Inactive'}
          color={row.active ? 'success' : 'default'}
          size="small"
        />
      ),
    },
    {
      id: 'actions',
      label: 'Actions',
      sortable: false,
      align: 'center',
      render: (row) => (
        <>
          <Tooltip title="Edit">
            <IconButton size="small" onClick={() => openEdit(row)}>
              <Edit fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton
              size="small"
              color="error"
              onClick={() => onDelete(row.id)}
            >
              <Delete fontSize="small" />
            </IconButton>
          </Tooltip>
        </>
      ),
    },
  ];

  return (
    <Card>
      <CardHeader
        title="Reporting Entities"
        subheader={`${entities.length} entities defined`}
        action={
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={openAdd}
            size="small"
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Add Entity
          </Button>
        }
      />
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <DataTable
          columns={columns}
          rows={entities}
          rowKey={(r) => r.id}
          dense
          searchPlaceholder="Search entities..."
        />
      </CardContent>

      {/* Add/Edit Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editEntity ? 'Edit Entity' : 'Add Entity'}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                required
                label="Entity Name"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Entity Type"
                value={form.entity_type}
                onChange={(e) => setForm({ ...form, entity_type: e.target.value })}
              >
                {ENTITY_TYPES.map((t) => (
                  <MenuItem key={t} value={t}>
                    {t.replace(/_/g, ' ')}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Country"
                value={form.country}
                onChange={(e) => setForm({ ...form, country: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Ownership %"
                value={form.ownership_pct}
                onChange={(e) =>
                  setForm({ ...form, ownership_pct: Number(e.target.value) })
                }
                inputProps={{ min: 0, max: 100 }}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                type="number"
                label="Employees"
                value={form.employees ?? ''}
                onChange={(e) =>
                  setForm({
                    ...form,
                    employees: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                type="number"
                label="Revenue (USD)"
                value={form.revenue ?? ''}
                onChange={(e) =>
                  setForm({
                    ...form,
                    revenue: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                type="number"
                label="Floor Area (m2)"
                value={form.floor_area_m2 ?? ''}
                onChange={(e) =>
                  setForm({
                    ...form,
                    floor_area_m2: e.target.value ? Number(e.target.value) : null,
                  })
                }
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={!form.name || !form.country || loading}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            {editEntity ? 'Save' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default EntityList;
