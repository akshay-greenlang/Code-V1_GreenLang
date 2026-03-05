/**
 * EmissionSourceTable - MUI DataGrid of emission sources
 *
 * Displays all emission sources for an inventory with inline actions
 * (quantify, delete), category filtering, and sorting.
 */

import React from 'react';
import {
  IconButton,
  Tooltip,
  Chip,
  Box,
} from '@mui/material';
import { Calculate, Delete } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import type { EmissionSource } from '../../types';
import { ISO_CATEGORY_SHORT_NAMES, GAS_COLORS, GHGGas } from '../../types';
import { formatNumber, getDataQualityColor } from '../../utils/formatters';

interface EmissionSourceTableProps {
  sources: EmissionSource[];
  onQuantify: (sourceId: string) => void;
  onDelete: (sourceId: string) => void;
}

const EmissionSourceTable: React.FC<EmissionSourceTableProps> = ({
  sources,
  onQuantify,
  onDelete,
}) => {
  const columns: Column<EmissionSource>[] = [
    { id: 'source_name', label: 'Source Name' },
    {
      id: 'category',
      label: 'Category',
      render: (row) => (
        <Chip
          label={ISO_CATEGORY_SHORT_NAMES[row.category]}
          size="small"
          variant="outlined"
        />
      ),
    },
    {
      id: 'gas',
      label: 'Gas',
      render: (row) => (
        <Chip
          label={row.gas}
          size="small"
          sx={{
            bgcolor: (GAS_COLORS[row.gas as GHGGas] || '#9e9e9e') + '22',
            color: GAS_COLORS[row.gas as GHGGas] || '#9e9e9e',
            fontWeight: 600,
          }}
        />
      ),
    },
    {
      id: 'activity_data',
      label: 'Activity Data',
      align: 'right',
      render: (row) => `${formatNumber(row.activity_data)} ${row.activity_unit}`,
    },
    {
      id: 'emission_factor',
      label: 'EF',
      align: 'right',
      render: (row) => formatNumber(row.emission_factor, 4),
    },
    {
      id: 'tco2e',
      label: 'tCO2e',
      align: 'right',
      render: (row) => (
        <Box sx={{ fontWeight: 600 }}>{formatNumber(row.tco2e, 3)}</Box>
      ),
      getValue: (row) => row.tco2e,
    },
    {
      id: 'data_quality_tier',
      label: 'Quality',
      align: 'center',
      render: (row) => (
        <Chip
          label={row.data_quality_tier.replace(/_/g, ' ')}
          size="small"
          color={getDataQualityColor(row.data_quality_tier)}
          variant="outlined"
        />
      ),
    },
    {
      id: 'method',
      label: 'Method',
      render: (row) => row.method.replace(/_/g, ' '),
    },
    {
      id: 'actions',
      label: 'Actions',
      sortable: false,
      align: 'center',
      render: (row) => (
        <>
          <Tooltip title="Quantify emissions">
            <IconButton
              size="small"
              color="primary"
              onClick={(e) => {
                e.stopPropagation();
                onQuantify(row.id);
              }}
            >
              <Calculate fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete source">
            <IconButton
              size="small"
              color="error"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(row.id);
              }}
            >
              <Delete fontSize="small" />
            </IconButton>
          </Tooltip>
        </>
      ),
    },
  ];

  return (
    <DataTable
      columns={columns}
      rows={sources}
      rowKey={(r) => r.id}
      dense
      searchPlaceholder="Search emission sources..."
      defaultSort="tco2e"
      defaultOrder="desc"
    />
  );
};

export default EmissionSourceTable;
