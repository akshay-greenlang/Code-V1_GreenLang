/**
 * RemovalSourceTable - Table of removals with credited amounts
 *
 * Displays removal sources with type, permanence, gross/credited amounts,
 * verification status, and inline actions.
 */

import React from 'react';
import { IconButton, Tooltip, Chip } from '@mui/material';
import { Edit, Delete } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import StatusChip from '../common/StatusChip';
import type { RemovalSource } from '../../types';
import { formatNumber, getDataQualityColor } from '../../utils/formatters';

interface RemovalSourceTableProps {
  removals: RemovalSource[];
  onEdit: (removal: RemovalSource) => void;
  onDelete: (removalId: string) => void;
}

const RemovalSourceTable: React.FC<RemovalSourceTableProps> = ({
  removals,
  onEdit,
  onDelete,
}) => {
  const columns: Column<RemovalSource>[] = [
    { id: 'source_name', label: 'Source Name' },
    {
      id: 'removal_type',
      label: 'Type',
      render: (row) => (
        <Chip
          label={row.removal_type.replace(/_/g, ' ')}
          size="small"
          variant="outlined"
        />
      ),
    },
    {
      id: 'permanence_level',
      label: 'Permanence',
      render: (row) => row.permanence_level.replace(/_/g, ' '),
    },
    {
      id: 'gross_removals_tco2e',
      label: 'Gross (tCO2e)',
      align: 'right',
      render: (row) => formatNumber(row.gross_removals_tco2e, 2),
      getValue: (row) => row.gross_removals_tco2e,
    },
    {
      id: 'permanence_discount_factor',
      label: 'Discount',
      align: 'right',
      render: (row) => `${(row.permanence_discount_factor * 100).toFixed(0)}%`,
    },
    {
      id: 'credited_removals_tco2e',
      label: 'Credited (tCO2e)',
      align: 'right',
      render: (row) => (
        <strong>{formatNumber(row.credited_removals_tco2e, 2)}</strong>
      ),
      getValue: (row) => row.credited_removals_tco2e,
    },
    {
      id: 'verification_status',
      label: 'Verification',
      render: (row) => <StatusChip status={row.verification_status} />,
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
      id: 'actions',
      label: 'Actions',
      sortable: false,
      align: 'center',
      render: (row) => (
        <>
          <Tooltip title="Edit">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onEdit(row);
              }}
            >
              <Edit fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
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
      rows={removals}
      rowKey={(r) => r.id}
      dense
      searchPlaceholder="Search removal sources..."
      defaultSort="credited_removals_tco2e"
      defaultOrder="desc"
    />
  );
};

export default RemovalSourceTable;
