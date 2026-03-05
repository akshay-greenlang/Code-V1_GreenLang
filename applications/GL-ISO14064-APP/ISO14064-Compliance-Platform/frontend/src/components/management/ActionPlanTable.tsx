/**
 * ActionPlanTable - Management actions table
 *
 * Displays management actions with status, priority, target reduction,
 * estimated cost, responsible person, and timeline.
 */

import React from 'react';
import { IconButton, Tooltip, Chip, Typography } from '@mui/material';
import { Edit, Delete } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import StatusChip from '../common/StatusChip';
import type { ManagementAction } from '../../types';
import { ISO_CATEGORY_SHORT_NAMES, ISOCategory } from '../../types';
import { formatNumber, formatDate } from '../../utils/formatters';

interface ActionPlanTableProps {
  actions: ManagementAction[];
  onEdit: (action: ManagementAction) => void;
  onDelete: (actionId: string) => void;
}

const PRIORITY_COLORS: Record<string, 'error' | 'warning' | 'info' | 'default'> = {
  high: 'error',
  medium: 'warning',
  low: 'info',
};

const ActionPlanTable: React.FC<ActionPlanTableProps> = ({
  actions,
  onEdit,
  onDelete,
}) => {
  const columns: Column<ManagementAction>[] = [
    { id: 'title', label: 'Action' },
    {
      id: 'action_category',
      label: 'Category',
      render: (row) => (
        <Chip
          label={row.action_category.replace(/_/g, ' ')}
          size="small"
          variant="outlined"
        />
      ),
    },
    {
      id: 'target_category',
      label: 'Target',
      render: (row) =>
        row.target_category
          ? ISO_CATEGORY_SHORT_NAMES[row.target_category as ISOCategory]
          : 'All',
    },
    {
      id: 'status',
      label: 'Status',
      render: (row) => <StatusChip status={row.status} />,
    },
    {
      id: 'priority',
      label: 'Priority',
      render: (row) => (
        <Chip
          label={row.priority}
          color={PRIORITY_COLORS[row.priority.toLowerCase()] || 'default'}
          size="small"
          sx={{ textTransform: 'capitalize' }}
        />
      ),
    },
    {
      id: 'estimated_reduction_tco2e',
      label: 'Reduction (tCO2e)',
      align: 'right',
      render: (row) =>
        row.estimated_reduction_tco2e != null
          ? formatNumber(row.estimated_reduction_tco2e, 1)
          : '--',
      getValue: (row) => row.estimated_reduction_tco2e ?? 0,
    },
    {
      id: 'estimated_cost_usd',
      label: 'Cost (USD)',
      align: 'right',
      render: (row) =>
        row.estimated_cost_usd != null
          ? `$${formatNumber(row.estimated_cost_usd, 0)}`
          : '--',
    },
    {
      id: 'responsible_person',
      label: 'Responsible',
      render: (row) => row.responsible_person || '--',
    },
    {
      id: 'target_date',
      label: 'Target Date',
      render: (row) => formatDate(row.target_date),
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
      rows={actions}
      rowKey={(r) => r.id}
      dense
      searchPlaceholder="Search management actions..."
      defaultSort="priority"
    />
  );
};

export default ActionPlanTable;
