/**
 * TargetTable - List of all targets with status and actions.
 */

import React from 'react';
import { Button, Box } from '@mui/material';
import { Add, Download } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import type { Target } from '../../types';
import { formatTargetMethod, formatPercentageAbs } from '../../utils/formatters';

interface TargetTableProps {
  targets: Target[];
  onRowClick: (target: Target) => void;
  onAddNew: () => void;
}

const TargetTable: React.FC<TargetTableProps> = ({ targets, onRowClick, onAddNew }) => {
  const columns: Column<Target>[] = [
    { id: 'name', label: 'Target Name', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'scope', label: 'Scope', accessor: (r) => r.target_scope.replace(/_/g, ' ').toUpperCase() },
    { id: 'timeframe', label: 'Timeframe', accessor: (r) => r.target_timeframe === 'near_term' ? 'Near-Term' : 'Long-Term' },
    { id: 'method', label: 'Method', accessor: (r) => formatTargetMethod(r.target_method) },
    { id: 'reduction', label: 'Reduction', accessor: (r) => formatPercentageAbs(r.target_reduction_pct), align: 'right', sortAccessor: (r) => r.target_reduction_pct },
    { id: 'years', label: 'Period', accessor: (r) => `${r.base_year}-${r.target_year}` },
    { id: 'progress', label: 'Progress', accessor: (r) => formatPercentageAbs(r.progress_pct), align: 'right', sortAccessor: (r) => r.progress_pct },
    { id: 'alignment', label: 'Alignment', accessor: (r) => <StatusBadge status={r.pathway_alignment} variant="alignment" /> },
    { id: 'status', label: 'Status', accessor: (r) => <StatusBadge status={r.status} variant="target" /> },
  ];

  return (
    <DataTable
      columns={columns}
      data={targets}
      keyAccessor={(r) => r.id}
      title="All Targets"
      onRowClick={onRowClick}
      defaultSortColumn="name"
      toolbar={
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button variant="contained" size="small" startIcon={<Add />} onClick={onAddNew}>New Target</Button>
          <Button variant="outlined" size="small" startIcon={<Download />}>Export</Button>
        </Box>
      }
    />
  );
};

export default TargetTable;
