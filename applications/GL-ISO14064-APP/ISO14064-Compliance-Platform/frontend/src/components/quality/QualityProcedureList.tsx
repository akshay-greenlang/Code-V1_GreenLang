/**
 * QualityProcedureList - Procedures table with status and review schedule
 *
 * Displays quality management procedures per ISO 14064-1 Clause 7
 * with status indicators, responsible person, frequency, and
 * next review date.
 */

import React from 'react';
import { Card, CardContent, CardHeader, Chip } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import type { QualityProcedure } from '../../types';
import { formatDate, getStatusColor } from '../../utils/formatters';

interface QualityProcedureListProps {
  procedures: QualityProcedure[];
}

const QualityProcedureList: React.FC<QualityProcedureListProps> = ({
  procedures,
}) => {
  const columns: Column<QualityProcedure>[] = [
    { id: 'title', label: 'Procedure' },
    {
      id: 'procedure_type',
      label: 'Type',
      render: (row) => (
        <Chip
          label={row.procedure_type.replace(/_/g, ' ')}
          size="small"
          variant="outlined"
        />
      ),
    },
    { id: 'responsible', label: 'Responsible' },
    { id: 'frequency', label: 'Frequency' },
    {
      id: 'status',
      label: 'Status',
      render: (row) => (
        <Chip
          label={row.status.replace(/_/g, ' ')}
          color={getStatusColor(row.status)}
          size="small"
        />
      ),
    },
    {
      id: 'last_review',
      label: 'Last Review',
      render: (row) => formatDate(row.last_review),
    },
    {
      id: 'next_review',
      label: 'Next Review',
      render: (row) => {
        if (!row.next_review) return '--';
        const isOverdue = new Date(row.next_review) < new Date();
        return (
          <Chip
            label={formatDate(row.next_review)}
            size="small"
            color={isOverdue ? 'error' : 'default'}
            variant={isOverdue ? 'filled' : 'outlined'}
          />
        );
      },
    },
    {
      id: 'description',
      label: 'Description',
      render: (row) =>
        row.description.length > 60
          ? row.description.substring(0, 57) + '...'
          : row.description,
    },
  ];

  return (
    <Card>
      <CardHeader
        title="Quality Management Procedures"
        subheader={`${procedures.length} procedures defined (ISO 14064-1 Clause 7)`}
      />
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <DataTable
          columns={columns}
          rows={procedures}
          rowKey={(r) => r.id}
          dense
          searchPlaceholder="Search procedures..."
        />
      </CardContent>
    </Card>
  );
};

export default QualityProcedureList;
