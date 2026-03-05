/**
 * SupplierList - Supplier management list with status
 */
import React from 'react';
import { Box, Typography } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import StatusChip from '../common/StatusChip';
import type { SupplierRequest } from '../../types';
import { formatDate } from '../../utils/formatters';

interface SupplierListProps { suppliers: SupplierRequest[]; onSelect: (s: SupplierRequest) => void; }

const columns: Column<SupplierRequest>[] = [
  { id: 'supplier_name', label: 'Supplier', render: (r) => <Typography variant="body2" fontWeight={500}>{r.supplier_name}</Typography> },
  { id: 'supplier_country', label: 'Country' },
  { id: 'supplier_sector', label: 'Sector' },
  { id: 'status', label: 'Status', render: (r) => <StatusChip status={r.status} /> },
  { id: 'score', label: 'Score', align: 'right', render: (r) => r.score != null ? `${r.score.toFixed(0)}%` : '--' },
  { id: 'invited_at', label: 'Invited', render: (r) => formatDate(r.invited_at) },
];

const SupplierList: React.FC<SupplierListProps> = ({ suppliers, onSelect }) => (
  <DataTable columns={columns} rows={suppliers} rowKey={(r) => r.id} onRowClick={onSelect} searchPlaceholder="Search suppliers..." />
);

export default SupplierList;
