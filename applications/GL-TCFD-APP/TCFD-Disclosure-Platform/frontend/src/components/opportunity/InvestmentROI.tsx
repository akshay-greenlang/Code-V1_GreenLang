import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import { formatCurrency } from '../../utils/formatters';

interface InvestmentROIProps { data: { id: string; name: string; investment: number; npv: number; irr: number; payback: number }[]; }

const InvestmentROI: React.FC<InvestmentROIProps> = ({ data }) => {
  const columns: Column<typeof data[0]>[] = [
    { id: 'name', label: 'Opportunity', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'investment', label: 'Investment', accessor: (r) => formatCurrency(r.investment, 'USD', true), align: 'right', sortAccessor: (r) => r.investment },
    { id: 'npv', label: 'NPV', accessor: (r) => formatCurrency(r.npv, 'USD', true), align: 'right', sortAccessor: (r) => r.npv },
    { id: 'irr', label: 'IRR', accessor: (r) => <Chip label={`${(r.irr * 100).toFixed(1)}%`} size="small" color={r.irr >= 0.15 ? 'success' : r.irr >= 0.08 ? 'warning' : 'default'} />, align: 'center', sortAccessor: (r) => r.irr },
    { id: 'payback', label: 'Payback (yrs)', accessor: (r) => r.payback.toFixed(1), align: 'center', sortAccessor: (r) => r.payback },
  ];
  return <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}><DataTable title="Investment & ROI Analysis" columns={columns} data={data} keyAccessor={(r) => r.id} /></CardContent></Card>;
};

export default InvestmentROI;
