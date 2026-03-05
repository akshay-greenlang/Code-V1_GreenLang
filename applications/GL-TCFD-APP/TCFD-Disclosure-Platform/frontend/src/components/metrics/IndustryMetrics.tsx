import React from 'react';
import { Card, CardContent, Typography, Chip, LinearProgress, Box } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';

interface IndustryMetricsProps { data: { metric: string; value: number; unit: string; industry_avg: number; percentile: number }[]; }

const IndustryMetrics: React.FC<IndustryMetricsProps> = ({ data }) => {
  const columns: Column<typeof data[0]>[] = [
    { id: 'metric', label: 'Metric', accessor: (r) => r.metric, sortAccessor: (r) => r.metric },
    { id: 'value', label: 'Our Value', accessor: (r) => `${r.value.toLocaleString()} ${r.unit}`, sortAccessor: (r) => r.value },
    { id: 'avg', label: 'Industry Avg', accessor: (r) => `${r.industry_avg.toLocaleString()} ${r.unit}`, sortAccessor: (r) => r.industry_avg },
    { id: 'vs', label: 'vs. Avg', accessor: (r) => {
      const diff = r.industry_avg > 0 ? ((r.value - r.industry_avg) / r.industry_avg * 100) : 0;
      return <Chip label={`${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`} size="small" color={diff <= 0 ? 'success' : 'error'} />;
    }, sortAccessor: (r) => r.value - r.industry_avg },
    { id: 'percentile', label: 'Percentile', accessor: (r) => <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><LinearProgress variant="determinate" value={r.percentile} sx={{ width: 80, height: 6, borderRadius: 3 }} color={r.percentile >= 75 ? 'success' : r.percentile >= 50 ? 'primary' : 'warning'} /><Typography variant="caption">{r.percentile}th</Typography></Box>, sortAccessor: (r) => r.percentile },
  ];
  return <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}><DataTable title="Industry-Specific Metrics" columns={columns} data={data} keyAccessor={(r) => r.metric} /></CardContent></Card>;
};

export default IndustryMetrics;
