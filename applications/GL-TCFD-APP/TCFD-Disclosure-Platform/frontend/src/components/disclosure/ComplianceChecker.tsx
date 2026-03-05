import React from 'react';
import { Card, CardContent, Typography, Chip, Box, LinearProgress } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import type { ComplianceCheck } from '../../types';

interface ComplianceCheckerProps { checks: ComplianceCheck[]; onRunCheck: () => void; }

const ComplianceChecker: React.FC<ComplianceCheckerProps> = ({ checks, onRunCheck }) => {
  const met = checks.filter((c) => c.status === 'met').length;
  const total = checks.filter((c) => c.status !== 'not_applicable').length;
  const pct = total > 0 ? (met / total * 100) : 0;
  const columns: Column<ComplianceCheck>[] = [
    { id: 'code', label: 'Code', accessor: (r) => r.requirement_code, sortAccessor: (r) => r.requirement_code },
    { id: 'req', label: 'Requirement', accessor: (r) => r.requirement, sortAccessor: (r) => r.requirement },
    { id: 'framework', label: 'Framework', accessor: (r) => r.framework.toUpperCase(), sortAccessor: (r) => r.framework },
    { id: 'status', label: 'Status', accessor: (r) => <Chip label={r.status.replace(/_/g, ' ')} size="small" color={r.status === 'met' ? 'success' : r.status === 'partial' ? 'warning' : r.status === 'not_met' ? 'error' : 'default'} />, sortAccessor: (r) => r.status },
    { id: 'quality', label: 'Evidence Quality', accessor: (r) => <Chip label={r.evidence_quality} size="small" variant="outlined" color={r.evidence_quality === 'strong' ? 'success' : r.evidence_quality === 'adequate' ? 'primary' : 'warning'} />, sortAccessor: (r) => r.evidence_quality },
    { id: 'gap', label: 'Gap / Recommendation', accessor: (r) => r.gap_description || r.recommendation || '-' },
  ];
  return (
    <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Compliance Checker</Typography>
        <Box sx={{ flexGrow: 1 }}><LinearProgress variant="determinate" value={pct} sx={{ height: 8, borderRadius: 4 }} color={pct >= 80 ? 'success' : pct >= 50 ? 'warning' : 'error'} /></Box>
        <Typography variant="body2" sx={{ fontWeight: 700 }}>{pct.toFixed(0)}%</Typography>
      </Box>
      <DataTable columns={columns} data={checks} keyAccessor={(r) => r.id} emptyMessage="Run a compliance check to see results" />
    </CardContent></Card>
  );
};

export default ComplianceChecker;
