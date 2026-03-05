import React from 'react';
import { Card, CardContent, Chip } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import RiskBadge from '../common/RiskBadge';
import type { RiskManagementRecord } from '../../types';

interface RiskRegisterProps { data: RiskManagementRecord[]; }

const RiskRegister: React.FC<RiskRegisterProps> = ({ data }) => {
  const columns: Column<RiskManagementRecord>[] = [
    { id: 'name', label: 'Risk', accessor: (r) => r.risk_name, sortAccessor: (r) => r.risk_name },
    { id: 'category', label: 'Category', accessor: (r) => r.risk_category, sortAccessor: (r) => r.risk_category },
    { id: 'inherent', label: 'Inherent Score', accessor: (r) => r.inherent_risk_score, align: 'center', sortAccessor: (r) => r.inherent_risk_score },
    { id: 'level', label: 'Risk Level', accessor: (r) => <RiskBadge level={r.risk_level} />, sortAccessor: (r) => r.risk_level },
    { id: 'strategy', label: 'Response', accessor: (r) => <Chip label={r.response_strategy} size="small" variant="outlined" />, sortAccessor: (r) => r.response_strategy },
    { id: 'residual', label: 'Residual', accessor: (r) => <RiskBadge level={r.residual_risk_level} variant="outlined" />, sortAccessor: (r) => r.residual_risk_level },
    { id: 'owner', label: 'Owner', accessor: (r) => r.owner, sortAccessor: (r) => r.owner },
    { id: 'status', label: 'Status', accessor: (r) => <Chip label={r.status} size="small" color={r.status === 'closed' ? 'success' : r.status === 'mitigating' ? 'primary' : 'default'} />, sortAccessor: (r) => r.status },
    { id: 'erm', label: 'ERM', accessor: (r) => r.erm_integrated ? 'Yes' : 'No', align: 'center' },
  ];
  return <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}><DataTable title="Risk Management Register" columns={columns} data={data} keyAccessor={(r) => r.id} /></CardContent></Card>;
};

export default RiskRegister;
