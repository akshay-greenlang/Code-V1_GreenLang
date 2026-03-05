import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import RiskBadge from '../common/RiskBadge';
import type { SupplyChainRiskNode } from '../../types';

interface SupplyChainRiskProps { data: SupplyChainRiskNode[]; }

const SupplyChainRisk: React.FC<SupplyChainRiskProps> = ({ data }) => {
  const columns: Column<SupplyChainRiskNode>[] = [
    { id: 'supplier', label: 'Supplier', accessor: (r) => r.supplier_name, sortAccessor: (r) => r.supplier_name },
    { id: 'tier', label: 'Tier', accessor: (r) => `Tier ${r.tier}`, align: 'center', sortAccessor: (r) => r.tier },
    { id: 'location', label: 'Location', accessor: (r) => r.location },
    { id: 'level', label: 'Risk Level', accessor: (r) => <RiskBadge level={r.risk_level} />, sortAccessor: (r) => r.risk_level },
    { id: 'hazards', label: 'Hazard Exposures', accessor: (r) => r.hazard_exposures.slice(0, 3).map((h) => <Chip key={h} label={h.replace(/_/g, ' ')} size="small" variant="outlined" sx={{ mr: 0.5, fontSize: 10, height: 20 }} />) },
    { id: 'dependency', label: 'Revenue Dep.', accessor: (r) => `${(r.revenue_dependency * 100).toFixed(0)}%`, align: 'center', sortAccessor: (r) => r.revenue_dependency },
    { id: 'alternatives', label: 'Alternatives', accessor: (r) => r.alternative_suppliers, align: 'center', sortAccessor: (r) => r.alternative_suppliers },
    { id: 'lead_time', label: 'Lead Time Impact', accessor: (r) => `${r.lead_time_impact_days}d`, align: 'center', sortAccessor: (r) => r.lead_time_impact_days },
  ];
  return (
    <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
      <DataTable title="Supply Chain Risk Propagation" columns={columns} data={data} keyAccessor={(r) => r.id} emptyMessage="No supply chain risk data" />
    </CardContent></Card>
  );
};

export default SupplyChainRisk;
