import React from 'react';
import { Card, CardContent, Chip } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import type { ClimateOpportunity } from '../../types';
import { formatCurrency, formatTimeHorizon } from '../../utils/formatters';

interface OpportunityRegistryProps {
  opportunities: ClimateOpportunity[];
}

const STATUS_COLORS: Record<string, 'default' | 'info' | 'warning' | 'success' | 'primary'> = {
  identified: 'default', evaluating: 'info', approved: 'warning', implementing: 'primary', realized: 'success',
};

const OpportunityRegistry: React.FC<OpportunityRegistryProps> = ({ opportunities }) => {
  const columns: Column<ClimateOpportunity>[] = [
    { id: 'name', label: 'Opportunity', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'type', label: 'Type', accessor: (r) => r.opportunity_type.replace(/_/g, ' '), sortAccessor: (r) => r.opportunity_type },
    { id: 'priority', label: 'Priority', accessor: (r) => <Chip label={r.strategic_priority} size="small" color={r.strategic_priority === 'critical' ? 'error' : r.strategic_priority === 'high' ? 'warning' : 'default'} />, sortAccessor: (r) => r.strategic_priority },
    { id: 'horizon', label: 'Time Horizon', accessor: (r) => formatTimeHorizon(r.time_horizon), sortAccessor: (r) => r.time_horizon },
    { id: 'revenue', label: 'Revenue Potential', accessor: (r) => formatCurrency(r.revenue_potential_mid, 'USD', true), align: 'right', sortAccessor: (r) => r.revenue_potential_mid },
    { id: 'savings', label: 'Cost Savings', accessor: (r) => formatCurrency(r.cost_savings_mid, 'USD', true), align: 'right', sortAccessor: (r) => r.cost_savings_mid },
    { id: 'investment', label: 'Investment', accessor: (r) => formatCurrency(r.investment_required, 'USD', true), align: 'right', sortAccessor: (r) => r.investment_required },
    { id: 'payback', label: 'Payback (yrs)', accessor: (r) => r.payback_period_years.toFixed(1), align: 'center', sortAccessor: (r) => r.payback_period_years },
    { id: 'status', label: 'Status', accessor: (r) => <Chip label={r.status.replace(/_/g, ' ')} size="small" color={STATUS_COLORS[r.status] || 'default'} />, sortAccessor: (r) => r.status },
  ];

  return (
    <Card>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <DataTable title="Climate Opportunity Registry" columns={columns} data={opportunities} keyAccessor={(r) => r.id} emptyMessage="No opportunities registered" />
      </CardContent>
    </Card>
  );
};

export default OpportunityRegistry;
