/**
 * StrategyOpportunities - Climate opportunity registry with detail drawer and revenue chart.
 */

import React, { useMemo, useState } from 'react';
import { Grid, Card, CardContent, Typography, Box, Button, Dialog, DialogTitle, DialogContent, DialogActions, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from 'recharts';
import { Add, EmojiObjects } from '@mui/icons-material';
import DataTable, { Column } from '../components/common/DataTable';
import StatusBadge from '../components/common/StatusBadge';

interface OppRow {
  id: string;
  name: string;
  type: string;
  time_horizon: string;
  revenue_potential: number;
  cost_savings: number;
  investment_required: number;
  net_value: number;
  status: string;
  responsible: string;
}

const DEMO_OPPS: OppRow[] = [
  { id: '1', name: 'Green product line expansion', type: 'products_services', time_horizon: 'medium_term', revenue_potential: 25000000, cost_savings: 3000000, investment_required: 8000000, net_value: 20000000, status: 'evaluating', responsible: 'Product Team' },
  { id: '2', name: 'Renewable energy procurement (PPA)', type: 'energy_source', time_horizon: 'short_term', revenue_potential: 0, cost_savings: 8500000, investment_required: 2000000, net_value: 6500000, status: 'implementing', responsible: 'Operations' },
  { id: '3', name: 'Energy efficiency retrofits', type: 'resource_efficiency', time_horizon: 'short_term', revenue_potential: 0, cost_savings: 4200000, investment_required: 6000000, net_value: -1800000, status: 'approved', responsible: 'Facilities' },
  { id: '4', name: 'Carbon offset marketplace', type: 'markets', time_horizon: 'medium_term', revenue_potential: 15000000, cost_savings: 0, investment_required: 5000000, net_value: 10000000, status: 'identified', responsible: 'Strategy' },
  { id: '5', name: 'Climate resilience consulting', type: 'products_services', time_horizon: 'long_term', revenue_potential: 12000000, cost_savings: 0, investment_required: 3000000, net_value: 9000000, status: 'evaluating', responsible: 'Consulting' },
  { id: '6', name: 'Supply chain decarbonization SaaS', type: 'products_services', time_horizon: 'medium_term', revenue_potential: 18000000, cost_savings: 1200000, investment_required: 7000000, net_value: 12200000, status: 'identified', responsible: 'Engineering' },
];

const TYPE_COLORS: Record<string, string> = {
  resource_efficiency: '#1B5E20',
  energy_source: '#0D47A1',
  products_services: '#7B1FA2',
  markets: '#EF6C00',
  resilience: '#00838F',
};

const StrategyOpportunities: React.FC = () => {
  const [selected, setSelected] = useState<OppRow | null>(null);

  const byTypeData = useMemo(() => {
    const groups: Record<string, { revenue: number; savings: number; investment: number }> = {};
    DEMO_OPPS.forEach((o) => {
      if (!groups[o.type]) groups[o.type] = { revenue: 0, savings: 0, investment: 0 };
      groups[o.type].revenue += o.revenue_potential;
      groups[o.type].savings += o.cost_savings;
      groups[o.type].investment += o.investment_required;
    });
    return Object.entries(groups).map(([type, vals]) => ({
      type: type.replace(/_/g, ' '),
      revenue: vals.revenue / 1e6,
      savings: vals.savings / 1e6,
      investment: vals.investment / 1e6,
    }));
  }, []);

  const columns: Column<OppRow>[] = [
    { id: 'name', label: 'Opportunity', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'type', label: 'Type', accessor: (r) => <Chip label={r.type.replace(/_/g, ' ')} size="small" sx={{ backgroundColor: TYPE_COLORS[r.type] || '#9E9E9E', color: 'white', textTransform: 'capitalize' }} />, sortAccessor: (r) => r.type },
    { id: 'time_horizon', label: 'Horizon', accessor: (r) => r.time_horizon.replace(/_/g, ' '), sortAccessor: (r) => r.time_horizon },
    { id: 'revenue_potential', label: 'Revenue Potential', accessor: (r) => r.revenue_potential > 0 ? `$${(r.revenue_potential / 1e6).toFixed(1)}M` : '-', sortAccessor: (r) => r.revenue_potential, align: 'right' },
    { id: 'cost_savings', label: 'Cost Savings', accessor: (r) => r.cost_savings > 0 ? `$${(r.cost_savings / 1e6).toFixed(1)}M` : '-', sortAccessor: (r) => r.cost_savings, align: 'right' },
    { id: 'net_value', label: 'Net Value', accessor: (r) => <Typography variant="body2" sx={{ color: r.net_value >= 0 ? 'success.main' : 'error.main', fontWeight: 600 }}>${(r.net_value / 1e6).toFixed(1)}M</Typography>, sortAccessor: (r) => r.net_value, align: 'right' },
    { id: 'status', label: 'Status', accessor: (r) => <StatusBadge status={r.status} /> },
  ];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Climate Opportunities</Typography>
          <Typography variant="body2" color="text.secondary">
            TCFD Strategy Disclosure (a) -- Opportunities identified across time horizons
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<Add />}>Add Opportunity</Button>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Revenue & Savings by Opportunity Type ($M)</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={byTypeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" fontSize={11} />
                  <YAxis tickFormatter={(v) => `$${v}M`} />
                  <Tooltip formatter={(value: number) => [`$${value.toFixed(1)}M`, '']} />
                  <Legend />
                  <Bar dataKey="revenue" name="Revenue Potential" fill="#1B5E20" />
                  <Bar dataKey="savings" name="Cost Savings" fill="#0D47A1" />
                  <Bar dataKey="investment" name="Investment Required" fill="#EF6C00" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <DataTable
        columns={columns}
        data={DEMO_OPPS}
        keyAccessor={(r) => r.id}
        defaultSortColumn="net_value"
        defaultSortDirection="desc"
        onRowClick={(r) => setSelected(r)}
        searchPlaceholder="Search opportunities..."
      />

      <Dialog open={!!selected} onClose={() => setSelected(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Opportunity Details</DialogTitle>
        <DialogContent>
          {selected && (
            <Box sx={{ pt: 1 }}>
              <Typography variant="h6" gutterBottom>{selected.name}</Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Type</Typography><Typography sx={{ textTransform: 'capitalize' }}>{selected.type.replace(/_/g, ' ')}</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Status</Typography><StatusBadge status={selected.status} /></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Revenue Potential</Typography><Typography>${(selected.revenue_potential / 1e6).toFixed(1)}M</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Cost Savings</Typography><Typography>${(selected.cost_savings / 1e6).toFixed(1)}M</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Investment Required</Typography><Typography>${(selected.investment_required / 1e6).toFixed(1)}M</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Net Value</Typography><Typography sx={{ fontWeight: 600, color: selected.net_value >= 0 ? 'success.main' : 'error.main' }}>${(selected.net_value / 1e6).toFixed(1)}M</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Responsible</Typography><Typography>{selected.responsible}</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Time Horizon</Typography><Typography sx={{ textTransform: 'capitalize' }}>{selected.time_horizon.replace(/_/g, ' ')}</Typography></Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelected(null)}>Close</Button>
          <Button variant="contained">Edit</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default StrategyOpportunities;
