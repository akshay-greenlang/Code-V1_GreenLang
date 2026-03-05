/**
 * StrategyRisks - Climate risk registry with filtering, detail drawer, and distribution chart.
 */

import React, { useMemo, useState } from 'react';
import { Grid, Card, CardContent, Typography, Box, Button, Dialog, DialogTitle, DialogContent, DialogActions, Chip, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';
import { Add, Warning } from '@mui/icons-material';
import DataTable, { Column } from '../components/common/DataTable';
import RiskBadge from '../components/common/RiskBadge';

interface RiskRow {
  id: string;
  name: string;
  category: string;
  risk_type: string;
  risk_level: string;
  likelihood: string;
  impact_severity: string;
  time_horizon: string;
  financial_impact: number;
  owner: string;
}

const DEMO_RISKS: RiskRow[] = [
  { id: '1', name: 'Carbon pricing regulation (EU CBAM)', category: 'transition', risk_type: 'policy_legal', risk_level: 'high', likelihood: 'very_likely', impact_severity: 'major', time_horizon: 'medium_term', financial_impact: 12500000, owner: 'Michael Park' },
  { id: '2', name: 'Extreme weather - coastal facilities', category: 'physical', risk_type: 'acute', risk_level: 'critical', likelihood: 'likely', impact_severity: 'catastrophic', time_horizon: 'long_term', financial_impact: 8300000, owner: 'Sarah Chen' },
  { id: '3', name: 'Technology substitution risk', category: 'transition', risk_type: 'technology', risk_level: 'medium', likelihood: 'possible', impact_severity: 'moderate', time_horizon: 'medium_term', financial_impact: 6200000, owner: 'James Mitchell' },
  { id: '4', name: 'Water stress - manufacturing sites', category: 'physical', risk_type: 'chronic', risk_level: 'high', likelihood: 'likely', impact_severity: 'major', time_horizon: 'long_term', financial_impact: 5100000, owner: 'Michael Park' },
  { id: '5', name: 'Consumer preference shift', category: 'transition', risk_type: 'market', risk_level: 'medium', likelihood: 'likely', impact_severity: 'moderate', time_horizon: 'short_term', financial_impact: 4800000, owner: 'Aisha Rahman' },
  { id: '6', name: 'Reputational risk - greenwashing', category: 'transition', risk_type: 'reputation', risk_level: 'high', likelihood: 'possible', impact_severity: 'major', time_horizon: 'short_term', financial_impact: 3200000, owner: 'Sarah Chen' },
  { id: '7', name: 'Supply chain disruption - heat', category: 'physical', risk_type: 'chronic', risk_level: 'medium', likelihood: 'possible', impact_severity: 'moderate', time_horizon: 'medium_term', financial_impact: 2800000, owner: 'Michael Park' },
  { id: '8', name: 'Mandatory disclosure regulation', category: 'transition', risk_type: 'policy_legal', risk_level: 'low', likelihood: 'almost_certain', impact_severity: 'minor', time_horizon: 'short_term', financial_impact: 1500000, owner: 'Aisha Rahman' },
];

const TYPE_COLORS: Record<string, string> = {
  policy_legal: '#0D47A1',
  technology: '#7B1FA2',
  market: '#EF6C00',
  reputation: '#C62828',
  acute: '#B71C1C',
  chronic: '#E65100',
};

const StrategyRisks: React.FC = () => {
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [horizonFilter, setHorizonFilter] = useState<string>('all');
  const [selectedRisk, setSelectedRisk] = useState<RiskRow | null>(null);

  const filteredRisks = useMemo(() => {
    let result = DEMO_RISKS;
    if (typeFilter !== 'all') result = result.filter((r) => r.category === typeFilter);
    if (horizonFilter !== 'all') result = result.filter((r) => r.time_horizon === horizonFilter);
    return result;
  }, [typeFilter, horizonFilter]);

  const byTypeData = useMemo(() => {
    const counts: Record<string, number> = {};
    DEMO_RISKS.forEach((r) => {
      counts[r.risk_type] = (counts[r.risk_type] || 0) + 1;
    });
    return Object.entries(counts).map(([type, count]) => ({
      type: type.replace(/_/g, ' '),
      count,
      fill: TYPE_COLORS[type] || '#9E9E9E',
    }));
  }, []);

  const byHorizonData = useMemo(() => {
    const sums: Record<string, number> = { short_term: 0, medium_term: 0, long_term: 0 };
    DEMO_RISKS.forEach((r) => { sums[r.time_horizon] += r.financial_impact; });
    return [
      { horizon: 'Short Term', value: sums.short_term },
      { horizon: 'Medium Term', value: sums.medium_term },
      { horizon: 'Long Term', value: sums.long_term },
    ];
  }, []);

  const columns: Column<RiskRow>[] = [
    { id: 'name', label: 'Risk Name', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'category', label: 'Category', accessor: (r) => <Chip label={r.category} size="small" variant="outlined" sx={{ textTransform: 'capitalize' }} />, sortAccessor: (r) => r.category },
    { id: 'risk_type', label: 'Type', accessor: (r) => <Chip label={r.risk_type.replace(/_/g, ' ')} size="small" sx={{ backgroundColor: TYPE_COLORS[r.risk_type] || '#9E9E9E', color: 'white', textTransform: 'capitalize' }} />, sortAccessor: (r) => r.risk_type },
    { id: 'risk_level', label: 'Level', accessor: (r) => <RiskBadge level={r.risk_level} />, sortAccessor: (r) => r.risk_level },
    { id: 'time_horizon', label: 'Time Horizon', accessor: (r) => r.time_horizon.replace(/_/g, ' '), sortAccessor: (r) => r.time_horizon },
    { id: 'financial_impact', label: 'Financial Impact', accessor: (r) => `$${(r.financial_impact / 1e6).toFixed(1)}M`, sortAccessor: (r) => r.financial_impact, align: 'right' },
    { id: 'owner', label: 'Owner', accessor: (r) => r.owner, sortAccessor: (r) => r.owner },
  ];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Climate Risks</Typography>
          <Typography variant="body2" color="text.secondary">
            TCFD Strategy Disclosure (a) -- Risks identified across time horizons
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<Add />}>Add Risk</Button>
      </Box>

      {/* Distribution Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Risk Distribution by Type</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={byTypeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" fontSize={11} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" name="Count">
                    {byTypeData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Financial Exposure by Time Horizon</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={byHorizonData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="horizon" fontSize={12} />
                  <YAxis tickFormatter={(v) => `$${(v / 1e6).toFixed(0)}M`} />
                  <Tooltip formatter={(value: number) => [`$${(value / 1e6).toFixed(1)}M`, 'Exposure']} />
                  <Bar dataKey="value" fill="#C62828" name="Exposure" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters + Table */}
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Category</InputLabel>
          <Select value={typeFilter} label="Category" onChange={(e: SelectChangeEvent) => setTypeFilter(e.target.value)}>
            <MenuItem value="all">All Categories</MenuItem>
            <MenuItem value="physical">Physical</MenuItem>
            <MenuItem value="transition">Transition</MenuItem>
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Time Horizon</InputLabel>
          <Select value={horizonFilter} label="Time Horizon" onChange={(e: SelectChangeEvent) => setHorizonFilter(e.target.value)}>
            <MenuItem value="all">All Horizons</MenuItem>
            <MenuItem value="short_term">Short Term</MenuItem>
            <MenuItem value="medium_term">Medium Term</MenuItem>
            <MenuItem value="long_term">Long Term</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <DataTable
        columns={columns}
        data={filteredRisks}
        keyAccessor={(r) => r.id}
        defaultSortColumn="financial_impact"
        defaultSortDirection="desc"
        onRowClick={(r) => setSelectedRisk(r)}
        searchPlaceholder="Search risks..."
      />

      {/* Risk Detail Dialog */}
      <Dialog open={!!selectedRisk} onClose={() => setSelectedRisk(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Risk Details</DialogTitle>
        <DialogContent>
          {selectedRisk && (
            <Box sx={{ pt: 1 }}>
              <Typography variant="h6" gutterBottom>{selectedRisk.name}</Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Category</Typography><Typography sx={{ textTransform: 'capitalize' }}>{selectedRisk.category}</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Type</Typography><Typography sx={{ textTransform: 'capitalize' }}>{selectedRisk.risk_type.replace(/_/g, ' ')}</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Risk Level</Typography><RiskBadge level={selectedRisk.risk_level} /></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Likelihood</Typography><RiskBadge level={selectedRisk.likelihood} variant="outlined" /></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Impact</Typography><RiskBadge level={selectedRisk.impact_severity} variant="outlined" /></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Time Horizon</Typography><Typography sx={{ textTransform: 'capitalize' }}>{selectedRisk.time_horizon.replace(/_/g, ' ')}</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Financial Impact</Typography><Typography sx={{ fontWeight: 600 }}>${(selectedRisk.financial_impact / 1e6).toFixed(1)}M</Typography></Grid>
                <Grid item xs={6}><Typography variant="body2" color="text.secondary">Owner</Typography><Typography>{selectedRisk.owner}</Typography></Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedRisk(null)}>Close</Button>
          <Button variant="contained">Edit Risk</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default StrategyRisks;
