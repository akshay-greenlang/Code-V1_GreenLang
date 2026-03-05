import React, { useState } from 'react';
import { Card, CardContent, Box, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import RiskBadge from '../common/RiskBadge';
import type { ClimateRisk } from '../../types';
import { formatCurrency, formatTimeHorizon } from '../../utils/formatters';

interface RiskRegistryProps {
  risks: ClimateRisk[];
}

const RiskRegistry: React.FC<RiskRegistryProps> = ({ risks }) => {
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [levelFilter, setLevelFilter] = useState('all');

  const filteredRisks = risks.filter((r) => {
    if (categoryFilter !== 'all' && r.category !== categoryFilter) return false;
    if (levelFilter !== 'all' && r.risk_level !== levelFilter) return false;
    return true;
  });

  const columns: Column<ClimateRisk>[] = [
    { id: 'name', label: 'Risk', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'category', label: 'Category', accessor: (r) => r.category === 'physical' ? 'Physical' : 'Transition', sortAccessor: (r) => r.category },
    { id: 'type', label: 'Type', accessor: (r) => r.risk_type.replace(/_/g, ' '), sortAccessor: (r) => r.risk_type },
    { id: 'level', label: 'Risk Level', accessor: (r) => <RiskBadge level={r.risk_level} />, sortAccessor: (r) => r.risk_level },
    { id: 'likelihood', label: 'Likelihood', accessor: (r) => <RiskBadge level={r.likelihood} variant="outlined" />, sortAccessor: (r) => r.likelihood },
    { id: 'impact', label: 'Impact', accessor: (r) => <RiskBadge level={r.impact_severity} variant="outlined" />, sortAccessor: (r) => r.impact_severity },
    { id: 'horizon', label: 'Time Horizon', accessor: (r) => formatTimeHorizon(r.time_horizon), sortAccessor: (r) => r.time_horizon },
    { id: 'financial', label: 'Financial Impact (Mid)', accessor: (r) => formatCurrency(r.financial_impact_mid, 'USD', true), align: 'right', sortAccessor: (r) => r.financial_impact_mid },
    { id: 'owner', label: 'Owner', accessor: (r) => r.owner, sortAccessor: (r) => r.owner },
  ];

  const toolbar = (
    <Box sx={{ display: 'flex', gap: 1 }}>
      <FormControl size="small" sx={{ minWidth: 120 }}>
        <InputLabel>Category</InputLabel>
        <Select value={categoryFilter} label="Category" onChange={(e: SelectChangeEvent) => setCategoryFilter(e.target.value)}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="physical">Physical</MenuItem>
          <MenuItem value="transition">Transition</MenuItem>
        </Select>
      </FormControl>
      <FormControl size="small" sx={{ minWidth: 120 }}>
        <InputLabel>Risk Level</InputLabel>
        <Select value={levelFilter} label="Risk Level" onChange={(e: SelectChangeEvent) => setLevelFilter(e.target.value)}>
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="critical">Critical</MenuItem>
          <MenuItem value="high">High</MenuItem>
          <MenuItem value="medium">Medium</MenuItem>
          <MenuItem value="low">Low</MenuItem>
        </Select>
      </FormControl>
    </Box>
  );

  return (
    <Card>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <DataTable
          title="Climate Risk Registry"
          columns={columns}
          data={filteredRisks}
          keyAccessor={(r) => r.id}
          defaultSortColumn="level"
          toolbar={toolbar}
          emptyMessage="No risks registered"
        />
      </CardContent>
    </Card>
  );
};

export default RiskRegistry;
