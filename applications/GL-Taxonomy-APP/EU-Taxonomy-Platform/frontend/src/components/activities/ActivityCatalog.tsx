/**
 * ActivityCatalog - Filterable list of taxonomy-eligible economic activities.
 */

import React from 'react';
import { Box } from '@mui/material';
import DataTable from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import { AlignmentStatus } from '../../types';

const DEMO_ACTIVITIES = [
  { id: '1', nace_code: 'D35.11', name: 'Electricity generation (solar PV)', sector: 'Energy', objective: 'CCM', type: 'Own Performance', status: AlignmentStatus.ALIGNED, turnover: 45000000 },
  { id: '2', nace_code: 'D35.11', name: 'Electricity generation (wind)', sector: 'Energy', objective: 'CCM', type: 'Own Performance', status: AlignmentStatus.ALIGNED, turnover: 38000000 },
  { id: '3', nace_code: 'F41.2', name: 'Renovation of existing buildings', sector: 'Construction', objective: 'CCM', type: 'Transitional', status: AlignmentStatus.DNSH_PASS, turnover: 22000000 },
  { id: '4', nace_code: 'H49.1', name: 'Passenger rail transport', sector: 'Transport', objective: 'CCM', type: 'Own Performance', status: AlignmentStatus.SC_PASS, turnover: 15000000 },
  { id: '5', nace_code: 'C23.1', name: 'Manufacture of flat glass', sector: 'Manufacturing', objective: 'CCM', type: 'Transitional', status: AlignmentStatus.ELIGIBLE, turnover: 12000000 },
  { id: '6', nace_code: 'J62.0', name: 'Data-driven climate solutions', sector: 'ICT', objective: 'CCM', type: 'Enabling', status: AlignmentStatus.ALIGNED, turnover: 8000000 },
  { id: '7', nace_code: 'E36.0', name: 'Water collection and treatment', sector: 'Water', objective: 'WTR', type: 'Own Performance', status: AlignmentStatus.NOT_STARTED, turnover: 6000000 },
];

const columns = [
  { key: 'nace_code' as const, label: 'NACE', width: 80 },
  { key: 'name' as const, label: 'Activity Name' },
  { key: 'sector' as const, label: 'Sector', width: 120 },
  { key: 'objective' as const, label: 'Objective', width: 80 },
  { key: 'type' as const, label: 'Type', width: 120 },
  { key: 'status' as const, label: 'Status', width: 130, format: (val: unknown) => <StatusBadge status={val as AlignmentStatus} /> },
  { key: 'turnover' as const, label: 'Turnover', align: 'right' as const, width: 120, format: (val: unknown) => `EUR ${((val as number) / 1e6).toFixed(1)}M` },
];

interface ActivityCatalogProps {
  onSelect?: (activity: typeof DEMO_ACTIVITIES[0]) => void;
}

const ActivityCatalog: React.FC<ActivityCatalogProps> = ({ onSelect }) => (
  <Box>
    <DataTable
      columns={columns}
      data={DEMO_ACTIVITIES}
      keyField="id"
      defaultSortBy="turnover"
      defaultSortDir="desc"
      onRowClick={onSelect}
    />
  </Box>
);

export default ActivityCatalog;
