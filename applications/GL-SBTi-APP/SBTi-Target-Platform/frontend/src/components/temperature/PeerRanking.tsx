/**
 * PeerRanking - Peer comparison table for temperature scores.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import type { PeerTemperatureRanking } from '../../types';
import { formatSector, formatTemperature } from '../../utils/formatters';

interface PeerRankingProps { rankings: PeerTemperatureRanking[]; }

const PeerRanking: React.FC<PeerRankingProps> = ({ rankings }) => {
  const columns: Column<PeerTemperatureRanking>[] = [
    { id: 'rank', label: '#', accessor: (r) => r.rank.toString(), sortAccessor: (r) => r.rank, width: 40 },
    { id: 'company', label: 'Company', accessor: (r) => r.is_current_org ? `${r.company_name} (You)` : r.company_name, sortAccessor: (r) => r.company_name },
    { id: 'sector', label: 'Sector', accessor: (r) => formatSector(r.sector) },
    { id: 'temperature', label: 'Temperature', accessor: (r) => formatTemperature(r.temperature_score), sortAccessor: (r) => r.temperature_score, align: 'right' },
    { id: 'alignment', label: 'Alignment', accessor: (r) => <StatusBadge status={r.alignment} variant="alignment" /> },
    { id: 'sbti', label: 'SBTi Status', accessor: (r) => <StatusBadge status={r.sbti_status} variant="target" /> },
  ];
  return (
    <DataTable columns={columns} data={rankings} keyAccessor={(r) => r.company_name} title="Peer Comparison" defaultSortColumn="rank" />
  );
};

export default PeerRanking;
