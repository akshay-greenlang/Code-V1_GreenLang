import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import type { GovernanceCommittee } from '../../types';
import { formatDate } from '../../utils/formatters';

interface CommitteeTrackerProps {
  committees: GovernanceCommittee[];
}

const CommitteeTracker: React.FC<CommitteeTrackerProps> = ({ committees }) => {
  const columns: Column<GovernanceCommittee>[] = [
    { id: 'name', label: 'Committee', accessor: (r) => r.name, sortAccessor: (r) => r.name },
    { id: 'type', label: 'Type', accessor: (r) => <Chip label={r.type.replace(/_/g, ' ')} size="small" variant="outlined" />, sortAccessor: (r) => r.type },
    { id: 'chair', label: 'Chair', accessor: (r) => r.chair, sortAccessor: (r) => r.chair },
    { id: 'members', label: 'Members', accessor: (r) => r.members.length, align: 'center', sortAccessor: (r) => r.members.length },
    { id: 'frequency', label: 'Meeting Freq.', accessor: (r) => r.meeting_frequency },
    { id: 'climate_pct', label: 'Climate Agenda %', accessor: (r) => {
      const pct = r.total_agenda_items > 0 ? (r.climate_agenda_items / r.total_agenda_items * 100) : 0;
      return <Chip label={`${pct.toFixed(0)}%`} size="small" color={pct >= 30 ? 'success' : pct >= 15 ? 'warning' : 'default'} />;
    }, align: 'center', sortAccessor: (r) => r.total_agenda_items > 0 ? r.climate_agenda_items / r.total_agenda_items : 0 },
    { id: 'last_meeting', label: 'Last Meeting', accessor: (r) => formatDate(r.last_meeting_date), sortAccessor: (r) => r.last_meeting_date },
    { id: 'next_meeting', label: 'Next Meeting', accessor: (r) => formatDate(r.next_meeting_date), sortAccessor: (r) => r.next_meeting_date },
  ];

  return (
    <Card>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <DataTable
          title="Committee Meetings & Composition"
          columns={columns}
          data={committees}
          keyAccessor={(r) => r.id}
          emptyMessage="No committees configured"
        />
      </CardContent>
    </Card>
  );
};

export default CommitteeTracker;
