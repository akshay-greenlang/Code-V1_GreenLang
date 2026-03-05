/**
 * EngagementTracker - Investee engagement table.
 */
import React from 'react';
import DataTable, { Column } from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import type { EngagementRecord } from '../../types';
import { formatDate } from '../../utils/formatters';

interface EngagementTrackerProps { engagements: EngagementRecord[]; }

const EngagementTracker: React.FC<EngagementTrackerProps> = ({ engagements }) => {
  const columns: Column<EngagementRecord>[] = [
    { id: 'company', label: 'Company', accessor: (r) => r.company_name, sortAccessor: (r) => r.company_name },
    { id: 'date', label: 'Date', accessor: (r) => formatDate(r.engagement_date), sortAccessor: (r) => r.engagement_date },
    { id: 'type', label: 'Type', accessor: (r) => r.engagement_type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) },
    { id: 'outcome', label: 'Outcome', accessor: (r) => r.outcome },
    { id: 'sbti', label: 'SBTi Committed', accessor: (r) => r.sbti_commitment_obtained ? 'Yes' : 'No', align: 'center' },
    { id: 'followup', label: 'Follow Up', accessor: (r) => r.follow_up_date ? formatDate(r.follow_up_date) : '-' },
  ];
  return <DataTable columns={columns} data={engagements} keyAccessor={(r) => r.id} title="Engagement Activity" defaultSortColumn="date" defaultSortDirection="desc" />;
};

export default EngagementTracker;
