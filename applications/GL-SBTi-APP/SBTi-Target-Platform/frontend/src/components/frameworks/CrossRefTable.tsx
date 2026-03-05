/**
 * CrossRefTable - SBTi-to-framework mapping table.
 */
import React from 'react';
import DataTable, { Column } from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import type { FrameworkMapping } from '../../types';

interface CrossRefTableProps { mappings: FrameworkMapping[]; }

const CrossRefTable: React.FC<CrossRefTableProps> = ({ mappings }) => {
  const columns: Column<FrameworkMapping>[] = [
    { id: 'sbti', label: 'SBTi Requirement', accessor: (r) => r.sbti_requirement },
    { id: 'sbti_code', label: 'SBTi Code', accessor: (r) => r.sbti_code },
    { id: 'framework', label: 'Framework', accessor: (r) => r.framework.toUpperCase() },
    { id: 'fw_req', label: 'Framework Requirement', accessor: (r) => r.framework_requirement },
    { id: 'mapping', label: 'Mapping', accessor: (r) => r.mapping_type.replace(/\b\w/g, (c) => c.toUpperCase()) },
    { id: 'status', label: 'Status', accessor: (r) => <StatusBadge status={r.organization_status} variant="validation" /> },
  ];
  return <DataTable columns={columns} data={mappings} keyAccessor={(r) => r.id} title="Cross-Reference Mapping" />;
};

export default CrossRefTable;
