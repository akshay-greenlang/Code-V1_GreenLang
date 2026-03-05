/**
 * MappingTable - TCFD-to-ISSB/IFRS S2 requirement mapping table.
 *
 * Displays the crosswalk between TCFD recommendations and ISSB/IFRS S2 paragraphs
 * with alignment status, compliance tracking, and gap identification.
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Tooltip as MUITooltip,
  IconButton,
} from '@mui/material';
import { Info, OpenInNew } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';

type AlignmentType = 'full' | 'partial' | 'enhanced' | 'new';
type ComplianceStatus = 'complete' | 'in_progress' | 'not_started';

interface CrosswalkMapping {
  id: string;
  tcfd_section: string;
  tcfd_code: string;
  tcfd_recommendation: string;
  issb_paragraph: string;
  issb_requirement: string;
  alignment: AlignmentType;
  tcfd_status: ComplianceStatus;
  issb_status: ComplianceStatus;
  gap_notes: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  migration_effort: string;
}

interface MappingTableProps {
  mappings: CrosswalkMapping[];
  onMappingClick?: (mappingId: string) => void;
}

const ALIGNMENT_CONFIG: Record<AlignmentType, { label: string; color: string; bgColor: string; description: string }> = {
  full: {
    label: 'Full Alignment',
    color: '#2E7D32',
    bgColor: '#E8F5E9',
    description: 'Direct mapping between TCFD and ISSB requirements',
  },
  partial: {
    label: 'Partial',
    color: '#F57F17',
    bgColor: '#FFF9C4',
    description: 'Partial overlap; some ISSB requirements extend beyond TCFD',
  },
  enhanced: {
    label: 'Enhanced Required',
    color: '#0D47A1',
    bgColor: '#E3F2FD',
    description: 'ISSB requires significantly more detail than TCFD',
  },
  new: {
    label: 'New in ISSB',
    color: '#C62828',
    bgColor: '#FFEBEE',
    description: 'No TCFD equivalent; entirely new ISSB requirement',
  },
};

const STATUS_COLORS: Record<ComplianceStatus, 'success' | 'warning' | 'default'> = {
  complete: 'success',
  in_progress: 'warning',
  not_started: 'default',
};

const MappingTable: React.FC<MappingTableProps> = ({ mappings, onMappingClick }) => {
  const [alignmentFilter, setAlignmentFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const filteredMappings = useMemo(() => {
    let result = mappings;
    if (alignmentFilter !== 'all') result = result.filter((m) => m.alignment === alignmentFilter);
    if (statusFilter !== 'all') result = result.filter((m) => m.issb_status === statusFilter);
    return result;
  }, [mappings, alignmentFilter, statusFilter]);

  const summary = useMemo(() => ({
    total: mappings.length,
    full: mappings.filter((m) => m.alignment === 'full').length,
    enhanced: mappings.filter((m) => m.alignment === 'enhanced').length,
    new_items: mappings.filter((m) => m.alignment === 'new').length,
    issb_complete: mappings.filter((m) => m.issb_status === 'complete').length,
    tcfd_complete: mappings.filter((m) => m.tcfd_status === 'complete').length,
  }), [mappings]);

  const columns: Column<CrosswalkMapping>[] = [
    {
      id: 'tcfd',
      label: 'TCFD Section',
      accessor: (r) => (
        <Box>
          <Chip
            label={r.tcfd_code}
            size="small"
            sx={{
              fontWeight: 600,
              fontSize: '0.7rem',
              backgroundColor: r.tcfd_code === 'N/A' ? '#F5F5F5' : '#E8F5E9',
              color: r.tcfd_code === 'N/A' ? '#9E9E9E' : '#1B5E20',
              mb: 0.5,
            }}
          />
          <Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: 500 }}>
            {r.tcfd_section}
          </Typography>
        </Box>
      ),
      sortAccessor: (r) => r.tcfd_code,
    },
    {
      id: 'issb',
      label: 'ISSB/IFRS S2',
      accessor: (r) => (
        <Box>
          <Chip
            label={r.issb_paragraph}
            size="small"
            sx={{
              fontWeight: 600,
              fontSize: '0.7rem',
              backgroundColor: '#E3F2FD',
              color: '#0D47A1',
              mb: 0.5,
            }}
          />
          <Typography variant="body2" sx={{ fontSize: '0.78rem' }}>
            {r.issb_requirement}
          </Typography>
        </Box>
      ),
      sortAccessor: (r) => r.issb_paragraph,
    },
    {
      id: 'alignment',
      label: 'Alignment',
      accessor: (r) => {
        const config = ALIGNMENT_CONFIG[r.alignment];
        return (
          <MUITooltip title={config.description} placement="top">
            <Chip
              label={config.label}
              size="small"
              sx={{
                backgroundColor: config.bgColor,
                color: config.color,
                fontWeight: 600,
                fontSize: '0.65rem',
                cursor: 'help',
              }}
            />
          </MUITooltip>
        );
      },
      sortAccessor: (r) => {
        const order: Record<string, number> = { new: 0, enhanced: 1, partial: 2, full: 3 };
        return order[r.alignment] ?? 4;
      },
    },
    {
      id: 'tcfd_status',
      label: 'TCFD Status',
      accessor: (r) => (
        <Chip
          label={r.tcfd_status.replace(/_/g, ' ')}
          size="small"
          color={STATUS_COLORS[r.tcfd_status]}
          sx={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
        />
      ),
      sortAccessor: (r) => r.tcfd_status,
      align: 'center',
    },
    {
      id: 'issb_status',
      label: 'ISSB Status',
      accessor: (r) => (
        <Chip
          label={r.issb_status.replace(/_/g, ' ')}
          size="small"
          color={STATUS_COLORS[r.issb_status]}
          sx={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
        />
      ),
      sortAccessor: (r) => r.issb_status,
      align: 'center',
    },
    {
      id: 'gap_notes',
      label: 'Gap Notes',
      accessor: (r) =>
        r.gap_notes ? (
          <MUITooltip title={r.gap_notes} placement="left">
            <Typography
              variant="body2"
              sx={{
                fontSize: '0.75rem',
                color: 'text.secondary',
                maxWidth: 200,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                cursor: 'help',
              }}
            >
              {r.gap_notes}
            </Typography>
          </MUITooltip>
        ) : (
          <Typography variant="body2" color="text.disabled" sx={{ fontSize: '0.75rem' }}>
            --
          </Typography>
        ),
    },
  ];

  return (
    <Card>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        {/* Header */}
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
            TCFD-to-ISSB Crosswalk Mapping
          </Typography>

          {/* Summary Chips */}
          <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
            <Chip
              label={`${summary.full} Full Alignment`}
              size="small"
              sx={{ backgroundColor: ALIGNMENT_CONFIG.full.bgColor, color: ALIGNMENT_CONFIG.full.color, fontWeight: 600 }}
            />
            <Chip
              label={`${summary.enhanced} Enhanced Required`}
              size="small"
              sx={{ backgroundColor: ALIGNMENT_CONFIG.enhanced.bgColor, color: ALIGNMENT_CONFIG.enhanced.color, fontWeight: 600 }}
            />
            <Chip
              label={`${summary.new_items} New in ISSB`}
              size="small"
              sx={{ backgroundColor: ALIGNMENT_CONFIG.new.bgColor, color: ALIGNMENT_CONFIG.new.color, fontWeight: 600 }}
            />
            <Chip
              label={`TCFD: ${summary.tcfd_complete}/${summary.total} complete`}
              size="small"
              variant="outlined"
              color="success"
            />
            <Chip
              label={`ISSB: ${summary.issb_complete}/${summary.total} complete`}
              size="small"
              variant="outlined"
              color="primary"
            />
          </Box>

          {/* Filters */}
          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl size="small" sx={{ minWidth: 160 }}>
              <InputLabel>Alignment Type</InputLabel>
              <Select
                value={alignmentFilter}
                label="Alignment Type"
                onChange={(e: SelectChangeEvent) => setAlignmentFilter(e.target.value)}
              >
                <MenuItem value="all">All Types</MenuItem>
                <MenuItem value="full">Full Alignment</MenuItem>
                <MenuItem value="partial">Partial</MenuItem>
                <MenuItem value="enhanced">Enhanced Required</MenuItem>
                <MenuItem value="new">New in ISSB</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 140 }}>
              <InputLabel>ISSB Status</InputLabel>
              <Select
                value={statusFilter}
                label="ISSB Status"
                onChange={(e: SelectChangeEvent) => setStatusFilter(e.target.value)}
              >
                <MenuItem value="all">All Statuses</MenuItem>
                <MenuItem value="complete">Complete</MenuItem>
                <MenuItem value="in_progress">In Progress</MenuItem>
                <MenuItem value="not_started">Not Started</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>

        <DataTable
          columns={columns}
          data={filteredMappings}
          keyAccessor={(r) => r.id}
          emptyMessage="No mappings match the current filters."
          onRowClick={(r) => onMappingClick?.(r.id)}
        />
      </CardContent>
    </Card>
  );
};

export default MappingTable;
