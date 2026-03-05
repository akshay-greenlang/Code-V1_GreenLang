/**
 * GapTable - Interactive gap requirements table with filtering, sorting, and inline gap details.
 *
 * Displays all identified disclosure gaps with priority, remediation actions,
 * owner assignments, due dates, and status tracking. Supports pillar and priority filtering.
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  IconButton,
  Collapse,
} from '@mui/material';
import { KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';

interface GapItem {
  id: string;
  requirement: string;
  requirement_code: string;
  pillar: string;
  framework: string;
  current_maturity: number;
  target_maturity: number;
  gap: number;
  priority: 'critical' | 'high' | 'medium' | 'low';
  action: string;
  owner: string;
  due_date: string;
  status: 'not_started' | 'planning' | 'in_progress' | 'completed' | 'deferred';
  effort: 'low' | 'medium' | 'high';
  dependencies: string[];
  notes: string;
}

interface GapTableProps {
  gaps: GapItem[];
  onStatusChange?: (gapId: string, newStatus: string) => void;
  onActionClick?: (gapId: string) => void;
}

const PRIORITY_COLORS: Record<string, 'error' | 'warning' | 'info' | 'default'> = {
  critical: 'error',
  high: 'warning',
  medium: 'info',
  low: 'default',
};

const STATUS_COLORS: Record<string, 'success' | 'warning' | 'info' | 'default' | 'error'> = {
  completed: 'success',
  in_progress: 'warning',
  planning: 'info',
  not_started: 'default',
  deferred: 'error',
};

const EFFORT_STYLES: Record<string, { label: string; color: string }> = {
  low: { label: 'Low Effort', color: '#E8F5E9' },
  medium: { label: 'Medium Effort', color: '#FFF3E0' },
  high: { label: 'High Effort', color: '#FFEBEE' },
};

const GapTable: React.FC<GapTableProps> = ({ gaps, onStatusChange, onActionClick }) => {
  const [pillarFilter, setPillarFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const filteredGaps = useMemo(() => {
    let result = gaps;
    if (pillarFilter !== 'all') result = result.filter((g) => g.pillar === pillarFilter);
    if (priorityFilter !== 'all') result = result.filter((g) => g.priority === priorityFilter);
    if (statusFilter !== 'all') result = result.filter((g) => g.status === statusFilter);
    return result;
  }, [gaps, pillarFilter, priorityFilter, statusFilter]);

  const summaryStats = useMemo(() => {
    const total = gaps.length;
    const completed = gaps.filter((g) => g.status === 'completed').length;
    const critical = gaps.filter((g) => g.priority === 'critical' && g.status !== 'completed').length;
    const avgGap = Math.round(gaps.reduce((s, g) => s + g.gap, 0) / gaps.length);
    return { total, completed, critical, avgGap };
  }, [gaps]);

  const pillars = useMemo(() => Array.from(new Set(gaps.map((g) => g.pillar))), [gaps]);

  const columns: Column<GapItem>[] = [
    {
      id: 'expand',
      label: '',
      accessor: (r) => (
        <IconButton size="small" onClick={() => setExpandedRow(expandedRow === r.id ? null : r.id)}>
          {expandedRow === r.id ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
        </IconButton>
      ),
    },
    {
      id: 'requirement',
      label: 'Requirement',
      accessor: (r) => (
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {r.requirement}
          </Typography>
          <Chip label={r.requirement_code} size="small" variant="outlined" sx={{ fontSize: '0.65rem', mt: 0.5, height: 18 }} />
        </Box>
      ),
      sortAccessor: (r) => r.requirement,
    },
    {
      id: 'pillar',
      label: 'Pillar',
      accessor: (r) => (
        <Chip label={r.pillar} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
      ),
      sortAccessor: (r) => r.pillar,
    },
    {
      id: 'gap',
      label: 'Gap',
      accessor: (r) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box sx={{ width: 40, height: 6, borderRadius: 3, backgroundColor: '#E0E0E0', overflow: 'hidden' }}>
            <Box
              sx={{
                width: `${Math.min(r.gap * 2.5, 100)}%`,
                height: '100%',
                borderRadius: 3,
                backgroundColor: r.gap > 25 ? '#C62828' : r.gap > 15 ? '#EF6C00' : '#2E7D32',
              }}
            />
          </Box>
          <Typography
            variant="body2"
            sx={{ fontWeight: 600, color: r.gap > 25 ? 'error.main' : r.gap > 15 ? 'warning.main' : 'text.primary' }}
          >
            {r.gap}pts
          </Typography>
        </Box>
      ),
      sortAccessor: (r) => r.gap,
      align: 'center',
    },
    {
      id: 'priority',
      label: 'Priority',
      accessor: (r) => (
        <Chip
          label={r.priority}
          size="small"
          color={PRIORITY_COLORS[r.priority]}
          sx={{ textTransform: 'capitalize' }}
        />
      ),
      sortAccessor: (r) => {
        const order: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };
        return order[r.priority] ?? 4;
      },
    },
    {
      id: 'owner',
      label: 'Owner',
      accessor: (r) => r.owner,
      sortAccessor: (r) => r.owner,
    },
    {
      id: 'due_date',
      label: 'Due',
      accessor: (r) => {
        const isOverdue = new Date(r.due_date) < new Date() && r.status !== 'completed';
        return (
          <Typography
            variant="body2"
            sx={{ fontSize: '0.8rem', color: isOverdue ? 'error.main' : 'text.primary', fontWeight: isOverdue ? 600 : 400 }}
          >
            {new Date(r.due_date).toLocaleDateString()}
          </Typography>
        );
      },
      sortAccessor: (r) => r.due_date,
    },
    {
      id: 'status',
      label: 'Status',
      accessor: (r) => (
        <Chip
          label={r.status.replace(/_/g, ' ')}
          size="small"
          color={STATUS_COLORS[r.status]}
          sx={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
        />
      ),
      sortAccessor: (r) => r.status,
    },
    {
      id: 'effort',
      label: 'Effort',
      accessor: (r) => (
        <Chip
          label={EFFORT_STYLES[r.effort].label}
          size="small"
          sx={{ backgroundColor: EFFORT_STYLES[r.effort].color, fontSize: '0.65rem', height: 20 }}
        />
      ),
      sortAccessor: (r) => r.effort,
    },
  ];

  return (
    <Card>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        {/* Summary Bar */}
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Gap Remediation Tracker
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
            <Chip label={`${summaryStats.total} gaps`} size="small" variant="outlined" />
            <Chip
              label={`${summaryStats.completed} complete`}
              size="small"
              color="success"
              variant="outlined"
            />
            {summaryStats.critical > 0 && (
              <Chip
                label={`${summaryStats.critical} critical`}
                size="small"
                color="error"
              />
            )}
            <Chip label={`Avg gap: ${summaryStats.avgGap}pts`} size="small" variant="outlined" />
          </Box>
        </Box>

        {/* Filters */}
        <Box sx={{ px: 2, pb: 2, display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Pillar</InputLabel>
            <Select
              value={pillarFilter}
              label="Pillar"
              onChange={(e: SelectChangeEvent) => setPillarFilter(e.target.value)}
            >
              <MenuItem value="all">All Pillars</MenuItem>
              {pillars.map((p) => (
                <MenuItem key={p} value={p}>
                  {p}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel>Priority</InputLabel>
            <Select
              value={priorityFilter}
              label="Priority"
              onChange={(e: SelectChangeEvent) => setPriorityFilter(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={statusFilter}
              label="Status"
              onChange={(e: SelectChangeEvent) => setStatusFilter(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="not_started">Not Started</MenuItem>
              <MenuItem value="planning">Planning</MenuItem>
              <MenuItem value="in_progress">In Progress</MenuItem>
              <MenuItem value="completed">Completed</MenuItem>
              <MenuItem value="deferred">Deferred</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <DataTable
          columns={columns}
          data={filteredGaps}
          keyAccessor={(r) => r.id}
          defaultSortColumn="gap"
          defaultSortDirection="desc"
          emptyMessage="No gaps match the current filters."
        />
      </CardContent>
    </Card>
  );
};

export default GapTable;
