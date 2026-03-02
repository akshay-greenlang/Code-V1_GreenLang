/**
 * FindingTracker - Verification findings management table
 *
 * Displays a DataTable of verification findings with severity chips,
 * status badges, scope tags, and emissions impact. Provides an
 * "Add Finding" dialog and "Resolve" action per row. Supports
 * filtering by severity and status.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  IconButton,
  Tooltip as MuiTooltip,
  SelectChangeEvent,
} from '@mui/material';
import {
  Add,
  CheckCircle,
  BugReport,
  FilterList,
  Flag,
} from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import type { Finding, AddFindingRequest, FindingSeverity, Scope } from '../../types';
import { formatDate } from '../../utils/formatters';

interface FindingTrackerProps {
  findings: Finding[];
  verificationId: string;
  onAddFinding: (verificationId: string, finding: AddFindingRequest) => void;
  onResolveFinding: (verificationId: string, findingId: string, resolution: string) => void;
  readOnly?: boolean;
}

const SEVERITY_COLORS: Record<string, 'error' | 'warning' | 'info' | 'default'> = {
  critical: 'error',
  high: 'warning',
  medium: 'info',
  low: 'default',
};

const SEVERITY_OPTIONS: { value: FindingSeverity; label: string }[] = [
  { value: 'critical' as FindingSeverity, label: 'Critical' },
  { value: 'high' as FindingSeverity, label: 'High' },
  { value: 'medium' as FindingSeverity, label: 'Medium' },
  { value: 'low' as FindingSeverity, label: 'Low' },
];

const STATUS_COLORS: Record<string, 'success' | 'warning' | 'info' | 'default'> = {
  resolved: 'success',
  accepted: 'success',
  in_progress: 'warning',
  open: 'info',
};

const SCOPE_OPTIONS: { value: Scope; label: string }[] = [
  { value: 'scope_1' as Scope, label: 'Scope 1' },
  { value: 'scope_2' as Scope, label: 'Scope 2' },
  { value: 'scope_3' as Scope, label: 'Scope 3' },
];

const CATEGORY_OPTIONS = [
  'Data accuracy',
  'Calculation methodology',
  'Emission factors',
  'Boundary completeness',
  'Documentation',
  'Internal controls',
  'Scope classification',
  'Double counting',
  'Other',
];

const FindingTracker: React.FC<FindingTrackerProps> = ({
  findings,
  verificationId,
  onAddFinding,
  onResolveFinding,
  readOnly = false,
}) => {
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [resolveDialogOpen, setResolveDialogOpen] = useState(false);
  const [selectedFinding, setSelectedFinding] = useState<Finding | null>(null);
  const [resolution, setResolution] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const [newFinding, setNewFinding] = useState<Partial<AddFindingRequest>>({
    category: '',
    severity: 'medium' as FindingSeverity,
    description: '',
    affected_scope: null,
    affected_category: null,
    emissions_impact_tco2e: null,
    recommendation: '',
  });

  // Filter findings
  const filteredFindings = findings.filter((f) => {
    if (severityFilter !== 'all' && f.severity !== severityFilter) return false;
    if (statusFilter !== 'all' && f.status !== statusFilter) return false;
    return true;
  });

  // Summary counts
  const openCount = findings.filter((f) => f.status === 'open' || f.status === 'in_progress').length;
  const resolvedCount = findings.filter((f) => f.status === 'resolved' || f.status === 'accepted').length;
  const criticalCount = findings.filter((f) => f.severity === 'critical').length;
  const highCount = findings.filter((f) => f.severity === 'high').length;

  const handleAddFinding = () => {
    if (newFinding.category && newFinding.description && newFinding.recommendation) {
      onAddFinding(verificationId, newFinding as AddFindingRequest);
      setAddDialogOpen(false);
      setNewFinding({
        category: '',
        severity: 'medium' as FindingSeverity,
        description: '',
        affected_scope: null,
        affected_category: null,
        emissions_impact_tco2e: null,
        recommendation: '',
      });
    }
  };

  const handleOpenResolve = (finding: Finding) => {
    setSelectedFinding(finding);
    setResolution(finding.management_response || '');
    setResolveDialogOpen(true);
  };

  const handleResolveFinding = () => {
    if (selectedFinding && resolution.trim()) {
      onResolveFinding(verificationId, selectedFinding.id, resolution);
      setResolveDialogOpen(false);
      setSelectedFinding(null);
      setResolution('');
    }
  };

  // Table columns
  const columns: Column<Finding>[] = [
    {
      id: 'severity',
      label: 'Severity',
      render: (row) => (
        <Chip
          label={row.severity}
          size="small"
          color={SEVERITY_COLORS[row.severity] || 'default'}
        />
      ),
      getValue: (row) => row.severity,
    },
    {
      id: 'category',
      label: 'Category',
      getValue: (row) => row.category,
    },
    {
      id: 'description',
      label: 'Description',
      render: (row) => (
        <Typography variant="body2" sx={{ maxWidth: 250 }} noWrap>
          {row.description}
        </Typography>
      ),
      getValue: (row) => row.description,
    },
    {
      id: 'affected_scope',
      label: 'Scope',
      render: (row) => (
        row.affected_scope ? (
          <Chip
            label={row.affected_scope.replace('_', ' ')}
            size="small"
            variant="outlined"
          />
        ) : (
          <Typography variant="body2" color="text.secondary">-</Typography>
        )
      ),
      getValue: (row) => row.affected_scope || '',
    },
    {
      id: 'emissions_impact_tco2e',
      label: 'Impact (tCO2e)',
      align: 'right',
      render: (row) => (
        <Typography variant="body2">
          {row.emissions_impact_tco2e != null
            ? row.emissions_impact_tco2e.toLocaleString()
            : '-'}
        </Typography>
      ),
      getValue: (row) => row.emissions_impact_tco2e ?? 0,
    },
    {
      id: 'status',
      label: 'Status',
      render: (row) => (
        <Chip
          label={row.status.replace('_', ' ')}
          size="small"
          color={STATUS_COLORS[row.status] || 'default'}
          variant="outlined"
        />
      ),
      getValue: (row) => row.status,
    },
    {
      id: 'created_at',
      label: 'Date',
      render: (row) => (
        <Typography variant="body2">{formatDate(row.created_at)}</Typography>
      ),
      getValue: (row) => row.created_at,
    },
    ...(!readOnly
      ? [{
          id: 'actions' as const,
          label: 'Actions',
          sortable: false,
          render: (row: Finding) =>
            row.status !== 'resolved' && row.status !== 'accepted' ? (
              <MuiTooltip title="Resolve Finding">
                <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleOpenResolve(row); }}>
                  <CheckCircle fontSize="small" color="success" />
                </IconButton>
              </MuiTooltip>
            ) : (
              <Typography variant="caption" color="success.main">Resolved</Typography>
            ),
        } as Column<Finding>]
      : []),
  ];

  return (
    <Box>
      {/* Summary cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary">Total</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{findings.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="info.main">Open</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'info.main' }}>{openCount}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="error.main">Critical/High</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'error.main' }}>{criticalCount + highCount}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="success.main">Resolved</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: 'success.main' }}>{resolvedCount}</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters and actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap', gap: 1 }}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel>Severity</InputLabel>
            <Select
              value={severityFilter}
              label="Severity"
              onChange={(e: SelectChangeEvent) => setSeverityFilter(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              {SEVERITY_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={statusFilter}
              label="Status"
              onChange={(e: SelectChangeEvent) => setStatusFilter(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="open">Open</MenuItem>
              <MenuItem value="in_progress">In Progress</MenuItem>
              <MenuItem value="resolved">Resolved</MenuItem>
              <MenuItem value="accepted">Accepted</MenuItem>
            </Select>
          </FormControl>
        </Box>
        {!readOnly && (
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setAddDialogOpen(true)}
            size="small"
          >
            Add Finding
          </Button>
        )}
      </Box>

      {/* Findings table */}
      {filteredFindings.length === 0 ? (
        <Alert severity="info">
          {findings.length === 0
            ? 'No findings recorded yet.'
            : 'No findings match the current filters.'}
        </Alert>
      ) : (
        <DataTable
          columns={columns}
          rows={filteredFindings}
          rowKey={(row) => row.id}
          searchPlaceholder="Search findings..."
          defaultSort="severity"
          defaultOrder="desc"
          dense
        />
      )}

      {/* Add Finding Dialog */}
      <Dialog open={addDialogOpen} onClose={() => setAddDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <BugReport color="warning" />
            Add Verification Finding
          </Box>
        </DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '16px !important' }}>
          <FormControl fullWidth size="small">
            <InputLabel>Category</InputLabel>
            <Select
              value={newFinding.category || ''}
              label="Category"
              onChange={(e: SelectChangeEvent) => setNewFinding({ ...newFinding, category: e.target.value })}
            >
              {CATEGORY_OPTIONS.map((cat) => (
                <MenuItem key={cat} value={cat}>{cat}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth size="small">
            <InputLabel>Severity</InputLabel>
            <Select
              value={newFinding.severity || 'medium'}
              label="Severity"
              onChange={(e: SelectChangeEvent) => setNewFinding({ ...newFinding, severity: e.target.value as FindingSeverity })}
            >
              {SEVERITY_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            label="Description"
            fullWidth
            multiline
            rows={3}
            value={newFinding.description || ''}
            onChange={(e) => setNewFinding({ ...newFinding, description: e.target.value })}
          />

          <Grid container spacing={2}>
            <Grid item xs={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Affected Scope</InputLabel>
                <Select
                  value={newFinding.affected_scope || ''}
                  label="Affected Scope"
                  onChange={(e: SelectChangeEvent) =>
                    setNewFinding({ ...newFinding, affected_scope: (e.target.value || null) as Scope | null })
                  }
                >
                  <MenuItem value="">None</MenuItem>
                  {SCOPE_OPTIONS.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="Emissions Impact (tCO2e)"
                type="number"
                fullWidth
                size="small"
                value={newFinding.emissions_impact_tco2e ?? ''}
                onChange={(e) =>
                  setNewFinding({
                    ...newFinding,
                    emissions_impact_tco2e: e.target.value ? parseFloat(e.target.value) : null,
                  })
                }
              />
            </Grid>
          </Grid>

          <TextField
            label="Recommendation"
            fullWidth
            multiline
            rows={2}
            value={newFinding.recommendation || ''}
            onChange={(e) => setNewFinding({ ...newFinding, recommendation: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAddFinding}
            disabled={!newFinding.category || !newFinding.description || !newFinding.recommendation}
          >
            Add Finding
          </Button>
        </DialogActions>
      </Dialog>

      {/* Resolve Finding Dialog */}
      <Dialog open={resolveDialogOpen} onClose={() => setResolveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CheckCircle color="success" />
            Resolve Finding
          </Box>
        </DialogTitle>
        <DialogContent sx={{ pt: '16px !important' }}>
          {selectedFinding && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                <Chip
                  label={selectedFinding.severity}
                  size="small"
                  color={SEVERITY_COLORS[selectedFinding.severity] || 'default'}
                />
                <Chip label={selectedFinding.category} size="small" variant="outlined" />
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {selectedFinding.description}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                Recommendation:
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {selectedFinding.recommendation}
              </Typography>
            </Box>
          )}

          <TextField
            label="Management Response / Resolution"
            fullWidth
            multiline
            rows={4}
            value={resolution}
            onChange={(e) => setResolution(e.target.value)}
            placeholder="Describe the corrective action taken to address this finding..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResolveDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            color="success"
            onClick={handleResolveFinding}
            disabled={!resolution.trim()}
            startIcon={<CheckCircle />}
          >
            Resolve
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FindingTracker;
