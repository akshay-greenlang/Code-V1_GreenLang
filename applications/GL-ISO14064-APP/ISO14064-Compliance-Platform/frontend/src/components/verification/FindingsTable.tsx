/**
 * FindingsTable - Findings with severity color-coding and status management
 *
 * Displays verification findings in a table with severity indicators,
 * status management, and inline resolution capability.
 */

import React, { useState } from 'react';
import {
  IconButton,
  Tooltip,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
} from '@mui/material';
import { CheckCircle } from '@mui/icons-material';
import DataTable, { Column } from '../common/DataTable';
import type { Finding } from '../../types';
import { ISO_CATEGORY_SHORT_NAMES, ISOCategory } from '../../types';
import { getSeverityColor, formatDate, formatNumber } from '../../utils/formatters';
import StatusChip from '../common/StatusChip';

interface FindingsTableProps {
  findings: Finding[];
  onResolve: (findingId: string, resolution: string) => void;
}

const FindingsTable: React.FC<FindingsTableProps> = ({ findings, onResolve }) => {
  const [resolveDialogOpen, setResolveDialogOpen] = useState(false);
  const [selectedFinding, setSelectedFinding] = useState<Finding | null>(null);
  const [resolution, setResolution] = useState('');

  const openResolve = (finding: Finding) => {
    setSelectedFinding(finding);
    setResolution('');
    setResolveDialogOpen(true);
  };

  const handleResolve = () => {
    if (selectedFinding) {
      onResolve(selectedFinding.id, resolution);
    }
    setResolveDialogOpen(false);
  };

  const columns: Column<Finding>[] = [
    {
      id: 'severity',
      label: 'Severity',
      render: (row) => (
        <Chip
          label={row.severity}
          color={getSeverityColor(row.severity)}
          size="small"
          sx={{ fontWeight: 600, textTransform: 'capitalize' }}
        />
      ),
    },
    { id: 'category', label: 'Category' },
    {
      id: 'description',
      label: 'Description',
      render: (row) => (
        <Typography variant="body2" sx={{ maxWidth: 300 }} noWrap>
          {row.description}
        </Typography>
      ),
    },
    {
      id: 'affected_category',
      label: 'Affected Category',
      render: (row) =>
        row.affected_category
          ? ISO_CATEGORY_SHORT_NAMES[row.affected_category as ISOCategory]
          : '--',
    },
    {
      id: 'emissions_impact_tco2e',
      label: 'Impact (tCO2e)',
      align: 'right',
      render: (row) =>
        row.emissions_impact_tco2e != null
          ? formatNumber(row.emissions_impact_tco2e, 2)
          : '--',
    },
    {
      id: 'status',
      label: 'Status',
      render: (row) => <StatusChip status={row.status} />,
    },
    {
      id: 'created_at',
      label: 'Created',
      render: (row) => formatDate(row.created_at),
    },
    {
      id: 'actions',
      label: 'Actions',
      sortable: false,
      align: 'center',
      render: (row) =>
        row.status !== 'resolved' && row.status !== 'accepted' ? (
          <Tooltip title="Resolve finding">
            <IconButton
              size="small"
              color="success"
              onClick={(e) => {
                e.stopPropagation();
                openResolve(row);
              }}
            >
              <CheckCircle fontSize="small" />
            </IconButton>
          </Tooltip>
        ) : (
          <Typography variant="caption" color="text.secondary">
            {formatDate(row.resolved_at)}
          </Typography>
        ),
    },
  ];

  return (
    <>
      <DataTable
        columns={columns}
        rows={findings}
        rowKey={(r) => r.id}
        dense
        searchPlaceholder="Search findings..."
      />

      {/* Resolve Dialog */}
      <Dialog
        open={resolveDialogOpen}
        onClose={() => setResolveDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Resolve Finding</DialogTitle>
        <DialogContent>
          {selectedFinding && (
            <>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {selectedFinding.description}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Recommendation: {selectedFinding.recommendation}
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Management Response / Resolution"
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                sx={{ mt: 2 }}
              />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResolveDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleResolve}
            variant="contained"
            disabled={!resolution.trim()}
            sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
          >
            Resolve
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default FindingsTable;
