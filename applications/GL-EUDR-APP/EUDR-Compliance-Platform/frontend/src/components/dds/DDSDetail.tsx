/**
 * DDSDetail - Due Diligence Statement detail view.
 *
 * Renders all 7 EUDR sections: Operator Information, Product Description,
 * Country of Production, Geolocation Data, Risk Assessment, Risk Mitigation,
 * and Conclusion. Includes a status banner, action bar, and status history.
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Stack,
  Divider,
  Button,
  Paper,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableRow,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import EditIcon from '@mui/icons-material/Edit';
import BusinessIcon from '@mui/icons-material/Business';
import CategoryIcon from '@mui/icons-material/Category';
import PublicIcon from '@mui/icons-material/Public';
import MapIcon from '@mui/icons-material/Map';
import SecurityIcon from '@mui/icons-material/Security';
import ShieldIcon from '@mui/icons-material/Shield';
import GavelIcon from '@mui/icons-material/Gavel';
import HistoryIcon from '@mui/icons-material/History';
import type {
  DueDiligenceStatement,
  DDSStatus,
  RiskLevel,
} from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<DDSStatus, { bg: string; color: string; severity: 'success' | 'info' | 'warning' | 'error' }> = {
  draft: { bg: '#e0e0e0', color: '#424242', severity: 'info' },
  pending_review: { bg: '#bbdefb', color: '#1565c0', severity: 'info' },
  validated: { bg: '#b2dfdb', color: '#00695c', severity: 'success' },
  submitted: { bg: '#ffe0b2', color: '#e65100', severity: 'warning' },
  accepted: { bg: '#c8e6c9', color: '#2e7d32', severity: 'success' },
  rejected: { bg: '#ffcdd2', color: '#c62828', severity: 'error' },
  amended: { bg: '#e1bee7', color: '#6a1b9a', severity: 'info' },
};

const RISK_COLORS: Record<RiskLevel, string> = {
  low: '#4caf50',
  standard: '#2196f3',
  high: '#ff9800',
  critical: '#f44336',
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface DDSStatusHistoryEntry {
  date: string;
  from_status: DDSStatus;
  to_status: DDSStatus;
  user: string;
  notes: string;
}

interface DDSDetailProps {
  dds: DueDiligenceStatement;
  plots?: Array<{ id: string; name: string; country: string; area_hectares: number }>;
  statusHistory?: DDSStatusHistoryEntry[];
  onValidate: (dds: DueDiligenceStatement) => void;
  onSubmit: (dds: DueDiligenceStatement) => void;
  onDownload: (dds: DueDiligenceStatement, format: 'pdf' | 'xml' | 'json') => void;
  onAmend: (dds: DueDiligenceStatement) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(d: string | null): string {
  if (!d) return '-';
  return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

function formatDateTime(d: string | null): string {
  if (!d) return '-';
  return new Date(d).toLocaleString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

// ---------------------------------------------------------------------------
// Section component
// ---------------------------------------------------------------------------

function Section({
  number,
  title,
  icon,
  children,
  defaultExpanded = true,
}: {
  number: number;
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultExpanded?: boolean;
}) {
  return (
    <Accordion defaultExpanded={defaultExpanded} variant="outlined" sx={{ mb: 1 }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Stack direction="row" alignItems="center" spacing={1.5}>
          <Chip label={number} size="small" color="primary" sx={{ fontWeight: 700, minWidth: 28 }} />
          {icon}
          <Typography variant="subtitle1" fontWeight={600}>{title}</Typography>
        </Stack>
      </AccordionSummary>
      <AccordionDetails>{children}</AccordionDetails>
    </Accordion>
  );
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <TableRow>
      <TableCell sx={{ fontWeight: 500, color: 'text.secondary', width: '35%', borderBottom: 'none', py: 0.75 }}>
        {label}
      </TableCell>
      <TableCell sx={{ borderBottom: 'none', py: 0.75 }}>{value}</TableCell>
    </TableRow>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DDSDetail: React.FC<DDSDetailProps> = ({
  dds,
  plots = [],
  statusHistory = [],
  onValidate,
  onSubmit,
  onDownload,
  onAmend,
}) => {
  const statusStyle = STATUS_COLORS[dds.status] ?? STATUS_COLORS.draft;

  return (
    <Box>
      {/* Status Banner */}
      <Alert
        severity={statusStyle.severity}
        sx={{ mb: 2 }}
        action={
          <Chip
            label={dds.status.replace('_', ' ')}
            sx={{
              backgroundColor: statusStyle.bg,
              color: statusStyle.color,
              fontWeight: 700,
              textTransform: 'capitalize',
            }}
          />
        }
      >
        <Typography variant="subtitle1" fontWeight={600}>
          Due Diligence Statement: {dds.reference_number}
        </Typography>
        <Typography variant="body2">
          Generated {formatDateTime(dds.generated_at)}
          {dds.submitted_at && ` | Submitted ${formatDateTime(dds.submitted_at)}`}
        </Typography>
      </Alert>

      {/* Action Bar */}
      <Stack direction="row" spacing={1} mb={3} flexWrap="wrap" useFlexGap>
        {(dds.status === 'draft' || dds.status === 'pending_review') && (
          <Button variant="contained" startIcon={<CheckCircleIcon />} onClick={() => onValidate(dds)}>
            Validate
          </Button>
        )}
        {dds.status === 'validated' && (
          <Button variant="contained" color="warning" startIcon={<SendIcon />} onClick={() => onSubmit(dds)}>
            Submit to EU Authority
          </Button>
        )}
        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={() => onDownload(dds, 'pdf')}>
          PDF
        </Button>
        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={() => onDownload(dds, 'xml')}>
          XML
        </Button>
        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={() => onDownload(dds, 'json')}>
          JSON
        </Button>
        {(dds.status === 'rejected' || dds.status === 'submitted') && (
          <Button variant="outlined" color="secondary" startIcon={<EditIcon />} onClick={() => onAmend(dds)}>
            Amend
          </Button>
        )}
      </Stack>

      {/* Section 1: Operator Information */}
      <Section number={1} title="Operator Information" icon={<BusinessIcon color="action" />}>
        <Table size="small">
          <TableBody>
            <InfoRow label="Operator Name" value={dds.operator_name} />
            <InfoRow label="Operator Address" value={dds.operator_address} />
            <InfoRow label="EORI Number" value={dds.operator_eori} />
            <InfoRow label="EU Authority" value={dds.eu_authority || '-'} />
          </TableBody>
        </Table>
      </Section>

      {/* Section 2: Product Description */}
      <Section number={2} title="Product Description" icon={<CategoryIcon color="action" />}>
        <Table size="small">
          <TableBody>
            <InfoRow
              label="Commodity"
              value={
                <Chip
                  label={dds.commodity.replace('_', ' ')}
                  color="primary"
                  size="small"
                  sx={{ textTransform: 'capitalize' }}
                />
              }
            />
            <InfoRow label="Total Quantity" value={`${dds.total_quantity_kg.toLocaleString()} kg`} />
            <InfoRow label="Total Area" value={`${dds.total_area_hectares.toFixed(1)} ha`} />
            <InfoRow label="Number of Plots" value={dds.plot_ids.length} />
          </TableBody>
        </Table>
      </Section>

      {/* Section 3: Country of Production */}
      <Section number={3} title="Country of Production" icon={<PublicIcon color="action" />}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Supplier: <strong>{dds.supplier_name}</strong>
        </Typography>
        <Chip
          label={`Risk: ${dds.risk_level}`}
          sx={{
            backgroundColor: RISK_COLORS[dds.risk_level],
            color: '#fff',
            textTransform: 'capitalize',
            fontWeight: 600,
          }}
        />
      </Section>

      {/* Section 4: Geolocation Data */}
      <Section number={4} title="Geolocation Data" icon={<MapIcon color="action" />}>
        {plots.length > 0 ? (
          <Table size="small">
            <TableBody>
              {plots.map((plot) => (
                <TableRow key={plot.id}>
                  <TableCell sx={{ py: 0.75 }}>{plot.name}</TableCell>
                  <TableCell sx={{ py: 0.75 }}>{plot.country}</TableCell>
                  <TableCell sx={{ py: 0.75 }} align="right">{plot.area_hectares.toFixed(1)} ha</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Typography variant="body2" color="text.secondary">
            {dds.plot_ids.length} plot(s) referenced. Load plot details to view geolocation data.
          </Typography>
        )}
      </Section>

      {/* Section 5: Risk Assessment */}
      <Section number={5} title="Risk Assessment" icon={<SecurityIcon color="action" />}>
        <Stack direction="row" alignItems="center" spacing={2} mb={1}>
          <Typography variant="body2" color="text.secondary">Overall Risk:</Typography>
          <Chip
            label={dds.risk_level.replace('_', ' ')}
            sx={{
              backgroundColor: RISK_COLORS[dds.risk_level],
              color: '#fff',
              textTransform: 'capitalize',
              fontWeight: 700,
            }}
          />
        </Stack>
        <Typography variant="body2" color="text.secondary">
          Risk assessment considers country-level deforestation risk, satellite imagery analysis,
          supplier compliance history, and document verification status.
        </Typography>
      </Section>

      {/* Section 6: Risk Mitigation */}
      <Section number={6} title="Risk Mitigation" icon={<ShieldIcon color="action" />}>
        {dds.risk_mitigation_measures.length > 0 ? (
          <List dense disablePadding>
            {dds.risk_mitigation_measures.map((measure, idx) => (
              <ListItem key={idx} disableGutters>
                <ListItemIcon sx={{ minWidth: 28 }}>
                  <CheckCircleIcon fontSize="small" color="success" />
                </ListItemIcon>
                <ListItemText primary={measure} />
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No risk mitigation measures recorded.
          </Typography>
        )}
      </Section>

      {/* Section 7: Conclusion */}
      <Section number={7} title="Conclusion" icon={<GavelIcon color="action" />}>
        <Alert
          severity={dds.risk_level === 'low' || dds.risk_level === 'standard' ? 'success' : 'warning'}
          variant="outlined"
        >
          <Typography variant="body2">
            {dds.risk_level === 'low' || dds.risk_level === 'standard'
              ? 'Based on the risk assessment and mitigation measures, the products covered by this statement are assessed as compliant with EU Regulation 2023/1115 (EUDR). No significant deforestation risk has been identified.'
              : 'Elevated risk has been identified. Additional due diligence measures are recommended before submission. Review risk mitigation measures and ensure all documentation is complete.'}
          </Typography>
        </Alert>
        <Stack direction="row" spacing={2} mt={1.5}>
          <Paper variant="outlined" sx={{ p: 1, flex: 1, textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Expiry Date</Typography>
            <Typography variant="body2" fontWeight={600}>{formatDate(dds.expiry_date)}</Typography>
          </Paper>
          <Paper variant="outlined" sx={{ p: 1, flex: 1, textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">Validated At</Typography>
            <Typography variant="body2" fontWeight={600}>{formatDateTime(dds.validated_at)}</Typography>
          </Paper>
        </Stack>
      </Section>

      {/* Status History Timeline */}
      {statusHistory.length > 0 && (
        <Card variant="outlined" sx={{ mt: 2 }}>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} mb={1.5}>
              <HistoryIcon color="action" />
              <Typography variant="subtitle1" fontWeight={600}>Status History</Typography>
            </Stack>
            <List dense disablePadding>
              {statusHistory.map((entry, idx) => {
                const toStyle = STATUS_COLORS[entry.to_status] ?? STATUS_COLORS.draft;
                return (
                  <ListItem key={idx} divider={idx < statusHistory.length - 1}>
                    <ListItemText
                      primary={
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Chip
                            label={entry.from_status.replace('_', ' ')}
                            size="small"
                            variant="outlined"
                            sx={{ textTransform: 'capitalize', fontSize: 11 }}
                          />
                          <Typography variant="body2">→</Typography>
                          <Chip
                            label={entry.to_status.replace('_', ' ')}
                            size="small"
                            sx={{
                              backgroundColor: toStyle.bg,
                              color: toStyle.color,
                              textTransform: 'capitalize',
                              fontWeight: 600,
                              fontSize: 11,
                            }}
                          />
                        </Stack>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {formatDateTime(entry.date)} by {entry.user}
                          {entry.notes && ` - ${entry.notes}`}
                        </Typography>
                      }
                    />
                  </ListItem>
                );
              })}
            </List>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default DDSDetail;
