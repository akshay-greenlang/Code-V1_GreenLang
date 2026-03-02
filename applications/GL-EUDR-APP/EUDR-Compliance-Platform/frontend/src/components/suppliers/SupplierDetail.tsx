/**
 * SupplierDetail - Supplier detail panel with five tabs.
 *
 * Tabs: Overview, Plots, Documents, DDS, History.
 * Includes action buttons for Edit, Generate DDS, Run Pipeline, and Delete.
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Grid,
  Chip,
  Stack,
  Divider,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper,
  Avatar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Timeline,
  IconButton,
  Tooltip,
  Alert,
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import DescriptionIcon from '@mui/icons-material/Description';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DeleteIcon from '@mui/icons-material/Delete';
import BusinessIcon from '@mui/icons-material/Business';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import EmailIcon from '@mui/icons-material/Email';
import PhoneIcon from '@mui/icons-material/Phone';
import BadgeIcon from '@mui/icons-material/Badge';
import MapIcon from '@mui/icons-material/Map';
import VerifiedIcon from '@mui/icons-material/Verified';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import HistoryIcon from '@mui/icons-material/History';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import PendingIcon from '@mui/icons-material/Pending';
import type {
  Supplier,
  Plot,
  Document,
  DueDiligenceStatement,
  RiskLevel,
  ComplianceStatus,
} from '../../types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface SupplierDetailProps {
  supplier: Supplier;
  plots?: Plot[];
  documents?: Document[];
  ddsList?: DueDiligenceStatement[];
  complianceHistory?: ComplianceHistoryEntry[];
  onEdit: (supplier: Supplier) => void;
  onGenerateDDS: (supplier: Supplier) => void;
  onRunPipeline: (supplier: Supplier) => void;
  onDelete: (supplier: Supplier) => void;
}

export interface ComplianceHistoryEntry {
  date: string;
  event: string;
  status: ComplianceStatus;
  details: string;
  user: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RISK_COLORS: Record<RiskLevel, string> = {
  low: '#4caf50',
  standard: '#2196f3',
  high: '#ff9800',
  critical: '#f44336',
};

const COMPLIANCE_COLORS: Record<ComplianceStatus, string> = {
  compliant: '#4caf50',
  non_compliant: '#f44336',
  pending: '#ff9800',
  under_review: '#2196f3',
  expired: '#9e9e9e',
};

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

function complianceIcon(status: ComplianceStatus) {
  switch (status) {
    case 'compliant': return <CheckCircleIcon color="success" fontSize="small" />;
    case 'non_compliant': return <CancelIcon color="error" fontSize="small" />;
    case 'pending': return <PendingIcon color="warning" fontSize="small" />;
    case 'under_review': return <PendingIcon color="info" fontSize="small" />;
    case 'expired': return <WarningAmberIcon color="disabled" fontSize="small" />;
    default: return <PendingIcon fontSize="small" />;
  }
}

// ---------------------------------------------------------------------------
// Tab Panel
// ---------------------------------------------------------------------------

function TabPanel({
  children,
  value,
  index,
}: {
  children: React.ReactNode;
  value: number;
  index: number;
}) {
  return (
    <Box role="tabpanel" hidden={value !== index} sx={{ pt: 2 }}>
      {value === index && children}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const SupplierDetail: React.FC<SupplierDetailProps> = ({
  supplier,
  plots = [],
  documents = [],
  ddsList = [],
  complianceHistory = [],
  onEdit,
  onGenerateDDS,
  onRunPipeline,
  onDelete,
}) => {
  const [tab, setTab] = useState(0);

  return (
    <Box>
      {/* Header + Actions */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
        <Stack direction="row" alignItems="center" spacing={2}>
          <Avatar sx={{ bgcolor: 'primary.main', width: 48, height: 48 }}>
            <BusinessIcon />
          </Avatar>
          <Box>
            <Typography variant="h5">{supplier.name}</Typography>
            <Stack direction="row" spacing={1} mt={0.5}>
              <Chip
                size="small"
                label={supplier.risk_level.replace('_', ' ')}
                sx={{
                  backgroundColor: RISK_COLORS[supplier.risk_level],
                  color: '#fff',
                  textTransform: 'capitalize',
                  fontWeight: 600,
                }}
              />
              <Chip
                size="small"
                label={supplier.compliance_status.replace('_', ' ')}
                sx={{
                  backgroundColor: COMPLIANCE_COLORS[supplier.compliance_status],
                  color: '#fff',
                  textTransform: 'capitalize',
                  fontWeight: 600,
                }}
              />
            </Stack>
          </Box>
        </Stack>

        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<EditIcon />} onClick={() => onEdit(supplier)}>
            Edit
          </Button>
          <Button variant="outlined" startIcon={<DescriptionIcon />} onClick={() => onGenerateDDS(supplier)}>
            Generate DDS
          </Button>
          <Button variant="outlined" color="secondary" startIcon={<PlayArrowIcon />} onClick={() => onRunPipeline(supplier)}>
            Run Pipeline
          </Button>
          <Button variant="outlined" color="error" startIcon={<DeleteIcon />} onClick={() => onDelete(supplier)}>
            Delete
          </Button>
        </Stack>
      </Stack>

      {/* Tabs */}
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tab label="Overview" />
        <Tab label={`Plots (${plots.length})`} />
        <Tab label={`Documents (${documents.length})`} />
        <Tab label={`DDS (${ddsList.length})`} />
        <Tab label="History" />
      </Tabs>

      {/* ---- OVERVIEW TAB ---- */}
      <TabPanel value={tab} index={0}>
        <Grid container spacing={3}>
          {/* Profile Info */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                  Profile Information
                </Typography>
                <List dense disablePadding>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 36 }}><BusinessIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Company" secondary={supplier.name} />
                  </ListItem>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 36 }}><LocationOnIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Location" secondary={`${supplier.region}, ${supplier.country}`} />
                  </ListItem>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 36 }}><BadgeIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Tax ID" secondary={supplier.tax_id || '-'} />
                  </ListItem>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 36 }}><LocationOnIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Address" secondary={supplier.address || '-'} />
                  </ListItem>
                  <ListItem disableGutters>
                    <ListItemIcon sx={{ minWidth: 36 }}><EmailIcon fontSize="small" /></ListItemIcon>
                    <ListItemText primary="Contact" secondary={`${supplier.contact_name} (${supplier.contact_email})`} />
                  </ListItem>
                  {supplier.contact_phone && (
                    <ListItem disableGutters>
                      <ListItemIcon sx={{ minWidth: 36 }}><PhoneIcon fontSize="small" /></ListItemIcon>
                      <ListItemText primary="Phone" secondary={supplier.contact_phone} />
                    </ListItem>
                  )}
                </List>
                <Divider sx={{ my: 1 }} />
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Commodities
                </Typography>
                <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                  {supplier.commodities.map((c) => (
                    <Chip key={c} label={c.replace('_', ' ')} size="small" color="primary" variant="outlined" sx={{ textTransform: 'capitalize' }} />
                  ))}
                </Stack>
                {supplier.certifications.length > 0 && (
                  <>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Certifications
                    </Typography>
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                      {supplier.certifications.map((cert, i) => (
                        <Chip key={i} icon={<VerifiedIcon />} label={cert} size="small" variant="outlined" />
                      ))}
                    </Stack>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Status Cards */}
          <Grid item xs={12} md={6}>
            <Stack spacing={2}>
              {/* Compliance Status Card */}
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                    Compliance Status
                  </Typography>
                  <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                    {complianceIcon(supplier.compliance_status)}
                    <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                      {supplier.compliance_status.replace('_', ' ')}
                    </Typography>
                  </Stack>
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Last Audit</Typography>
                      <Typography variant="body2">{formatDate(supplier.last_audit_date)}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Next Audit</Typography>
                      <Typography variant="body2">{formatDate(supplier.next_audit_date)}</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>

              {/* Risk Summary Card */}
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                    Risk Summary
                  </Typography>
                  <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                    <Chip
                      label={supplier.risk_level.replace('_', ' ')}
                      sx={{
                        backgroundColor: RISK_COLORS[supplier.risk_level],
                        color: '#fff',
                        textTransform: 'capitalize',
                        fontWeight: 600,
                      }}
                    />
                  </Stack>
                  <Grid container spacing={1}>
                    <Grid item xs={4}>
                      <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="h5">{supplier.total_plots}</Typography>
                        <Typography variant="caption" color="text.secondary">Plots</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={4}>
                      <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="h5">{supplier.active_dds_count}</Typography>
                        <Typography variant="caption" color="text.secondary">Active DDS</Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={4}>
                      <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="h5">{supplier.commodities.length}</Typography>
                        <Typography variant="caption" color="text.secondary">Commodities</Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>

              {supplier.notes && (
                <Alert severity="info" variant="outlined">
                  <Typography variant="body2">{supplier.notes}</Typography>
                </Alert>
              )}
            </Stack>
          </Grid>
        </Grid>
      </TabPanel>

      {/* ---- PLOTS TAB ---- */}
      <TabPanel value={tab} index={1}>
        {plots.length === 0 ? (
          <Typography color="text.secondary">No plots registered for this supplier.</Typography>
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Country</TableCell>
                  <TableCell>Commodity</TableCell>
                  <TableCell align="right">Area (ha)</TableCell>
                  <TableCell>Risk</TableCell>
                  <TableCell>Deforestation Free</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {plots.map((plot) => (
                  <TableRow key={plot.id} hover>
                    <TableCell>{plot.name}</TableCell>
                    <TableCell>{plot.country}</TableCell>
                    <TableCell sx={{ textTransform: 'capitalize' }}>{plot.commodity.replace('_', ' ')}</TableCell>
                    <TableCell align="right">{plot.area_hectares.toFixed(1)}</TableCell>
                    <TableCell>
                      <Chip
                        label={plot.risk_level}
                        size="small"
                        sx={{
                          backgroundColor: RISK_COLORS[plot.risk_level],
                          color: '#fff',
                          textTransform: 'capitalize',
                          fontWeight: 600,
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      {plot.deforestation_free === null
                        ? <Chip label="Pending" size="small" color="default" />
                        : plot.deforestation_free
                        ? <Chip label="Yes" size="small" color="success" />
                        : <Chip label="No" size="small" color="error" />}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </TabPanel>

      {/* ---- DOCUMENTS TAB ---- */}
      <TabPanel value={tab} index={2}>
        {documents.length === 0 ? (
          <Typography color="text.secondary">No documents linked to this supplier.</Typography>
        ) : (
          <List>
            {documents.map((doc) => (
              <ListItem key={doc.id} divider>
                <ListItemIcon>
                  <DescriptionIcon />
                </ListItemIcon>
                <ListItemText
                  primary={doc.name}
                  secondary={`${doc.document_type.replace('_', ' ')} | ${formatDate(doc.uploaded_at)}`}
                />
                <Chip
                  label={doc.verification_status}
                  size="small"
                  color={
                    doc.verification_status === 'verified' ? 'success'
                    : doc.verification_status === 'rejected' ? 'error'
                    : doc.verification_status === 'expired' ? 'default'
                    : 'warning'
                  }
                  sx={{ textTransform: 'capitalize' }}
                />
              </ListItem>
            ))}
          </List>
        )}
      </TabPanel>

      {/* ---- DDS TAB ---- */}
      <TabPanel value={tab} index={3}>
        {ddsList.length === 0 ? (
          <Typography color="text.secondary">No Due Diligence Statements for this supplier.</Typography>
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Reference</TableCell>
                  <TableCell>Commodity</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Risk</TableCell>
                  <TableCell>Generated</TableCell>
                  <TableCell>Submitted</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {ddsList.map((dds) => (
                  <TableRow key={dds.id} hover>
                    <TableCell>{dds.reference_number}</TableCell>
                    <TableCell sx={{ textTransform: 'capitalize' }}>{dds.commodity.replace('_', ' ')}</TableCell>
                    <TableCell>
                      <Chip
                        label={dds.status.replace('_', ' ')}
                        size="small"
                        sx={{ textTransform: 'capitalize' }}
                        color={
                          dds.status === 'accepted' ? 'success'
                          : dds.status === 'rejected' ? 'error'
                          : dds.status === 'submitted' ? 'warning'
                          : dds.status === 'validated' ? 'info'
                          : 'default'
                        }
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={dds.risk_level}
                        size="small"
                        sx={{
                          backgroundColor: RISK_COLORS[dds.risk_level],
                          color: '#fff',
                          textTransform: 'capitalize',
                        }}
                      />
                    </TableCell>
                    <TableCell>{formatDate(dds.generated_at)}</TableCell>
                    <TableCell>{formatDate(dds.submitted_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </TabPanel>

      {/* ---- HISTORY TAB ---- */}
      <TabPanel value={tab} index={4}>
        {complianceHistory.length === 0 ? (
          <Typography color="text.secondary">No compliance history available.</Typography>
        ) : (
          <List>
            {complianceHistory.map((entry, idx) => (
              <ListItem key={idx} divider sx={{ alignItems: 'flex-start' }}>
                <ListItemIcon sx={{ mt: 0.5 }}>
                  {complianceIcon(entry.status)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography variant="body2" fontWeight={600}>{entry.event}</Typography>
                      <Chip
                        label={entry.status.replace('_', ' ')}
                        size="small"
                        sx={{
                          backgroundColor: COMPLIANCE_COLORS[entry.status],
                          color: '#fff',
                          textTransform: 'capitalize',
                          fontSize: 11,
                        }}
                      />
                    </Stack>
                  }
                  secondary={
                    <>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(entry.date)} by {entry.user}
                      </Typography>
                      {entry.details && (
                        <Typography variant="body2" sx={{ mt: 0.5 }}>
                          {entry.details}
                        </Typography>
                      )}
                    </>
                  }
                />
              </ListItem>
            ))}
          </List>
        )}
      </TabPanel>
    </Box>
  );
};

export default SupplierDetail;
