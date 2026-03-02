/**
 * DocumentLibrary - Page for managing EUDR compliance documents.
 *
 * Features drag-and-drop upload zone, grid/list view toggle, filter by type
 * and verification status, document cards with metadata, and click-to-view
 * detail with verification results.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  InputAdornment,
  ToggleButton,
  ToggleButtonGroup,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  CircularProgress,
  Alert,
  Snackbar,
  LinearProgress,
  SelectChangeEvent,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import GridViewIcon from '@mui/icons-material/GridView';
import ViewListIcon from '@mui/icons-material/ViewList';
import SearchIcon from '@mui/icons-material/Search';
import DescriptionIcon from '@mui/icons-material/Description';
import VerifiedIcon from '@mui/icons-material/Verified';
import PendingIcon from '@mui/icons-material/Pending';
import CancelIcon from '@mui/icons-material/Cancel';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import apiClient from '../services/api';
import type { Document, DocumentType, DocumentVerificationResult } from '../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DOC_TYPE_LABELS: Record<DocumentType, string> = {
  certificate_of_origin: 'Certificate of Origin',
  phytosanitary_certificate: 'Phytosanitary Certificate',
  bill_of_lading: 'Bill of Lading',
  customs_declaration: 'Customs Declaration',
  sustainability_certificate: 'Sustainability Certificate',
  land_title: 'Land Title',
  satellite_imagery: 'Satellite Imagery',
  audit_report: 'Audit Report',
  supplier_declaration: 'Supplier Declaration',
  gps_coordinates: 'GPS Coordinates',
  other: 'Other',
};

const VERIFICATION_COLORS: Record<string, 'success' | 'warning' | 'error' | 'default'> = {
  verified: 'success',
  pending: 'warning',
  rejected: 'error',
  expired: 'default',
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DocumentLibrary: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Filters
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');

  // Upload
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Detail dialog
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [verification, setVerification] = useState<DocumentVerificationResult | null>(null);

  // Snackbar
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false, message: '', severity: 'success',
  });

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiClient.getDocuments({
        search: search || undefined,
        document_type: (typeFilter as DocumentType) || undefined,
        verification_status: statusFilter || undefined,
        per_page: 100,
      });
      setDocuments(result.items);
    } catch {
      setError('Failed to load documents.');
    } finally {
      setLoading(false);
    }
  }, [search, typeFilter, statusFilter]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // File upload handler
  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', file.name);
        formData.append('document_type', 'other');
        await apiClient.uploadDocument(formData);
      }
      setSnackbar({ open: true, message: `${files.length} document(s) uploaded.`, severity: 'success' });
      fetchDocuments();
    } catch {
      setSnackbar({ open: true, message: 'Upload failed.', severity: 'error' });
    } finally {
      setUploading(false);
    }
  };

  // Drag-and-drop handlers
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFileUpload(e.dataTransfer.files);
  };

  // Verify document
  const handleVerify = async (doc: Document) => {
    try {
      const result = await apiClient.verifyDocument(doc.id);
      setVerification(result);
      setSelectedDoc(doc);
      setSnackbar({ open: true, message: 'Verification complete.', severity: 'success' });
    } catch {
      setSnackbar({ open: true, message: 'Verification failed.', severity: 'error' });
    }
  };

  // Delete document
  const handleDelete = async (doc: Document) => {
    if (!window.confirm(`Delete document "${doc.name}"?`)) return;
    try {
      await apiClient.deleteDocument(doc.id);
      setSnackbar({ open: true, message: 'Document deleted.', severity: 'success' });
      fetchDocuments();
    } catch {
      setSnackbar({ open: true, message: 'Delete failed.', severity: 'error' });
    }
  };

  const verificationIcon = (status: string) => {
    switch (status) {
      case 'verified': return <VerifiedIcon color="success" fontSize="small" />;
      case 'pending': return <PendingIcon color="warning" fontSize="small" />;
      case 'rejected': return <CancelIcon color="error" fontSize="small" />;
      case 'expired': return <WarningAmberIcon color="disabled" fontSize="small" />;
      default: return <PendingIcon fontSize="small" />;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h4" fontWeight={700}>Document Library</Typography>
        <Button
          variant="contained"
          startIcon={<UploadFileIcon />}
          onClick={() => fileInputRef.current?.click()}
        >
          Upload Document
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => handleFileUpload(e.target.files)}
        />
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Upload Zone */}
      <Paper
        variant="outlined"
        sx={{
          p: 4,
          mb: 3,
          textAlign: 'center',
          borderStyle: 'dashed',
          borderColor: dragOver ? 'primary.main' : 'grey.400',
          backgroundColor: dragOver ? 'primary.50' : 'grey.50',
          cursor: 'pointer',
          transition: 'all 0.2s',
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <CloudUploadIcon sx={{ fontSize: 48, color: dragOver ? 'primary.main' : 'grey.400', mb: 1 }} />
        <Typography variant="body1" color={dragOver ? 'primary.main' : 'text.secondary'}>
          {uploading ? 'Uploading...' : 'Drag and drop files here, or click to browse'}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Supported: PDF, JPEG, PNG, TIFF, XLSX, CSV, XML
        </Typography>
        {uploading && <LinearProgress sx={{ mt: 1 }} />}
      </Paper>

      {/* Filter Bar */}
      <Stack direction="row" spacing={1.5} mb={2} alignItems="center" flexWrap="wrap" useFlexGap>
        <TextField
          size="small"
          placeholder="Search documents..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          sx={{ minWidth: 200, flex: 1 }}
          InputProps={{
            startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment>,
          }}
        />
        <FormControl size="small" sx={{ minWidth: 180 }}>
          <InputLabel>Type</InputLabel>
          <Select value={typeFilter} label="Type" onChange={(e: SelectChangeEvent) => setTypeFilter(e.target.value)}>
            <MenuItem value="">All</MenuItem>
            {Object.entries(DOC_TYPE_LABELS).map(([val, label]) => (
              <MenuItem key={val} value={val}>{label}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Status</InputLabel>
          <Select value={statusFilter} label="Status" onChange={(e: SelectChangeEvent) => setStatusFilter(e.target.value)}>
            <MenuItem value="">All</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="verified">Verified</MenuItem>
            <MenuItem value="rejected">Rejected</MenuItem>
            <MenuItem value="expired">Expired</MenuItem>
          </Select>
        </FormControl>
        <ToggleButtonGroup
          size="small"
          value={viewMode}
          exclusive
          onChange={(_, v) => v && setViewMode(v)}
        >
          <ToggleButton value="grid"><GridViewIcon fontSize="small" /></ToggleButton>
          <ToggleButton value="list"><ViewListIcon fontSize="small" /></ToggleButton>
        </ToggleButtonGroup>
      </Stack>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}><CircularProgress /></Box>
      ) : documents.length === 0 ? (
        <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
          No documents found. Upload documents or adjust your filters.
        </Typography>
      ) : viewMode === 'grid' ? (
        /* Grid View */
        <Grid container spacing={2}>
          {documents.map((doc) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={doc.id}>
              <Card
                variant="outlined"
                sx={{ cursor: 'pointer', '&:hover': { borderColor: 'primary.main' } }}
                onClick={() => { setSelectedDoc(doc); setVerification(null); }}
              >
                <CardContent sx={{ pb: 1 }}>
                  <Stack direction="row" alignItems="flex-start" spacing={1}>
                    <DescriptionIcon color="action" />
                    <Box sx={{ overflow: 'hidden', flex: 1 }}>
                      <Typography variant="body2" fontWeight={600} noWrap>{doc.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatFileSize(doc.file_size_bytes)} | {formatDate(doc.uploaded_at)}
                      </Typography>
                    </Box>
                  </Stack>
                  <Stack direction="row" spacing={0.5} mt={1}>
                    <Chip
                      label={DOC_TYPE_LABELS[doc.document_type] ?? doc.document_type}
                      size="small"
                      sx={{ fontSize: 10 }}
                    />
                    <Chip
                      icon={verificationIcon(doc.verification_status) as React.ReactElement}
                      label={doc.verification_status}
                      size="small"
                      color={VERIFICATION_COLORS[doc.verification_status]}
                      sx={{ textTransform: 'capitalize', fontSize: 10 }}
                    />
                  </Stack>
                  {doc.supplier_name && (
                    <Typography variant="caption" color="text.secondary" display="block" mt={0.5}>
                      Supplier: {doc.supplier_name}
                    </Typography>
                  )}
                </CardContent>
                <CardActions sx={{ pt: 0, justifyContent: 'flex-end' }}>
                  <Tooltip title="Verify">
                    <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleVerify(doc); }}>
                      <CheckCircleIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete">
                    <IconButton size="small" color="error" onClick={(e) => { e.stopPropagation(); handleDelete(doc); }}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        /* List View */
        <Paper variant="outlined">
          <List disablePadding>
            {documents.map((doc, idx) => (
              <ListItem
                key={doc.id}
                divider={idx < documents.length - 1}
                button
                onClick={() => { setSelectedDoc(doc); setVerification(null); }}
                secondaryAction={
                  <Stack direction="row" spacing={0.5}>
                    <Tooltip title="Verify">
                      <IconButton size="small" onClick={(e) => { e.stopPropagation(); handleVerify(doc); }}>
                        <CheckCircleIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton size="small" color="error" onClick={(e) => { e.stopPropagation(); handleDelete(doc); }}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                }
              >
                <ListItemIcon><DescriptionIcon /></ListItemIcon>
                <ListItemText
                  primary={doc.name}
                  secondary={`${DOC_TYPE_LABELS[doc.document_type] ?? doc.document_type} | ${formatFileSize(doc.file_size_bytes)} | ${formatDate(doc.uploaded_at)}`}
                />
                <Chip
                  label={doc.verification_status}
                  size="small"
                  color={VERIFICATION_COLORS[doc.verification_status]}
                  sx={{ textTransform: 'capitalize', mr: 8 }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Detail Dialog */}
      <Dialog
        open={Boolean(selectedDoc)}
        onClose={() => { setSelectedDoc(null); setVerification(null); }}
        maxWidth="sm"
        fullWidth
      >
        {selectedDoc && (
          <>
            <DialogTitle>{selectedDoc.name}</DialogTitle>
            <DialogContent>
              <Stack spacing={1}>
                <Typography variant="body2"><strong>Type:</strong> {DOC_TYPE_LABELS[selectedDoc.document_type]}</Typography>
                <Typography variant="body2"><strong>File:</strong> {selectedDoc.file_name}</Typography>
                <Typography variant="body2"><strong>Size:</strong> {formatFileSize(selectedDoc.file_size_bytes)}</Typography>
                <Typography variant="body2"><strong>Uploaded:</strong> {formatDate(selectedDoc.uploaded_at)} by {selectedDoc.uploaded_by}</Typography>
                <Typography variant="body2"><strong>Verification:</strong> {selectedDoc.verification_status}</Typography>
                {selectedDoc.verification_notes && (
                  <Typography variant="body2"><strong>Notes:</strong> {selectedDoc.verification_notes}</Typography>
                )}
                {selectedDoc.supplier_name && (
                  <Typography variant="body2"><strong>Supplier:</strong> {selectedDoc.supplier_name}</Typography>
                )}
                {selectedDoc.expiry_date && (
                  <Typography variant="body2"><strong>Expires:</strong> {formatDate(selectedDoc.expiry_date)}</Typography>
                )}
              </Stack>
              {verification && (
                <Box mt={2}>
                  <Divider sx={{ mb: 1 }} />
                  <Alert severity={verification.is_verified ? 'success' : 'warning'} sx={{ mb: 1 }}>
                    {verification.is_verified
                      ? `Document verified with ${(verification.confidence_score * 100).toFixed(0)}% confidence.`
                      : 'Document verification found issues.'}
                  </Alert>
                  {verification.issues.length > 0 && (
                    <List dense>
                      {verification.issues.map((issue, idx) => (
                        <ListItem key={idx} disableGutters>
                          <ListItemIcon sx={{ minWidth: 28 }}>
                            {issue.severity === 'error' ? <CancelIcon color="error" fontSize="small" /> : <WarningAmberIcon color="warning" fontSize="small" />}
                          </ListItemIcon>
                          <ListItemText primary={issue.message} secondary={issue.code} />
                        </ListItem>
                      ))}
                    </List>
                  )}
                </Box>
              )}
            </DialogContent>
          </>
        )}
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setSnackbar((s) => ({ ...s, open: false }))} severity={snackbar.severity} variant="filled">
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DocumentLibrary;
