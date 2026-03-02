/**
 * DDSTable - MUI table listing Due Diligence Statements.
 *
 * Displays DDS records with status badges, filters (status, commodity, year,
 * supplier search), and action buttons (View, Validate, Submit, Download, Amend).
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Button,
  Stack,
  InputAdornment,
  IconButton,
  Tooltip,
  Typography,
  Menu,
  SelectChangeEvent,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import EditIcon from '@mui/icons-material/Edit';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import ClearIcon from '@mui/icons-material/Clear';
import type { DueDiligenceStatement, DDSStatus, EUDRCommodity } from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<DDSStatus, { bg: string; color: string }> = {
  draft: { bg: '#e0e0e0', color: '#424242' },
  pending_review: { bg: '#bbdefb', color: '#1565c0' },
  validated: { bg: '#b2dfdb', color: '#00695c' },
  submitted: { bg: '#ffe0b2', color: '#e65100' },
  accepted: { bg: '#c8e6c9', color: '#2e7d32' },
  rejected: { bg: '#ffcdd2', color: '#c62828' },
  amended: { bg: '#e1bee7', color: '#6a1b9a' },
};

const STATUS_OPTIONS: { value: DDSStatus; label: string }[] = [
  { value: 'draft' as DDSStatus, label: 'Draft' },
  { value: 'pending_review' as DDSStatus, label: 'Pending Review' },
  { value: 'validated' as DDSStatus, label: 'Validated' },
  { value: 'submitted' as DDSStatus, label: 'Submitted' },
  { value: 'accepted' as DDSStatus, label: 'Accepted' },
  { value: 'rejected' as DDSStatus, label: 'Rejected' },
  { value: 'amended' as DDSStatus, label: 'Amended' },
];

const COMMODITY_OPTIONS: { value: EUDRCommodity; label: string }[] = [
  { value: 'cattle' as EUDRCommodity, label: 'Cattle' },
  { value: 'cocoa' as EUDRCommodity, label: 'Cocoa' },
  { value: 'coffee' as EUDRCommodity, label: 'Coffee' },
  { value: 'oil_palm' as EUDRCommodity, label: 'Oil Palm' },
  { value: 'rubber' as EUDRCommodity, label: 'Rubber' },
  { value: 'soya' as EUDRCommodity, label: 'Soya' },
  { value: 'wood' as EUDRCommodity, label: 'Wood' },
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DDSTableProps {
  ddsList: DueDiligenceStatement[];
  totalCount: number;
  page: number;
  rowsPerPage: number;
  loading?: boolean;
  onPageChange: (page: number) => void;
  onRowsPerPageChange: (perPage: number) => void;
  onView: (dds: DueDiligenceStatement) => void;
  onValidate: (dds: DueDiligenceStatement) => void;
  onSubmit: (dds: DueDiligenceStatement) => void;
  onDownload: (dds: DueDiligenceStatement, format: 'pdf' | 'xml' | 'json') => void;
  onAmend: (dds: DueDiligenceStatement) => void;
  onFilterChange: (filters: DDSFilters) => void;
}

export interface DDSFilters {
  search: string;
  status: string;
  commodity: string;
  year: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DDSTable: React.FC<DDSTableProps> = ({
  ddsList,
  totalCount,
  page,
  rowsPerPage,
  loading = false,
  onPageChange,
  onRowsPerPageChange,
  onView,
  onValidate,
  onSubmit,
  onDownload,
  onAmend,
  onFilterChange,
}) => {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [commodityFilter, setCommodityFilter] = useState('');
  const [yearFilter, setYearFilter] = useState('');

  // Action menu
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [menuDDS, setMenuDDS] = useState<DueDiligenceStatement | null>(null);

  const emitFilters = useCallback(
    (overrides: Partial<DDSFilters> = {}) => {
      onFilterChange({
        search,
        status: statusFilter,
        commodity: commodityFilter,
        year: yearFilter,
        ...overrides,
      });
    },
    [search, statusFilter, commodityFilter, yearFilter, onFilterChange]
  );

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearch(val);
    emitFilters({ search: val });
  };

  const handleSelectChange =
    (setter: (v: string) => void, field: keyof DDSFilters) =>
    (e: SelectChangeEvent<string>) => {
      const val = e.target.value;
      setter(val);
      emitFilters({ [field]: val });
    };

  const handleClearFilters = () => {
    setSearch('');
    setStatusFilter('');
    setCommodityFilter('');
    setYearFilter('');
    emitFilters({ search: '', status: '', commodity: '', year: '' });
  };

  const handleMenuOpen = (e: React.MouseEvent<HTMLElement>, dds: DueDiligenceStatement) => {
    e.stopPropagation();
    setMenuAnchor(e.currentTarget);
    setMenuDDS(dds);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setMenuDDS(null);
  };

  const formatDate = (d: string | null) => {
    if (!d) return '-';
    return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  };

  // Generate year options from data
  const years = Array.from(
    new Set(ddsList.map((d) => new Date(d.generated_at).getFullYear().toString()))
  ).sort((a, b) => b.localeCompare(a));

  const hasFilters = search || statusFilter || commodityFilter || yearFilter;

  return (
    <Paper sx={{ width: '100%' }}>
      {/* Filter Bar */}
      <Box sx={{ p: 2, display: 'flex', flexWrap: 'wrap', gap: 1.5, alignItems: 'center' }}>
        <TextField
          size="small"
          placeholder="Search by supplier or reference..."
          value={search}
          onChange={handleSearchChange}
          sx={{ minWidth: 240, flex: 1 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment>
            ),
          }}
        />

        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Status</InputLabel>
          <Select value={statusFilter} label="Status" onChange={handleSelectChange(setStatusFilter, 'status')}>
            <MenuItem value="">All</MenuItem>
            {STATUS_OPTIONS.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel>Commodity</InputLabel>
          <Select value={commodityFilter} label="Commodity" onChange={handleSelectChange(setCommodityFilter, 'commodity')}>
            <MenuItem value="">All</MenuItem>
            {COMMODITY_OPTIONS.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 100 }}>
          <InputLabel>Year</InputLabel>
          <Select value={yearFilter} label="Year" onChange={handleSelectChange(setYearFilter, 'year')}>
            <MenuItem value="">All</MenuItem>
            {years.map((y) => (
              <MenuItem key={y} value={y}>{y}</MenuItem>
            ))}
          </Select>
        </FormControl>

        {hasFilters && (
          <Tooltip title="Clear all filters">
            <IconButton size="small" onClick={handleClearFilters}><ClearIcon fontSize="small" /></IconButton>
          </Tooltip>
        )}
      </Box>

      {/* Table */}
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Reference</TableCell>
              <TableCell>Supplier</TableCell>
              <TableCell>Commodity</TableCell>
              <TableCell align="right">Quantity (kg)</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Submission Date</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {ddsList.map((dds) => {
              const statusStyle = STATUS_COLORS[dds.status] ?? STATUS_COLORS.draft;
              return (
                <TableRow key={dds.id} hover sx={{ cursor: 'pointer' }} onClick={() => onView(dds)}>
                  <TableCell>
                    <Typography variant="body2" fontWeight={500}>{dds.reference_number}</Typography>
                  </TableCell>
                  <TableCell>{dds.supplier_name}</TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>
                    {dds.commodity.replace('_', ' ')}
                  </TableCell>
                  <TableCell align="right">{dds.total_quantity_kg.toLocaleString()}</TableCell>
                  <TableCell>
                    <Chip
                      label={dds.status.replace('_', ' ')}
                      size="small"
                      sx={{
                        backgroundColor: statusStyle.bg,
                        color: statusStyle.color,
                        fontWeight: 600,
                        textTransform: 'capitalize',
                      }}
                    />
                  </TableCell>
                  <TableCell>{formatDate(dds.submitted_at)}</TableCell>
                  <TableCell align="center" onClick={(e) => e.stopPropagation()}>
                    <Stack direction="row" spacing={0.5} justifyContent="center">
                      <Tooltip title="View"><IconButton size="small" onClick={() => onView(dds)}><VisibilityIcon fontSize="small" /></IconButton></Tooltip>
                      {(dds.status === 'draft' || dds.status === 'pending_review') && (
                        <Tooltip title="Validate"><IconButton size="small" onClick={() => onValidate(dds)}><CheckCircleIcon fontSize="small" color="info" /></IconButton></Tooltip>
                      )}
                      {dds.status === 'validated' && (
                        <Tooltip title="Submit"><IconButton size="small" onClick={() => onSubmit(dds)}><SendIcon fontSize="small" color="warning" /></IconButton></Tooltip>
                      )}
                      <Tooltip title="More actions">
                        <IconButton size="small" onClick={(e) => handleMenuOpen(e, dds)}>
                          <MoreVertIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  </TableCell>
                </TableRow>
              );
            })}
            {ddsList.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">No due diligence statements found.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      <TablePagination
        component="div"
        count={totalCount}
        page={page}
        rowsPerPage={rowsPerPage}
        rowsPerPageOptions={[10, 25, 50]}
        onPageChange={(_, p) => onPageChange(p)}
        onRowsPerPageChange={(e) => onRowsPerPageChange(parseInt(e.target.value, 10))}
      />

      {/* Actions Menu */}
      <Menu anchorEl={menuAnchor} open={Boolean(menuAnchor)} onClose={handleMenuClose}>
        <MenuItem onClick={() => { if (menuDDS) onDownload(menuDDS, 'pdf'); handleMenuClose(); }}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} /> Download PDF
        </MenuItem>
        <MenuItem onClick={() => { if (menuDDS) onDownload(menuDDS, 'xml'); handleMenuClose(); }}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} /> Download XML
        </MenuItem>
        <MenuItem onClick={() => { if (menuDDS) onDownload(menuDDS, 'json'); handleMenuClose(); }}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} /> Download JSON
        </MenuItem>
        {menuDDS && (menuDDS.status === 'rejected' || menuDDS.status === 'submitted') && (
          <MenuItem onClick={() => { if (menuDDS) onAmend(menuDDS); handleMenuClose(); }}>
            <EditIcon fontSize="small" sx={{ mr: 1 }} /> Amend
          </MenuItem>
        )}
      </Menu>
    </Paper>
  );
};

export default DDSTable;
