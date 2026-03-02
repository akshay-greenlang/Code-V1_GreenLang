/**
 * SupplierTable - Interactive MUI table for supplier listing.
 *
 * Displays suppliers with sortable columns, filter bar (country, commodity,
 * risk level, compliance status), search, pagination, row navigation,
 * and bulk selection with action buttons.
 */

import React, { useState, useMemo, useCallback } from 'react';
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
  Checkbox,
  Button,
  Stack,
  InputAdornment,
  IconButton,
  Tooltip,
  Typography,
  SelectChangeEvent,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import DescriptionIcon from '@mui/icons-material/Description';
import ClearIcon from '@mui/icons-material/Clear';
import type {
  Supplier,
  EUDRCommodity,
  RiskLevel,
  ComplianceStatus,
} from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COMMODITY_OPTIONS: { value: EUDRCommodity; label: string }[] = [
  { value: 'cattle' as EUDRCommodity, label: 'Cattle' },
  { value: 'cocoa' as EUDRCommodity, label: 'Cocoa' },
  { value: 'coffee' as EUDRCommodity, label: 'Coffee' },
  { value: 'oil_palm' as EUDRCommodity, label: 'Oil Palm' },
  { value: 'rubber' as EUDRCommodity, label: 'Rubber' },
  { value: 'soya' as EUDRCommodity, label: 'Soya' },
  { value: 'wood' as EUDRCommodity, label: 'Wood' },
];

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

type SortField = keyof Pick<
  Supplier,
  'name' | 'country' | 'risk_level' | 'compliance_status' | 'total_plots' | 'active_dds_count'
>;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface SupplierTableProps {
  suppliers: Supplier[];
  totalCount: number;
  page: number;
  rowsPerPage: number;
  loading?: boolean;
  onPageChange: (page: number) => void;
  onRowsPerPageChange: (perPage: number) => void;
  onRowClick: (supplier: Supplier) => void;
  onFilterChange: (filters: SupplierFilters) => void;
  onBulkExport?: (ids: string[]) => void;
  onBulkGenerateDDS?: (ids: string[]) => void;
}

export interface SupplierFilters {
  search: string;
  country: string;
  commodity: string;
  risk_level: string;
  compliance_status: string;
  sort_by: string;
  sort_order: 'asc' | 'desc';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const SupplierTable: React.FC<SupplierTableProps> = ({
  suppliers,
  totalCount,
  page,
  rowsPerPage,
  loading = false,
  onPageChange,
  onRowsPerPageChange,
  onRowClick,
  onFilterChange,
  onBulkExport,
  onBulkGenerateDDS,
}) => {
  // Filter state
  const [search, setSearch] = useState('');
  const [countryFilter, setCountryFilter] = useState('');
  const [commodityFilter, setCommodityFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [complianceFilter, setComplianceFilter] = useState('');
  const [sortBy, setSortBy] = useState<SortField>('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  // Bulk selection
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // Country list derived from data
  const countries = useMemo(() => {
    const set = new Set(suppliers.map((s) => s.country));
    return Array.from(set).sort();
  }, [suppliers]);

  // Emit filter changes
  const emitFilters = useCallback(
    (overrides: Partial<SupplierFilters> = {}) => {
      onFilterChange({
        search,
        country: countryFilter,
        commodity: commodityFilter,
        risk_level: riskFilter,
        compliance_status: complianceFilter,
        sort_by: sortBy,
        sort_order: sortOrder,
        ...overrides,
      });
    },
    [search, countryFilter, commodityFilter, riskFilter, complianceFilter, sortBy, sortOrder, onFilterChange]
  );

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearch(val);
    emitFilters({ search: val });
  };

  const handleSelectChange =
    (setter: (v: string) => void, field: keyof SupplierFilters) =>
    (e: SelectChangeEvent<string>) => {
      const val = e.target.value;
      setter(val);
      emitFilters({ [field]: val });
    };

  const handleSort = (field: SortField) => {
    const isAsc = sortBy === field && sortOrder === 'asc';
    const newOrder = isAsc ? 'desc' : 'asc';
    setSortBy(field);
    setSortOrder(newOrder);
    emitFilters({ sort_by: field, sort_order: newOrder });
  };

  const handleClearFilters = () => {
    setSearch('');
    setCountryFilter('');
    setCommodityFilter('');
    setRiskFilter('');
    setComplianceFilter('');
    emitFilters({
      search: '',
      country: '',
      commodity: '',
      risk_level: '',
      compliance_status: '',
    });
  };

  const handleSelectAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      setSelected(new Set(suppliers.map((s) => s.id)));
    } else {
      setSelected(new Set());
    }
  };

  const handleSelectOne = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const hasFilters = search || countryFilter || commodityFilter || riskFilter || complianceFilter;
  const selectedArray = Array.from(selected);

  return (
    <Paper sx={{ width: '100%' }}>
      {/* Filter Bar */}
      <Box sx={{ p: 2, display: 'flex', flexWrap: 'wrap', gap: 1.5, alignItems: 'center' }}>
        <TextField
          size="small"
          placeholder="Search suppliers..."
          value={search}
          onChange={handleSearchChange}
          sx={{ minWidth: 220, flex: 1 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
          }}
        />

        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Country</InputLabel>
          <Select
            value={countryFilter}
            label="Country"
            onChange={handleSelectChange(setCountryFilter, 'country')}
          >
            <MenuItem value="">All</MenuItem>
            {countries.map((c) => (
              <MenuItem key={c} value={c}>{c}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Commodity</InputLabel>
          <Select
            value={commodityFilter}
            label="Commodity"
            onChange={handleSelectChange(setCommodityFilter, 'commodity')}
          >
            <MenuItem value="">All</MenuItem>
            {COMMODITY_OPTIONS.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel>Risk Level</InputLabel>
          <Select
            value={riskFilter}
            label="Risk Level"
            onChange={handleSelectChange(setRiskFilter, 'risk_level')}
          >
            <MenuItem value="">All</MenuItem>
            <MenuItem value="low">Low</MenuItem>
            <MenuItem value="standard">Standard</MenuItem>
            <MenuItem value="high">High</MenuItem>
            <MenuItem value="critical">Critical</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Compliance</InputLabel>
          <Select
            value={complianceFilter}
            label="Compliance"
            onChange={handleSelectChange(setComplianceFilter, 'compliance_status')}
          >
            <MenuItem value="">All</MenuItem>
            <MenuItem value="compliant">Compliant</MenuItem>
            <MenuItem value="non_compliant">Non-Compliant</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="under_review">Under Review</MenuItem>
            <MenuItem value="expired">Expired</MenuItem>
          </Select>
        </FormControl>

        {hasFilters && (
          <Tooltip title="Clear all filters">
            <IconButton size="small" onClick={handleClearFilters}>
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>

      {/* Bulk Actions */}
      {selected.size > 0 && (
        <Box sx={{ px: 2, pb: 1, display: 'flex', gap: 1, alignItems: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            {selected.size} selected
          </Typography>
          {onBulkExport && (
            <Button
              size="small"
              variant="outlined"
              startIcon={<FileDownloadIcon />}
              onClick={() => onBulkExport(selectedArray)}
            >
              Export
            </Button>
          )}
          {onBulkGenerateDDS && (
            <Button
              size="small"
              variant="outlined"
              startIcon={<DescriptionIcon />}
              onClick={() => onBulkGenerateDDS(selectedArray)}
            >
              Generate DDS
            </Button>
          )}
        </Box>
      )}

      {/* Table */}
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={selected.size > 0 && selected.size < suppliers.length}
                  checked={suppliers.length > 0 && selected.size === suppliers.length}
                  onChange={handleSelectAll}
                />
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === 'name'}
                  direction={sortBy === 'name' ? sortOrder : 'asc'}
                  onClick={() => handleSort('name')}
                >
                  Name
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === 'country'}
                  direction={sortBy === 'country' ? sortOrder : 'asc'}
                  onClick={() => handleSort('country')}
                >
                  Country
                </TableSortLabel>
              </TableCell>
              <TableCell>Commodities</TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === 'risk_level'}
                  direction={sortBy === 'risk_level' ? sortOrder : 'asc'}
                  onClick={() => handleSort('risk_level')}
                >
                  Risk Level
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortBy === 'compliance_status'}
                  direction={sortBy === 'compliance_status' ? sortOrder : 'asc'}
                  onClick={() => handleSort('compliance_status')}
                >
                  Compliance
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={sortBy === 'total_plots'}
                  direction={sortBy === 'total_plots' ? sortOrder : 'asc'}
                  onClick={() => handleSort('total_plots')}
                >
                  Plots
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={sortBy === 'active_dds_count'}
                  direction={sortBy === 'active_dds_count' ? sortOrder : 'asc'}
                  onClick={() => handleSort('active_dds_count')}
                >
                  DDS Count
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {suppliers.map((supplier) => (
              <TableRow
                key={supplier.id}
                hover
                onClick={() => onRowClick(supplier)}
                sx={{ cursor: 'pointer' }}
                selected={selected.has(supplier.id)}
              >
                <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                  <Checkbox
                    checked={selected.has(supplier.id)}
                    onChange={() => handleSelectOne(supplier.id)}
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontWeight={500}>
                    {supplier.name}
                  </Typography>
                </TableCell>
                <TableCell>{supplier.country}</TableCell>
                <TableCell>
                  <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                    {supplier.commodities.map((c) => (
                      <Chip
                        key={c}
                        label={c.replace('_', ' ')}
                        size="small"
                        sx={{ textTransform: 'capitalize', fontSize: 11 }}
                      />
                    ))}
                  </Stack>
                </TableCell>
                <TableCell>
                  <Chip
                    label={supplier.risk_level.replace('_', ' ')}
                    size="small"
                    sx={{
                      backgroundColor: RISK_COLORS[supplier.risk_level],
                      color:
                        supplier.risk_level === 'low' || supplier.risk_level === 'standard'
                          ? '#fff'
                          : '#fff',
                      textTransform: 'capitalize',
                      fontWeight: 600,
                    }}
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={supplier.compliance_status.replace('_', ' ')}
                    size="small"
                    sx={{
                      backgroundColor: COMPLIANCE_COLORS[supplier.compliance_status],
                      color: '#fff',
                      textTransform: 'capitalize',
                      fontWeight: 600,
                    }}
                  />
                </TableCell>
                <TableCell align="right">{supplier.total_plots}</TableCell>
                <TableCell align="right">{supplier.active_dds_count}</TableCell>
              </TableRow>
            ))}

            {suppliers.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={8} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">
                    No suppliers found. Adjust filters or add a new supplier.
                  </Typography>
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
        rowsPerPageOptions={[10, 25, 50, 100]}
        onPageChange={(_, newPage) => onPageChange(newPage)}
        onRowsPerPageChange={(e) => onRowsPerPageChange(parseInt(e.target.value, 10))}
      />
    </Paper>
  );
};

export default SupplierTable;
