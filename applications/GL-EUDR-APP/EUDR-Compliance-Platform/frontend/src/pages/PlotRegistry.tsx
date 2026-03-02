/**
 * PlotRegistry - Page for managing geolocation plots.
 *
 * Full-width PlotMap (400px), plot data table with filter/sort, "Add Plot"
 * and "Validate All" buttons, and a side panel for selected plot details
 * plus satellite overlay.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Stack,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Chip,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  InputAdornment,
  Drawer,
  Divider,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Snackbar,
  SelectChangeEvent,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SearchIcon from '@mui/icons-material/Search';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import PlotMap from '../components/maps/PlotMap';
import SatelliteOverlay, { SatelliteAssessment } from '../components/maps/SatelliteOverlay';
import apiClient from '../services/api';
import type { Plot, RiskLevel, EUDRCommodity } from '../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RISK_COLORS: Record<RiskLevel, string> = {
  low: '#4caf50',
  standard: '#2196f3',
  high: '#ff9800',
  critical: '#f44336',
};

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
// Component
// ---------------------------------------------------------------------------

const PlotRegistry: React.FC = () => {
  // Plot data
  const [plots, setPlots] = useState<Plot[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [search, setSearch] = useState('');
  const [countryFilter, setCountryFilter] = useState('');
  const [commodityFilter, setCommodityFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  // Selected plot
  const [selectedPlot, setSelectedPlot] = useState<Plot | null>(null);
  const [satelliteData, setSatelliteData] = useState<SatelliteAssessment[]>([]);

  // Snackbar
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false, message: '', severity: 'success',
  });

  // Fetch plots
  const fetchPlots = useCallback(async () => {
    try {
      setLoading(true);
      const result = await apiClient.getPlots({
        page: page + 1,
        per_page: rowsPerPage,
        search: search || undefined,
        country: countryFilter || undefined,
        commodity: (commodityFilter as EUDRCommodity) || undefined,
        risk_level: (riskFilter as RiskLevel) || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      setPlots(result.items);
      setTotalCount(result.total);
    } catch {
      setError('Failed to load plots.');
    } finally {
      setLoading(false);
    }
  }, [page, rowsPerPage, search, countryFilter, commodityFilter, riskFilter, sortBy, sortOrder]);

  useEffect(() => {
    fetchPlots();
  }, [fetchPlots]);

  // Handle plot selection
  const handlePlotClick = (plot: Plot) => {
    setSelectedPlot(plot);
    // Mock satellite data -- in real app, fetch from API
    setSatelliteData([
      {
        id: `sa-${plot.id}-1`,
        plot_id: plot.id,
        assessment_date: new Date().toISOString(),
        satellite_source: 'Sentinel-2',
        ndvi_baseline: 0.72,
        ndvi_current: plot.deforestation_free ? 0.68 : 0.45,
        ndvi_change: plot.deforestation_free ? -0.04 : -0.27,
        forest_cover_baseline_pct: 87.5,
        forest_cover_current_pct: plot.deforestation_free ? 84.2 : 62.1,
        tree_loss_pct: plot.deforestation_free ? 3.3 : 25.4,
        deforestation_detected: !plot.deforestation_free,
        confidence_score: 0.92,
        classification: plot.deforestation_free ? 'no_change' : 'deforestation',
        reference_date: '2020-01-01',
        imagery_resolution_m: 10,
        notes: '',
      },
    ]);
  };

  const handlePolygonDraw = (geojson: GeoJSON.Geometry) => {
    setSnackbar({ open: true, message: 'New polygon drawn. Opening plot form...', severity: 'success' });
  };

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder((o) => (o === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
  };

  const handleValidateAll = async () => {
    setSnackbar({ open: true, message: 'Validating all plots...', severity: 'success' });
  };

  // Unique countries from data
  const countries = [...new Set(plots.map((p) => p.country))].sort();

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h4" fontWeight={700}>
          Plot Registry
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<CheckCircleIcon />} onClick={handleValidateAll}>
            Validate All
          </Button>
          <Button variant="contained" startIcon={<AddIcon />}>
            Add Plot
          </Button>
        </Stack>
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Map */}
      <Paper sx={{ mb: 3, overflow: 'hidden', borderRadius: 2 }}>
        <PlotMap
          plots={plots}
          height={400}
          onPlotClick={handlePlotClick}
          onPolygonDraw={handlePolygonDraw}
          selectedPlotId={selectedPlot?.id}
        />
      </Paper>

      {/* Filter bar */}
      <Stack direction="row" spacing={1.5} mb={2} flexWrap="wrap" useFlexGap>
        <TextField
          size="small"
          placeholder="Search plots..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          sx={{ minWidth: 200, flex: 1 }}
          InputProps={{
            startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" /></InputAdornment>,
          }}
        />
        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel>Country</InputLabel>
          <Select value={countryFilter} label="Country" onChange={(e: SelectChangeEvent) => { setCountryFilter(e.target.value); setPage(0); }}>
            <MenuItem value="">All</MenuItem>
            {countries.map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel>Commodity</InputLabel>
          <Select value={commodityFilter} label="Commodity" onChange={(e: SelectChangeEvent) => { setCommodityFilter(e.target.value); setPage(0); }}>
            <MenuItem value="">All</MenuItem>
            {COMMODITY_OPTIONS.map((o) => <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>)}
          </Select>
        </FormControl>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Risk</InputLabel>
          <Select value={riskFilter} label="Risk" onChange={(e: SelectChangeEvent) => { setRiskFilter(e.target.value); setPage(0); }}>
            <MenuItem value="">All</MenuItem>
            <MenuItem value="low">Low</MenuItem>
            <MenuItem value="standard">Standard</MenuItem>
            <MenuItem value="high">High</MenuItem>
            <MenuItem value="critical">Critical</MenuItem>
          </Select>
        </FormControl>
      </Stack>

      {/* Plot Table */}
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel active={sortBy === 'name'} direction={sortBy === 'name' ? sortOrder : 'asc'} onClick={() => handleSort('name')}>
                  Name
                </TableSortLabel>
              </TableCell>
              <TableCell>Supplier</TableCell>
              <TableCell>
                <TableSortLabel active={sortBy === 'country'} direction={sortBy === 'country' ? sortOrder : 'asc'} onClick={() => handleSort('country')}>
                  Country
                </TableSortLabel>
              </TableCell>
              <TableCell>Commodity</TableCell>
              <TableCell align="right">
                <TableSortLabel active={sortBy === 'area_hectares'} direction={sortBy === 'area_hectares' ? sortOrder : 'asc'} onClick={() => handleSort('area_hectares')}>
                  Area (ha)
                </TableSortLabel>
              </TableCell>
              <TableCell>Risk</TableCell>
              <TableCell>Deforestation Free</TableCell>
              <TableCell>Last Check</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {plots.map((plot) => (
              <TableRow
                key={plot.id}
                hover
                selected={selectedPlot?.id === plot.id}
                onClick={() => handlePlotClick(plot)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell><Typography variant="body2" fontWeight={500}>{plot.name}</Typography></TableCell>
                <TableCell>{plot.supplier_name}</TableCell>
                <TableCell>{plot.country}</TableCell>
                <TableCell sx={{ textTransform: 'capitalize' }}>{plot.commodity.replace('_', ' ')}</TableCell>
                <TableCell align="right">{plot.area_hectares.toFixed(1)}</TableCell>
                <TableCell>
                  <Chip label={plot.risk_level} size="small" sx={{ backgroundColor: RISK_COLORS[plot.risk_level], color: '#fff', textTransform: 'capitalize', fontWeight: 600 }} />
                </TableCell>
                <TableCell>
                  {plot.deforestation_free === null
                    ? <Chip label="Pending" size="small" />
                    : plot.deforestation_free
                    ? <Chip label="Yes" size="small" color="success" />
                    : <Chip label="No" size="small" color="error" />}
                </TableCell>
                <TableCell>
                  {plot.deforestation_check_date
                    ? new Date(plot.deforestation_check_date).toLocaleDateString('en-GB')
                    : '-'}
                </TableCell>
              </TableRow>
            ))}
            {plots.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={8} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">No plots found.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <TablePagination
          component="div"
          count={totalCount}
          page={page}
          rowsPerPage={rowsPerPage}
          rowsPerPageOptions={[10, 25, 50, 100]}
          onPageChange={(_, p) => setPage(p)}
          onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value, 10)); setPage(0); }}
        />
      </TableContainer>

      {/* Side Panel (Drawer) */}
      <Drawer
        anchor="right"
        open={Boolean(selectedPlot)}
        onClose={() => { setSelectedPlot(null); setSatelliteData([]); }}
        PaperProps={{ sx: { width: { xs: '100%', md: 520 }, p: 2 } }}
      >
        {selectedPlot && (
          <Box>
            <Typography variant="h6" gutterBottom>{selectedPlot.name}</Typography>
            <Stack spacing={1} mb={2}>
              <Typography variant="body2"><strong>Supplier:</strong> {selectedPlot.supplier_name}</Typography>
              <Typography variant="body2"><strong>Country:</strong> {selectedPlot.country}, {selectedPlot.region}</Typography>
              <Typography variant="body2" sx={{ textTransform: 'capitalize' }}><strong>Commodity:</strong> {selectedPlot.commodity.replace('_', ' ')}</Typography>
              <Typography variant="body2"><strong>Area:</strong> {selectedPlot.area_hectares.toFixed(2)} ha</Typography>
              <Stack direction="row" spacing={1}>
                <Chip label={selectedPlot.risk_level} size="small" sx={{ backgroundColor: RISK_COLORS[selectedPlot.risk_level], color: '#fff', textTransform: 'capitalize' }} />
                {selectedPlot.deforestation_free !== null && (
                  <Chip
                    label={selectedPlot.deforestation_free ? 'Deforestation Free' : 'Deforestation Detected'}
                    size="small"
                    color={selectedPlot.deforestation_free ? 'success' : 'error'}
                  />
                )}
              </Stack>
            </Stack>
            <Divider sx={{ my: 2 }} />
            <SatelliteOverlay plotId={selectedPlot.id} assessments={satelliteData} />
          </Box>
        )}
      </Drawer>

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

export default PlotRegistry;
