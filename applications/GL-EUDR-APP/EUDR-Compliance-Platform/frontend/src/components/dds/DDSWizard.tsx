/**
 * DDSWizard - Multi-step wizard for generating a new Due Diligence Statement.
 *
 * Steps: Select Supplier, Select Commodity & Year, Select Plots,
 * Review Documents, Risk Assessment Summary, Review & Generate.
 * Uses MUI Stepper with per-step validation and a summary sidebar.
 */

import React, { useState, useMemo, useEffect } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  Grid,
  TextField,
  Autocomplete,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Chip,
  Stack,
  Divider,
  Alert,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableRow,
  LinearProgress,
  SelectChangeEvent,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import type {
  Supplier,
  Plot,
  Document,
  EUDRCommodity,
  RiskLevel,
  DocumentGapAnalysis,
  RequiredDocument,
  DDSGenerateRequest,
} from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STEPS = [
  'Select Supplier',
  'Commodity & Year',
  'Select Plots',
  'Review Documents',
  'Risk Assessment',
  'Review & Generate',
];

const COMMODITIES: { value: EUDRCommodity; label: string }[] = [
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

const CURRENT_YEAR = new Date().getFullYear();
const YEAR_OPTIONS = Array.from({ length: 5 }, (_, i) => CURRENT_YEAR - i);

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DDSWizardProps {
  suppliers: Supplier[];
  /** Called to fetch plots when supplier is selected. */
  onFetchPlots: (supplierId: string) => Promise<Plot[]>;
  /** Called to fetch document gap analysis. */
  onFetchDocGap: (supplierId: string) => Promise<DocumentGapAnalysis>;
  /** Final generation callback. */
  onGenerate: (request: DDSGenerateRequest) => void;
  onCancel: () => void;
  loading?: boolean;
  /** Operator defaults from settings. */
  operatorDefaults?: { name: string; address: string; eori: string };
}

// ---------------------------------------------------------------------------
// Wizard State
// ---------------------------------------------------------------------------

interface WizardState {
  supplier: Supplier | null;
  commodity: EUDRCommodity | null;
  year: number;
  selectedPlotIds: Set<string>;
  plots: Plot[];
  docGap: DocumentGapAnalysis | null;
  quantity: string;
  riskMitigation: string[];
  operatorName: string;
  operatorAddress: string;
  operatorEORI: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DDSWizard: React.FC<DDSWizardProps> = ({
  suppliers,
  onFetchPlots,
  onFetchDocGap,
  onGenerate,
  onCancel,
  loading = false,
  operatorDefaults,
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [state, setState] = useState<WizardState>({
    supplier: null,
    commodity: null,
    year: CURRENT_YEAR,
    selectedPlotIds: new Set(),
    plots: [],
    docGap: null,
    quantity: '',
    riskMitigation: [],
    operatorName: operatorDefaults?.name ?? '',
    operatorAddress: operatorDefaults?.address ?? '',
    operatorEORI: operatorDefaults?.eori ?? '',
  });
  const [stepErrors, setStepErrors] = useState<Record<number, string>>({});
  const [plotsLoading, setPlotsLoading] = useState(false);
  const [docGapLoading, setDocGapLoading] = useState(false);

  // Load plots when supplier changes
  useEffect(() => {
    if (state.supplier) {
      setPlotsLoading(true);
      onFetchPlots(state.supplier.id)
        .then((plots) => setState((s) => ({ ...s, plots, selectedPlotIds: new Set() })))
        .catch(() => setState((s) => ({ ...s, plots: [] })))
        .finally(() => setPlotsLoading(false));
    }
  }, [state.supplier, onFetchPlots]);

  // Load doc gap when entering step 3
  useEffect(() => {
    if (activeStep === 3 && state.supplier && !state.docGap) {
      setDocGapLoading(true);
      onFetchDocGap(state.supplier.id)
        .then((gap) => setState((s) => ({ ...s, docGap: gap })))
        .catch(() => {})
        .finally(() => setDocGapLoading(false));
    }
  }, [activeStep, state.supplier, state.docGap, onFetchDocGap]);

  // Filtered plots by commodity
  const filteredPlots = useMemo(() => {
    if (!state.commodity) return state.plots;
    return state.plots.filter((p) => p.commodity === state.commodity);
  }, [state.plots, state.commodity]);

  // Selected plots
  const selectedPlots = useMemo(
    () => state.plots.filter((p) => state.selectedPlotIds.has(p.id)),
    [state.plots, state.selectedPlotIds]
  );

  // Calculated risk (simple heuristic)
  const calculatedRisk = useMemo((): RiskLevel => {
    if (selectedPlots.length === 0) return 'standard';
    const scores = selectedPlots.map((p) => {
      switch (p.risk_level) {
        case 'critical': return 1.0;
        case 'high': return 0.7;
        case 'standard': return 0.4;
        case 'low': return 0.1;
        default: return 0.5;
      }
    });
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    if (avg >= 0.8) return 'critical';
    if (avg >= 0.55) return 'high';
    if (avg >= 0.25) return 'standard';
    return 'low';
  }, [selectedPlots]);

  const totalArea = useMemo(
    () => selectedPlots.reduce((sum, p) => sum + p.area_hectares, 0),
    [selectedPlots]
  );

  // Validate current step
  const validateStep = (): boolean => {
    const newErrors: Record<number, string> = {};

    switch (activeStep) {
      case 0:
        if (!state.supplier) newErrors[0] = 'Please select a supplier.';
        break;
      case 1:
        if (!state.commodity) newErrors[1] = 'Please select a commodity.';
        break;
      case 2:
        if (state.selectedPlotIds.size === 0) newErrors[2] = 'Select at least one plot.';
        break;
      case 3:
        // Documents step -- warnings only, not blocking
        break;
      case 4:
        // Risk summary -- informational
        break;
      case 5:
        if (!state.quantity.trim() || isNaN(Number(state.quantity))) {
          newErrors[5] = 'Enter a valid total quantity in kg.';
        }
        if (!state.operatorName.trim()) newErrors[5] = 'Operator name is required.';
        break;
    }

    setStepErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep()) {
      setActiveStep((s) => Math.min(s + 1, STEPS.length - 1));
    }
  };

  const handleBack = () => {
    setActiveStep((s) => Math.max(s - 1, 0));
  };

  const handleGenerate = () => {
    if (!validateStep()) return;
    if (!state.supplier || !state.commodity) return;

    onGenerate({
      supplier_id: state.supplier.id,
      commodity: state.commodity,
      plot_ids: Array.from(state.selectedPlotIds),
      total_quantity_kg: Number(state.quantity),
      operator_name: state.operatorName,
      operator_address: state.operatorAddress,
      operator_eori: state.operatorEORI,
      risk_mitigation_measures: state.riskMitigation,
    });
  };

  const handlePlotToggle = (plotId: string) => {
    setState((s) => {
      const next = new Set(s.selectedPlotIds);
      if (next.has(plotId)) next.delete(plotId);
      else next.add(plotId);
      return { ...s, selectedPlotIds: next };
    });
  };

  const handleSelectAllPlots = () => {
    setState((s) => ({
      ...s,
      selectedPlotIds: new Set(filteredPlots.map((p) => p.id)),
    }));
  };

  // ---------------------------------------------------------------------------
  // Step renderers
  // ---------------------------------------------------------------------------

  const renderStep0 = () => (
    <Box>
      <Typography variant="subtitle1" gutterBottom>Search and select a supplier</Typography>
      <Autocomplete
        options={suppliers}
        getOptionLabel={(opt) => `${opt.name} (${opt.country})`}
        value={state.supplier}
        onChange={(_, val) => setState((s) => ({ ...s, supplier: val, docGap: null }))}
        renderInput={(params) => (
          <TextField {...params} label="Supplier" placeholder="Type to search..." fullWidth />
        )}
        sx={{ mb: 2 }}
      />
      {state.supplier && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="body2"><strong>Name:</strong> {state.supplier.name}</Typography>
            <Typography variant="body2"><strong>Country:</strong> {state.supplier.country}</Typography>
            <Typography variant="body2"><strong>Commodities:</strong> {state.supplier.commodities.map((c) => c.replace('_', ' ')).join(', ')}</Typography>
            <Typography variant="body2"><strong>Risk:</strong> {state.supplier.risk_level}</Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );

  const renderStep1 = () => (
    <Box>
      <Typography variant="subtitle1" gutterBottom>Select the commodity and reporting year</Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Commodity</InputLabel>
            <Select
              value={state.commodity ?? ''}
              label="Commodity"
              onChange={(e: SelectChangeEvent<string>) =>
                setState((s) => ({
                  ...s,
                  commodity: (e.target.value as EUDRCommodity) || null,
                  selectedPlotIds: new Set(),
                }))
              }
            >
              {COMMODITIES.filter(
                (c) => !state.supplier || state.supplier.commodities.includes(c.value)
              ).map((c) => (
                <MenuItem key={c.value} value={c.value}>{c.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>Year</InputLabel>
            <Select
              value={state.year.toString()}
              label="Year"
              onChange={(e) => setState((s) => ({ ...s, year: Number(e.target.value) }))}
            >
              {YEAR_OPTIONS.map((y) => (
                <MenuItem key={y} value={y.toString()}>{y}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>
    </Box>
  );

  const renderStep2 = () => (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="subtitle1">
          Select plots for this DDS ({filteredPlots.length} available)
        </Typography>
        <Button size="small" onClick={handleSelectAllPlots}>Select All</Button>
      </Stack>
      {plotsLoading ? (
        <LinearProgress sx={{ my: 2 }} />
      ) : filteredPlots.length === 0 ? (
        <Alert severity="warning">
          No plots found for the selected commodity. Add plots first or change commodity.
        </Alert>
      ) : (
        <List dense sx={{ maxHeight: 320, overflow: 'auto', border: '1px solid #e0e0e0', borderRadius: 1 }}>
          {filteredPlots.map((plot) => (
            <ListItem key={plot.id} disablePadding>
              <ListItemButton onClick={() => handlePlotToggle(plot.id)}>
                <ListItemIcon>
                  <Checkbox checked={state.selectedPlotIds.has(plot.id)} edge="start" />
                </ListItemIcon>
                <ListItemText
                  primary={plot.name}
                  secondary={`${plot.country} | ${plot.area_hectares.toFixed(1)} ha | Risk: ${plot.risk_level}`}
                />
                <Chip
                  size="small"
                  label={plot.risk_level}
                  sx={{
                    backgroundColor: RISK_COLORS[plot.risk_level],
                    color: '#fff',
                    textTransform: 'capitalize',
                    fontWeight: 600,
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      )}
      <Typography variant="body2" color="text.secondary" mt={1}>
        {state.selectedPlotIds.size} plot(s) selected | Total area: {totalArea.toFixed(1)} ha
      </Typography>
    </Box>
  );

  const renderStep3 = () => (
    <Box>
      <Typography variant="subtitle1" gutterBottom>Document Review</Typography>
      {docGapLoading ? (
        <LinearProgress sx={{ my: 2 }} />
      ) : state.docGap ? (
        <>
          <Stack direction="row" spacing={2} mb={2}>
            <Paper variant="outlined" sx={{ p: 1, flex: 1, textAlign: 'center' }}>
              <Typography variant="h5">{state.docGap.completeness_percentage.toFixed(0)}%</Typography>
              <Typography variant="caption" color="text.secondary">Completeness</Typography>
            </Paper>
            <Paper variant="outlined" sx={{ p: 1, flex: 1, textAlign: 'center' }}>
              <Typography variant="h5" color="error.main">{state.docGap.missing_critical}</Typography>
              <Typography variant="caption" color="text.secondary">Missing Critical</Typography>
            </Paper>
            <Paper variant="outlined" sx={{ p: 1, flex: 1, textAlign: 'center' }}>
              <Typography variant="h5" color="warning.main">{state.docGap.missing_recommended}</Typography>
              <Typography variant="caption" color="text.secondary">Missing Recommended</Typography>
            </Paper>
          </Stack>
          <List dense>
            {state.docGap.required_documents.map((doc, idx) => (
              <ListItem key={idx} divider>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  {doc.provided ? (
                    <CheckCircleIcon color="success" fontSize="small" />
                  ) : doc.required ? (
                    <ErrorOutlineIcon color="error" fontSize="small" />
                  ) : (
                    <WarningAmberIcon color="warning" fontSize="small" />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={doc.label}
                  secondary={
                    doc.provided
                      ? `Verification: ${doc.verification_status ?? 'pending'}`
                      : doc.required
                      ? 'MISSING - Required for DDS submission'
                      : 'MISSING - Recommended'
                  }
                />
              </ListItem>
            ))}
          </List>
          {state.docGap.missing_critical > 0 && (
            <Alert severity="warning" sx={{ mt: 1 }}>
              {state.docGap.missing_critical} critical document(s) are missing. The DDS can be
              generated as draft but cannot be submitted until all required documents are provided.
            </Alert>
          )}
        </>
      ) : (
        <Alert severity="info">No document gap analysis available. Proceed to generate the DDS.</Alert>
      )}
    </Box>
  );

  const renderStep4 = () => (
    <Box>
      <Typography variant="subtitle1" gutterBottom>Risk Assessment Summary</Typography>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Typography variant="body1">Calculated Risk Level:</Typography>
            <Chip
              label={calculatedRisk.replace('_', ' ')}
              sx={{
                backgroundColor: RISK_COLORS[calculatedRisk],
                color: '#fff',
                fontWeight: 700,
                textTransform: 'capitalize',
                fontSize: 14,
                px: 1,
              }}
            />
          </Stack>
        </CardContent>
      </Card>

      <Typography variant="subtitle2" gutterBottom>Risk Factors</Typography>
      <List dense>
        <ListItem>
          <ListItemText
            primary="Country Risk"
            secondary={state.supplier ? `${state.supplier.country} - ${state.supplier.risk_level} risk` : '-'}
          />
        </ListItem>
        <ListItem>
          <ListItemText
            primary="Plots"
            secondary={`${selectedPlots.filter((p) => p.deforestation_free === false).length} plot(s) with deforestation detected`}
          />
        </ListItem>
        <ListItem>
          <ListItemText
            primary="Supplier Compliance"
            secondary={state.supplier?.compliance_status.replace('_', ' ') ?? '-'}
          />
        </ListItem>
      </List>

      {(calculatedRisk === 'high' || calculatedRisk === 'critical') && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          High or critical risk detected. Consider adding risk mitigation measures before submission.
        </Alert>
      )}
    </Box>
  );

  const renderStep5 = () => (
    <Box>
      <Typography variant="subtitle1" gutterBottom>Review and Generate</Typography>

      {/* Summary */}
      <Table size="small" sx={{ mb: 2 }}>
        <TableBody>
          <TableRow>
            <TableCell sx={{ fontWeight: 500 }}>Supplier</TableCell>
            <TableCell>{state.supplier?.name ?? '-'}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ fontWeight: 500 }}>Commodity</TableCell>
            <TableCell sx={{ textTransform: 'capitalize' }}>{state.commodity?.replace('_', ' ') ?? '-'}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ fontWeight: 500 }}>Year</TableCell>
            <TableCell>{state.year}</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ fontWeight: 500 }}>Plots</TableCell>
            <TableCell>{state.selectedPlotIds.size} ({totalArea.toFixed(1)} ha)</TableCell>
          </TableRow>
          <TableRow>
            <TableCell sx={{ fontWeight: 500 }}>Risk Level</TableCell>
            <TableCell>
              <Chip
                label={calculatedRisk}
                size="small"
                sx={{ backgroundColor: RISK_COLORS[calculatedRisk], color: '#fff', textTransform: 'capitalize' }}
              />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>

      <Divider sx={{ my: 2 }} />

      {/* Operator info + quantity */}
      <Typography variant="subtitle2" gutterBottom>Operator Details</Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            required
            label="Operator Name"
            value={state.operatorName}
            onChange={(e) => setState((s) => ({ ...s, operatorName: e.target.value }))}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="EORI Number"
            value={state.operatorEORI}
            onChange={(e) => setState((s) => ({ ...s, operatorEORI: e.target.value }))}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Operator Address"
            value={state.operatorAddress}
            onChange={(e) => setState((s) => ({ ...s, operatorAddress: e.target.value }))}
            multiline
            rows={2}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            required
            label="Total Quantity (kg)"
            type="number"
            value={state.quantity}
            onChange={(e) => setState((s) => ({ ...s, quantity: e.target.value }))}
            helperText="Total product quantity covered by this DDS"
          />
        </Grid>
      </Grid>
    </Box>
  );

  const stepRenderers = [renderStep0, renderStep1, renderStep2, renderStep3, renderStep4, renderStep5];

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Generate Due Diligence Statement
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Grid container spacing={3}>
        {/* Main content */}
        <Grid item xs={12} md={8}>
          {stepErrors[activeStep] && (
            <Alert severity="error" sx={{ mb: 2 }}>{stepErrors[activeStep]}</Alert>
          )}
          {stepRenderers[activeStep]()}
        </Grid>

        {/* Summary sidebar */}
        <Grid item xs={12} md={4}>
          <Card variant="outlined" sx={{ position: 'sticky', top: 16 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>DDS Summary</Typography>
              <Divider sx={{ mb: 1 }} />
              <Stack spacing={0.75}>
                <Typography variant="body2" color="text.secondary">
                  Supplier: <strong>{state.supplier?.name ?? 'Not selected'}</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Commodity:{' '}
                  <strong style={{ textTransform: 'capitalize' }}>
                    {state.commodity?.replace('_', ' ') ?? 'Not selected'}
                  </strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Year: <strong>{state.year}</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Plots: <strong>{state.selectedPlotIds.size}</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Area: <strong>{totalArea.toFixed(1)} ha</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Risk:{' '}
                  <Chip
                    label={calculatedRisk}
                    size="small"
                    sx={{
                      backgroundColor: RISK_COLORS[calculatedRisk],
                      color: '#fff',
                      textTransform: 'capitalize',
                      fontWeight: 600,
                      height: 20,
                      fontSize: 11,
                    }}
                  />
                </Typography>
                {state.quantity && (
                  <Typography variant="body2" color="text.secondary">
                    Quantity: <strong>{Number(state.quantity).toLocaleString()} kg</strong>
                  </Typography>
                )}
              </Stack>

              <Divider sx={{ my: 1.5 }} />

              <Typography variant="caption" color="text.secondary">
                Step {activeStep + 1} of {STEPS.length}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={((activeStep + 1) / STEPS.length) * 100}
                sx={{ mt: 0.5, height: 6, borderRadius: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Navigation */}
      <Divider sx={{ my: 2 }} />
      <Stack direction="row" justifyContent="space-between">
        <Button
          variant="outlined"
          onClick={activeStep === 0 ? onCancel : handleBack}
          startIcon={activeStep === 0 ? undefined : <ArrowBackIcon />}
        >
          {activeStep === 0 ? 'Cancel' : 'Back'}
        </Button>

        {activeStep < STEPS.length - 1 ? (
          <Button variant="contained" onClick={handleNext} endIcon={<ArrowForwardIcon />}>
            Next
          </Button>
        ) : (
          <Button
            variant="contained"
            color="success"
            onClick={handleGenerate}
            startIcon={<AutoFixHighIcon />}
            disabled={loading}
          >
            {loading ? 'Generating...' : 'Generate DDS'}
          </Button>
        )}
      </Stack>
    </Paper>
  );
};

export default DDSWizard;
