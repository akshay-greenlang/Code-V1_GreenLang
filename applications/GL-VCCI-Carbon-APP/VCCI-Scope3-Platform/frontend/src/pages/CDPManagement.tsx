/**
 * CDPManagement - CDP questionnaire management page
 *
 * Provides a full-page layout for editing CDP Climate Change questionnaires
 * with year selection, auto-population, validation, export, and progress
 * tracking. Left column shows the questionnaire editor, right column shows
 * the progress tracker, and a collapsible bottom section shows data mappings.
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  AlertTitle,
  Collapse,
  IconButton,
  Divider,
  Chip,
  Menu,
  SelectChangeEvent,
} from '@mui/material';
import {
  AutoFixHigh,
  CheckCircle,
  Download,
  ExpandMore,
  ExpandLess,
  Refresh,
  FactCheck,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchCDPQuestionnaire,
  autoPopulateCDP,
  saveCDPDraft,
  validateCDP,
  exportCDP,
  fetchCDPProgress,
  fetchScorePrediction,
  setSelectedYear,
  updateQuestionValue,
  clearValidation,
} from '../store/slices/cdpSlice';
import {
  CDPQuestionnaireEditor,
  CDPProgressTracker,
  CDPDataMapping,
} from '../components/cdp';
import LoadingSpinner from '../components/LoadingSpinner';

// =============================================================================
// Constants
// =============================================================================

const AVAILABLE_YEARS = [2024, 2025, 2026];

// =============================================================================
// Main Component
// =============================================================================

const CDPManagement: React.FC = () => {
  const dispatch = useAppDispatch();
  const {
    questionnaire,
    progress,
    mappings,
    scorePrediction,
    validation,
    selectedYear,
    loading,
    saving,
    error,
  } = useAppSelector((state) => state.cdp);

  const [mappingExpanded, setMappingExpanded] = useState(false);
  const [exportAnchorEl, setExportAnchorEl] = useState<null | HTMLElement>(null);

  // Load data on mount and year change
  useEffect(() => {
    dispatch(fetchCDPQuestionnaire(selectedYear));
    dispatch(fetchCDPProgress(selectedYear));
    dispatch(fetchScorePrediction(selectedYear));
  }, [dispatch, selectedYear]);

  // Handlers
  const handleYearChange = useCallback(
    (e: SelectChangeEvent<number>) => {
      dispatch(setSelectedYear(Number(e.target.value)));
    },
    [dispatch]
  );

  const handleAutoPopulate = useCallback(() => {
    dispatch(autoPopulateCDP(selectedYear));
  }, [dispatch, selectedYear]);

  const handleSave = useCallback(
    (data: any) => {
      dispatch(saveCDPDraft({ year: selectedYear, data }));
    },
    [dispatch, selectedYear]
  );

  const handleQuestionChange = useCallback(
    (sectionId: string, questionId: string, value: any) => {
      dispatch(updateQuestionValue({ sectionId, questionId, value }));
    },
    [dispatch]
  );

  const handleValidate = useCallback(() => {
    dispatch(validateCDP(selectedYear));
  }, [dispatch, selectedYear]);

  const handleExportClick = useCallback(
    (event: React.MouseEvent<HTMLElement>) => {
      setExportAnchorEl(event.currentTarget);
    },
    []
  );

  const handleExportClose = useCallback(() => {
    setExportAnchorEl(null);
  }, []);

  const handleExport = useCallback(
    (format: string) => {
      dispatch(exportCDP({ year: selectedYear, format }));
      setExportAnchorEl(null);
    },
    [dispatch, selectedYear]
  );

  const handleRefresh = useCallback(() => {
    dispatch(fetchCDPQuestionnaire(selectedYear));
    dispatch(fetchCDPProgress(selectedYear));
    dispatch(fetchScorePrediction(selectedYear));
    dispatch(clearValidation());
  }, [dispatch, selectedYear]);

  const toggleMapping = useCallback(() => {
    setMappingExpanded((prev) => !prev);
  }, []);

  // Compute deadline for the progress tracker
  const deadline = questionnaire?.submissionDeadline;

  // Loading state
  if (loading && !questionnaire) {
    return <LoadingSpinner message="Loading CDP questionnaire..." />;
  }

  return (
    <Box>
      {/* Page Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
          flexWrap: 'wrap',
          gap: 1,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h4">CDP Management</Typography>
          {scorePrediction && (
            <Chip
              label={`Predicted Score: ${scorePrediction}`}
              color={
                scorePrediction.startsWith('A')
                  ? 'success'
                  : scorePrediction.startsWith('B')
                  ? 'info'
                  : scorePrediction.startsWith('C')
                  ? 'warning'
                  : 'error'
              }
              sx={{ fontWeight: 'bold' }}
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Year selector */}
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel>Year</InputLabel>
            <Select
              value={selectedYear}
              label="Year"
              onChange={handleYearChange}
            >
              {AVAILABLE_YEARS.map((year) => (
                <MenuItem key={year} value={year}>
                  {year}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            variant="outlined"
            startIcon={<AutoFixHigh />}
            onClick={handleAutoPopulate}
            disabled={loading}
            size="small"
          >
            Auto-populate
          </Button>

          <Button
            variant="outlined"
            startIcon={<FactCheck />}
            onClick={handleValidate}
            disabled={loading}
            size="small"
          >
            Validate
          </Button>

          <Button
            variant="outlined"
            startIcon={<Download />}
            onClick={handleExportClick}
            disabled={loading}
            size="small"
          >
            Export
          </Button>
          <Menu
            anchorEl={exportAnchorEl}
            open={Boolean(exportAnchorEl)}
            onClose={handleExportClose}
          >
            <MenuItem onClick={() => handleExport('pdf')}>Export as PDF</MenuItem>
            <MenuItem onClick={() => handleExport('excel')}>Export as Excel</MenuItem>
            <MenuItem onClick={() => handleExport('json')}>Export as JSON</MenuItem>
          </Menu>

          <IconButton onClick={handleRefresh} disabled={loading} size="small">
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Deadline Alert */}
      {deadline && (
        <Alert
          severity={
            new Date(deadline).getTime() - Date.now() < 7 * 24 * 60 * 60 * 1000
              ? 'warning'
              : 'info'
          }
          sx={{ mb: 2 }}
          icon={false}
        >
          <AlertTitle>Submission Deadline</AlertTitle>
          CDP Climate Change {selectedYear} questionnaire is due by{' '}
          {new Date(deadline).toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
          .
        </Alert>
      )}

      {/* Validation results */}
      {validation && (
        <Alert
          severity={validation.isValid ? 'success' : 'error'}
          sx={{ mb: 2 }}
          onClose={() => dispatch(clearValidation())}
        >
          <AlertTitle>
            {validation.isValid ? 'Validation Passed' : 'Validation Failed'}
          </AlertTitle>
          {validation.isValid
            ? 'All required fields are complete and valid.'
            : `${validation.totalErrors} error${validation.totalErrors !== 1 ? 's' : ''} and ${validation.totalWarnings} warning${validation.totalWarnings !== 1 ? 's' : ''} found. Please review and correct the highlighted fields.`}
        </Alert>
      )}

      {/* Error state */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Main content */}
      {questionnaire ? (
        <>
          <Grid container spacing={3}>
            {/* Left column: Questionnaire Editor */}
            <Grid item xs={12} md={8}>
              <CDPQuestionnaireEditor
                year={selectedYear}
                questionnaire={questionnaire}
                onSave={handleSave}
                onQuestionChange={handleQuestionChange}
                saving={saving}
              />
            </Grid>

            {/* Right column: Progress Tracker */}
            <Grid item xs={12} md={4}>
              {progress && (
                <CDPProgressTracker
                  progress={progress}
                  deadline={deadline}
                />
              )}
            </Grid>
          </Grid>

          {/* Bottom: Data Mapping (collapsible) */}
          <Box sx={{ mt: 3 }}>
            <Divider sx={{ mb: 1 }}>
              <Button
                onClick={toggleMapping}
                endIcon={mappingExpanded ? <ExpandLess /> : <ExpandMore />}
                size="small"
                color="inherit"
              >
                Data Source Mapping ({mappings.length} fields)
              </Button>
            </Divider>
            <Collapse in={mappingExpanded}>
              <CDPDataMapping mappings={mappings} />
            </Collapse>
          </Box>
        </>
      ) : (
        !loading && (
          <Alert severity="info" sx={{ mt: 2 }}>
            No CDP questionnaire data available for {selectedYear}. Click
            "Auto-populate" to initialize from your existing emissions data.
          </Alert>
        )
      )}
    </Box>
  );
};

export default CDPManagement;
