/**
 * CDPQuestionnaireEditor - Full CDP Climate Change questionnaire editor
 *
 * Provides a step-by-step editor for the 13 CDP Climate Change questionnaire
 * sections (C0-C12). Supports auto-populated and manual entry fields with
 * validation status indicators. Integrates with Redux for state management.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepButton,
  Button,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Chip,
  Alert,
  Collapse,
  IconButton,
  Tooltip,
  Divider,
  LinearProgress,
  Paper,
  SelectChangeEvent,
} from '@mui/material';
import {
  NavigateBefore,
  NavigateNext,
  Save,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  HelpOutline,
  AutoFixHigh,
  Edit as EditIcon,
} from '@mui/icons-material';
import type {
  CDPQuestionnaire,
  CDPSection,
  CDPQuestion,
} from '../../store/slices/cdpSlice';

// =============================================================================
// CDP Section Definitions
// =============================================================================

const CDP_SECTION_LABELS: Record<string, string> = {
  C0: 'C0 - Introduction',
  C1: 'C1 - Governance',
  C2: 'C2 - Risks & Opportunities',
  C3: 'C3 - Business Strategy',
  C4: 'C4 - Targets & Performance',
  C5: 'C5 - Emissions Methodology',
  C6: 'C6 - Emissions Data',
  C7: 'C7 - Emissions Breakdown',
  C8: 'C8 - Energy',
  C9: 'C9 - Additional Metrics',
  C10: 'C10 - Verification',
  C11: 'C11 - Carbon Pricing',
  C12: 'C12 - Engagement',
};

const CDP_SECTION_SHORT_LABELS: Record<string, string> = {
  C0: 'C0',
  C1: 'C1',
  C2: 'C2',
  C3: 'C3',
  C4: 'C4',
  C5: 'C5',
  C6: 'C6',
  C7: 'C7',
  C8: 'C8',
  C9: 'C9',
  C10: 'C10',
  C11: 'C11',
  C12: 'C12',
};

// =============================================================================
// Props Interface
// =============================================================================

interface CDPQuestionnaireEditorProps {
  year: number;
  questionnaire: CDPQuestionnaire;
  onSave: (data: CDPQuestionnaire) => void;
  onQuestionChange?: (sectionId: string, questionId: string, value: any) => void;
  saving?: boolean;
}

// =============================================================================
// Sub-Components
// =============================================================================

interface QuestionFieldProps {
  question: CDPQuestion;
  onChange: (questionId: string, value: any) => void;
}

const QuestionField: React.FC<QuestionFieldProps> = React.memo(
  ({ question, onChange }) => {
    const [helpOpen, setHelpOpen] = useState(false);

    const handleTextChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        onChange(question.id, e.target.value);
      },
      [question.id, onChange]
    );

    const handleSelectChange = useCallback(
      (e: SelectChangeEvent<string>) => {
        onChange(question.id, e.target.value);
      },
      [question.id, onChange]
    );

    const handleMultiSelectChange = useCallback(
      (e: SelectChangeEvent<string[]>) => {
        onChange(question.id, e.target.value);
      },
      [question.id, onChange]
    );

    const handleCheckboxChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        onChange(question.id, e.target.checked);
      },
      [question.id, onChange]
    );

    const renderField = () => {
      switch (question.fieldType) {
        case 'text':
          return (
            <TextField
              fullWidth
              label={question.questionText}
              value={question.value || ''}
              onChange={handleTextChange}
              required={question.required}
              error={!!question.validationError}
              helperText={question.validationError}
              size="small"
              InputProps={{
                readOnly: question.autoPopulated,
              }}
              sx={{
                '& .MuiOutlinedInput-root': question.autoPopulated
                  ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                  : {},
              }}
            />
          );

        case 'textarea':
          return (
            <TextField
              fullWidth
              label={question.questionText}
              value={question.value || ''}
              onChange={handleTextChange}
              required={question.required}
              error={!!question.validationError}
              helperText={question.validationError}
              multiline
              rows={4}
              size="small"
              InputProps={{
                readOnly: question.autoPopulated,
              }}
              sx={{
                '& .MuiOutlinedInput-root': question.autoPopulated
                  ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                  : {},
              }}
            />
          );

        case 'number':
          return (
            <TextField
              fullWidth
              label={question.questionText}
              value={question.value ?? ''}
              onChange={handleTextChange}
              required={question.required}
              error={!!question.validationError}
              helperText={question.validationError}
              type="number"
              size="small"
              InputProps={{
                readOnly: question.autoPopulated,
              }}
              sx={{
                '& .MuiOutlinedInput-root': question.autoPopulated
                  ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                  : {},
              }}
            />
          );

        case 'select':
          return (
            <FormControl
              fullWidth
              size="small"
              required={question.required}
              error={!!question.validationError}
            >
              <InputLabel>{question.questionText}</InputLabel>
              <Select
                value={question.value || ''}
                label={question.questionText}
                onChange={handleSelectChange}
                readOnly={question.autoPopulated}
                sx={
                  question.autoPopulated
                    ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                    : {}
                }
              >
                {(question.options || []).map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
              {question.validationError && (
                <Typography variant="caption" color="error" sx={{ mt: 0.5, ml: 1.5 }}>
                  {question.validationError}
                </Typography>
              )}
            </FormControl>
          );

        case 'multiselect':
          return (
            <FormControl
              fullWidth
              size="small"
              required={question.required}
              error={!!question.validationError}
            >
              <InputLabel>{question.questionText}</InputLabel>
              <Select
                multiple
                value={question.value || []}
                label={question.questionText}
                onChange={handleMultiSelectChange}
                readOnly={question.autoPopulated}
                renderValue={(selected: string[]) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((val) => (
                      <Chip key={val} label={val} size="small" />
                    ))}
                  </Box>
                )}
                sx={
                  question.autoPopulated
                    ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                    : {}
                }
              >
                {(question.options || []).map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
              {question.validationError && (
                <Typography variant="caption" color="error" sx={{ mt: 0.5, ml: 1.5 }}>
                  {question.validationError}
                </Typography>
              )}
            </FormControl>
          );

        case 'checkbox':
          return (
            <FormGroup>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={!!question.value}
                    onChange={handleCheckboxChange}
                    disabled={question.autoPopulated}
                  />
                }
                label={
                  <Typography variant="body2">
                    {question.questionText}
                    {question.required && (
                      <Typography component="span" color="error">
                        {' *'}
                      </Typography>
                    )}
                  </Typography>
                }
              />
              {question.validationError && (
                <Typography variant="caption" color="error" sx={{ ml: 4 }}>
                  {question.validationError}
                </Typography>
              )}
            </FormGroup>
          );

        case 'date':
          return (
            <TextField
              fullWidth
              label={question.questionText}
              value={question.value || ''}
              onChange={handleTextChange}
              required={question.required}
              error={!!question.validationError}
              helperText={question.validationError}
              type="date"
              size="small"
              InputLabelProps={{ shrink: true }}
              InputProps={{
                readOnly: question.autoPopulated,
              }}
              sx={{
                '& .MuiOutlinedInput-root': question.autoPopulated
                  ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                  : {},
              }}
            />
          );

        case 'table':
          return (
            <TextField
              fullWidth
              label={question.questionText}
              value={
                typeof question.value === 'object'
                  ? JSON.stringify(question.value, null, 2)
                  : question.value || ''
              }
              onChange={handleTextChange}
              required={question.required}
              error={!!question.validationError}
              helperText={
                question.validationError ||
                'Enter structured data in JSON format'
              }
              multiline
              rows={6}
              size="small"
              InputProps={{
                readOnly: question.autoPopulated,
                sx: { fontFamily: 'monospace', fontSize: '0.85rem' },
              }}
              sx={{
                '& .MuiOutlinedInput-root': question.autoPopulated
                  ? { backgroundColor: 'rgba(46, 125, 50, 0.04)' }
                  : {},
              }}
            />
          );

        default:
          return (
            <TextField
              fullWidth
              label={question.questionText}
              value={question.value || ''}
              onChange={handleTextChange}
              required={question.required}
              size="small"
            />
          );
      }
    };

    return (
      <Paper
        variant="outlined"
        sx={{
          p: 2,
          mb: 2,
          borderLeft: question.validationError
            ? '3px solid'
            : question.autoPopulated
            ? '3px solid'
            : '3px solid',
          borderLeftColor: question.validationError
            ? 'error.main'
            : question.autoPopulated
            ? 'success.main'
            : 'grey.300',
        }}
      >
        {/* Question header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mb: 1.5,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              {question.questionNumber}
            </Typography>
            {question.required && (
              <Typography component="span" color="error" variant="body2">
                *
              </Typography>
            )}
            {question.autoPopulated ? (
              <Chip
                label="Auto-filled"
                size="small"
                color="success"
                variant="outlined"
                icon={<AutoFixHigh />}
                sx={{ height: 24, fontSize: '0.75rem' }}
              />
            ) : question.value ? (
              <Chip
                label="Manual Entry"
                size="small"
                color="warning"
                variant="outlined"
                icon={<EditIcon />}
                sx={{ height: 24, fontSize: '0.75rem' }}
              />
            ) : null}
            {question.dataSource && (
              <Chip
                label={question.dataSource}
                size="small"
                variant="outlined"
                sx={{ height: 24, fontSize: '0.7rem' }}
              />
            )}
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {question.confidence && (
              <Chip
                label={question.confidence}
                size="small"
                color={
                  question.confidence === 'high'
                    ? 'success'
                    : question.confidence === 'medium'
                    ? 'warning'
                    : 'error'
                }
                sx={{ height: 20, fontSize: '0.7rem', textTransform: 'capitalize' }}
              />
            )}
            {question.helpText && (
              <Tooltip title="Toggle help text">
                <IconButton
                  size="small"
                  onClick={() => setHelpOpen(!helpOpen)}
                  color={helpOpen ? 'primary' : 'default'}
                >
                  <HelpOutline fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        {/* Help text */}
        {question.helpText && (
          <Collapse in={helpOpen}>
            <Alert severity="info" sx={{ mb: 1.5, py: 0 }} icon={false}>
              <Typography variant="body2" color="text.secondary">
                {question.helpText}
              </Typography>
            </Alert>
          </Collapse>
        )}

        {/* Question field */}
        {renderField()}
      </Paper>
    );
  }
);

QuestionField.displayName = 'QuestionField';

// =============================================================================
// Validation Status Component
// =============================================================================

interface ValidationStatusProps {
  section: CDPSection;
}

const ValidationStatus: React.FC<ValidationStatusProps> = ({ section }) => {
  const answeredCount = section.questions.filter(
    (q) => q.value !== null && q.value !== undefined && q.value !== ''
  ).length;
  const requiredCount = section.questions.filter((q) => q.required).length;
  const requiredAnswered = section.questions.filter(
    (q) => q.required && q.value !== null && q.value !== undefined && q.value !== ''
  ).length;
  const errorCount = section.questions.filter((q) => q.validationError).length;

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
      <Chip
        label={`${answeredCount}/${section.questions.length} answered`}
        size="small"
        color={
          answeredCount === section.questions.length
            ? 'success'
            : answeredCount > 0
            ? 'warning'
            : 'default'
        }
        variant="outlined"
      />
      <Chip
        label={`${requiredAnswered}/${requiredCount} required`}
        size="small"
        color={requiredAnswered === requiredCount ? 'success' : 'error'}
        variant="outlined"
      />
      {errorCount > 0 && (
        <Chip
          label={`${errorCount} error${errorCount > 1 ? 's' : ''}`}
          size="small"
          color="error"
          icon={<ErrorIcon />}
        />
      )}
      {section.isValid && (
        <CheckCircle color="success" fontSize="small" />
      )}
    </Box>
  );
};

// =============================================================================
// Main Component
// =============================================================================

const CDPQuestionnaireEditor: React.FC<CDPQuestionnaireEditorProps> = ({
  year,
  questionnaire,
  onSave,
  onQuestionChange,
  saving = false,
}) => {
  const [activeStep, setActiveStep] = useState(0);

  const sections = useMemo(() => questionnaire.sections, [questionnaire.sections]);

  const currentSection = useMemo(
    () => sections[activeStep] || null,
    [sections, activeStep]
  );

  // Compute overall validation
  const overallValidation = useMemo(() => {
    const totalQuestions = sections.reduce((sum, s) => sum + s.questions.length, 0);
    const answeredQuestions = sections.reduce(
      (sum, s) =>
        sum +
        s.questions.filter(
          (q) => q.value !== null && q.value !== undefined && q.value !== ''
        ).length,
      0
    );
    const completionPercentage =
      totalQuestions > 0 ? (answeredQuestions / totalQuestions) * 100 : 0;
    const allValid = sections.every((s) => s.isValid);

    return { totalQuestions, answeredQuestions, completionPercentage, allValid };
  }, [sections]);

  // Handlers
  const handleStepClick = useCallback((stepIndex: number) => {
    setActiveStep(stepIndex);
  }, []);

  const handleNext = useCallback(() => {
    setActiveStep((prev) => Math.min(prev + 1, sections.length - 1));
  }, [sections.length]);

  const handleBack = useCallback(() => {
    setActiveStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleQuestionChange = useCallback(
    (questionId: string, value: any) => {
      if (onQuestionChange && currentSection) {
        onQuestionChange(currentSection.id, questionId, value);
      }
    },
    [onQuestionChange, currentSection]
  );

  const handleSave = useCallback(() => {
    onSave(questionnaire);
  }, [onSave, questionnaire]);

  // Step icon logic
  const getStepIcon = useCallback(
    (index: number) => {
      const section = sections[index];
      if (!section) return undefined;

      const hasErrors = section.questions.some((q) => q.validationError);
      const isComplete = section.completionPercentage === 100;

      if (hasErrors) return <ErrorIcon color="error" />;
      if (isComplete) return <CheckCircle color="success" />;
      if (section.completionPercentage > 0) return <Warning color="warning" />;
      return undefined;
    },
    [sections]
  );

  if (!currentSection) {
    return (
      <Alert severity="warning">
        No questionnaire sections available for year {year}.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Box>
          <Typography variant="h6">CDP Climate Change {year}</Typography>
          <Typography variant="body2" color="text.secondary">
            {overallValidation.answeredQuestions} of {overallValidation.totalQuestions}{' '}
            questions answered ({overallValidation.completionPercentage.toFixed(0)}%)
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {questionnaire.lastSavedAt && (
            <Typography variant="caption" color="text.secondary">
              Last saved: {new Date(questionnaire.lastSavedAt).toLocaleString()}
            </Typography>
          )}
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSave}
            disabled={saving}
            size="small"
          >
            {saving ? 'Saving...' : 'Save Draft'}
          </Button>
        </Box>
      </Box>

      {/* Overall progress */}
      <LinearProgress
        variant="determinate"
        value={overallValidation.completionPercentage}
        sx={{
          mb: 2,
          height: 8,
          borderRadius: 4,
          backgroundColor: 'grey.200',
          '& .MuiLinearProgress-bar': {
            borderRadius: 4,
            backgroundColor:
              overallValidation.completionPercentage >= 80
                ? 'success.main'
                : overallValidation.completionPercentage >= 50
                ? 'warning.main'
                : 'error.main',
          },
        }}
      />

      {/* Section Stepper */}
      <Stepper
        nonLinear
        activeStep={activeStep}
        alternativeLabel
        sx={{
          mb: 3,
          '& .MuiStepLabel-label': { fontSize: '0.7rem' },
          overflowX: 'auto',
          flexWrap: 'nowrap',
        }}
      >
        {sections.map((section, index) => (
          <Step key={section.id} completed={section.completionPercentage === 100}>
            <StepButton onClick={() => handleStepClick(index)}>
              <StepLabel
                icon={getStepIcon(index)}
                optional={
                  <Typography variant="caption" color="text.secondary">
                    {section.completionPercentage.toFixed(0)}%
                  </Typography>
                }
              >
                {CDP_SECTION_SHORT_LABELS[section.code] || section.code}
              </StepLabel>
            </StepButton>
          </Step>
        ))}
      </Stepper>

      {/* Current Section Content */}
      <Card variant="outlined">
        <CardContent>
          {/* Section header */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              mb: 2,
            }}
          >
            <Box>
              <Typography variant="h6">
                {CDP_SECTION_LABELS[currentSection.code] || currentSection.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {currentSection.description}
              </Typography>
            </Box>
            <ValidationStatus section={currentSection} />
          </Box>

          <Divider sx={{ mb: 2 }} />

          {/* Section progress */}
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                Section Progress
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {currentSection.completionPercentage.toFixed(0)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={currentSection.completionPercentage}
              sx={{
                height: 4,
                borderRadius: 2,
                backgroundColor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 2,
                  backgroundColor:
                    currentSection.completionPercentage >= 80
                      ? 'success.main'
                      : currentSection.completionPercentage >= 50
                      ? 'warning.main'
                      : 'error.main',
                },
              }}
            />
          </Box>

          {/* Questions */}
          {currentSection.questions.length === 0 ? (
            <Alert severity="info">
              No questions configured for this section yet.
            </Alert>
          ) : (
            currentSection.questions.map((question) => (
              <QuestionField
                key={question.id}
                question={question}
                onChange={handleQuestionChange}
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Navigation */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mt: 2,
        }}
      >
        <Button
          variant="outlined"
          startIcon={<NavigateBefore />}
          onClick={handleBack}
          disabled={activeStep === 0}
        >
          Previous
        </Button>

        <Typography variant="body2" color="text.secondary">
          Section {activeStep + 1} of {sections.length}
        </Typography>

        <Button
          variant="outlined"
          endIcon={<NavigateNext />}
          onClick={handleNext}
          disabled={activeStep === sections.length - 1}
        >
          Next
        </Button>
      </Box>
    </Box>
  );
};

export default CDPQuestionnaireEditor;
