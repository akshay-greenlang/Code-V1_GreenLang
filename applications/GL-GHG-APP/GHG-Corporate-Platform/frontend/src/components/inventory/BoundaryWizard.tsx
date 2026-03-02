/**
 * BoundaryWizard - Multi-step wizard for GHG inventory setup
 *
 * A 5-step MUI Stepper that guides the user through:
 *   Step 1: Organization Info (name, industry, country)
 *   Step 2: Consolidation Approach (operational / financial / equity share)
 *   Step 3: Scope Selection (Scope 1, 2, 3 with descriptions)
 *   Step 4: Base Year (year picker, justification)
 *   Step 5: Review and Create (summary of all selections)
 *
 * Validates each step before allowing the user to proceed.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  TextField,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Checkbox,
  FormGroup,
  Select,
  MenuItem,
  InputLabel,
  Card,
  CardContent,
  Alert,
  Divider,
  Grid,
} from '@mui/material';
import { ConsolidationApproach, Scope3Category, SCOPE3_CATEGORY_NAMES } from '../../types';
import { isRequired, isValidYear } from '../../utils/validators';

const STEPS = [
  'Organization Info',
  'Consolidation Approach',
  'Scope Selection',
  'Base Year',
  'Review & Create',
];

const INDUSTRIES = [
  'Energy', 'Materials', 'Industrials', 'Consumer Discretionary',
  'Consumer Staples', 'Health Care', 'Financials', 'Information Technology',
  'Communication Services', 'Utilities', 'Real Estate', 'Other',
];

const COUNTRIES = [
  'United States', 'United Kingdom', 'Germany', 'France', 'Netherlands',
  'Japan', 'Canada', 'Australia', 'India', 'China', 'Brazil', 'Other',
];

interface WizardData {
  orgName: string;
  industry: string;
  country: string;
  consolidation: ConsolidationApproach;
  scope1: boolean;
  scope2: boolean;
  scope3: boolean;
  scope3Categories: Scope3Category[];
  baseYear: number;
  justification: string;
}

interface BoundaryWizardProps {
  onComplete: (data: WizardData) => void;
}

const BoundaryWizard: React.FC<BoundaryWizardProps> = ({ onComplete }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [data, setData] = useState<WizardData>({
    orgName: '',
    industry: '',
    country: '',
    consolidation: ConsolidationApproach.OPERATIONAL_CONTROL,
    scope1: true,
    scope2: true,
    scope3: false,
    scope3Categories: [],
    baseYear: new Date().getFullYear() - 1,
    justification: '',
  });

  const updateField = useCallback(
    <K extends keyof WizardData>(field: K, value: WizardData[K]) => {
      setData((prev) => ({ ...prev, [field]: value }));
      setErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    },
    []
  );

  const validateStep = (): boolean => {
    const errs: Record<string, string> = {};

    if (activeStep === 0) {
      const nameErr = isRequired(data.orgName);
      if (nameErr) errs.orgName = nameErr;
      if (!data.industry) errs.industry = 'Please select an industry';
      if (!data.country) errs.country = 'Please select a country';
    }

    if (activeStep === 3) {
      const yearErr = isValidYear(data.baseYear);
      if (yearErr) errs.baseYear = yearErr;
      const justErr = isRequired(data.justification);
      if (justErr) errs.justification = 'Base year justification is required';
    }

    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleNext = () => {
    if (!validateStep()) return;
    if (activeStep === STEPS.length - 1) {
      onComplete(data);
    } else {
      setActiveStep((s) => s + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((s) => s - 1);
  };

  const toggleScope3Cat = (cat: Scope3Category) => {
    setData((prev) => {
      const cats = prev.scope3Categories.includes(cat)
        ? prev.scope3Categories.filter((c) => c !== cat)
        : [...prev.scope3Categories, cat];
      return { ...prev, scope3Categories: cats };
    });
  };

  /* ---- Step renderers ---- */

  const renderStep0 = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Organization Name"
          value={data.orgName}
          onChange={(e) => updateField('orgName', e.target.value)}
          error={!!errors.orgName}
          helperText={errors.orgName}
          required
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <FormControl fullWidth error={!!errors.industry}>
          <InputLabel>Industry</InputLabel>
          <Select
            value={data.industry}
            label="Industry"
            onChange={(e) => updateField('industry', e.target.value)}
          >
            {INDUSTRIES.map((ind) => (
              <MenuItem key={ind} value={ind}>{ind}</MenuItem>
            ))}
          </Select>
          {errors.industry && (
            <Typography variant="caption" color="error" sx={{ ml: 1.75, mt: 0.5 }}>
              {errors.industry}
            </Typography>
          )}
        </FormControl>
      </Grid>
      <Grid item xs={12} sm={6}>
        <FormControl fullWidth error={!!errors.country}>
          <InputLabel>Country</InputLabel>
          <Select
            value={data.country}
            label="Country"
            onChange={(e) => updateField('country', e.target.value)}
          >
            {COUNTRIES.map((c) => (
              <MenuItem key={c} value={c}>{c}</MenuItem>
            ))}
          </Select>
          {errors.country && (
            <Typography variant="caption" color="error" sx={{ ml: 1.75, mt: 0.5 }}>
              {errors.country}
            </Typography>
          )}
        </FormControl>
      </Grid>
    </Grid>
  );

  const renderStep1 = () => (
    <FormControl component="fieldset">
      <FormLabel component="legend" sx={{ mb: 2 }}>
        Select the consolidation approach for your GHG inventory
      </FormLabel>
      <RadioGroup
        value={data.consolidation}
        onChange={(e) => updateField('consolidation', e.target.value as ConsolidationApproach)}
      >
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <FormControlLabel
              value={ConsolidationApproach.OPERATIONAL_CONTROL}
              control={<Radio />}
              label={
                <Box>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    Operational Control
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Account for 100% of emissions from operations over which the company
                    has operational control. Most common for corporate reporting.
                  </Typography>
                </Box>
              }
            />
          </CardContent>
        </Card>
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <FormControlLabel
              value={ConsolidationApproach.FINANCIAL_CONTROL}
              control={<Radio />}
              label={
                <Box>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    Financial Control
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Account for 100% of emissions from operations over which the company
                    has financial control (can direct financial and operating policies).
                  </Typography>
                </Box>
              }
            />
          </CardContent>
        </Card>
        <Card variant="outlined">
          <CardContent>
            <FormControlLabel
              value={ConsolidationApproach.EQUITY_SHARE}
              control={<Radio />}
              label={
                <Box>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    Equity Share
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Account for emissions based on the company's equity share in each
                    operation. Aligns with financial accounting for equity investments.
                  </Typography>
                </Box>
              }
            />
          </CardContent>
        </Card>
      </RadioGroup>
    </FormControl>
  );

  const renderStep2 = () => (
    <Box>
      <FormGroup>
        <FormControlLabel
          control={<Checkbox checked={data.scope1} onChange={(e) => updateField('scope1', e.target.checked)} />}
          label={
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Scope 1 - Direct Emissions
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Emissions from sources owned or controlled by the company (stationary combustion,
                mobile combustion, process emissions, fugitive emissions).
              </Typography>
            </Box>
          }
        />
        <FormControlLabel
          control={<Checkbox checked={data.scope2} onChange={(e) => updateField('scope2', e.target.checked)} />}
          label={
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Scope 2 - Indirect Emissions (Energy)
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Emissions from purchased electricity, steam, heating, and cooling.
                Dual reporting of location-based and market-based methods required.
              </Typography>
            </Box>
          }
        />
        <FormControlLabel
          control={<Checkbox checked={data.scope3} onChange={(e) => updateField('scope3', e.target.checked)} />}
          label={
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Scope 3 - Other Indirect Emissions (Value Chain)
              </Typography>
              <Typography variant="body2" color="text.secondary">
                All other indirect emissions across 15 categories in the company's
                value chain (upstream and downstream).
              </Typography>
            </Box>
          }
        />
      </FormGroup>

      {data.scope3 && (
        <Box sx={{ mt: 3, pl: 4 }}>
          <Typography variant="subtitle2" gutterBottom>
            Select Scope 3 categories to include:
          </Typography>
          <Grid container spacing={1}>
            {Object.entries(SCOPE3_CATEGORY_NAMES).map(([key, name]) => (
              <Grid item xs={12} sm={6} key={key}>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={data.scope3Categories.includes(key as Scope3Category)}
                      onChange={() => toggleScope3Cat(key as Scope3Category)}
                    />
                  }
                  label={
                    <Typography variant="body2">
                      Cat {key.replace('cat_', '')}: {name}
                    </Typography>
                  }
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );

  const renderStep3 = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Base Year"
          type="number"
          value={data.baseYear}
          onChange={(e) => updateField('baseYear', parseInt(e.target.value, 10))}
          error={!!errors.baseYear}
          helperText={errors.baseYear || 'The reference year against which future emissions are compared'}
          required
        />
      </Grid>
      <Grid item xs={12}>
        <TextField
          fullWidth
          multiline
          rows={4}
          label="Base Year Justification"
          value={data.justification}
          onChange={(e) => updateField('justification', e.target.value)}
          error={!!errors.justification}
          helperText={
            errors.justification ||
            'Explain why this year was chosen (e.g., earliest year with reliable data, no significant structural changes)'
          }
          required
        />
      </Grid>
    </Grid>
  );

  const renderStep4 = () => {
    const approachLabel =
      data.consolidation === ConsolidationApproach.OPERATIONAL_CONTROL
        ? 'Operational Control'
        : data.consolidation === ConsolidationApproach.FINANCIAL_CONTROL
          ? 'Financial Control'
          : 'Equity Share';

    const selectedScopes = [
      data.scope1 && 'Scope 1',
      data.scope2 && 'Scope 2',
      data.scope3 && 'Scope 3',
    ].filter(Boolean);

    return (
      <Box>
        <Alert severity="info" sx={{ mb: 3 }}>
          Review your selections before creating the inventory. You can go back to make changes.
        </Alert>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Organization</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{data.orgName}</Typography>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Typography variant="body2" color="text.secondary">Industry</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{data.industry}</Typography>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Typography variant="body2" color="text.secondary">Country</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{data.country}</Typography>
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Consolidation Approach</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{approachLabel}</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="body2" color="text.secondary">Scopes Included</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              {selectedScopes.join(', ')}
            </Typography>
          </Grid>
        </Grid>

        {data.scope3 && data.scope3Categories.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Scope 3 Categories ({data.scope3Categories.length})
            </Typography>
            <Typography variant="body2">
              {data.scope3Categories
                .map((c) => `Cat ${c.replace('cat_', '')}: ${SCOPE3_CATEGORY_NAMES[c]}`)
                .join(', ')}
            </Typography>
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <Typography variant="body2" color="text.secondary">Base Year</Typography>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{data.baseYear}</Typography>
          </Grid>
          <Grid item xs={12} sm={8}>
            <Typography variant="body2" color="text.secondary">Justification</Typography>
            <Typography variant="body2">{data.justification}</Typography>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const stepContent = [renderStep0, renderStep1, renderStep2, renderStep3, renderStep4];

  return (
    <Box>
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box sx={{ minHeight: 300, mb: 3 }}>{stepContent[activeStep]()}</Box>

      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button disabled={activeStep === 0} onClick={handleBack} variant="outlined">
          Back
        </Button>
        <Button variant="contained" onClick={handleNext}>
          {activeStep === STEPS.length - 1 ? 'Create Inventory' : 'Next'}
        </Button>
      </Box>
    </Box>
  );
};

export default BoundaryWizard;
