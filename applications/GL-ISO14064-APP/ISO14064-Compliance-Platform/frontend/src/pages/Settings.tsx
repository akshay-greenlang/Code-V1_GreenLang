/**
 * Settings Page - Platform configuration
 *
 * Provides settings for GWP source, base year, reporting period,
 * significance threshold, and notification preferences.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  TextField,
  MenuItem,
  Button,
  Divider,
  Switch,
  FormControlLabel,
  Alert,
  Snackbar,
} from '@mui/material';
import { Save } from '@mui/icons-material';
import {
  GWPSource,
  ConsolidationApproach,
  ReportingPeriod,
} from '../types';

const Settings: React.FC = () => {
  const [gwpSource, setGwpSource] = useState<GWPSource>(GWPSource.AR6);
  const [baseYear, setBaseYear] = useState(2019);
  const [reportingPeriod, setReportingPeriod] = useState<ReportingPeriod>(
    ReportingPeriod.CALENDAR_YEAR,
  );
  const [significanceThreshold, setSignificanceThreshold] = useState(50);
  const [recalcThreshold, setRecalcThreshold] = useState(5);
  const [consolidation, setConsolidation] = useState<ConsolidationApproach>(
    ConsolidationApproach.OPERATIONAL_CONTROL,
  );
  const [autoSignificance, setAutoSignificance] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [uncertaintyIterations, setUncertaintyIterations] = useState(10000);
  const [snackOpen, setSnackOpen] = useState(false);

  const handleSave = () => {
    // Would dispatch save settings action
    setSnackOpen(true);
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Settings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure ISO 14064-1 platform parameters and preferences
      </Typography>

      <Grid container spacing={3}>
        {/* Quantification Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Quantification Settings" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    select
                    label="GWP Source"
                    value={gwpSource}
                    onChange={(e) => setGwpSource(e.target.value as GWPSource)}
                    helperText="Select the IPCC Assessment Report for GWP values"
                  >
                    <MenuItem value={GWPSource.AR5}>IPCC AR5 (2014)</MenuItem>
                    <MenuItem value={GWPSource.AR6}>IPCC AR6 (2021)</MenuItem>
                    <MenuItem value={GWPSource.CUSTOM}>Custom</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    select
                    label="Default Consolidation Approach"
                    value={consolidation}
                    onChange={(e) =>
                      setConsolidation(e.target.value as ConsolidationApproach)
                    }
                  >
                    {Object.values(ConsolidationApproach).map((a) => (
                      <MenuItem key={a} value={a}>
                        {a.replace(/_/g, ' ')}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Uncertainty Monte Carlo Iterations"
                    value={uncertaintyIterations}
                    onChange={(e) => setUncertaintyIterations(Number(e.target.value))}
                    helperText="Number of simulation iterations (1,000 - 100,000)"
                    inputProps={{ min: 1000, max: 100000, step: 1000 }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Reporting Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Reporting Configuration" />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Base Year"
                    value={baseYear}
                    onChange={(e) => setBaseYear(Number(e.target.value))}
                    helperText="ISO 14064-1 Clause 5.3"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    select
                    label="Reporting Period"
                    value={reportingPeriod}
                    onChange={(e) =>
                      setReportingPeriod(e.target.value as ReportingPeriod)
                    }
                  >
                    <MenuItem value={ReportingPeriod.CALENDAR_YEAR}>
                      Calendar Year
                    </MenuItem>
                    <MenuItem value={ReportingPeriod.FISCAL_YEAR}>
                      Fiscal Year
                    </MenuItem>
                    <MenuItem value={ReportingPeriod.CUSTOM}>Custom</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Significance Threshold"
                    value={significanceThreshold}
                    onChange={(e) =>
                      setSignificanceThreshold(Number(e.target.value))
                    }
                    helperText="Score above which a category is significant"
                    inputProps={{ min: 0, max: 100 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Recalculation Threshold (%)"
                    value={recalcThreshold}
                    onChange={(e) => setRecalcThreshold(Number(e.target.value))}
                    helperText="Base year recalculation trigger threshold"
                    inputProps={{ min: 0, max: 25 }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Preferences */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader title="Preferences" />
            <CardContent>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoSignificance}
                    onChange={(e) => setAutoSignificance(e.target.checked)}
                    color="success"
                  />
                }
                label="Auto-assess significance for new indirect categories"
              />
              <Divider sx={{ my: 1.5 }} />
              <FormControlLabel
                control={
                  <Switch
                    checked={emailNotifications}
                    onChange={(e) => setEmailNotifications(e.target.checked)}
                    color="success"
                  />
                }
                label="Email notifications for verification stage changes"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Save */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={handleSave}
              sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
            >
              Save Settings
            </Button>
          </Box>
        </Grid>
      </Grid>

      <Snackbar
        open={snackOpen}
        autoHideDuration={3000}
        onClose={() => setSnackOpen(false)}
        message="Settings saved successfully"
      />
    </Box>
  );
};

export default Settings;
