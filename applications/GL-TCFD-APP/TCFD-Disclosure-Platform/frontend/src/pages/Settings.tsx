/**
 * Settings - Organization settings, scenario preferences, regulatory jurisdictions, data source configuration.
 */

import React, { useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, TextField, Button,
  Select, MenuItem, FormControl, InputLabel, SelectChangeEvent,
  Switch, FormControlLabel, Divider, Chip, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  List, ListItem, ListItemText, ListItemSecondaryAction, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions,
} from '@mui/material';
import {
  Save, Business, Settings as SettingsIcon, Security, Storage,
  Add, Delete, Edit, CheckCircle, Warning,
} from '@mui/icons-material';

/* ── Demo Data ────────────────────────────────────────────────── */

interface OrgSettings {
  name: string;
  industry: string;
  sector: string;
  size: string;
  reportingYear: string;
  baseYear: string;
  currency: string;
  emissionsUnit: string;
}

interface DataSource {
  id: string;
  name: string;
  type: string;
  status: 'connected' | 'disconnected' | 'error';
  lastSync: string;
  recordCount: number;
}

const DEFAULT_ORG: OrgSettings = {
  name: 'GreenLang Corp',
  industry: 'Technology',
  sector: 'Software & Services',
  size: 'Large (>250 employees)',
  reportingYear: '2024',
  baseYear: '2019',
  currency: 'USD',
  emissionsUnit: 'tCO2e',
};

const DATA_SOURCES: DataSource[] = [
  { id: '1', name: 'GHG Protocol Calculator', type: 'Emissions', status: 'connected', lastSync: '2025-02-15T10:30:00', recordCount: 12450 },
  { id: '2', name: 'SAP ERP System', type: 'Financial', status: 'connected', lastSync: '2025-02-14T22:00:00', recordCount: 85200 },
  { id: '3', name: 'Asset Management DB', type: 'Assets', status: 'connected', lastSync: '2025-02-15T08:00:00', recordCount: 342 },
  { id: '4', name: 'Climate Risk API', type: 'Risk Data', status: 'error', lastSync: '2025-02-10T14:30:00', recordCount: 0 },
  { id: '5', name: 'Supply Chain Platform', type: 'Supply Chain', status: 'disconnected', lastSync: '', recordCount: 0 },
];

const JURISDICTIONS = [
  { code: 'EU', name: 'European Union', frameworks: ['CSRD', 'EU Taxonomy', 'CBAM', 'EU ETS'], enabled: true },
  { code: 'UK', name: 'United Kingdom', frameworks: ['UK Climate Disclosure', 'UK ETS', 'TCFD (FCA)'], enabled: true },
  { code: 'US', name: 'United States', frameworks: ['SEC Climate Disclosure', 'EPA GHG Reporting'], enabled: true },
  { code: 'JP', name: 'Japan', frameworks: ['ISSB Adoption', 'GX Carbon Levy'], enabled: false },
  { code: 'AU', name: 'Australia', frameworks: ['AASB Climate Disclosure', 'ISSB Adoption'], enabled: false },
  { code: 'SG', name: 'Singapore', frameworks: ['SGX Climate Reporting', 'MAS Guidelines'], enabled: true },
];

const SCENARIO_PRESETS = [
  { id: 'nze', name: 'IEA Net Zero 2050', source: 'IEA WEO 2024', warmingTarget: '1.5C', carbonPrice2030: 130, carbonPrice2050: 250, enabled: true },
  { id: 'aps', name: 'IEA Announced Pledges', source: 'IEA WEO 2024', warmingTarget: '1.7C', carbonPrice2030: 90, carbonPrice2050: 175, enabled: true },
  { id: 'steps', name: 'IEA STEPS', source: 'IEA WEO 2024', warmingTarget: '2.4C', carbonPrice2030: 45, carbonPrice2050: 65, enabled: true },
  { id: 'delayed', name: 'Delayed Transition', source: 'NGFS', warmingTarget: '1.8C', carbonPrice2030: 30, carbonPrice2050: 350, enabled: false },
  { id: 'hothouse', name: 'Hot House World', source: 'NGFS', warmingTarget: '>3C', carbonPrice2030: 10, carbonPrice2050: 15, enabled: false },
];

/* ── Component ─────────────────────────────────────────────────── */

const Settings: React.FC = () => {
  const [org, setOrg] = useState<OrgSettings>(DEFAULT_ORG);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [dataSources, setDataSources] = useState(DATA_SOURCES);
  const [jurisdictions, setJurisdictions] = useState(JURISDICTIONS);
  const [scenarios, setScenarios] = useState(SCENARIO_PRESETS);
  const [addSourceOpen, setAddSourceOpen] = useState(false);

  const handleOrgChange = (field: keyof OrgSettings, value: string) => {
    setOrg((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = () => {
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const handleJurisdictionToggle = (code: string) => {
    setJurisdictions((prev) =>
      prev.map((j) => (j.code === code ? { ...j, enabled: !j.enabled } : j))
    );
  };

  const handleScenarioToggle = (id: string) => {
    setScenarios((prev) =>
      prev.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s))
    );
  };

  const getStatusColor = (status: string): 'success' | 'error' | 'default' => {
    if (status === 'connected') return 'success';
    if (status === 'error') return 'error';
    return 'default';
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Settings</Typography>
        <Typography variant="body2" color="text.secondary">
          Configure organization, scenarios, jurisdictions, and data sources
        </Typography>
      </Box>

      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSaveSuccess(false)}>
          Settings saved successfully.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Organization Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Business color="primary" />
                <Typography variant="h6">Organization Settings</Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Organization Name"
                    value={org.name}
                    onChange={(e) => handleOrgChange('name', e.target.value)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Industry</InputLabel>
                    <Select
                      value={org.industry}
                      label="Industry"
                      onChange={(e: SelectChangeEvent) => handleOrgChange('industry', e.target.value)}
                    >
                      <MenuItem value="Technology">Technology</MenuItem>
                      <MenuItem value="Manufacturing">Manufacturing</MenuItem>
                      <MenuItem value="Energy">Energy</MenuItem>
                      <MenuItem value="Financial Services">Financial Services</MenuItem>
                      <MenuItem value="Healthcare">Healthcare</MenuItem>
                      <MenuItem value="Real Estate">Real Estate</MenuItem>
                      <MenuItem value="Transportation">Transportation</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Sector"
                    value={org.sector}
                    onChange={(e) => handleOrgChange('sector', e.target.value)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Size</InputLabel>
                    <Select
                      value={org.size}
                      label="Size"
                      onChange={(e: SelectChangeEvent) => handleOrgChange('size', e.target.value)}
                    >
                      <MenuItem value="Small (<50 employees)">Small (&lt;50 employees)</MenuItem>
                      <MenuItem value="Medium (50-249 employees)">Medium (50-249 employees)</MenuItem>
                      <MenuItem value="Large (>250 employees)">Large (&gt;250 employees)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Reporting Year</InputLabel>
                    <Select
                      value={org.reportingYear}
                      label="Reporting Year"
                      onChange={(e: SelectChangeEvent) => handleOrgChange('reportingYear', e.target.value)}
                    >
                      {['2020', '2021', '2022', '2023', '2024', '2025'].map((y) => (
                        <MenuItem key={y} value={y}>{y}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Base Year</InputLabel>
                    <Select
                      value={org.baseYear}
                      label="Base Year"
                      onChange={(e: SelectChangeEvent) => handleOrgChange('baseYear', e.target.value)}
                    >
                      {['2015', '2016', '2017', '2018', '2019', '2020', '2021'].map((y) => (
                        <MenuItem key={y} value={y}>{y}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Currency</InputLabel>
                    <Select
                      value={org.currency}
                      label="Currency"
                      onChange={(e: SelectChangeEvent) => handleOrgChange('currency', e.target.value)}
                    >
                      <MenuItem value="USD">USD</MenuItem>
                      <MenuItem value="EUR">EUR</MenuItem>
                      <MenuItem value="GBP">GBP</MenuItem>
                      <MenuItem value="JPY">JPY</MenuItem>
                      <MenuItem value="AUD">AUD</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Units</InputLabel>
                    <Select
                      value={org.emissionsUnit}
                      label="Units"
                      onChange={(e: SelectChangeEvent) => handleOrgChange('emissionsUnit', e.target.value)}
                    >
                      <MenuItem value="tCO2e">tCO2e</MenuItem>
                      <MenuItem value="kgCO2e">kgCO2e</MenuItem>
                      <MenuItem value="mtCO2e">mtCO2e</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <Button variant="contained" startIcon={<Save />} onClick={handleSave}>
                  Save Organization
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Regulatory Jurisdictions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Security color="primary" />
                <Typography variant="h6">Regulatory Jurisdictions</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Enable the jurisdictions relevant to your organization. This determines which regulatory frameworks are tracked.
              </Typography>
              <List dense>
                {jurisdictions.map((j) => (
                  <React.Fragment key={j.code}>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>{j.name}</Typography>
                            <Chip label={j.code} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 18 }} />
                          </Box>
                        }
                        secondary={
                          <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
                            {j.frameworks.map((fw) => (
                              <Chip key={fw} label={fw} size="small" sx={{ fontSize: '0.6rem', height: 18, backgroundColor: j.enabled ? '#E8F5E9' : '#F5F5F5' }} />
                            ))}
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <Switch
                          edge="end"
                          checked={j.enabled}
                          onChange={() => handleJurisdictionToggle(j.code)}
                          color="success"
                        />
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider component="li" />
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Scenario Preferences */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <SettingsIcon color="primary" />
                <Typography variant="h6">Scenario Preferences</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Configure which climate scenarios are used for analysis and reporting.
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Scenario</TableCell>
                      <TableCell>Source</TableCell>
                      <TableCell align="center">Warming</TableCell>
                      <TableCell align="center">C-Price 2030</TableCell>
                      <TableCell align="center">C-Price 2050</TableCell>
                      <TableCell align="center">Enabled</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {scenarios.map((s) => (
                      <TableRow key={s.id} hover sx={{ opacity: s.enabled ? 1 : 0.5 }}>
                        <TableCell sx={{ fontWeight: 500 }}>{s.name}</TableCell>
                        <TableCell sx={{ fontSize: '0.8rem' }}>{s.source}</TableCell>
                        <TableCell align="center">
                          <Chip
                            label={s.warmingTarget}
                            size="small"
                            sx={{
                              backgroundColor: s.warmingTarget === '1.5C' ? '#E8F5E9' : s.warmingTarget.includes('2') ? '#FFF9C4' : '#FFCDD2',
                              fontWeight: 600,
                              fontSize: '0.7rem',
                            }}
                          />
                        </TableCell>
                        <TableCell align="center">${s.carbonPrice2030}</TableCell>
                        <TableCell align="center">${s.carbonPrice2050}</TableCell>
                        <TableCell align="center">
                          <Switch
                            size="small"
                            checked={s.enabled}
                            onChange={() => handleScenarioToggle(s.id)}
                            color="success"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Sources */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Storage color="primary" />
                  <Typography variant="h6">Data Sources</Typography>
                </Box>
                <Button variant="outlined" size="small" startIcon={<Add />} onClick={() => setAddSourceOpen(true)}>
                  Add Source
                </Button>
              </Box>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Source</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell align="center">Status</TableCell>
                      <TableCell>Last Sync</TableCell>
                      <TableCell align="right">Records</TableCell>
                      <TableCell align="center">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {dataSources.map((source) => (
                      <TableRow key={source.id} hover>
                        <TableCell sx={{ fontWeight: 500 }}>{source.name}</TableCell>
                        <TableCell>
                          <Chip label={source.type} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={source.status}
                            size="small"
                            color={getStatusColor(source.status)}
                            sx={{ textTransform: 'capitalize' }}
                            icon={source.status === 'connected' ? <CheckCircle /> : source.status === 'error' ? <Warning /> : undefined}
                          />
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.8rem' }}>
                          {source.lastSync ? new Date(source.lastSync).toLocaleString() : '--'}
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: 500 }}>
                          {source.recordCount > 0 ? source.recordCount.toLocaleString() : '--'}
                        </TableCell>
                        <TableCell align="center">
                          <IconButton size="small">
                            <Edit fontSize="small" />
                          </IconButton>
                          <IconButton size="small" color="error">
                            <Delete fontSize="small" />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Add Data Source Dialog */}
      <Dialog open={addSourceOpen} onClose={() => setAddSourceOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Data Source</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid item xs={12}>
              <TextField fullWidth label="Source Name" size="small" />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Type</InputLabel>
                <Select label="Type" defaultValue="">
                  <MenuItem value="emissions">Emissions</MenuItem>
                  <MenuItem value="financial">Financial</MenuItem>
                  <MenuItem value="assets">Assets</MenuItem>
                  <MenuItem value="risk_data">Risk Data</MenuItem>
                  <MenuItem value="supply_chain">Supply Chain</MenuItem>
                  <MenuItem value="weather">Weather/Climate</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Connection Type</InputLabel>
                <Select label="Connection Type" defaultValue="">
                  <MenuItem value="api">REST API</MenuItem>
                  <MenuItem value="database">Database</MenuItem>
                  <MenuItem value="file_upload">File Upload</MenuItem>
                  <MenuItem value="sftp">SFTP</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField fullWidth label="Connection URL / Path" size="small" />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField fullWidth label="API Key / Username" size="small" />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField fullWidth label="Secret / Password" type="password" size="small" />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddSourceOpen(false)}>Cancel</Button>
          <Button variant="outlined" onClick={() => setAddSourceOpen(false)}>Test Connection</Button>
          <Button variant="contained" onClick={() => setAddSourceOpen(false)}>Add Source</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Settings;
