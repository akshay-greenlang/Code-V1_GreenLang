/**
 * Settings - Organization settings, SBTi preferences, data source configuration, and notification settings.
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
  Add, Delete, Edit, CheckCircle, Warning, Notifications,
} from '@mui/icons-material';

/* Demo Data */
interface OrgSettings {
  name: string;
  industry: string;
  sector: string;
  size: string;
  baseYear: string;
  reportingYear: string;
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
  baseYear: '2019',
  reportingYear: '2024',
  currency: 'USD',
  emissionsUnit: 'tCO2e',
};

const DATA_SOURCES: DataSource[] = [
  { id: '1', name: 'GHG Protocol Calculator', type: 'Emissions', status: 'connected', lastSync: '2025-02-28T10:30:00', recordCount: 15200 },
  { id: '2', name: 'SAP ERP System', type: 'Financial', status: 'connected', lastSync: '2025-02-27T22:00:00', recordCount: 92400 },
  { id: '3', name: 'SBTi Target Registry', type: 'Targets', status: 'connected', lastSync: '2025-02-28T08:00:00', recordCount: 6 },
  { id: '4', name: 'CDP Questionnaire Platform', type: 'Reporting', status: 'disconnected', lastSync: '', recordCount: 0 },
];

const SBTI_PREFERENCES = [
  { id: 'alignment', label: 'Default Alignment Level', value: '1.5C', options: ['1.5C', 'WB2C', '2C'] },
  { id: 'method', label: 'Default Pathway Method', value: 'ACA', options: ['ACA', 'SDA'] },
  { id: 'recalc_threshold', label: 'Recalculation Threshold (%)', value: '5', options: ['1', '2', '5', '10'] },
  { id: 'review_cycle', label: 'Review Cycle (years)', value: '5', options: ['5'] },
];

const NOTIFICATION_SETTINGS = [
  { id: 'target_expiry', label: 'Target expiry reminders', description: '90, 60, 30, 7 days before target expiry', enabled: true },
  { id: 'progress_due', label: 'Annual progress data collection', description: 'Reminder when annual progress data is due', enabled: true },
  { id: 'recalc_trigger', label: 'Recalculation trigger alerts', description: 'Alert when structural changes exceed threshold', enabled: true },
  { id: 'review_countdown', label: '5-year review countdown', description: 'Monthly reminders as review date approaches', enabled: false },
  { id: 'pathway_deviation', label: 'Pathway deviation alerts', description: 'Alert when actual emissions deviate from pathway', enabled: true },
  { id: 'framework_updates', label: 'Framework update notifications', description: 'Alert when SBTi criteria or methods are updated', enabled: true },
];

const Settings: React.FC = () => {
  const [org, setOrg] = useState<OrgSettings>(DEFAULT_ORG);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [dataSources, setDataSources] = useState(DATA_SOURCES);
  const [sbtiPrefs, setSbtiPrefs] = useState(SBTI_PREFERENCES);
  const [notifications, setNotifications] = useState(NOTIFICATION_SETTINGS);
  const [addSourceOpen, setAddSourceOpen] = useState(false);

  const handleOrgChange = (field: keyof OrgSettings, value: string) => {
    setOrg((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = () => {
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const handlePrefChange = (id: string, value: string) => {
    setSbtiPrefs((prev) => prev.map((p) => (p.id === id ? { ...p, value } : p)));
  };

  const handleNotifToggle = (id: string) => {
    setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, enabled: !n.enabled } : n)));
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
          Configure organization, SBTi preferences, data sources, and notifications
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
                  <TextField fullWidth label="Organization Name" value={org.name} onChange={(e) => handleOrgChange('name', e.target.value)} size="small" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Industry</InputLabel>
                    <Select value={org.industry} label="Industry" onChange={(e: SelectChangeEvent) => handleOrgChange('industry', e.target.value)}>
                      {['Technology', 'Manufacturing', 'Energy', 'Financial Services', 'Healthcare', 'Real Estate', 'Transportation', 'Agriculture'].map((i) => (
                        <MenuItem key={i} value={i}>{i}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField fullWidth label="Sector" value={org.sector} onChange={(e) => handleOrgChange('sector', e.target.value)} size="small" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Size</InputLabel>
                    <Select value={org.size} label="Size" onChange={(e: SelectChangeEvent) => handleOrgChange('size', e.target.value)}>
                      <MenuItem value="Small (<50 employees)">Small (&lt;50 employees)</MenuItem>
                      <MenuItem value="Medium (50-249 employees)">Medium (50-249 employees)</MenuItem>
                      <MenuItem value="Large (>250 employees)">Large (&gt;250 employees)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Base Year</InputLabel>
                    <Select value={org.baseYear} label="Base Year" onChange={(e: SelectChangeEvent) => handleOrgChange('baseYear', e.target.value)}>
                      {['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'].map((y) => (
                        <MenuItem key={y} value={y}>{y}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Reporting Year</InputLabel>
                    <Select value={org.reportingYear} label="Reporting Year" onChange={(e: SelectChangeEvent) => handleOrgChange('reportingYear', e.target.value)}>
                      {['2020', '2021', '2022', '2023', '2024', '2025'].map((y) => (
                        <MenuItem key={y} value={y}>{y}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Currency</InputLabel>
                    <Select value={org.currency} label="Currency" onChange={(e: SelectChangeEvent) => handleOrgChange('currency', e.target.value)}>
                      {['USD', 'EUR', 'GBP', 'JPY', 'AUD'].map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Units</InputLabel>
                    <Select value={org.emissionsUnit} label="Units" onChange={(e: SelectChangeEvent) => handleOrgChange('emissionsUnit', e.target.value)}>
                      {['tCO2e', 'kgCO2e', 'mtCO2e'].map((u) => <MenuItem key={u} value={u}>{u}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <Button variant="contained" startIcon={<Save />} onClick={handleSave}>Save Organization</Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* SBTi Preferences */}
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <SettingsIcon color="primary" />
                <Typography variant="h6">SBTi Preferences</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Default settings for target validation and pathway calculation.
              </Typography>
              <Grid container spacing={2}>
                {sbtiPrefs.map((pref) => (
                  <Grid item xs={12} sm={6} key={pref.id}>
                    <FormControl fullWidth size="small">
                      <InputLabel>{pref.label}</InputLabel>
                      <Select value={pref.value} label={pref.label} onChange={(e: SelectChangeEvent) => handlePrefChange(pref.id, e.target.value)}>
                        {pref.options.map((o) => <MenuItem key={o} value={o}>{o}</MenuItem>)}
                      </Select>
                    </FormControl>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>

          {/* Notifications */}
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Notifications color="primary" />
                <Typography variant="h6">Notifications</Typography>
              </Box>
              <List dense>
                {notifications.map((n) => (
                  <React.Fragment key={n.id}>
                    <ListItem>
                      <ListItemText
                        primary={<Typography variant="body2" sx={{ fontWeight: 600 }}>{n.label}</Typography>}
                        secondary={n.description}
                      />
                      <ListItemSecondaryAction>
                        <Switch edge="end" checked={n.enabled} onChange={() => handleNotifToggle(n.id)} color="success" />
                      </ListItemSecondaryAction>
                    </ListItem>
                    <Divider component="li" />
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Sources */}
        <Grid item xs={12}>
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
                        <TableCell><Chip label={source.type} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} /></TableCell>
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
                          <IconButton size="small"><Edit fontSize="small" /></IconButton>
                          <IconButton size="small" color="error"><Delete fontSize="small" /></IconButton>
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
                  <MenuItem value="targets">Targets</MenuItem>
                  <MenuItem value="reporting">Reporting</MenuItem>
                  <MenuItem value="supply_chain">Supply Chain</MenuItem>
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
