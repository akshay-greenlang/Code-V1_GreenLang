/**
 * Settings - Organization config, reporting periods, thresholds, MRV mapping.
 */

import React, { useState } from 'react';
import {
  Typography, Box, Card, CardContent, Grid, TextField, FormControl, InputLabel,
  Select, MenuItem, Button, Switch, FormControlLabel, Divider, Chip, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow, Alert,
} from '@mui/material';
import { Save } from '@mui/icons-material';

const Settings: React.FC = () => {
  const [orgName, setOrgName] = useState('GreenLang Demo Corp');
  const [lei, setLei] = useState('ABCD1234567890ABCDEF');
  const [sector, setSector] = useState('Energy');
  const [currency, setCurrency] = useState('EUR');
  const [daVersion, setDaVersion] = useState('2024');
  const [deMinimis, setDeMinimis] = useState(5);
  const [autoRegulatory, setAutoRegulatory] = useState(true);

  return (
    <Box>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>Settings</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Organization</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField label="Organization Name" value={orgName} onChange={(e) => setOrgName(e.target.value)} fullWidth size="small" />
                <TextField label="LEI Code" value={lei} onChange={(e) => setLei(e.target.value)} fullWidth size="small" />
                <FormControl fullWidth size="small">
                  <InputLabel>Sector</InputLabel>
                  <Select value={sector} onChange={(e) => setSector(e.target.value)} label="Sector">
                    {['Energy', 'Manufacturing', 'Transport', 'Construction', 'ICT', 'Financial Services'].map(s => (
                      <MenuItem key={s} value={s}>{s}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth size="small">
                  <InputLabel>Default Currency</InputLabel>
                  <Select value={currency} onChange={(e) => setCurrency(e.target.value)} label="Default Currency">
                    {['EUR', 'USD', 'GBP', 'CHF'].map(c => <MenuItem key={c} value={c}>{c}</MenuItem>)}
                  </Select>
                </FormControl>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Reporting Configuration</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth size="small">
                  <InputLabel>Delegated Act Version</InputLabel>
                  <Select value={daVersion} onChange={(e) => setDaVersion(e.target.value)} label="Delegated Act Version">
                    <MenuItem value="2021">Climate DA (June 2021)</MenuItem>
                    <MenuItem value="2023">Environmental DA (June 2023)</MenuItem>
                    <MenuItem value="2024">Climate DA Amendment (2024)</MenuItem>
                  </Select>
                </FormControl>
                <TextField label="De Minimis Threshold (%)" type="number" value={deMinimis} onChange={(e) => setDeMinimis(Number(e.target.value))} fullWidth size="small" />
                <FormControlLabel control={<Switch checked={autoRegulatory} onChange={(e) => setAutoRegulatory(e.target.checked)} />} label="Auto-update on regulatory changes" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Reporting Periods</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Period</TableCell>
                      <TableCell>Start</TableCell>
                      <TableCell>End</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow><TableCell>FY 2025</TableCell><TableCell>2025-01-01</TableCell><TableCell>2025-12-31</TableCell><TableCell><Chip label="Active" size="small" color="success" /></TableCell></TableRow>
                    <TableRow><TableCell>FY 2024</TableCell><TableCell>2024-01-01</TableCell><TableCell>2024-12-31</TableCell><TableCell><Chip label="Closed" size="small" /></TableCell></TableRow>
                    <TableRow><TableCell>FY 2023</TableCell><TableCell>2023-01-01</TableCell><TableCell>2023-12-31</TableCell><TableCell><Chip label="Closed" size="small" /></TableCell></TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>MRV Agent Mapping</Typography>
              <Alert severity="info" sx={{ mb: 2 }}>
                Map GreenLang MRV agents to Taxonomy environmental objectives for automated data flow.
              </Alert>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>MRV Agent</TableCell>
                      <TableCell>Objective</TableCell>
                      <TableCell>Type</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow><TableCell>Stationary Combustion</TableCell><TableCell><Chip label="CCM" size="small" color="primary" /></TableCell><TableCell>Direct</TableCell></TableRow>
                    <TableRow><TableCell>Scope 2 Location</TableCell><TableCell><Chip label="CCM" size="small" color="primary" /></TableCell><TableCell>Calculated</TableCell></TableRow>
                    <TableRow><TableCell>Fugitive Emissions</TableCell><TableCell><Chip label="PPC" size="small" color="primary" /></TableCell><TableCell>Direct</TableCell></TableRow>
                    <TableRow><TableCell>Water Treatment</TableCell><TableCell><Chip label="WTR" size="small" color="primary" /></TableCell><TableCell>Direct</TableCell></TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button variant="contained" startIcon={<Save />}>Save Settings</Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;
