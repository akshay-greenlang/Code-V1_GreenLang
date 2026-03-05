/**
 * ScenarioAnalysis - THE MOST IMPORTANT PAGE
 *
 * Scenario selector, side-by-side comparison, parameter display, financial impact
 * waterfall chart, sensitivity tornado chart, and custom scenario builder dialog.
 */

import React, { useMemo, useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Button, Tabs, Tab, Chip, Dialog,
  DialogTitle, DialogContent, DialogActions, TextField, Slider, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Paper, IconButton, Alert,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  Cell, LineChart, Line, ReferenceLine,
} from 'recharts';
import { Add, PlayArrow, Compare, Thermostat, TrendingDown, TrendingUp } from '@mui/icons-material';

interface ScenarioCard {
  id: string;
  name: string;
  type: string;
  temperature: string;
  carbonPrice2030: number;
  carbonPrice2050: number;
  renewableShare2050: number;
  netImpact: number;
  color: string;
}

const SCENARIOS: ScenarioCard[] = [
  { id: 's1', name: 'Net Zero 2050 (IEA NZE)', type: 'orderly_transition', temperature: '1.5C', carbonPrice2030: 130, carbonPrice2050: 250, renewableShare2050: 90, netImpact: -12500000, color: '#1B5E20' },
  { id: 's2', name: 'Announced Pledges (APS)', type: 'disorderly_transition', temperature: '2.0C', carbonPrice2030: 90, carbonPrice2050: 160, renewableShare2050: 70, netImpact: -5800000, color: '#0D47A1' },
  { id: 's3', name: 'Current Policies (STEPS)', type: 'hot_house', temperature: '3.0C', carbonPrice2030: 35, carbonPrice2050: 50, renewableShare2050: 45, netImpact: -28400000, color: '#E65100' },
  { id: 's4', name: 'Delayed Transition (NGFS)', type: 'delayed_transition', temperature: '2.0C', carbonPrice2030: 25, carbonPrice2050: 350, renewableShare2050: 65, netImpact: -18200000, color: '#7B1FA2' },
];

const WATERFALL_DATA = [
  { name: 'Baseline Revenue', value: 500, cumulative: 500, fill: '#0D47A1' },
  { name: 'Carbon Costs', value: -45, cumulative: 455, fill: '#C62828' },
  { name: 'Energy Costs', value: -22, cumulative: 433, fill: '#C62828' },
  { name: 'Stranded Assets', value: -35, cumulative: 398, fill: '#C62828' },
  { name: 'New Revenue', value: 62, cumulative: 460, fill: '#2E7D32' },
  { name: 'Cost Savings', value: 28, cumulative: 488, fill: '#2E7D32' },
  { name: 'Net Adjusted', value: 488, cumulative: 488, fill: '#1B5E20' },
];

const SENSITIVITY_DATA = [
  { parameter: 'Carbon Price', low_impact: -25, high_impact: 15, unit: '$/tCO2' },
  { parameter: 'Energy Costs', low_impact: -18, high_impact: 8, unit: '% change' },
  { parameter: 'Demand Shift', low_impact: -12, high_impact: 22, unit: '% change' },
  { parameter: 'Technology Cost', low_impact: -8, high_impact: 15, unit: '% change' },
  { parameter: 'Physical Damage', low_impact: -20, high_impact: 0, unit: '$M' },
  { parameter: 'Regulatory Timing', low_impact: -15, high_impact: 5, unit: 'years' },
];

const ScenarioAnalysis: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>(['s1', 's2', 's3']);
  const [customDialogOpen, setCustomDialogOpen] = useState(false);

  const selectedCards = SCENARIOS.filter((s) => selectedScenarios.includes(s.id));

  const comparisonData = useMemo(() => [
    { metric: 'Carbon Price 2030', ...Object.fromEntries(selectedCards.map((s) => [s.name, s.carbonPrice2030])) },
    { metric: 'Carbon Price 2050', ...Object.fromEntries(selectedCards.map((s) => [s.name, s.carbonPrice2050])) },
    { metric: 'Renewable Share 2050 (%)', ...Object.fromEntries(selectedCards.map((s) => [s.name, s.renewableShare2050])) },
  ], [selectedCards]);

  const toggleScenario = (id: string) => {
    setSelectedScenarios((prev) =>
      prev.includes(id)
        ? prev.filter((s) => s !== id)
        : prev.length < 4
        ? [...prev, id]
        : prev
    );
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Scenario Analysis</Typography>
          <Typography variant="body2" color="text.secondary">
            TCFD Strategy Disclosure (c) -- Resilience under different climate scenarios
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button variant="outlined" startIcon={<Add />} onClick={() => setCustomDialogOpen(true)}>
            Custom Scenario
          </Button>
          <Button variant="contained" startIcon={<PlayArrow />}>Run Analysis</Button>
        </Box>
      </Box>

      {/* Scenario Selector Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {SCENARIOS.map((scenario) => {
          const isSelected = selectedScenarios.includes(scenario.id);
          return (
            <Grid item xs={12} sm={6} md={3} key={scenario.id}>
              <Card
                onClick={() => toggleScenario(scenario.id)}
                sx={{
                  cursor: 'pointer',
                  border: isSelected ? `2px solid ${scenario.color}` : '1px solid #E0E0E0',
                  backgroundColor: isSelected ? `${scenario.color}08` : 'white',
                  transition: 'all 0.2s',
                  '&:hover': { boxShadow: 3 },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Chip label={scenario.temperature} size="small" sx={{ backgroundColor: scenario.color, color: 'white', fontWeight: 600 }} />
                    {isSelected && <Chip label="Selected" size="small" color="primary" variant="outlined" />}
                  </Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>{scenario.name}</Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Carbon price 2030: ${scenario.carbonPrice2030}/tCO2
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Renewable share 2050: {scenario.renewableShare2050}%
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{ mt: 1, fontWeight: 600, color: scenario.netImpact >= 0 ? 'success.main' : 'error.main' }}
                  >
                    Net Impact: ${(scenario.netImpact / 1e6).toFixed(1)}M
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Tabs */}
      <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 3 }}>
        <Tab label="Side-by-Side Comparison" />
        <Tab label="Financial Waterfall" />
        <Tab label="Sensitivity Analysis" />
      </Tabs>

      {/* Comparison View */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>Scenario Parameter Comparison</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 700 }}>Parameter</TableCell>
                        {selectedCards.map((s) => (
                          <TableCell key={s.id} align="center" sx={{ fontWeight: 700 }}>
                            <Chip label={s.temperature} size="small" sx={{ backgroundColor: s.color, color: 'white', mr: 1 }} />
                            {s.name}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Temperature Target</TableCell>
                        {selectedCards.map((s) => <TableCell key={s.id} align="center">{s.temperature}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell>Carbon Price 2030 ($/tCO2)</TableCell>
                        {selectedCards.map((s) => <TableCell key={s.id} align="center">${s.carbonPrice2030}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell>Carbon Price 2050 ($/tCO2)</TableCell>
                        {selectedCards.map((s) => <TableCell key={s.id} align="center">${s.carbonPrice2050}</TableCell>)}
                      </TableRow>
                      <TableRow>
                        <TableCell>Renewable Energy Share 2050</TableCell>
                        {selectedCards.map((s) => <TableCell key={s.id} align="center">{s.renewableShare2050}%</TableCell>)}
                      </TableRow>
                      <TableRow sx={{ backgroundColor: '#FAFAFA' }}>
                        <TableCell sx={{ fontWeight: 600 }}>Net Financial Impact</TableCell>
                        {selectedCards.map((s) => (
                          <TableCell key={s.id} align="center" sx={{ fontWeight: 700, color: s.netImpact >= 0 ? 'success.main' : 'error.main' }}>
                            ${(s.netImpact / 1e6).toFixed(1)}M
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>Net Impact Comparison ($M)</Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={selectedCards.map((s) => ({ name: s.name, impact: s.netImpact / 1e6 }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" fontSize={11} />
                    <YAxis tickFormatter={(v) => `$${v}M`} />
                    <Tooltip formatter={(v: number) => [`$${v.toFixed(1)}M`, 'Net Impact']} />
                    <ReferenceLine y={0} stroke="#666" />
                    <Bar dataKey="impact" name="Net Impact">
                      {selectedCards.map((s) => (
                        <Cell key={s.id} fill={s.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Waterfall Chart */}
      {activeTab === 1 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>Financial Impact Waterfall -- Net Zero 2050 Scenario ($M)</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={WATERFALL_DATA}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" fontSize={11} />
                <YAxis tickFormatter={(v) => `$${v}M`} />
                <Tooltip formatter={(v: number) => [`$${v}M`, '']} />
                <Bar dataKey="value" name="Impact">
                  {WATERFALL_DATA.map((entry, idx) => (
                    <Cell key={idx} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <Box sx={{ display: 'flex', gap: 3, mt: 2, justifyContent: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 12, height: 12, backgroundColor: '#C62828', borderRadius: 1 }} />
                <Typography variant="caption">Risk/Cost Drivers</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 12, height: 12, backgroundColor: '#2E7D32', borderRadius: 1 }} />
                <Typography variant="caption">Opportunity Drivers</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 12, height: 12, backgroundColor: '#1B5E20', borderRadius: 1 }} />
                <Typography variant="caption">Net Result</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Sensitivity Tornado */}
      {activeTab === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>Sensitivity Tornado Chart -- Key Parameter Impact ($M)</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={SENSITIVITY_DATA} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={(v) => `$${v}M`} />
                <YAxis type="category" dataKey="parameter" width={130} fontSize={12} />
                <Tooltip formatter={(v: number) => [`$${v}M`, '']} />
                <ReferenceLine x={0} stroke="#666" />
                <Bar dataKey="low_impact" name="Downside" fill="#C62828" />
                <Bar dataKey="high_impact" name="Upside" fill="#2E7D32" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Custom Scenario Builder Dialog */}
      <Dialog open={customDialogOpen} onClose={() => setCustomDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Build Custom Scenario</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
            <TextField label="Scenario Name" fullWidth size="small" />
            <TextField label="Description" fullWidth multiline rows={2} size="small" />
            <Box>
              <Typography gutterBottom>Temperature Target</Typography>
              <Slider defaultValue={2.0} min={1.5} max={4.0} step={0.5} marks valueLabelDisplay="auto" valueLabelFormat={(v) => `${v}C`} />
            </Box>
            <Box>
              <Typography gutterBottom>Carbon Price 2030 ($/tCO2)</Typography>
              <Slider defaultValue={50} min={0} max={300} valueLabelDisplay="auto" valueLabelFormat={(v) => `$${v}`} />
            </Box>
            <Box>
              <Typography gutterBottom>Carbon Price 2050 ($/tCO2)</Typography>
              <Slider defaultValue={150} min={0} max={500} valueLabelDisplay="auto" valueLabelFormat={(v) => `$${v}`} />
            </Box>
            <Box>
              <Typography gutterBottom>Renewable Energy Share 2050 (%)</Typography>
              <Slider defaultValue={60} min={20} max={100} valueLabelDisplay="auto" valueLabelFormat={(v) => `${v}%`} />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCustomDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" startIcon={<PlayArrow />}>Create & Analyze</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ScenarioAnalysis;
