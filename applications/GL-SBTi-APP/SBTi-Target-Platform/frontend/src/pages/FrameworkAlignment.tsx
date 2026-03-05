/**
 * FrameworkAlignment - Cross-framework alignment analysis page.
 *
 * Shows alignment bar chart, cross-reference mapping table,
 * and gap indicators across GHG Protocol, CDP, TCFD, CSRD, ISO 14064, ISSB, and SEC.
 */

import React, { useState, useMemo } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Chip, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent,
  LinearProgress,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import FrameworkAlignmentComponent from '../components/frameworks/FrameworkAlignment';
import CrossRefTable from '../components/frameworks/CrossRefTable';
import GapIndicator from '../components/frameworks/GapIndicator';

const DEMO_ALIGNMENT = [
  { framework: 'ghg_protocol', alignment_pct: 92, met: 11, partial: 1, not_met: 0 },
  { framework: 'cdp', alignment_pct: 78, met: 14, partial: 4, not_met: 2 },
  { framework: 'tcfd', alignment_pct: 72, met: 8, partial: 3, not_met: 1 },
  { framework: 'csrd', alignment_pct: 65, met: 10, partial: 5, not_met: 3 },
  { framework: 'iso_14064', alignment_pct: 88, met: 9, partial: 1, not_met: 0 },
  { framework: 'issb', alignment_pct: 70, met: 7, partial: 3, not_met: 2 },
  { framework: 'sec_climate', alignment_pct: 55, met: 5, partial: 3, not_met: 4 },
];

const DEMO_MAPPINGS = [
  { id: '1', sbti_requirement: 'Base Year Emissions Inventory', sbti_code: 'C1.1', framework: 'ghg_protocol', framework_requirement: 'Complete GHG inventory', mapping_type: 'direct', organization_status: 'met' as const, gap_description: null },
  { id: '2', sbti_requirement: 'Scope 1+2 Coverage >= 95%', sbti_code: 'C2.1', framework: 'cdp', framework_requirement: 'C6.1 - GHG emissions data', mapping_type: 'direct', organization_status: 'met' as const, gap_description: null },
  { id: '3', sbti_requirement: 'Near-term Target 5-10yr', sbti_code: 'C4.1', framework: 'tcfd', framework_requirement: 'MT-C: Climate targets', mapping_type: 'partial', organization_status: 'met' as const, gap_description: null },
  { id: '4', sbti_requirement: 'Scope 3 Screening', sbti_code: 'C9.1', framework: 'csrd', framework_requirement: 'ESRS E1-6: Value chain emissions', mapping_type: 'partial', organization_status: 'partial' as const, gap_description: 'CSRD requires broader value chain disclosure than SBTi screening' },
  { id: '5', sbti_requirement: 'Minimum Ambition Level', sbti_code: 'C5.1', framework: 'iso_14064', framework_requirement: 'Clause 6: Reduction targets', mapping_type: 'complementary', organization_status: 'met' as const, gap_description: null },
  { id: '6', sbti_requirement: 'Annual Progress Reporting', sbti_code: 'C10.1', framework: 'cdp', framework_requirement: 'C4.1b - Emissions reduction target progress', mapping_type: 'direct', organization_status: 'met' as const, gap_description: null },
  { id: '7', sbti_requirement: 'Temperature Scoring', sbti_code: 'C11.1', framework: 'issb', framework_requirement: 'IFRS S2: Climate-related metrics', mapping_type: 'partial', organization_status: 'partial' as const, gap_description: 'ISSB requires scenario-based temperature disclosure' },
  { id: '8', sbti_requirement: 'Base Year Recalculation', sbti_code: 'C12.1', framework: 'sec_climate', framework_requirement: 'GHG emissions restatement', mapping_type: 'complementary', organization_status: 'not_met' as const, gap_description: 'SEC requires materiality-based disclosure framework' },
  { id: '9', sbti_requirement: 'FLAG Target Setting', sbti_code: 'C8.1', framework: 'csrd', framework_requirement: 'ESRS E4: Biodiversity', mapping_type: 'indirect', organization_status: 'not_met' as const, gap_description: 'CSRD biodiversity requirements extend beyond FLAG scope' },
];

const FRAMEWORK_COLORS: Record<string, string> = {
  ghg_protocol: '#1B5E20', cdp: '#7B1FA2', tcfd: '#EF6C00', csrd: '#C62828', iso_14064: '#006064', issb: '#33691E', sec_climate: '#880E4F',
};

const FrameworkAlignment: React.FC = () => {
  const [frameworkFilter, setFrameworkFilter] = useState('all');

  const filteredMappings = frameworkFilter === 'all'
    ? DEMO_MAPPINGS
    : DEMO_MAPPINGS.filter((m) => m.framework === frameworkFilter);

  const gaps = DEMO_MAPPINGS.filter((m) => m.organization_status !== 'met');

  const radarData = DEMO_ALIGNMENT.map((a) => ({
    framework: a.framework.toUpperCase().replace(/_/g, ' '),
    alignment: a.alignment_pct,
    target: 80,
  }));

  const overallAlignment = Math.round(DEMO_ALIGNMENT.reduce((s, a) => s + a.alignment_pct, 0) / DEMO_ALIGNMENT.length);

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Framework Alignment</Typography>
        <Typography variant="body2" color="text.secondary">
          Cross-framework alignment between SBTi and GHG Protocol, CDP, TCFD, CSRD, ISO 14064, ISSB, and SEC
        </Typography>
      </Box>

      {/* KPI Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Overall Alignment</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: overallAlignment >= 75 ? '#2E7D32' : '#EF6C00' }}>
                {overallAlignment}%
              </Typography>
              <Typography variant="caption" color="text.secondary">across {DEMO_ALIGNMENT.length} frameworks</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Frameworks Tracked</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700 }}>{DEMO_ALIGNMENT.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Requirements Met</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#2E7D32' }}>
                {DEMO_MAPPINGS.filter((m) => m.organization_status === 'met').length}
              </Typography>
              <Typography variant="caption" color="text.secondary">of {DEMO_MAPPINGS.length} total</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Gaps Identified</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#C62828' }}>{gaps.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Radar + Alignment Bar */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Alignment Radar</Typography>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="framework" fontSize={9} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="Alignment" dataKey="alignment" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                  <Radar name="Target (80%)" dataKey="target" stroke="#C62828" fill="none" strokeWidth={1.5} strokeDasharray="5 5" />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <FrameworkAlignmentComponent items={DEMO_ALIGNMENT as any} />
        </Grid>
      </Grid>

      {/* Cross-Reference Table */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Cross-Reference Mapping</Typography>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>Framework</InputLabel>
            <Select value={frameworkFilter} label="Framework" onChange={(e: SelectChangeEvent) => setFrameworkFilter(e.target.value)}>
              <MenuItem value="all">All Frameworks</MenuItem>
              {DEMO_ALIGNMENT.map((a) => (
                <MenuItem key={a.framework} value={a.framework}>{a.framework.toUpperCase().replace(/_/g, ' ')}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        <CrossRefTable mappings={filteredMappings as any} />
      </Box>

      {/* Gap Indicator */}
      <GapIndicator gaps={gaps as any} />
    </Box>
  );
};

export default FrameworkAlignment;
