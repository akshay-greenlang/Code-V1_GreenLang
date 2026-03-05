/**
 * ISSBCrossWalk - TCFD-to-ISSB/IFRS S2 mapping table, dual scorecard, migration checklist.
 */

import React, { useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Chip,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  LinearProgress, Tabs, Tab, Checkbox, FormControlLabel, Divider,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import { CompareArrows, CheckCircle, Sync, Assignment } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';

/* ── Demo Data ────────────────────────────────────────────────── */

interface CrosswalkMapping {
  id: string;
  tcfdSection: string;
  tcfdCode: string;
  issbParagraph: string;
  issbRequirement: string;
  alignment: 'full' | 'partial' | 'new' | 'enhanced';
  tcfdStatus: 'complete' | 'in_progress' | 'not_started';
  issbStatus: 'complete' | 'in_progress' | 'not_started';
  notes: string;
}

const CROSSWALK_MAPPINGS: CrosswalkMapping[] = [
  { id: '1', tcfdSection: 'Governance - Board Oversight', tcfdCode: 'GOV-A', issbParagraph: 'IFRS S2 6(a)', issbRequirement: 'Governance body oversight of climate-related risks and opportunities', alignment: 'full', tcfdStatus: 'complete', issbStatus: 'complete', notes: 'Direct mapping, ISSB adds competency disclosure requirement' },
  { id: '2', tcfdSection: 'Governance - Management Role', tcfdCode: 'GOV-B', issbParagraph: 'IFRS S2 6(b)', issbRequirement: 'Management role in governance processes, controls, and procedures', alignment: 'enhanced', tcfdStatus: 'complete', issbStatus: 'in_progress', notes: 'ISSB requires more detail on controls and procedures' },
  { id: '3', tcfdSection: 'Strategy - Risks & Opportunities', tcfdCode: 'STR-A', issbParagraph: 'IFRS S2 10-12', issbRequirement: 'Climate-related risks and opportunities with reasonable expectation of affecting entity', alignment: 'enhanced', tcfdStatus: 'complete', issbStatus: 'in_progress', notes: 'ISSB requires value chain analysis and current/anticipated effects' },
  { id: '4', tcfdSection: 'Strategy - Business Impact', tcfdCode: 'STR-B', issbParagraph: 'IFRS S2 13-15', issbRequirement: 'Effects on business model, value chain, financial position, and performance', alignment: 'enhanced', tcfdStatus: 'in_progress', issbStatus: 'not_started', notes: 'ISSB significantly expands financial impact quantification requirements' },
  { id: '5', tcfdSection: 'Strategy - Resilience', tcfdCode: 'STR-C', issbParagraph: 'IFRS S2 22', issbRequirement: 'Climate resilience assessment including scenario analysis', alignment: 'enhanced', tcfdStatus: 'in_progress', issbStatus: 'not_started', notes: 'ISSB mandates quantitative scenario analysis with specific methodological disclosure' },
  { id: '6', tcfdSection: 'Strategy - Transition Plan', tcfdCode: 'N/A', issbParagraph: 'IFRS S2 14(a)', issbRequirement: 'Transition plan for managing climate risks including targets and actions', alignment: 'new', tcfdStatus: 'not_started', issbStatus: 'not_started', notes: 'NEW: ISSB requires explicit transition plan disclosure (not in TCFD)' },
  { id: '7', tcfdSection: 'Risk Management - Identification', tcfdCode: 'RM-A', issbParagraph: 'IFRS S2 25(a)', issbRequirement: 'Processes to identify, assess and prioritize climate-related risks', alignment: 'full', tcfdStatus: 'complete', issbStatus: 'complete', notes: 'Direct mapping' },
  { id: '8', tcfdSection: 'Risk Management - Process', tcfdCode: 'RM-B', issbParagraph: 'IFRS S2 25(b)', issbRequirement: 'Processes to monitor, manage and mitigate climate-related risks', alignment: 'full', tcfdStatus: 'in_progress', issbStatus: 'in_progress', notes: 'Direct mapping, slight wording differences' },
  { id: '9', tcfdSection: 'Risk Management - ERM Integration', tcfdCode: 'RM-C', issbParagraph: 'IFRS S2 25(c)', issbRequirement: 'Integration into overall risk management process', alignment: 'full', tcfdStatus: 'not_started', issbStatus: 'not_started', notes: 'Direct mapping' },
  { id: '10', tcfdSection: 'Metrics - Climate Metrics', tcfdCode: 'MT-A', issbParagraph: 'IFRS S2 29', issbRequirement: 'Climate-related metrics including cross-industry and industry-specific', alignment: 'enhanced', tcfdStatus: 'complete', issbStatus: 'in_progress', notes: 'ISSB adds industry-specific metrics (SASB-based) and internal carbon pricing' },
  { id: '11', tcfdSection: 'Metrics - GHG Emissions', tcfdCode: 'MT-B', issbParagraph: 'IFRS S2 29(a)', issbRequirement: 'Scope 1, 2, 3 GHG emissions (GHG Protocol aligned)', alignment: 'full', tcfdStatus: 'complete', issbStatus: 'complete', notes: 'Direct mapping, ISSB mandates Scope 3 (TCFD recommends)' },
  { id: '12', tcfdSection: 'Metrics - Targets', tcfdCode: 'MT-C', issbParagraph: 'IFRS S2 33-36', issbRequirement: 'Climate-related targets including GHG reduction targets with methodology', alignment: 'enhanced', tcfdStatus: 'in_progress', issbStatus: 'not_started', notes: 'ISSB requires validated targets, interim milestones, third-party verification' },
  { id: '13', tcfdSection: 'Metrics - Remuneration', tcfdCode: 'N/A', issbParagraph: 'IFRS S2 29(g)', issbRequirement: 'Climate-related remuneration including proportion linked to climate performance', alignment: 'new', tcfdStatus: 'not_started', issbStatus: 'not_started', notes: 'NEW: ISSB requires climate-linked remuneration disclosure' },
];

const DUAL_SCORECARD = [
  { dimension: 'Board Oversight', tcfd: 85, issb: 75 },
  { dimension: 'Management Controls', tcfd: 80, issb: 55 },
  { dimension: 'Risk Identification', tcfd: 75, issb: 70 },
  { dimension: 'Scenario Analysis', tcfd: 55, issb: 35 },
  { dimension: 'Financial Impact', tcfd: 50, issb: 25 },
  { dimension: 'Transition Plan', tcfd: 10, issb: 10 },
  { dimension: 'GHG Emissions', tcfd: 90, issb: 85 },
  { dimension: 'Targets', tcfd: 72, issb: 45 },
  { dimension: 'Industry Metrics', tcfd: 60, issb: 30 },
  { dimension: 'Remuneration', tcfd: 20, issb: 15 },
];

interface MigrationItem {
  id: string;
  task: string;
  category: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  effort: string;
  completed: boolean;
}

const MIGRATION_CHECKLIST: MigrationItem[] = [
  { id: '1', task: 'Map all TCFD disclosures to IFRS S2 paragraphs', category: 'Planning', priority: 'critical', effort: '2 weeks', completed: true },
  { id: '2', task: 'Identify new ISSB requirements not covered by TCFD', category: 'Planning', priority: 'critical', effort: '1 week', completed: true },
  { id: '3', task: 'Develop transition plan disclosure (IFRS S2 14a)', category: 'Strategy', priority: 'critical', effort: '6 weeks', completed: false },
  { id: '4', task: 'Quantify financial impacts for all material risks', category: 'Strategy', priority: 'critical', effort: '8 weeks', completed: false },
  { id: '5', task: 'Implement industry-specific metrics (SASB-aligned)', category: 'Metrics', priority: 'high', effort: '4 weeks', completed: false },
  { id: '6', task: 'Document climate-linked remuneration policies', category: 'Governance', priority: 'high', effort: '2 weeks', completed: false },
  { id: '7', task: 'Enhance scenario analysis with quantitative outputs', category: 'Strategy', priority: 'high', effort: '6 weeks', completed: false },
  { id: '8', task: 'Add interim milestones to all climate targets', category: 'Metrics', priority: 'medium', effort: '2 weeks', completed: false },
  { id: '9', task: 'Document management controls and procedures', category: 'Governance', priority: 'medium', effort: '3 weeks', completed: false },
  { id: '10', task: 'Obtain third-party verification for GHG emissions', category: 'Metrics', priority: 'medium', effort: '4 weeks', completed: false },
  { id: '11', task: 'Prepare value chain analysis for Scope 3', category: 'Metrics', priority: 'medium', effort: '6 weeks', completed: false },
  { id: '12', task: 'Train finance team on ISSB-specific requirements', category: 'Planning', priority: 'low', effort: '1 week', completed: false },
];

/* ── Component ─────────────────────────────────────────────────── */

const ISSBCrossWalk: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  const fullAligned = CROSSWALK_MAPPINGS.filter((m) => m.alignment === 'full').length;
  const enhancedRequired = CROSSWALK_MAPPINGS.filter((m) => m.alignment === 'enhanced').length;
  const newRequired = CROSSWALK_MAPPINGS.filter((m) => m.alignment === 'new').length;
  const issbComplete = CROSSWALK_MAPPINGS.filter((m) => m.issbStatus === 'complete').length;
  const migrationComplete = MIGRATION_CHECKLIST.filter((m) => m.completed).length;

  const tcfdAvg = Math.round(DUAL_SCORECARD.reduce((s, d) => s + d.tcfd, 0) / DUAL_SCORECARD.length);
  const issbAvg = Math.round(DUAL_SCORECARD.reduce((s, d) => s + d.issb, 0) / DUAL_SCORECARD.length);

  const getAlignmentColor = (alignment: string): string => {
    if (alignment === 'full') return '#2E7D32';
    if (alignment === 'partial') return '#F57F17';
    if (alignment === 'enhanced') return '#0D47A1';
    return '#C62828';
  };

  const getAlignmentLabel = (alignment: string): string => {
    if (alignment === 'full') return 'Full Alignment';
    if (alignment === 'partial') return 'Partial';
    if (alignment === 'enhanced') return 'Enhanced Required';
    return 'New in ISSB';
  };

  const getStatusChip = (status: string) => {
    const colors: Record<string, 'success' | 'warning' | 'default'> = {
      complete: 'success',
      in_progress: 'warning',
      not_started: 'default',
    };
    return (
      <Chip
        label={status.replace(/_/g, ' ')}
        size="small"
        color={colors[status] || 'default'}
        sx={{ textTransform: 'capitalize', fontSize: '0.7rem' }}
      />
    );
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">ISSB Crosswalk</Typography>
        <Typography variant="body2" color="text.secondary">
          TCFD-to-ISSB/IFRS S2 mapping, dual compliance scorecard, and migration planning
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="TCFD Score"
            value={tcfdAvg}
            format="percent"
            icon={<CompareArrows />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="ISSB Score"
            value={issbAvg}
            format="percent"
            icon={<Sync />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="ISSB Sections Complete"
            value={issbComplete}
            icon={<CheckCircle />}
            subtitle={`of ${CROSSWALK_MAPPINGS.length} total`}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Migration Progress"
            value={migrationComplete}
            icon={<Assignment />}
            subtitle={`of ${MIGRATION_CHECKLIST.length} tasks`}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Tabs */}
      <Card sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ px: 2, pt: 1 }}>
          <Tab label="Crosswalk Mapping" />
          <Tab label="Dual Scorecard" />
          <Tab label="Migration Checklist" />
        </Tabs>

        <CardContent>
          {/* Tab 0: Crosswalk Mapping Table */}
          {activeTab === 0 && (
            <>
              <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
                <Chip label={`${fullAligned} Full Alignment`} sx={{ backgroundColor: '#E8F5E9', color: '#2E7D32', fontWeight: 600 }} size="small" />
                <Chip label={`${enhancedRequired} Enhanced Required`} sx={{ backgroundColor: '#E3F2FD', color: '#0D47A1', fontWeight: 600 }} size="small" />
                <Chip label={`${newRequired} New in ISSB`} sx={{ backgroundColor: '#FFEBEE', color: '#C62828', fontWeight: 600 }} size="small" />
              </Box>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>TCFD Section</TableCell>
                      <TableCell>TCFD Code</TableCell>
                      <TableCell>ISSB Para.</TableCell>
                      <TableCell>ISSB Requirement</TableCell>
                      <TableCell align="center">Alignment</TableCell>
                      <TableCell align="center">TCFD Status</TableCell>
                      <TableCell align="center">ISSB Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {CROSSWALK_MAPPINGS.map((mapping) => (
                      <TableRow key={mapping.id} hover>
                        <TableCell sx={{ fontWeight: 500, fontSize: '0.8rem' }}>{mapping.tcfdSection}</TableCell>
                        <TableCell>
                          <Chip label={mapping.tcfdCode} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                        </TableCell>
                        <TableCell>
                          <Chip label={mapping.issbParagraph} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.8rem', maxWidth: 250 }}>{mapping.issbRequirement}</TableCell>
                        <TableCell align="center">
                          <Chip
                            label={getAlignmentLabel(mapping.alignment)}
                            size="small"
                            sx={{
                              backgroundColor: `${getAlignmentColor(mapping.alignment)}15`,
                              color: getAlignmentColor(mapping.alignment),
                              fontWeight: 600,
                              fontSize: '0.65rem',
                            }}
                          />
                        </TableCell>
                        <TableCell align="center">{getStatusChip(mapping.tcfdStatus)}</TableCell>
                        <TableCell align="center">{getStatusChip(mapping.issbStatus)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}

          {/* Tab 1: Dual Scorecard */}
          {activeTab === 1 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>Compliance Radar</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={DUAL_SCORECARD}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="dimension" fontSize={9} />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                    <Radar name="TCFD" dataKey="tcfd" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                    <Radar name="ISSB" dataKey="issb" stroke="#0D47A1" fill="#0D47A1" fillOpacity={0.2} strokeWidth={2} />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>Side-by-Side Comparison</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={DUAL_SCORECARD} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <YAxis type="category" dataKey="dimension" width={110} fontSize={10} />
                    <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                    <Legend />
                    <Bar dataKey="tcfd" name="TCFD" fill="#1B5E20" barSize={8} />
                    <Bar dataKey="issb" name="ISSB" fill="#0D47A1" barSize={8} />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
            </Grid>
          )}

          {/* Tab 2: Migration Checklist */}
          {activeTab === 2 && (
            <>
              <Box sx={{ mb: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={(migrationComplete / MIGRATION_CHECKLIST.length) * 100}
                  sx={{ height: 10, borderRadius: 5 }}
                  color="success"
                />
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {migrationComplete}/{MIGRATION_CHECKLIST.length} tasks completed ({Math.round((migrationComplete / MIGRATION_CHECKLIST.length) * 100)}%)
                </Typography>
              </Box>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ width: 40 }}></TableCell>
                      <TableCell>Task</TableCell>
                      <TableCell>Category</TableCell>
                      <TableCell>Priority</TableCell>
                      <TableCell>Effort</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {MIGRATION_CHECKLIST.map((item) => (
                      <TableRow key={item.id} hover sx={{ opacity: item.completed ? 0.6 : 1 }}>
                        <TableCell>
                          <Checkbox checked={item.completed} size="small" disabled />
                        </TableCell>
                        <TableCell sx={{ fontWeight: 500, textDecoration: item.completed ? 'line-through' : 'none', fontSize: '0.85rem' }}>
                          {item.task}
                        </TableCell>
                        <TableCell>
                          <Chip label={item.category} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={item.priority}
                            size="small"
                            color={item.priority === 'critical' ? 'error' : item.priority === 'high' ? 'warning' : item.priority === 'medium' ? 'info' : 'default'}
                            sx={{ textTransform: 'capitalize' }}
                          />
                        </TableCell>
                        <TableCell sx={{ fontSize: '0.8rem' }}>{item.effort}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ISSBCrossWalk;
