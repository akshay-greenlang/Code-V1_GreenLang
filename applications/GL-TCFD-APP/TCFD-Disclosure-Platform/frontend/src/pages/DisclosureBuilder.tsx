/**
 * DisclosureBuilder - 11-section TCFD disclosure checklist, section editor, compliance checker, export.
 */

import React, { useState } from 'react';
import {
  Grid, Card, CardContent, Typography, Box, Button, Chip, TextField,
  List, ListItemButton, ListItemIcon, ListItemText, LinearProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Dialog, DialogTitle, DialogContent, DialogActions, Divider, Alert,
} from '@mui/material';
import {
  CheckCircle, RadioButtonUnchecked, Edit, Download, PictureAsPdf,
  TableChart, Code, Visibility, Warning, Description,
} from '@mui/icons-material';
import StatCard from '../components/common/StatCard';

/* ── Demo Data ────────────────────────────────────────────────── */

interface DisclosureSection {
  id: string;
  pillar: string;
  code: string;
  title: string;
  description: string;
  status: 'complete' | 'in_progress' | 'not_started';
  content: string;
  wordCount: number;
  lastEdited: string;
  complianceNotes: string[];
}

const SECTIONS: DisclosureSection[] = [
  { id: '1', pillar: 'Governance', code: 'GOV-A', title: 'Board Oversight', description: 'Describe the board\'s oversight of climate-related risks and opportunities.', status: 'complete', content: 'The Board of Directors maintains oversight of climate-related risks and opportunities through the Sustainability Committee, which meets quarterly to review climate strategy, risk assessments, and progress toward net-zero targets. The Committee receives reports from management on material climate risks, regulatory developments, and scenario analysis results. Board members have completed climate literacy training covering TCFD, ISSB, and emerging regulatory requirements.', wordCount: 285, lastEdited: '2025-02-15', complianceNotes: [] },
  { id: '2', pillar: 'Governance', code: 'GOV-B', title: 'Management Role', description: 'Describe management\'s role in assessing and managing climate-related risks and opportunities.', status: 'complete', content: 'The Chief Sustainability Officer (CSO) leads the Climate Risk Management team, reporting directly to the CEO and presenting quarterly to the Board Sustainability Committee. The CSO oversees a cross-functional Climate Working Group comprising representatives from Risk, Finance, Operations, and Supply Chain departments.', wordCount: 210, lastEdited: '2025-02-14', complianceNotes: [] },
  { id: '3', pillar: 'Strategy', code: 'STR-A', title: 'Climate Risks & Opportunities', description: 'Describe the climate-related risks and opportunities the organization has identified over the short, medium, and long term.', status: 'complete', content: 'The organization has identified transition and physical risks across three time horizons: Short-term (0-3 years): Carbon pricing regulations, particularly EU CBAM and UK ETS enhancements, represent the most significant near-term transition risk with estimated annual impact of $12.5M. Medium-term (3-10 years): Technology disruption in clean energy and process efficiency could require $51M in capital expenditure. Long-term (10+ years): Physical risks from extreme weather events and sea-level rise threaten $180M in asset value.', wordCount: 380, lastEdited: '2025-02-12', complianceNotes: ['Consider adding quantitative probability assessments for each risk'] },
  { id: '4', pillar: 'Strategy', code: 'STR-B', title: 'Impact on Business', description: 'Describe the impact of climate-related risks and opportunities on the organization\'s businesses, strategy, and financial planning.', status: 'in_progress', content: 'Climate-related factors are integrated into strategic planning through annual scenario analysis exercises. Key impacts include: (1) Revenue: Potential $43M reduction under NZE 2050 scenario due to demand shifts and carbon costs. (2) Costs: Operating expenses expected to increase 10% from compliance requirements. (3) Capital allocation: 15% of CapEx redirected to green investments.', wordCount: 195, lastEdited: '2025-02-10', complianceNotes: ['Needs financial planning integration details', 'Add supply chain impact assessment'] },
  { id: '5', pillar: 'Strategy', code: 'STR-C', title: 'Resilience of Strategy', description: 'Describe the resilience of the organization\'s strategy, taking into consideration different climate-related scenarios.', status: 'in_progress', content: 'The organization conducts annual scenario analysis using IEA NZE 2050, APS, and STEPS pathways. Under the NZE 2050 scenario, EBITDA impact is estimated at -19% by 2030. Mitigation strategies include diversification into green products (projected $43M revenue by 2028) and operational efficiency improvements targeting 25% emissions reduction.', wordCount: 165, lastEdited: '2025-02-08', complianceNotes: ['Expand 2 degree scenario analysis', 'Include quantitative resilience metrics'] },
  { id: '6', pillar: 'Risk Management', code: 'RM-A', title: 'Risk Identification', description: 'Describe the organization\'s processes for identifying and assessing climate-related risks.', status: 'complete', content: 'Climate risk identification follows a structured process integrated with the Enterprise Risk Management (ERM) framework. The process includes: quarterly risk scanning across physical and transition risk categories, asset-level vulnerability assessments using RCP/SSP climate scenarios, supply chain risk mapping covering Tier 1-3 suppliers, and stakeholder engagement including investor expectations and regulatory tracking.', wordCount: 245, lastEdited: '2025-02-06', complianceNotes: [] },
  { id: '7', pillar: 'Risk Management', code: 'RM-B', title: 'Risk Management Process', description: 'Describe the organization\'s processes for managing climate-related risks.', status: 'in_progress', content: 'Climate risks are managed through a combination of mitigation, transfer, acceptance, and avoidance strategies. The risk management team maintains a climate risk register with 6 identified risks, of which 4 are actively being mitigated. Key management activities include insurance coverage for physical risks, carbon offset procurement, and technology investment.', wordCount: 180, lastEdited: '2025-02-05', complianceNotes: ['Add specific risk response details for each material risk'] },
  { id: '8', pillar: 'Risk Management', code: 'RM-C', title: 'ERM Integration', description: 'Describe how processes for identifying, assessing, and managing climate-related risks are integrated into the organization\'s overall risk management.', status: 'not_started', content: '', wordCount: 0, lastEdited: '', complianceNotes: ['Section not yet started', 'Critical for TCFD compliance'] },
  { id: '9', pillar: 'Metrics & Targets', code: 'MT-A', title: 'Climate Metrics', description: 'Disclose the metrics used by the organization to assess climate-related risks and opportunities.', status: 'complete', content: 'The organization tracks the following climate metrics: GHG emissions (Scope 1, 2, 3) following GHG Protocol; emissions intensity per $M revenue, per FTE, and per sqm; weighted average carbon intensity (WACI) for investment portfolio; internal carbon price ($85/tCO2e); climate value-at-risk across RCP scenarios; and percentage of revenue from low-carbon products.', wordCount: 220, lastEdited: '2025-02-04', complianceNotes: [] },
  { id: '10', pillar: 'Metrics & Targets', code: 'MT-B', title: 'GHG Emissions', description: 'Disclose Scope 1, Scope 2, and Scope 3 greenhouse gas emissions.', status: 'complete', content: 'Scope 1 Direct Emissions: 12,500 tCO2e (stationary combustion, mobile sources, fugitive emissions, process emissions). Scope 2 Emissions: 8,200 tCO2e (market-based) / 9,100 tCO2e (location-based). Scope 3 Indirect Emissions: 85,000 tCO2e across 15 categories, with Category 1 (Purchased Goods & Services) representing 37.6% of total Scope 3. Methodology follows GHG Protocol Corporate Standard and Scope 3 Standard.', wordCount: 275, lastEdited: '2025-02-03', complianceNotes: [] },
  { id: '11', pillar: 'Metrics & Targets', code: 'MT-C', title: 'Targets & Performance', description: 'Describe the targets used by the organization to manage climate-related risks and opportunities and performance against targets.', status: 'in_progress', content: 'The organization has set the following climate targets: Net Zero by 2050 (all scopes, SBTi validated); 42% reduction in Scope 1+2 emissions by 2030 from 2019 baseline (SBTi validated); 25% reduction in Scope 3 emissions by 2030; 100% renewable electricity by 2030; 80% fleet electrification by 2028.', wordCount: 155, lastEdited: '2025-02-01', complianceNotes: ['Add performance against each target', 'Include interim milestones'] },
];

const COMPLIANCE_CHECKS = [
  { requirement: 'Board oversight described', section: 'GOV-A', met: true },
  { requirement: 'Management role defined', section: 'GOV-B', met: true },
  { requirement: 'Short/medium/long-term risks identified', section: 'STR-A', met: true },
  { requirement: 'Financial impact quantified', section: 'STR-B', met: false },
  { requirement: 'Scenario analysis (2C or below)', section: 'STR-C', met: false },
  { requirement: 'Risk identification process described', section: 'RM-A', met: true },
  { requirement: 'Risk management process described', section: 'RM-B', met: true },
  { requirement: 'ERM integration described', section: 'RM-C', met: false },
  { requirement: 'Scope 1, 2, 3 emissions disclosed', section: 'MT-B', met: true },
  { requirement: 'Climate targets set and tracked', section: 'MT-C', met: true },
  { requirement: 'Intensity metrics reported', section: 'MT-A', met: true },
  { requirement: 'All 11 recommended disclosures addressed', section: 'All', met: false },
];

/* ── Component ─────────────────────────────────────────────────── */

const DisclosureBuilder: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('1');
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [previewOpen, setPreviewOpen] = useState(false);

  const completedSections = SECTIONS.filter((s) => s.status === 'complete').length;
  const inProgressSections = SECTIONS.filter((s) => s.status === 'in_progress').length;
  const totalWordCount = SECTIONS.reduce((sum, s) => sum + s.wordCount, 0);
  const complianceMet = COMPLIANCE_CHECKS.filter((c) => c.met).length;
  const complianceScore = Math.round((complianceMet / COMPLIANCE_CHECKS.length) * 100);

  const currentSection = SECTIONS.find((s) => s.id === activeSection) || SECTIONS[0];

  const handleEditOpen = () => {
    setEditContent(currentSection.content);
    setEditDialogOpen(true);
  };

  const pillarColors: Record<string, string> = {
    Governance: '#1B5E20',
    Strategy: '#0D47A1',
    'Risk Management': '#E65100',
    'Metrics & Targets': '#4527A0',
  };

  const getStatusIcon = (status: string) => {
    if (status === 'complete') return <CheckCircle sx={{ color: 'success.main' }} fontSize="small" />;
    if (status === 'in_progress') return <Edit sx={{ color: 'warning.main' }} fontSize="small" />;
    return <RadioButtonUnchecked sx={{ color: 'text.disabled' }} fontSize="small" />;
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Disclosure Builder</Typography>
        <Typography variant="body2" color="text.secondary">
          Build, edit, and validate TCFD-aligned climate disclosure across all 11 recommended disclosures
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Sections Complete"
            value={completedSections}
            icon={<CheckCircle />}
            subtitle={`of ${SECTIONS.length} total`}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="In Progress"
            value={inProgressSections}
            icon={<Edit />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Word Count"
            value={totalWordCount}
            icon={<Description />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Compliance Score"
            value={complianceScore}
            format="percent"
            icon={<Warning />}
            color={complianceScore >= 80 ? 'success' : complianceScore >= 60 ? 'warning' : 'error'}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Sidebar - Section Checklist */}
        <Grid item xs={12} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ p: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, p: 1 }}>Disclosure Sections</Typography>
              <List dense sx={{ maxHeight: 600, overflow: 'auto' }}>
                {['Governance', 'Strategy', 'Risk Management', 'Metrics & Targets'].map((pillar) => (
                  <React.Fragment key={pillar}>
                    <Typography
                      variant="caption"
                      sx={{
                        display: 'block',
                        px: 2,
                        pt: 1.5,
                        pb: 0.5,
                        fontWeight: 700,
                        color: pillarColors[pillar],
                        textTransform: 'uppercase',
                        fontSize: '0.65rem',
                        letterSpacing: 1,
                      }}
                    >
                      {pillar}
                    </Typography>
                    {SECTIONS.filter((s) => s.pillar === pillar).map((section) => (
                      <ListItemButton
                        key={section.id}
                        selected={activeSection === section.id}
                        onClick={() => setActiveSection(section.id)}
                        sx={{ borderRadius: 1, mx: 0.5, py: 0.5 }}
                      >
                        <ListItemIcon sx={{ minWidth: 28 }}>
                          {getStatusIcon(section.status)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: activeSection === section.id ? 600 : 400 }}>
                              {section.code}: {section.title}
                            </Typography>
                          }
                        />
                      </ListItemButton>
                    ))}
                  </React.Fragment>
                ))}
              </List>
              <Box sx={{ p: 1, mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Overall Progress
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(completedSections / SECTIONS.length) * 100}
                  sx={{ height: 8, borderRadius: 4, mt: 0.5 }}
                  color="success"
                />
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {completedSections}/{SECTIONS.length} complete
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Main Content - Section Editor */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Chip
                      label={currentSection.pillar}
                      size="small"
                      sx={{
                        backgroundColor: pillarColors[currentSection.pillar],
                        color: 'white',
                        fontWeight: 600,
                        fontSize: '0.7rem',
                      }}
                    />
                    <Chip
                      label={currentSection.status.replace(/_/g, ' ')}
                      size="small"
                      color={currentSection.status === 'complete' ? 'success' : currentSection.status === 'in_progress' ? 'warning' : 'default'}
                      sx={{ textTransform: 'capitalize' }}
                    />
                  </Box>
                  <Typography variant="h6">{currentSection.code}: {currentSection.title}</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    {currentSection.description}
                  </Typography>
                </Box>
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<Edit />}
                  onClick={handleEditOpen}
                >
                  Edit
                </Button>
              </Box>

              <Divider sx={{ mb: 2 }} />

              {currentSection.content ? (
                <Box sx={{ p: 2, backgroundColor: '#FAFAFA', borderRadius: 1, minHeight: 200 }}>
                  <Typography variant="body2" sx={{ lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                    {currentSection.content}
                  </Typography>
                </Box>
              ) : (
                <Alert severity="info" sx={{ minHeight: 200 }}>
                  This section has not been started yet. Click "Edit" to begin drafting your disclosure content.
                </Alert>
              )}

              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  {currentSection.wordCount} words
                  {currentSection.lastEdited && ` | Last edited: ${new Date(currentSection.lastEdited).toLocaleDateString()}`}
                </Typography>
              </Box>

              {currentSection.complianceNotes.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Compliance Notes</Typography>
                  {currentSection.complianceNotes.map((note, idx) => (
                    <Alert key={idx} severity="warning" sx={{ mb: 0.5, py: 0 }}>
                      <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>{note}</Typography>
                    </Alert>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Right Panel - Compliance Checker + Export */}
        <Grid item xs={12} md={3}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Compliance Checker</Typography>
              <Box sx={{ mb: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={complianceScore}
                  sx={{ height: 10, borderRadius: 5 }}
                  color={complianceScore >= 80 ? 'success' : complianceScore >= 60 ? 'warning' : 'error'}
                />
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  {complianceMet}/{COMPLIANCE_CHECKS.length} requirements met ({complianceScore}%)
                </Typography>
              </Box>
              <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                {COMPLIANCE_CHECKS.map((check, idx) => (
                  <Box key={idx} sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.5, mb: 0.5 }}>
                    {check.met ? (
                      <CheckCircle sx={{ color: 'success.main', fontSize: 16, mt: 0.3 }} />
                    ) : (
                      <RadioButtonUnchecked sx={{ color: 'error.main', fontSize: 16, mt: 0.3 }} />
                    )}
                    <Box>
                      <Typography variant="body2" sx={{ fontSize: '0.75rem', lineHeight: 1.3 }}>
                        {check.requirement}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                        {check.section}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </List>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>Export Disclosure</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button variant="outlined" startIcon={<PictureAsPdf />} fullWidth size="small">
                  Export PDF
                </Button>
                <Button variant="outlined" startIcon={<TableChart />} fullWidth size="small">
                  Export Excel
                </Button>
                <Button variant="outlined" startIcon={<Code />} fullWidth size="small">
                  Export JSON
                </Button>
                <Button variant="outlined" startIcon={<Visibility />} fullWidth size="small" onClick={() => setPreviewOpen(true)}>
                  Preview Report
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Edit: {currentSection.code} - {currentSection.title}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {currentSection.description}
          </Typography>
          <TextField
            multiline
            rows={12}
            fullWidth
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            variant="outlined"
            placeholder="Enter your disclosure content here..."
            sx={{ '& .MuiInputBase-root': { fontFamily: 'Georgia, serif', fontSize: '0.95rem', lineHeight: 1.8 } }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Word count: {editContent.split(/\s+/).filter(Boolean).length}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setEditDialogOpen(false)}>Save Draft</Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog open={previewOpen} onClose={() => setPreviewOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>TCFD Disclosure Report Preview</DialogTitle>
        <DialogContent>
          {['Governance', 'Strategy', 'Risk Management', 'Metrics & Targets'].map((pillar) => (
            <Box key={pillar} sx={{ mb: 3 }}>
              <Typography variant="h5" sx={{ color: pillarColors[pillar], mb: 1 }}>{pillar}</Typography>
              <Divider sx={{ mb: 2 }} />
              {SECTIONS.filter((s) => s.pillar === pillar).map((section) => (
                <Box key={section.id} sx={{ mb: 2 }}>
                  <Typography variant="h6" sx={{ mb: 0.5 }}>
                    {section.code}: {section.title}
                  </Typography>
                  {section.content ? (
                    <Typography variant="body2" sx={{ lineHeight: 1.8, color: 'text.secondary' }}>
                      {section.content}
                    </Typography>
                  ) : (
                    <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'error.main' }}>
                      [Section not yet drafted]
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>Close</Button>
          <Button variant="contained" startIcon={<PictureAsPdf />}>Export PDF</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DisclosureBuilder;
