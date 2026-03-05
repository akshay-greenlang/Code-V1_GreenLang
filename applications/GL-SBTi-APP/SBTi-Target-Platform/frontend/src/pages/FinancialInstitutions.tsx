/**
 * FinancialInstitutions - SBTi for Financial Institutions (FI) dashboard.
 *
 * Portfolio coverage, financed emissions, PCAF data quality, WACI trend,
 * and engagement tracker for investees.
 */

import React, { useEffect } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Chip, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  LinearProgress,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, AreaChart, Area,
} from 'recharts';
import { AccountBalance, TrendingDown } from '@mui/icons-material';
import PortfolioCoverage from '../components/fi/PortfolioCoverage';
import FinancedEmissions from '../components/fi/FinancedEmissions';
import PCAFQuality from '../components/fi/PCAFQuality';
import WACITrend from '../components/fi/WACITrend';
import EngagementTracker from '../components/fi/EngagementTracker';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchPortfolios, fetchFICoverage, fetchFinancedEmissions, fetchWACI,
  selectPortfolios, selectFICoverage, selectFinancedEmissions, selectWACI, selectFILoading,
} from '../store/slices/fiSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_COVERAGE = {
  overall_pct: 62,
  by_asset_class: [
    { asset_class: 'Listed Equity', coverage_pct: 75, target_2030: 100, target_2040: 100 },
    { asset_class: 'Corporate Bonds', coverage_pct: 68, target_2030: 100, target_2040: 100 },
    { asset_class: 'Project Finance', coverage_pct: 55, target_2030: 90, target_2040: 100 },
    { asset_class: 'Commercial Real Estate', coverage_pct: 42, target_2030: 80, target_2040: 100 },
    { asset_class: 'Mortgages', coverage_pct: 30, target_2030: 70, target_2040: 95 },
    { asset_class: 'Motor Vehicle Loans', coverage_pct: 20, target_2030: 60, target_2040: 90 },
  ],
  path_to_100: [
    { year: 2024, coverage_pct: 62 },
    { year: 2025, coverage_pct: 68 },
    { year: 2027, coverage_pct: 78 },
    { year: 2030, coverage_pct: 90 },
    { year: 2035, coverage_pct: 95 },
    { year: 2040, coverage_pct: 100 },
  ],
};

const DEMO_FINANCED = {
  total: 2450000,
  by_asset_class: [
    { asset_class: 'Listed Equity', emissions: 820000, pct: 33.5 },
    { asset_class: 'Corporate Bonds', emissions: 650000, pct: 26.5 },
    { asset_class: 'Project Finance', emissions: 420000, pct: 17.1 },
    { asset_class: 'Commercial Real Estate', emissions: 310000, pct: 12.7 },
    { asset_class: 'Mortgages', emissions: 180000, pct: 7.3 },
    { asset_class: 'Motor Vehicle Loans', emissions: 70000, pct: 2.9 },
  ],
  by_sector: [
    { sector: 'Energy', emissions: 680000, pct: 27.8 },
    { sector: 'Materials', emissions: 520000, pct: 21.2 },
    { sector: 'Industrials', emissions: 410000, pct: 16.7 },
    { sector: 'Utilities', emissions: 380000, pct: 15.5 },
    { sector: 'Real Estate', emissions: 280000, pct: 11.4 },
    { sector: 'Other', emissions: 180000, pct: 7.3 },
  ],
};

const DEMO_WACI = {
  current_waci: 142,
  trend: [
    { year: 2020, waci: 195 },
    { year: 2021, waci: 178 },
    { year: 2022, waci: 165 },
    { year: 2023, waci: 155 },
    { year: 2024, waci: 142 },
  ],
  benchmark: 168,
};

const DEMO_ENGAGEMENTS = [
  { id: '1', company_name: 'EnergyMax Corp', engagement_date: '2025-02-15', engagement_type: 'direct_dialogue', outcome: 'Committed to set SBTi targets', sbti_commitment_obtained: true, follow_up_date: '2025-06-15' },
  { id: '2', company_name: 'SteelWorks Inc', engagement_date: '2025-01-20', engagement_type: 'collaborative_engagement', outcome: 'In process of target development', sbti_commitment_obtained: false, follow_up_date: '2025-04-20' },
  { id: '3', company_name: 'ChemPlus Ltd', engagement_date: '2024-12-10', engagement_type: 'proxy_voting', outcome: 'Voted for climate disclosure', sbti_commitment_obtained: false, follow_up_date: '2025-03-10' },
  { id: '4', company_name: 'AutoDrive Holdings', engagement_date: '2024-11-05', engagement_type: 'direct_dialogue', outcome: 'SBTi targets validated', sbti_commitment_obtained: true, follow_up_date: null },
];

const SECTOR_COLORS = ['#C62828', '#E65100', '#EF6C00', '#F9A825', '#1B5E20', '#9E9E9E'];
const ASSET_COLORS = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#64B5F6'];

const FinancialInstitutions: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const coverage = useAppSelector(selectFICoverage);
  const financedEmissions = useAppSelector(selectFinancedEmissions);
  const waci = useAppSelector(selectWACI);
  const loading = useAppSelector(selectFILoading);

  useEffect(() => {
    dispatch(fetchPortfolios(orgId));
    dispatch(fetchFICoverage(orgId));
    dispatch(fetchFinancedEmissions(orgId));
    dispatch(fetchWACI(orgId));
  }, [dispatch, orgId]);

  const coverageData = coverage || DEMO_COVERAGE;
  const financedData = financedEmissions || DEMO_FINANCED;
  const waciData = waci || DEMO_WACI;

  if (loading && !coverage) return <LoadingSpinner message="Loading FI dashboard..." />;

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Financial Institutions</Typography>
        <Typography variant="body2" color="text.secondary">
          SBTi for Financial Institutions - portfolio coverage, financed emissions, and engagement tracking
        </Typography>
      </Box>

      {/* KPI Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Portfolio Coverage</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: coverageData.overall_pct >= 50 ? '#2E7D32' : '#C62828' }}>
                {coverageData.overall_pct}%
              </Typography>
              <Typography variant="caption" color="text.secondary">of AUM with SBTi targets</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Financed Emissions</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>
                {(financedData.total / 1e6).toFixed(1)}M
              </Typography>
              <Typography variant="caption" color="text.secondary">tCO2e total</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">WACI</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: waciData.current_waci <= waciData.benchmark ? '#2E7D32' : '#C62828' }}>
                {waciData.current_waci}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                tCO2e/$M (benchmark: {waciData.benchmark})
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary">Engagements</Typography>
              <Typography variant="h3" sx={{ fontWeight: 700 }}>
                {DEMO_ENGAGEMENTS.filter((e) => e.sbti_commitment_obtained).length}/{DEMO_ENGAGEMENTS.length}
              </Typography>
              <Typography variant="caption" color="text.secondary">commitments obtained</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Coverage + WACI Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Coverage by Asset Class</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={coverageData.by_asset_class} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="asset_class" width={140} fontSize={11} />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                  <Legend />
                  <Bar dataKey="coverage_pct" name="Current" fill="#1B5E20" barSize={10} />
                  <Bar dataKey="target_2030" name="2030 Target" fill="#E0E0E0" barSize={10} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>WACI Trend (tCO2e/$M)</Typography>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={waciData.trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis domain={[0, 250]} />
                  <Tooltip formatter={(value: number) => [`${value} tCO2e/$M`, 'WACI']} />
                  <Line type="monotone" dataKey="waci" stroke="#1B5E20" strokeWidth={2.5} dot={{ r: 4 }} name="WACI" />
                </LineChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1, gap: 2 }}>
                <Chip label={`Current: ${waciData.current_waci}`} size="small" color="primary" />
                <Chip label={`Benchmark: ${waciData.benchmark}`} size="small" variant="outlined" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Financed Emissions */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Financed Emissions by Sector</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={financedData.by_sector} cx="50%" cy="50%" innerRadius={55} outerRadius={85} dataKey="emissions"
                    label={({ sector, pct }) => `${sector}: ${pct}%`}
                  >
                    {financedData.by_sector.map((_: any, idx: number) => <Cell key={idx} fill={SECTOR_COLORS[idx]} />)}
                  </Pie>
                  <Tooltip formatter={(value: number) => [value.toLocaleString() + ' tCO2e', '']} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Path to 100% Coverage</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={coverageData.path_to_100}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(value: number) => [`${value}%`, 'Coverage']} />
                  <Area type="monotone" dataKey="coverage_pct" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.2} name="Coverage" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Engagements */}
      <EngagementTracker engagements={DEMO_ENGAGEMENTS as any} />
    </Box>
  );
};

export default FinancialInstitutions;
