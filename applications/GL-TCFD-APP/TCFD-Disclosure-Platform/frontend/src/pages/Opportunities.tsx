/**
 * Opportunities - Pipeline kanban, revenue sizing chart, priority matrix scatter plot.
 */

import React, { useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip, Paper } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ScatterChart, Scatter, Cell, ZAxis } from 'recharts';
import { TrendingUp, EmojiObjects, AttachMoney, Star } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';
import StatusBadge from '../components/common/StatusBadge';

interface KanbanItem {
  id: string;
  name: string;
  type: string;
  value: number;
  status: string;
}

const PIPELINE: Record<string, KanbanItem[]> = {
  Identified: [
    { id: '1', name: 'Carbon offset marketplace', type: 'markets', value: 10000000, status: 'identified' },
    { id: '2', name: 'Scope 3 SaaS platform', type: 'products_services', value: 12200000, status: 'identified' },
  ],
  Evaluating: [
    { id: '3', name: 'Green product line', type: 'products_services', value: 20000000, status: 'evaluating' },
    { id: '4', name: 'Climate consulting', type: 'products_services', value: 9000000, status: 'evaluating' },
  ],
  Approved: [
    { id: '5', name: 'Energy efficiency retrofits', type: 'resource_efficiency', value: -1800000, status: 'approved' },
  ],
  Implementing: [
    { id: '6', name: 'Renewable PPA', type: 'energy_source', value: 6500000, status: 'implementing' },
  ],
};

const REVENUE_SIZING = [
  { type: 'Products & Services', low: 15, mid: 43, high: 65 },
  { type: 'Energy Source', low: 5, mid: 8.5, high: 12 },
  { type: 'Resource Efficiency', low: 2, mid: 4.2, high: 7 },
  { type: 'Markets', low: 8, mid: 15, high: 22 },
  { type: 'Resilience', low: 1, mid: 3, high: 6 },
];

const PRIORITY_MATRIX = [
  { id: '1', name: 'Green product line', impact: 85, feasibility: 70, size: 20, type: 'products_services' },
  { id: '2', name: 'Renewable PPA', impact: 60, feasibility: 90, size: 8.5, type: 'energy_source' },
  { id: '3', name: 'Energy retrofits', impact: 45, feasibility: 80, size: 4.2, type: 'resource_efficiency' },
  { id: '4', name: 'Carbon marketplace', impact: 75, feasibility: 50, size: 15, type: 'markets' },
  { id: '5', name: 'Climate consulting', impact: 65, feasibility: 65, size: 9, type: 'products_services' },
  { id: '6', name: 'Scope 3 SaaS', impact: 80, feasibility: 55, size: 12.2, type: 'products_services' },
];

const TYPE_COLORS: Record<string, string> = {
  products_services: '#7B1FA2',
  energy_source: '#0D47A1',
  resource_efficiency: '#1B5E20',
  markets: '#EF6C00',
  resilience: '#00838F',
};

const STAGE_COLORS: Record<string, string> = {
  Identified: '#E0E0E0',
  Evaluating: '#BBDEFB',
  Approved: '#C8E6C9',
  Implementing: '#FFF9C4',
};

const Opportunities: React.FC = () => {
  const totalPipeline = Object.values(PIPELINE).flat().reduce((s, i) => s + Math.max(i.value, 0), 0);

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Opportunity Pipeline</Typography>
        <Typography variant="body2" color="text.secondary">
          Climate opportunity assessment, sizing, and prioritization
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Pipeline Value" value={totalPipeline} format="currency" icon={<TrendingUp />} color="success" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Total Opportunities" value={Object.values(PIPELINE).flat().length} icon={<EmojiObjects />} color="primary" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Revenue Potential (Mid)" value={71200000} format="currency" icon={<AttachMoney />} color="secondary" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Avg Feasibility" value={68} format="percent" icon={<Star />} color="info" />
        </Grid>
      </Grid>

      {/* Kanban Pipeline */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>Opportunity Pipeline</Typography>
          <Box sx={{ display: 'flex', gap: 2, overflowX: 'auto', pb: 1 }}>
            {Object.entries(PIPELINE).map(([stage, items]) => (
              <Paper key={stage} elevation={0} sx={{
                minWidth: 260, flex: '1 0 260px', p: 2,
                backgroundColor: STAGE_COLORS[stage] || '#F5F5F5',
                borderRadius: 2,
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{stage}</Typography>
                  <Chip label={items.length} size="small" />
                </Box>
                {items.map((item) => (
                  <Paper key={item.id} sx={{ p: 1.5, mb: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500, mb: 0.5 }}>{item.name}</Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Chip label={item.type.replace(/_/g, ' ')} size="small" sx={{ backgroundColor: TYPE_COLORS[item.type] || '#9E9E9E', color: 'white', fontSize: '0.65rem', height: 20, textTransform: 'capitalize' }} />
                      <Typography variant="caption" sx={{ fontWeight: 600, color: item.value >= 0 ? 'success.main' : 'error.main' }}>
                        ${(item.value / 1e6).toFixed(1)}M
                      </Typography>
                    </Box>
                  </Paper>
                ))}
              </Paper>
            ))}
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Revenue Sizing */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Revenue Sizing by Type ($M -- Low/Mid/High)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={REVENUE_SIZING}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" fontSize={10} />
                  <YAxis tickFormatter={(v) => `$${v}M`} />
                  <Tooltip formatter={(v: number) => [`$${v}M`, '']} />
                  <Legend />
                  <Bar dataKey="low" name="Low" fill="#C8E6C9" />
                  <Bar dataKey="mid" name="Mid" fill="#4CAF50" />
                  <Bar dataKey="high" name="High" fill="#1B5E20" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Priority Matrix */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Priority Matrix (Impact vs Feasibility)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="feasibility" name="Feasibility" domain={[30, 100]} unit="%" />
                  <YAxis type="number" dataKey="impact" name="Impact" domain={[30, 100]} unit="%" />
                  <ZAxis type="number" dataKey="size" range={[80, 400]} name="Value ($M)" />
                  <Tooltip formatter={(value: number, name: string) => [name === 'Value ($M)' ? `$${value}M` : `${value}%`, name]} />
                  <Scatter name="Opportunities" data={PRIORITY_MATRIX}>
                    {PRIORITY_MATRIX.map((entry) => (
                      <Cell key={entry.id} fill={TYPE_COLORS[entry.type] || '#9E9E9E'} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', gap: 2, mt: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
                {Object.entries(TYPE_COLORS).map(([type, color]) => (
                  <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: color }} />
                    <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>{type.replace(/_/g, ' ')}</Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Opportunities;
