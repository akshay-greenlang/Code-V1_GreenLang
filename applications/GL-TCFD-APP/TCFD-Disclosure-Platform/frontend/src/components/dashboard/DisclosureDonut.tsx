import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import type { DisclosureStatus } from '../../types';

interface DisclosureDonutProps {
  sectionStatuses: { code: string; title: string; status: DisclosureStatus }[];
  overallPct: number;
}

const STATUS_COLORS: Record<string, string> = {
  published: '#1B5E20',
  final: '#2E7D32',
  review: '#0D47A1',
  draft: '#F57F17',
  in_progress: '#E65100',
  not_started: '#BDBDBD',
};

const DisclosureDonut: React.FC<DisclosureDonutProps> = ({ sectionStatuses, overallPct }) => {
  const statusCounts = sectionStatuses.reduce<Record<string, number>>((acc, s) => {
    acc[s.status] = (acc[s.status] || 0) + 1;
    return acc;
  }, {});

  const chartData = Object.entries(statusCounts).map(([status, count]) => ({
    name: status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    value: count,
    color: STATUS_COLORS[status] || '#9E9E9E',
  }));

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
          Disclosure Completeness
        </Typography>
        <Box sx={{ position: 'relative' }}>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(v: number) => [`${v} sections`, 'Count']} />
            </PieChart>
          </ResponsiveContainer>
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
            }}
          >
            <Typography variant="h4" sx={{ fontWeight: 700, color: 'primary.main' }}>
              {overallPct.toFixed(0)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Complete
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default DisclosureDonut;
