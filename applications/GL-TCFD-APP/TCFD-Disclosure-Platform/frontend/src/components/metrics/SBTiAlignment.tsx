import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface SBTiAlignmentProps { data: { target_id: string; target_name: string; sbti_pathway: number[]; actual: number[]; aligned: boolean }[]; }

const SBTiAlignment: React.FC<SBTiAlignmentProps> = ({ data }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>SBTi Pathway Alignment</Typography>
    {data.map((target) => {
      const chartData = target.actual.map((v, i) => ({
        year: (2020 + i).toString(),
        actual: v,
        pathway: target.sbti_pathway[i] || 0,
      }));
      return (
        <Box key={target.target_id} sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{target.target_name}</Typography>
            <Chip label={target.aligned ? 'Aligned' : 'Not Aligned'} size="small" color={target.aligned ? 'success' : 'error'} />
          </Box>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="year" /><YAxis /><Tooltip /><Legend />
              <Line type="monotone" dataKey="pathway" name="SBTi Pathway" stroke="#1B5E20" strokeWidth={2} strokeDasharray="5 5" />
              <Line type="monotone" dataKey="actual" name="Actual" stroke="#0D47A1" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      );
    })}
    {data.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No SBTi alignment data</Typography>}
  </CardContent></Card>
);

export default SBTiAlignment;
