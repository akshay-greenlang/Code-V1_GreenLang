/**
 * FrameworkAlignment - Cross-framework alignment table.
 */
import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { AlignmentItem } from '../../types';

interface FrameworkAlignmentProps { items: AlignmentItem[]; }

const COLORS: Record<string, string> = { sbti: '#1B5E20', ghg_protocol: '#0D47A1', cdp: '#7B1FA2', tcfd: '#EF6C00', csrd: '#C62828', iso_14064: '#006064', issb: '#33691E', sec_climate: '#880E4F' };

const FrameworkAlignmentComponent: React.FC<FrameworkAlignmentProps> = ({ items }) => {
  const data = items.map((i) => ({ name: i.framework.toUpperCase().replace(/_/g, ' '), alignment: i.alignment_pct, met: i.met, partial: i.partial, not_met: i.not_met }));
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Framework Alignment</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={10} />
            <YAxis domain={[0, 100]} fontSize={11} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(0)}%`, 'Alignment']} />
            <Bar dataKey="alignment" name="Alignment %">
              {data.map((d, i) => <Cell key={i} fill={Object.values(COLORS)[i % Object.values(COLORS).length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default FrameworkAlignmentComponent;
