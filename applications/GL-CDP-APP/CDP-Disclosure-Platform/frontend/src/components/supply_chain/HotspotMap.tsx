/**
 * HotspotMap - Emission hotspot visualization (bar chart)
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { EmissionHotspot } from '../../types';

interface HotspotMapProps { hotspots: EmissionHotspot[]; }

const HotspotMap: React.FC<HotspotMapProps> = ({ hotspots }) => {
  const data = hotspots.map((h) => ({ name: h.category, emissions: h.total_emissions, suppliers: h.supplier_count, pct: h.percentage_of_total }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Emission Hotspots</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
            <YAxis />
            <Tooltip formatter={(value: number, name: string) => [name === 'emissions' ? `${value.toFixed(0)} tCO2e` : value, name === 'emissions' ? 'Emissions' : 'Suppliers']} />
            <Bar dataKey="emissions" fill="#e53935" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default HotspotMap;
