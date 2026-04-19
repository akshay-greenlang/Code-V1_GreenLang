import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
} from 'recharts';
import { Box, Typography, Paper } from '@mui/material';

const COLORS = ['#1976d2', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#ffeb3b', '#795548'];

interface ChartProps {
  data: any[];
  title?: string;
  height?: number;
}

export const CategoryPieChart: React.FC<ChartProps> = ({ data, title, height = 300 }) => {
  return (
    <Paper sx={{ p: 2 }}>
      {title && <Typography variant="h6" gutterBottom>{title}</Typography>}
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            dataKey="emissionsTCO2e"
            nameKey="categoryName"
            cx="50%"
            cy="50%"
            outerRadius={80}
            label={(entry) => `${entry.categoryName}: ${entry.percentage.toFixed(1)}%`}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => `${value.toFixed(2)} t CO₂e`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const MonthlyTrendChart: React.FC<ChartProps> = ({ data, title, height = 300 }) => {
  return (
    <Paper sx={{ p: 2 }}>
      {title && <Typography variant="h6" gutterBottom>{title}</Typography>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis yAxisId="left" label={{ value: 't CO₂e', angle: -90, position: 'insideLeft' }} />
          <YAxis yAxisId="right" orientation="right" label={{ value: 'USD', angle: 90, position: 'insideRight' }} />
          <Tooltip />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="emissionsTCO2e"
            stroke="#4caf50"
            strokeWidth={2}
            name="Emissions (t CO₂e)"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="spendUsd"
            stroke="#1976d2"
            strokeWidth={2}
            name="Spend (USD)"
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export const TopSuppliersChart: React.FC<ChartProps> = ({ data, title, height = 300 }) => {
  return (
    <Paper sx={{ p: 2 }}>
      {title && <Typography variant="h6" gutterBottom>{title}</Typography>}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" label={{ value: 't CO₂e', position: 'insideBottom', offset: -5 }} />
          <YAxis dataKey="supplierName" type="category" width={150} />
          <Tooltip formatter={(value: number) => `${value.toFixed(2)} t CO₂e`} />
          <Bar dataKey="emissionsTCO2e" fill="#1976d2" />
        </BarChart>
      </ResponsiveContainer>
    </Paper>
  );
};
