/**
 * PriorityMatrix - Impact vs effort matrix (scatter plot)
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';
import type { GapItem, GapEffort } from '../../types';

interface PriorityMatrixProps { gaps: GapItem[]; }

const effortToValue = (e: GapEffort | string): number => {
  if (e === 'low') return 1; if (e === 'medium') return 2; return 3;
};

const PriorityMatrix: React.FC<PriorityMatrixProps> = ({ gaps }) => {
  const data = gaps.filter((g) => !g.is_resolved).map((g) => ({
    x: effortToValue(g.effort),
    y: g.uplift_points,
    name: g.question_number,
    severity: g.severity,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Priority Matrix (Impact vs Effort)</Typography>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ bottom: 20, left: 20 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="x" domain={[0, 4]} name="Effort" tick={{ fontSize: 10 }} label={{ value: 'Effort (Low to High)', position: 'bottom', fontSize: 11 }} />
            <YAxis type="number" dataKey="y" name="Uplift" label={{ value: 'Score Uplift (pts)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <ZAxis range={[60, 200]} />
            <Tooltip formatter={(value: number, name: string) => [name === 'Effort' ? ['Low', 'Med', 'High'][Math.round(value as number) - 1] : `${value.toFixed(1)} pts`, name]} />
            <Scatter name="Gaps" data={data} fill="#1b5e20" />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default PriorityMatrix;
