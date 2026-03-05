/**
 * GL-ISO14064-APP v1.0 - Facility Emissions Map (Placeholder)
 *
 * Placeholder component for the facility-level emissions map.
 * In production this would integrate with Leaflet or Mapbox GL
 * to render facility locations with proportional emission bubbles.
 * Currently renders a data summary table by facility.
 */

import React from 'react';
import {
  Card, CardContent, Typography, Box,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow,
} from '@mui/material';
import MapIcon from '@mui/icons-material/Map';

interface FacilityEntry {
  facility_id: string;
  name: string;
  tco2e: number;
}

interface Props {
  facilities: FacilityEntry[];
  title?: string;
}

const FacilityMap: React.FC<Props> = ({ facilities = [], title = 'Facility Emissions' }) => {
  if (facilities.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
            {title}
          </Typography>
          <Box
            display="flex"
            flexDirection="column"
            justifyContent="center"
            alignItems="center"
            height={200}
            gap={1}
          >
            <MapIcon sx={{ fontSize: 48, color: 'text.disabled' }} />
            <Typography color="text.secondary">No facility data available</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  const sorted = [...facilities].sort((a, b) => b.tco2e - a.tco2e);
  const total = sorted.reduce((s, f) => s + f.tco2e, 0);

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        <TableContainer sx={{ maxHeight: 300 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Facility</TableCell>
                <TableCell align="right" sx={{ fontWeight: 600 }}>Emissions (tCO2e)</TableCell>
                <TableCell align="right" sx={{ fontWeight: 600 }}>% of Total</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sorted.map((f) => (
                <TableRow key={f.facility_id} hover>
                  <TableCell>{f.name}</TableCell>
                  <TableCell align="right">
                    {f.tco2e.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                  </TableCell>
                  <TableCell align="right">
                    {total > 0 ? ((f.tco2e / total) * 100).toFixed(1) : '0.0'}%
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default FacilityMap;
