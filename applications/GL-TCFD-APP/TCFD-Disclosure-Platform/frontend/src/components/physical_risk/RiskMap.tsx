import React from 'react';
import { Card, CardContent, Typography, Box, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import type { AssetLocation, RiskLevel, HazardType } from '../../types';

interface RiskMapProps { assets: AssetLocation[]; }

const RISK_COLORS: Record<RiskLevel, string> = { critical: '#B71C1C', high: '#E65100', medium: '#F57F17', low: '#2E7D32', negligible: '#388E3C' };

const RiskMap: React.FC<RiskMapProps> = ({ assets }) => {
  const [hazardFilter, setHazardFilter] = React.useState<string>('all');
  const filtered = hazardFilter === 'all' ? assets : assets.filter((a) => a.hazards.some((h) => h.hazard_type === hazardFilter));
  const center: [number, number] = assets.length > 0 ? [assets[0].latitude, assets[0].longitude] : [20, 0];

  return (
    <Card>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Physical Risk Map</Typography>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Hazard Type</InputLabel>
            <Select value={hazardFilter} label="Hazard Type" onChange={(e: SelectChangeEvent) => setHazardFilter(e.target.value)}>
              <MenuItem value="all">All Hazards</MenuItem>
              {['flood', 'drought', 'wildfire', 'tropical_cyclone', 'extreme_heat', 'sea_level_rise'].map((h) => (
                <MenuItem key={h} value={h}>{h.replace(/_/g, ' ')}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        <Box sx={{ height: 450 }}>
          <MapContainer center={center} zoom={2} style={{ height: '100%', width: '100%' }} scrollWheelZoom>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; OpenStreetMap' />
            {filtered.map((asset) => (
              <CircleMarker key={asset.id} center={[asset.latitude, asset.longitude]}
                radius={Math.max(6, Math.min(20, asset.overall_risk_score / 5))}
                pathOptions={{ fillColor: RISK_COLORS[asset.risk_level] || '#9E9E9E', fillOpacity: 0.7, color: '#FFF', weight: 2 }}>
                <Popup>
                  <Box sx={{ minWidth: 180 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{asset.name}</Typography>
                    <Typography variant="body2">Type: {asset.asset_type}</Typography>
                    <Typography variant="body2">Risk Score: {asset.overall_risk_score.toFixed(0)}</Typography>
                    <Typography variant="body2">Book Value: ${(asset.book_value / 1e6).toFixed(1)}M</Typography>
                    <Typography variant="body2">Hazards: {asset.hazards.length}</Typography>
                  </Box>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RiskMap;
