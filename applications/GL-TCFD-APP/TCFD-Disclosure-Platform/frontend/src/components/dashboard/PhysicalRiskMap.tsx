import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import type { AssetLocation, RiskLevel } from '../../types';
import 'leaflet/dist/leaflet.css';

interface PhysicalRiskMapProps {
  assets: AssetLocation[];
}

const RISK_COLORS: Record<RiskLevel, string> = {
  critical: '#B71C1C',
  high: '#E65100',
  medium: '#F57F17',
  low: '#2E7D32',
  negligible: '#388E3C',
};

const PhysicalRiskMap: React.FC<PhysicalRiskMapProps> = ({ assets }) => {
  const center: [number, number] = assets.length > 0
    ? [assets[0].latitude, assets[0].longitude]
    : [20, 0];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
        <Typography variant="h6" sx={{ fontWeight: 600, p: 2, pb: 1 }}>
          Physical Risk Exposure Map
        </Typography>
        <Box sx={{ height: 350, position: 'relative' }}>
          <MapContainer
            center={center}
            zoom={2}
            style={{ height: '100%', width: '100%' }}
            scrollWheelZoom={false}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            />
            {assets.map((asset) => (
              <CircleMarker
                key={asset.id}
                center={[asset.latitude, asset.longitude]}
                radius={Math.max(6, Math.min(20, asset.overall_risk_score / 5))}
                pathOptions={{
                  fillColor: RISK_COLORS[asset.risk_level] || '#9E9E9E',
                  fillOpacity: 0.7,
                  color: '#FFFFFF',
                  weight: 2,
                }}
              >
                <Popup>
                  <Box sx={{ minWidth: 160 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{asset.name}</Typography>
                    <Typography variant="body2">Type: {asset.asset_type}</Typography>
                    <Typography variant="body2">Country: {asset.country}</Typography>
                    <Typography variant="body2">
                      Risk Score: {asset.overall_risk_score.toFixed(0)}
                    </Typography>
                    <Typography variant="body2">
                      Book Value: ${(asset.book_value / 1_000_000).toFixed(1)}M
                    </Typography>
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

export default PhysicalRiskMap;
