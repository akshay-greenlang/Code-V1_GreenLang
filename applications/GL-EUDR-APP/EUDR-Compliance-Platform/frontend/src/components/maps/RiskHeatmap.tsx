/**
 * RiskHeatmap - Leaflet choropleth map showing country-level risk scores.
 *
 * Renders simplified world boundaries colored by risk score, with tooltips
 * displaying country name, score, and top commodities. Uses a continuous
 * green-yellow-orange-red color scale from 0 to 1.
 */

import React, { useMemo, useCallback, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import L, { Layer, PathOptions } from 'leaflet';
import {
  Box,
  Paper,
  Typography,
  Stack,
  Chip,
} from '@mui/material';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface RiskHeatmapProps {
  /** Mapping of ISO-3 country code to overall risk score (0..1). */
  countryRisks: Record<string, number>;
  /** Optional commodity filter shown in header. */
  selectedCommodity?: string;
  /** Country-level detail: top commodities per country. */
  countryDetails?: Record<string, { commodities: string[]; supplierCount: number }>;
  height?: number | string;
}

// ---------------------------------------------------------------------------
// Color scale
// ---------------------------------------------------------------------------

function riskColor(score: number): string {
  if (score <= 0) return '#c8e6c9';
  if (score <= 0.3) {
    const t = score / 0.3;
    return interpolateColor('#4caf50', '#ffeb3b', t);
  }
  if (score <= 0.5) {
    const t = (score - 0.3) / 0.2;
    return interpolateColor('#ffeb3b', '#ff9800', t);
  }
  if (score <= 0.7) {
    const t = (score - 0.5) / 0.2;
    return interpolateColor('#ff9800', '#f44336', t);
  }
  const t = Math.min((score - 0.7) / 0.3, 1);
  return interpolateColor('#f44336', '#b71c1c', t);
}

function interpolateColor(c1: string, c2: string, t: number): string {
  const r1 = parseInt(c1.slice(1, 3), 16);
  const g1 = parseInt(c1.slice(3, 5), 16);
  const b1 = parseInt(c1.slice(5, 7), 16);
  const r2 = parseInt(c2.slice(1, 3), 16);
  const g2 = parseInt(c2.slice(3, 5), 16);
  const b2 = parseInt(c2.slice(5, 7), 16);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// Simplified world GeoJSON (inline minimal country boundaries)
// In production, this would be loaded from a static JSON asset.
// Here we provide a stub structure that renders placeholders.
// ---------------------------------------------------------------------------

/** Placeholder GeoJSON for demo -- in a real build, replace with Natural Earth data. */
function getWorldGeoJSON(): GeoJSON.FeatureCollection {
  // Minimal set of country bounding boxes as simple polygons
  const countries: Array<{ iso3: string; name: string; bounds: number[][] }> = [
    { iso3: 'BRA', name: 'Brazil', bounds: [[-73.99,-33.75],[-73.99,5.27],[-34.79,5.27],[-34.79,-33.75]] },
    { iso3: 'IDN', name: 'Indonesia', bounds: [[95.29,-10.36],[95.29,5.48],[141.03,5.48],[141.03,-10.36]] },
    { iso3: 'COL', name: 'Colombia', bounds: [[-79.0,-4.23],[-79.0,12.45],[-66.87,12.45],[-66.87,-4.23]] },
    { iso3: 'GHA', name: 'Ghana', bounds: [[-3.24,4.74],[-3.24,11.17],[1.19,11.17],[1.19,4.74]] },
    { iso3: 'CIV', name: "Cote d'Ivoire", bounds: [[-8.60,4.36],[-8.60,10.74],[-2.49,10.74],[-2.49,4.36]] },
    { iso3: 'MYS', name: 'Malaysia', bounds: [[99.64,0.85],[99.64,7.36],[119.27,7.36],[119.27,0.85]] },
    { iso3: 'PER', name: 'Peru', bounds: [[-81.33,-18.35],[-81.33,-0.04],[-68.65,-0.04],[-68.65,-18.35]] },
    { iso3: 'COD', name: 'DR Congo', bounds: [[12.18,-13.46],[12.18,5.39],[31.31,5.39],[31.31,-13.46]] },
    { iso3: 'CMR', name: 'Cameroon', bounds: [[8.49,1.65],[8.49,13.08],[16.19,13.08],[16.19,1.65]] },
    { iso3: 'ETH', name: 'Ethiopia', bounds: [[32.99,3.40],[32.99,14.89],[47.99,14.89],[47.99,3.40]] },
    { iso3: 'VNM', name: 'Vietnam', bounds: [[102.14,8.56],[102.14,23.39],[109.46,23.39],[109.46,8.56]] },
    { iso3: 'ARG', name: 'Argentina', bounds: [[-73.58,-55.06],[-73.58,-21.78],[-53.59,-21.78],[-53.59,-55.06]] },
    { iso3: 'PRY', name: 'Paraguay', bounds: [[-62.65,-27.59],[-62.65,-19.29],[-54.24,-19.29],[-54.24,-27.59]] },
    { iso3: 'BOL', name: 'Bolivia', bounds: [[-69.59,-22.90],[-69.59,-9.68],[-57.45,-9.68],[-57.45,-22.90]] },
    { iso3: 'NGA', name: 'Nigeria', bounds: [[2.67,4.27],[2.67,13.89],[14.68,13.89],[14.68,4.27]] },
    { iso3: 'THA', name: 'Thailand', bounds: [[97.35,5.61],[97.35,20.46],[105.64,20.46],[105.64,5.61]] },
    { iso3: 'PNG', name: 'Papua New Guinea', bounds: [[140.84,-10.65],[140.84,-1.35],[156.02,-1.35],[156.02,-10.65]] },
    { iso3: 'LBR', name: 'Liberia', bounds: [[-11.49,4.35],[-11.49,8.55],[-7.37,8.55],[-7.37,4.35]] },
    { iso3: 'GTM', name: 'Guatemala', bounds: [[-92.23,13.74],[-92.23,17.82],[-88.22,17.82],[-88.22,13.74]] },
    { iso3: 'HND', name: 'Honduras', bounds: [[-89.35,12.98],[-89.35,16.51],[-83.13,16.51],[-83.13,12.98]] },
  ];

  return {
    type: 'FeatureCollection',
    features: countries.map((c) => ({
      type: 'Feature' as const,
      properties: { iso3: c.iso3, name: c.name },
      geometry: {
        type: 'Polygon' as const,
        coordinates: [[...c.bounds, c.bounds[0]]],
      },
    })),
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const RiskHeatmap: React.FC<RiskHeatmapProps> = ({
  countryRisks,
  selectedCommodity,
  countryDetails = {},
  height = 450,
}) => {
  const geoJsonRef = useRef<L.GeoJSON | null>(null);
  const worldGeo = useMemo(() => getWorldGeoJSON(), []);

  const styleFeature = useCallback(
    (feature?: GeoJSON.Feature): PathOptions => {
      const iso3 = feature?.properties?.iso3 as string | undefined;
      const score = iso3 ? (countryRisks[iso3] ?? -1) : -1;

      if (score < 0) {
        return {
          fillColor: '#e0e0e0',
          color: '#bdbdbd',
          weight: 1,
          fillOpacity: 0.3,
          opacity: 0.6,
        };
      }

      return {
        fillColor: riskColor(score),
        color: '#fff',
        weight: 1.5,
        fillOpacity: 0.7,
        opacity: 1,
      };
    },
    [countryRisks]
  );

  const onEachFeature = useCallback(
    (feature: GeoJSON.Feature, layer: Layer) => {
      const props = feature.properties;
      if (!props) return;

      const iso3 = props.iso3 as string;
      const name = props.name as string;
      const score = countryRisks[iso3];
      const details = countryDetails[iso3];

      layer.on('mouseover', (e: L.LeafletEvent) => {
        const target = e.target as L.Path;
        target.setStyle({
          weight: 3,
          color: '#333',
          fillOpacity: 0.85,
        });
        target.bringToFront();
      });

      layer.on('mouseout', () => {
        if (geoJsonRef.current) {
          geoJsonRef.current.resetStyle(layer as L.Path);
        }
      });

      const scoreText = score !== undefined ? (score * 100).toFixed(0) + '%' : 'N/A';
      const commoditiesHtml = details?.commodities
        ? details.commodities
            .map((c) => `<span style="background:#e3f2fd;padding:1px 6px;border-radius:4px;font-size:11px;margin-right:4px;text-transform:capitalize">${c.replace('_', ' ')}</span>`)
            .join('')
        : '<span style="color:#999">None tracked</span>';
      const supplierCountText = details?.supplierCount ?? 0;

      layer.bindTooltip(
        `<div style="min-width:180px">
          <strong style="font-size:14px">${name}</strong>
          <hr style="margin:4px 0;border-color:#eee"/>
          <div style="margin-bottom:4px"><span style="color:#666">Risk Score:</span> <strong>${scoreText}</strong></div>
          <div style="margin-bottom:4px"><span style="color:#666">Suppliers:</span> ${supplierCountText}</div>
          <div><span style="color:#666">Commodities:</span><br/>${commoditiesHtml}</div>
        </div>`,
        { sticky: true, direction: 'top', opacity: 0.95 }
      );
    },
    [countryRisks, countryDetails]
  );

  // Re-render when risk data changes
  useEffect(() => {
    if (geoJsonRef.current) {
      geoJsonRef.current.clearLayers();
      geoJsonRef.current.addData(worldGeo);
    }
  }, [countryRisks, worldGeo]);

  return (
    <Box sx={{ position: 'relative', height, width: '100%' }}>
      {/* Header */}
      {selectedCommodity && (
        <Paper
          sx={{
            position: 'absolute',
            top: 12,
            left: 60,
            zIndex: 1000,
            px: 2,
            py: 0.5,
          }}
          elevation={2}
        >
          <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            Filtered:
            <Chip
              size="small"
              label={selectedCommodity.replace('_', ' ')}
              color="primary"
              sx={{ textTransform: 'capitalize' }}
            />
          </Typography>
        </Paper>
      )}

      <MapContainer
        center={[20, 0]}
        zoom={2}
        minZoom={2}
        maxZoom={6}
        style={{ height: '100%', width: '100%', borderRadius: '8px' }}
        scrollWheelZoom
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
        />

        <GeoJSON
          ref={(ref) => {
            geoJsonRef.current = ref as unknown as L.GeoJSON | null;
          }}
          data={worldGeo}
          style={styleFeature}
          onEachFeature={onEachFeature}
        />
      </MapContainer>

      {/* Color scale legend */}
      <Paper
        sx={{
          position: 'absolute',
          bottom: 24,
          right: 16,
          zIndex: 1000,
          p: 1.5,
          minWidth: 180,
        }}
        elevation={3}
      >
        <Typography variant="subtitle2" gutterBottom>
          Risk Score
        </Typography>
        <Box
          sx={{
            height: 12,
            borderRadius: 1,
            background: 'linear-gradient(to right, #4caf50, #ffeb3b, #ff9800, #f44336, #b71c1c)',
            mb: 0.5,
          }}
        />
        <Stack direction="row" justifyContent="space-between">
          <Typography variant="caption" color="text.secondary">0%</Typography>
          <Typography variant="caption" color="text.secondary">30%</Typography>
          <Typography variant="caption" color="text.secondary">50%</Typography>
          <Typography variant="caption" color="text.secondary">70%</Typography>
          <Typography variant="caption" color="text.secondary">100%</Typography>
        </Stack>
      </Paper>
    </Box>
  );
};

export default RiskHeatmap;
