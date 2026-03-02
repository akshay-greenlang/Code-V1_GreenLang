/**
 * PlotMap - Interactive Leaflet map showing plot polygons on OpenStreetMap tiles.
 *
 * Renders supplier plot boundaries color-coded by risk level, with popups,
 * layer controls, deforestation alert markers, polygon draw controls,
 * and an auto-fit-bounds feature.
 */

import React, { useEffect, useMemo, useRef, useCallback } from 'react';
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Marker,
  Popup,
  LayersControl,
  useMap,
} from 'react-leaflet';
import L, { LatLngBoundsExpression, Layer, PathOptions } from 'leaflet';
import {
  Box,
  Paper,
  Typography,
  Chip,
  Stack,
  Divider,
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import type { Plot, RiskAlert, RiskLevel } from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RISK_COLORS: Record<RiskLevel, string> = {
  low: '#4caf50',
  standard: '#ffeb3b',
  high: '#ff9800',
  critical: '#f44336',
};

const RISK_LABELS: Record<RiskLevel, string> = {
  low: 'Low',
  standard: 'Standard',
  high: 'High',
  critical: 'Critical',
};

const DEFAULT_CENTER: [number, number] = [20, 0];
const DEFAULT_ZOOM = 3;

const STREET_TILE = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const SATELLITE_TILE =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

// Alert marker icon
const alertIcon = new L.Icon({
  iconUrl:
    'data:image/svg+xml;base64,' +
    btoa(
      '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">' +
        '<path fill="#f44336" d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>' +
        '</svg>'
    ),
  iconSize: [24, 24],
  iconAnchor: [12, 24],
  popupAnchor: [0, -24],
});

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PlotMapProps {
  plots: Plot[];
  alerts?: RiskAlert[];
  onPlotClick?: (plot: Plot) => void;
  onPolygonDraw?: (geojson: GeoJSON.Geometry) => void;
  height?: number | string;
  selectedPlotId?: string;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Automatically fits the map bounds to cover all provided plots. */
function FitBounds({ plots }: { plots: Plot[] }) {
  const map = useMap();

  useEffect(() => {
    if (plots.length === 0) return;

    const allCoords: [number, number][] = [];
    plots.forEach((plot) => {
      plot.coordinates.forEach((c) => {
        allCoords.push([c.latitude, c.longitude]);
      });
    });

    if (allCoords.length > 0) {
      const bounds = L.latLngBounds(allCoords);
      map.fitBounds(bounds as LatLngBoundsExpression, { padding: [40, 40], maxZoom: 14 });
    }
  }, [plots, map]);

  return null;
}

/** Simple draw control using a button that toggles polygon draw mode. */
function DrawControl({
  onPolygonDraw,
}: {
  onPolygonDraw?: (geojson: GeoJSON.Geometry) => void;
}) {
  const map = useMap();
  const drawingRef = useRef(false);
  const pointsRef = useRef<L.LatLng[]>([]);
  const polylineRef = useRef<L.Polyline | null>(null);

  useEffect(() => {
    if (!onPolygonDraw) return;

    const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
    const button = L.DomUtil.create('a', '', container);
    button.href = '#';
    button.title = 'Draw polygon';
    button.innerHTML = '&#x2B21;';
    button.style.fontSize = '18px';
    button.style.lineHeight = '30px';
    button.style.textAlign = 'center';
    button.style.width = '30px';
    button.style.height = '30px';
    button.style.display = 'block';
    button.style.textDecoration = 'none';
    button.style.color = '#333';
    button.style.cursor = 'pointer';

    const control = L.Control.extend({
      options: { position: 'topleft' as L.ControlPosition },
      onAdd() {
        return container;
      },
    });

    const ctrl = new control();
    map.addControl(ctrl);

    const handleClick = (e: L.LeafletMouseEvent) => {
      if (!drawingRef.current) return;

      pointsRef.current.push(e.latlng);

      if (polylineRef.current) {
        polylineRef.current.setLatLngs(pointsRef.current);
      } else {
        polylineRef.current = L.polyline(pointsRef.current, {
          color: '#1976d2',
          weight: 2,
          dashArray: '5,5',
        }).addTo(map);
      }
    };

    const handleDblClick = () => {
      if (!drawingRef.current || pointsRef.current.length < 3) return;

      const coords = pointsRef.current.map((p) => [p.lng, p.lat]);
      coords.push(coords[0]); // close ring

      const geojson: GeoJSON.Geometry = {
        type: 'Polygon',
        coordinates: [coords],
      };

      onPolygonDraw(geojson);

      // Cleanup
      if (polylineRef.current) {
        map.removeLayer(polylineRef.current);
        polylineRef.current = null;
      }
      pointsRef.current = [];
      drawingRef.current = false;
      button.style.backgroundColor = '';
      map.getContainer().style.cursor = '';
    };

    L.DomEvent.on(button, 'click', (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
      drawingRef.current = !drawingRef.current;
      button.style.backgroundColor = drawingRef.current ? '#bbdefb' : '';
      map.getContainer().style.cursor = drawingRef.current ? 'crosshair' : '';

      if (!drawingRef.current) {
        if (polylineRef.current) {
          map.removeLayer(polylineRef.current);
          polylineRef.current = null;
        }
        pointsRef.current = [];
      }
    });

    map.on('click', handleClick);
    map.on('dblclick', handleDblClick);

    return () => {
      map.removeControl(ctrl);
      map.off('click', handleClick);
      map.off('dblclick', handleDblClick);
    };
  }, [map, onPolygonDraw]);

  return null;
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function Legend() {
  return (
    <Paper
      sx={{
        position: 'absolute',
        bottom: 24,
        right: 16,
        zIndex: 1000,
        p: 1.5,
        minWidth: 140,
      }}
      elevation={3}
    >
      <Typography variant="subtitle2" gutterBottom>
        Risk Level
      </Typography>
      {Object.entries(RISK_COLORS).map(([level, color]) => (
        <Stack key={level} direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
          <Box
            sx={{
              width: 16,
              height: 16,
              borderRadius: '2px',
              backgroundColor: color,
              border: '1px solid rgba(0,0,0,0.2)',
              flexShrink: 0,
            }}
          />
          <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
            {RISK_LABELS[level as RiskLevel]}
          </Typography>
        </Stack>
      ))}
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function plotToGeoJSON(plot: Plot): GeoJSON.Feature {
  const coords = plot.coordinates.map((c) => [c.longitude, c.latitude]);
  if (coords.length > 0 && (coords[0][0] !== coords[coords.length - 1][0] ||
    coords[0][1] !== coords[coords.length - 1][1])) {
    coords.push(coords[0]);
  }

  return {
    type: 'Feature',
    properties: {
      id: plot.id,
      name: plot.name,
      commodity: plot.commodity,
      area_hectares: plot.area_hectares,
      risk_level: plot.risk_level,
      supplier_name: plot.supplier_name,
      deforestation_check_date: plot.deforestation_check_date,
      deforestation_free: plot.deforestation_free,
    },
    geometry: {
      type: 'Polygon',
      coordinates: [coords],
    },
  };
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'Not assessed';
  return new Date(dateStr).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

// ---------------------------------------------------------------------------
// PlotMap Component
// ---------------------------------------------------------------------------

const PlotMap: React.FC<PlotMapProps> = ({
  plots,
  alerts = [],
  onPlotClick,
  onPolygonDraw,
  height = 500,
  selectedPlotId,
}) => {
  const geoJsonRef = useRef<L.GeoJSON | null>(null);

  const geojsonData = useMemo<GeoJSON.FeatureCollection>(() => {
    return {
      type: 'FeatureCollection',
      features: plots.map(plotToGeoJSON),
    };
  }, [plots]);

  const plotMap = useMemo(() => {
    const map = new Map<string, Plot>();
    plots.forEach((p) => map.set(p.id, p));
    return map;
  }, [plots]);

  const styleFeature = useCallback(
    (feature?: GeoJSON.Feature): PathOptions => {
      const riskLevel = (feature?.properties?.risk_level ?? 'standard') as RiskLevel;
      const isSelected = feature?.properties?.id === selectedPlotId;
      return {
        fillColor: RISK_COLORS[riskLevel] ?? RISK_COLORS.standard,
        color: isSelected ? '#1565c0' : '#333',
        weight: isSelected ? 3 : 1.5,
        opacity: 1,
        fillOpacity: isSelected ? 0.6 : 0.4,
      };
    },
    [selectedPlotId]
  );

  const onEachFeature = useCallback(
    (feature: GeoJSON.Feature, layer: Layer) => {
      const props = feature.properties;
      if (!props) return;

      const riskColor = RISK_COLORS[props.risk_level as RiskLevel] ?? '#999';

      layer.bindPopup(
        `<div style="min-width:200px">
          <h4 style="margin:0 0 8px">${props.name}</h4>
          <table style="width:100%;font-size:13px;border-collapse:collapse">
            <tr><td style="padding:2px 8px 2px 0;color:#666">Supplier</td>
                <td>${props.supplier_name}</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#666">Commodity</td>
                <td style="text-transform:capitalize">${(props.commodity as string).replace('_', ' ')}</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#666">Area</td>
                <td>${Number(props.area_hectares).toFixed(1)} ha</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#666">Risk</td>
                <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${riskColor};margin-right:4px;vertical-align:middle"></span>
                ${RISK_LABELS[props.risk_level as RiskLevel]}</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#666">Assessed</td>
                <td>${formatDate(props.deforestation_check_date)}</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#666">Deforestation Free</td>
                <td>${props.deforestation_free === null ? 'Pending' : props.deforestation_free ? 'Yes' : 'No'}</td></tr>
          </table>
        </div>`,
        { maxWidth: 320 }
      );

      layer.on('click', () => {
        const plot = plotMap.get(props.id);
        if (plot && onPlotClick) {
          onPlotClick(plot);
        }
      });
    },
    [onPlotClick, plotMap]
  );

  // Re-render GeoJSON when data changes
  useEffect(() => {
    if (geoJsonRef.current) {
      geoJsonRef.current.clearLayers();
      geoJsonRef.current.addData(geojsonData);
    }
  }, [geojsonData]);

  return (
    <Box sx={{ position: 'relative', height, width: '100%' }}>
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        style={{ height: '100%', width: '100%', borderRadius: '8px' }}
        scrollWheelZoom
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Street Map">
            <TileLayer
              url={STREET_TILE}
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url={SATELLITE_TILE}
              attribution='&copy; Esri, Maxar, Earthstar Geographics'
            />
          </LayersControl.BaseLayer>
        </LayersControl>

        {/* Plot polygons */}
        <GeoJSON
          ref={(ref) => {
            geoJsonRef.current = ref as unknown as L.GeoJSON | null;
          }}
          data={geojsonData}
          style={styleFeature}
          onEachFeature={onEachFeature}
        />

        {/* Deforestation alert markers */}
        {alerts.map((alert) => {
          // Attempt to locate alert near its supplier's first plot
          const supplierPlot = plots.find((p) => p.supplier_id === alert.supplier_id);
          if (!supplierPlot) return null;

          return (
            <Marker
              key={alert.id}
              position={[supplierPlot.centroid.latitude, supplierPlot.centroid.longitude]}
              icon={alertIcon}
            >
              <Popup>
                <div>
                  <strong>{alert.title}</strong>
                  <br />
                  <span style={{ fontSize: '12px', color: '#666' }}>
                    {alert.description}
                  </span>
                  <br />
                  <Chip
                    size="small"
                    label={RISK_LABELS[alert.severity]}
                    sx={{
                      mt: 0.5,
                      backgroundColor: RISK_COLORS[alert.severity],
                      color: alert.severity === 'low' || alert.severity === 'standard' ? '#333' : '#fff',
                    }}
                  />
                </div>
              </Popup>
            </Marker>
          );
        })}

        {/* Auto-fit bounds */}
        <FitBounds plots={plots} />

        {/* Draw control */}
        {onPolygonDraw && <DrawControl onPolygonDraw={onPolygonDraw} />}
      </MapContainer>

      {/* Legend */}
      <Legend />

      {/* Alert count badge */}
      {alerts.length > 0 && (
        <Paper
          sx={{
            position: 'absolute',
            top: 16,
            left: 16,
            zIndex: 1000,
            px: 1.5,
            py: 0.5,
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
          }}
          elevation={3}
        >
          <WarningAmberIcon fontSize="small" color="error" />
          <Typography variant="body2" fontWeight={600}>
            {alerts.length} Alert{alerts.length !== 1 ? 's' : ''}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default PlotMap;
