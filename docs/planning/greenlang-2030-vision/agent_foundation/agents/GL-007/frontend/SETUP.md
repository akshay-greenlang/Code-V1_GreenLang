# GL-007 Furnace Performance Monitor - Frontend Setup Guide

Complete setup guide for the GL-007 Industrial Furnace Performance Monitoring Dashboard.

## System Requirements

- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **Operating System**: Windows, macOS, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge (latest 2 versions)
- **Memory**: 8GB RAM minimum (4GB for development, charts are memory-intensive)
- **Disk Space**: 3GB for node_modules (includes Storybook)

## Quick Start

```bash
# 1. Navigate to frontend directory
cd GreenLang_2030/agent_foundation/agents/GL-007/frontend

# 2. Install dependencies (exact versions from package-lock.json)
npm ci

# 3. Create environment file
cp .env.example .env

# 4. Start development server
npm run dev
```

The application will be available at http://localhost:3000

## Installation Steps

### 1. Verify Prerequisites

```bash
# Check Node.js version (must be >= 18.0.0)
node --version

# Check npm version (must be >= 9.0.0)
npm --version
```

### 2. Install Dependencies

Using `npm ci` (recommended):
```bash
npm ci
```

This ensures exact dependency versions are installed from package-lock.json.

Using `npm install` (alternative):
```bash
npm install
```

This may update dependencies within semver ranges.

### 3. Configure Environment Variables

Create `.env` file:

```env
# Backend API Configuration
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws

# Feature Flags
VITE_ENABLE_REALTIME=true
VITE_ENABLE_THERMAL_PROFILING=true
VITE_ENABLE_PREDICTIVE_MAINTENANCE=true

# Monitoring
VITE_ENABLE_PERFORMANCE_MONITORING=true

# Localization
VITE_DEFAULT_LANGUAGE=en
VITE_DEFAULT_TIMEZONE=UTC
```

### 4. Verify Installation

```bash
# Type check
npm run type-check

# Run tests
npm test

# Build
npm run build
```

## Available Scripts

### Development

```bash
# Start Vite development server
npm run dev
# Server: http://localhost:3000
# Features: Hot Module Replacement (HMR), Fast Refresh

# Start Storybook component explorer
npm run storybook
# Server: http://localhost:6006
# Features: Component isolation, visual testing
```

### Building

```bash
# Type check without emitting files
npm run type-check

# Build production bundle
npm run build
# Output: dist/ directory

# Preview production build
npm run preview
# Server: http://localhost:4173
```

### Testing

```bash
# Run tests in watch mode
npm test

# Run tests with UI dashboard
npm run test:ui
# Server: http://localhost:51204

# Run tests with coverage
npm run test:coverage
# Reports: coverage/ directory

# Run specific test file
npm test -- GaugeChart.test.tsx
```

### Code Quality

```bash
# Lint code
npm run lint

# Format code with Prettier
npm run format
```

### Storybook

```bash
# Start Storybook dev server
npm run storybook

# Build static Storybook site
npm run build-storybook
# Output: storybook-static/ directory
```

## Technology Stack

### Core Framework

- **React 18.3**: Latest React with Concurrent Features
- **TypeScript 5.4**: Strict type checking
- **Vite 5.2**: Lightning-fast build tool

### UI Framework

- **Material-UI (MUI) 5.15**: Enterprise component library
  - @mui/material: Core components
  - @mui/icons-material: 2000+ Material icons
  - @emotion/react: CSS-in-JS styling
  - @emotion/styled: Styled components

### Data Visualization

- **Chart.js 4.4**: Versatile charting library
- **react-chartjs-2 5.2**: React wrapper for Chart.js
- **Recharts 2.12**: Composable React charts
- **D3.js 7.9**: Advanced visualizations
- **@nivo/heatmap 0.87**: Beautiful heatmaps
- **Plotly.js 2.32**: Interactive scientific charts

### State Management

- **Zustand 4.5**: Lightweight state management
- **@tanstack/react-query 5.45**: Server state management
  - Caching
  - Automatic refetching
  - Optimistic updates

### Real-time Communication

- **Socket.IO Client 4.7**: Real-time bidirectional events
  - WebSocket connections
  - Automatic reconnection
  - Event-based messaging

### Forms & Validation

- **React Hook Form 7.51**: Performant form library
- **Yup 1.4**: Schema validation

### Routing

- **React Router 6.23**: Declarative routing

### Internationalization

- **i18next 23.11**: Translation framework
- **react-i18next 14.1**: React integration

### Utilities

- **Axios 1.7**: HTTP client
- **date-fns 3.6**: Modern date utility library
- **Lodash 4.17**: Utility functions
- **clsx 2.1**: Conditional className utility
- **react-toastify 10**: Toast notifications

### Testing & Development

- **Vitest 1.6**: Fast unit test framework
- **@testing-library/react 15**: React testing utilities
- **@testing-library/user-event 14**: User interaction simulation
- **Storybook 8.1**: Component development environment
- **ESLint 8.57**: Code linting
- **Prettier 3.3**: Code formatting

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── charts/
│   │   │   ├── GaugeChart.tsx           # Circular gauge visualization
│   │   │   └── KPICard.tsx              # KPI metric cards
│   │   └── dashboards/
│   │       ├── ExecutiveDashboard.tsx   # Executive summary view
│   │       ├── OperationsDashboard.tsx  # Operations monitoring
│   │       └── ThermalProfilingView.tsx # Thermal analysis
│   ├── services/
│   │   ├── apiClient.ts                 # HTTP API client
│   │   └── websocket.ts                 # WebSocket manager
│   ├── store/
│   │   └── furnaceStore.ts              # Zustand state store
│   ├── types/
│   │   └── index.ts                     # TypeScript type definitions
│   ├── App.tsx                          # Main application component
│   └── main.tsx                         # React entry point
├── tests/
│   └── KPICard.test.tsx                 # Component tests
├── public/                              # Static assets
├── package.json                         # Dependencies
├── package-lock.json                    # Locked versions
├── tsconfig.json                        # TypeScript config
├── tsconfig.node.json                   # TypeScript for Vite
├── vite.config.ts                       # Vite configuration
└── .env.example                         # Environment template
```

## Key Features

### Real-time Monitoring

- **Live Data Streaming**: WebSocket integration for real-time furnace telemetry
- **Auto-refresh**: Configurable refresh intervals (1s, 5s, 10s, 30s)
- **Connection Status**: Visual indicators for WebSocket connection state

### Executive Dashboard

- **KPI Cards**:
  - Thermal Efficiency (%)
  - Fuel Consumption (kg/hr)
  - Production Rate (tonnes/hr)
  - CO₂ Emissions (tCO₂e/hr)
- **Trend Charts**: Historical performance trends
- **Alert Summary**: Critical alerts and warnings
- **System Health**: Overall furnace health score

### Operations Dashboard

- **Zone Monitoring**: Individual zone temperature tracking
- **Fuel Flow Rates**: Real-time fuel consumption
- **Pressure Monitoring**: System pressure across zones
- **Emission Levels**: CO₂, NOx, SOx, particulates
- **Sensor Grid**: Live sensor readings with heatmap

### Thermal Profiling

- **3D Heatmap**: Temperature distribution visualization
- **Zone Comparison**: Side-by-side zone analysis
- **Thermal Gradients**: Temperature differential analysis
- **Historical Trends**: Time-series thermal data

### Alerts & Notifications

- **Real-time Alerts**: Toast notifications for critical events
- **Alert History**: Searchable alert log
- **Priority Levels**: Critical, Warning, Info
- **Acknowledgement**: Manual alert acknowledgement

### Internationalization

- **Multi-language**: English, German, Chinese, Spanish
- **Timezone Support**: Automatic timezone conversion
- **Number Formatting**: Locale-specific number/date formats

## Development Workflow

### 1. Create a New Chart Component

```typescript
// src/components/charts/MyChart.tsx
import React from 'react';
import { Box, Typography } from '@mui/material';
import { Line } from 'react-chartjs-2';

interface MyChartProps {
  data: number[];
  labels: string[];
  title?: string;
}

const MyChart: React.FC<MyChartProps> = ({ data, labels, title }) => {
  const chartData = {
    labels,
    datasets: [
      {
        label: 'Dataset',
        data,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  return (
    <Box sx={{ p: 2 }}>
      {title && <Typography variant="h6">{title}</Typography>}
      <Line data={chartData} />
    </Box>
  );
};

export default MyChart;
```

### 2. Create Storybook Story

```typescript
// src/components/charts/MyChart.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import MyChart from './MyChart';

const meta: Meta<typeof MyChart> = {
  title: 'Charts/MyChart',
  component: MyChart,
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof MyChart>;

export const Default: Story = {
  args: {
    data: [10, 20, 30, 40, 50],
    labels: ['A', 'B', 'C', 'D', 'E'],
    title: 'Sample Chart',
  },
};
```

### 3. Write Tests

```typescript
// tests/MyChart.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import MyChart from '../src/components/charts/MyChart';

describe('MyChart', () => {
  it('renders title', () => {
    render(<MyChart data={[]} labels={[]} title="Test" />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

### 4. Test in Storybook

```bash
npm run storybook
```

Navigate to http://localhost:6006 and view your component.

### 5. Run Tests

```bash
npm test
```

## WebSocket Integration

### Connect to Real-time Data

```typescript
import { useEffect } from 'react';
import { io } from 'socket.io-client';
import { useFurnaceStore } from './store/furnaceStore';

function App() {
  const updateMetrics = useFurnaceStore((state) => state.updateMetrics);

  useEffect(() => {
    const socket = io(import.meta.env.VITE_WS_URL);

    socket.on('furnace_telemetry', (data) => {
      updateMetrics(data);
    });

    socket.on('furnace_alert', (alert) => {
      // Handle alert
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return <div>...</div>;
}
```

## State Management with Zustand

```typescript
import create from 'zustand';

interface FurnaceState {
  metrics: any;
  alerts: any[];
  updateMetrics: (metrics: any) => void;
  addAlert: (alert: any) => void;
}

export const useFurnaceStore = create<FurnaceState>((set) => ({
  metrics: null,
  alerts: [],
  updateMetrics: (metrics) => set({ metrics }),
  addAlert: (alert) => set((state) => ({ alerts: [...state.alerts, alert] })),
}));
```

## Troubleshooting

### Port Already in Use

```bash
# Change port in vite.config.ts or use environment variable
PORT=3001 npm run dev
```

### WebSocket Connection Refused

1. Check backend is running
2. Verify VITE_WS_URL in `.env`
3. Check firewall/proxy settings
4. Inspect browser console for errors

### Chart Not Rendering

```bash
# Ensure Chart.js is registered
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);
```

### Memory Issues with Charts

- Limit data points displayed
- Use virtualization for large datasets
- Implement pagination
- Use `useMemo` for chart data

### TypeScript Errors

```bash
# Strict type checking - fix before building
npm run type-check
```

### Build Errors

```bash
# Clear caches
rm -rf node_modules/.vite dist

# Rebuild
npm run build
```

## Performance Optimization

### Lazy Loading

```typescript
import { lazy, Suspense } from 'react';

const ThermalProfilingView = lazy(() => import('./components/dashboards/ThermalProfilingView'));

function App() {
  return (
    <Suspense fallback={<CircularProgress />}>
      <ThermalProfilingView />
    </Suspense>
  );
}
```

### Memoization

```typescript
import { useMemo } from 'react';

function Dashboard({ data }) {
  const chartData = useMemo(() => {
    return processData(data);
  }, [data]);

  return <Chart data={chartData} />;
}
```

### React Query Caching

```typescript
import { useQuery } from '@tanstack/react-query';

function Dashboard() {
  const { data } = useQuery({
    queryKey: ['furnace', 'metrics'],
    queryFn: fetchMetrics,
    staleTime: 5000, // Cache for 5 seconds
    refetchInterval: 10000, // Refetch every 10 seconds
  });
}
```

## Deployment

### Build Production Bundle

```bash
npm run build
```

Output: `dist/` directory

### Environment Variables for Production

```env
VITE_API_URL=https://api.greenlang.io/v1
VITE_WS_URL=wss://api.greenlang.io/ws
VITE_ENABLE_REALTIME=true
```

### Deploy to Production

#### Docker

```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-007-frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-007-frontend
  template:
    metadata:
      labels:
        app: gl-007-frontend
    spec:
      containers:
      - name: frontend
        image: greenlang/gl-007-frontend:latest
        ports:
        - containerPort: 80
```

## Security Best Practices

1. **API Keys**: Use environment variables, never hardcode
2. **HTTPS**: Always use HTTPS in production
3. **CSP**: Configure Content Security Policy headers
4. **Dependencies**: Regular security audits with `npm audit`
5. **Authentication**: Implement proper JWT handling
6. **XSS Protection**: React automatically escapes content

## Additional Resources

- [Material-UI Documentation](https://mui.com/)
- [Chart.js Documentation](https://www.chartjs.org/)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Zustand Documentation](https://github.com/pmndrs/zustand)
- [Storybook Documentation](https://storybook.js.org/)

## Support

For technical support:
- GitHub Issues: https://github.com/greenlang/agents/issues
- Email: support@greenlang.ai
- Documentation: https://docs.greenlang.ai
