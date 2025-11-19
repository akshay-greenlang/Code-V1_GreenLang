# GL-007 Furnace Performance Monitor - Frontend

A comprehensive, real-time monitoring dashboard suite for industrial furnace performance, built with React, TypeScript, and Material-UI.

## Overview

The GL-007 Frontend provides a modern, responsive interface for monitoring and analyzing furnace performance across multiple dimensions:

- **Executive Dashboard**: High-level KPIs, cost savings, and sustainability metrics
- **Operations Dashboard**: Real-time monitoring with 20+ KPIs, multi-zone temperature tracking
- **Thermal Profiling**: Advanced thermal visualization with heatmaps and hot spot detection
- **Maintenance Dashboard**: Predictive maintenance alerts and equipment health scoring
- **Analytics Dashboard**: Performance trends, benchmarking, and optimization opportunities
- **Alert Management**: Real-time alert configuration and root cause analysis
- **Configuration Panel**: Furnace settings, sensor calibration, and user preferences
- **Reporting Module**: Automated report generation with multiple export formats

## Features

### Core Capabilities

- **Real-time Data Streaming**: WebSocket integration for live performance updates
- **20+ KPIs**: Comprehensive performance metrics across efficiency, production, emissions
- **Multi-Zone Monitoring**: Track temperature, pressure, and flow across furnace zones
- **Advanced Visualizations**: Charts, gauges, heatmaps, and 3D thermal profiling
- **Predictive Analytics**: AI-powered maintenance predictions and optimization recommendations
- **Alert System**: Multi-level alerts with escalation and acknowledgment workflows
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Dark Mode**: Full theme support with light/dark modes
- **Accessibility**: WCAG 2.1 AA compliant

### Technical Features

- **Type Safety**: Full TypeScript coverage with comprehensive type definitions
- **State Management**: Zustand for global state with persistence
- **Data Fetching**: React Query for efficient data loading and caching
- **Real-time Updates**: Socket.io for WebSocket communication
- **Performance**: Code splitting, lazy loading, and optimized bundle size
- **Testing**: Unit tests with React Testing Library
- **Component Library**: 40+ reusable, documented components
- **Internationalization**: Multi-language support (i18n ready)

## Technology Stack

### Core
- **React 18**: Modern React with hooks and concurrent features
- **TypeScript 5**: Full type safety and IntelliSense support
- **Vite**: Lightning-fast build tool and dev server

### UI Framework
- **Material-UI v5**: Comprehensive component library
- **Emotion**: CSS-in-JS styling solution
- **React Router v6**: Client-side routing

### Data Management
- **React Query**: Server state management and caching
- **Zustand**: Lightweight global state management
- **Socket.io Client**: Real-time WebSocket communication

### Visualization
- **Chart.js**: Versatile charting library
- **Recharts**: Composable charting components
- **D3.js**: Advanced data visualization
- **Nivo**: High-level charting components

### Development
- **ESLint**: Code linting and style enforcement
- **Prettier**: Code formatting
- **Vitest**: Fast unit testing
- **Storybook**: Component development and documentation

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0

### Installation

```bash
# Clone the repository
cd GL-007/frontend

# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env

# Update .env with your API credentials
# VITE_API_URL=https://api.greenlang.io/v1
# VITE_WS_URL=wss://ws.greenlang.io
# VITE_API_KEY=your_api_key_here
```

### Development

```bash
# Start development server
npm run dev

# Open browser to http://localhost:3000
```

The development server includes:
- Hot Module Replacement (HMR)
- Fast refresh for instant updates
- API proxy to avoid CORS issues
- WebSocket proxy for real-time connections

### Building for Production

```bash
# Create optimized production build
npm run build

# Preview production build locally
npm run preview
```

Build output includes:
- Minified JavaScript and CSS
- Code splitting for optimal loading
- Source maps for debugging
- Compressed assets (gzip/brotli)

Target bundle size: < 500KB gzipped

### Testing

```bash
# Run unit tests
npm run test

# Run tests with UI
npm run test:ui

# Generate coverage report
npm run test:coverage
```

### Linting & Formatting

```bash
# Lint code
npm run lint

# Format code
npm run format
```

### Component Development

```bash
# Start Storybook
npm run storybook

# Build Storybook for deployment
npm run build-storybook
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── charts/          # Reusable chart components
│   │   │   ├── KPICard.tsx
│   │   │   ├── GaugeChart.tsx
│   │   │   └── ...
│   │   ├── dashboards/      # Main dashboard views
│   │   │   ├── ExecutiveDashboard.tsx
│   │   │   ├── OperationsDashboard.tsx
│   │   │   ├── ThermalProfilingView.tsx
│   │   │   └── ...
│   │   ├── alerts/          # Alert management components
│   │   ├── config/          # Configuration components
│   │   └── common/          # Shared UI components
│   ├── services/            # API and WebSocket clients
│   │   ├── apiClient.ts     # REST API client
│   │   └── websocket.ts     # WebSocket service
│   ├── store/               # State management
│   │   └── furnaceStore.ts  # Global furnace state
│   ├── types/               # TypeScript definitions
│   │   └── index.ts         # All type definitions
│   ├── hooks/               # Custom React hooks
│   ├── utils/               # Helper functions
│   ├── styles/              # Global styles
│   ├── App.tsx              # Main application component
│   └── main.tsx             # Application entry point
├── public/                  # Static assets
├── tests/                   # Test files
├── package.json             # Dependencies and scripts
├── vite.config.ts           # Vite configuration
├── tsconfig.json            # TypeScript configuration
└── README.md                # This file
```

## Key Components

### Dashboards

#### Executive Dashboard
- Overall performance summary with 8 primary KPIs
- Fleet-wide metrics (if multiple furnaces)
- Cost savings opportunities with ROI analysis
- Carbon impact and sustainability metrics
- Energy efficiency trends over time
- Top optimization recommendations

**Route**: `/executive`

#### Operations Dashboard
- Real-time performance with 20+ KPIs
- Multi-zone temperature monitoring with live charts
- Fuel consumption tracking and combustion efficiency
- Production correlation and quality metrics
- Live alert feed with severity indicators
- Zone-by-zone performance table

**Route**: `/operations`

#### Thermal Profiling View
- Temperature distribution heatmap
- Hot spot and cold spot detection
- Multi-zone thermal analysis
- Temperature uniformity index
- Thermal control recommendations
- Historical thermal patterns

**Route**: `/thermal`

### Chart Components

#### KPICard
Displays key performance indicators with:
- Current value and unit
- Target comparison
- Trend indicator (increasing/decreasing/stable)
- Status color coding (good/warning/critical)
- Optional icon and click handler

```tsx
<KPICard
  title="Overall Efficiency"
  value={92.5}
  unit="%"
  target={95}
  trend="stable"
  trendValue={2.3}
  status="good"
  icon={<Assessment />}
/>
```

#### GaugeChart
Circular gauge for real-time metrics:
- Configurable thresholds (good/warning/critical)
- Color-coded segments
- Center value display
- Threshold indicators

```tsx
<GaugeChart
  value={85}
  maxValue={100}
  title="Thermal Efficiency"
  unit="%"
  thresholds={{ good: 85, warning: 75, critical: 65 }}
  size={200}
/>
```

## API Integration

### REST API Client

The `apiClient` service provides type-safe methods for all API endpoints:

```typescript
import { apiClient } from '@services/apiClient';

// Authenticate
await apiClient.authenticate(clientId, clientSecret);

// Get furnace data
const furnace = await apiClient.getFurnace(furnaceId);

// Get real-time performance
const performance = await apiClient.getPerformance(furnaceId);

// Get alerts
const alerts = await apiClient.getAlerts(furnaceId, {
  severity: 'critical',
  status: 'active'
});
```

### WebSocket Integration

Real-time updates are handled via WebSocket:

```typescript
import { useWebSocket } from '@services/websocket';

const MyComponent = () => {
  const { service, isConnected } = useWebSocket({
    furnaceId: 'furnace-1',
    autoConnect: true
  });

  useEffect(() => {
    const unsubscribe = service.on('performance_update', (data) => {
      console.log('Performance updated:', data);
    });

    return unsubscribe;
  }, [service]);
};
```

### React Query Integration

Data fetching with automatic caching and refetching:

```typescript
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@services/apiClient';

const { data, isLoading, error } = useQuery({
  queryKey: ['performance', furnaceId],
  queryFn: () => apiClient.getPerformance(furnaceId),
  refetchInterval: 5000, // Refresh every 5 seconds
});
```

## State Management

Global state is managed with Zustand:

```typescript
import { useFurnaceStore } from '@store/furnaceStore';

const MyComponent = () => {
  const {
    selectedFurnaceId,
    setSelectedFurnace,
    performance,
    activeAlerts
  } = useFurnaceStore();

  return (
    <div>
      <Select
        value={selectedFurnaceId}
        onChange={(e) => setSelectedFurnace(e.target.value)}
      >
        {/* options */}
      </Select>
    </div>
  );
};
```

## Performance Optimization

### Bundle Size
- Code splitting by route and feature
- Tree shaking to remove unused code
- Dynamic imports for heavy components
- Compression (gzip/brotli)

Target: < 500KB gzipped main bundle

### Runtime Performance
- React.memo for expensive components
- useMemo/useCallback for computed values
- Virtual scrolling for large lists
- Debounced/throttled event handlers
- Lazy loading of images and charts

Target: < 2 seconds initial load, 60 FPS animations

### Data Optimization
- React Query caching strategy
- WebSocket connection pooling
- Selective re-rendering with Zustand
- Local storage for user preferences

## Accessibility

WCAG 2.1 AA compliance includes:
- Semantic HTML structure
- ARIA labels and roles
- Keyboard navigation support
- Focus management
- Color contrast ratios > 4.5:1
- Screen reader compatibility
- Skip navigation links

## Browser Support

- Chrome/Edge >= 90
- Firefox >= 88
- Safari >= 14
- Mobile browsers (iOS Safari, Chrome Mobile)

## Environment Variables

```bash
# API Configuration
VITE_API_URL=https://api.greenlang.io/v1
VITE_API_KEY=your_api_key_here

# WebSocket Configuration
VITE_WS_URL=wss://ws.greenlang.io

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_THERMAL_IMAGING=true
VITE_ENABLE_PREDICTIVE_MAINTENANCE=true

# Authentication
VITE_AUTH_ENABLED=true
VITE_AUTH_PROVIDER=oauth2
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions including:
- Docker containerization
- Kubernetes deployment
- CDN configuration
- SSL/TLS setup
- Monitoring and logging

## Contributing

1. Create a feature branch
2. Make changes with appropriate tests
3. Run linting and tests
4. Submit pull request

Code style:
- Use TypeScript strict mode
- Follow Material-UI patterns
- Write comprehensive type definitions
- Add JSDoc comments for complex logic
- Include unit tests for utilities

## License

Copyright (c) 2024 GreenLang. All rights reserved.

## Support

For issues or questions:
- Email: support@greenlang.io
- Documentation: https://docs.greenlang.io
- GitHub Issues: https://github.com/greenlang/gl-007

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Executive Dashboard with 8 primary KPIs
- Operations Dashboard with 20+ real-time KPIs
- Thermal Profiling with heatmaps
- WebSocket integration for live updates
- Full TypeScript coverage
- Responsive design
- Dark mode support
- WCAG 2.1 AA accessibility
