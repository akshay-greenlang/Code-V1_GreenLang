# GreenLang Visual Workflow Builder - Frontend Setup Guide

Complete setup guide for the GreenLang Visual Workflow Builder frontend application.

## System Requirements

- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **Operating System**: Windows, macOS, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge (latest 2 versions)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 2GB for node_modules

## Quick Start

```bash
# 1. Navigate to frontend directory
cd greenlang/frontend

# 2. Install dependencies (this will use the existing package-lock.json)
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
# Check Node.js version (should be >= 18.0.0)
node --version

# Check npm version (should be >= 9.0.0)
npm --version
```

If Node.js is not installed or outdated, download from https://nodejs.org/

### 2. Install Dependencies

Using `npm ci` (recommended for CI/CD and production):
```bash
npm ci
```

This installs exact versions from package-lock.json (faster and more reliable).

Using `npm install` (for development):
```bash
npm install
```

This respects version ranges in package.json and may update package-lock.json.

### 3. Configure Environment

Create `.env` file in the frontend directory:

```env
# API Configuration
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_WORKFLOW_EXPORT=true

# Development
VITE_DEV_MODE=true
```

## Available Scripts

### Development

```bash
# Start Vite development server with HMR
npm run dev

# This starts the server at http://localhost:3000
# Hot Module Replacement (HMR) provides instant updates
```

### Building

```bash
# Type check TypeScript code
npm run type-check

# Build for production
npm run build

# Build output is in dist/ directory
```

### Testing

```bash
# Run tests in watch mode
npm test

# Run tests with UI dashboard
npm run test:ui

# Run tests with coverage report
npm run test:coverage

# Coverage reports are in coverage/ directory
```

### Code Quality

```bash
# Lint TypeScript/TSX files
npm run lint

# Auto-fix linting issues
npm run lint:fix

# Format code with Prettier
npm run format
```

### Preview Production Build

```bash
# Build and preview locally
npm run build
npm run preview

# Preview server runs at http://localhost:4173
```

## Technology Stack

### Core

- **React 18.3**: Modern React with Hooks and Concurrent Features
- **TypeScript 5.4**: Type-safe JavaScript
- **Vite 5.2**: Next-generation build tool with lightning-fast HMR

### Workflow & Visualization

- **ReactFlow 11.11**: Visual workflow builder with drag-and-drop
- **Dagre 0.8**: Automatic graph layout algorithm
- **D3.js 7.9**: Advanced data visualizations
- **ECharts 5.5**: Enterprise-grade charting library
- **Recharts 2.12**: Composable charting components

### State Management

- **Zustand 4.5**: Lightweight state management
- **Immer 10.1**: Immutable state updates

### Routing & HTTP

- **React Router 6.23**: Declarative routing
- **Axios 1.7**: Promise-based HTTP client
- **Socket.IO Client 4.7**: Real-time bidirectional communication

### Styling

- **Tailwind CSS 3.4**: Utility-first CSS framework
- **PostCSS 8.4**: CSS transformations
- **Autoprefixer 10.4**: Automatic vendor prefixing
- **Lucide React**: Icon library

### Testing

- **Vitest 1.6**: Fast unit test framework
- **@testing-library/react 15**: React testing utilities
- **jsdom 24**: DOM implementation for Node.js
- **@vitest/coverage-v8**: Code coverage reports

### Development Tools

- **ESLint 8.57**: JavaScript linter
- **Prettier 3.3**: Code formatter
- **TypeScript ESLint 7.12**: TypeScript linting rules

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── WorkflowBuilder/
│   │   │   ├── WorkflowCanvas.tsx        # Main canvas component
│   │   │   ├── AgentPalette.tsx          # Drag-and-drop agent palette
│   │   │   ├── DAGEditor.tsx             # DAG visualization editor
│   │   │   ├── ExecutionMonitor.tsx      # Real-time execution monitoring
│   │   │   ├── ExecutionTimeline.tsx     # Timeline visualization
│   │   │   ├── Collaboration.tsx         # Multi-user collaboration
│   │   │   └── hooks/
│   │   │       └── useWorkflowValidation.ts
│   │   └── Analytics/
│   │       ├── Dashboard.tsx             # Analytics dashboard
│   │       ├── AlertManager.tsx          # Alert management
│   │       ├── MetricService.ts          # Metrics service
│   │       └── widgets/
│   │           ├── LineChart.tsx         # Time series charts
│   │           ├── BarChart.tsx          # Bar charts
│   │           ├── PieChart.tsx          # Pie charts
│   │           ├── GaugeChart.tsx        # Gauge charts
│   │           ├── HeatmapChart.tsx      # Heatmap visualizations
│   │           ├── StatCard.tsx          # KPI stat cards
│   │           └── TableWidget.tsx       # Data tables
│   ├── App.tsx                           # Main App component
│   ├── index.css                         # Global styles
│   └── main.tsx                          # React entry point
├── public/                               # Static assets
├── tests/                                # Test files
├── index.html                            # HTML template
├── package.json                          # Dependencies
├── package-lock.json                     # Locked dependency versions
├── tsconfig.json                         # TypeScript config
├── vite.config.ts                        # Vite configuration
├── tailwind.config.js                    # Tailwind CSS config
├── postcss.config.js                     # PostCSS config
├── vitest.config.ts                      # Vitest config
└── .eslintrc.cjs                         # ESLint config
```

## Features

### Visual Workflow Builder

- **Drag-and-Drop Interface**: Intuitive agent composition
- **Real-time Validation**: Instant feedback on workflow validity
- **Auto-Layout**: Automatic graph layout with Dagre
- **Zoom & Pan**: Navigate complex workflows
- **Minimap**: Overview of large workflows
- **Undo/Redo**: Full history management
- **Export/Import**: JSON workflow definitions

### Analytics Dashboard

- **Real-time Metrics**: Live data streaming via WebSockets
- **Interactive Charts**: Drill-down capabilities
- **Custom Dashboards**: Configurable widget layouts
- **Alert Management**: Real-time notifications
- **Data Export**: CSV, Excel, JSON formats

### Collaboration

- **Multi-user Editing**: Real-time collaborative workflow editing
- **Comments & Annotations**: Inline discussion threads
- **Version Control**: Workflow versioning
- **Permissions**: Role-based access control

## Development Workflow

### 1. Create a New Component

```bash
# Create component file
touch src/components/MyComponent.tsx

# Create test file
touch src/components/__tests__/MyComponent.test.tsx
```

### 2. Component Template

```typescript
import React from 'react';

interface MyComponentProps {
  title: string;
  data: any[];
}

const MyComponent: React.FC<MyComponentProps> = ({ title, data }) => {
  return (
    <div>
      <h2>{title}</h2>
      {/* Component implementation */}
    </div>
  );
};

export default MyComponent;
```

### 3. Write Tests

```typescript
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  it('renders title', () => {
    render(<MyComponent title="Test" data={[]} />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

### 4. Run Tests

```bash
npm test
```

## Troubleshooting

### Port 3000 Already in Use

```bash
# Use a different port
PORT=3001 npm run dev
```

Or update `vite.config.ts`:
```typescript
export default defineConfig({
  server: {
    port: 3001
  }
});
```

### Dependency Installation Errors

```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### TypeScript Errors

```bash
# Run type check to see all errors
npm run type-check

# Restart TypeScript server in VS Code
# Command Palette > TypeScript: Restart TS Server
```

### Vite Build Errors

```bash
# Clear Vite cache
rm -rf node_modules/.vite

# Rebuild
npm run build
```

### WebSocket Connection Issues

Check these in your `.env`:
```env
VITE_WS_URL=ws://localhost:8000/ws
```

Verify backend WebSocket server is running.

## Performance Optimization

### Code Splitting

Vite automatically code-splits:
- Route-based splitting
- Dynamic imports
- Vendor chunk separation

### Bundle Analysis

```bash
# Build with analysis
npm run build

# Check dist/ directory for chunk sizes
ls -lh dist/assets/
```

### Lazy Loading

```typescript
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

## Deployment

### Build for Production

```bash
npm run build
```

Output in `dist/` directory is ready for deployment.

### Deploy to Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod --dir=dist
```

### Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

### Deploy to AWS S3

```bash
# Build
npm run build

# Upload to S3
aws s3 sync dist/ s3://your-bucket-name/
```

## Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **API Keys**: Use environment variables, not hardcoded values
3. **Dependencies**: Regularly update with `npm audit fix`
4. **HTTPS**: Always use HTTPS in production
5. **CSP**: Configure Content Security Policy headers

## Additional Resources

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [ReactFlow Documentation](https://reactflow.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [Vitest Documentation](https://vitest.dev/)

## Support

For issues or questions:
- Create an issue in the GitHub repository
- Contact: support@greenlang.ai
