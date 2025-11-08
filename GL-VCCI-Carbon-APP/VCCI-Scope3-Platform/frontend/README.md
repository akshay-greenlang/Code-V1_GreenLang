# GL-VCCI Carbon Intelligence Platform - Frontend

React-based frontend application for the GL-VCCI Scope 3 Carbon Intelligence Platform.

## Technology Stack

- **React 18.2** - UI framework
- **TypeScript 5.3** - Type-safe JavaScript
- **Material-UI 5.14** - Component library
- **Redux Toolkit 1.9** - State management
- **React Router 6.20** - Routing
- **Recharts 2.10** - Data visualization
- **Axios 1.6** - HTTP client

## Project Structure

```
frontend/
├── public/                 # Static assets
│   ├── index.html         # HTML template
│   └── manifest.json      # PWA manifest
├── src/
│   ├── components/        # Reusable components
│   │   ├── Layout.tsx
│   │   ├── AppBar.tsx
│   │   ├── Sidebar.tsx
│   │   ├── DataTable.tsx
│   │   ├── EmissionsChart.tsx
│   │   ├── StatCard.tsx
│   │   └── LoadingSpinner.tsx
│   ├── pages/             # Page components
│   │   ├── Dashboard.tsx
│   │   ├── DataUpload.tsx
│   │   ├── SupplierManagement.tsx
│   │   ├── Reports.tsx
│   │   └── Settings.tsx
│   ├── services/          # API services
│   │   └── api.ts
│   ├── store/             # Redux store
│   │   ├── index.ts
│   │   ├── hooks.ts
│   │   └── slices/
│   │       ├── dashboardSlice.ts
│   │       ├── transactionsSlice.ts
│   │       ├── suppliersSlice.ts
│   │       └── reportsSlice.ts
│   ├── types/             # TypeScript types
│   │   └── index.ts
│   ├── utils/             # Utility functions
│   │   └── formatters.ts
│   ├── App.tsx            # Main App component
│   ├── index.tsx          # React entry point
│   ├── theme.ts           # Material-UI theme
│   └── reportWebVitals.ts # Performance monitoring
├── package.json           # Dependencies
└── tsconfig.json         # TypeScript config
```

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm 9.x or higher

### Installation

```bash
# Install dependencies
npm install
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000/api/v1
```

### Development

```bash
# Start development server (http://localhost:3000)
npm start
```

The app will automatically reload when you make changes.

### Production Build

```bash
# Create optimized production build
npm run build
```

Build output will be in the `build/` directory.

### Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### Linting & Formatting

```bash
# Lint code
npm run lint

# Format code
npm run format
```

## Features

### Dashboard
- Real-time emissions metrics
- Category breakdown (pie chart)
- Monthly trends (line chart)
- Top suppliers analysis (bar chart)
- Hotspot identification with reduction potential

### Data Upload
- Multi-format support (CSV, Excel, JSON, XML)
- Real-time upload progress tracking
- Error reporting with row-level details
- Automatic format detection
- Validation feedback

### Supplier Management
- Searchable supplier directory
- Engagement campaign creation
- Response rate tracking
- PCF submission status
- Data quality indicators (Tier 1/2/3)

### Reports
- Multi-standard reporting:
  - ESRS E1 (EU CSRD)
  - CDP Questionnaire
  - GHG Protocol
  - ISO 14083
  - IFRS S2
- PDF, Excel, and JSON export formats
- Custom date range selection
- Download management

### Settings
- User profile management
- Localization (language, timezone, currency)
- Notification preferences
- Dashboard customization
- Theme selection (light/dark)

## API Integration

The frontend communicates with the backend API through the `api.ts` service layer:

- **Base URL**: Configured via `REACT_APP_API_URL`
- **Authentication**: JWT token in `Authorization` header
- **Error Handling**: Global error interceptor
- **Request Timeout**: 30 seconds (configurable)

### API Endpoints Used

- `GET /api/v1/dashboard/metrics` - Dashboard data
- `GET /api/v1/dashboard/hotspots` - Hotspot analysis
- `POST /api/v1/intake/upload` - File upload
- `GET /api/v1/intake/transactions` - Transaction list
- `GET /api/v1/engagement/suppliers` - Supplier list
- `POST /api/v1/engagement/campaigns` - Create campaign
- `GET /api/v1/reporting/reports` - Report list
- `POST /api/v1/reporting/generate` - Generate report
- `GET /api/v1/reporting/reports/:id/download` - Download report

## State Management

Redux Toolkit is used for global state management with the following slices:

- **dashboard**: Metrics, hotspots, analytics
- **transactions**: Transaction data, upload status
- **suppliers**: Supplier directory, campaigns
- **reports**: Report generation, downloads

## Component Library

Material-UI provides:
- Consistent design system
- Responsive layout components
- Pre-built form components
- Data grid with sorting/filtering/pagination
- Theme customization
- Accessibility support

## Browser Support

- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

## Performance

- Code splitting for optimal load times
- Lazy loading of routes
- Image optimization
- Bundle size: ~300 KB (gzipped)
- First Contentful Paint: <2s
- Time to Interactive: <3s

## Security

- HTTPS enforced in production
- XSS protection via React's built-in escaping
- CSRF token support
- Content Security Policy headers
- No sensitive data in localStorage

## Deployment

### Docker (Recommended)

```bash
# Build Docker image
docker build -t vcci-frontend .

# Run container
docker run -p 3000:80 vcci-frontend
```

### Static Hosting

Build and deploy to any static hosting service:
- AWS S3 + CloudFront
- Netlify
- Vercel
- GitHub Pages

## Troubleshooting

### Common Issues

**Port 3000 already in use:**
```bash
# Use different port
PORT=3001 npm start
```

**API connection errors:**
- Check `REACT_APP_API_URL` in `.env`
- Verify backend is running
- Check CORS configuration

**Build failures:**
- Clear node_modules: `rm -rf node_modules && npm install`
- Clear build cache: `rm -rf build`

## Contributing

1. Follow TypeScript best practices
2. Use functional components with hooks
3. Write unit tests for components
4. Follow Material-UI design patterns
5. Update documentation for new features

## License

Proprietary - GL-VCCI Team

## Support

For issues or questions, contact: support@greenlang.ai
