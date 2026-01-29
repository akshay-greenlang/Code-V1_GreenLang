# Review Console Web UI

React-based web interface for the GreenLang Normalizer Review Console.

## Overview

The Review Console provides a human-in-the-loop interface for:

- **Vocabulary Governance**: Review and approve vocabulary changes
- **Resolution Review**: Review low-confidence matches
- **Policy Overrides**: Approve policy exception requests
- **Audit Trail**: View complete audit history

## Technology Stack

- React 18
- TypeScript 5
- Vite
- TailwindCSS
- Tanstack Query (React Query)
- React Router

## Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Variables

Create `.env.local`:

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_AUTH_DOMAIN=auth.greenlang.io
VITE_AUTH_CLIENT_ID=your-client-id
```

## Project Structure

```
web/
├── src/
│   ├── api/           # API client and types
│   ├── components/    # Reusable components
│   ├── hooks/         # Custom React hooks
│   ├── pages/         # Page components
│   ├── styles/        # Global styles
│   ├── utils/         # Utility functions
│   ├── App.tsx        # Root component
│   └── main.tsx       # Entry point
├── public/            # Static assets
├── index.html         # HTML template
├── vite.config.ts     # Vite configuration
└── tailwind.config.js # Tailwind configuration
```

## Features

### Review Queue

- View pending reviews by type
- Sort and filter by priority, date, confidence
- Batch approve/reject multiple items
- Inline review with context

### Vocabulary Management

- Browse vocabulary entries
- Submit change requests
- View change history
- Compare versions

### Resolution Review

- View low-confidence matches
- Confirm or correct matches
- Add new vocabulary entries
- Train matching model

### Dashboard

- Queue statistics
- Review velocity
- Top reviewers
- Recent activity

## Components

### ReviewQueue

```tsx
import { ReviewQueue } from './components/ReviewQueue';

<ReviewQueue
  type="resolution"
  onReview={handleReview}
  filters={{ minConfidence: 50, maxConfidence: 90 }}
/>
```

### VocabBrowser

```tsx
import { VocabBrowser } from './components/VocabBrowser';

<VocabBrowser
  vocabulary="fuels"
  onSelect={handleSelect}
  searchable
/>
```

### ResolutionReview

```tsx
import { ResolutionReview } from './components/ResolutionReview';

<ResolutionReview
  item={resolutionItem}
  onConfirm={handleConfirm}
  onReject={handleReject}
/>
```

## API Integration

The web UI communicates with the Review Console API:

```typescript
import { useQueue } from './hooks/useQueue';

const { data, isLoading, error } = useQueue({
  type: 'resolution',
  status: 'pending',
  limit: 50,
});
```

## Deployment

### Docker

```bash
docker build -t gl-review-console-web .
docker run -p 3000:80 gl-review-console-web
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: review-console-web
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: web
        image: greenlang/review-console-web:latest
        ports:
        - containerPort: 80
```

## License

Copyright (c) 2024-2026 GreenLang. All rights reserved.
