# GreenLang Review Console

Web-based review interface for GL-FOUND-X-003 Entity Resolution system. This application provides a streamlined workflow for human reviewers to validate and correct entity normalization decisions.

## Features

- **Dashboard**: Overview statistics including pending items, resolution rates, and performance metrics
- **Queue Management**: Paginated list with filtering by entity type, status, date range, and search
- **Item Detail View**: Comprehensive view of original input, candidate matches with confidence scores
- **Resolution Workflow**: Quick actions for accept, reject, defer, and escalate decisions
- **Keyboard Shortcuts**: Power user support with comprehensive keyboard navigation

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type-safe development
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first styling
- **React Query** - Server state management
- **React Router** - Client-side routing
- **Heroicons** - Icon library

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

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

### Environment Configuration

Copy `.env.example` to `.env.local` and configure:

```env
VITE_API_URL=/api/v1
VITE_APP_TITLE=GreenLang Review Console
```

## Project Structure

```
src/
├── api/                 # API client and types
│   ├── client.ts       # Axios-based API client
│   └── types.ts        # TypeScript interfaces
├── components/          # React components
│   ├── Dashboard.tsx   # Overview dashboard
│   ├── Sidebar.tsx     # Navigation sidebar
│   ├── QueueList.tsx   # Queue listing page
│   ├── QueueItem.tsx   # Queue item card
│   ├── CandidateCard.tsx    # Candidate match display
│   ├── ResolutionForm.tsx   # Resolution submission form
│   └── ItemDetailView.tsx   # Full item detail page
├── hooks/              # Custom React hooks
│   ├── useQueue.ts     # Queue data fetching
│   ├── useResolution.ts     # Resolution management
│   └── useKeyboardShortcuts.ts  # Keyboard support
├── styles/             # Global styles
│   └── globals.css     # Tailwind + custom CSS
├── App.tsx             # Root component with routing
└── main.tsx            # Application entry point
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` | Accept top candidate |
| `R` | Reject all candidates |
| `D` | Defer item |
| `E` | Escalate item |
| `S` | Skip item |
| `1-9` | Select candidate by number |
| `Ctrl+Enter` | Submit resolution |
| `Escape` | Cancel / Reset |
| `?` | Show shortcuts help |

## API Integration

The frontend expects a REST API at `/api/v1` with the following endpoints:

- `GET /review/queue` - List queue items
- `GET /review/queue/:id` - Get item details
- `POST /review/queue/:id/claim` - Claim item for review
- `POST /review/queue/:id/resolve` - Submit resolution
- `POST /review/queue/:id/skip` - Skip item
- `POST /review/queue/:id/escalate` - Escalate item
- `GET /review/dashboard/stats` - Dashboard statistics

## Development

### Code Style

- ESLint for linting
- Prettier for formatting (configure as needed)
- TypeScript strict mode enabled

### Testing

```bash
# Run tests
npm run test

# Run tests with UI
npm run test:ui

# Generate coverage report
npm run test:coverage
```

## Accessibility

This application follows WCAG 2.1 AA guidelines:

- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- Focus management
- Color contrast compliance
- Screen reader friendly

## License

Proprietary - GreenLang Inc.
