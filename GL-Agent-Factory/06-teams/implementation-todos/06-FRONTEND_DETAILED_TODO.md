# GreenLang Agent Factory - Frontend UI/UX Development Detailed TODO

**Version:** 1.0.0
**Date:** December 4, 2025
**Team Lead:** GL-FrontendDeveloper
**Priority:** P1-P2
**Status:** Ready for Implementation
**Total Tasks:** 910 tasks

---

## Executive Summary

This document provides the comprehensive frontend implementation roadmap for the GreenLang Agent Factory platform. The frontend encompasses three major applications: Admin Dashboard, User Portal, and Agent Builder UI, all built on a robust design system foundation.

### Technology Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Framework** | React | 18.2+ |
| **Language** | TypeScript | 5.0+ |
| **Build Tool** | Vite | 5.0+ |
| **State Management** | React Query + Zustand | Latest |
| **Styling** | TailwindCSS + CSS Modules | 3.4+ |
| **UI Components** | Radix UI Primitives | Latest |
| **Forms** | React Hook Form + Zod | Latest |
| **Charts** | Plotly.js + Recharts | Latest |
| **Maps** | Mapbox GL / Leaflet | Latest |
| **Testing** | Jest + Playwright + axe-core | Latest |
| **i18n** | react-i18next | Latest |

### Task Distribution Summary

| Section | Tasks | Priority | Timeline |
|---------|-------|----------|----------|
| Design System Foundation | 296 | P1 | Week 1-6 |
| Admin Dashboard | 204 | P1 | Week 7-14 |
| User Portal | 242 | P1-P2 | Week 15-24 |
| Agent Builder UI | 168 | P2 | Week 25-32 |
| **TOTAL** | **910** | - | **32 weeks** |

---

## Phase 1: Design System Foundation (296 tasks)

### 1.1 Design Tokens (48 tasks)

#### 1.1.1 Color System (18 tasks)

- [ ] **P1** Define primary brand color palette (GreenLang green: #10B981)
  - Success Criteria: Primary color with 10 shade variants (50-900)
- [ ] **P1** Define primary color variants (50, 100, 200, 300, 400, 500, 600, 700, 800, 900)
  - Success Criteria: All variants pass WCAG contrast requirements
- [ ] **P1** Define secondary color palette (blue: #3B82F6)
  - Success Criteria: Secondary color with 10 shade variants
- [ ] **P1** Define secondary color variants (50-900)
  - Success Criteria: All variants documented with usage guidelines
- [ ] **P1** Define accent color palette (amber: #F59E0B)
  - Success Criteria: Accent color complements primary/secondary
- [ ] **P1** Define accent color variants (50-900)
  - Success Criteria: All variants accessible
- [ ] **P1** Define semantic success color (#22C55E) with variants
  - Success Criteria: Success states clearly distinguishable
- [ ] **P1** Define semantic warning color (#EAB308) with variants
  - Success Criteria: Warning states clearly distinguishable
- [ ] **P1** Define semantic error color (#EF4444) with variants
  - Success Criteria: Error states clearly distinguishable
- [ ] **P1** Define semantic info color (#3B82F6) with variants
  - Success Criteria: Info states clearly distinguishable
- [ ] **P1** Define neutral gray scale (50-950)
  - Success Criteria: 11 gray shades for backgrounds, borders, text
- [ ] **P1** Define dark mode primary colors
  - Success Criteria: All primary variants for dark theme
- [ ] **P1** Define dark mode secondary colors
  - Success Criteria: All secondary variants for dark theme
- [ ] **P1** Define dark mode semantic colors
  - Success Criteria: All semantic colors for dark theme
- [ ] **P1** Create CSS custom properties for all colors
  - Success Criteria: CSS variables in :root and .dark selectors
- [ ] **P1** Create Tailwind configuration color extensions
  - Success Criteria: tailwind.config.ts with all custom colors
- [ ] **P1** Validate WCAG 2.1 AA contrast ratios for all text/background combinations
  - Success Criteria: All text meets 4.5:1 minimum, large text 3:1
- [ ] **P1** Document color usage guidelines with examples
  - Success Criteria: Design system documentation with color usage

#### 1.1.2 Typography System (16 tasks)

- [ ] **P1** Select and configure primary font family (Inter for UI)
  - Success Criteria: Font loaded with fallbacks, variable font optimized
- [ ] **P1** Select and configure secondary font family (JetBrains Mono for code)
  - Success Criteria: Monospace font for code blocks and terminals
- [ ] **P1** Define font size scale (xs: 12px, sm: 14px, base: 16px, lg: 18px, xl: 20px, 2xl: 24px, 3xl: 30px, 4xl: 36px, 5xl: 48px)
  - Success Criteria: 9 font sizes with rem values
- [ ] **P1** Define font weight scale (normal: 400, medium: 500, semibold: 600, bold: 700)
  - Success Criteria: 4 font weights configured
- [ ] **P1** Define line height scale (tight: 1.25, snug: 1.375, normal: 1.5, relaxed: 1.625, loose: 2)
  - Success Criteria: 5 line height options
- [ ] **P1** Define letter spacing values (tighter: -0.05em, tight: -0.025em, normal: 0, wide: 0.025em, wider: 0.05em)
  - Success Criteria: 5 letter spacing options
- [ ] **P1** Create heading styles (h1: 48px/bold, h2: 36px/bold, h3: 30px/semibold, h4: 24px/semibold, h5: 20px/medium, h6: 18px/medium)
  - Success Criteria: 6 heading levels with consistent hierarchy
- [ ] **P1** Create body text styles (body-lg, body-base, body-sm)
  - Success Criteria: 3 body text variants
- [ ] **P1** Create label styles (label-lg, label-base, label-sm)
  - Success Criteria: 3 label variants for form elements
- [ ] **P1** Create caption styles (caption-base, caption-sm)
  - Success Criteria: 2 caption variants for supplementary text
- [ ] **P1** Create display text styles (display-lg, display-base)
  - Success Criteria: Large hero/display text styles
- [ ] **P1** Create code/mono text styles
  - Success Criteria: Inline code and code block styles
- [ ] **P1** Implement responsive typography scaling
  - Success Criteria: Font sizes adjust for mobile/tablet/desktop
- [ ] **P1** Configure font loading optimization (font-display: swap)
  - Success Criteria: No FOUT/FOIT, fast font loading
- [ ] **P1** Create text truncation utilities (line-clamp-1, 2, 3)
  - Success Criteria: Multi-line text truncation support
- [ ] **P1** Document typography guidelines with examples
  - Success Criteria: Typography documentation complete

#### 1.1.3 Spacing & Layout Tokens (14 tasks)

- [ ] **P1** Define spacing scale (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96)
  - Success Criteria: 19 spacing values in 4px increments
- [ ] **P1** Define border radius scale (none: 0, sm: 2px, md: 4px, lg: 8px, xl: 12px, 2xl: 16px, 3xl: 24px, full: 9999px)
  - Success Criteria: 8 border radius options
- [ ] **P1** Define shadow scale (sm, md, lg, xl, 2xl, inner)
  - Success Criteria: 6 shadow options for elevation
- [ ] **P1** Define responsive breakpoints (sm: 640px, md: 768px, lg: 1024px, xl: 1280px, 2xl: 1536px)
  - Success Criteria: 5 breakpoints configured
- [ ] **P1** Define z-index scale (dropdown: 50, sticky: 100, fixed: 200, modal-backdrop: 300, modal: 400, popover: 500, tooltip: 600)
  - Success Criteria: 7 z-index levels
- [ ] **P1** Define container max widths (sm: 640px, md: 768px, lg: 1024px, xl: 1280px, 2xl: 1536px, full: 100%)
  - Success Criteria: 6 container width options
- [ ] **P1** Define grid column system (12-column grid)
  - Success Criteria: 12-column grid with gap utilities
- [ ] **P1** Define aspect ratio utilities (square, video, photo, portrait)
  - Success Criteria: 4 aspect ratio presets
- [ ] **P1** Create layout spacing utilities (gap, space-x, space-y)
  - Success Criteria: Consistent spacing between elements
- [ ] **P1** Define transition durations (fast: 100ms, normal: 200ms, slow: 300ms, slower: 500ms)
  - Success Criteria: 4 animation timing options
- [ ] **P1** Define transition timing functions (ease-in, ease-out, ease-in-out, spring)
  - Success Criteria: 4 easing functions
- [ ] **P1** Create CSS custom properties for all spacing/layout tokens
  - Success Criteria: CSS variables for all tokens
- [ ] **P1** Configure Tailwind with all spacing/layout extensions
  - Success Criteria: tailwind.config.ts updated
- [ ] **P1** Document spacing and layout guidelines
  - Success Criteria: Layout documentation complete

### 1.2 Icon Library (24 tasks)

#### 1.2.1 Icon System Setup (8 tasks)

- [ ] **P1** Select and integrate primary icon library (Heroicons)
  - Success Criteria: Heroicons installed and configured
- [ ] **P1** Select and integrate secondary icon library (Lucide)
  - Success Criteria: Lucide icons available as fallback
- [ ] **P1** Create IconWrapper component with size variants (xs: 12px, sm: 16px, md: 20px, lg: 24px, xl: 32px)
  - Success Criteria: Consistent icon sizing component
- [ ] **P1** Implement icon color inheritance from parent
  - Success Criteria: Icons inherit text color by default
- [ ] **P1** Create icon button component (IconButton)
  - Success Criteria: Accessible icon-only buttons
- [ ] **P1** Implement icon loading optimization (tree-shaking)
  - Success Criteria: Only used icons in bundle
- [ ] **P1** Create SVG sprite generation for custom icons
  - Success Criteria: Custom icons in sprite sheet
- [ ] **P1** Document icon usage patterns and guidelines
  - Success Criteria: Icon documentation complete

#### 1.2.2 Icon Sets Creation (16 tasks)

- [ ] **P1** Create navigation icons set (home, dashboard, settings, users, agents, reports, data, analytics)
  - Success Criteria: 8 navigation icons
- [ ] **P1** Create action icons set (add, edit, delete, save, cancel, copy, paste, undo, redo, refresh)
  - Success Criteria: 10 action icons
- [ ] **P1** Create status icons set (success, warning, error, info, pending, loading, complete, blocked)
  - Success Criteria: 8 status icons
- [ ] **P1** Create chart icons set (bar, line, pie, area, scatter, donut, treemap, sankey)
  - Success Criteria: 8 chart type icons
- [ ] **P1** Create file icons set (pdf, excel, csv, json, xml, word, image, generic)
  - Success Criteria: 8 file type icons
- [ ] **P1** Create emissions icons set (co2, scope1, scope2, scope3, offset, reduction, target, footprint)
  - Success Criteria: 8 emissions-related icons
- [ ] **P1** Create regulatory icons set (cbam, csrd, eudr, sbti, taxonomy, gri, cdp, tcfd)
  - Success Criteria: 8 regulatory framework icons
- [ ] **P1** Create agent icons set (analyzer, calculator, validator, reporter, builder, monitor)
  - Success Criteria: 6 agent type icons
- [ ] **P1** Create data icons set (upload, download, import, export, sync, connect, database, api)
  - Success Criteria: 8 data operation icons
- [ ] **P1** Create user icons set (user, users, team, organization, role, permission, profile, avatar)
  - Success Criteria: 8 user-related icons
- [ ] **P1** Create notification icons set (bell, mail, message, alert, announcement, badge)
  - Success Criteria: 6 notification icons
- [ ] **P1** Create GreenLang logo variations (full, icon, wordmark, favicon)
  - Success Criteria: 4 logo variations
- [ ] **P1** Create favicon set (16x16, 32x32, 48x48, apple-touch-icon 180x180)
  - Success Criteria: Complete favicon set
- [ ] **P1** Create loading/spinner icons (spinner, dots, bars, pulse)
  - Success Criteria: 4 loading indicators
- [ ] **P1** Create empty state illustrations (no-data, no-results, error, maintenance)
  - Success Criteria: 4 empty state illustrations
- [ ] **P1** Export all icons as React components with TypeScript types
  - Success Criteria: Type-safe icon components

### 1.3 Core UI Components (96 tasks)

#### 1.3.1 Button Components (20 tasks)

- [ ] **P1** Create Button component base structure with forwardRef
  - Success Criteria: Button component with ref forwarding
- [ ] **P1** Implement Button variant: primary (filled, brand color)
  - Success Criteria: Primary button with hover/focus states
- [ ] **P1** Implement Button variant: secondary (filled, secondary color)
  - Success Criteria: Secondary button with hover/focus states
- [ ] **P1** Implement Button variant: outline (bordered, transparent background)
  - Success Criteria: Outline button with hover/focus states
- [ ] **P1** Implement Button variant: ghost (no border, transparent)
  - Success Criteria: Ghost button with hover/focus states
- [ ] **P1** Implement Button variant: destructive (danger/error color)
  - Success Criteria: Destructive button for delete actions
- [ ] **P1** Implement Button variant: link (text-only, underlined)
  - Success Criteria: Link-styled button
- [ ] **P1** Implement Button sizes: sm (32px), md (40px), lg (48px)
  - Success Criteria: 3 button size options
- [ ] **P1** Implement Button with left icon slot
  - Success Criteria: Icon before text support
- [ ] **P1** Implement Button with right icon slot
  - Success Criteria: Icon after text support
- [ ] **P1** Implement Button loading state with spinner
  - Success Criteria: Loading indicator replaces content
- [ ] **P1** Implement Button disabled state
  - Success Criteria: Disabled styling and aria-disabled
- [ ] **P1** Implement IconButton component (icon-only button)
  - Success Criteria: Square button for icons only
- [ ] **P1** Implement ButtonGroup component (grouped buttons)
  - Success Criteria: Connected button group with dividers
- [ ] **P1** Add Button keyboard navigation (Enter, Space)
  - Success Criteria: Keyboard activation works
- [ ] **P1** Add Button ARIA attributes (role, aria-pressed, aria-expanded)
  - Success Criteria: Screen reader compatible
- [ ] **P1** Implement Button as link (asChild prop with Radix Slot)
  - Success Criteria: Button renders as anchor when needed
- [ ] **P1** Write Button unit tests (render, click, disabled, loading)
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Button Storybook stories (all variants, sizes, states)
  - Success Criteria: Complete Storybook documentation
- [ ] **P1** Document Button component API and usage
  - Success Criteria: Component documentation complete

#### 1.3.2 Input Components (24 tasks)

- [ ] **P1** Create TextInput component base structure
  - Success Criteria: Text input with forwardRef
- [ ] **P1** Implement TextInput sizes: sm (32px), md (40px), lg (48px)
  - Success Criteria: 3 input size options
- [ ] **P1** Implement TextInput with label (FormField wrapper)
  - Success Criteria: Label associated with input
- [ ] **P1** Implement TextInput with helper text
  - Success Criteria: Description text below input
- [ ] **P1** Implement TextInput with error state and message
  - Success Criteria: Error styling and aria-invalid
- [ ] **P1** Implement TextInput with prefix (text/icon before)
  - Success Criteria: Prefix slot support
- [ ] **P1** Implement TextInput with suffix (text/icon after)
  - Success Criteria: Suffix slot support
- [ ] **P1** Implement TextInput with clear button
  - Success Criteria: X button to clear value
- [ ] **P1** Create NumberInput component with increment/decrement buttons
  - Success Criteria: Number input with stepper controls
- [ ] **P1** Implement NumberInput min/max/step validation
  - Success Criteria: Number constraints enforced
- [ ] **P1** Implement NumberInput with unit display (kg, tCO2e, kWh)
  - Success Criteria: Unit suffix in input
- [ ] **P1** Create TextArea component with auto-resize
  - Success Criteria: Textarea grows with content
- [ ] **P1** Implement TextArea character count display
  - Success Criteria: Character count with max limit
- [ ] **P1** Implement TextArea min/max rows configuration
  - Success Criteria: Height constraints
- [ ] **P1** Create SearchInput component with icon
  - Success Criteria: Search input with magnifying glass
- [ ] **P1** Implement SearchInput with autocomplete dropdown
  - Success Criteria: Suggestions as user types
- [ ] **P1** Implement SearchInput with keyboard navigation
  - Success Criteria: Arrow keys navigate suggestions
- [ ] **P1** Create PasswordInput component with show/hide toggle
  - Success Criteria: Eye icon toggles visibility
- [ ] **P1** Implement PasswordInput strength indicator
  - Success Criteria: Visual strength meter
- [ ] **P1** Add Input keyboard navigation (Tab, Escape to clear)
  - Success Criteria: Keyboard interactions work
- [ ] **P1** Add Input ARIA attributes (describedby, invalid, required)
  - Success Criteria: Screen reader compatible
- [ ] **P1** Write Input component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Input Storybook stories
  - Success Criteria: Complete Storybook documentation
- [ ] **P1** Document Input component API and usage
  - Success Criteria: Component documentation complete

#### 1.3.3 Select Components (16 tasks)

- [ ] **P1** Create Select component with Radix Select primitive
  - Success Criteria: Accessible dropdown select
- [ ] **P1** Implement Select with search/filter functionality
  - Success Criteria: Typeahead filtering of options
- [ ] **P1** Implement Select multi-select variant
  - Success Criteria: Multiple selection with chips
- [ ] **P1** Implement Select with option groups
  - Success Criteria: Grouped options with headers
- [ ] **P1** Implement Select with custom option rendering
  - Success Criteria: Custom option content (icons, descriptions)
- [ ] **P1** Implement Select disabled options
  - Success Criteria: Individual options can be disabled
- [ ] **P1** Create Combobox component (Select + custom input)
  - Success Criteria: Editable dropdown with suggestions
- [ ] **P1** Create CountrySelect component (ISO 3166 countries)
  - Success Criteria: Pre-built country selector with flags
- [ ] **P1** Create CurrencySelect component (ISO 4217 currencies)
  - Success Criteria: Pre-built currency selector
- [ ] **P1** Create UnitSelect component (emissions units: tCO2e, kgCO2e, etc.)
  - Success Criteria: Pre-built unit selector
- [ ] **P1** Create YearSelect component (range of years)
  - Success Criteria: Year selector with range
- [ ] **P1** Add Select keyboard navigation (Arrow, Enter, Escape, Type-ahead)
  - Success Criteria: Full keyboard support
- [ ] **P1** Add Select ARIA attributes (expanded, activedescendant, listbox)
  - Success Criteria: Screen reader compatible
- [ ] **P1** Write Select component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Select Storybook stories
  - Success Criteria: Complete Storybook documentation
- [ ] **P1** Document Select component API and usage
  - Success Criteria: Component documentation complete

#### 1.3.4 Checkbox & Radio Components (12 tasks)

- [ ] **P1** Create Checkbox component with Radix Checkbox primitive
  - Success Criteria: Accessible checkbox with custom styling
- [ ] **P1** Implement Checkbox indeterminate state
  - Success Criteria: Minus icon for partial selection
- [ ] **P1** Implement Checkbox with label and description
  - Success Criteria: Label and helper text support
- [ ] **P1** Create CheckboxGroup component (multiple checkboxes)
  - Success Criteria: Group with Select All option
- [ ] **P1** Create Radio component with Radix RadioGroup primitive
  - Success Criteria: Accessible radio button
- [ ] **P1** Create RadioGroup component with layout options
  - Success Criteria: Horizontal and vertical layouts
- [ ] **P1** Implement RadioGroup with descriptions per option
  - Success Criteria: Rich radio card options
- [ ] **P1** Create Switch/Toggle component
  - Success Criteria: On/off toggle switch
- [ ] **P1** Implement Switch with label and description
  - Success Criteria: Label support with placement options
- [ ] **P1** Add Checkbox/Radio keyboard navigation (Space, Tab)
  - Success Criteria: Keyboard activation works
- [ ] **P1** Write Checkbox/Radio unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Checkbox/Radio Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.3.5 Date & Time Components (12 tasks)

- [ ] **P1** Create DatePicker component with calendar popup
  - Success Criteria: Date selection with calendar UI
- [ ] **P1** Implement DatePicker with date formatting options (locale-aware)
  - Success Criteria: Localized date display
- [ ] **P1** Implement DatePicker min/max date constraints
  - Success Criteria: Date range restrictions
- [ ] **P1** Implement DatePicker with disabled dates (weekends, holidays)
  - Success Criteria: Specific dates can be disabled
- [ ] **P1** Create DateRangePicker component (start/end dates)
  - Success Criteria: Two-date range selection
- [ ] **P1** Create MonthPicker component
  - Success Criteria: Month/year selection only
- [ ] **P1** Create YearPicker component
  - Success Criteria: Year selection only
- [ ] **P1** Create TimePicker component (hours, minutes)
  - Success Criteria: Time selection UI
- [ ] **P1** Create DateTimePicker component (date + time)
  - Success Criteria: Combined date and time selection
- [ ] **P1** Implement date localization (date-fns locales)
  - Success Criteria: Dates formatted per locale
- [ ] **P1** Write DatePicker unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create DatePicker Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.3.6 Form Wrapper Components (12 tasks)

- [ ] **P1** Create Form component with React Hook Form integration
  - Success Criteria: FormProvider wrapper component
- [ ] **P1** Create FormField component (label + input + error)
  - Success Criteria: Consistent form field structure
- [ ] **P1** Implement FormField with required indicator (*)
  - Success Criteria: Visual required marker
- [ ] **P1** Implement FormField error message display
  - Success Criteria: Error messages below field
- [ ] **P1** Implement FormField with helper/description text
  - Success Criteria: Description above or below field
- [ ] **P1** Create FormSection component (grouped fields with title)
  - Success Criteria: Section grouping with heading
- [ ] **P1** Create FormActions component (submit/cancel buttons)
  - Success Criteria: Consistent form footer
- [ ] **P1** Implement form validation schemas with Zod
  - Success Criteria: Type-safe validation
- [ ] **P1** Implement form dirty state tracking
  - Success Criteria: Unsaved changes detection
- [ ] **P1** Implement form auto-save functionality
  - Success Criteria: Debounced auto-save option
- [ ] **P1** Write Form component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Form Storybook stories
  - Success Criteria: Complete Storybook documentation

### 1.4 Feedback Components (44 tasks)

#### 1.4.1 Alert Components (12 tasks)

- [ ] **P1** Create Alert component base structure
  - Success Criteria: Accessible alert container
- [ ] **P1** Implement Alert variant: info (blue, info icon)
  - Success Criteria: Informational alert style
- [ ] **P1** Implement Alert variant: success (green, check icon)
  - Success Criteria: Success alert style
- [ ] **P1** Implement Alert variant: warning (amber, warning icon)
  - Success Criteria: Warning alert style
- [ ] **P1** Implement Alert variant: error (red, error icon)
  - Success Criteria: Error alert style
- [ ] **P1** Implement Alert with title and description
  - Success Criteria: Two-line alert content
- [ ] **P1** Implement Alert with dismiss button
  - Success Criteria: X button to close alert
- [ ] **P1** Implement Alert with action button
  - Success Criteria: CTA button within alert
- [ ] **P1** Create AlertBanner component (full-width, sticky)
  - Success Criteria: Page-level banner alerts
- [ ] **P1** Add Alert ARIA live region attributes
  - Success Criteria: role="alert", aria-live="polite"
- [ ] **P1** Write Alert unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Alert Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.4.2 Toast/Notification Components (14 tasks)

- [ ] **P1** Create Toast component with Radix Toast primitive
  - Success Criteria: Accessible toast notification
- [ ] **P1** Implement Toast variants (info, success, warning, error)
  - Success Criteria: 4 toast variants with icons
- [ ] **P1** Implement Toast with progress indicator (auto-dismiss timer)
  - Success Criteria: Visual countdown bar
- [ ] **P1** Implement Toast auto-dismiss with configurable duration
  - Success Criteria: 3/5/10 second options
- [ ] **P1** Implement Toast with action button
  - Success Criteria: Undo/View action in toast
- [ ] **P1** Create ToastProvider context
  - Success Criteria: Context for toast management
- [ ] **P1** Create useToast hook for programmatic toasts
  - Success Criteria: Hook returns toast methods
- [ ] **P1** Implement Toast stack positioning (top-right, bottom-right, etc.)
  - Success Criteria: Configurable position
- [ ] **P1** Implement Toast max visible limit (3-5 toasts)
  - Success Criteria: Queue management for many toasts
- [ ] **P1** Implement Toast swipe-to-dismiss (touch devices)
  - Success Criteria: Swipe gesture support
- [ ] **P1** Add Toast ARIA attributes (status, alert)
  - Success Criteria: Screen reader compatible
- [ ] **P1** Implement Toast persistence option (no auto-dismiss)
  - Success Criteria: Sticky toasts for critical messages
- [ ] **P1** Write Toast unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Toast Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.4.3 Modal/Dialog Components (18 tasks)

- [ ] **P1** Create Modal component with Radix Dialog primitive
  - Success Criteria: Accessible modal dialog
- [ ] **P1** Implement Modal sizes (sm: 400px, md: 560px, lg: 720px, xl: 900px, full: 100%)
  - Success Criteria: 5 modal size options
- [ ] **P1** Implement Modal header with title and close button
  - Success Criteria: Consistent modal header
- [ ] **P1** Implement Modal body with scroll area
  - Success Criteria: Scrollable content area
- [ ] **P1** Implement Modal footer with action buttons
  - Success Criteria: Consistent footer with alignment
- [ ] **P1** Implement Modal with form content support
  - Success Criteria: Forms work correctly in modal
- [ ] **P1** Create ConfirmDialog component (confirm/cancel actions)
  - Success Criteria: Pre-built confirmation modal
- [ ] **P1** Create AlertDialog component (acknowledge only)
  - Success Criteria: Pre-built alert modal
- [ ] **P1** Create DeleteDialog component (destructive confirmation)
  - Success Criteria: Pre-built delete confirmation
- [ ] **P1** Implement Modal backdrop click to close (optional)
  - Success Criteria: Configurable backdrop behavior
- [ ] **P1** Implement Modal escape key to close
  - Success Criteria: Keyboard dismiss support
- [ ] **P1** Implement focus trap within Modal
  - Success Criteria: Tab stays within modal
- [ ] **P1** Implement focus restoration on Modal close
  - Success Criteria: Focus returns to trigger
- [ ] **P1** Implement Modal entry/exit animations
  - Success Criteria: Smooth open/close transitions
- [ ] **P1** Add Modal ARIA attributes (dialog, labelledby, describedby)
  - Success Criteria: Screen reader compatible
- [ ] **P1** Implement nested Modal support (stacked modals)
  - Success Criteria: Multiple modals can stack
- [ ] **P1** Write Modal unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Modal Storybook stories
  - Success Criteria: Complete Storybook documentation

### 1.5 Loading & Progress Components (20 tasks)

- [ ] **P1** Create Spinner component with size variants
  - Success Criteria: Loading spinner animation
- [ ] **P1** Implement Spinner sizes (xs, sm, md, lg, xl)
  - Success Criteria: 5 spinner sizes
- [ ] **P1** Create Skeleton component for loading states
  - Success Criteria: Placeholder loading animation
- [ ] **P1** Implement Skeleton variants (text, circle, rectangle)
  - Success Criteria: 3 skeleton shapes
- [ ] **P1** Create SkeletonCard component (card placeholder)
  - Success Criteria: Pre-built card skeleton
- [ ] **P1** Create SkeletonTable component (table placeholder)
  - Success Criteria: Pre-built table skeleton
- [ ] **P1** Create SkeletonList component (list placeholder)
  - Success Criteria: Pre-built list skeleton
- [ ] **P1** Create ProgressBar component (horizontal bar)
  - Success Criteria: Linear progress indicator
- [ ] **P1** Implement ProgressBar determinate mode (percentage)
  - Success Criteria: Shows actual progress
- [ ] **P1** Implement ProgressBar indeterminate mode (animation)
  - Success Criteria: Continuous animation for unknown duration
- [ ] **P1** Implement ProgressBar with label
  - Success Criteria: Shows percentage or custom label
- [ ] **P1** Create ProgressCircle component (circular progress)
  - Success Criteria: Circular progress indicator
- [ ] **P1** Create LoadingOverlay component (covers parent)
  - Success Criteria: Overlay with spinner
- [ ] **P1** Create LoadingButton component (button with loading state)
  - Success Criteria: Button shows spinner when loading
- [ ] **P1** Implement loading state transitions
  - Success Criteria: Smooth loading state changes
- [ ] **P1** Create Suspense fallback components
  - Success Criteria: React Suspense boundaries
- [ ] **P1** Add loading component ARIA attributes (busy, live)
  - Success Criteria: Screen reader announces loading
- [ ] **P1** Implement reduced motion preference for loaders
  - Success Criteria: Respects prefers-reduced-motion
- [ ] **P1** Write Loading component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Loading Storybook stories
  - Success Criteria: Complete Storybook documentation

### 1.6 Layout Components (40 tasks)

#### 1.6.1 Container & Grid (12 tasks)

- [ ] **P1** Create Container component with max-width variants
  - Success Criteria: Centered content container
- [ ] **P1** Implement Container fluid option (full-width)
  - Success Criteria: No max-width constraint
- [ ] **P1** Create Grid component with 12-column system
  - Success Criteria: CSS Grid-based layout
- [ ] **P1** Implement Grid responsive columns (sm, md, lg, xl, 2xl)
  - Success Criteria: Column counts per breakpoint
- [ ] **P1** Implement Grid gap variants (sm, md, lg, xl)
  - Success Criteria: Configurable grid gaps
- [ ] **P1** Create Stack component (vertical stack)
  - Success Criteria: Flexbox column layout
- [ ] **P1** Create HStack component (horizontal stack)
  - Success Criteria: Flexbox row layout
- [ ] **P1** Implement Stack/HStack spacing options
  - Success Criteria: Configurable gaps
- [ ] **P1** Create Flex component (flexible layout)
  - Success Criteria: General-purpose flex container
- [ ] **P1** Implement responsive grid breakpoints
  - Success Criteria: Grid adapts to screen size
- [ ] **P1** Write Layout component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Layout Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.6.2 Card Components (12 tasks)

- [ ] **P1** Create Card component base structure
  - Success Criteria: Basic card container
- [ ] **P1** Implement Card header slot (title, actions)
  - Success Criteria: Card header section
- [ ] **P1** Implement Card body slot (content area)
  - Success Criteria: Card content section
- [ ] **P1** Implement Card footer slot (actions, metadata)
  - Success Criteria: Card footer section
- [ ] **P1** Implement Card with action buttons
  - Success Criteria: Button slots in header/footer
- [ ] **P1** Implement Card hover state (elevation change)
  - Success Criteria: Interactive card styling
- [ ] **P1** Implement Card clickable variant (as link)
  - Success Criteria: Card navigates on click
- [ ] **P1** Create StatCard component (KPI display)
  - Success Criteria: Number + label + trend
- [ ] **P1** Create MetricCard component (emissions metrics)
  - Success Criteria: Value + unit + comparison
- [ ] **P1** Create FeatureCard component (icon + title + description)
  - Success Criteria: Feature highlight card
- [ ] **P1** Write Card component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Card Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.6.3 Navigation Components (16 tasks)

- [ ] **P1** Create Navbar component (horizontal navigation)
  - Success Criteria: Top navigation bar
- [ ] **P1** Implement Navbar with logo slot
  - Success Criteria: Logo placement
- [ ] **P1** Implement Navbar with navigation links
  - Success Criteria: Nav items with active state
- [ ] **P1** Implement Navbar with user menu dropdown
  - Success Criteria: Profile/settings/logout menu
- [ ] **P1** Create Sidebar component (vertical navigation)
  - Success Criteria: Side navigation panel
- [ ] **P1** Implement Sidebar collapsible (icon-only mode)
  - Success Criteria: Sidebar collapses to icons
- [ ] **P1** Implement Sidebar nested navigation (accordion)
  - Success Criteria: Multi-level nav items
- [ ] **P1** Create Breadcrumb component (page hierarchy)
  - Success Criteria: Breadcrumb trail
- [ ] **P1** Create Tabs component (horizontal tabs)
  - Success Criteria: Tab navigation with panels
- [ ] **P1** Implement Tabs vertical variant
  - Success Criteria: Vertical tab layout
- [ ] **P1** Create Pagination component
  - Success Criteria: Page navigation controls
- [ ] **P1** Implement Pagination with page size selector
  - Success Criteria: Rows per page dropdown
- [ ] **P1** Create StepIndicator component (wizard steps)
  - Success Criteria: Multi-step progress indicator
- [ ] **P1** Implement StepIndicator with completed/active/pending states
  - Success Criteria: Step status visualization
- [ ] **P1** Write Navigation component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Navigation Storybook stories
  - Success Criteria: Complete Storybook documentation

### 1.7 Data Display Components (48 tasks)

#### 1.7.1 Table Components (24 tasks)

- [ ] **P1** Create Table component base structure
  - Success Criteria: Semantic HTML table
- [ ] **P1** Implement Table with sticky header
  - Success Criteria: Header stays visible on scroll
- [ ] **P1** Implement Table sortable columns (asc/desc)
  - Success Criteria: Click header to sort
- [ ] **P1** Implement Table column resizing (drag handle)
  - Success Criteria: Adjust column widths
- [ ] **P1** Implement Table row selection (single)
  - Success Criteria: Click to select one row
- [ ] **P1** Implement Table row selection (multi with checkboxes)
  - Success Criteria: Select multiple rows
- [ ] **P1** Implement Table row expansion (expandable rows)
  - Success Criteria: Click to expand row details
- [ ] **P1** Implement Table pagination integration
  - Success Criteria: Pagination below table
- [ ] **P1** Implement Table column filtering (per-column)
  - Success Criteria: Filter dropdown per column
- [ ] **P1** Implement Table global search
  - Success Criteria: Search across all columns
- [ ] **P1** Implement Table empty state display
  - Success Criteria: Empty state message
- [ ] **P1** Implement Table loading state (skeleton rows)
  - Success Criteria: Loading placeholder
- [ ] **P1** Implement Table error state display
  - Success Criteria: Error message with retry
- [ ] **P1** Create TableCell variants (text, number, date, status, actions)
  - Success Criteria: Pre-styled cell types
- [ ] **P1** Implement Table column visibility toggle
  - Success Criteria: Show/hide columns menu
- [ ] **P1** Implement Table column reordering (drag-drop)
  - Success Criteria: Rearrange columns
- [ ] **P1** Implement Table export to CSV
  - Success Criteria: Download as CSV file
- [ ] **P1** Implement Table export to Excel
  - Success Criteria: Download as XLSX file
- [ ] **P1** Implement Table virtualization for large datasets (react-virtual)
  - Success Criteria: 10,000+ rows performant
- [ ] **P1** Implement Table responsive behavior (horizontal scroll)
  - Success Criteria: Table scrolls on mobile
- [ ] **P1** Add Table keyboard navigation (Arrow keys, Tab)
  - Success Criteria: Navigate cells with keyboard
- [ ] **P1** Add Table ARIA attributes (grid, rowgroup, columnheader)
  - Success Criteria: Screen reader compatible
- [ ] **P1** Write Table component unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Table Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.7.2 Badge & Tag Components (10 tasks)

- [ ] **P1** Create Badge component (status indicator)
  - Success Criteria: Inline status badge
- [ ] **P1** Implement Badge variants (default, success, warning, error, info)
  - Success Criteria: 5 badge colors
- [ ] **P1** Implement Badge sizes (sm, md, lg)
  - Success Criteria: 3 badge sizes
- [ ] **P1** Implement Badge with dot indicator
  - Success Criteria: Small dot badge variant
- [ ] **P1** Create Tag component (removable label)
  - Success Criteria: Tag with optional remove button
- [ ] **P1** Implement Tag with remove button
  - Success Criteria: X button to remove tag
- [ ] **P1** Create TagInput component (multi-tag input)
  - Success Criteria: Type to add tags
- [ ] **P1** Implement TagInput with suggestions
  - Success Criteria: Autocomplete for tags
- [ ] **P1** Write Badge/Tag unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Badge/Tag Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.7.3 Tooltip & Popover Components (8 tasks)

- [ ] **P1** Create Tooltip component with Radix Tooltip primitive
  - Success Criteria: Accessible tooltip on hover
- [ ] **P1** Implement Tooltip positions (top, right, bottom, left)
  - Success Criteria: 4 positioning options
- [ ] **P1** Implement Tooltip with arrow indicator
  - Success Criteria: Arrow pointing to trigger
- [ ] **P1** Create Popover component with Radix Popover primitive
  - Success Criteria: Click-triggered popover
- [ ] **P1** Implement Popover with close button
  - Success Criteria: X button to close
- [ ] **P1** Implement Popover controlled/uncontrolled modes
  - Success Criteria: Programmatic control option
- [ ] **P1** Write Tooltip/Popover unit tests
  - Success Criteria: 95%+ test coverage
- [ ] **P1** Create Tooltip/Popover Storybook stories
  - Success Criteria: Complete Storybook documentation

#### 1.7.4 Avatar & User Display Components (6 tasks)

- [ ] **P1** Create Avatar component with image support
  - Success Criteria: User avatar circle
- [ ] **P1** Implement Avatar with initials fallback
  - Success Criteria: Initials when no image
- [ ] **P1** Implement Avatar sizes (xs, sm, md, lg, xl)
  - Success Criteria: 5 avatar sizes
- [ ] **P1** Create AvatarGroup component (stacked avatars)
  - Success Criteria: Multiple avatars with overlap
- [ ] **P1** Create UserInfo component (avatar + name + role)
  - Success Criteria: User display with details
- [ ] **P1** Write Avatar unit tests and Storybook stories
  - Success Criteria: Tests and documentation complete

### 1.8 Accessibility Implementation (24 tasks)

- [ ] **P1** Implement skip-to-content link (bypass navigation)
  - Success Criteria: Skip link appears on focus
- [ ] **P1** Implement focus visible styles (keyboard focus ring)
  - Success Criteria: Clear focus indicators
- [ ] **P1** Implement focus trap utility for modals/dialogs
  - Success Criteria: Tab trapped in modal
- [ ] **P1** Implement keyboard navigation for all interactive components
  - Success Criteria: All components keyboard accessible
- [ ] **P1** Add ARIA labels to all form inputs
  - Success Criteria: Inputs have accessible names
- [ ] **P1** Add ARIA describedby for error messages
  - Success Criteria: Errors announced by screen reader
- [ ] **P1** Implement screen reader announcements for dynamic content
  - Success Criteria: Live regions for updates
- [ ] **P1** Add role attributes to custom components
  - Success Criteria: Semantic roles applied
- [ ] **P1** Implement reduced motion preference support
  - Success Criteria: Respects prefers-reduced-motion
- [ ] **P1** Implement high contrast mode support
  - Success Criteria: Works with forced-colors
- [ ] **P1** Test all components with NVDA screen reader
  - Success Criteria: All components readable
- [ ] **P1** Test all components with VoiceOver (macOS)
  - Success Criteria: All components readable
- [ ] **P1** Test all components with keyboard only
  - Success Criteria: All functions accessible
- [ ] **P1** Run axe-core accessibility audit on all components
  - Success Criteria: Zero critical issues
- [ ] **P1** Fix all critical accessibility issues
  - Success Criteria: Critical issues resolved
- [ ] **P1** Fix all serious accessibility issues
  - Success Criteria: Serious issues resolved
- [ ] **P1** Document accessibility patterns
  - Success Criteria: A11y documentation complete
- [ ] **P1** Create accessibility testing checklist
  - Success Criteria: Checklist for QA
- [ ] **P1** Implement ARIA live regions for notifications
  - Success Criteria: Toasts announced
- [ ] **P1** Add alt text guidelines for images
  - Success Criteria: Image alt text guide
- [ ] **P1** Implement heading hierarchy validation
  - Success Criteria: Proper heading levels
- [ ] **P1** Add landmark regions (main, nav, aside, footer)
  - Success Criteria: Semantic landmarks
- [ ] **P1** Create accessibility statement page
  - Success Criteria: Public a11y statement
- [ ] **P1** Set up automated accessibility testing in CI
  - Success Criteria: axe-core in CI pipeline

---

## Phase 2: Admin Dashboard (204 tasks)

### 2.1 Dashboard Shell & Navigation (36 tasks)

#### 2.1.1 App Shell (20 tasks)

- [ ] **P1** Create AdminLayout component (shell structure)
  - Success Criteria: Main layout wrapper
- [ ] **P1** Implement AdminLayout with sidebar navigation
  - Success Criteria: Sidebar integrated
- [ ] **P1** Implement AdminLayout with top navbar
  - Success Criteria: Navbar integrated
- [ ] **P1** Implement AdminLayout with breadcrumbs
  - Success Criteria: Breadcrumb trail shown
- [ ] **P1** Implement AdminLayout responsive behavior (mobile drawer)
  - Success Criteria: Sidebar becomes drawer on mobile
- [ ] **P1** Create sidebar navigation structure (menu items, icons)
  - Success Criteria: Complete nav menu
- [ ] **P1** Implement sidebar collapse/expand toggle
  - Success Criteria: Toggle button works
- [ ] **P1** Implement sidebar active state highlighting
  - Success Criteria: Current page highlighted
- [ ] **P1** Implement sidebar nested menu items (submenu accordion)
  - Success Criteria: Multi-level navigation
- [ ] **P1** Create mobile navigation drawer (off-canvas)
  - Success Criteria: Mobile nav drawer
- [ ] **P1** Implement mobile hamburger menu button
  - Success Criteria: Menu toggle on mobile
- [ ] **P1** Create user profile dropdown (navbar)
  - Success Criteria: User menu with options
- [ ] **P1** Implement user profile menu items (profile, settings, logout)
  - Success Criteria: Menu actions work
- [ ] **P1** Create notification dropdown (navbar)
  - Success Criteria: Notification menu
- [ ] **P1** Implement notification badge counter
  - Success Criteria: Unread count shown
- [ ] **P1** Create global search command palette (Cmd+K)
  - Success Criteria: Quick search modal
- [ ] **P1** Implement quick actions menu
  - Success Criteria: Common actions shortcut
- [ ] **P1** Create footer component (version, links)
  - Success Criteria: Footer with info
- [ ] **P1** Implement breadcrumb auto-generation from routes
  - Success Criteria: Breadcrumbs match route
- [ ] **P1** Add route transition animations
  - Success Criteria: Smooth page transitions

#### 2.1.2 Error & Utility Pages (16 tasks)

- [ ] **P1** Implement page title sync with document.title
  - Success Criteria: Browser title matches page
- [ ] **P1** Create 404 Not Found page
  - Success Criteria: Custom 404 page
- [ ] **P1** Create 500 Error page
  - Success Criteria: Custom error page
- [ ] **P1** Create 403 Forbidden page
  - Success Criteria: Access denied page
- [ ] **P1** Create maintenance mode page
  - Success Criteria: Maintenance notification
- [ ] **P1** Implement session timeout warning modal
  - Success Criteria: Warning before timeout
- [ ] **P1** Implement session timeout redirect to login
  - Success Criteria: Redirect on expiry
- [ ] **P1** Create help/support modal
  - Success Criteria: Help resources modal
- [ ] **P1** Create keyboard shortcuts help modal (?)
  - Success Criteria: Shortcuts reference
- [ ] **P1** Implement theme switcher (light/dark mode)
  - Success Criteria: Theme toggle works
- [ ] **P1** Persist theme preference to localStorage
  - Success Criteria: Theme persists
- [ ] **P1** Implement system theme preference detection
  - Success Criteria: Respects prefers-color-scheme
- [ ] **P1** Add loading indicator for route changes
  - Success Criteria: Loading bar during navigation
- [ ] **P1** Implement scroll restoration on navigation
  - Success Criteria: Scroll position preserved
- [ ] **P1** Write AdminLayout unit tests
  - Success Criteria: Layout tests complete
- [ ] **P1** Create AdminLayout Storybook stories
  - Success Criteria: Layout documentation

### 2.2 Agent Management UI (60 tasks)

#### 2.2.1 Agent List View (20 tasks)

- [ ] **P1** Create AgentListPage component
  - Success Criteria: Agent list page route
- [ ] **P1** Implement agent list table with columns (name, version, status, type, updated)
  - Success Criteria: Table displays agents
- [ ] **P1** Implement agent list sorting (all columns)
  - Success Criteria: Sortable columns
- [ ] **P1** Implement agent list filtering by status (active, draft, deprecated)
  - Success Criteria: Status filter dropdown
- [ ] **P1** Implement agent list filtering by type (regulatory, calculator, analyzer)
  - Success Criteria: Type filter dropdown
- [ ] **P1** Implement agent list search by name/description
  - Success Criteria: Search input works
- [ ] **P1** Implement agent list pagination
  - Success Criteria: Pagination controls
- [ ] **P1** Create agent status badge component
  - Success Criteria: Status badges with colors
- [ ] **P1** Implement agent quick actions menu (view, edit, deploy, delete)
  - Success Criteria: Row actions dropdown
- [ ] **P1** Create agent bulk actions toolbar
  - Success Criteria: Bulk action buttons
- [ ] **P1** Implement agent bulk selection (select all, select some)
  - Success Criteria: Checkbox selection
- [ ] **P1** Implement agent bulk delete action
  - Success Criteria: Delete multiple agents
- [ ] **P1** Implement agent bulk deploy action
  - Success Criteria: Deploy multiple agents
- [ ] **P1** Implement agent bulk export action
  - Success Criteria: Export selected agents
- [ ] **P1** Create agent list empty state
  - Success Criteria: Empty state with CTA
- [ ] **P1** Implement agent list loading skeleton
  - Success Criteria: Loading placeholder
- [ ] **P1** Implement agent list error state
  - Success Criteria: Error with retry
- [ ] **P1** Add agent list keyboard shortcuts (n: new, /: search)
  - Success Criteria: Keyboard navigation
- [ ] **P1** Write AgentList unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create AgentList Storybook stories
  - Success Criteria: Documentation complete

#### 2.2.2 Agent Detail View (20 tasks)

- [ ] **P1** Create AgentDetailPage component
  - Success Criteria: Agent detail page route
- [ ] **P1** Implement agent header with name, version, status
  - Success Criteria: Agent overview header
- [ ] **P1** Implement agent description section
  - Success Criteria: Description displayed
- [ ] **P1** Create agent specification YAML viewer (syntax highlighted)
  - Success Criteria: YAML code view
- [ ] **P1** Implement agent tools list display
  - Success Criteria: Tools with descriptions
- [ ] **P1** Implement agent inputs schema display
  - Success Criteria: Input schema table
- [ ] **P1** Implement agent outputs schema display
  - Success Criteria: Output schema table
- [ ] **P1** Create agent version history timeline
  - Success Criteria: Version history list
- [ ] **P1** Implement agent deployment history tab
  - Success Criteria: Deployment log
- [ ] **P1** Create agent metrics summary cards (executions, latency, errors)
  - Success Criteria: KPI cards
- [ ] **P1** Implement agent recent executions table
  - Success Criteria: Execution history
- [ ] **P1** Create agent error log viewer
  - Success Criteria: Error log display
- [ ] **P1** Implement agent configuration editor
  - Success Criteria: Config form
- [ ] **P1** Create agent test runner panel
  - Success Criteria: Run agent with test inputs
- [ ] **P1** Implement agent documentation tab (rendered markdown)
  - Success Criteria: README display
- [ ] **P1** Add agent detail page tabs navigation
  - Success Criteria: Tab navigation
- [ ] **P1** Implement agent detail loading state
  - Success Criteria: Loading skeleton
- [ ] **P1** Implement agent detail error state
  - Success Criteria: Error handling
- [ ] **P1** Write AgentDetail unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create AgentDetail Storybook stories
  - Success Criteria: Documentation complete

#### 2.2.3 Agent Creation/Editing (20 tasks)

- [ ] **P1** Create AgentCreatePage component
  - Success Criteria: Create agent page route
- [ ] **P1** Implement agent creation wizard (multi-step form)
  - Success Criteria: Wizard UI
- [ ] **P1** Create Step 1: Basic Info form (name, description, type, category)
  - Success Criteria: Basic info step
- [ ] **P1** Create Step 2: Inputs configuration (schema builder)
  - Success Criteria: Input schema step
- [ ] **P1** Create Step 3: Tools selection (available tools list)
  - Success Criteria: Tool selection step
- [ ] **P1** Create Step 4: Outputs configuration (schema builder)
  - Success Criteria: Output schema step
- [ ] **P1** Create Step 5: Review and submit
  - Success Criteria: Review step
- [ ] **P1** Implement agent YAML editor (Monaco Editor)
  - Success Criteria: Code editor integration
- [ ] **P1** Implement YAML syntax validation
  - Success Criteria: Syntax error highlighting
- [ ] **P1** Implement YAML schema validation (against AgentSpec)
  - Success Criteria: Schema error messages
- [ ] **P1** Create AgentEditPage component
  - Success Criteria: Edit agent page route
- [ ] **P1** Implement version comparison view (diff)
  - Success Criteria: Side-by-side diff
- [ ] **P1** Implement save as draft functionality
  - Success Criteria: Draft save works
- [ ] **P1** Implement publish workflow (draft -> published)
  - Success Criteria: Publish action works
- [ ] **P1** Create agent deletion confirmation modal
  - Success Criteria: Delete confirmation
- [ ] **P1** Implement agent clone functionality
  - Success Criteria: Clone agent works
- [ ] **P1** Add unsaved changes warning (beforeunload)
  - Success Criteria: Prevent accidental leave
- [ ] **P1** Implement form validation with error messages
  - Success Criteria: Validation works
- [ ] **P1** Write AgentCreate/Edit unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create AgentCreate/Edit Storybook stories
  - Success Criteria: Documentation complete

### 2.3 User Management UI (40 tasks)

#### 2.3.1 User List View (16 tasks)

- [ ] **P1** Create UserListPage component
  - Success Criteria: User list page route
- [ ] **P1** Implement user list table (name, email, role, status, last active)
  - Success Criteria: Table displays users
- [ ] **P1** Implement user list sorting
  - Success Criteria: Sortable columns
- [ ] **P1** Implement user list filtering by role
  - Success Criteria: Role filter
- [ ] **P1** Implement user list filtering by status (active, pending, disabled)
  - Success Criteria: Status filter
- [ ] **P1** Implement user list search
  - Success Criteria: Search by name/email
- [ ] **P1** Implement user list pagination
  - Success Criteria: Pagination controls
- [ ] **P1** Create user status badge component
  - Success Criteria: Status badges
- [ ] **P1** Create user role badge component
  - Success Criteria: Role badges
- [ ] **P1** Implement user quick actions menu
  - Success Criteria: Row actions
- [ ] **P1** Implement user bulk actions
  - Success Criteria: Bulk operations
- [ ] **P1** Create user list empty state
  - Success Criteria: Empty state
- [ ] **P1** Implement user invitation button
  - Success Criteria: Invite user CTA
- [ ] **P1** Implement user export (CSV)
  - Success Criteria: Export functionality
- [ ] **P1** Write UserList unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create UserList Storybook stories
  - Success Criteria: Documentation complete

#### 2.3.2 User Detail & Management Forms (24 tasks)

- [ ] **P1** Create UserDetailPage component
  - Success Criteria: User detail page route
- [ ] **P1** Implement user profile header (avatar, name, role, status)
  - Success Criteria: Profile header
- [ ] **P1** Implement user activity log tab
  - Success Criteria: Activity timeline
- [ ] **P1** Implement user permissions display
  - Success Criteria: Permissions list
- [ ] **P1** Implement user session history tab
  - Success Criteria: Session log
- [ ] **P1** Create user audit trail tab
  - Success Criteria: Audit events
- [ ] **P1** Create UserInviteModal component
  - Success Criteria: Invite modal
- [ ] **P1** Implement invite form (email, role selection)
  - Success Criteria: Invite form
- [ ] **P1** Implement bulk invite (CSV upload)
  - Success Criteria: Bulk invite
- [ ] **P1** Create UserEditModal component
  - Success Criteria: Edit modal
- [ ] **P1** Implement role assignment form
  - Success Criteria: Role dropdown
- [ ] **P1** Implement permission overrides form
  - Success Criteria: Custom permissions
- [ ] **P1** Create UserDeactivateModal component
  - Success Criteria: Deactivate confirmation
- [ ] **P1** Create UserReactivateModal component
  - Success Criteria: Reactivate confirmation
- [ ] **P1** Implement password reset trigger button
  - Success Criteria: Reset password
- [ ] **P1** Create RoleListPage component
  - Success Criteria: Roles management page
- [ ] **P1** Implement role creation form
  - Success Criteria: Create role
- [ ] **P1** Implement permission matrix editor
  - Success Criteria: Permission checkboxes
- [ ] **P1** Create role assignment interface
  - Success Criteria: Assign role to users
- [ ] **P1** Implement role hierarchy display
  - Success Criteria: Role inheritance view
- [ ] **P1** Create custom role builder
  - Success Criteria: Custom role creation
- [ ] **P1** Implement permission inheritance visualization
  - Success Criteria: Inheritance diagram
- [ ] **P1** Write User Management unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create User Management Storybook stories
  - Success Criteria: Documentation complete

### 2.4 Tenant Management UI (28 tasks)

- [ ] **P1** Create TenantListPage component
  - Success Criteria: Tenant list page
- [ ] **P1** Implement tenant list table (name, status, users, usage)
  - Success Criteria: Tenant table
- [ ] **P1** Implement tenant filtering and search
  - Success Criteria: Filter/search
- [ ] **P1** Create TenantDetailPage component
  - Success Criteria: Tenant detail page
- [ ] **P1** Implement tenant profile section
  - Success Criteria: Tenant info display
- [ ] **P1** Implement tenant usage metrics display
  - Success Criteria: Usage charts
- [ ] **P1** Implement tenant users tab
  - Success Criteria: Users in tenant
- [ ] **P1** Implement tenant agents tab
  - Success Criteria: Agents in tenant
- [ ] **P1** Create TenantCreateModal component
  - Success Criteria: Create tenant modal
- [ ] **P1** Implement tenant creation form (name, settings, quotas)
  - Success Criteria: Create form
- [ ] **P1** Create TenantEditModal component
  - Success Criteria: Edit tenant modal
- [ ] **P1** Implement tenant branding settings (logo, colors)
  - Success Criteria: Branding config
- [ ] **P1** Implement tenant domain configuration
  - Success Criteria: Custom domain
- [ ] **P1** Implement tenant resource quotas form
  - Success Criteria: Quota limits
- [ ] **P1** Implement tenant billing settings
  - Success Criteria: Billing config
- [ ] **P1** Create tenant SSO configuration section
  - Success Criteria: SSO setup
- [ ] **P1** Implement tenant audit settings
  - Success Criteria: Audit config
- [ ] **P1** Create tenant suspension modal
  - Success Criteria: Suspend tenant
- [ ] **P1** Create tenant deletion modal (with data handling)
  - Success Criteria: Delete tenant
- [ ] **P1** Implement tenant data export
  - Success Criteria: Export tenant data
- [ ] **P1** Create tenant onboarding checklist
  - Success Criteria: Setup progress
- [ ] **P1** Implement tenant switching for super admins
  - Success Criteria: Switch tenant context
- [ ] **P1** Create tenant comparison view
  - Success Criteria: Compare tenants
- [ ] **P1** Implement tenant health dashboard
  - Success Criteria: Tenant health metrics
- [ ] **P1** Create tenant activity feed
  - Success Criteria: Activity timeline
- [ ] **P1** Implement tenant notification settings
  - Success Criteria: Notification config
- [ ] **P1** Write Tenant Management unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Tenant Management Storybook stories
  - Success Criteria: Documentation complete

### 2.5 System Configuration UI (20 tasks)

- [ ] **P1** Create SystemConfigPage component
  - Success Criteria: Config page route
- [ ] **P1** Implement API keys management section
  - Success Criteria: API key list
- [ ] **P1** Implement API key creation form (name, permissions, expiry)
  - Success Criteria: Create API key
- [ ] **P1** Implement API key revocation with confirmation
  - Success Criteria: Revoke API key
- [ ] **P1** Implement API key usage display (requests, last used)
  - Success Criteria: Key usage stats
- [ ] **P1** Create environment variables editor
  - Success Criteria: Env var config
- [ ] **P1** Implement secrets management UI (masked values)
  - Success Criteria: Secrets config
- [ ] **P1** Create feature flags toggle UI
  - Success Criteria: Feature toggles
- [ ] **P1** Implement rate limiting configuration
  - Success Criteria: Rate limit config
- [ ] **P1** Create webhook configuration UI
  - Success Criteria: Webhook setup
- [ ] **P1** Implement webhook test button
  - Success Criteria: Test webhook
- [ ] **P1** Create email (SMTP) integration setup
  - Success Criteria: SMTP config
- [ ] **P1** Create S3/storage integration setup
  - Success Criteria: Storage config
- [ ] **P1** Implement integration health status display
  - Success Criteria: Health indicators
- [ ] **P1** Create integration test connection button
  - Success Criteria: Test connection
- [ ] **P1** Create notification settings page
  - Success Criteria: Notification config
- [ ] **P1** Implement email notification preferences
  - Success Criteria: Email preferences
- [ ] **P1** Create notification template editor
  - Success Criteria: Template editing
- [ ] **P1** Write System Configuration unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create System Configuration Storybook stories
  - Success Criteria: Documentation complete

### 2.6 Monitoring Dashboard (20 tasks)

- [ ] **P1** Create MonitoringDashboardPage component
  - Success Criteria: Monitoring page route
- [ ] **P1** Implement system health summary cards (uptime, latency, errors)
  - Success Criteria: Health KPIs
- [ ] **P1** Create active agents count widget
  - Success Criteria: Agent count card
- [ ] **P1** Create total executions today widget
  - Success Criteria: Execution count
- [ ] **P1** Create average latency widget (P50, P95, P99)
  - Success Criteria: Latency stats
- [ ] **P1** Create error rate widget
  - Success Criteria: Error percentage
- [ ] **P1** Implement real-time execution counter (WebSocket)
  - Success Criteria: Live updates
- [ ] **P1** Create system status indicator (operational/degraded/down)
  - Success Criteria: Status banner
- [ ] **P1** Create ExecutionMetricsChart component (line chart)
  - Success Criteria: Executions over time
- [ ] **P1** Create LatencyDistributionChart (histogram)
  - Success Criteria: Latency histogram
- [ ] **P1** Create ErrorRateTrendChart (line chart)
  - Success Criteria: Error trend
- [ ] **P1** Create TokenUsageChart (area chart)
  - Success Criteria: Token usage over time
- [ ] **P1** Create CostTrackingChart (bar chart)
  - Success Criteria: Cost by agent
- [ ] **P1** Create AlertsListWidget (active alerts)
  - Success Criteria: Alert list
- [ ] **P1** Implement alert acknowledgment action
  - Success Criteria: Acknowledge alert
- [ ] **P1** Create infrastructure status panel (K8s pods, DB, Redis)
  - Success Criteria: Infra status
- [ ] **P1** Implement time range selector (1h, 6h, 24h, 7d, 30d)
  - Success Criteria: Time range filter
- [ ] **P1** Implement auto-refresh toggle (30s, 1m, 5m)
  - Success Criteria: Auto refresh
- [ ] **P1** Write Monitoring Dashboard unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Monitoring Dashboard Storybook stories
  - Success Criteria: Documentation complete

---

## Phase 3: User Portal (242 tasks)

### 3.1 Authentication Flows (38 tasks)

#### 3.1.1 Login Flow (14 tasks)

- [ ] **P1** Create LoginPage component
  - Success Criteria: Login page route
- [ ] **P1** Implement email/password login form
  - Success Criteria: Login form
- [ ] **P1** Implement login form validation
  - Success Criteria: Validation messages
- [ ] **P1** Implement login error handling (invalid credentials, locked account)
  - Success Criteria: Error messages
- [ ] **P1** Create "Remember me" checkbox
  - Success Criteria: Persistent session option
- [ ] **P1** Create "Forgot password" link
  - Success Criteria: Link to reset
- [ ] **P1** Implement login rate limiting feedback
  - Success Criteria: Rate limit message
- [ ] **P1** Create login loading state (button spinner)
  - Success Criteria: Loading indicator
- [ ] **P1** Implement SSO login buttons (Google, Microsoft, Okta)
  - Success Criteria: SSO buttons
- [ ] **P1** Implement SSO redirect handling
  - Success Criteria: OAuth callback handling
- [ ] **P1** Create MFA verification step
  - Success Criteria: MFA input screen
- [ ] **P1** Implement MFA code input (6 digits)
  - Success Criteria: Code input
- [ ] **P1** Implement MFA backup codes option
  - Success Criteria: Backup code input
- [ ] **P1** Implement redirect after login (return URL)
  - Success Criteria: Post-login redirect

#### 3.1.2 Signup & Password Management (16 tasks)

- [ ] **P1** Create SignupPage component
  - Success Criteria: Signup page route
- [ ] **P1** Implement signup form (email, password, name, company)
  - Success Criteria: Signup form
- [ ] **P1** Implement password strength indicator
  - Success Criteria: Strength meter
- [ ] **P1** Implement password requirements display
  - Success Criteria: Requirements checklist
- [ ] **P1** Implement terms & conditions checkbox
  - Success Criteria: T&C agreement
- [ ] **P1** Create email verification step
  - Success Criteria: Verify email screen
- [ ] **P1** Implement verification code input
  - Success Criteria: Code input
- [ ] **P1** Create organization setup step
  - Success Criteria: Org creation
- [ ] **P1** Implement invite code acceptance
  - Success Criteria: Join via invite
- [ ] **P1** Create ForgotPasswordPage component
  - Success Criteria: Forgot password page
- [ ] **P1** Implement password reset email request
  - Success Criteria: Request reset
- [ ] **P1** Create ResetPasswordPage component
  - Success Criteria: Reset password page
- [ ] **P1** Implement password reset form
  - Success Criteria: New password form
- [ ] **P1** Implement password reset success message
  - Success Criteria: Success confirmation
- [ ] **P1** Create ChangePasswordPage component (authenticated)
  - Success Criteria: Change password
- [ ] **P1** Implement current password verification
  - Success Criteria: Verify current password

#### 3.1.3 Session Management (8 tasks)

- [ ] **P1** Implement auth state persistence (JWT in httpOnly cookies)
  - Success Criteria: Secure token storage
- [ ] **P1** Implement token refresh mechanism
  - Success Criteria: Auto refresh tokens
- [ ] **P1** Create session expiry warning modal
  - Success Criteria: Expiry warning
- [ ] **P1** Implement session timeout redirect
  - Success Criteria: Redirect to login
- [ ] **P1** Create active sessions list (in profile)
  - Success Criteria: Session list
- [ ] **P1** Implement session revocation (logout other devices)
  - Success Criteria: Revoke sessions
- [ ] **P1** Write Authentication unit tests
  - Success Criteria: Auth tests complete
- [ ] **P1** Create Authentication Storybook stories
  - Success Criteria: Documentation complete

### 3.2 Dashboard Home (32 tasks)

#### 3.2.1 Quick Actions & Widgets (16 tasks)

- [ ] **P1** Create UserDashboardPage component
  - Success Criteria: Dashboard home page
- [ ] **P1** Implement welcome message with user name
  - Success Criteria: Personalized greeting
- [ ] **P1** Create quick action cards grid
  - Success Criteria: Action buttons grid
- [ ] **P1** Implement "New Calculation" quick action
  - Success Criteria: Opens calculation UI
- [ ] **P1** Implement "View Reports" quick action
  - Success Criteria: Opens reports
- [ ] **P1** Implement "Upload Data" quick action
  - Success Criteria: Opens upload
- [ ] **P1** Implement "Browse Agents" quick action
  - Success Criteria: Opens agent catalog
- [ ] **P1** Create emissions summary card
  - Success Criteria: Total emissions KPI
- [ ] **P1** Implement emissions trend mini-chart (sparkline)
  - Success Criteria: Trend indicator
- [ ] **P1** Create compliance status summary card
  - Success Criteria: Compliance KPI
- [ ] **P1** Implement compliance checklist progress
  - Success Criteria: Progress bar
- [ ] **P1** Create recent activity feed widget
  - Success Criteria: Activity list
- [ ] **P1** Implement activity item components
  - Success Criteria: Activity cards
- [ ] **P1** Create upcoming deadlines widget
  - Success Criteria: Deadline list
- [ ] **P1** Implement deadline countdown display
  - Success Criteria: Days until deadline
- [ ] **P1** Create announcements banner
  - Success Criteria: System announcements

#### 3.2.2 Data Overview & Personalization (16 tasks)

- [ ] **P1** Create scope breakdown donut chart
  - Success Criteria: Scope 1/2/3 pie
- [ ] **P1** Implement scope 1/2/3 comparison
  - Success Criteria: Scope breakdown
- [ ] **P1** Create top emission sources list
  - Success Criteria: Top sources
- [ ] **P1** Create data quality score display
  - Success Criteria: Quality gauge
- [ ] **P1** Implement data coverage percentage
  - Success Criteria: Coverage metric
- [ ] **P1** Create missing data alerts widget
  - Success Criteria: Data gap alerts
- [ ] **P1** Implement data freshness indicator
  - Success Criteria: Last updated info
- [ ] **P1** Create reduction progress widget (vs target)
  - Success Criteria: Target progress
- [ ] **P1** Implement dashboard widget reordering (drag-drop)
  - Success Criteria: Rearrange widgets
- [ ] **P1** Implement dashboard widget visibility toggle
  - Success Criteria: Show/hide widgets
- [ ] **P1** Save dashboard layout to user preferences
  - Success Criteria: Persist layout
- [ ] **P1** Create dashboard layout presets (compact, detailed, executive)
  - Success Criteria: Layout presets
- [ ] **P1** Implement dashboard export (PDF snapshot)
  - Success Criteria: Export dashboard
- [ ] **P1** Create dashboard refresh button
  - Success Criteria: Manual refresh
- [ ] **P1** Write Dashboard Home unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Dashboard Home Storybook stories
  - Success Criteria: Documentation complete

### 3.3 Agent Interaction UI (72 tasks)

#### 3.3.1 Agent Catalog (12 tasks)

- [ ] **P1** Create AgentCatalogPage component
  - Success Criteria: Catalog page route
- [ ] **P1** Implement agent card grid layout
  - Success Criteria: Agent cards grid
- [ ] **P1** Implement agent category filters
  - Success Criteria: Category filter
- [ ] **P1** Implement agent search (name, description)
  - Success Criteria: Search input
- [ ] **P1** Create agent detail preview modal
  - Success Criteria: Quick preview
- [ ] **P1** Implement agent favoriting (add to favorites)
  - Success Criteria: Favorite toggle
- [ ] **P1** Create "My Favorites" filter
  - Success Criteria: Favorites filter
- [ ] **P1** Implement agent usage count display
  - Success Criteria: Popularity indicator
- [ ] **P1** Create "Recently Used" section
  - Success Criteria: Recent agents
- [ ] **P1** Implement agent category icons
  - Success Criteria: Category visuals
- [ ] **P1** Write AgentCatalog unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create AgentCatalog Storybook stories
  - Success Criteria: Documentation complete

#### 3.3.2 Fuel Emissions Analyzer UI (10 tasks)

- [ ] **P1** Create FuelEmissionsPage component
  - Success Criteria: Fuel analyzer page
- [ ] **P1** Implement fuel type selector (dropdown with categories)
  - Success Criteria: Fuel type selection
- [ ] **P1** Implement quantity input with unit selector (liters, gallons, kg)
  - Success Criteria: Quantity input
- [ ] **P1** Implement region selector (for region-specific factors)
  - Success Criteria: Region dropdown
- [ ] **P1** Implement reporting period selector (month/year)
  - Success Criteria: Period selection
- [ ] **P1** Create calculation result display card
  - Success Criteria: Result display
- [ ] **P1** Implement result breakdown by GHG (CO2, CH4, N2O)
  - Success Criteria: GHG breakdown
- [ ] **P1** Implement emission factor source display
  - Success Criteria: Factor provenance
- [ ] **P1** Create calculation history table
  - Success Criteria: History list
- [ ] **P1** Implement result comparison (side-by-side)
  - Success Criteria: Compare results

#### 3.3.3 CBAM Calculator UI (12 tasks)

- [ ] **P1** Create CBAMCalculatorPage component
  - Success Criteria: CBAM page route
- [ ] **P1** Implement product category selector (11 CBAM categories)
  - Success Criteria: Category dropdown
- [ ] **P1** Implement import data form (weight, origin country, dates)
  - Success Criteria: Import form
- [ ] **P1** Implement production method selector (BAT, standard)
  - Success Criteria: Method selection
- [ ] **P1** Implement precursor inputs form (steel, aluminium, etc.)
  - Success Criteria: Precursor inputs
- [ ] **P1** Create carbon intensity result display
  - Success Criteria: Intensity result
- [ ] **P1** Implement CBAM certificate estimate calculation
  - Success Criteria: Certificate estimate
- [ ] **P1** Create data quality score display
  - Success Criteria: Quality indicator
- [ ] **P1** Implement CBAM report generator
  - Success Criteria: Generate report
- [ ] **P1** Create supplementary documentation uploader
  - Success Criteria: Document upload
- [ ] **P1** Implement supplier data request workflow
  - Success Criteria: Request supplier data
- [ ] **P1** Create CBAM submission tracker
  - Success Criteria: Submission status

#### 3.3.4 Building Energy Performance UI (10 tasks)

- [ ] **P2** Create BuildingEnergyPage component
  - Success Criteria: Building page route
- [ ] **P2** Implement building profile form (type, size, location, year)
  - Success Criteria: Building form
- [ ] **P2** Implement energy consumption data entry (electricity, gas, steam)
  - Success Criteria: Energy inputs
- [ ] **P2** Implement renewable energy inputs
  - Success Criteria: Renewable tracking
- [ ] **P2** Create energy use intensity (EUI) display
  - Success Criteria: EUI calculation
- [ ] **P2** Implement benchmark comparison (ENERGY STAR, ASHRAE)
  - Success Criteria: Benchmark display
- [ ] **P2** Implement NYC LL97 compliance checker
  - Success Criteria: LL97 assessment
- [ ] **P2** Create improvement recommendations section
  - Success Criteria: Recommendations
- [ ] **P2** Implement energy certification display
  - Success Criteria: Certification info
- [ ] **P2** Create building comparison tool
  - Success Criteria: Compare buildings

#### 3.3.5 EUDR Compliance UI (12 tasks)

- [ ] **P1** Create EUDRCompliancePage component
  - Success Criteria: EUDR page route
- [ ] **P1** Implement commodity type selector (palm, soy, beef, coffee, cocoa, rubber, wood)
  - Success Criteria: Commodity selection
- [ ] **P1** Implement supplier information form
  - Success Criteria: Supplier form
- [ ] **P1** Implement geolocation data uploader (shapefile, GeoJSON)
  - Success Criteria: Geo data upload
- [ ] **P1** Create interactive map for plot visualization (Mapbox/Leaflet)
  - Success Criteria: Map display
- [ ] **P1** Implement supply chain traceability display
  - Success Criteria: Supply chain view
- [ ] **P1** Create risk assessment result display
  - Success Criteria: Risk results
- [ ] **P1** Implement due diligence statement generator
  - Success Criteria: Generate statement
- [ ] **P1** Create document upload section
  - Success Criteria: Document upload
- [ ] **P1** Implement compliance status tracker
  - Success Criteria: Status tracking
- [ ] **P1** Create EUDR submission workflow
  - Success Criteria: Submit workflow
- [ ] **P1** Implement supplier notification system
  - Success Criteria: Notify suppliers

#### 3.3.6 Additional Agent UIs (16 tasks)

- [ ] **P2** Create Scope3Page component
  - Success Criteria: Scope 3 page
- [ ] **P2** Implement category selection interface (15 categories)
  - Success Criteria: Category tabs
- [ ] **P2** Create spend-based data entry form
  - Success Criteria: Spend input
- [ ] **P2** Create activity-based data entry form
  - Success Criteria: Activity input
- [ ] **P2** Implement supplier data collection requests
  - Success Criteria: Request supplier data
- [ ] **P2** Create emissions breakdown by category chart
  - Success Criteria: Category breakdown
- [ ] **P2** Implement data quality indicators per category
  - Success Criteria: Quality by category
- [ ] **P2** Create CSRDReportingPage component
  - Success Criteria: CSRD page
- [ ] **P2** Implement materiality assessment wizard
  - Success Criteria: Materiality wizard
- [ ] **P2** Create ESRS data collection forms
  - Success Criteria: ESRS forms
- [ ] **P2** Implement gap analysis display
  - Success Criteria: Gap analysis
- [ ] **P2** Create report preview
  - Success Criteria: Report preview
- [ ] **P2** Implement iXBRL export
  - Success Criteria: XBRL export
- [ ] **P2** Create GenericAgentPage component (dynamic form generation)
  - Success Criteria: Generic agent UI
- [ ] **P2** Implement dynamic form generation from agent schema
  - Success Criteria: Schema-driven forms
- [ ] **P2** Write Agent Interaction unit tests
  - Success Criteria: Tests complete

### 3.4 Results Visualization (52 tasks)

#### 3.4.1 Emissions Visualization (16 tasks)

- [ ] **P1** Create EmissionsVisualizationPage component
  - Success Criteria: Visualization page
- [ ] **P1** Implement total emissions summary card
  - Success Criteria: Total KPI
- [ ] **P1** Implement emissions trend line chart
  - Success Criteria: Trend chart
- [ ] **P1** Implement emissions comparison bar chart (YoY, QoQ)
  - Success Criteria: Comparison chart
- [ ] **P1** Create scope breakdown pie chart
  - Success Criteria: Scope pie
- [ ] **P1** Create scope breakdown treemap
  - Success Criteria: Scope treemap
- [ ] **P1** Implement emissions by location map (choropleth)
  - Success Criteria: Location map
- [ ] **P1** Create emissions by source category chart
  - Success Criteria: Source breakdown
- [ ] **P1** Implement emissions intensity metrics (per revenue, per employee)
  - Success Criteria: Intensity metrics
- [ ] **P1** Create emission factor sources table
  - Success Criteria: Factor sources
- [ ] **P1** Implement chart drill-down (click to explore)
  - Success Criteria: Drill-down navigation
- [ ] **P1** Implement chart time range selector
  - Success Criteria: Date range filter
- [ ] **P1** Create chart export functionality (PNG, SVG, CSV)
  - Success Criteria: Export charts
- [ ] **P1** Implement chart full-screen mode
  - Success Criteria: Full-screen view
- [ ] **P1** Write Emissions Visualization unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Emissions Visualization Storybook stories
  - Success Criteria: Documentation complete

#### 3.4.2 Compliance Visualization (16 tasks)

- [ ] **P1** Create ComplianceDashboardPage component
  - Success Criteria: Compliance page
- [ ] **P1** Implement compliance scorecard
  - Success Criteria: Overall score
- [ ] **P1** Create compliance by regulation breakdown
  - Success Criteria: Regulation cards
- [ ] **P1** Implement compliance gap analysis chart
  - Success Criteria: Gap analysis
- [ ] **P1** Create compliance timeline (deadlines calendar)
  - Success Criteria: Deadline calendar
- [ ] **P1** Implement regulatory requirement checklist
  - Success Criteria: Requirement checklist
- [ ] **P1** Create non-compliance alerts display
  - Success Criteria: Alert list
- [ ] **P1** Implement remediation action tracker
  - Success Criteria: Action items
- [ ] **P1** Create CBAM compliance dashboard
  - Success Criteria: CBAM dashboard
- [ ] **P1** Create CSRD compliance dashboard
  - Success Criteria: CSRD dashboard
- [ ] **P1** Create EUDR compliance dashboard
  - Success Criteria: EUDR dashboard
- [ ] **P1** Create SBTi progress dashboard
  - Success Criteria: SBTi dashboard
- [ ] **P1** Implement target progress gauge
  - Success Criteria: Target gauge
- [ ] **P1** Create pathway alignment chart
  - Success Criteria: Pathway chart
- [ ] **P1** Write Compliance Visualization unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Compliance Visualization Storybook stories
  - Success Criteria: Documentation complete

#### 3.4.3 Advanced Visualization (20 tasks)

- [ ] **P2** Create DataQualityPage component
  - Success Criteria: Data quality page
- [ ] **P2** Implement overall data quality score gauge
  - Success Criteria: Quality gauge
- [ ] **P2** Create data quality by source chart
  - Success Criteria: Source quality
- [ ] **P2** Implement data coverage heatmap
  - Success Criteria: Coverage heatmap
- [ ] **P2** Create data freshness timeline
  - Success Criteria: Freshness chart
- [ ] **P2** Create ComparisonPage component
  - Success Criteria: Comparison page
- [ ] **P2** Implement period-over-period comparison
  - Success Criteria: Period comparison
- [ ] **P2** Implement facility comparison
  - Success Criteria: Facility comparison
- [ ] **P2** Implement benchmark comparison
  - Success Criteria: Benchmark comparison
- [ ] **P2** Create TrendAnalysisPage component
  - Success Criteria: Trend analysis page
- [ ] **P2** Implement moving average overlay
  - Success Criteria: Moving average
- [ ] **P2** Create forecasting projection chart
  - Success Criteria: Forecast chart
- [ ] **P2** Create InteractiveMapPage component
  - Success Criteria: Map page
- [ ] **P2** Implement facility location markers
  - Success Criteria: Facility markers
- [ ] **P2** Create emissions heatmap layer
  - Success Criteria: Emissions heatmap
- [ ] **P2** Create SankeyDiagram component
  - Success Criteria: Sankey chart
- [ ] **P2** Implement emissions flow visualization
  - Success Criteria: Flow visualization
- [ ] **P2** Implement chart zoom/pan controls
  - Success Criteria: Zoom/pan
- [ ] **P2** Write Advanced Visualization unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Advanced Visualization Storybook stories
  - Success Criteria: Documentation complete

### 3.5 Report Generation UI (28 tasks)

- [ ] **P1** Create ReportBuilderPage component
  - Success Criteria: Report builder page
- [ ] **P1** Implement report template selection
  - Success Criteria: Template selector
- [ ] **P1** Create CBAM report template
  - Success Criteria: CBAM template
- [ ] **P1** Create CSRD report template
  - Success Criteria: CSRD template
- [ ] **P1** Create CDP report template
  - Success Criteria: CDP template
- [ ] **P1** Create GRI report template
  - Success Criteria: GRI template
- [ ] **P1** Create custom report template
  - Success Criteria: Custom template
- [ ] **P1** Implement section selection checklist
  - Success Criteria: Section picker
- [ ] **P1** Implement date range selector for report data
  - Success Criteria: Date range
- [ ] **P1** Implement data scope selector (facilities, products)
  - Success Criteria: Scope selector
- [ ] **P1** Implement report title/cover page editor
  - Success Criteria: Cover editor
- [ ] **P1** Implement section ordering (drag-drop)
  - Success Criteria: Section reorder
- [ ] **P1** Implement chart inclusion selection
  - Success Criteria: Chart picker
- [ ] **P1** Implement commentary/notes editor
  - Success Criteria: Notes editor
- [ ] **P1** Create branding/logo uploader
  - Success Criteria: Logo upload
- [ ] **P1** Create ReportPreviewPage component
  - Success Criteria: Preview page
- [ ] **P1** Implement PDF preview renderer
  - Success Criteria: PDF preview
- [ ] **P1** Implement page navigation in preview
  - Success Criteria: Page nav
- [ ] **P1** Implement zoom controls in preview
  - Success Criteria: Zoom controls
- [ ] **P1** Implement PDF export (client-side generation)
  - Success Criteria: PDF export
- [ ] **P1** Implement Excel export
  - Success Criteria: Excel export
- [ ] **P1** Implement Word export
  - Success Criteria: Word export
- [ ] **P1** Implement JSON data export
  - Success Criteria: JSON export
- [ ] **P1** Implement email report delivery
  - Success Criteria: Email report
- [ ] **P1** Create scheduled report configuration
  - Success Criteria: Report scheduling
- [ ] **P1** Implement report sharing links
  - Success Criteria: Share links
- [ ] **P1** Write Report Generation unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Report Generation Storybook stories
  - Success Criteria: Documentation complete

### 3.6 Data Management UI (20 tasks)

- [ ] **P1** Create DataUploadPage component
  - Success Criteria: Upload page
- [ ] **P1** Implement file upload drag & drop zone
  - Success Criteria: Drop zone
- [ ] **P1** Implement CSV upload with parsing
  - Success Criteria: CSV upload
- [ ] **P1** Implement Excel upload with sheet selection
  - Success Criteria: Excel upload
- [ ] **P1** Create upload progress indicator
  - Success Criteria: Upload progress
- [ ] **P1** Implement file validation preview
  - Success Criteria: Validation preview
- [ ] **P1** Create column mapping interface
  - Success Criteria: Column mapping
- [ ] **P1** Implement data type detection
  - Success Criteria: Auto-detect types
- [ ] **P1** Create validation error display
  - Success Criteria: Error display
- [ ] **P1** Implement partial upload (skip errors) option
  - Success Criteria: Skip errors
- [ ] **P1** Create data upload templates download
  - Success Criteria: Template downloads
- [ ] **P1** Create DataManagementPage component
  - Success Criteria: Data management page
- [ ] **P1** Implement uploaded files list
  - Success Criteria: Files list
- [ ] **P1** Implement file metadata display
  - Success Criteria: File details
- [ ] **P1** Implement file delete functionality
  - Success Criteria: Delete files
- [ ] **P1** Create DataHistoryPage component
  - Success Criteria: History page
- [ ] **P1** Implement upload history list
  - Success Criteria: History list
- [ ] **P1** Create audit trail table
  - Success Criteria: Audit trail
- [ ] **P1** Write Data Management unit tests
  - Success Criteria: Tests complete
- [ ] **P1** Create Data Management Storybook stories
  - Success Criteria: Documentation complete

---

## Phase 4: Agent Builder UI (168 tasks)

### 4.1 Visual Agent Designer (52 tasks)

#### 4.1.1 Canvas Interface (20 tasks)

- [ ] **P2** Create AgentDesignerPage component
  - Success Criteria: Designer page route
- [ ] **P2** Implement drag & drop canvas (React Flow)
  - Success Criteria: Canvas with drag-drop
- [ ] **P2** Create node palette sidebar (draggable nodes)
  - Success Criteria: Node palette
- [ ] **P2** Implement canvas zoom controls (+/- buttons)
  - Success Criteria: Zoom buttons
- [ ] **P2** Implement canvas pan controls (drag to pan)
  - Success Criteria: Pan gesture
- [ ] **P2** Implement canvas grid background
  - Success Criteria: Grid lines
- [ ] **P2** Implement minimap navigator
  - Success Criteria: Minimap overlay
- [ ] **P2** Create undo/redo functionality
  - Success Criteria: Undo/redo buttons
- [ ] **P2** Implement auto-layout algorithm
  - Success Criteria: Auto arrange nodes
- [ ] **P2** Create canvas background themes (light, dark, blueprint)
  - Success Criteria: Theme options
- [ ] **P2** Implement canvas save state
  - Success Criteria: Save canvas
- [ ] **P2** Implement canvas load state
  - Success Criteria: Load canvas
- [ ] **P2** Create canvas keyboard shortcuts (delete, copy, paste)
  - Success Criteria: Keyboard shortcuts
- [ ] **P2** Implement node selection (single, multi with Shift)
  - Success Criteria: Node selection
- [ ] **P2** Implement node alignment tools
  - Success Criteria: Align nodes
- [ ] **P2** Create canvas export as image
  - Success Criteria: Export PNG/SVG
- [ ] **P2** Implement canvas validation overlay (show errors)
  - Success Criteria: Error indicators
- [ ] **P2** Create canvas help overlay (hints)
  - Success Criteria: Help hints
- [ ] **P2** Write Canvas unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Canvas Storybook stories
  - Success Criteria: Documentation complete

#### 4.1.2 Node Types (16 tasks)

- [ ] **P2** Create Input node component (agent inputs)
  - Success Criteria: Input node
- [ ] **P2** Create Output node component (agent outputs)
  - Success Criteria: Output node
- [ ] **P2** Create Tool node component (tool invocation)
  - Success Criteria: Tool node
- [ ] **P2** Create Condition node component (if/else branching)
  - Success Criteria: Condition node
- [ ] **P2** Create Loop node component (iteration)
  - Success Criteria: Loop node
- [ ] **P2** Create Parallel node component (concurrent execution)
  - Success Criteria: Parallel node
- [ ] **P2** Create Comment/Note node (documentation)
  - Success Criteria: Comment node
- [ ] **P2** Create Group/Container node (grouping)
  - Success Criteria: Group node
- [ ] **P2** Implement node styling (colors by type, icons)
  - Success Criteria: Node styles
- [ ] **P2** Implement node resize handles
  - Success Criteria: Resize nodes
- [ ] **P2** Implement node minimize/expand
  - Success Criteria: Collapse nodes
- [ ] **P2** Create node connection ports (input/output)
  - Success Criteria: Node ports
- [ ] **P2** Implement port type indicators (data types)
  - Success Criteria: Type indicators
- [ ] **P2** Create node status indicators (valid, error, warning)
  - Success Criteria: Status icons
- [ ] **P2** Write Node unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Node Storybook stories
  - Success Criteria: Documentation complete

#### 4.1.3 Node Editing & Edges (16 tasks)

- [ ] **P2** Create node properties panel (sidebar)
  - Success Criteria: Properties panel
- [ ] **P2** Implement node name editing
  - Success Criteria: Edit name
- [ ] **P2** Implement node description editing
  - Success Criteria: Edit description
- [ ] **P2** Implement node configuration form (dynamic)
  - Success Criteria: Config form
- [ ] **P2** Create input/output port editing
  - Success Criteria: Port editor
- [ ] **P2** Implement port data type selection
  - Success Criteria: Type dropdown
- [ ] **P2** Implement port required/optional toggle
  - Success Criteria: Required toggle
- [ ] **P2** Create node validation rules editor
  - Success Criteria: Validation rules
- [ ] **P2** Implement edge creation (drag between ports)
  - Success Criteria: Edge creation
- [ ] **P2** Implement edge deletion (click delete, backspace)
  - Success Criteria: Edge deletion
- [ ] **P2** Implement edge validation (type compatibility)
  - Success Criteria: Edge validation
- [ ] **P2** Create edge labels (data flow description)
  - Success Criteria: Edge labels
- [ ] **P2** Implement edge routing (straight, curved, stepped)
  - Success Criteria: Edge routing
- [ ] **P2** Create edge animation (data flow visualization)
  - Success Criteria: Flow animation
- [ ] **P2** Write Edge unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Edge Storybook stories
  - Success Criteria: Documentation complete

### 4.2 Tool Configuration UI (36 tasks)

#### 4.2.1 Tool Library (16 tasks)

- [ ] **P2** Create ToolLibraryPage component
  - Success Criteria: Tool library page
- [ ] **P2** Implement tool list view (table)
  - Success Criteria: Tool list
- [ ] **P2** Implement tool card view (grid)
  - Success Criteria: Tool cards
- [ ] **P2** Implement tool search
  - Success Criteria: Search tools
- [ ] **P2** Implement tool category filters
  - Success Criteria: Category filter
- [ ] **P2** Create tool detail modal
  - Success Criteria: Tool details
- [ ] **P2** Implement tool favoriting
  - Success Criteria: Favorite tools
- [ ] **P2** Display tool usage statistics
  - Success Criteria: Usage stats
- [ ] **P2** Create EmissionFactorLookupTool config UI
  - Success Criteria: EF tool config
- [ ] **P2** Create CalculatorTool config UI
  - Success Criteria: Calculator config
- [ ] **P2** Create ValidatorTool config UI
  - Success Criteria: Validator config
- [ ] **P2** Create APIConnectorTool config UI
  - Success Criteria: API connector config
- [ ] **P2** Create DataTransformTool config UI
  - Success Criteria: Transform config
- [ ] **P2** Create ConditionalTool config UI
  - Success Criteria: Conditional config
- [ ] **P2** Write Tool Library unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Tool Library Storybook stories
  - Success Criteria: Documentation complete

#### 4.2.2 Custom Tool Creation (20 tasks)

- [ ] **P2** Create CustomToolBuilder component
  - Success Criteria: Tool builder page
- [ ] **P2** Implement tool name/description form
  - Success Criteria: Basic info form
- [ ] **P2** Implement input schema builder (add/remove fields)
  - Success Criteria: Input schema UI
- [ ] **P2** Implement output schema builder
  - Success Criteria: Output schema UI
- [ ] **P2** Create field type selector (string, number, boolean, object, array)
  - Success Criteria: Type selector
- [ ] **P2** Implement field validation rules editor
  - Success Criteria: Validation rules
- [ ] **P2** Create tool implementation code editor (Monaco)
  - Success Criteria: Code editor
- [ ] **P2** Implement code syntax highlighting (Python)
  - Success Criteria: Syntax highlighting
- [ ] **P2** Implement code auto-completion
  - Success Criteria: Auto-complete
- [ ] **P2** Create tool testing interface
  - Success Criteria: Test tool
- [ ] **P2** Implement test input form
  - Success Criteria: Test inputs
- [ ] **P2** Display test results
  - Success Criteria: Test output
- [ ] **P2** Create tool documentation generator
  - Success Criteria: Generate docs
- [ ] **P2** Implement tool versioning
  - Success Criteria: Version tools
- [ ] **P2** Create test case management
  - Success Criteria: Test cases
- [ ] **P2** Implement test case save/load
  - Success Criteria: Save test cases
- [ ] **P2** Create tool preview (simulate execution)
  - Success Criteria: Tool preview
- [ ] **P2** Implement tool publish workflow
  - Success Criteria: Publish tool
- [ ] **P2** Write Custom Tool unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Custom Tool Storybook stories
  - Success Criteria: Documentation complete

### 4.3 Workflow Builder (40 tasks)

#### 4.3.1 Workflow Design (20 tasks)

- [ ] **P2** Create WorkflowBuilderPage component
  - Success Criteria: Workflow builder page
- [ ] **P2** Implement workflow timeline view
  - Success Criteria: Timeline view
- [ ] **P2** Create step node component
  - Success Criteria: Step nodes
- [ ] **P2** Implement step sequencing (drag to reorder)
  - Success Criteria: Step order
- [ ] **P2** Create conditional branching UI (if/else)
  - Success Criteria: Branching UI
- [ ] **P2** Implement parallel execution paths
  - Success Criteria: Parallel paths
- [ ] **P2** Create loop/iteration configuration
  - Success Criteria: Loop config
- [ ] **P2** Implement error handling paths (try/catch)
  - Success Criteria: Error handling
- [ ] **P2** Create retry configuration (count, backoff)
  - Success Criteria: Retry config
- [ ] **P2** Create DataCollectionStep component
  - Success Criteria: Data step
- [ ] **P2** Create CalculationStep component
  - Success Criteria: Calculation step
- [ ] **P2** Create ValidationStep component
  - Success Criteria: Validation step
- [ ] **P2** Create TransformationStep component
  - Success Criteria: Transform step
- [ ] **P2** Create NotificationStep component
  - Success Criteria: Notification step
- [ ] **P2** Create ApprovalStep component (human-in-loop)
  - Success Criteria: Approval step
- [ ] **P2** Create ExportStep component
  - Success Criteria: Export step
- [ ] **P2** Create APICallStep component
  - Success Criteria: API step
- [ ] **P2** Write Workflow Design unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Workflow Design Storybook stories
  - Success Criteria: Documentation complete
- [ ] **P2** Create workflow templates library
  - Success Criteria: Templates

#### 4.3.2 Workflow Configuration (20 tasks)

- [ ] **P2** Implement step configuration forms
  - Success Criteria: Step forms
- [ ] **P2** Create variable/context editor
  - Success Criteria: Variable editor
- [ ] **P2** Implement input/output mapping between steps
  - Success Criteria: Data mapping
- [ ] **P2** Create expression builder (simple formulas)
  - Success Criteria: Expression builder
- [ ] **P2** Implement condition builder (AND, OR, comparisons)
  - Success Criteria: Condition builder
- [ ] **P2** Create schedule configuration (cron, interval)
  - Success Criteria: Schedule config
- [ ] **P2** Implement trigger configuration (event, webhook, manual)
  - Success Criteria: Trigger config
- [ ] **P2** Create workflow execution viewer (live view)
  - Success Criteria: Execution viewer
- [ ] **P2** Implement step-by-step progress display
  - Success Criteria: Step progress
- [ ] **P2** Create execution log viewer
  - Success Criteria: Log viewer
- [ ] **P2** Implement pause/resume controls
  - Success Criteria: Pause/resume
- [ ] **P2** Create execution history list
  - Success Criteria: History list
- [ ] **P2** Implement execution comparison (diff runs)
  - Success Criteria: Compare runs
- [ ] **P2** Create workflow export (YAML, JSON)
  - Success Criteria: Export workflow
- [ ] **P2** Implement workflow import
  - Success Criteria: Import workflow
- [ ] **P2** Create workflow validation (check for errors)
  - Success Criteria: Validate workflow
- [ ] **P2** Implement workflow duplication
  - Success Criteria: Clone workflow
- [ ] **P2** Create workflow versioning
  - Success Criteria: Version control
- [ ] **P2** Write Workflow Config unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Workflow Config Storybook stories
  - Success Criteria: Documentation complete

### 4.4 Testing Playground (24 tasks)

- [ ] **P2** Create TestPlaygroundPage component
  - Success Criteria: Playground page
- [ ] **P2** Implement agent selector dropdown
  - Success Criteria: Agent selector
- [ ] **P2** Create input form generator (from agent schema)
  - Success Criteria: Dynamic form
- [ ] **P2** Implement test data presets (saved inputs)
  - Success Criteria: Presets
- [ ] **P2** Create randomized test data generator
  - Success Criteria: Random data
- [ ] **P2** Implement test execution button
  - Success Criteria: Execute test
- [ ] **P2** Display execution progress (steps, timing)
  - Success Criteria: Progress display
- [ ] **P2** Create test result viewer
  - Success Criteria: Result viewer
- [ ] **P2** Implement JSON result display (formatted)
  - Success Criteria: JSON display
- [ ] **P2** Implement formatted result display (human-readable)
  - Success Criteria: Formatted view
- [ ] **P2** Create diff view (expected vs actual)
  - Success Criteria: Diff view
- [ ] **P2** Implement result validation (pass/fail)
  - Success Criteria: Validation
- [ ] **P2** Create error stack trace viewer
  - Success Criteria: Error display
- [ ] **P2** Create test case creator
  - Success Criteria: Create test case
- [ ] **P2** Implement test case save
  - Success Criteria: Save test case
- [ ] **P2** Implement test case load
  - Success Criteria: Load test case
- [ ] **P2** Create test suite builder (group tests)
  - Success Criteria: Test suites
- [ ] **P2** Implement batch test execution
  - Success Criteria: Run all tests
- [ ] **P2** Create test report generator
  - Success Criteria: Test report
- [ ] **P2** Implement test coverage display
  - Success Criteria: Coverage stats
- [ ] **P2** Create performance test runner
  - Success Criteria: Performance tests
- [ ] **P2** Implement latency measurement display
  - Success Criteria: Latency stats
- [ ] **P2** Write Testing Playground unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Testing Playground Storybook stories
  - Success Criteria: Documentation complete

### 4.5 Deployment UI (16 tasks)

- [ ] **P2** Create DeploymentConfigPage component
  - Success Criteria: Deployment page
- [ ] **P2** Implement environment selector (dev, staging, prod)
  - Success Criteria: Env selector
- [ ] **P2** Create resource allocation form (CPU, memory)
  - Success Criteria: Resource config
- [ ] **P2** Implement scaling configuration (min, max replicas)
  - Success Criteria: Scaling config
- [ ] **P2** Create environment variable editor
  - Success Criteria: Env vars
- [ ] **P2** Implement secrets configuration (masked values)
  - Success Criteria: Secrets config
- [ ] **P2** Create health check configuration
  - Success Criteria: Health checks
- [ ] **P2** Create DeploymentWizard component
  - Success Criteria: Deploy wizard
- [ ] **P2** Implement pre-deployment validation
  - Success Criteria: Validation check
- [ ] **P2** Create deployment confirmation modal
  - Success Criteria: Confirm deploy
- [ ] **P2** Implement deployment progress display
  - Success Criteria: Deploy progress
- [ ] **P2** Create deployment log viewer
  - Success Criteria: Deploy logs
- [ ] **P2** Implement deployment success confirmation
  - Success Criteria: Success message
- [ ] **P2** Implement rollback functionality
  - Success Criteria: Rollback
- [ ] **P2** Write Deployment UI unit tests
  - Success Criteria: Tests complete
- [ ] **P2** Create Deployment UI Storybook stories
  - Success Criteria: Documentation complete

---

## Technical Implementation Guidelines

### Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ui/              # Design system primitives
│   │   ├── forms/           # Form components
│   │   ├── charts/          # Chart components
│   │   ├── tables/          # Table components
│   │   └── layout/          # Layout components
│   ├── features/            # Feature-specific components
│   │   ├── admin/           # Admin dashboard features
│   │   ├── portal/          # User portal features
│   │   ├── builder/         # Agent builder features
│   │   └── auth/            # Authentication features
│   ├── hooks/               # Custom React hooks
│   ├── stores/              # Zustand state stores
│   ├── api/                 # API client (React Query)
│   ├── lib/                 # Utility functions
│   ├── types/               # TypeScript types
│   └── styles/              # Global styles, Tailwind config
├── tests/                   # Test files
├── storybook/               # Storybook configuration
└── public/                  # Static assets
```

### Coding Standards

- TypeScript strict mode enabled
- ESLint with Airbnb config
- Prettier for formatting
- 100% TypeScript coverage (no any types in production)
- Component files: PascalCase.tsx
- Hook files: useCamelCase.ts
- Utility files: camelCase.ts

### Performance Targets

- First Contentful Paint (FCP): < 1.5s
- Largest Contentful Paint (LCP): < 2.5s
- Time to Interactive (TTI): < 3.5s
- Cumulative Layout Shift (CLS): < 0.1
- Bundle size (gzipped): < 500KB initial load

### Testing Requirements

- Unit test coverage: 80%+
- Integration test coverage: 70%+
- E2E test coverage: Key user flows
- Accessibility tests: axe-core passing
- Visual regression: Chromatic or Percy

---

## Summary

### Phase Distribution

| Phase | Tasks | Priority | Duration |
|-------|-------|----------|----------|
| Phase 1: Design System Foundation | 296 | P1 | 6 weeks |
| Phase 2: Admin Dashboard | 204 | P1 | 8 weeks |
| Phase 3: User Portal | 242 | P1-P2 | 10 weeks |
| Phase 4: Agent Builder UI | 168 | P2 | 8 weeks |
| **TOTAL** | **910** | - | **32 weeks** |

### Key Milestones

| Milestone | Target Date | Deliverables |
|-----------|-------------|--------------|
| Design System v1 | Week 6 | Component library, Storybook, Tokens |
| Admin Dashboard MVP | Week 14 | Agent mgmt, User mgmt, Monitoring |
| User Portal MVP | Week 24 | Auth, Dashboard, Agent UIs, Reports |
| Agent Builder Alpha | Week 32 | Visual designer, Workflow builder, Testing |

### Resource Requirements

| Role | Count | Responsibilities |
|------|-------|------------------|
| Senior Frontend Developer | 2 | Architecture, code review, complex components |
| Frontend Developer | 4 | Component development, pages, integration |
| UI/UX Designer | 2 | Design system, user research, prototypes |
| QA Engineer | 1 | Testing, accessibility audits |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-FrontendDeveloper | Initial comprehensive frontend TODO |

---

## Approvals

- [ ] **Frontend Lead:** _____________________ Date: _______
- [ ] **Engineering Lead:** _____________________ Date: _______
- [ ] **Product Manager:** _____________________ Date: _______
- [ ] **UX Lead:** _____________________ Date: _______

---

**END OF DOCUMENT**