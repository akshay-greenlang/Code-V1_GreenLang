/**
 * KPICard Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import KPICard from '../src/components/charts/KPICard';
import { Assessment } from '@mui/icons-material';

describe('KPICard', () => {
  it('renders title and value correctly', () => {
    render(
      <KPICard
        title="Overall Efficiency"
        value={92.5}
        unit="%"
      />
    );

    expect(screen.getByText('Overall Efficiency')).toBeInTheDocument();
    expect(screen.getByText('92.5')).toBeInTheDocument();
    expect(screen.getByText('%')).toBeInTheDocument();
  });

  it('displays trend indicator when provided', () => {
    render(
      <KPICard
        title="Efficiency"
        value={90}
        unit="%"
        trend="increasing"
        trendValue={5.2}
      />
    );

    expect(screen.getByText('+5.2%')).toBeInTheDocument();
  });

  it('shows status color based on value', () => {
    const { rerender } = render(
      <KPICard
        title="Temperature"
        value={100}
        unit="°C"
        status="good"
      />
    );

    let valueElement = screen.getByText('100');
    expect(valueElement).toHaveStyle({ color: expect.stringContaining('success') });

    rerender(
      <KPICard
        title="Temperature"
        value={100}
        unit="°C"
        status="critical"
      />
    );

    valueElement = screen.getByText('100');
    expect(valueElement).toHaveStyle({ color: expect.stringContaining('error') });
  });

  it('displays target comparison chip', () => {
    render(
      <KPICard
        title="Production"
        value={90}
        unit="t/hr"
        target={100}
      />
    );

    expect(screen.getByText('90% of target')).toBeInTheDocument();
  });

  it('calls onClick handler when clicked', () => {
    const handleClick = vi.fn();

    render(
      <KPICard
        title="Efficiency"
        value={95}
        unit="%"
        onClick={handleClick}
      />
    );

    const card = screen.getByText('Efficiency').closest('.MuiCard-root');
    fireEvent.click(card!);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('shows loading state', () => {
    render(
      <KPICard
        title="Loading KPI"
        value={0}
        unit="%"
        loading={true}
      />
    );

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('formats value using custom formatter', () => {
    const formatter = (value: number) => `$${value.toFixed(2)}`;

    render(
      <KPICard
        title="Cost"
        value={1234.567}
        format={formatter}
      />
    );

    expect(screen.getByText('$1234.57')).toBeInTheDocument();
  });

  it('displays icon when provided', () => {
    render(
      <KPICard
        title="Efficiency"
        value={95}
        unit="%"
        icon={<Assessment data-testid="assessment-icon" />}
      />
    );

    expect(screen.getByTestId('assessment-icon')).toBeInTheDocument();
  });
});
