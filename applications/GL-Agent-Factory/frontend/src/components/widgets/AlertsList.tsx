/**
 * AlertsList Component
 *
 * Display system alerts with severity indicators.
 */

import * as React from 'react';
import {
  AlertCircle,
  AlertTriangle,
  Info,
  XCircle,
  Check,
  X,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/utils/cn';
import type { SystemAlert } from '@/api/types';
import { formatRelativeTime } from '@/utils/format';

interface AlertsListProps {
  alerts: SystemAlert[];
  onAcknowledge?: (alertId: string) => void;
  onDismiss?: (alertId: string) => void;
  loading?: boolean;
  maxItems?: number;
  className?: string;
}

const severityConfig = {
  info: {
    icon: Info,
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    textColor: 'text-blue-700',
    iconColor: 'text-blue-500',
    badgeVariant: 'info' as const,
  },
  warning: {
    icon: AlertTriangle,
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    textColor: 'text-amber-700',
    iconColor: 'text-amber-500',
    badgeVariant: 'warning' as const,
  },
  error: {
    icon: AlertCircle,
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    textColor: 'text-red-700',
    iconColor: 'text-red-500',
    badgeVariant: 'destructive' as const,
  },
  critical: {
    icon: XCircle,
    bgColor: 'bg-red-100',
    borderColor: 'border-red-300',
    textColor: 'text-red-800',
    iconColor: 'text-red-600',
    badgeVariant: 'destructive' as const,
  },
};

export function AlertsList({
  alerts,
  onAcknowledge,
  onDismiss,
  loading,
  maxItems,
  className,
}: AlertsListProps) {
  const displayedAlerts = maxItems ? alerts.slice(0, maxItems) : alerts;

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>System Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="animate-pulse rounded-lg border p-4">
                <div className="flex items-start gap-3">
                  <div className="h-5 w-5 rounded-full bg-muted" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 w-3/4 rounded bg-muted" />
                    <div className="h-3 w-1/2 rounded bg-muted" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (alerts.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>System Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="rounded-full bg-greenlang-100 p-3 mb-3">
              <Check className="h-6 w-6 text-greenlang-600" />
            </div>
            <p className="font-medium">All Systems Operational</p>
            <p className="text-sm text-muted-foreground">
              No active alerts at this time
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>System Alerts</CardTitle>
        <Badge variant="destructive">{alerts.length} Active</Badge>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {displayedAlerts.map((alert) => {
            const config = severityConfig[alert.severity];
            const Icon = config.icon;

            return (
              <div
                key={alert.id}
                className={cn(
                  'rounded-lg border p-4 transition-all',
                  config.bgColor,
                  config.borderColor,
                  alert.acknowledgedAt && 'opacity-60'
                )}
              >
                <div className="flex items-start gap-3">
                  <Icon className={cn('h-5 w-5 mt-0.5', config.iconColor)} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <h4 className={cn('font-medium', config.textColor)}>
                        {alert.title}
                      </h4>
                      <div className="flex items-center gap-1">
                        {!alert.acknowledgedAt && onAcknowledge && (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => onAcknowledge(alert.id)}
                            className="h-6 w-6"
                          >
                            <Check className="h-4 w-4" />
                          </Button>
                        )}
                        {onDismiss && (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => onDismiss(alert.id)}
                            className="h-6 w-6"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                    <p className={cn('text-sm mt-1', config.textColor)}>
                      {alert.message}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge variant={config.badgeVariant} size="sm">
                        {alert.severity.toUpperCase()}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {formatRelativeTime(alert.createdAt)}
                      </span>
                      {alert.acknowledgedAt && (
                        <span className="text-xs text-muted-foreground">
                          (Acknowledged)
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {maxItems && alerts.length > maxItems && (
          <div className="mt-4 text-center">
            <Button variant="outline" size="sm">
              View All {alerts.length} Alerts
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
