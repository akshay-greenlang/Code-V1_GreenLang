/**
 * AgentStatusCard Component
 *
 * Display agent status with metrics and quick actions.
 */

import * as React from 'react';
import { Link } from 'react-router-dom';
import {
  Bot,
  MoreVertical,
  Play,
  Pause,
  RefreshCw,
  Settings,
  ExternalLink,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@radix-ui/react-dropdown-menu';
import { cn } from '@/utils/cn';
import type { Agent, AgentStatus } from '@/api/types';
import { formatRelativeTime, formatNumber } from '@/utils/format';

interface AgentStatusCardProps {
  agent: Agent;
  onRestart?: (agentId: string) => void;
  onToggle?: (agentId: string) => void;
  className?: string;
}

const statusConfig: Record<
  AgentStatus,
  { variant: 'active' | 'inactive' | 'error' | 'pending' | 'processing'; label: string }
> = {
  active: { variant: 'active', label: 'Active' },
  inactive: { variant: 'inactive', label: 'Inactive' },
  error: { variant: 'error', label: 'Error' },
  maintenance: { variant: 'pending', label: 'Maintenance' },
  deploying: { variant: 'processing', label: 'Deploying' },
};

const agentTypeIcons: Record<string, string> = {
  cbam: 'CBAM',
  eudr: 'EUDR',
  fuel: 'FUEL',
  building: 'BLD',
  sb253: 'SB253',
};

export function AgentStatusCard({
  agent,
  onRestart,
  onToggle,
  className,
}: AgentStatusCardProps) {
  const config = statusConfig[agent.status];

  return (
    <Card className={cn('relative', className)} variant="interactive">
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary font-semibold text-sm">
            {agentTypeIcons[agent.type] || <Bot className="h-5 w-5" />}
          </div>
          <div>
            <CardTitle className="text-base">{agent.name}</CardTitle>
            <p className="text-xs text-muted-foreground">v{agent.version}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant={config.variant} dot>
            {config.label}
          </Badge>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon-sm">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              className="w-48 rounded-md border bg-popover p-1 shadow-md"
            >
              <DropdownMenuItem
                className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
                onClick={() => onToggle?.(agent.id)}
              >
                {agent.status === 'active' ? (
                  <>
                    <Pause className="h-4 w-4" /> Pause Agent
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" /> Start Agent
                  </>
                )}
              </DropdownMenuItem>
              <DropdownMenuItem
                className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
                onClick={() => onRestart?.(agent.id)}
              >
                <RefreshCw className="h-4 w-4" /> Restart Agent
              </DropdownMenuItem>
              <DropdownMenuItem
                className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
                asChild
              >
                <Link to={`/admin/agents/${agent.id}`}>
                  <Settings className="h-4 w-4" /> Configure
                </Link>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-muted-foreground line-clamp-2 mb-4">
          {agent.description}
        </p>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Requests Today</p>
            <p className="font-semibold">{formatNumber(agent.metrics.requestsToday)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Avg Response</p>
            <p className="font-semibold">{agent.metrics.avgResponseTime}ms</p>
          </div>
          <div>
            <p className="text-muted-foreground">Error Rate</p>
            <p className={cn('font-semibold', agent.metrics.errorRate > 5 && 'text-destructive')}>
              {agent.metrics.errorRate.toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Uptime</p>
            <p className="font-semibold">{agent.metrics.uptime.toFixed(1)}%</p>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between border-t pt-4">
          <p className="text-xs text-muted-foreground">
            Last deployed {formatRelativeTime(agent.lastDeployedAt)}
          </p>
          <Link
            to={`/admin/agents/${agent.id}`}
            className="text-xs text-primary hover:underline inline-flex items-center gap-1"
          >
            View Details <ExternalLink className="h-3 w-3" />
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}

// Loading skeleton for agent card
export function AgentStatusCardSkeleton() {
  return (
    <Card className="animate-pulse">
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-muted" />
          <div className="space-y-1">
            <div className="h-4 w-24 rounded bg-muted" />
            <div className="h-3 w-12 rounded bg-muted" />
          </div>
        </div>
        <div className="h-5 w-16 rounded-full bg-muted" />
      </CardHeader>
      <CardContent>
        <div className="h-4 w-full rounded bg-muted mb-2" />
        <div className="h-4 w-3/4 rounded bg-muted mb-4" />
        <div className="grid grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i}>
              <div className="h-3 w-16 rounded bg-muted mb-1" />
              <div className="h-5 w-12 rounded bg-muted" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
