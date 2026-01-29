/**
 * Sidebar Component
 *
 * Main navigation sidebar for the Review Console.
 * Includes navigation links, keyboard shortcuts panel, and user info.
 */

import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  QueueListIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  QuestionMarkCircleIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CommandLineIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number;
}

const navigation: NavItem[] = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Review Queue', href: '/queue', icon: QueueListIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
];

const shortcuts = [
  { key: 'A', description: 'Accept top match' },
  { key: 'R', description: 'Reject all' },
  { key: 'S', description: 'Skip item' },
  { key: 'N', description: 'Next item' },
  { key: '1-9', description: 'Select candidate' },
  { key: 'Enter', description: 'Submit' },
  { key: 'Esc', description: 'Cancel' },
  { key: '?', description: 'Show shortcuts' },
];

interface SidebarProps {
  pendingCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ pendingCount = 0 }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const location = useLocation();

  return (
    <>
      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed inset-y-0 left-0 z-50 flex flex-col',
          'bg-gl-neutral-900 transition-all duration-300',
          collapsed ? 'w-16' : 'w-64'
        )}
      >
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b border-gl-neutral-800">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-8 h-8 bg-gl-primary-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">GL</span>
            </div>
            {!collapsed && (
              <div className="animate-fade-in">
                <h1 className="text-white font-semibold text-sm">GreenLang</h1>
                <p className="text-gl-neutral-400 text-xs">Review Console</p>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            const showBadge = item.href === '/queue' && pendingCount > 0;

            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors',
                  'text-sm font-medium',
                  isActive
                    ? 'bg-gl-primary-600 text-white'
                    : 'text-gl-neutral-300 hover:bg-gl-neutral-800 hover:text-white'
                )}
                title={collapsed ? item.name : undefined}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && (
                  <>
                    <span className="flex-1">{item.name}</span>
                    {showBadge && (
                      <span className="px-2 py-0.5 text-xs font-medium bg-red-500 text-white rounded-full">
                        {pendingCount > 99 ? '99+' : pendingCount}
                      </span>
                    )}
                  </>
                )}
                {collapsed && showBadge && (
                  <span className="absolute left-10 top-1 w-2 h-2 bg-red-500 rounded-full" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Keyboard shortcuts toggle */}
        {!collapsed && (
          <div className="px-2 py-2 border-t border-gl-neutral-800">
            <button
              onClick={() => setShowShortcuts(!showShortcuts)}
              className="flex items-center gap-3 w-full px-3 py-2.5 rounded-md
                         text-sm font-medium text-gl-neutral-300
                         hover:bg-gl-neutral-800 hover:text-white transition-colors"
            >
              <CommandLineIcon className="w-5 h-5" />
              <span>Shortcuts</span>
              <span className="ml-auto kbd text-xs">?</span>
            </button>
          </div>
        )}

        {/* Help link */}
        <div className="px-2 py-2 border-t border-gl-neutral-800">
          <a
            href="/help"
            className="flex items-center gap-3 px-3 py-2.5 rounded-md
                       text-sm font-medium text-gl-neutral-300
                       hover:bg-gl-neutral-800 hover:text-white transition-colors"
            title={collapsed ? 'Help' : undefined}
          >
            <QuestionMarkCircleIcon className="w-5 h-5" />
            {!collapsed && <span>Help</span>}
          </a>
        </div>

        {/* Collapse toggle */}
        <div className="px-2 py-2 border-t border-gl-neutral-800">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex items-center justify-center w-full px-3 py-2.5 rounded-md
                       text-sm font-medium text-gl-neutral-300
                       hover:bg-gl-neutral-800 hover:text-white transition-colors"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? (
              <ChevronRightIcon className="w-5 h-5" />
            ) : (
              <>
                <ChevronLeftIcon className="w-5 h-5" />
                <span className="ml-2">Collapse</span>
              </>
            )}
          </button>
        </div>
      </aside>

      {/* Keyboard shortcuts panel */}
      {showShortcuts && !collapsed && (
        <div
          className="fixed left-64 bottom-4 z-50 w-64 p-4 bg-white rounded-lg shadow-lg border border-gl-neutral-200 animate-in"
          role="dialog"
          aria-label="Keyboard shortcuts"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gl-neutral-900">
              Keyboard Shortcuts
            </h3>
            <button
              onClick={() => setShowShortcuts(false)}
              className="text-gl-neutral-400 hover:text-gl-neutral-600"
              aria-label="Close shortcuts panel"
            >
              <span className="text-lg">&times;</span>
            </button>
          </div>
          <dl className="space-y-2">
            {shortcuts.map((shortcut) => (
              <div key={shortcut.key} className="flex items-center justify-between">
                <dt className="text-xs text-gl-neutral-600">{shortcut.description}</dt>
                <dd>
                  <kbd className="kbd">{shortcut.key}</kbd>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {/* Spacer for main content */}
      <div className={clsx('flex-shrink-0', collapsed ? 'w-16' : 'w-64')} />
    </>
  );
};

export default Sidebar;
