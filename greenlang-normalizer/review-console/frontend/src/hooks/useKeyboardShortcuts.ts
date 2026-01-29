/**
 * useKeyboardShortcuts Hook
 *
 * Global keyboard shortcut handler for power users.
 */

import { useEffect, useCallback, useRef } from 'react';

interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  action: () => void;
  description: string;
  when?: () => boolean; // Conditional execution
}

interface UseKeyboardShortcutsOptions {
  enabled?: boolean;
  ignoreInputs?: boolean; // Ignore when focused on input/textarea
}

/**
 * Hook for registering and managing keyboard shortcuts
 */
export function useKeyboardShortcuts(
  shortcuts: ShortcutConfig[],
  options: UseKeyboardShortcutsOptions = {}
) {
  const { enabled = true, ignoreInputs = true } = options;
  const shortcutsRef = useRef(shortcuts);

  // Update ref when shortcuts change
  useEffect(() => {
    shortcutsRef.current = shortcuts;
  }, [shortcuts]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      // Ignore when typing in inputs
      if (ignoreInputs) {
        const target = event.target as HTMLElement;
        if (
          target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable
        ) {
          return;
        }
      }

      const key = event.key.toLowerCase();

      for (const shortcut of shortcutsRef.current) {
        const matchesKey = shortcut.key.toLowerCase() === key;
        const matchesCtrl = shortcut.ctrl ? event.ctrlKey || event.metaKey : !event.ctrlKey;
        const matchesShift = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const matchesAlt = shortcut.alt ? event.altKey : !event.altKey;
        const matchesMeta = shortcut.meta ? event.metaKey : true; // Meta is optional

        const conditionMet = shortcut.when ? shortcut.when() : true;

        if (matchesKey && matchesCtrl && matchesShift && matchesAlt && matchesMeta && conditionMet) {
          event.preventDefault();
          shortcut.action();
          return;
        }
      }
    },
    [enabled, ignoreInputs]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

/**
 * Hook for showing a keyboard shortcuts help modal
 */
export function useShortcutHelp() {
  const shortcuts: Array<{ key: string; description: string; category?: string }> = [
    // Navigation
    { key: 'G then D', description: 'Go to Dashboard', category: 'Navigation' },
    { key: 'G then Q', description: 'Go to Queue', category: 'Navigation' },
    { key: 'G then S', description: 'Go to Settings', category: 'Navigation' },

    // Queue actions
    { key: 'N', description: 'Next item', category: 'Queue' },
    { key: 'S', description: 'Skip item', category: 'Queue' },
    { key: 'R', description: 'Refresh queue', category: 'Queue' },

    // Review actions
    { key: 'A', description: 'Accept top candidate', category: 'Review' },
    { key: '1-9', description: 'Select candidate by number', category: 'Review' },
    { key: 'R', description: 'Reject all candidates', category: 'Review' },
    { key: 'D', description: 'Defer item', category: 'Review' },
    { key: 'E', description: 'Escalate item', category: 'Review' },
    { key: 'Ctrl+Enter', description: 'Submit resolution', category: 'Review' },
    { key: 'Escape', description: 'Cancel / Reset', category: 'Review' },

    // General
    { key: '?', description: 'Show keyboard shortcuts', category: 'General' },
    { key: '/', description: 'Focus search', category: 'General' },
  ];

  // Group by category
  const groupedShortcuts = shortcuts.reduce(
    (acc, shortcut) => {
      const category = shortcut.category || 'Other';
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(shortcut);
      return acc;
    },
    {} as Record<string, typeof shortcuts>
  );

  return { shortcuts, groupedShortcuts };
}

/**
 * Common shortcut presets
 */
export const shortcutPresets = {
  // Submit form
  submit: (action: () => void): ShortcutConfig => ({
    key: 'Enter',
    ctrl: true,
    action,
    description: 'Submit form',
  }),

  // Cancel/close
  cancel: (action: () => void): ShortcutConfig => ({
    key: 'Escape',
    action,
    description: 'Cancel / Close',
  }),

  // Search focus
  search: (action: () => void): ShortcutConfig => ({
    key: '/',
    action,
    description: 'Focus search',
  }),

  // Help
  help: (action: () => void): ShortcutConfig => ({
    key: '?',
    shift: true,
    action,
    description: 'Show help',
  }),

  // Refresh
  refresh: (action: () => void): ShortcutConfig => ({
    key: 'r',
    action,
    description: 'Refresh',
  }),
};

export default useKeyboardShortcuts;
