#!/usr/bin/env node
/**
 * Minimal CLI for @greenlang/factors.
 *
 * Usage:
 *   glfactors search "steel" [--geography US] [--limit 20]
 *   glfactors resolve <config.json>
 *   glfactors explain <factor_id> [--edition ef_2026_q1]
 *
 * The CLI intentionally avoids commander.js to keep the install weight
 * at zero; if you want a richer command surface, install `commander`
 * and wrap the methods from `client.ts` yourself.
 *
 * Configuration is picked up from environment variables:
 *   GL_FACTORS_API_URL  (default: https://api.greenlang.io)
 *   GL_FACTORS_API_KEY
 *   GL_FACTORS_JWT
 *   GL_FACTORS_EDITION
 */

import { readFileSync } from 'node:fs';

import { FactorsClient } from './client';

interface ParsedArgs {
  command: string;
  positional: string[];
  flags: Record<string, string | boolean>;
}

function parseArgs(argv: string[]): ParsedArgs {
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};
  let command = '';
  let i = 0;
  while (i < argv.length) {
    const tok = argv[i];
    if (!command && !tok.startsWith('-')) {
      command = tok;
    } else if (tok.startsWith('--')) {
      const key = tok.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) {
        flags[key] = next;
        i += 1;
      } else {
        flags[key] = true;
      }
    } else if (tok.startsWith('-')) {
      flags[tok.slice(1)] = true;
    } else {
      positional.push(tok);
    }
    i += 1;
  }
  return { command, positional, flags };
}

function usage(): never {
  // eslint-disable-next-line no-console
  console.error(
    [
      'glfactors — GreenLang Factors CLI',
      '',
      'Usage:',
      '  glfactors search <query> [--geography CODE] [--limit N] [--edition ID]',
      '  glfactors resolve <request.json> [--edition ID]',
      '  glfactors explain <factor_id> [--edition ID]',
      '  glfactors list-editions',
      '  glfactors coverage [--edition ID]',
      '',
      'Environment:',
      '  GL_FACTORS_API_URL  (default: https://api.greenlang.io)',
      '  GL_FACTORS_API_KEY  (or GL_FACTORS_JWT for JWT auth)',
      '  GL_FACTORS_EDITION  (default edition pin)',
    ].join('\n'),
  );
  process.exit(2);
}

function buildClient(cliEdition?: string): FactorsClient {
  const baseUrl = process.env.GL_FACTORS_API_URL ?? 'https://api.greenlang.io';
  return new FactorsClient({
    baseUrl,
    apiKey: process.env.GL_FACTORS_API_KEY,
    jwtToken: process.env.GL_FACTORS_JWT,
    edition: cliEdition ?? process.env.GL_FACTORS_EDITION,
  });
}

function printJson(value: unknown): void {
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(value, null, 2));
}

export async function main(argv: string[] = process.argv.slice(2)): Promise<void> {
  if (argv.length === 0) usage();
  const args = parseArgs(argv);
  const cliEdition = typeof args.flags.edition === 'string' ? args.flags.edition : undefined;
  const client = buildClient(cliEdition);

  switch (args.command) {
    case 'search': {
      const query = args.positional[0];
      if (!query) usage();
      const result = await client.search(query, {
        geography:
          typeof args.flags.geography === 'string' ? args.flags.geography : undefined,
        limit:
          typeof args.flags.limit === 'string' ? parseInt(args.flags.limit, 10) : undefined,
        edition: cliEdition,
      });
      printJson(result);
      break;
    }
    case 'resolve': {
      const path = args.positional[0];
      if (!path) usage();
      const payload = JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
      const result = await client.resolve(payload, { edition: cliEdition });
      printJson(result);
      break;
    }
    case 'explain': {
      const factorId = args.positional[0];
      if (!factorId) usage();
      const result = await client.resolveExplain(factorId, { edition: cliEdition });
      printJson(result);
      break;
    }
    case 'list-editions': {
      const result = await client.listEditions();
      printJson(result);
      break;
    }
    case 'coverage': {
      const result = await client.coverage({ edition: cliEdition });
      printJson(result);
      break;
    }
    case 'help':
    case '--help':
    case '-h':
      usage();
      break;
    default:
      usage();
  }
}

// Auto-invoke when used as a script. We avoid `require.main === module`
// (CJS-only) and instead compare `process.argv[1]` to this file's URL,
// which works in both CJS and ESM builds.
const isCliEntrypoint = (() => {
  try {
    const entry = process.argv[1] ?? '';
    return (
      entry.endsWith('cli.js') ||
      entry.endsWith('cli.ts') ||
      entry.endsWith('glfactors')
    );
  } catch {
    return false;
  }
})();

if (isCliEntrypoint) {
  main().catch((err: Error) => {
    // eslint-disable-next-line no-console
    console.error(err.message);
    process.exit(1);
  });
}
