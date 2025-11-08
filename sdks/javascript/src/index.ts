/**
 * GreenLang JavaScript/TypeScript SDK
 *
 * Official JavaScript/TypeScript SDK for the GreenLang API.
 *
 * @example
 * ```typescript
 * import { GreenLangClient } from '@greenlang/sdk';
 *
 * const client = new GreenLangClient({ apiKey: 'your-api-key' });
 *
 * const result = await client.executeWorkflow('wf_123', {
 *   query: 'What is carbon footprint?'
 * });
 *
 * console.log(result.data);
 * ```
 */

export { GreenLangClient } from './client';
export * from './types';
export * from './errors';
export { version } from './version';
