/**
 * @greenlang/factors — TypeScript SDK for the GreenLang Factors REST API.
 *
 * Public surface. Import from here; internal modules may be reorganised
 * without notice.
 *
 * @example
 * ```ts
 * import { FactorsClient } from "@greenlang/factors";
 *
 * const client = new FactorsClient({
 *   baseUrl: "https://api.greenlang.io",
 *   apiKey: process.env.GL_API_KEY!,
 *   edition: "ef_2026_q1",
 * });
 *
 * const hits = await client.search("diesel scope 1 US");
 * for (const f of hits.factors) {
 *   console.log(f.factor_id, f.co2e_per_unit);
 * }
 * ```
 */

export const SDK_VERSION = '1.3.0';

// Client
export {
  FactorsClient,
  GREENLANG_CA_FINGERPRINT_SHA256,
  PINNED_HOST_SUFFIXES,
  computeSubresourceIntegrity,
} from './client';
export type { FactorsClientOptions, SearchV2Options } from './client';

// Models
export type {
  ActivitySchema,
  AuditBundle,
  BatchJobHandle,
  BatchJobStatus,
  ChosenFactor,
  CoverageReport,
  DeprecationStatus,
  Edition,
  Factor,
  FactorDiff,
  FactorMatch,
  GasBreakdown,
  Jurisdiction,
  LicensingEnvelope,
  MethodPack,
  MethodPackCoverage,
  MethodPackCoverageReport,
  Override,
  QualityEnvelope,
  QualityScore,
  ResolutionRequest,
  ResolvedFactor,
  SearchResponse,
  SignedReceiptEnvelope,
  Source,
  SourceDescriptor,
  Uncertainty,
  UncertaintyEnvelope,
} from './models';
export {
  inflateDeprecationStatus,
  inflateMethodPackCoverage,
  isTerminalBatchStatus,
} from './models';

// Errors
export {
  AuthError,
  CertificatePinError,
  EditionMismatchError,
  EditionPinError,
  EntitlementError,
  FactorCannotResolveSafelyError,
  FactorNotFoundError,
  FactorsAPIError,
  LicenseError,
  LicensingGapError,
  RateLimitError,
  TierError,
  ValidationError,
  errorFromResponse,
} from './errors';
export type { FactorsAPIErrorOptions } from './errors';

// Receipt verification (offline)
export {
  ReceiptVerificationError,
  verifyReceipt,
  DEFAULT_FUTURE_TOLERANCE_SEC,
} from './verify';
export type {
  SignedReceipt,
  VerifiedReceipt,
  VerifyReceiptOptions,
} from './verify';

// Auth
export { APIKeyAuth, HMACAuth, JWTAuth, composeAuthHeaders } from './auth';
export type {
  APIKeyAuthOptions,
  AuthContext,
  AuthProvider,
  HMACAuthOptions,
} from './auth';

// Transport
export {
  DEFAULT_MAX_RETRIES,
  DEFAULT_TIMEOUT_MS,
  DEFAULT_USER_AGENT,
  ETagCache,
  Transport,
} from './transport';
export type {
  FetchLike,
  FetchResponseLike,
  RateLimitInfo,
  TransportOptions,
  TransportResponse,
} from './transport';

// Pagination
export {
  CursorPaginator,
  OffsetPaginator,
  extractItems,
} from './pagination';
export type { CursorFetcher, OffsetFetcher, PageInfo } from './pagination';

// Webhooks
export {
  WebhookVerificationError,
  canonicalJsonStringify,
  computeSignature,
  computeSignatureBytes,
  parseSignatureHeader,
  signWebhook,
  verifyWebhook,
  verifyWebhookBytes,
  verifyWebhookStrict,
} from './webhooks';
export type { WebhookPayload } from './webhooks';

// Hash (low-level helpers, exposed mainly for advanced integrations)
export { constantTimeEqual, hmacSha256Hex, sha256Hex } from './hash';

// Canonical JSON (exposed so integrators can pre-canonicalise if needed)
export { canonicalJsonBytes } from './canonical';

import { FactorsClient } from './client';
export default FactorsClient;
