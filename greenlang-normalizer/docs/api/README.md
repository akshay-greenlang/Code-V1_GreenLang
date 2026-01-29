# API Documentation

This directory contains API documentation for the GreenLang Normalizer Service.

## Contents

- `openapi.yaml` - OpenAPI 3.0 specification
- `endpoints.md` - Endpoint reference documentation
- `authentication.md` - Authentication guide
- `rate-limiting.md` - Rate limiting policies
- `error-codes.md` - Error code reference

## API Overview

The GreenLang Normalizer provides a RESTful API for:

- **Unit Conversion**: Convert values between compatible units
- **Reference Resolution**: Match strings to canonical vocabulary entries
- **Unit Parsing**: Parse quantity strings into components
- **Vocabulary Management**: Query and manage vocabularies

## Base URL

```
Production: https://api.greenlang.io/normalizer/v1
Staging: https://api-staging.greenlang.io/normalizer/v1
Development: http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication via:
- API Key: `X-API-Key` header
- Bearer Token: `Authorization: Bearer <token>` header

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/convert` | 100 req/min |
| `/resolve` | 100 req/min |
| `/convert/batch` | 20 req/min |
| `/vocabularies` | 50 req/min |

## Response Format

All responses follow this structure:

```json
{
  "data": { ... },
  "meta": {
    "request_id": "req-xxx",
    "timestamp": "2026-01-30T00:00:00Z"
  }
}
```

## Error Handling

Errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://api.greenlang.io/errors/conversion-failed",
  "title": "Conversion Failed",
  "status": 400,
  "detail": "Cannot convert from 'kg' to 'liter'",
  "instance": "/api/v1/convert"
}
```
