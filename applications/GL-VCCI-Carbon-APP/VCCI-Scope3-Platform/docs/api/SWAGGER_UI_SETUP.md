# GL-VCCI Scope 3 Platform - Swagger UI Setup Guide

**Version**: 1.0
**Last Updated**: 2025-11-07
**Platform**: GL-VCCI Carbon Intelligence Platform

---

## Table of Contents

1. [Introduction](#introduction)
2. [Swagger UI Overview](#swagger-ui-overview)
3. [OpenAPI Specification](#openapi-specification)
4. [Installation Options](#installation-options)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Standalone HTML Deployment](#standalone-html-deployment)
8. [Configuration](#configuration)
9. [Authentication Setup](#authentication-setup)
10. [Customization](#customization)
11. [Production Best Practices](#production-best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Swagger UI?

Swagger UI is an open-source tool that automatically generates interactive API documentation from OpenAPI (formerly Swagger) specifications. It provides:

- **Interactive Documentation**: Test API endpoints directly in the browser
- **Code Generation**: Generate client code in multiple languages
- **API Explorer**: Browse and understand API structure
- **Request/Response Examples**: See real-world API usage

### Benefits for GL-VCCI Platform

- **Developer Onboarding**: Quickly understand API capabilities
- **Testing**: Test endpoints without writing code
- **Documentation**: Always up-to-date API reference
- **Client Development**: Generate client SDKs automatically

### Architecture Overview

```
┌─────────────────┐
│   Developers    │
└────────┬────────┘
         │
         │ HTTPS
         ▼
┌─────────────────┐
│   Swagger UI    │ ← Static HTML/JS application
│   (Frontend)    │
└────────┬────────┘
         │
         │ Fetch OpenAPI spec
         ▼
┌─────────────────┐
│   API Server    │ ← Serves OpenAPI specification
│   /api/spec     │    at /api/v1/openapi.json
└────────┬────────┘
         │
         │ API calls
         ▼
┌─────────────────┐
│   API Endpoints │ ← Actual API implementation
│   /api/v1/*     │
└─────────────────┘
```

---

## Swagger UI Overview

### Key Features

**Interactive API Explorer**:
- Execute API requests directly from the browser
- See request/response examples
- Test authentication flows
- Validate request parameters

**Automatic Documentation**:
- Generated from OpenAPI spec
- Always synchronized with code
- Includes models and schemas
- Shows all available endpoints

**Code Generation**:
- Client SDKs in multiple languages
- Server stubs for new APIs
- Request/response models
- Authentication code

**Collaboration**:
- Share API documentation with team
- Embed in developer portals
- Export to PDF or Markdown
- Version control integration

### Supported OpenAPI Versions

- **OpenAPI 3.0.x**: Recommended (current standard)
- **OpenAPI 3.1.x**: Latest version
- **Swagger 2.0**: Legacy support (upgrade recommended)

---

## OpenAPI Specification

### Generating OpenAPI Spec for GL-VCCI API

**Using Flask-RESTX**:
```python
# app.py
from flask import Flask
from flask_restx import Api, Resource, fields, Namespace

app = Flask(__name__)

# Configure API with metadata
api = Api(
    app,
    version='1.0',
    title='GL-VCCI Scope 3 Carbon Intelligence API',
    description='API for managing Scope 3 carbon emissions data',
    doc='/api/docs',  # Swagger UI endpoint
    prefix='/api/v1',
    contact='support@gl-vcci.com',
    license='Proprietary',
    terms_url='https://gl-vcci.com/terms',
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT token. Format: "Bearer {token}"'
        },
        'ApiKey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-Key',
            'description': 'API Key for service-to-service authentication'
        }
    },
    security='Bearer'
)

# Define namespaces
transactions_ns = Namespace('transactions', description='Transaction operations')
suppliers_ns = Namespace('suppliers', description='Supplier operations')
emissions_ns = Namespace('emissions', description='Emissions calculations')

api.add_namespace(transactions_ns, path='/transactions')
api.add_namespace(suppliers_ns, path='/suppliers')
api.add_namespace(emissions_ns, path='/emissions')

# Define models
transaction_model = api.model('Transaction', {
    'transaction_id': fields.String(required=True, description='Unique transaction ID'),
    'date': fields.Date(required=True, description='Transaction date'),
    'supplier_id': fields.String(required=True, description='Supplier ID'),
    'supplier_name': fields.String(required=True, description='Supplier name'),
    'product_name': fields.String(required=True, description='Product/service name'),
    'product_category': fields.String(required=True, description='UNSPSC code'),
    'quantity': fields.Float(required=True, description='Quantity purchased'),
    'unit': fields.String(required=True, description='Unit of measurement'),
    'spend_usd': fields.Float(required=True, description='Spend in USD'),
    'currency': fields.String(required=True, description='ISO 4217 currency code'),
    'ghg_category': fields.Integer(required=True, description='GHG Protocol category (1-15)'),
    'country': fields.String(required=True, description='ISO 3166-1 alpha-2 country code'),
    'description': fields.String(description='Additional notes'),
})

# Define endpoints
@transactions_ns.route('/')
class TransactionList(Resource):
    @transactions_ns.doc('list_transactions',
                         security='Bearer',
                         params={
                             'page': 'Page number',
                             'per_page': 'Items per page',
                             'supplier_id': 'Filter by supplier',
                             'date_from': 'Filter by start date (YYYY-MM-DD)',
                             'date_to': 'Filter by end date (YYYY-MM-DD)'
                         })
    @transactions_ns.marshal_list_with(transaction_model)
    def get(self):
        """List all transactions"""
        # Implementation
        pass

    @transactions_ns.doc('create_transaction',
                         security='Bearer',
                         responses={
                             201: 'Created',
                             400: 'Validation Error',
                             401: 'Unauthorized',
                             409: 'Conflict (duplicate transaction_id)'
                         })
    @transactions_ns.expect(transaction_model)
    @transactions_ns.marshal_with(transaction_model, code=201)
    def post(self):
        """Create a new transaction"""
        # Implementation
        pass

@transactions_ns.route('/<string:transaction_id>')
@transactions_ns.param('transaction_id', 'Transaction identifier')
class Transaction(Resource):
    @transactions_ns.doc('get_transaction',
                         security='Bearer',
                         responses={
                             200: 'Success',
                             404: 'Not Found'
                         })
    @transactions_ns.marshal_with(transaction_model)
    def get(self, transaction_id):
        """Get transaction by ID"""
        # Implementation
        pass

    @transactions_ns.doc('update_transaction',
                         security='Bearer',
                         responses={
                             200: 'Success',
                             404: 'Not Found',
                             400: 'Validation Error'
                         })
    @transactions_ns.expect(transaction_model)
    @transactions_ns.marshal_with(transaction_model)
    def put(self, transaction_id):
        """Update transaction"""
        # Implementation
        pass

    @transactions_ns.doc('delete_transaction',
                         security='Bearer',
                         responses={
                             204: 'Deleted',
                             404: 'Not Found'
                         })
    def delete(self, transaction_id):
        """Delete transaction"""
        # Implementation
        pass

# Export OpenAPI spec
@app.route('/api/v1/openapi.json')
def openapi_spec():
    """Serve OpenAPI specification"""
    return jsonify(api.__schema__)

if __name__ == '__main__':
    app.run(debug=True)
```

**Using Flask-Swagger (Manual Spec)**:
```python
# openapi.yaml
openapi: 3.0.0
info:
  title: GL-VCCI Scope 3 Carbon Intelligence API
  version: 1.0.0
  description: |
    API for managing Scope 3 carbon emissions data and calculations.

    # Authentication
    All endpoints require authentication using JWT Bearer tokens or API keys.

    # Rate Limiting
    - Standard tier: 100 requests/hour
    - Premium tier: 1000 requests/hour

  contact:
    name: API Support
    email: support@gl-vcci.com
    url: https://gl-vcci.com/support

  license:
    name: Proprietary
    url: https://gl-vcci.com/license

servers:
  - url: https://api.gl-vcci.com/api/v1
    description: Production server
  - url: https://staging-api.gl-vcci.com/api/v1
    description: Staging server
  - url: http://localhost:8080/api/v1
    description: Development server

security:
  - BearerAuth: []
  - ApiKeyAuth: []

tags:
  - name: transactions
    description: Transaction management operations
  - name: suppliers
    description: Supplier master data operations
  - name: emissions
    description: Emissions calculation operations
  - name: reports
    description: Report generation operations

paths:
  /transactions:
    get:
      tags:
        - transactions
      summary: List transactions
      description: Retrieve a paginated list of transactions with optional filters
      operationId: listTransactions
      parameters:
        - name: page
          in: query
          description: Page number (1-indexed)
          required: false
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: per_page
          in: query
          description: Number of items per page
          required: false
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 50
        - name: supplier_id
          in: query
          description: Filter by supplier ID
          required: false
          schema:
            type: string
        - name: date_from
          in: query
          description: Filter by start date (inclusive)
          required: false
          schema:
            type: string
            format: date
        - name: date_to
          in: query
          description: Filter by end date (inclusive)
          required: false
          schema:
            type: string
            format: date
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Transaction'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

    post:
      tags:
        - transactions
      summary: Create transaction
      description: Create a new transaction record
      operationId: createTransaction
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TransactionCreate'
      responses:
        '201':
          description: Transaction created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Transaction'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '409':
          description: Conflict - transaction_id already exists

  /transactions/{transaction_id}:
    get:
      tags:
        - transactions
      summary: Get transaction
      description: Retrieve a specific transaction by ID
      operationId: getTransaction
      parameters:
        - name: transaction_id
          in: path
          required: true
          description: Transaction identifier
          schema:
            type: string
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Transaction'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT authentication token

    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for service-to-service authentication

  schemas:
    Transaction:
      type: object
      required:
        - transaction_id
        - date
        - supplier_id
        - supplier_name
        - product_name
        - product_category
        - quantity
        - unit
        - spend_usd
        - currency
        - ghg_category
        - country
      properties:
        transaction_id:
          type: string
          description: Unique transaction identifier
          example: TXN-2024-MFG-00001
        date:
          type: string
          format: date
          description: Transaction date
          example: '2024-08-15'
        supplier_id:
          type: string
          description: Supplier identifier
          example: SUP-1001
        supplier_name:
          type: string
          description: Supplier name
          example: Precision Steel Manufacturing Co.
        product_name:
          type: string
          description: Product or service name
          example: Cold-rolled steel sheets - Grade 304
        product_category:
          type: string
          description: UNSPSC product category code
          example: '3310.15.10.00'
        quantity:
          type: number
          format: float
          minimum: 0
          exclusiveMinimum: true
          description: Quantity purchased
          example: 5000
        unit:
          type: string
          description: Unit of measurement
          enum: [kg, tonnes, liters, m3, kWh, pieces, units, hours, compute-hours, containers, trips]
          example: kg
        spend_usd:
          type: number
          format: float
          minimum: 0
          exclusiveMinimum: true
          description: Spend amount in USD
          example: 12500.00
        currency:
          type: string
          pattern: '^[A-Z]{3}$'
          description: ISO 4217 currency code
          example: USD
        ghg_category:
          type: integer
          minimum: 1
          maximum: 15
          description: GHG Protocol Scope 3 category
          example: 1
        country:
          type: string
          pattern: '^[A-Z]{2}$'
          description: ISO 3166-1 alpha-2 country code
          example: US
        description:
          type: string
          description: Additional notes or context
          example: High-grade stainless steel for automotive parts manufacturing
        created_at:
          type: string
          format: date-time
          readOnly: true
          description: Record creation timestamp
        updated_at:
          type: string
          format: date-time
          readOnly: true
          description: Record last update timestamp

    TransactionCreate:
      type: object
      required:
        - transaction_id
        - date
        - supplier_id
        - supplier_name
        - product_name
        - product_category
        - quantity
        - unit
        - spend_usd
        - currency
        - ghg_category
        - country
      properties:
        transaction_id:
          type: string
        date:
          type: string
          format: date
        supplier_id:
          type: string
        supplier_name:
          type: string
        product_name:
          type: string
        product_category:
          type: string
        quantity:
          type: number
          format: float
          minimum: 0
          exclusiveMinimum: true
        unit:
          type: string
        spend_usd:
          type: number
          format: float
          minimum: 0
          exclusiveMinimum: true
        currency:
          type: string
          pattern: '^[A-Z]{3}$'
        ghg_category:
          type: integer
          minimum: 1
          maximum: 15
        country:
          type: string
          pattern: '^[A-Z]{2}$'
        description:
          type: string

    Pagination:
      type: object
      properties:
        page:
          type: integer
          description: Current page number
        per_page:
          type: integer
          description: Items per page
        total:
          type: integer
          description: Total number of items
        total_pages:
          type: integer
          description: Total number of pages
        has_next:
          type: boolean
          description: Whether there is a next page
        has_prev:
          type: boolean
          description: Whether there is a previous page

    Error:
      type: object
      properties:
        error:
          type: string
          description: Error type
        message:
          type: string
          description: Human-readable error message
        details:
          type: object
          description: Additional error details

  responses:
    BadRequest:
      description: Bad request - validation error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    Unauthorized:
      description: Unauthorized - authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

---

## Installation Options

### Option 1: Docker Container (Recommended)

**Advantages**:
- Isolated environment
- Easy to deploy and scale
- Consistent across environments
- Simple updates

**Disadvantages**:
- Requires Docker
- Additional resource overhead

### Option 2: Kubernetes Deployment

**Advantages**:
- Native Kubernetes integration
- Automatic scaling and healing
- Service mesh integration
- Production-ready

**Disadvantages**:
- More complex setup
- Requires Kubernetes cluster

### Option 3: Standalone HTML

**Advantages**:
- No server required
- Can be served as static files
- Lightweight
- Easy to customize

**Disadvantages**:
- Limited configuration options
- CORS challenges
- No server-side authentication

---

## Docker Deployment

### Basic Docker Setup

**Dockerfile**:
```dockerfile
# Dockerfile.swagger
FROM swaggerapi/swagger-ui:latest

# Set environment variables
ENV SWAGGER_JSON=/config/openapi.yaml
ENV BASE_URL=/api/docs
ENV PORT=8080

# Copy OpenAPI spec
COPY openapi.yaml /config/openapi.yaml

# Copy custom configuration
COPY swagger-config.json /usr/share/nginx/html/swagger-config.json

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/ || exit 1
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  swagger-ui:
    build:
      context: .
      dockerfile: Dockerfile.swagger
    container_name: gl-vcci-swagger-ui
    ports:
      - "8080:8080"
    environment:
      - SWAGGER_JSON=/config/openapi.yaml
      - BASE_URL=/api/docs
      - VALIDATOR_URL=null  # Disable validator
      - DEEP_LINKING=true
      - DISPLAY_OPERATION_ID=true
      - DEFAULT_MODELS_EXPAND_DEPTH=1
      - DEFAULT_MODEL_EXPAND_DEPTH=1
      - DISPLAY_REQUEST_DURATION=true
      - FILTER=true
      - SHOW_EXTENSIONS=true
      - SHOW_COMMON_EXTENSIONS=true
      - TRY_IT_OUT_ENABLED=true
    volumes:
      - ./openapi.yaml:/config/openapi.yaml:ro
      - ./swagger-config.json:/usr/share/nginx/html/swagger-config.json:ro
    networks:
      - gl-vcci-network
    restart: unless-stopped

  # API server (for reference)
  api:
    image: gl-vcci/api:latest
    container_name: gl-vcci-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/gl_vcci_db
    networks:
      - gl-vcci-network
    depends_on:
      - postgres

  # PostgreSQL
  postgres:
    image: postgres:15
    container_name: gl-vcci-postgres
    environment:
      - POSTGRES_DB=gl_vcci_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    networks:
      - gl-vcci-network
    volumes:
      - postgres-data:/var/lib/postgresql/data

networks:
  gl-vcci-network:
    driver: bridge

volumes:
  postgres-data:
```

**Build and Run**:
```bash
# Build image
docker build -f Dockerfile.swagger -t gl-vcci/swagger-ui:latest .

# Run standalone
docker run -d \
  --name gl-vcci-swagger-ui \
  -p 8080:8080 \
  -e SWAGGER_JSON=/config/openapi.yaml \
  -v $(pwd)/openapi.yaml:/config/openapi.yaml:ro \
  gl-vcci/swagger-ui:latest

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f swagger-ui

# Stop
docker-compose down
```

### Advanced Docker Configuration

**Nginx Reverse Proxy**:
```nginx
# nginx.conf
server {
    listen 80;
    server_name api-docs.gl-vcci.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api-docs.gl-vcci.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/gl-vcci.crt;
    ssl_certificate_key /etc/nginx/ssl/gl-vcci.key;

    # Swagger UI location
    location /api/docs {
        proxy_pass http://swagger-ui:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API proxy (for try-it-out functionality)
    location /api/v1 {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type' always;

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
}
```

---

## Kubernetes Deployment

### Kubernetes Manifests

**ConfigMap for OpenAPI Spec**:
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: swagger-openapi-spec
  namespace: gl-vcci
data:
  openapi.yaml: |
    openapi: 3.0.0
    info:
      title: GL-VCCI Scope 3 Carbon Intelligence API
      version: 1.0.0
    # ... (full OpenAPI spec)
```

**Deployment**:
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swagger-ui
  namespace: gl-vcci
  labels:
    app: swagger-ui
spec:
  replicas: 2
  selector:
    matchLabels:
      app: swagger-ui
  template:
    metadata:
      labels:
        app: swagger-ui
    spec:
      containers:
      - name: swagger-ui
        image: swaggerapi/swagger-ui:v5.10.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: SWAGGER_JSON
          value: /config/openapi.yaml
        - name: BASE_URL
          value: /api/docs
        - name: VALIDATOR_URL
          value: "null"
        - name: DEEP_LINKING
          value: "true"
        - name: DISPLAY_OPERATION_ID
          value: "true"
        - name: TRY_IT_OUT_ENABLED
          value: "true"
        volumeMounts:
        - name: openapi-spec
          mountPath: /config
          readOnly: true
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "100m"
        livenessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: openapi-spec
        configMap:
          name: swagger-openapi-spec
```

**Service**:
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: swagger-ui
  namespace: gl-vcci
  labels:
    app: swagger-ui
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app: swagger-ui
```

**Ingress**:
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: swagger-ui-ingress
  namespace: gl-vcci
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api-docs.gl-vcci.com
    secretName: swagger-ui-tls
  rules:
  - host: api-docs.gl-vcci.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: swagger-ui
            port:
              number: 80
```

**Deploy to Kubernetes**:
```bash
# Create namespace
kubectl create namespace gl-vcci

# Apply manifests
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment
kubectl get pods -n gl-vcci
kubectl get svc -n gl-vcci
kubectl get ingress -n gl-vcci

# View logs
kubectl logs -f deployment/swagger-ui -n gl-vcci

# Port forward for local testing
kubectl port-forward svc/swagger-ui 8080:80 -n gl-vcci
```

### Helm Chart

**Chart.yaml**:
```yaml
apiVersion: v2
name: swagger-ui
description: Swagger UI for GL-VCCI API
type: application
version: 1.0.0
appVersion: "5.10.0"
maintainers:
  - name: GL-VCCI Platform Team
    email: platform@gl-vcci.com
```

**values.yaml**:
```yaml
replicaCount: 2

image:
  repository: swaggerapi/swagger-ui
  tag: v5.10.0
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api-docs.gl-vcci.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: swagger-ui-tls
      hosts:
        - api-docs.gl-vcci.com

resources:
  requests:
    memory: 64Mi
    cpu: 50m
  limits:
    memory: 128Mi
    cpu: 100m

config:
  swaggerJson: /config/openapi.yaml
  baseUrl: /api/docs
  validatorUrl: null
  deepLinking: true
  displayOperationId: true
  tryItOutEnabled: true

openapi:
  spec: |
    openapi: 3.0.0
    info:
      title: GL-VCCI Scope 3 Carbon Intelligence API
      version: 1.0.0
    # ... (full spec)
```

**templates/deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "swagger-ui.fullname" . }}
  labels:
    {{- include "swagger-ui.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "swagger-ui.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "swagger-ui.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: swagger-ui
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: http
          containerPort: {{ .Values.service.targetPort }}
        env:
        - name: SWAGGER_JSON
          value: {{ .Values.config.swaggerJson }}
        - name: BASE_URL
          value: {{ .Values.config.baseUrl }}
        - name: VALIDATOR_URL
          value: "{{ .Values.config.validatorUrl }}"
        volumeMounts:
        - name: openapi-spec
          mountPath: /config
          readOnly: true
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
      volumes:
      - name: openapi-spec
        configMap:
          name: {{ include "swagger-ui.fullname" . }}-openapi
```

**Install with Helm**:
```bash
# Install chart
helm install swagger-ui ./swagger-ui-chart -n gl-vcci

# Upgrade
helm upgrade swagger-ui ./swagger-ui-chart -n gl-vcci

# Uninstall
helm uninstall swagger-ui -n gl-vcci
```

---

## Standalone HTML Deployment

### Generate Standalone HTML

**index.html**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GL-VCCI API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui.css">
    <link rel="icon" href="favicon.png">
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }

        *, *:before, *:after {
            box-sizing: inherit;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        .topbar {
            background-color: #1a5f3f;
        }

        .swagger-ui .topbar .download-url-wrapper input[type=text] {
            border: 2px solid #1a5f3f;
        }

        .swagger-ui .btn.authorize {
            background-color: #1a5f3f;
            border-color: #1a5f3f;
        }

        .swagger-ui .btn.authorize:hover {
            background-color: #145034;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>

    <script src="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.10.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "openapi.yaml",  // Path to your OpenAPI spec
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",

                // Configuration
                displayOperationId: true,
                displayRequestDuration: true,
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                tryItOutEnabled: true,
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                docExpansion: "list",
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],

                // Authentication
                persistAuthorization: true,

                // Custom request interceptor
                requestInterceptor: (request) => {
                    // Add custom headers or modify request
                    console.log('Request:', request);
                    return request;
                },

                // Custom response interceptor
                responseInterceptor: (response) => {
                    // Process response
                    console.log('Response:', response);
                    return response;
                },

                // Custom error handler
                onComplete: () => {
                    console.log('Swagger UI loaded');
                }
            });

            window.ui = ui;
        };
    </script>
</body>
</html>
```

### Deploy as Static Site

**Nginx Configuration**:
```nginx
server {
    listen 80;
    server_name api-docs.gl-vcci.com;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /openapi.yaml {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Content-Type';
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

**Deploy to S3 + CloudFront**:
```bash
# Sync to S3
aws s3 sync ./swagger-ui s3://api-docs.gl-vcci.com/ \
  --delete \
  --cache-control "public, max-age=3600"

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id DISTRIBUTION_ID \
  --paths "/*"
```

---

## Configuration

### Environment Variables

**Common Configuration Options**:
```bash
# OpenAPI spec location
SWAGGER_JSON=/config/openapi.yaml
SWAGGER_JSON_URL=https://api.gl-vcci.com/api/v1/openapi.json

# Base URL for Swagger UI
BASE_URL=/api/docs

# Validator URL (set to null to disable)
VALIDATOR_URL=null

# Deep linking
DEEP_LINKING=true

# Display settings
DISPLAY_OPERATION_ID=true
DISPLAY_REQUEST_DURATION=true
DEFAULT_MODELS_EXPAND_DEPTH=1
DEFAULT_MODEL_EXPAND_DEPTH=1
DOC_EXPANSION=list

# Filter
FILTER=true

# Extensions
SHOW_EXTENSIONS=true
SHOW_COMMON_EXTENSIONS=true

# Try it out
TRY_IT_OUT_ENABLED=true

# Supported methods
SUPPORTED_SUBMIT_METHODS=["get","post","put","delete","patch","head","options"]

# Authentication persistence
PERSIST_AUTHORIZATION=true

# Syntax highlighting
SYNTAX_HIGHLIGHT_ACTIVATE=true
SYNTAX_HIGHLIGHT_THEME=agate
```

### swagger-config.json

**Custom Configuration File**:
```json
{
  "url": "https://api.gl-vcci.com/api/v1/openapi.json",
  "urls": [
    {
      "name": "Production",
      "url": "https://api.gl-vcci.com/api/v1/openapi.json"
    },
    {
      "name": "Staging",
      "url": "https://staging-api.gl-vcci.com/api/v1/openapi.json"
    },
    {
      "name": "Development",
      "url": "http://localhost:8000/api/v1/openapi.json"
    }
  ],
  "validatorUrl": null,
  "deepLinking": true,
  "displayOperationId": true,
  "displayRequestDuration": true,
  "filter": true,
  "showExtensions": true,
  "showCommonExtensions": true,
  "tryItOutEnabled": true,
  "persistAuthorization": true,
  "docExpansion": "list",
  "defaultModelsExpandDepth": 1,
  "defaultModelExpandDepth": 1,
  "supportedSubmitMethods": [
    "get",
    "post",
    "put",
    "delete",
    "patch"
  ],
  "oauth2RedirectUrl": "https://api-docs.gl-vcci.com/oauth2-redirect.html",
  "requestInterceptor": null,
  "responseInterceptor": null
}
```

---

## Authentication Setup

### JWT Bearer Token

**OpenAPI Security Definition**:
```yaml
components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWT authentication token.
        Obtain token from /api/v1/auth/login endpoint.
        Format: "Bearer {token}"

security:
  - BearerAuth: []
```

**Swagger UI Authentication**:
```javascript
// In index.html or swagger-config.json
const ui = SwaggerUIBundle({
    // ... other config

    // Pre-authorize with token
    onComplete: () => {
        // Get token from localStorage or prompt user
        const token = localStorage.getItem('api_token');

        if (token) {
            ui.preauthorizeApiKey('BearerAuth', `Bearer ${token}`);
        }
    }
});
```

### API Key Authentication

**OpenAPI Security Definition**:
```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for authentication

security:
  - ApiKeyAuth: []
```

### OAuth 2.0

**OpenAPI Security Definition**:
```yaml
components:
  securitySchemes:
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://auth.gl-vcci.com/oauth/authorize
          tokenUrl: https://auth.gl-vcci.com/oauth/token
          scopes:
            read:transactions: Read transaction data
            write:transactions: Create and update transactions
            read:suppliers: Read supplier data
            write:suppliers: Manage suppliers

security:
  - OAuth2:
    - read:transactions
    - write:transactions
```

**Swagger UI OAuth Configuration**:
```javascript
const ui = SwaggerUIBundle({
    // ... other config

    oauth2RedirectUrl: 'https://api-docs.gl-vcci.com/oauth2-redirect.html',

    initOAuth: {
        clientId: 'swagger-ui-client',
        clientSecret: 'your-client-secret',
        realm: 'gl-vcci',
        appName: 'GL-VCCI API Documentation',
        scopeSeparator: ' ',
        scopes: 'read:transactions write:transactions',
        additionalQueryStringParams: {},
        useBasicAuthenticationWithAccessCodeGrant: false,
        usePkceWithAuthorizationCodeGrant: true
    }
});
```

---

## Customization

### Custom CSS

**Custom Styling**:
```html
<style>
    /* Custom brand colors */
    .swagger-ui .topbar {
        background-color: #1a5f3f;
    }

    .swagger-ui .topbar .download-url-wrapper input[type=text] {
        border: 2px solid #1a5f3f;
    }

    /* Custom button colors */
    .swagger-ui .btn.authorize {
        background-color: #1a5f3f;
        border-color: #1a5f3f;
    }

    .swagger-ui .btn.authorize svg {
        fill: #ffffff;
    }

    /* Custom operation colors */
    .swagger-ui .opblock.opblock-get {
        border-color: #1a5f3f;
        background: rgba(26, 95, 63, 0.1);
    }

    .swagger-ui .opblock.opblock-post {
        border-color: #2d8a5e;
        background: rgba(45, 138, 94, 0.1);
    }

    /* Custom logo */
    .swagger-ui .topbar-wrapper img {
        content: url('logo.png');
    }

    /* Hide Swagger UI logo */
    .swagger-ui .topbar-wrapper .link {
        display: none;
    }
</style>
```

### Custom Logo and Branding

**Add Custom Header**:
```javascript
const ui = SwaggerUIBundle({
    // ... other config

    // Custom layout with logo
    plugins: [
        () => ({
            components: {
                Logo: () => React.createElement('img', {
                    src: 'https://gl-vcci.com/logo.png',
                    alt: 'GL-VCCI',
                    height: 40
                })
            }
        })
    ]
});
```

### Custom Request/Response Interceptors

**Add Custom Headers**:
```javascript
const ui = SwaggerUIBundle({
    // ... other config

    requestInterceptor: (request) => {
        // Add custom headers
        request.headers['X-Client-Version'] = '1.0.0';
        request.headers['X-Request-ID'] = generateRequestId();

        // Log request
        console.log('API Request:', request);

        return request;
    },

    responseInterceptor: (response) => {
        // Log response
        console.log('API Response:', response);

        // Handle errors
        if (response.status >= 400) {
            console.error('API Error:', response);
        }

        return response;
    }
});

function generateRequestId() {
    return 'req-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}
```

---

## Production Best Practices

### Security

**1. Authentication**:
- Always require authentication for production APIs
- Use JWT or OAuth 2.0 for secure authentication
- Implement proper token expiration and refresh
- Store tokens securely (HttpOnly cookies, secure storage)

**2. HTTPS**:
```nginx
# Force HTTPS
server {
    listen 80;
    server_name api-docs.gl-vcci.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api-docs.gl-vcci.com;

    ssl_certificate /etc/nginx/ssl/gl-vcci.crt;
    ssl_certificate_key /etc/nginx/ssl/gl-vcci.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

**3. CORS Configuration**:
```yaml
# In OpenAPI spec
servers:
  - url: https://api.gl-vcci.com/api/v1
    description: Production server

# In Nginx
add_header 'Access-Control-Allow-Origin' 'https://api-docs.gl-vcci.com' always;
add_header 'Access-Control-Allow-Credentials' 'true' always;
add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type' always;
```

**4. Rate Limiting**:
```nginx
# Rate limit Swagger UI
limit_req_zone $binary_remote_addr zone=swagger_limit:10m rate=10r/s;

location /api/docs {
    limit_req zone=swagger_limit burst=20 nodelay;
    proxy_pass http://swagger-ui:8080;
}
```

### Performance

**1. Caching**:
```nginx
# Cache OpenAPI spec
location /api/v1/openapi.json {
    proxy_pass http://api:8000;
    proxy_cache_valid 200 1h;
    add_header X-Cache-Status $upstream_cache_status;
}

# Cache static assets
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

**2. CDN**:
- Serve Swagger UI static assets from CDN
- Cache OpenAPI spec at edge locations
- Use CloudFront or similar for global distribution

**3. Compression**:
```nginx
gzip on;
gzip_types application/json text/css application/javascript;
gzip_min_length 1000;
```

### Monitoring

**1. Access Logs**:
```nginx
log_format swagger_log '$remote_addr - $remote_user [$time_local] '
                       '"$request" $status $body_bytes_sent '
                       '"$http_referer" "$http_user_agent"';

access_log /var/log/nginx/swagger-access.log swagger_log;
```

**2. Health Checks**:
```yaml
# Kubernetes health checks
livenessProbe:
  httpGet:
    path: /
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

**3. Metrics**:
- Track API documentation page views
- Monitor API try-it-out usage
- Measure authentication success rates
- Track error rates

### Version Management

**Multiple API Versions**:
```json
{
  "urls": [
    {
      "name": "API v1 (Current)",
      "url": "https://api.gl-vcci.com/api/v1/openapi.json"
    },
    {
      "name": "API v2 (Beta)",
      "url": "https://api.gl-vcci.com/api/v2/openapi.json"
    },
    {
      "name": "API v1 (Legacy)",
      "url": "https://api.gl-vcci.com/api/v0/openapi.json"
    }
  ],
  "urls.primaryName": "API v1 (Current)"
}
```

---

## Troubleshooting

### Common Issues

#### Issue: CORS Errors

**Symptoms**:
```
Access to fetch at 'https://api.gl-vcci.com/api/v1/transactions'
from origin 'https://api-docs.gl-vcci.com' has been blocked by CORS policy
```

**Solutions**:
```nginx
# Add CORS headers to API
location /api/v1 {
    add_header 'Access-Control-Allow-Origin' 'https://api-docs.gl-vcci.com' always;
    add_header 'Access-Control-Allow-Credentials' 'true' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Authorization, Content-Type' always;

    if ($request_method = 'OPTIONS') {
        return 204;
    }

    proxy_pass http://api:8000;
}
```

#### Issue: OpenAPI Spec Not Loading

**Symptoms**:
- Blank Swagger UI page
- "Failed to load API definition" error

**Solutions**:
1. Check spec URL is accessible:
   ```bash
   curl https://api.gl-vcci.com/api/v1/openapi.json
   ```

2. Validate OpenAPI spec:
   ```bash
   # Install swagger-cli
   npm install -g @apidevtools/swagger-cli

   # Validate spec
   swagger-cli validate openapi.yaml
   ```

3. Check CORS headers:
   ```bash
   curl -I -H "Origin: https://api-docs.gl-vcci.com" \
     https://api.gl-vcci.com/api/v1/openapi.json
   ```

#### Issue: Authentication Not Working

**Symptoms**:
- 401 Unauthorized errors
- Token not being sent with requests

**Solutions**:
1. Check security definition in OpenAPI spec
2. Verify token format (Bearer prefix)
3. Check token expiration
4. Enable `persistAuthorization` in Swagger UI config

#### Issue: Try-It-Out Not Working

**Symptoms**:
- Execute button doesn't send requests
- Network errors in browser console

**Solutions**:
1. Check CORS configuration
2. Verify API server is accessible
3. Check browser network tab for errors
4. Enable `tryItOutEnabled` in config

### Debugging

**Enable Debug Logging**:
```javascript
const ui = SwaggerUIBundle({
    // ... other config

    // Enable debug logging
    onComplete: () => {
        console.log('Swagger UI loaded');
        console.log('Config:', ui.getConfigs());
    },

    requestInterceptor: (request) => {
        console.log('Request:', request);
        return request;
    },

    responseInterceptor: (response) => {
        console.log('Response:', response);
        return response;
    }
});
```

**Check Browser Console**:
- Look for JavaScript errors
- Check network requests in DevTools
- Verify API responses
- Check CORS and authentication headers

---

## Conclusion

Swagger UI provides a powerful, interactive way to document and test your GL-VCCI API. Follow this guide to deploy and configure Swagger UI for your environment.

### Quick Start Checklist

- [ ] Generate OpenAPI specification
- [ ] Choose deployment method (Docker/Kubernetes/Standalone)
- [ ] Configure authentication
- [ ] Customize branding and styling
- [ ] Set up CORS properly
- [ ] Enable HTTPS in production
- [ ] Test all endpoints
- [ ] Monitor usage and performance

### Resources

- **Swagger UI GitHub**: https://github.com/swagger-api/swagger-ui
- **OpenAPI Specification**: https://swagger.io/specification/
- **Swagger Editor**: https://editor.swagger.io/
- **API Documentation Best Practices**: https://swagger.io/resources/articles/documenting-apis/

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Maintained By**: GL-VCCI Platform Team
