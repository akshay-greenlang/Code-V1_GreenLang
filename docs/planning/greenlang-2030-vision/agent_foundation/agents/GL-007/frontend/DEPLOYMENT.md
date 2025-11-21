# GL-007 Frontend - Deployment Guide

Complete guide for deploying the GL-007 Furnace Performance Monitor frontend to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build Configuration](#build-configuration)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [CDN Configuration](#cdn-configuration)
6. [Environment Configuration](#environment-configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Performance Optimization](#performance-optimization)
9. [Security Hardening](#security-hardening)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

- Node.js >= 18.0.0
- Docker >= 20.10 (for containerized deployment)
- Kubernetes >= 1.24 (for K8s deployment)
- SSL/TLS certificates
- CDN account (CloudFlare, AWS CloudFront, or Akamai)

## Build Configuration

### Production Build

```bash
# Install dependencies
npm ci --production=false

# Run tests
npm run test

# Lint code
npm run lint

# Create production build
npm run build

# Output will be in ./dist directory
```

### Build Optimization

The build process includes:
- **Code Splitting**: Automatic chunking by route and vendor
- **Tree Shaking**: Removal of unused code
- **Minification**: JavaScript and CSS compression
- **Asset Optimization**: Image compression and lazy loading
- **Source Maps**: For debugging (optional in production)

Target metrics:
- Initial bundle: < 500KB gzipped
- Load time: < 2 seconds on 3G
- Lighthouse score: > 90

### Build Verification

```bash
# Verify build output
npm run preview

# Run production build locally on http://localhost:4173
# Test all features before deployment
```

## Docker Deployment

### Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --production=false

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

### Nginx Configuration

Create `nginx.conf`:

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript
               application/x-javascript application/xml+rss
               application/javascript application/json;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # React Router - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }

    # API proxy
    location /api/ {
        proxy_pass https://api.greenlang.io/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket proxy
    location /ws/ {
        proxy_pass https://ws.greenlang.io/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Build and Run Docker Container

```bash
# Build image
docker build -t gl-007-frontend:latest .

# Run container
docker run -d \
  --name gl-007-frontend \
  -p 80:80 \
  -e VITE_API_URL=https://api.greenlang.io/v1 \
  -e VITE_WS_URL=wss://ws.greenlang.io \
  gl-007-frontend:latest

# Check logs
docker logs -f gl-007-frontend

# Health check
curl http://localhost/health
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    environment:
      - VITE_API_URL=https://api.greenlang.io/v1
      - VITE_WS_URL=wss://ws.greenlang.io
      - VITE_API_KEY=${API_KEY}
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
```

## Kubernetes Deployment

### Deployment Manifest

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-007-frontend
  namespace: greenlang
  labels:
    app: gl-007-frontend
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-007-frontend
  template:
    metadata:
      labels:
        app: gl-007-frontend
        version: v1
    spec:
      containers:
      - name: frontend
        image: greenlang/gl-007-frontend:latest
        ports:
        - containerPort: 80
          name: http
        env:
        - name: VITE_API_URL
          valueFrom:
            configMapKeyRef:
              name: gl-007-config
              key: api-url
        - name: VITE_WS_URL
          valueFrom:
            configMapKeyRef:
              name: gl-007-config
              key: ws-url
        - name: VITE_API_KEY
          valueFrom:
            secretKeyRef:
              name: gl-007-secrets
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: gl-007-frontend
  namespace: greenlang
spec:
  selector:
    app: gl-007-frontend
  ports:
  - name: http
    port: 80
    targetPort: 80
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-007-config
  namespace: greenlang
data:
  api-url: "https://api.greenlang.io/v1"
  ws-url: "wss://ws.greenlang.io"
---
apiVersion: v1
kind: Secret
metadata:
  name: gl-007-secrets
  namespace: greenlang
type: Opaque
stringData:
  api-key: "your-api-key-here"
```

### Horizontal Pod Autoscaler

Create `k8s-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-007-frontend-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-007-frontend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Ingress Configuration

Create `k8s-ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gl-007-frontend
  namespace: greenlang
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - furnace.greenlang.io
    secretName: gl-007-tls
  rules:
  - host: furnace.greenlang.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: gl-007-frontend
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace greenlang

# Apply configurations
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-hpa.yaml
kubectl apply -f k8s-ingress.yaml

# Verify deployment
kubectl get pods -n greenlang
kubectl get svc -n greenlang
kubectl get ingress -n greenlang

# Check logs
kubectl logs -f deployment/gl-007-frontend -n greenlang

# Scale manually if needed
kubectl scale deployment/gl-007-frontend --replicas=5 -n greenlang
```

## CDN Configuration

### CloudFlare Configuration

1. **DNS Setup**:
   - Add A record: `furnace.greenlang.io` → Load Balancer IP
   - Enable CloudFlare proxy (orange cloud)

2. **Caching Rules**:
   ```
   Cache Level: Standard
   Browser TTL: 4 hours
   Edge TTL: 1 day

   Always Cache:
   - *.js
   - *.css
   - *.png, *.jpg, *.svg
   - *.woff, *.woff2, *.ttf

   Bypass Cache:
   - /api/*
   - /ws/*
   - index.html
   ```

3. **Performance Settings**:
   - Enable Auto Minify (JS, CSS, HTML)
   - Enable Brotli compression
   - Enable HTTP/2 and HTTP/3
   - Enable 0-RTT Connection Resumption

4. **Security Settings**:
   - SSL/TLS: Full (strict)
   - Min TLS Version: 1.2
   - Enable HSTS
   - Enable WAF rules
   - Rate limiting: 100 req/min per IP

### AWS CloudFront Configuration

```javascript
// CloudFront distribution config
{
  "Origins": [
    {
      "DomainName": "gl-007-frontend.s3.amazonaws.com",
      "S3OriginConfig": {
        "OriginAccessIdentity": "origin-access-identity/cloudfront/..."
      }
    }
  ],
  "DefaultCacheBehavior": {
    "ViewerProtocolPolicy": "redirect-to-https",
    "Compress": true,
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6", // CachingOptimized
    "TargetOriginId": "S3-gl-007-frontend"
  },
  "CacheBehaviors": [
    {
      "PathPattern": "/api/*",
      "ViewerProtocolPolicy": "https-only",
      "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad" // CachingDisabled
    }
  ]
}
```

## Environment Configuration

### Production Environment Variables

```bash
# API Configuration
VITE_API_URL=https://api.greenlang.io/v1
VITE_WS_URL=wss://ws.greenlang.io
VITE_API_KEY=prod_api_key_xxxxx

# Application
VITE_APP_NAME=GL-007 Furnace Monitor
VITE_APP_VERSION=1.0.0
VITE_APP_ENV=production

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_THERMAL_IMAGING=true
VITE_ENABLE_PREDICTIVE_MAINTENANCE=true

# Authentication
VITE_AUTH_ENABLED=true
VITE_AUTH_PROVIDER=oauth2
VITE_AUTH_DOMAIN=auth.greenlang.io

# Monitoring
VITE_SENTRY_DSN=https://...@sentry.io/...
VITE_GA_TRACKING_ID=UA-XXXXXXXXX-X
```

## Monitoring & Logging

### Application Monitoring

**Sentry Integration**:

```typescript
// src/main.tsx
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  environment: import.meta.env.VITE_APP_ENV,
  tracesSampleRate: 0.1,
  beforeSend(event) {
    // Filter sensitive data
    return event;
  }
});
```

**Google Analytics**:

```typescript
// src/services/analytics.ts
import ReactGA from 'react-ga4';

ReactGA.initialize(import.meta.env.VITE_GA_TRACKING_ID);

export const trackPageView = (path: string) => {
  ReactGA.send({ hitType: "pageview", page: path });
};
```

### Infrastructure Monitoring

**Prometheus Metrics**:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gl-007-frontend'
    static_configs:
      - targets: ['frontend:9090']
```

**Grafana Dashboard**:
- Request rate and latency
- Error rate and types
- Active users and sessions
- Resource usage (CPU, memory)
- Cache hit rates

## Performance Optimization

### Production Checklist

- [ ] Enable gzip/brotli compression
- [ ] Configure CDN caching
- [ ] Implement service worker for offline support
- [ ] Enable HTTP/2 or HTTP/3
- [ ] Optimize images (WebP format, lazy loading)
- [ ] Remove console.log statements
- [ ] Enable production mode React optimizations
- [ ] Configure proper cache headers
- [ ] Minimize third-party scripts
- [ ] Use CDN for static assets

### Performance Metrics Targets

| Metric | Target | Current |
|--------|--------|---------|
| First Contentful Paint | < 1.5s | - |
| Largest Contentful Paint | < 2.5s | - |
| Time to Interactive | < 3.5s | - |
| Cumulative Layout Shift | < 0.1 | - |
| Bundle Size (gzipped) | < 500KB | - |

## Security Hardening

### Security Headers

```nginx
# Add to nginx.conf
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

### Content Security Policy

```nginx
add_header Content-Security-Policy "
  default-src 'self';
  script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.greenlang.io;
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self' data:;
  connect-src 'self' https://api.greenlang.io wss://ws.greenlang.io;
  frame-ancestors 'none';
  base-uri 'self';
  form-action 'self';
" always;
```

### Secrets Management

**Never commit**:
- API keys
- Authentication tokens
- Private certificates
- Environment-specific configs

Use:
- Kubernetes Secrets
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault

## Troubleshooting

### Common Issues

**Issue**: Blank page after deployment
**Solution**:
- Check browser console for errors
- Verify API_URL and WS_URL are correct
- Check CORS configuration on backend
- Verify routing configuration (try_files in nginx)

**Issue**: WebSocket connection fails
**Solution**:
- Check WebSocket proxy configuration
- Verify SSL/TLS certificates
- Check firewall rules for WebSocket port
- Test with ws:// (non-SSL) locally first

**Issue**: Slow initial load
**Solution**:
- Enable compression (gzip/brotli)
- Configure CDN caching
- Optimize bundle size
- Enable code splitting
- Use lazy loading for heavy components

**Issue**: High memory usage
**Solution**:
- Check for memory leaks in components
- Implement proper cleanup in useEffect
- Optimize WebSocket message handling
- Use React.memo for expensive renders

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
VITE_DEBUG=true npm run build

# Or in .env
VITE_DEBUG=true
```

### Health Check Endpoints

```bash
# Application health
curl https://furnace.greenlang.io/health

# API health
curl https://api.greenlang.io/health

# WebSocket health
wscat -c wss://ws.greenlang.io/health
```

## Rollback Procedure

If deployment fails:

```bash
# Docker
docker pull greenlang/gl-007-frontend:previous-tag
docker service update --image greenlang/gl-007-frontend:previous-tag frontend

# Kubernetes
kubectl rollout undo deployment/gl-007-frontend -n greenlang
kubectl rollout status deployment/gl-007-frontend -n greenlang

# Verify
kubectl get pods -n greenlang
```

## Support

For deployment issues:
- DevOps Team: devops@greenlang.io
- Documentation: https://docs.greenlang.io/deployment
- Status Page: https://status.greenlang.io
