# ==============================================================================
# GL-VCCI Frontend - React SPA Dockerfile
# Multi-stage build: Build with Node.js, serve with Nginx
# ==============================================================================

# ==============================================================================
# Stage 1: Build React Application
# ==============================================================================
FROM node:18-alpine as builder

# Metadata
LABEL maintainer="GL-VCCI Team <support@greenlang.ai>"

# Set working directory
WORKDIR /build

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --production=false

# Copy source code
COPY . .

# Build application
ENV NODE_ENV=production
ENV GENERATE_SOURCEMAP=false
ENV INLINE_RUNTIME_CHUNK=false

RUN npm run build

# ==============================================================================
# Stage 2: Production Nginx Server
# ==============================================================================
FROM nginx:1.25-alpine

# Metadata
LABEL maintainer="GL-VCCI Team <support@greenlang.ai>" \
      version="2.0.0" \
      description="GL-VCCI Scope 3 Platform - Frontend (React SPA)"

# Install curl for healthcheck
RUN apk add --no-cache curl

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built React app from builder
COPY --from=builder /build/build /usr/share/nginx/html

# Create nginx user and set permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chmod -R 755 /usr/share/nginx/html

# Create directories for nginx to write
RUN mkdir -p /var/run && \
    chown -R nginx:nginx /var/run

# Switch to non-root user
USER nginx

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
