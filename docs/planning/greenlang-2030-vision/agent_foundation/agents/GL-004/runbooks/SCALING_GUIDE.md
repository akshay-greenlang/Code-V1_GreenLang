# GL-004 - Scaling Guide
## Horizontal Scaling
Scale replicas based on load:
- 1-3 burners: 3 replicas
- 4-10 burners: 5 replicas  
- 11-20 burners: 8 replicas
- 20+ burners: 10+ replicas

Commands:
\`\`\`bash
kubectl scale deployment/gl-004 --replicas=5 -n greenlang
kubectl autoscale deployment/gl-004 --min=3 --max=10 --cpu-percent=70 -n greenlang
\`\`\`

## Vertical Scaling
Increase resources in deployment.yaml:
\`\`\`yaml
resources:
  requests: {cpu: 1000m, memory: 1Gi}
  limits: {cpu: 2000m, memory: 2Gi}
\`\`\`

## Database Scaling
Increase connection pool: DB_POOL_SIZE=20, DB_MAX_OVERFLOW=40

## Performance Tuning
- Reduce optimization frequency for stability
- Increase cache TTL for frequently accessed data
- Enable connection pooling
- Use read replicas for queries
