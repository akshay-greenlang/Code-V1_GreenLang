{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "greenlang.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "greenlang.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenlang.labels" -}}
helm.sh/chart: {{ include "greenlang.chart" . }}
{{ include "greenlang.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
environment: {{ .Values.global.environment }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenlang.selectorLabels" -}}
app.kubernetes.io/name: {{ include "greenlang.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenlang.serviceAccountName" -}}
{{- if .Values.rbac.serviceAccount.create }}
{{- default (include "greenlang.fullname" .) .Values.rbac.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.rbac.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Executor labels
*/}}
{{- define "greenlang.executor.labels" -}}
{{ include "greenlang.labels" . }}
app: greenlang
component: executor
{{- end }}

{{/*
Worker labels
*/}}
{{- define "greenlang.worker.labels" -}}
{{ include "greenlang.labels" . }}
app: greenlang
component: worker
{{- end }}

{{/*
Redis labels
*/}}
{{- define "greenlang.redis.labels" -}}
{{ include "greenlang.labels" . }}
app: redis-sentinel
component: cache
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "greenlang.imagePullSecrets" -}}
{{- if .Values.image.pullSecrets }}
imagePullSecrets:
{{- range .Values.image.pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "greenlang.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Return the proper executor image name
*/}}
{{- define "greenlang.executor.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.executor.image.repository -}}
{{- $tag := .Values.executor.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Return the proper worker image name
*/}}
{{- define "greenlang.worker.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.worker.image.repository -}}
{{- $tag := .Values.worker.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Return the storage class name
*/}}
{{- define "greenlang.storageClass" -}}
{{- .Values.global.storageClass | default "fast-ssd" -}}
{{- end }}

{{/*
Return the namespace
*/}}
{{- define "greenlang.namespace" -}}
{{- .Values.namespace.name | default .Release.Namespace -}}
{{- end }}

{{/*
Create deployment annotations
*/}}
{{- define "greenlang.deploymentAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "9090"
prometheus.io/path: "/metrics"
{{- end }}

{{/*
Create pod annotations
*/}}
{{- define "greenlang.podAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "9090"
prometheus.io/path: "/metrics"
{{- if .checksumConfig }}
checksum/config: {{ .checksumConfig }}
{{- end }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "greenlang.commonEnv" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
{{- end }}

{{/*
Pod security context
*/}}
{{- define "greenlang.podSecurityContext" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
{{- end }}

{{/*
Container security context
*/}}
{{- define "greenlang.containerSecurityContext" -}}
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  capabilities:
    drop:
      - ALL
{{- end }}

{{/*
Liveness probe
*/}}
{{- define "greenlang.livenessProbe" -}}
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
{{- end }}

{{/*
Readiness probe
*/}}
{{- define "greenlang.readinessProbe" -}}
readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  successThreshold: 1
{{- end }}

{{/*
Startup probe
*/}}
{{- define "greenlang.startupProbe" -}}
startupProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 30
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "greenlang.imagePullSecrets" -}}
{{- if .Values.image.pullSecrets }}
imagePullSecrets:
  {{- range .Values.image.pullSecrets }}
  - name: {{ . }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Common volume mounts
*/}}
{{- define "greenlang.commonVolumeMounts" -}}
- name: config
  mountPath: /app/config
  readOnly: true
- name: tmp
  mountPath: /tmp
- name: cache
  mountPath: /app/.cache
{{- end }}

{{/*
Common volumes
*/}}
{{- define "greenlang.commonVolumes" -}}
- name: config
  configMap:
    name: {{ include "greenlang.fullname" . }}-config
- name: tmp
  emptyDir:
    sizeLimit: 100Mi
- name: cache
  emptyDir:
    sizeLimit: 500Mi
{{- end }}

{{/*
Redis host helper
*/}}
{{- define "greenlang.redisHost" -}}
{{- if .Values.externalRedis.enabled }}
{{- .Values.externalRedis.host }}
{{- else }}
{{- printf "%s-redis-master" .Release.Name }}
{{- end }}
{{- end }}

{{/*
Redis port helper
*/}}
{{- define "greenlang.redisPort" -}}
{{- if .Values.externalRedis.enabled }}
{{- .Values.externalRedis.port | default 6379 }}
{{- else }}
{{- 6379 }}
{{- end }}
{{- end }}

{{/*
PostgreSQL host helper
*/}}
{{- define "greenlang.postgresqlHost" -}}
{{- if .Values.externalDatabase.enabled }}
{{- .Values.externalDatabase.host }}
{{- else }}
{{- printf "%s-postgresql" .Release.Name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL port helper
*/}}
{{- define "greenlang.postgresqlPort" -}}
{{- if .Values.externalDatabase.enabled }}
{{- .Values.externalDatabase.port | default 5432 }}
{{- else }}
{{- 5432 }}
{{- end }}
{{- end }}

{{/*
PostgreSQL database helper
*/}}
{{- define "greenlang.postgresqlDatabase" -}}
{{- if .Values.externalDatabase.enabled }}
{{- .Values.externalDatabase.database | default "greenlang" }}
{{- else if .Values.postgresql.auth.database }}
{{- .Values.postgresql.auth.database }}
{{- else }}
{{- "greenlang" }}
{{- end }}
{{- end }}

{{/*
Pod anti-affinity helper
*/}}
{{- define "greenlang.podAntiAffinity" -}}
{{- if eq .Values.affinity.podAntiAffinity "required" }}
podAntiAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          {{- include "greenlang.selectorLabels" . | nindent 10 }}
      topologyKey: kubernetes.io/hostname
{{- else if eq .Values.affinity.podAntiAffinity "preferred" }}
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            {{- include "greenlang.selectorLabels" . | nindent 12 }}
        topologyKey: kubernetes.io/hostname
{{- end }}
{{- end }}

{{/*
Topology spread constraints helper
*/}}
{{- define "greenlang.topologySpreadConstraints" -}}
{{- if .Values.topologySpreadConstraints.enabled }}
topologySpreadConstraints:
  - maxSkew: {{ .Values.topologySpreadConstraints.maxSkew | default 1 }}
    topologyKey: {{ .Values.topologySpreadConstraints.topologyKey | default "topology.kubernetes.io/zone" }}
    whenUnsatisfiable: {{ .Values.topologySpreadConstraints.whenUnsatisfiable | default "DoNotSchedule" }}
    labelSelector:
      matchLabels:
        {{- include "greenlang.selectorLabels" . | nindent 8 }}
{{- end }}
{{- end }}
