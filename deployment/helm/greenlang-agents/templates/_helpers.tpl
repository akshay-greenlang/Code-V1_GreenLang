{{/*
GreenLang Agent Factory - Template Helpers
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang-agents.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "greenlang-agents.fullname" -}}
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
{{- define "greenlang-agents.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenlang-agents.labels" -}}
helm.sh/chart: {{ include "greenlang-agents.chart" . }}
{{ include "greenlang-agents.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- with .Values.global.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenlang-agents.selectorLabels" -}}
app.kubernetes.io/name: {{ include "greenlang-agents.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenlang-agents.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "greenlang-agents.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the namespace name
*/}}
{{- define "greenlang-agents.namespace" -}}
{{- if .Values.namespace.create }}
{{- default .Release.Namespace .Values.namespace.name }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Create image name with registry
*/}}
{{- define "greenlang-agents.image" -}}
{{- $registry := .Values.global.imageRegistry | default "ghcr.io" }}
{{- $repository := .Values.image.repository | default "greenlang" }}
{{- $tag := .Values.image.tag | default "latest" }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}

{{/*
Agent image helper
*/}}
{{- define "greenlang-agents.agentImage" -}}
{{- $registry := .root.Values.global.imageRegistry | default "ghcr.io" }}
{{- $repository := .agent.image.repository }}
{{- $tag := .agent.image.tag | default "latest" }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}

{{/*
Resource limits and requests
*/}}
{{- define "greenlang-agents.resources" -}}
{{- if .resources }}
resources:
  {{- toYaml .resources | nindent 2 }}
{{- else if .common.resources }}
resources:
  {{- toYaml .common.resources | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "greenlang-agents.podSecurityContext" -}}
{{- if .podSecurityContext }}
securityContext:
  {{- toYaml .podSecurityContext | nindent 2 }}
{{- else if .common.podSecurityContext }}
securityContext:
  {{- toYaml .common.podSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "greenlang-agents.containerSecurityContext" -}}
{{- if .containerSecurityContext }}
securityContext:
  {{- toYaml .containerSecurityContext | nindent 2 }}
{{- else if .common.containerSecurityContext }}
securityContext:
  {{- toYaml .common.containerSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Liveness probe
*/}}
{{- define "greenlang-agents.livenessProbe" -}}
{{- if .livenessProbe }}
livenessProbe:
  {{- toYaml .livenessProbe | nindent 2 }}
{{- else if .common.livenessProbe }}
livenessProbe:
  {{- toYaml .common.livenessProbe | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Readiness probe
*/}}
{{- define "greenlang-agents.readinessProbe" -}}
{{- if .readinessProbe }}
readinessProbe:
  {{- toYaml .readinessProbe | nindent 2 }}
{{- else if .common.readinessProbe }}
readinessProbe:
  {{- toYaml .common.readinessProbe | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Startup probe
*/}}
{{- define "greenlang-agents.startupProbe" -}}
{{- if .startupProbe }}
startupProbe:
  {{- toYaml .startupProbe | nindent 2 }}
{{- else if .common.startupProbe }}
startupProbe:
  {{- toYaml .common.startupProbe | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Agent component labels
*/}}
{{- define "greenlang-agents.componentLabels" -}}
{{- $component := .component | default "agent" }}
{{- include "greenlang-agents.labels" .root }}
app.kubernetes.io/component: {{ $component }}
{{- end }}

{{/*
Agent selector labels with component
*/}}
{{- define "greenlang-agents.componentSelectorLabels" -}}
{{- $component := .component | default "agent" }}
{{- include "greenlang-agents.selectorLabels" .root }}
app.kubernetes.io/component: {{ $component }}
{{- end }}

{{/*
Create annotations for deployments
*/}}
{{- define "greenlang-agents.deploymentAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "9090"
prometheus.io/path: "/metrics"
{{- if .Values.global.annotations }}
{{- toYaml .Values.global.annotations }}
{{- end }}
{{- end }}

{{/*
Create pod annotations
*/}}
{{- define "greenlang-agents.podAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "9090"
prometheus.io/path: "/metrics"
{{- if .checksumConfig }}
checksum/config: {{ .checksumConfig }}
{{- end }}
{{- if .Values.common.podAnnotations }}
{{- toYaml .Values.common.podAnnotations | nindent 0 }}
{{- end }}
{{- end }}

{{/*
Environment variables common to all agents
*/}}
{{- define "greenlang-agents.commonEnv" -}}
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
- name: ENVIRONMENT
  value: {{ .Values.global.environment | quote }}
- name: LOG_LEVEL
  value: {{ .Values.config.LOG_LEVEL | default "info" | quote }}
- name: LOG_FORMAT
  value: {{ .Values.config.LOG_FORMAT | default "json" | quote }}
- name: METRICS_ENABLED
  value: "true"
- name: METRICS_PORT
  value: "9090"
{{- end }}

{{/*
Common volume mounts for agents
*/}}
{{- define "greenlang-agents.commonVolumeMounts" -}}
- name: config
  mountPath: /app/config
  readOnly: true
- name: tmp
  mountPath: /tmp
- name: cache
  mountPath: /app/.cache
{{- end }}

{{/*
Common volumes for agents
*/}}
{{- define "greenlang-agents.commonVolumes" -}}
- name: config
  configMap:
    name: {{ include "greenlang-agents.fullname" . }}-config
- name: tmp
  emptyDir:
    sizeLimit: 100Mi
- name: cache
  emptyDir:
    sizeLimit: 500Mi
{{- end }}

{{/*
Pod affinity helper - zone spread
*/}}
{{- define "greenlang-agents.zoneAffinity" -}}
{{- $component := .component | default "agent" }}
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            {{- include "greenlang-agents.selectorLabels" .root | nindent 12 }}
            app.kubernetes.io/component: {{ $component }}
        topologyKey: topology.kubernetes.io/zone
{{- end }}

{{/*
Strict pod affinity helper - host spread (for critical agents)
*/}}
{{- define "greenlang-agents.strictAffinity" -}}
{{- $component := .component | default "agent" }}
podAntiAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          {{- include "greenlang-agents.selectorLabels" .root | nindent 10 }}
          app.kubernetes.io/component: {{ $component }}
      topologyKey: kubernetes.io/hostname
{{- end }}

{{/*
Topology spread constraints
*/}}
{{- define "greenlang-agents.topologySpreadConstraints" -}}
{{- $component := .component | default "agent" }}
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
        {{- include "greenlang-agents.selectorLabels" .root | nindent 8 }}
        app.kubernetes.io/component: {{ $component }}
{{- end }}

{{/*
Create image pull secrets list
*/}}
{{- define "greenlang-agents.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
  {{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
  {{- end }}
{{- end }}
{{- end }}

{{/*
Redis URL helper
*/}}
{{- define "greenlang-agents.redisUrl" -}}
{{- if .Values.externalRedis.enabled }}
{{- printf "redis://%s:%s" .Values.externalRedis.host (.Values.externalRedis.port | toString) }}
{{- else }}
{{- printf "redis://%s-redis-master:6379" .Release.Name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL URL helper
*/}}
{{- define "greenlang-agents.postgresqlUrl" -}}
{{- if .Values.externalPostgresql.enabled }}
{{- printf "postgresql://%s:%s/%s" .Values.externalPostgresql.host (.Values.externalPostgresql.port | toString) .Values.externalPostgresql.database }}
{{- else }}
{{- printf "postgresql://%s-postgresql:5432/greenlang" .Release.Name }}
{{- end }}
{{- end }}

{{/*
Priority class name helper
*/}}
{{- define "greenlang-agents.priorityClassName" -}}
{{- if .priorityClassName }}
priorityClassName: {{ .priorityClassName }}
{{- else if eq .component "eudr-compliance" }}
priorityClassName: high-priority
{{- end }}
{{- end }}

{{/*
Termination grace period helper
*/}}
{{- define "greenlang-agents.terminationGracePeriod" -}}
{{- if .terminationGracePeriodSeconds }}
terminationGracePeriodSeconds: {{ .terminationGracePeriodSeconds }}
{{- else if eq .component "worker" }}
terminationGracePeriodSeconds: 300
{{- else }}
terminationGracePeriodSeconds: 30
{{- end }}
{{- end }}
