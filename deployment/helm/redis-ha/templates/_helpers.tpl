{{/*
Expand the name of the chart.
*/}}
{{- define "redis-ha.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "redis-ha.fullname" -}}
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
{{- define "redis-ha.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "redis-ha.labels" -}}
helm.sh/chart: {{ include "redis-ha.chart" . }}
{{ include "redis-ha.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: database
app.kubernetes.io/part-of: {{ include "redis-ha.name" . }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "redis-ha.selectorLabels" -}}
app.kubernetes.io/name: {{ include "redis-ha.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "redis-ha.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "redis-ha.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for PodDisruptionBudget
*/}}
{{- define "redis-ha.pdb.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "policy/v1" }}
{{- print "policy/v1" }}
{{- else }}
{{- print "policy/v1beta1" }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for NetworkPolicy
*/}}
{{- define "redis-ha.networkPolicy.apiVersion" -}}
{{- print "networking.k8s.io/v1" }}
{{- end }}

{{/*
Return the secret name for Redis AUTH
*/}}
{{- define "redis-ha.secretName" -}}
{{- if .Values.auth.existingSecret }}
{{- .Values.auth.existingSecret }}
{{- else }}
{{- include "redis-ha.fullname" . }}
{{- end }}
{{- end }}

{{/*
Return Redis password
*/}}
{{- define "redis-ha.password" -}}
{{- if .Values.auth.password }}
{{- .Values.auth.password }}
{{- else }}
{{- randAlphaNum 32 }}
{{- end }}
{{- end }}

{{/*
Return Redis master service name
*/}}
{{- define "redis-ha.masterServiceName" -}}
{{- printf "%s-master" (include "redis-ha.fullname" .) }}
{{- end }}

{{/*
Return Redis read service name
*/}}
{{- define "redis-ha.readServiceName" -}}
{{- printf "%s-read" (include "redis-ha.fullname" .) }}
{{- end }}

{{/*
Return Redis sentinel service name
*/}}
{{- define "redis-ha.sentinelServiceName" -}}
{{- printf "%s-sentinel" (include "redis-ha.fullname" .) }}
{{- end }}

{{/*
Return Redis headless service name
*/}}
{{- define "redis-ha.headlessServiceName" -}}
{{- printf "%s-headless" (include "redis-ha.fullname" .) }}
{{- end }}

{{/*
Return the Redis configmap name
*/}}
{{- define "redis-ha.configmapName" -}}
{{- printf "%s-config" (include "redis-ha.fullname" .) }}
{{- end }}

{{/*
Return the Redis scripts configmap name
*/}}
{{- define "redis-ha.scriptsConfigmapName" -}}
{{- printf "%s-scripts" (include "redis-ha.fullname" .) }}
{{- end }}

{{/*
Pod labels
*/}}
{{- define "redis-ha.podLabels" -}}
{{ include "redis-ha.selectorLabels" . }}
{{- if .Values.podLabels }}
{{ toYaml .Values.podLabels }}
{{- end }}
{{- end }}

{{/*
Pod annotations
*/}}
{{- define "redis-ha.podAnnotations" -}}
checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- if .Values.podAnnotations }}
{{ toYaml .Values.podAnnotations }}
{{- end }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "redis-ha.containerSecurityContext" -}}
{{- if .Values.containerSecurityContext.enabled }}
securityContext:
  runAsUser: {{ .Values.containerSecurityContext.runAsUser }}
  runAsNonRoot: {{ .Values.containerSecurityContext.runAsNonRoot }}
  runAsGroup: {{ .Values.containerSecurityContext.runAsGroup }}
  allowPrivilegeEscalation: {{ .Values.containerSecurityContext.allowPrivilegeEscalation }}
  readOnlyRootFilesystem: {{ .Values.containerSecurityContext.readOnlyRootFilesystem }}
  capabilities:
    drop:
    {{- range .Values.containerSecurityContext.capabilities.drop }}
      - {{ . }}
    {{- end }}
{{- end }}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "redis-ha.podSecurityContext" -}}
{{- if .Values.securityContext.enabled }}
securityContext:
  fsGroup: {{ .Values.securityContext.fsGroup }}
  runAsUser: {{ .Values.securityContext.runAsUser }}
  runAsNonRoot: {{ .Values.securityContext.runAsNonRoot }}
  runAsGroup: {{ .Values.securityContext.runAsGroup }}
{{- end }}
{{- end }}

{{/*
Redis AUTH arguments
*/}}
{{- define "redis-ha.authArgs" -}}
{{- if .Values.auth.enabled }}
--requirepass $(REDIS_PASSWORD) --masterauth $(REDIS_PASSWORD)
{{- end }}
{{- end }}

{{/*
Return the number of sentinel replicas
*/}}
{{- define "redis-ha.sentinelReplicas" -}}
{{- if .Values.sentinel.enabled }}
{{- .Values.sentinel.replicas }}
{{- else }}
{{- 0 }}
{{- end }}
{{- end }}

{{/*
Affinity rules
*/}}
{{- define "redis-ha.affinity" -}}
{{- if .Values.affinity.enabled }}
affinity:
  podAntiAffinity:
    {{- if eq .Values.affinity.type "hard" }}
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            {{- include "redis-ha.selectorLabels" . | nindent 12 }}
        topologyKey: {{ .Values.affinity.topologyKey }}
      {{- if .Values.affinity.zoneAntiAffinity.enabled }}
      - labelSelector:
          matchLabels:
            {{- include "redis-ha.selectorLabels" . | nindent 12 }}
        topologyKey: {{ .Values.affinity.zoneAntiAffinity.topologyKey }}
      {{- end }}
    {{- else }}
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              {{- include "redis-ha.selectorLabels" . | nindent 14 }}
          topologyKey: {{ .Values.affinity.topologyKey }}
      {{- if .Values.affinity.zoneAntiAffinity.enabled }}
      - weight: 90
        podAffinityTerm:
          labelSelector:
            matchLabels:
              {{- include "redis-ha.selectorLabels" . | nindent 14 }}
          topologyKey: {{ .Values.affinity.zoneAntiAffinity.topologyKey }}
      {{- end }}
    {{- end }}
{{- end }}
{{- end }}
