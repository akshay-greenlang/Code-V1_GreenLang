{{/*
Expand the name of the chart.
*/}}
{{- define "timescaledb-ha.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "timescaledb-ha.fullname" -}}
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
{{- define "timescaledb-ha.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "timescaledb-ha.labels" -}}
helm.sh/chart: {{ include "timescaledb-ha.chart" . }}
{{ include "timescaledb-ha.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: database
app.kubernetes.io/part-of: greenlang
{{- end }}

{{/*
Selector labels
*/}}
{{- define "timescaledb-ha.selectorLabels" -}}
app.kubernetes.io/name: {{ include "timescaledb-ha.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
cluster-name: {{ .Values.cluster.name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "timescaledb-ha.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "timescaledb-ha.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the secret containing credentials
*/}}
{{- define "timescaledb-ha.secretName" -}}
{{- printf "%s-credentials" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Create the name of the Patroni ConfigMap
*/}}
{{- define "timescaledb-ha.patroniConfigMapName" -}}
{{- printf "%s-patroni" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Create the name of the PostgreSQL ConfigMap
*/}}
{{- define "timescaledb-ha.postgresConfigMapName" -}}
{{- printf "%s-postgres" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Create the name of the primary service
*/}}
{{- define "timescaledb-ha.primaryServiceName" -}}
{{- printf "%s-primary" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Create the name of the replica service
*/}}
{{- define "timescaledb-ha.replicaServiceName" -}}
{{- printf "%s-replica" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Create the name of the headless service
*/}}
{{- define "timescaledb-ha.headlessServiceName" -}}
{{- printf "%s-headless" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Create the name of the PgBouncer service
*/}}
{{- define "timescaledb-ha.pgbouncerServiceName" -}}
{{- printf "%s-pgbouncer" (include "timescaledb-ha.fullname" .) }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "timescaledb-ha.image" -}}
{{- $registryName := .Values.global.imageRegistry | default "" -}}
{{- $repositoryName := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper PgBouncer image name
*/}}
{{- define "timescaledb-ha.pgbouncerImage" -}}
{{- $registryName := .Values.global.imageRegistry | default "" -}}
{{- $repositoryName := .Values.pgbouncer.image.repository -}}
{{- $tag := .Values.pgbouncer.image.tag -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper metrics exporter image name
*/}}
{{- define "timescaledb-ha.metricsImage" -}}
{{- $registryName := .Values.global.imageRegistry | default "" -}}
{{- $repositoryName := .Values.metrics.image.repository -}}
{{- $tag := .Values.metrics.image.tag -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper backup image name
*/}}
{{- define "timescaledb-ha.backupImage" -}}
{{- $registryName := .Values.global.imageRegistry | default "" -}}
{{- $repositoryName := .Values.backup.image.repository -}}
{{- $tag := .Values.backup.image.tag -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Create namespace
*/}}
{{- define "timescaledb-ha.namespace" -}}
{{- default .Release.Namespace .Values.cluster.namespace }}
{{- end }}

{{/*
Return the storage class
*/}}
{{- define "timescaledb-ha.storageClass" -}}
{{- if .Values.persistence.storageClass }}
{{- if (eq "-" .Values.persistence.storageClass) }}
storageClassName: ""
{{- else }}
storageClassName: {{ .Values.persistence.storageClass | quote }}
{{- end }}
{{- else if .Values.global.storageClass }}
storageClassName: {{ .Values.global.storageClass | quote }}
{{- end }}
{{- end }}

{{/*
Return WAL storage class
*/}}
{{- define "timescaledb-ha.walStorageClass" -}}
{{- if .Values.persistence.wal.storageClass }}
{{- if (eq "-" .Values.persistence.wal.storageClass) }}
storageClassName: ""
{{- else }}
storageClassName: {{ .Values.persistence.wal.storageClass | quote }}
{{- end }}
{{- else }}
{{- include "timescaledb-ha.storageClass" . }}
{{- end }}
{{- end }}

{{/*
Generate random password
*/}}
{{- define "timescaledb-ha.randomPassword" -}}
{{- randAlphaNum 32 -}}
{{- end }}

{{/*
Return the TLS secret name
*/}}
{{- define "timescaledb-ha.tlsSecretName" -}}
{{- if .Values.tls.existingSecret }}
{{- .Values.tls.existingSecret }}
{{- else }}
{{- printf "%s-tls" (include "timescaledb-ha.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Return pod anti-affinity rules
*/}}
{{- define "timescaledb-ha.podAntiAffinity" -}}
{{- if .Values.podAntiAffinity.enabled }}
{{- if eq .Values.podAntiAffinity.type "hard" }}
requiredDuringSchedulingIgnoredDuringExecution:
  - labelSelector:
      matchLabels:
        {{- include "timescaledb-ha.selectorLabels" . | nindent 8 }}
    topologyKey: {{ .Values.podAntiAffinity.topologyKey }}
{{- else }}
preferredDuringSchedulingIgnoredDuringExecution:
  - weight: 100
    podAffinityTerm:
      labelSelector:
        matchLabels:
          {{- include "timescaledb-ha.selectorLabels" . | nindent 10 }}
      topologyKey: {{ .Values.podAntiAffinity.topologyKey }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return topology spread constraints
*/}}
{{- define "timescaledb-ha.topologySpreadConstraints" -}}
{{- if .Values.topologySpreadConstraints.enabled }}
- maxSkew: {{ .Values.topologySpreadConstraints.maxSkew }}
  topologyKey: {{ .Values.topologySpreadConstraints.topologyKey }}
  whenUnsatisfiable: {{ .Values.topologySpreadConstraints.whenUnsatisfiable }}
  labelSelector:
    matchLabels:
      {{- include "timescaledb-ha.selectorLabels" . | nindent 6 }}
{{- end }}
{{- end }}
