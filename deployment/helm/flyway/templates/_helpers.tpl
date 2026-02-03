{{/*
=============================================================================
GreenLang Flyway Helm Chart Helper Templates
=============================================================================
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "flyway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "flyway.fullname" -}}
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
{{- define "flyway.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "flyway.labels" -}}
helm.sh/chart: {{ include "flyway.chart" . }}
{{ include "flyway.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- end }}

{{/*
Selector labels
*/}}
{{- define "flyway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "flyway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: flyway
component: database-migration
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "flyway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "flyway.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Generate JDBC URL for Flyway
*/}}
{{- define "flyway.jdbcUrl" -}}
{{- if .Values.database.jdbcUrl }}
{{- .Values.database.jdbcUrl }}
{{- else }}
{{- printf "jdbc:postgresql://%s:%s/%s?sslmode=%s" .Values.database.host (toString .Values.database.port) .Values.database.name .Values.database.sslMode }}
{{- end }}
{{- end }}

{{/*
Generate direct JDBC URL (bypassing PgBouncer for DDL operations)
*/}}
{{- define "flyway.directJdbcUrl" -}}
{{- if .Values.database.directJdbcUrl }}
{{- .Values.database.directJdbcUrl }}
{{- else }}
{{- printf "jdbc:postgresql://%s:%s/%s?sslmode=%s" .Values.database.directHost (toString .Values.database.directPort) .Values.database.name .Values.database.sslMode }}
{{- end }}
{{- end }}

{{/*
Secret name for database credentials
*/}}
{{- define "flyway.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-credentials" (include "flyway.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Secret name for AWS credentials
*/}}
{{- define "flyway.awsSecretName" -}}
{{- if .Values.aws.existingSecret }}
{{- .Values.aws.existingSecret }}
{{- else }}
{{- printf "%s-aws-credentials" (include "flyway.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Generate migration locations as comma-separated string
*/}}
{{- define "flyway.locations" -}}
{{- join "," .Values.migration.locations }}
{{- end }}

{{/*
Generate schemas as comma-separated string
*/}}
{{- define "flyway.schemas" -}}
{{- join "," .Values.migration.schemas }}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "flyway.podSecurityContext" -}}
runAsNonRoot: {{ .Values.securityContext.pod.runAsNonRoot }}
runAsUser: {{ .Values.securityContext.pod.runAsUser }}
runAsGroup: {{ .Values.securityContext.pod.runAsGroup }}
fsGroup: {{ .Values.securityContext.pod.fsGroup }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "flyway.containerSecurityContext" -}}
allowPrivilegeEscalation: {{ .Values.securityContext.container.allowPrivilegeEscalation }}
readOnlyRootFilesystem: {{ .Values.securityContext.container.readOnlyRootFilesystem }}
capabilities:
  drop:
    {{- toYaml .Values.securityContext.container.capabilities.drop | nindent 4 }}
{{- end }}

{{/*
Check if external secrets are enabled
*/}}
{{- define "flyway.useExternalSecrets" -}}
{{- if and .Values.secrets.externalSecrets .Values.secrets.externalSecrets.enabled }}
{{- "true" }}
{{- else }}
{{- "false" }}
{{- end }}
{{- end }}

{{/*
Annotations for Helm hooks
*/}}
{{- define "flyway.hookAnnotations" -}}
{{- if .Values.job.hooks.enabled }}
helm.sh/hook: pre-upgrade,pre-install
helm.sh/hook-weight: "{{ .Values.job.hooks.weight }}"
helm.sh/hook-delete-policy: {{ .Values.job.hooks.deletePolicy }}
{{- end }}
{{- end }}

{{/*
Environment variables common to all Flyway containers
*/}}
{{- define "flyway.commonEnv" -}}
- name: FLYWAY_URL
  value: {{ include "flyway.jdbcUrl" . | quote }}
- name: FLYWAY_USER
  valueFrom:
    secretKeyRef:
      name: {{ include "flyway.secretName" . }}
      key: {{ .Values.secrets.usernameKey }}
- name: FLYWAY_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "flyway.secretName" . }}
      key: {{ .Values.secrets.passwordKey }}
- name: ENVIRONMENT
  value: {{ .Values.global.environment | quote }}
- name: RETENTION_DAYS
  value: {{ .Values.timescaledb.retentionDays | quote }}
- name: COMPRESSION_AFTER_DAYS
  value: {{ .Values.timescaledb.compressionAfterDays | quote }}
- name: CHUNK_INTERVAL
  value: {{ .Values.timescaledb.chunkInterval | quote }}
{{- if .Values.aws.existingSecret }}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "flyway.awsSecretName" . }}
      key: access_key_id
      optional: true
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "flyway.awsSecretName" . }}
      key: secret_access_key
      optional: true
{{- end }}
- name: AWS_REGION
  value: {{ .Values.aws.region | quote }}
{{- end }}

{{/*
Volume mounts common to all Flyway containers
*/}}
{{- define "flyway.commonVolumeMounts" -}}
- name: flyway-config
  mountPath: /flyway/conf
  readOnly: true
- name: flyway-sql
  mountPath: /flyway/sql
  readOnly: true
{{- end }}

{{/*
Volumes common to all Flyway pods
*/}}
{{- define "flyway.commonVolumes" -}}
- name: flyway-config
  configMap:
    name: {{ include "flyway.fullname" . }}-config
- name: flyway-sql
  configMap:
    name: {{ .Values.sqlMigrations.configMap.name | default (printf "%s-sql" (include "flyway.fullname" .)) }}
{{- end }}
