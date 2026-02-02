{{/*
GL-017_Condensync Helm Chart - Template Helpers
Condenser Optimization Agent
Version: 1.0.0
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "gl-017-condensync.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "gl-017-condensync.fullname" -}}
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
{{- define "gl-017-condensync.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "gl-017-condensync.labels" -}}
helm.sh/chart: {{ include "gl-017-condensync.chart" . }}
{{ include "gl-017-condensync.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
app.kubernetes.io/component: condenser-optimization
greenlang.io/agent-id: {{ .Values.agent.id | quote }}
greenlang.io/agent-group: {{ .Values.agent.group | quote }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "gl-017-condensync.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gl-017-condensync.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "gl-017-condensync.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "gl-017-condensync.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "gl-017-condensync.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- $digest := .Values.image.digest | default "" -}}
{{- if $registry }}
{{- if $digest }}
{{- printf "%s/%s@%s" $registry $repository $digest }}
{{- else }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}
{{- else }}
{{- if $digest }}
{{- printf "%s@%s" $repository $digest }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create ConfigMap name
*/}}
{{- define "gl-017-condensync.configMapName" -}}
{{- printf "%s-config" (include "gl-017-condensync.fullname" .) }}
{{- end }}

{{/*
Create Secret name
*/}}
{{- define "gl-017-condensync.secretName" -}}
{{- printf "%s-secrets" (include "gl-017-condensync.fullname" .) }}
{{- end }}

{{/*
Create PDB name
*/}}
{{- define "gl-017-condensync.pdbName" -}}
{{- printf "%s-pdb" (include "gl-017-condensync.fullname" .) }}
{{- end }}

{{/*
Create HPA name
*/}}
{{- define "gl-017-condensync.hpaName" -}}
{{- printf "%s-hpa" (include "gl-017-condensync.fullname" .) }}
{{- end }}

{{/*
Create Ingress host
*/}}
{{- define "gl-017-condensync.ingressHost" -}}
{{- printf "%s.greenlang.io" .Values.agent.id }}
{{- end }}

{{/*
Return the proper image pull secrets
*/}}
{{- define "gl-017-condensync.imagePullSecrets" -}}
{{- $pullSecrets := list }}
{{- if .Values.global.imagePullSecrets }}
{{- range .Values.global.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if .Values.imagePullSecrets }}
{{- range .Values.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if $pullSecrets }}
imagePullSecrets:
{{- range $pullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper storage class
*/}}
{{- define "gl-017-condensync.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else if .Values.persistence.storageClass }}
{{- .Values.persistence.storageClass }}
{{- end }}
{{- end }}

{{/*
Generate environment variables from values
*/}}
{{- define "gl-017-condensync.env" -}}
{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- if .Values.extraEnv }}
{{- toYaml .Values.extraEnv }}
{{- end }}
{{- end }}

{{/*
Generate container ports
*/}}
{{- define "gl-017-condensync.ports" -}}
{{- range .Values.ports }}
- name: {{ .name }}
  containerPort: {{ .containerPort }}
  protocol: {{ .protocol | default "TCP" }}
{{- end }}
{{- end }}

{{/*
Check if we should create RBAC resources
*/}}
{{- define "gl-017-condensync.createRbac" -}}
{{- and .Values.serviceAccount.create .Values.rbac.create }}
{{- end }}

{{/*
Return checksum of configmap for deployment annotation
*/}}
{{- define "gl-017-condensync.configChecksum" -}}
{{- if .Values.configMap.enabled }}
checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- end }}
{{- end }}

{{/*
Return checksum of secret for deployment annotation
*/}}
{{- define "gl-017-condensync.secretChecksum" -}}
{{- if .Values.secrets.enabled }}
checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- end }}
{{- end }}

{{/*
Generate condenser-specific labels
*/}}
{{- define "gl-017-condensync.condenserLabels" -}}
greenlang.io/optimization-type: condenser
greenlang.io/thermal-subsystem: cooling
{{- end }}
