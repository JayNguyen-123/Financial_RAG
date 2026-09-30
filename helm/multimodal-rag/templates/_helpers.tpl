{{- define "rag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "rag.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/* Selector labels for a component: include "rag.selectorLabels" (dict "ctx" . "component" "api") */}}
{{- define "rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rag.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "rag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "rag.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "rag.secretName" -}}
{{- default (printf "%s-secrets" (include "rag.fullname" .)) .Values.secrets.existingSecret -}}
{{- end -}}

{{- define "rag.claimName" -}}
{{- default (printf "%s-data" (include "rag.fullname" .)) .Values.persistence.existingClaim -}}
{{- end -}}

{{- define "rag.redisAddress" -}}
{{- if .Values.worker.autoscaling.keda.redisAddress -}}
{{- .Values.worker.autoscaling.keda.redisAddress -}}
{{- else -}}
{{- printf "%s-redis.%s.svc.cluster.local:6379" (include "rag.fullname" .) .Release.Namespace -}}
{{- end -}}
{{- end -}}

{{- define "rag.redisUrl" -}}
{{- if .Values.config.REDIS_URL -}}
{{- .Values.config.REDIS_URL -}}
{{- else if .Values.redis.enabled -}}
{{- printf "redis://%s-redis:6379/0" (include "rag.fullname" .) -}}
{{- else -}}
{{- fail "Set config.REDIS_URL or enable redis.enabled" -}}
{{- end -}}
{{- end -}}

{{/* Env + volume wiring shared by API and worker pods */}}
{{- define "rag.appEnv" -}}
envFrom:
  - configMapRef:
      name: {{ include "rag.fullname" . }}-config
  - secretRef:
      name: {{ include "rag.secretName" . }}
env:
  - name: STORAGE_PATH
    value: /app/data/store
  - name: SCRATCH_DIR
    value: /scratch
  - name: MPLCONFIGDIR          # writable cache dir under readOnlyRootFilesystem
    value: /tmp
{{- end -}}

{{- define "rag.appVolumeMounts" -}}
{{- if eq .Values.storage.backend "local" }}
- name: data
  mountPath: /app/data
{{- end }}
- name: scratch
  mountPath: /scratch
- name: tmp
  mountPath: /tmp
{{- end -}}

{{- define "rag.appVolumes" -}}
{{- if eq .Values.storage.backend "local" }}
- name: data
  persistentVolumeClaim:
    claimName: {{ include "rag.claimName" . }}
{{- end }}
- name: scratch
  emptyDir:
    sizeLimit: {{ .Values.storage.scratchSizeLimit }}
- name: tmp
  emptyDir: {}
{{- end -}}

{{- define "rag.configChecksums" -}}
checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- end -}}
