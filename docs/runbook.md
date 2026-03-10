# Operations Runbook – AI-Driven Beamforming for Starlink in the Amazon

**System**: AI-Driven Beamforming Controller  
**Version**: 1.0  
**Owner**: Platform Operations Team  
**Last Updated**: 2026-03  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [System Initialisation](#2-system-initialisation)
3. [Dashboard Monitoring](#3-dashboard-monitoring)
4. [Alert Interpretation and Recommended Actions](#4-alert-interpretation-and-recommended-actions)
5. [Manual Fallback Procedures](#5-manual-fallback-procedures)
6. [Model Update – Canary Rollout](#6-model-update--canary-rollout)
7. [Model Registry Backup and Restore](#7-model-registry-backup-and-restore)
8. [Disaster Recovery](#8-disaster-recovery)
9. [Routine Maintenance](#9-routine-maintenance)
10. [Contact Escalation Matrix](#10-contact-escalation-matrix)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                   │
│                                                         │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │ BeamController  │───▶│  NullPhasedArrayDriver   │   │
│  │ (Deployment)    │    │  / EthernetDriver        │   │
│  └────────┬────────┘    └──────────────────────────┘   │
│           │                                             │
│           │ Prometheus metrics (:8080/metrics)          │
│           ▼                                             │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │  Prometheus     │───▶│  Grafana Dashboard       │   │
│  └─────────────────┘    └──────────────────────────┘   │
│                                                         │
│  ┌─────────────────┐                                    │
│  │  RetainCronJob  │  (weekly retrain, Sunday 03:00)   │
│  └─────────────────┘                                    │
│                                                         │
│  ┌─────────────────┐                                    │
│  │  Model Registry │  (PVC: /models)                   │
│  │  PVC            │                                    │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

**Data flow**:
1. `TelemetryDataset` / `RadarDataset` / `WeatherForecast` adapters stream live data.
2. `OnlineBeamController` or `HardwareBeamController` runs inference every step (~500 ms).
3. The `PhasedArrayDriver` issues beam commands to hardware.
4. Metrics are scraped by Prometheus and visualised in Grafana.
5. The `RetainCronJob` retrains the model weekly and promotes it to the registry.

---

## 2. System Initialisation

### 2.1 Prerequisites

- Kubernetes ≥ 1.27, Helm ≥ 3.12
- `kubectl` configured with cluster access
- Docker registry access for the controller image
- SpaceTrack credentials (stored in `beamforming-spacetrack-secret` K8s Secret)
- CPTEC API endpoint reachable from the cluster

### 2.2 Deploy with Helm

```bash
# First deployment
helm install beamforming ./helm/beamforming \
  --namespace beamforming \
  --create-namespace \
  --set image.tag=<VERSION> \
  --set production.enabled=true

# Verify pods are running
kubectl -n beamforming get pods

# Check controller logs
kubectl -n beamforming logs -l app.kubernetes.io/name=beamforming -f
```

### 2.3 Validate Startup

Within 60 seconds of pod start, you should see:

```
{"level":"INFO","event":"controller_ready","snr_threshold":5.0,"max_failures":3}
```

Run a quick smoke test:

```bash
kubectl -n beamforming exec deploy/beamforming -- \
  python scripts/field_test_hardware.py --driver null --steps 10
```

All 5 scenarios should report `✅ PASS`.

### 2.4 Apply Field-Test Calibration (if available)

```bash
# Copy calibration file into the pod
kubectl -n beamforming cp /tmp/calibration.json \
  $(kubectl -n beamforming get pods -l app=beamforming -o name | head -1):/tmp/

# Apply at runtime via the controller's REST endpoint (if exposed)
# or restart with the calibration environment variable:
kubectl -n beamforming set env deploy/beamforming \
  CALIBRATION_PATH=/tmp/calibration.json
```

---

## 3. Dashboard Monitoring

### 3.1 Key Grafana Panels

| Panel | Metric | Normal Range | Alert Threshold |
|---|---|---|---|
| SNR (dB) | `beam_snr_db` | 10–30 dB | < 5 dB → outage |
| Inference Latency P95 | `inference_latency_p95_ms` | < 50 ms | > 500 ms |
| Outage Rate (%) | `outage_rate_1m` | < 0.5 % | > 1 % |
| Fallback Rate (%) | `fallback_rate_1m` | < 1 % | > 5 % |
| Watchdog Alerts | `watchdog_alert_total` | 0 | > 0 |
| Model Version | `model_version` | Latest | Stale > 7 days |
| CPU Usage | `container_cpu_usage_seconds_total` | < 60 % | > 80 % |
| Memory Usage | `container_memory_working_set_bytes` | < 500 MB | > 800 MB |

### 3.2 Accessing the Dashboard

```bash
# Port-forward Grafana
kubectl -n monitoring port-forward svc/grafana 3000:3000

# Open in browser
open http://localhost:3000
# Dashboard: "AI Beamforming – Amazon LEO"
```

### 3.3 Checking Alert Status

```bash
# List active Prometheus alerts
kubectl -n monitoring exec -it deploy/prometheus -- \
  promtool query instant http://localhost:9090 'ALERTS{alertstate="firing"}'
```

---

## 4. Alert Interpretation and Recommended Actions

> **Regulatory Note**: The system enforces constraints derived from
> **Anatel Resolução 723/2020** (LEO satellite regulation for Brazil),
> **FCC Part 25** (Satellite Communications Services), and
> **ITU-R S.580-6 / SM.1448** (radiation diagrams and coordination).
> Run `python scripts/generate_compliance_report.py --verbose` to generate a
> current compliance snapshot.

### 4.1 `BeamOutageRateHigh`

**Trigger**: Outage rate > 1 % over 5 minutes.

**Likely causes**:
- Heavy rain event (Amazon afternoon convective cells).
- Satellite handover failure.
- Controller fallback loop.

**Actions**:
1. Check current SNR trend in Grafana (last 30 min).
2. Check weather data: `kubectl exec ... -- python -c "from data.weather_forecast import make_forecast; print(make_forecast('synthetic'))"`.
3. If rain-related: no action required; system should self-recover.
4. If persistent (> 30 min): escalate to L2; trigger manual fallback (§5).

### 4.2 `InferenceLatencyHigh`

**Trigger**: P95 inference latency > 500 ms.

**Likely causes**:
- CPU throttling (pod resource limit hit).
- Model checkpoint too large (slow load).
- Network timeout to hardware driver.

**Actions**:
1. Check CPU/memory in Grafana.
2. `kubectl -n beamforming top pod`.
3. If CPU: scale up – `kubectl -n beamforming scale deploy/beamforming --replicas=2`.
4. If model: trigger retrain with smaller architecture (see §6).

### 4.3 `FallbackRateHigh`

**Trigger**: Fallback rate > 5 % over 5 minutes.

**Likely causes**:
- SNR consistently near threshold (miscalibrated threshold).
- Hardware driver connectivity issue.
- Stale model producing poor actions.

**Actions**:
1. Check calibration: compare `snr_threshold_db` against current median SNR.
2. Re-run calibration: `python scripts/analyze_field_test.py --report <latest_ft_report>`.
3. Trigger model retrain if model is > 7 days old (see §6).

### 4.4 `WatchdogAlert`

**Trigger**: `max_failures` consecutive inference failures.

**Likely causes**:
- Hardware driver disconnected.
- Kubernetes pod OOMKilled.

**Actions**:
1. `kubectl -n beamforming describe pod <pod>` – check for OOMKilled.
2. Check hardware driver: `kubectl exec ... -- python -c "from hardware.phaser_driver import NullPhasedArrayDriver; d=NullPhasedArrayDriver(); d.connect(); print(d.read_telemetry())"`.
3. If driver disconnected: restart pod (`kubectl -n beamforming rollout restart deploy/beamforming`).

### 4.5 `ModelRegistryStale`

**Trigger**: Model version not updated for > 7 days.

**Actions**:
1. Check CronJob status: `kubectl -n beamforming get cronjob`.
2. Trigger manual retrain: `kubectl -n beamforming create job --from=cronjob/retrain-job retrain-manual-$(date +%s)`.
3. Monitor job logs: `kubectl -n beamforming logs job/retrain-manual-<ID> -f`.

---

## 5. Manual Fallback Procedures

### 5.1 Initiate Fallback

The controller falls back automatically when `max_failures` consecutive inference
steps fail.  To force fallback manually:

```bash
# Scale the controller to 0 – hardware driver enters safe boresight state
kubectl -n beamforming scale deploy/beamforming --replicas=0

# The NullPhasedArrayDriver reset() method returns to boresight:
kubectl exec <driver-pod> -- python -c \
  "from hardware.phaser_driver import NullPhasedArrayDriver; d=NullPhasedArrayDriver(); d.reset()"
```

### 5.2 Resume from Fallback

```bash
# Restore controller
kubectl -n beamforming scale deploy/beamforming --replicas=1

# Verify recovery
kubectl -n beamforming logs -l app=beamforming --tail=20
```

### 5.3 Override SNR Threshold

To temporarily raise the threshold during a known outage event:

```bash
kubectl -n beamforming set env deploy/beamforming SNR_THRESHOLD_DB=3.0
```

Revert after the event:

```bash
kubectl -n beamforming set env deploy/beamforming SNR_THRESHOLD_DB=5.0
```

---

## 6. Model Update – Canary Rollout

### 6.1 Prepare New Model Version

```bash
# Run retrain job manually
python scripts/retrain_job.py \
  --registry /models \
  --model-name ppo_amazon \
  --n-episodes 50 \
  --rounds 5
```

Note the new version directory (e.g., `/models/ppo_amazon/v3`).

### 6.2 Validate Before Rollout

```bash
# Run acceptance test with the new model
python scripts/acceptance_test.py \
  --steps 1000 \
  --output-json /tmp/acceptance_v3.json \
  --verbose
```

If `all_passed: true`, proceed.

### 6.3 Canary Rollout (10 % → 50 % → 100 %)

```bash
# Set new model version in values.yaml
# production.modelVersion: "v3"

# Deploy 10 % canary
helm upgrade beamforming ./helm/beamforming \
  --set production.modelVersion=v3 \
  --set canary.enabled=true \
  --set canary.weight=10

# Monitor for 1 hour; if metrics stable:
helm upgrade beamforming ./helm/beamforming \
  --set production.modelVersion=v3 \
  --set canary.weight=50

# Full rollout after 24 hours
helm upgrade beamforming ./helm/beamforming \
  --set production.modelVersion=v3 \
  --set canary.enabled=false
```

### 6.4 Rollback

```bash
helm rollback beamforming
# or explicitly:
helm upgrade beamforming ./helm/beamforming \
  --set production.modelVersion=v2
```

---

## 7. Model Registry Backup and Restore

### 7.1 Backup

The model registry PVC is mounted at `/models`.  Take a snapshot:

```bash
# Using kubectl cp (small registries)
kubectl -n beamforming cp \
  $(kubectl -n beamforming get pods -l app=beamforming -o name | head -1):/models \
  /backup/models-$(date +%Y%m%d)

# For large registries, use a Velero scheduled backup:
velero schedule create model-registry-daily \
  --schedule="0 2 * * *" \
  --include-namespaces beamforming \
  --include-resources persistentvolumeclaims
```

### 7.2 Restore

```bash
# Restore from local backup
kubectl -n beamforming cp \
  /backup/models-20260301 \
  $(kubectl -n beamforming get pods -l app=beamforming -o name | head -1):/models

# Verify registry integrity
kubectl -n beamforming exec deploy/beamforming -- \
  python -c "
import sys; sys.path.insert(0,'src')
from utils.model_registry import ModelRegistry
r = ModelRegistry('/models')
print('Latest:', r.latest_version('ppo_amazon'))
"
```

---

## 8. Disaster Recovery

### 8.1 Complete Pod Loss

```bash
# Re-deploy from Helm
helm upgrade --install beamforming ./helm/beamforming \
  --namespace beamforming \
  --set production.enabled=true

# Restore model registry (§7.2)
# Re-apply calibration (§2.4)
```

### 8.2 PVC Data Loss

If the model registry PVC is lost:

```bash
# 1. Restore from backup (§7.2)
# 2. If no backup available, trigger immediate retrain:
kubectl -n beamforming create job \
  --from=cronjob/retrain-job retrain-recovery-$(date +%s)

# 3. Monitor until model version v1 is promoted:
kubectl -n beamforming logs job/retrain-recovery-<ID> -f
```

### 8.3 Full Cluster Recovery

```bash
# 1. Provision new cluster
# 2. Restore secrets
kubectl -n beamforming create secret generic beamforming-spacetrack-secret \
  --from-literal=username=<ST_USER> \
  --from-literal=password=<ST_PASS>

# 3. Deploy Helm chart
helm install beamforming ./helm/beamforming ...

# 4. Restore model registry from off-site backup
# 5. Validate with field test:
python scripts/field_test_hardware.py --driver null --steps 10
```

**Recovery Time Objective (RTO)**: 30 minutes  
**Recovery Point Objective (RPO)**: 24 hours (last backup)

---

## 9. Routine Maintenance

| Task | Frequency | Owner | Script |
|---|---|---|---|
| Model retrain | Weekly (automated) | MLOps CronJob | `retrain_job.py` |
| Field-test validation | Monthly | Operations | `field_test_hardware.py` |
| Calibration update | After each field test | Operations | `analyze_field_test.py` |
| Compliance audit | Quarterly | Compliance | `generate_compliance_report.py` |
| Registry backup | Daily (automated) | Velero | – |
| Acceptance test | Before each major release | QA | `acceptance_test.py` |
| Runbook review | Bi-annual | Operations | This document |

---

## 10. Contact Escalation Matrix

| Level | Condition | Action |
|---|---|---|
| L1 – On-call | Single alert firing | Follow alert playbook (§4) |
| L2 – Senior Ops | Alert unresolved > 30 min | Page senior on-call; engage vendor |
| L3 – Engineering | System-wide outage; data loss | Page on-call engineer; open P1 incident |
| Regulatory | Compliance violation detected | Notify compliance officer; generate report |
