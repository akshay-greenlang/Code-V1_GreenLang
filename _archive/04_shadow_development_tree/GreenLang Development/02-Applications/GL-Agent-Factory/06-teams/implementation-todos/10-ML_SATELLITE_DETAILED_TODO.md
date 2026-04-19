# ML & Satellite Deforestation Detection System - Detailed Implementation TODO

**Team:** ML/Satellite Analysis Team
**Version:** 1.0.0
**Date:** 2025-12-04
**Priority:** P2
**Total Duration:** 32 weeks (Phases 0-3)
**Total Tasks:** 140+ tasks

---

## Executive Summary

This document provides the detailed implementation plan for the ML and Satellite Deforestation Detection System for EUDR (EU Deforestation Regulation) compliance. The system will process satellite imagery (Sentinel-2, Landsat, Planet Labs) to detect deforestation events, classify land use, and provide verification services for supply chain compliance.

**Key Deliverables:**
- Satellite data integration pipelines (Sentinel-2, Landsat, Planet Labs, Google Earth Engine)
- Forest cover classification ML models (U-Net, ResNet backbone)
- Change detection algorithms for deforestation monitoring
- Training data collection for Amazon, Southeast Asia, and Africa regions
- MLOps infrastructure (MLflow, Feature Store, Model Registry)
- Inference optimization (ONNX, TensorRT)
- EUDR Agent API integration

**Success Metrics:**
- Forest classification accuracy: >95%
- Deforestation detection recall: >90%
- False positive rate: <10%
- Detection latency: <30 days from event
- Model inference time: <5 seconds/tile

---

## Phase 0: Foundation and Environment Setup (Weeks 1-4)

### Week 1: Environment and Access Setup

#### Day 1-2: Google Earth Engine Setup (6 tasks)
- [ ] GEES-001: Create Google Earth Engine project for GreenLang deforestation monitoring
- [ ] GEES-002: Request and obtain Earth Engine API access approval (may require 2-3 business days)
- [ ] GEES-003: Generate Earth Engine service account credentials and store in secrets manager
- [ ] GEES-004: Configure Earth Engine Python API authentication (earthengine-api package)
- [ ] GEES-005: Test Earth Engine connection with sample NDVI query for Amazon region
- [ ] GEES-006: Document Earth Engine quota limits (100,000 compute seconds/day) and implement rate limiting

#### Day 3: AWS Earth Data Setup (6 tasks)
- [ ] AWSE-001: Set up AWS account with Earth data access permissions
- [ ] AWSE-002: Configure Sentinel-2 Cloud-Optimized GeoTIFF (COG) access via AWS Open Data Registry
- [ ] AWSE-003: Configure Landsat Collection 2 COG access via AWS Open Data Registry
- [ ] AWSE-004: Set up S3 bucket (greenlang-satellite-data) for processed imagery storage with lifecycle policies
- [ ] AWSE-005: Configure IAM roles and policies for satellite data access (least privilege)
- [ ] AWSE-006: Test Sentinel-2 tile download from AWS (verify 10m resolution bands)

#### Day 4-5: Development Environment Setup (8 tasks)
- [ ] DEVE-001: Set up Python 3.11+ environment with conda/venv for ML development
- [ ] DEVE-002: Install core geospatial libraries: rasterio 1.3+, GDAL 3.6+, geopandas 0.14+
- [ ] DEVE-003: Install ML frameworks: TensorFlow 2.15+ or PyTorch 2.1+ with CUDA 12.x support
- [ ] DEVE-004: Install satellite-specific libraries: eo-learn 1.5+, sentinelhub 3.9+, rio-tiler
- [ ] DEVE-005: Install Google Earth Engine Python API (earthengine-api 0.1.380+)
- [ ] DEVE-006: Configure GPU access for model training (verify NVIDIA drivers, CUDA toolkit)
- [ ] DEVE-007: Set up Jupyter Lab environment for exploration and prototyping
- [ ] DEVE-008: Create requirements.txt and pyproject.toml for reproducibility

---

## Phase 1: Satellite Data Integration (Weeks 2-6) - 35 Tasks

### Week 2: Sentinel-2 Integration (8 tasks)

#### Sentinel-2 API and Authentication
- [ ] S2-001: Implement Sentinel-2 API authentication using Copernicus Data Space Ecosystem credentials
- [ ] S2-002: Create SentinelHubConfig class for managing API tokens and rate limits
- [ ] S2-003: Implement token refresh mechanism for long-running data collection jobs

#### Sentinel-2 Scene Selection
- [ ] S2-004: Implement Sentinel-2 scene discovery by bounding box (WGS84 coordinates)
- [ ] S2-005: Implement Sentinel-2 scene discovery by date range with configurable windows
- [ ] S2-006: Implement Sentinel-2 cloud cover threshold filtering (default <20%)

#### Sentinel-2 Band Processing (RGB, NIR, SWIR)
- [ ] S2-007: Build Sentinel-2 band extraction for 10m bands (B02-Blue, B03-Green, B04-Red, B08-NIR)
- [ ] S2-008: Build Sentinel-2 band extraction for 20m bands (B05-B07 Red Edge, B11-SWIR1, B12-SWIR2)

### Week 3: Landsat Integration (8 tasks)

#### Landsat API and Authentication
- [ ] LS-001: Implement USGS EarthExplorer API authentication for Landsat data access
- [ ] LS-002: Configure AWS Open Data access for Landsat Collection 2 Level-2 products
- [ ] LS-003: Implement scene ID parsing (e.g., LC09_L2SP_001010_20240101_...)

#### Landsat Scene Selection
- [ ] LS-004: Implement Landsat Collection 2 tile discovery by WRS-2 path/row
- [ ] LS-005: Implement Landsat tile discovery by date range and cloud cover threshold
- [ ] LS-006: Build cross-reference mapping between Sentinel-2 MGRS tiles and Landsat WRS-2 path/row

#### Landsat Band Processing
- [ ] LS-007: Build Landsat band extraction (Blue, Green, Red, NIR, SWIR1, SWIR2)
- [ ] LS-008: Implement Landsat surface reflectance scaling and quality flag parsing

### Week 4: Planet Labs Integration (7 tasks)

#### Planet Labs API Setup
- [ ] PL-001: Evaluate Planet Labs API pricing (NICFI Basemaps for tropical forests - free for research)
- [ ] PL-002: Obtain Planet Labs API key and configure authentication
- [ ] PL-003: Implement Planet Labs API client with rate limiting and retry logic

#### Planet Labs Scene Search and Download
- [ ] PL-004: Implement Planet Labs scene search by AOI and date range
- [ ] PL-005: Build Planet Labs 4-band download pipeline (Blue, Green, Red, NIR at 4.77m)
- [ ] PL-006: Implement Planet Labs order management for bulk downloads
- [ ] PL-007: Test Planet Labs data quality for 5 sample locations in Amazon basin

### Week 5: Cloud Removal Pipeline (6 tasks)

#### Cloud Detection Algorithms
- [ ] CLR-001: Research and implement cloud masking algorithms (Fmask 4.0, s2cloudless)
- [ ] CLR-002: Implement Sentinel-2 cloud detection using Scene Classification Layer (SCL band)
- [ ] CLR-003: Implement Sentinel-2 cloud detection using s2cloudless ML model (threshold 0.4)
- [ ] CLR-004: Implement Landsat cloud detection using QA_PIXEL band (Fmask algorithm)

#### Cloud Shadow and Composite Generation
- [ ] CLR-005: Build cloud shadow detection algorithm using geometric projection
- [ ] CLR-006: Create cloud-free composite generation using temporal median with 90-day window

---

## Phase 2: Deforestation Detection ML (Weeks 7-18) - 55 Tasks

### Week 7-8: Training Data Collection - Amazon Region (6 tasks)

#### Amazon Deforestation Data Collection
- [ ] TDA-001: Download Global Forest Change (Hansen) dataset for Amazon basin (2020-2024)
- [ ] TDA-002: Extract deforestation polygons for Brazil (Mato Grosso, Para, Rondonia states)
- [ ] TDA-003: Extract deforestation polygons for Peru, Colombia, Bolivia (Amazon portions)
- [ ] TDA-004: Collect 5,000 forest samples from intact Amazon rainforest (Sentinel-2 patches)
- [ ] TDA-005: Collect 5,000 non-forest samples from Amazon (agriculture, urban, water)
- [ ] TDA-006: Collect 2,000 deforestation event samples with before/after imagery pairs

### Week 9: Training Data Collection - Southeast Asia Region (6 tasks)

#### Southeast Asia Deforestation Data Collection
- [ ] TDSE-001: Download Global Forest Change (Hansen) dataset for SE Asia (Indonesia, Malaysia, PNG)
- [ ] TDSE-002: Extract 2020-2024 deforestation polygons for Indonesia (Sumatra, Kalimantan, Papua)
- [ ] TDSE-003: Extract 2020-2024 deforestation polygons for Malaysia and Papua New Guinea
- [ ] TDSE-004: Collect 5,000 forest samples from tropical rainforest (Sentinel-2 patches)
- [ ] TDSE-005: Collect 5,000 non-forest samples including 2,000 palm oil plantation samples
- [ ] TDSE-006: Collect 2,000 deforestation event samples with before/after imagery pairs

### Week 10: Training Data Collection - Africa Region (6 tasks)

#### Africa Deforestation Data Collection
- [ ] TDAF-001: Download Global Forest Change (Hansen) dataset for Africa (Congo Basin, West Africa)
- [ ] TDAF-002: Extract 2020-2024 deforestation polygons for Congo Basin (DRC, Gabon, Cameroon)
- [ ] TDAF-003: Extract 2020-2024 deforestation polygons for West Africa (Ghana, Ivory Coast)
- [ ] TDAF-004: Collect 5,000 forest samples from tropical rainforest (Sentinel-2 patches)
- [ ] TDAF-005: Collect 5,000 non-forest samples including 2,000 cocoa plantation samples
- [ ] TDAF-006: Collect 2,000 deforestation event samples with before/after imagery pairs

### Week 11-12: Forest Cover Classification Model Architecture (15 tasks)

#### Model Architecture Design
- [ ] FCC-001: Research CNN architectures for forest classification (ResNet-50, U-Net, DeepLabV3+)
- [ ] FCC-002: Design forest/non-forest binary classification model architecture
- [ ] FCC-003: Implement U-Net encoder-decoder architecture with skip connections
- [ ] FCC-004: Implement ResNet-50 backbone for feature extraction (ImageNet pretrained)
- [ ] FCC-005: Implement DeepLabV3+ architecture with atrous spatial pyramid pooling
- [ ] FCC-006: Design multi-scale feature pyramid network for multi-resolution inputs
- [ ] FCC-007: Implement attention mechanism for forest edge detection
- [ ] FCC-008: Create model configuration YAML schema for hyperparameter management

#### Training Data Preparation
- [ ] FCC-009: Implement data augmentation pipeline (rotation, flip, scale, brightness)
- [ ] FCC-010: Split datasets: 70% train, 15% validation, 15% test for each region
- [ ] FCC-011: Implement class balancing (forest/non-forest) using weighted sampling

#### Model Training Pipeline
- [ ] FCC-012: Train U-Net model on Amazon Sentinel-2 data (batch size 32, epochs 100)
- [ ] FCC-013: Implement early stopping based on validation loss (patience 10 epochs)
- [ ] FCC-014: Implement learning rate scheduling (cosine annealing with warm restarts)
- [ ] FCC-015: Log training metrics to MLflow/Weights & Biases

### Week 13-14: Change Detection Algorithms (12 tasks)

#### Bi-temporal Change Detection
- [ ] CHD-001: Research Siamese networks for bi-temporal change detection
- [ ] CHD-002: Design Siamese CNN architecture with shared weight encoder
- [ ] CHD-003: Implement difference/concatenation layer for before/after comparison
- [ ] CHD-004: Implement change classification head with binary output (change/no-change)
- [ ] CHD-005: Create before/after image pairs dataset from deforestation events

#### Time Series Analysis
- [ ] CHD-006: Implement monthly NDVI time series extraction for target regions
- [ ] CHD-007: Design recurrent architecture for time series (LSTM or ConvLSTM)
- [ ] CHD-008: Implement BFAST (Breaks For Additive Season and Trend) algorithm
- [ ] CHD-009: Train time series model on 2-year NDVI sequences

#### Alert Generation
- [ ] CHD-010: Implement breakpoint detection algorithm for NDVI time series
- [ ] CHD-011: Design alert generation system with confidence scores
- [ ] CHD-012: Implement multi-temporal confirmation (3 consecutive observations)

### Week 15-16: Model Validation and Accuracy Metrics (10 tasks)

#### Accuracy Metrics Calculation
- [ ] VAL-001: Calculate overall accuracy for forest classification (Amazon test set)
- [ ] VAL-002: Calculate precision, recall, F1-score for forest class (Amazon)
- [ ] VAL-003: Calculate IoU (Intersection over Union) for forest class (Amazon)
- [ ] VAL-004: Repeat accuracy metrics for SE Asia model
- [ ] VAL-005: Repeat accuracy metrics for Africa model
- [ ] VAL-006: Calculate confusion matrix for each region (forest, non-forest, change)

#### Change Detection Validation
- [ ] VAL-007: Calculate change detection accuracy on independent test set
- [ ] VAL-008: Calculate false positive rate and false negative rate
- [ ] VAL-009: Validate detection latency (average days from event to detection)
- [ ] VAL-010: Create comprehensive validation report with per-region accuracy breakdown

---

## Phase 3: ML Infrastructure and MLOps (Weeks 19-28) - 50 Tasks

### Week 19-20: MLOps Pipeline Setup with MLflow (12 tasks)

#### MLflow Server Setup
- [ ] MLF-001: Deploy MLflow tracking server on Kubernetes (PostgreSQL backend, S3 artifacts)
- [ ] MLF-002: Configure MLflow backend storage (PostgreSQL database schema)
- [ ] MLF-003: Configure MLflow artifact storage (S3 bucket for model artifacts)
- [ ] MLF-004: Set up MLflow authentication and access control

#### Experiment Tracking
- [ ] MLF-005: Implement experiment logging for all training runs (hyperparams, metrics)
- [ ] MLF-006: Create standardized experiment naming convention (model_region_date)
- [ ] MLF-007: Implement artifact logging (model checkpoints, confusion matrices, plots)
- [ ] MLF-008: Create experiment comparison dashboard

#### Training Pipeline Automation
- [ ] MLF-009: Build automated training pipeline with Kubeflow/Airflow
- [ ] MLF-010: Implement hyperparameter sweep automation (Optuna integration)
- [ ] MLF-011: Create training job scheduling for nightly model updates
- [ ] MLF-012: Implement training failure alerting (Slack/PagerDuty)

### Week 21-22: Model Registry (8 tasks)

#### Model Registry Implementation
- [ ] MRG-001: Design model registry schema (model name, version, stage, metrics)
- [ ] MRG-002: Implement semantic versioning for models (major.minor.patch)
- [ ] MRG-003: Implement model staging workflow (None -> Staging -> Production -> Archived)
- [ ] MRG-004: Create model promotion API with approval gates

#### Model Lifecycle Management
- [ ] MRG-005: Implement model deprecation and archival workflow
- [ ] MRG-006: Create model lineage tracking (training data version, code version)
- [ ] MRG-007: Implement model rollback capability with one-click restore
- [ ] MRG-008: Create model comparison API for A/B testing

### Week 23-24: Feature Store (8 tasks)

#### Feature Store Design
- [ ] FST-001: Evaluate feature store options (Feast, Tecton, custom implementation)
- [ ] FST-002: Design feature store schema for satellite imagery features
- [ ] FST-003: Implement feature: NDVI monthly averages (last 12 months)
- [ ] FST-004: Implement feature: NDVI trend slope (linear regression coefficient)

#### Feature Engineering Pipeline
- [ ] FST-005: Implement feature: cloud-free observation count per month
- [ ] FST-006: Implement feature: precipitation data integration (ERA5 or CHIRPS)
- [ ] FST-007: Implement feature: elevation and slope from SRTM DEM
- [ ] FST-008: Implement feature: distance to roads and settlements (OSM derived)

### Week 25-26: Training Infrastructure - GPU (10 tasks)

#### GPU Cluster Setup
- [ ] GPU-001: Provision GPU nodes (NVIDIA A100/V100) on cloud provider (AWS/GCP/Azure)
- [ ] GPU-002: Configure Kubernetes GPU operator and device plugin
- [ ] GPU-003: Set up multi-GPU training environment (DistributedDataParallel)
- [ ] GPU-004: Implement data parallelism for large satellite image datasets

#### Training Optimization
- [ ] GPU-005: Implement mixed precision training (FP16) with gradient scaling
- [ ] GPU-006: Implement gradient checkpointing for memory efficiency
- [ ] GPU-007: Configure NVIDIA DALI for GPU-accelerated data loading
- [ ] GPU-008: Set up training monitoring dashboard (GPU utilization, memory, throughput)

#### Resource Management
- [ ] GPU-009: Implement spot instance training with checkpointing for cost savings
- [ ] GPU-010: Create training job queue with priority-based scheduling

### Week 27-28: Inference Optimization - ONNX and TensorRT (12 tasks)

#### Model Export and Optimization
- [ ] INF-001: Implement ONNX export for PyTorch/TensorFlow models
- [ ] INF-002: Validate ONNX model outputs match original model (tolerance 1e-5)
- [ ] INF-003: Implement TensorRT optimization for NVIDIA GPU inference
- [ ] INF-004: Implement model quantization (INT8) with calibration dataset

#### Inference Pipeline
- [ ] INF-005: Build tile-based inference pipeline for large area processing
- [ ] INF-006: Implement tile stitching with overlap handling (50% overlap)
- [ ] INF-007: Implement batch inference across multiple tiles (GPU batching)
- [ ] INF-008: Create inference job queue with Redis/Celery

#### Performance Benchmarking
- [ ] INF-009: Benchmark inference speed: target <5 seconds per 10km x 10km tile
- [ ] INF-010: Benchmark memory usage and optimize for 16GB GPU memory
- [ ] INF-011: Implement inference result caching (Redis) for repeated queries
- [ ] INF-012: Create inference performance monitoring dashboard

---

## Phase 4: EUDR Agent Integration (Weeks 29-32)

### Week 29-30: API Endpoints for EUDR Verification

#### Deforestation Check API
- [ ] API-001: Design REST API for deforestation verification endpoint
  ```
  POST /v1/deforestation/check
  {
    "coordinates": {"lat": -3.123, "lon": -60.456},
    "polygon": "GeoJSON polygon",
    "date_range": {"start": "2020-01-01", "end": "2024-12-01"},
    "commodity": "soy"
  }
  ```
- [ ] API-002: Implement coordinate validation (WGS84, 6 decimal precision minimum)
- [ ] API-003: Implement polygon area validation (max 10,000 hectares per request)
- [ ] API-004: Implement date range validation (cutoff date: 2020-12-31 for EUDR)

#### Response Format
- [ ] API-005: Design deforestation check response schema
  ```json
  {
    "status": "deforestation_free|deforestation_detected|insufficient_data",
    "confidence": 0.95,
    "deforestation_events": [
      {
        "date": "2023-05-15",
        "area_ha": 2.5,
        "coordinates": {...}
      }
    ],
    "forest_cover_history": [...],
    "metadata": {
      "satellite": "Sentinel-2",
      "model_version": "1.2.0",
      "processing_date": "2024-12-04"
    }
  }
  ```
- [ ] API-006: Implement error response schema with clear error codes

### Week 31: Batch Processing Pipeline

#### Batch Job Management
- [ ] BPP-001: Design batch processing API for large-scale supply chain verification
  ```
  POST /v1/deforestation/batch
  {
    "locations": [
      {"id": "plot_001", "polygon": "..."},
      {"id": "plot_002", "polygon": "..."}
    ],
    "date_range": {...}
  }
  ```
- [ ] BPP-002: Implement job queue with Celery/Redis for background processing
- [ ] BPP-003: Implement job status tracking and progress reporting
- [ ] BPP-004: Implement webhook notifications for batch job completion

#### Performance and Scaling
- [ ] BPP-005: Implement parallel processing for batch jobs (concurrent workers)
- [ ] BPP-006: Implement result aggregation and summary statistics
- [ ] BPP-007: Create batch job dashboard for monitoring queue depth and throughput

### Week 32: Real-Time Monitoring and Alerts

#### Alert System
- [ ] RTA-001: Implement near-real-time deforestation monitoring (weekly scans)
- [ ] RTA-002: Design alert schema for deforestation detection events
  ```json
  {
    "alert_id": "ALT-2024-00001",
    "severity": "high|medium|low",
    "location": {"lat": -3.123, "lon": -60.456},
    "area_ha": 5.2,
    "detection_date": "2024-12-04",
    "confidence": 0.92,
    "satellite_image_url": "s3://..."
  }
  ```
- [ ] RTA-003: Implement alert filtering by region, severity, area threshold

#### Notification System
- [ ] RTA-004: Implement webhook integration for alert delivery
- [ ] RTA-005: Implement email notification for critical alerts
- [ ] RTA-006: Create alert dashboard with map visualization

#### Integration with EUDR Agent
- [ ] RTA-007: Create gRPC interface for low-latency agent communication
- [ ] RTA-008: Implement authentication and authorization (API keys, JWT)
- [ ] RTA-009: Create rate limiting and quota management per customer
- [ ] RTA-010: Document API integration guide for EUDR Agent team

---

## Model Specifications

### Forest Cover Classification Model

```yaml
model_spec:
  name: "gl-forest-classifier"
  version: "1.0.0"
  architecture: "U-Net with ResNet-50 backbone"
  input:
    channels: 10  # B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12
    resolution: "10m"
    tile_size: 512x512 pixels
    projection: "UTM"
  output:
    classes: 2  # forest, non-forest
    format: "GeoTIFF with probability layer"
  training:
    dataset: "Hansen GFC + manual annotations"
    samples: 50,000 per region (Amazon, SE Asia, Africa)
    augmentation: "rotation, flip, scale, brightness"
    optimizer: "AdamW"
    learning_rate: 1e-4
    batch_size: 32
    epochs: 100
    early_stopping: "validation loss, patience 10"
  performance:
    target_accuracy: ">95%"
    target_f1: ">0.92"
    target_iou: ">0.85"
    inference_time: "<5s per 10km x 10km tile"
```

### Change Detection Model

```yaml
model_spec:
  name: "gl-deforestation-detector"
  version: "1.0.0"
  architecture: "Siamese U-Net with temporal attention"
  input:
    before_image: "Sentinel-2 10-band, 512x512"
    after_image: "Sentinel-2 10-band, 512x512"
    time_gap: "minimum 90 days"
  output:
    classes: 2  # change, no-change
    change_magnitude: "0-1 continuous"
    format: "GeoTIFF with change probability"
  training:
    dataset: "Hansen deforestation events (2020-2024)"
    positive_samples: 20,000 change events
    negative_samples: 60,000 no-change pairs
    augmentation: "synchronized augmentation for before/after"
  performance:
    target_recall: ">90%"
    target_precision: ">85%"
    target_detection_latency: "<30 days"
    false_positive_rate: "<10%"
```

---

## Data Pipeline Architecture

```
+---------------------+     +------------------------+     +-------------------+
|  Satellite Sources  |     |   Data Ingestion       |     |   Feature Store   |
|  - Sentinel-2       | --> |   - AWS Lambda         | --> |   - NDVI monthly  |
|  - Landsat 8/9      |     |   - Scene filtering    |     |   - Cloud-free %  |
|  - Planet NICFI     |     |   - Cloud masking      |     |   - Precipitation |
+---------------------+     +------------------------+     +-------------------+
                                      |
                                      v
+---------------------+     +------------------------+     +-------------------+
|   Model Training    |     |   MLflow Tracking      |     |   Model Registry  |
|   - GPU cluster     | --> |   - Experiments        | --> |   - Versioning    |
|   - Hyperparameter  |     |   - Metrics            |     |   - Staging       |
|     tuning          |     |   - Artifacts          |     |   - Production    |
+---------------------+     +------------------------+     +-------------------+
                                      |
                                      v
+---------------------+     +------------------------+     +-------------------+
| Inference Pipeline  |     |   EUDR Agent API       |     |   Alert System    |
|   - TensorRT        | --> |   - /deforestation     | --> |   - Webhooks      |
|   - Batch processing|     |   - /batch             |     |   - Email         |
|   - Tile stitching  |     |   - /monitor           |     |   - Dashboard     |
+---------------------+     +------------------------+     +-------------------+
```

---

## Infrastructure Configuration

### Kubernetes Resources

```yaml
# GPU Training Pod Spec
apiVersion: v1
kind: Pod
metadata:
  name: gl-training-pod
spec:
  containers:
  - name: training
    image: greenlang/ml-training:1.0.0
    resources:
      limits:
        nvidia.com/gpu: 4
        memory: 128Gi
        cpu: 32
    volumeMounts:
    - name: training-data
      mountPath: /data
    - name: model-artifacts
      mountPath: /artifacts
```

### MLflow Configuration

```yaml
# MLflow Tracking Server
mlflow:
  backend_store_uri: "postgresql://mlflow:****@db.greenlang.io:5432/mlflow"
  artifact_root: "s3://greenlang-mlflow-artifacts"
  host: "mlflow.greenlang.io"
  port: 5000

# Experiment Tracking
experiments:
  - name: "forest-classification-amazon"
    tags:
      region: "amazon"
      model_type: "u-net"
  - name: "deforestation-detection-global"
    tags:
      model_type: "siamese"
```

### Feature Store Schema

```yaml
# Feast Feature Store Configuration
feature_store:
  project: "greenlang_satellite"
  registry: "s3://greenlang-feast/registry.db"
  provider: "aws"

feature_views:
  - name: "ndvi_monthly"
    entities: ["location_id"]
    features:
      - name: "ndvi_mean"
        dtype: FLOAT
      - name: "ndvi_std"
        dtype: FLOAT
      - name: "cloud_free_days"
        dtype: INT32
    ttl: 2592000  # 30 days

  - name: "forest_probability"
    entities: ["location_id"]
    features:
      - name: "forest_prob"
        dtype: FLOAT
      - name: "model_version"
        dtype: STRING
    ttl: 604800  # 7 days
```

---

## Dependencies and Integration

### External Dependencies

| Dependency | Provider | Phase | Description |
|------------|----------|-------|-------------|
| Google Earth Engine | Google | 0 | Satellite data processing engine |
| Sentinel-2 L2A | ESA/Copernicus | 1 | Primary satellite imagery (10m resolution) |
| Landsat Collection 2 | USGS | 1 | Secondary imagery (30m resolution) |
| Planet NICFI | Planet Labs | 1 | High-frequency tropical monitoring |
| Hansen Global Forest Change | UMD | 2 | Training labels for deforestation |
| NVIDIA GPU (A100/V100) | Cloud provider | 2-3 | Model training and inference |
| MLflow | Self-hosted | 3 | Experiment tracking and model registry |

### Internal Dependencies

| Dependency | Provider Team | Phase | Description |
|------------|---------------|-------|-------------|
| EUDR Agent integration | AI/Agent Team | 4 | API consumer for deforestation checks |
| Climate Science validation | Climate Science Team | 2 | Golden test cases for validation |
| Infrastructure (K8s, S3) | DevOps Team | 0-3 | Compute and storage infrastructure |
| Agent Registry | Platform Team | 3 | Model deployment integration |

### Deliverables to Other Teams

| Deliverable | Consumer Team | Phase | Description |
|-------------|---------------|-------|-------------|
| Deforestation Check API | AI/Agent (EUDR) | 4 | REST API for supply chain verification |
| Batch Processing API | AI/Agent (EUDR) | 4 | Bulk location verification |
| Alert Webhook | Platform | 4 | Real-time deforestation alerts |
| Model Artifacts | ML Platform | 3 | Trained models for registry |

---

## Success Metrics and KPIs

### Phase 1 Metrics (Satellite Data Integration)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sentinel-2 scene availability | >95% | % of requested scenes successfully downloaded |
| Cloud-free composite quality | <5% residual cloud | Visual inspection of 100 sample composites |
| Data ingestion latency | <24 hours | Time from scene availability to processed tile |
| Pipeline uptime | 99.5% | Automated monitoring |

### Phase 2 Metrics (Deforestation Detection)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Forest classification accuracy | >95% | Overall accuracy on test set |
| Deforestation detection recall | >90% | True positive rate on known events |
| False positive rate | <10% | False alarms per 1000 km2 per month |
| Detection latency | <30 days | Days from deforestation to detection |
| F1 score by region | >0.90 | Amazon, SE Asia, Africa separately |

### Phase 3 Metrics (ML Infrastructure)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model training time | <24 hours | Full training run on GPU cluster |
| Inference latency | <5 seconds/tile | 10km x 10km tile at 10m resolution |
| Model deployment time | <1 hour | Staging to production promotion |
| Feature store latency | <100ms | Feature retrieval time |

### Phase 4 Metrics (EUDR Integration)

| Metric | Target | Measurement |
|--------|--------|-------------|
| API availability | 99.9% | Uptime over 30-day period |
| API latency (single location) | <10 seconds | P95 response time |
| Batch processing throughput | 1000 locations/hour | Sustained throughput |
| Alert delivery latency | <1 hour | Time from detection to notification |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cloud cover limits usable imagery | High | Medium | Multi-sensor fusion (S2 + Landsat + Planet); longer compositing windows |
| Training data quality issues | Medium | High | Manual validation sampling; active learning for label correction |
| Model performance degrades over time | Medium | High | Drift monitoring; quarterly retraining with new data |
| GPU compute costs exceed budget | Medium | Medium | Spot instances; inference optimization (INT8 quantization) |
| Earth Engine quota limits | Low | High | Backup AWS-based pipeline; query optimization |
| False positive alerts cause user fatigue | Medium | Medium | Multi-temporal confirmation; confidence thresholds; human review queue |

---

## Team Resource Allocation

| Phase | Weeks | ML Engineers | Remote Sensing | Data Engineers | DevOps | Total FTE-weeks |
|-------|-------|--------------|----------------|----------------|--------|-----------------|
| Phase 0 | 1-4 | 1.0 | 0.5 | 0.5 | 0.5 | 10 |
| Phase 1 | 5-10 | 2.0 | 1.0 | 0.5 | 0.25 | 22.5 |
| Phase 2 | 11-18 | 3.0 | 1.0 | 0.5 | 0.25 | 38 |
| Phase 3 | 19-28 | 2.0 | 0.5 | 1.0 | 1.0 | 45 |
| Phase 4 | 29-32 | 2.0 | 0.5 | 0.5 | 0.5 | 14 |
| **Total** | **1-32** | - | - | - | - | **129.5 FTE-weeks** |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | ML/Satellite Team Lead | Initial detailed TODO list |

---

**Total Tasks: 140+**
**Estimated Duration: 32 weeks (8 months)**
**Team Size: 3-4 ML Engineers + 1-2 Remote Sensing Specialists + 1 Data Engineer**

---

## Appendix A: Sample Code - Sentinel-2 Data Loader

```python
"""
Sentinel-2 Data Loader for Deforestation Detection.

This module provides utilities for loading and preprocessing Sentinel-2
imagery for the GreenLang deforestation detection pipeline.
"""

import rasterio
from rasterio.windows import Window
from typing import Tuple, List, Optional
import numpy as np
from datetime import datetime
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    bbox_to_dimensions
)


class Sentinel2Loader:
    """
    Load and preprocess Sentinel-2 L2A imagery.

    Attributes:
        config: SentinelHub configuration with API credentials
        bands: List of band names to load
        resolution: Spatial resolution in meters

    Example:
        >>> loader = Sentinel2Loader(config)
        >>> image = loader.load_tile(
        ...     bbox=BBox((-60.5, -3.5, -60.0, -3.0), CRS.WGS84),
        ...     date_range=("2024-01-01", "2024-03-31"),
        ...     max_cloud_cover=0.2
        ... )
    """

    BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B11", "B12"]

    def __init__(
        self,
        config: SHConfig,
        bands: Optional[List[str]] = None,
        resolution: int = 10
    ):
        self.config = config
        self.bands = bands or self.BAND_NAMES
        self.resolution = resolution

    def load_tile(
        self,
        bbox: BBox,
        date_range: Tuple[str, str],
        max_cloud_cover: float = 0.2
    ) -> np.ndarray:
        """
        Load Sentinel-2 tile for specified bounding box and date range.

        Args:
            bbox: Bounding box in WGS84 coordinates
            date_range: Tuple of (start_date, end_date) in ISO format
            max_cloud_cover: Maximum cloud cover fraction (0-1)

        Returns:
            numpy array of shape (height, width, num_bands)

        Raises:
            ValueError: If no valid scenes found for date range
        """
        size = bbox_to_dimensions(bbox, resolution=self.resolution)

        evalscript = self._build_evalscript()

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=date_range,
                    maxcc=max_cloud_cover
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox,
            size=size,
            config=self.config
        )

        images = request.get_data()

        if not images:
            raise ValueError(f"No valid scenes found for {date_range}")

        return np.array(images[0])

    def _build_evalscript(self) -> str:
        """Build Sentinel Hub evalscript for band extraction."""
        band_list = ", ".join([f'"{b}"' for b in self.bands])

        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{{bands: [{band_list}]}}],
                output: {{bands: {len(self.bands)}, sampleType: "FLOAT32"}}
            }};
        }}

        function evaluatePixel(sample) {{
            return [{", ".join([f"sample.{b}" for b in self.bands])}];
        }}
        """


class CloudMasker:
    """
    Apply cloud masking to Sentinel-2 imagery using SCL band.

    The Scene Classification Layer (SCL) provides cloud/shadow classification:
    - 0: No data
    - 3: Cloud shadows
    - 8: Cloud medium probability
    - 9: Cloud high probability
    - 10: Thin cirrus
    """

    CLOUD_CLASSES = [3, 8, 9, 10]  # SCL values for clouds/shadows

    def mask_clouds(
        self,
        image: np.ndarray,
        scl: np.ndarray,
        fill_value: float = np.nan
    ) -> np.ndarray:
        """
        Apply cloud mask to multi-band image.

        Args:
            image: Input image array (height, width, bands)
            scl: Scene Classification Layer (height, width)
            fill_value: Value to use for masked pixels

        Returns:
            Masked image array
        """
        cloud_mask = np.isin(scl, self.CLOUD_CLASSES)
        masked = image.copy()
        masked[cloud_mask] = fill_value
        return masked
```

---

## Appendix B: Sample Code - NDVI Time Series Analysis

```python
"""
NDVI Time Series Analysis for Deforestation Detection.

This module provides utilities for analyzing NDVI (Normalized Difference
Vegetation Index) time series to detect vegetation loss events.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import theilslopes
from typing import Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DeforestationEvent:
    """Detected deforestation event from NDVI time series."""
    date: datetime
    location: Tuple[float, float]  # (lat, lon)
    ndvi_drop: float  # Magnitude of NDVI decrease
    confidence: float  # Detection confidence (0-1)
    area_ha: Optional[float] = None


class NDVITimeSeriesAnalyzer:
    """
    Analyze NDVI time series for deforestation detection.

    Uses BFAST-lite algorithm for breakpoint detection in
    vegetation index time series.

    Attributes:
        window_size: Savitzky-Golay filter window size
        threshold: NDVI drop threshold for event detection

    Example:
        >>> analyzer = NDVITimeSeriesAnalyzer(threshold=0.3)
        >>> events = analyzer.detect_breakpoints(
        ...     ndvi_series=ndvi_values,
        ...     dates=date_array,
        ...     location=(-3.5, -60.0)
        ... )
    """

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 0.3,
        min_confidence: float = 0.7
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_confidence = min_confidence

    def calculate_ndvi(
        self,
        nir: np.ndarray,
        red: np.ndarray
    ) -> np.ndarray:
        """
        Calculate NDVI from NIR and Red bands.

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            nir: Near-infrared band values
            red: Red band values

        Returns:
            NDVI values in range [-1, 1]
        """
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = np.nan_to_num(ndvi, nan=0.0)
        return ndvi

    def smooth_series(
        self,
        ndvi_series: np.ndarray
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to NDVI time series.

        Args:
            ndvi_series: Raw NDVI values

        Returns:
            Smoothed NDVI series
        """
        if len(ndvi_series) < self.window_size:
            return ndvi_series

        return savgol_filter(
            ndvi_series,
            window_length=self.window_size,
            polyorder=2
        )

    def detect_trend(
        self,
        ndvi_series: np.ndarray
    ) -> Tuple[float, float]:
        """
        Detect linear trend in NDVI series using Theil-Sen estimator.

        Args:
            ndvi_series: NDVI time series values

        Returns:
            Tuple of (slope, intercept)
        """
        x = np.arange(len(ndvi_series))
        slope, intercept, _, _ = theilslopes(ndvi_series, x)
        return slope, intercept

    def detect_breakpoints(
        self,
        ndvi_series: np.ndarray,
        dates: List[datetime],
        location: Tuple[float, float]
    ) -> List[DeforestationEvent]:
        """
        Detect deforestation events using breakpoint analysis.

        Args:
            ndvi_series: NDVI time series values
            dates: Corresponding observation dates
            location: (latitude, longitude) of the pixel

        Returns:
            List of detected DeforestationEvent objects
        """
        events = []
        smoothed = self.smooth_series(ndvi_series)

        # Calculate first-order differences
        diff = np.diff(smoothed)

        # Find large negative changes (potential deforestation)
        for i in range(len(diff)):
            if diff[i] < -self.threshold:
                # Calculate confidence based on magnitude and persistence
                ndvi_drop = abs(diff[i])

                # Check if drop persists (not just noise)
                if i + 3 < len(smoothed):
                    persistence = smoothed[i] - np.mean(smoothed[i+1:i+4])
                    confidence = min(1.0, persistence / self.threshold)
                else:
                    confidence = ndvi_drop / self.threshold

                if confidence >= self.min_confidence:
                    events.append(DeforestationEvent(
                        date=dates[i+1],
                        location=location,
                        ndvi_drop=ndvi_drop,
                        confidence=confidence
                    ))

        return events
```

---

## Appendix C: API Specification - Deforestation Check

```yaml
openapi: 3.0.3
info:
  title: GreenLang Deforestation Detection API
  description: API for EUDR compliance deforestation verification
  version: 1.0.0

servers:
  - url: https://api.greenlang.io/satellite/v1
    description: Production server

paths:
  /deforestation/check:
    post:
      summary: Check location for deforestation
      description: |
        Analyze a geographic location or polygon for deforestation events
        since the EUDR cutoff date (2020-12-31).
      operationId: checkDeforestation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DeforestationCheckRequest'
      responses:
        '200':
          description: Successful deforestation check
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DeforestationCheckResponse'
        '400':
          description: Invalid request parameters
        '429':
          description: Rate limit exceeded
        '500':
          description: Internal server error

  /deforestation/batch:
    post:
      summary: Batch deforestation check
      description: Submit multiple locations for batch processing
      operationId: batchCheckDeforestation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/BatchCheckRequest'
      responses:
        '202':
          description: Batch job accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BatchJobResponse'

  /deforestation/batch/{job_id}:
    get:
      summary: Get batch job status
      description: Retrieve status and results of a batch deforestation check
      operationId: getBatchJobStatus
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Batch job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BatchJobStatusResponse'

components:
  schemas:
    DeforestationCheckRequest:
      type: object
      required:
        - location
        - date_range
      properties:
        location:
          oneOf:
            - $ref: '#/components/schemas/PointLocation'
            - $ref: '#/components/schemas/PolygonLocation'
        date_range:
          $ref: '#/components/schemas/DateRange'
        commodity:
          type: string
          enum: [cattle, cocoa, coffee, palm_oil, rubber, soya, wood]
          description: EUDR commodity type for context

    PointLocation:
      type: object
      required:
        - lat
        - lon
      properties:
        lat:
          type: number
          minimum: -90
          maximum: 90
          description: Latitude (WGS84)
        lon:
          type: number
          minimum: -180
          maximum: 180
          description: Longitude (WGS84)

    PolygonLocation:
      type: object
      required:
        - type
        - coordinates
      properties:
        type:
          type: string
          enum: [Polygon]
        coordinates:
          type: array
          items:
            type: array
            items:
              type: array
              items:
                type: number
          description: GeoJSON polygon coordinates

    DateRange:
      type: object
      required:
        - start
        - end
      properties:
        start:
          type: string
          format: date
          description: Start date (ISO 8601)
        end:
          type: string
          format: date
          description: End date (ISO 8601)

    DeforestationCheckResponse:
      type: object
      properties:
        status:
          type: string
          enum: [deforestation_free, deforestation_detected, insufficient_data]
        confidence:
          type: number
          minimum: 0
          maximum: 1
        deforestation_events:
          type: array
          items:
            $ref: '#/components/schemas/DeforestationEvent'
        forest_cover_history:
          type: array
          items:
            $ref: '#/components/schemas/ForestCoverObservation'
        metadata:
          $ref: '#/components/schemas/ResponseMetadata'

    DeforestationEvent:
      type: object
      properties:
        date:
          type: string
          format: date
        area_ha:
          type: number
        coordinates:
          $ref: '#/components/schemas/PointLocation'
        confidence:
          type: number

    ForestCoverObservation:
      type: object
      properties:
        date:
          type: string
          format: date
        forest_cover_percent:
          type: number
        ndvi:
          type: number

    ResponseMetadata:
      type: object
      properties:
        satellite:
          type: string
        model_version:
          type: string
        processing_date:
          type: string
          format: date-time
        request_id:
          type: string

    BatchCheckRequest:
      type: object
      required:
        - locations
        - date_range
      properties:
        locations:
          type: array
          maxItems: 10000
          items:
            type: object
            properties:
              id:
                type: string
              location:
                oneOf:
                  - $ref: '#/components/schemas/PointLocation'
                  - $ref: '#/components/schemas/PolygonLocation'
        date_range:
          $ref: '#/components/schemas/DateRange'
        webhook_url:
          type: string
          format: uri
          description: URL for completion notification

    BatchJobResponse:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [queued, processing, completed, failed]
        estimated_completion:
          type: string
          format: date-time
        locations_count:
          type: integer

    BatchJobStatusResponse:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
        progress:
          type: number
          minimum: 0
          maximum: 100
        results:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
              result:
                $ref: '#/components/schemas/DeforestationCheckResponse'
        error:
          type: string
```

---

**END OF DOCUMENT**
