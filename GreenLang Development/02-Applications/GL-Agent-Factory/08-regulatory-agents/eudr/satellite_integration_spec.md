# EUDR Satellite Data Integration Specification

**Document ID:** GL-EUDR-SAT-SPEC-001
**Version:** 1.0.0
**Date:** December 4, 2025
**Author:** GL-Satellite-ML-Specialist
**Status:** APPROVED

---

## 1. Executive Summary

This specification defines the satellite data integration architecture for the EUDR Compliance Agent. The system uses multi-source satellite imagery to verify deforestation-free status of production plots against the December 31, 2020 cutoff date mandated by EU Regulation 2023/1115.

### Key Requirements

| Requirement | Target | Metric |
|-------------|--------|--------|
| Deforestation Detection Accuracy | >90% | F1 Score |
| False Positive Rate | <5% | Precision |
| False Negative Rate | <3% | Recall |
| Processing Time | <30s | Per polygon |
| Temporal Coverage | 2018-2025 | Years |
| Spatial Resolution | 10m | Sentinel-2 |

---

## 2. Satellite Data Sources

### 2.1 Primary Source: Sentinel-2

**Description:** European Space Agency's Sentinel-2 mission provides high-resolution multispectral imagery ideal for vegetation monitoring.

**Technical Specifications:**
| Parameter | Value |
|-----------|-------|
| Spatial Resolution | 10m (VNIR), 20m (SWIR), 60m (Coastal/Cirrus) |
| Temporal Resolution | 5 days (2 satellites) |
| Spectral Bands | 13 bands |
| Swath Width | 290 km |
| Archive Start | June 2015 |
| Cost | Free (Copernicus Open Access Hub) |

**Relevant Bands for Forest Monitoring:**
| Band | Wavelength (nm) | Resolution | Use Case |
|------|-----------------|------------|----------|
| B02 (Blue) | 490 | 10m | True color |
| B03 (Green) | 560 | 10m | True color |
| B04 (Red) | 665 | 10m | NDVI calculation |
| B08 (NIR) | 842 | 10m | NDVI calculation |
| B11 (SWIR) | 1610 | 20m | Burn detection |
| B12 (SWIR) | 2190 | 20m | Moisture content |

**API Integration:**
```python
class Sentinel2Client:
    BASE_URL = "https://dataspace.copernicus.eu/odata/v1"

    def __init__(self, client_id: str, client_secret: str):
        self.auth = OAuth2Session(client_id, client_secret)

    def search_scenes(
        self,
        polygon: Polygon,
        start_date: date,
        end_date: date,
        cloud_cover_max: float = 20.0
    ) -> List[Scene]:
        """
        Search for Sentinel-2 scenes covering the polygon.

        Args:
            polygon: GeoJSON polygon of area of interest
            start_date: Start of date range
            end_date: End of date range
            cloud_cover_max: Maximum cloud cover percentage

        Returns:
            List of matching scenes sorted by date
        """
        query = f"""
        Products?$filter=
            Collection/Name eq 'SENTINEL-2' and
            ContentDate/Start ge {start_date}T00:00:00.000Z and
            ContentDate/Start le {end_date}T23:59:59.999Z and
            OData.CSC.Intersects(area=geography'SRID=4326;{polygon.wkt}') and
            Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover_max})
        &$orderby=ContentDate/Start desc
        """
        return self._execute_query(query)

    def download_bands(
        self,
        product_id: str,
        bands: List[str] = ["B02", "B03", "B04", "B08"]
    ) -> Dict[str, np.ndarray]:
        """Download specific bands from a product."""
        pass
```

### 2.2 Secondary Source: Landsat 8/9

**Description:** NASA/USGS Landsat program provides longer historical archive and backup coverage.

**Technical Specifications:**
| Parameter | Value |
|-----------|-------|
| Spatial Resolution | 30m (Multispectral), 15m (Pan) |
| Temporal Resolution | 16 days (per satellite) |
| Spectral Bands | 11 bands |
| Archive Start | 2013 (Landsat 8), 2021 (Landsat 9) |
| Cost | Free (USGS Earth Explorer) |

**API Integration:**
```python
class LandsatClient:
    BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

    def search_scenes(
        self,
        polygon: Polygon,
        start_date: date,
        end_date: date,
        collection: str = "landsat_ot_c2_l2"
    ) -> List[Scene]:
        """Search for Landsat scenes covering the polygon."""
        pass
```

### 2.3 Alert Source: Global Forest Watch

**Description:** World Resources Institute's Global Forest Watch provides pre-processed deforestation alerts.

**Data Products:**
| Product | Resolution | Latency | Coverage |
|---------|------------|---------|----------|
| GLAD Alerts | 30m | ~1 week | Global (tropics) |
| RADD Alerts | 10m | ~1 week | Tropical forests |
| Hansen Tree Cover Loss | 30m | Annual | Global |

**API Integration:**
```python
class GlobalForestWatchClient:
    BASE_URL = "https://data-api.globalforestwatch.org"

    def get_forest_loss(
        self,
        polygon: Polygon,
        years: List[int]
    ) -> Dict[int, float]:
        """
        Get forest loss in hectares by year.

        Args:
            polygon: GeoJSON polygon
            years: List of years to query

        Returns:
            Dictionary of year -> hectares lost
        """
        endpoint = f"{self.BASE_URL}/analysis/umd-loss-gain"
        response = self.session.post(endpoint, json={
            "geometry": polygon.__geo_interface__,
            "aggregate_values": True,
            "geostore": None
        })
        return self._parse_loss_by_year(response.json(), years)

    def get_glad_alerts(
        self,
        polygon: Polygon,
        start_date: date,
        end_date: date
    ) -> List[Alert]:
        """Get GLAD deforestation alerts for polygon."""
        pass
```

### 2.4 Backup Source: Sentinel-1 SAR

**Description:** Sentinel-1 Synthetic Aperture Radar provides cloud-penetrating capability for persistent cloud regions.

**Use Cases:**
- Cloud gap filling in tropical regions
- Wet season monitoring
- Forest structure change detection

**Technical Specifications:**
| Parameter | Value |
|-----------|-------|
| Spatial Resolution | 10m (IW mode) |
| Polarization | VV, VH |
| Revisit Time | 6 days |
| Cloud Penetration | Yes |

---

## 3. Forest Change Detection Algorithm

### 3.1 Baseline Establishment

The system establishes a December 31, 2020 baseline for each production plot.

**Algorithm:**
```python
def establish_baseline(polygon: Polygon) -> BaselineResult:
    """
    Establish forest cover baseline for December 31, 2020.

    Args:
        polygon: Production plot polygon

    Returns:
        BaselineResult with forest cover metrics
    """
    # 1. Query imagery closest to cutoff date
    scenes = sentinel2_client.search_scenes(
        polygon=polygon,
        start_date=date(2020, 10, 1),
        end_date=date(2021, 2, 28),
        cloud_cover_max=30.0
    )

    # 2. Select best scene (closest to Dec 31, 2020)
    best_scene = select_optimal_scene(scenes, target_date=date(2020, 12, 31))

    # 3. Calculate NDVI
    ndvi = calculate_ndvi(best_scene)

    # 4. Generate forest mask (NDVI > 0.4)
    forest_mask = ndvi > 0.4

    # 5. Calculate forest area
    forest_area_ha = calculate_area(forest_mask, polygon)

    return BaselineResult(
        date=best_scene.date,
        forest_area_ha=forest_area_ha,
        forest_percentage=forest_area_ha / polygon.area_ha * 100,
        ndvi_mean=ndvi.mean(),
        confidence=calculate_confidence(best_scene)
    )
```

### 3.2 NDVI Calculation

**Normalized Difference Vegetation Index (NDVI):**
```
NDVI = (NIR - Red) / (NIR + Red)
```

**Implementation:**
```python
def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI from red and NIR bands.

    Args:
        red_band: Red band reflectance (0-10000 scale for Sentinel-2)
        nir_band: NIR band reflectance

    Returns:
        NDVI array (-1 to 1 scale)
    """
    # Avoid division by zero
    denominator = nir_band.astype(float) + red_band.astype(float)
    denominator[denominator == 0] = np.nan

    ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator

    # Mask invalid values
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[np.isnan(ndvi)] = -9999  # NoData value

    return ndvi
```

**NDVI Thresholds for Forest Classification:**
| NDVI Range | Classification |
|------------|----------------|
| > 0.6 | Dense forest |
| 0.4 - 0.6 | Forest/woodland |
| 0.2 - 0.4 | Shrubland/degraded |
| 0.0 - 0.2 | Sparse vegetation |
| < 0.0 | Non-vegetation |

### 3.3 Change Detection Model

**Architecture:** U-Net semantic segmentation model for forest/non-forest classification.

**Model Specification:**
```python
class ForestChangeModel:
    """
    U-Net based forest change detection model.

    Input: 4-channel stack (R, G, B, NIR) at two time points
    Output: 3-class segmentation (No Change, Deforestation, Reforestation)
    """

    def __init__(self):
        self.model = self._build_unet()
        self.input_shape = (256, 256, 8)  # 4 bands x 2 timepoints
        self.output_classes = 3

    def _build_unet(self) -> tf.keras.Model:
        """Build U-Net architecture."""
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Encoder (contracting path)
        c1 = self._conv_block(inputs, 64)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = self._conv_block(p1, 128)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = self._conv_block(p2, 256)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = self._conv_block(p3, 512)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        # Bridge
        c5 = self._conv_block(p4, 1024)

        # Decoder (expanding path)
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = self._conv_block(u6, 512)

        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2))(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = self._conv_block(u7, 256)

        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = self._conv_block(u8, 128)

        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2))(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = self._conv_block(u9, 64)

        outputs = tf.keras.layers.Conv2D(
            self.output_classes, (1, 1), activation='softmax'
        )(c9)

        return tf.keras.Model(inputs=[inputs], outputs=[outputs])

    def predict_change(
        self,
        baseline_imagery: np.ndarray,
        current_imagery: np.ndarray,
        polygon: Polygon
    ) -> ChangeResult:
        """
        Predict forest change between baseline and current imagery.

        Args:
            baseline_imagery: 4-band imagery from baseline date
            current_imagery: 4-band imagery from current date
            polygon: Production plot polygon

        Returns:
            ChangeResult with classification and metrics
        """
        # Stack baseline and current
        stacked = np.concatenate([baseline_imagery, current_imagery], axis=-1)

        # Run prediction
        prediction = self.model.predict(stacked[np.newaxis, ...])[0]

        # Calculate class areas
        pixel_area_ha = 0.01  # 10m resolution = 0.01 ha per pixel

        deforestation_pixels = (prediction.argmax(axis=-1) == 1).sum()
        deforestation_area_ha = deforestation_pixels * pixel_area_ha

        reforestation_pixels = (prediction.argmax(axis=-1) == 2).sum()
        reforestation_area_ha = reforestation_pixels * pixel_area_ha

        # Calculate confidence
        confidence = prediction.max(axis=-1).mean()

        return ChangeResult(
            deforestation_detected=deforestation_area_ha > 0.1,  # >0.1 ha threshold
            deforestation_area_ha=deforestation_area_ha,
            reforestation_area_ha=reforestation_area_ha,
            net_change_ha=reforestation_area_ha - deforestation_area_ha,
            confidence=confidence,
            change_map=prediction
        )
```

### 3.4 Multi-Source Data Fusion

**Algorithm for combining multiple satellite sources:**

```python
def fuse_satellite_sources(
    sentinel2_result: Optional[ChangeResult],
    landsat_result: Optional[ChangeResult],
    gfw_alerts: List[Alert]
) -> FusedResult:
    """
    Fuse results from multiple satellite sources.

    Priority:
    1. Sentinel-2 (highest resolution)
    2. Landsat (backup)
    3. GFW alerts (independent verification)

    Args:
        sentinel2_result: Change detection from Sentinel-2
        landsat_result: Change detection from Landsat
        gfw_alerts: GLAD/RADD alerts from Global Forest Watch

    Returns:
        FusedResult with weighted consensus
    """
    results = []
    weights = []

    # Add Sentinel-2 result
    if sentinel2_result is not None:
        results.append(sentinel2_result)
        weights.append(0.5)  # Highest weight

    # Add Landsat result
    if landsat_result is not None:
        results.append(landsat_result)
        weights.append(0.3)

    # Add GFW alert-based result
    if gfw_alerts:
        gfw_result = alerts_to_change_result(gfw_alerts)
        results.append(gfw_result)
        weights.append(0.2)

    # Weighted consensus
    if not results:
        return FusedResult(
            status="INSUFFICIENT_DATA",
            confidence=0.0
        )

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Calculate weighted deforestation detection
    deforestation_score = sum(
        w * (1 if r.deforestation_detected else 0)
        for r, w in zip(results, weights)
    )

    # Calculate weighted area
    deforestation_area = sum(
        w * r.deforestation_area_ha
        for r, w in zip(results, weights)
    )

    # Calculate weighted confidence
    confidence = sum(
        w * r.confidence
        for r, w in zip(results, weights)
    )

    # Determine final status
    if deforestation_score > 0.5:
        status = "DEFORESTATION_DETECTED"
    elif deforestation_score > 0.2:
        status = "POTENTIAL_DEFORESTATION"
    else:
        status = "NO_DEFORESTATION"

    return FusedResult(
        status=status,
        deforestation_detected=deforestation_score > 0.5,
        deforestation_area_ha=deforestation_area,
        confidence=confidence,
        sources_used=len(results),
        agreement_score=1 - np.std([
            1 if r.deforestation_detected else 0
            for r in results
        ])
    )
```

---

## 4. Data Quality Assessment

### 4.1 Quality Metrics

```python
@dataclass
class SatelliteDataQuality:
    """Quality assessment for satellite data."""

    cloud_cover_percentage: float
    temporal_proximity_days: int  # Days from target date
    spatial_coverage_percentage: float  # Coverage of AOI
    atmospheric_quality: str  # Good, Moderate, Poor
    sensor_quality: str  # Good, Degraded
    overall_score: float  # 0-100

    @classmethod
    def assess(cls, scene: Scene, target_date: date, polygon: Polygon) -> 'SatelliteDataQuality':
        """Assess quality of a satellite scene."""

        # Cloud cover score (lower is better)
        cloud_score = max(0, 100 - scene.cloud_cover * 2)

        # Temporal proximity score
        days_diff = abs((scene.date - target_date).days)
        temporal_score = max(0, 100 - days_diff * 2)

        # Spatial coverage score
        coverage = calculate_coverage(scene.footprint, polygon)
        spatial_score = coverage * 100

        # Atmospheric quality
        atm_quality = assess_atmosphere(scene)
        atm_score = {"Good": 100, "Moderate": 70, "Poor": 30}[atm_quality]

        # Sensor quality
        sensor_quality = assess_sensor(scene)
        sensor_score = {"Good": 100, "Degraded": 50}[sensor_quality]

        # Overall score (weighted average)
        overall = (
            cloud_score * 0.3 +
            temporal_score * 0.25 +
            spatial_score * 0.2 +
            atm_score * 0.15 +
            sensor_score * 0.1
        )

        return cls(
            cloud_cover_percentage=scene.cloud_cover,
            temporal_proximity_days=days_diff,
            spatial_coverage_percentage=coverage * 100,
            atmospheric_quality=atm_quality,
            sensor_quality=sensor_quality,
            overall_score=overall
        )
```

### 4.2 Minimum Quality Thresholds

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Cloud Cover | <50% | <20% |
| Temporal Proximity | <90 days | <30 days |
| Spatial Coverage | >80% | >95% |
| Overall Score | >50 | >70 |

---

## 5. Processing Pipeline

### 5.1 Pipeline Architecture

```
Input: Polygon + Date Range
           |
           v
    +-------------+
    | Scene Query |-----> Sentinel-2, Landsat, GFW
    +-------------+
           |
           v
    +---------------+
    | Scene Selection|-----> Best available scenes
    +---------------+
           |
           v
    +---------------+
    | Preprocessing |-----> Atmospheric correction, resampling
    +---------------+
           |
           v
    +---------------+
    | NDVI Calculation|
    +---------------+
           |
           v
    +------------------+
    | Change Detection |-----> U-Net model
    +------------------+
           |
           v
    +--------------+
    | Data Fusion  |-----> Multi-source consensus
    +--------------+
           |
           v
    +------------------+
    | Quality Assessment|
    +------------------+
           |
           v
    +----------------+
    | Report Generation|
    +----------------+
           |
           v
Output: DeforestationResult
```

### 5.2 Pipeline Implementation

```python
class SatelliteVerificationPipeline:
    """End-to-end satellite verification pipeline."""

    def __init__(self):
        self.sentinel2 = Sentinel2Client()
        self.landsat = LandsatClient()
        self.gfw = GlobalForestWatchClient()
        self.model = ForestChangeModel()

    async def verify_deforestation(
        self,
        polygon: Polygon,
        baseline_date: date = date(2020, 12, 31),
        current_date: date = None
    ) -> DeforestationResult:
        """
        Verify deforestation-free status for a production plot.

        Args:
            polygon: Production plot polygon
            baseline_date: EUDR cutoff date (default: Dec 31, 2020)
            current_date: Verification date (default: today)

        Returns:
            DeforestationResult with compliance determination
        """
        if current_date is None:
            current_date = date.today()

        # 1. Query all sources in parallel
        sentinel2_baseline, sentinel2_current, landsat_baseline, landsat_current, gfw_alerts = await asyncio.gather(
            self.sentinel2.search_scenes(polygon, baseline_date - timedelta(days=90), baseline_date + timedelta(days=90)),
            self.sentinel2.search_scenes(polygon, current_date - timedelta(days=30), current_date),
            self.landsat.search_scenes(polygon, baseline_date - timedelta(days=90), baseline_date + timedelta(days=90)),
            self.landsat.search_scenes(polygon, current_date - timedelta(days=30), current_date),
            self.gfw.get_glad_alerts(polygon, baseline_date, current_date)
        )

        # 2. Select best scenes
        best_s2_baseline = self._select_best_scene(sentinel2_baseline, baseline_date)
        best_s2_current = self._select_best_scene(sentinel2_current, current_date)
        best_ls_baseline = self._select_best_scene(landsat_baseline, baseline_date)
        best_ls_current = self._select_best_scene(landsat_current, current_date)

        # 3. Download and preprocess imagery
        s2_baseline_img = await self._download_and_preprocess(best_s2_baseline) if best_s2_baseline else None
        s2_current_img = await self._download_and_preprocess(best_s2_current) if best_s2_current else None
        ls_baseline_img = await self._download_and_preprocess(best_ls_baseline) if best_ls_baseline else None
        ls_current_img = await self._download_and_preprocess(best_ls_current) if best_ls_current else None

        # 4. Run change detection
        s2_result = None
        if s2_baseline_img is not None and s2_current_img is not None:
            s2_result = self.model.predict_change(s2_baseline_img, s2_current_img, polygon)

        ls_result = None
        if ls_baseline_img is not None and ls_current_img is not None:
            ls_result = self.model.predict_change(ls_baseline_img, ls_current_img, polygon)

        # 5. Fuse results
        fused = fuse_satellite_sources(s2_result, ls_result, gfw_alerts)

        # 6. Assess quality
        quality = self._assess_overall_quality(
            best_s2_baseline, best_s2_current, best_ls_baseline, best_ls_current
        )

        # 7. Generate result
        return DeforestationResult(
            polygon_id=polygon.id,
            baseline_date=baseline_date,
            verification_date=current_date,
            deforestation_detected=fused.deforestation_detected,
            deforestation_area_ha=fused.deforestation_area_ha,
            status=fused.status,
            confidence=fused.confidence,
            data_quality=quality,
            sources_used=fused.sources_used,
            gfw_alerts_count=len(gfw_alerts),
            compliance_determination=self._determine_compliance(fused)
        )

    def _determine_compliance(self, fused: FusedResult) -> str:
        """Determine EUDR compliance based on satellite analysis."""
        if fused.confidence < 0.5:
            return "INSUFFICIENT_DATA"
        elif fused.deforestation_detected and fused.deforestation_area_ha > 0.1:
            return "NON_COMPLIANT"
        elif fused.status == "POTENTIAL_DEFORESTATION":
            return "MANUAL_REVIEW_REQUIRED"
        else:
            return "COMPLIANT"
```

---

## 6. API Endpoints

### 6.1 Satellite Verification Endpoint

```yaml
/api/v1/verify/satellite:
  post:
    summary: Verify deforestation status using satellite imagery
    requestBody:
      content:
        application/json:
          schema:
            type: object
            required:
              - polygon
            properties:
              polygon:
                type: object
                description: GeoJSON polygon
              baseline_date:
                type: string
                format: date
                default: "2020-12-31"
              priority:
                type: string
                enum: [normal, urgent]
                default: normal
    responses:
      200:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DeforestationResult'
      202:
        description: Processing queued (for large polygons)
      400:
        description: Invalid polygon
      503:
        description: Satellite data unavailable
```

### 6.2 Response Schema

```json
{
  "polygon_id": "uuid",
  "baseline_date": "2020-12-31",
  "verification_date": "2025-12-04",
  "deforestation_detected": false,
  "deforestation_area_ha": 0.0,
  "status": "NO_DEFORESTATION",
  "confidence": 0.92,
  "compliance_determination": "COMPLIANT",
  "data_quality": {
    "cloud_cover_percentage": 12.5,
    "temporal_proximity_days": 15,
    "spatial_coverage_percentage": 98.5,
    "overall_score": 85
  },
  "sources_used": 2,
  "gfw_alerts_count": 0,
  "imagery_metadata": {
    "baseline_scene": "S2A_20201228_T23KPQ",
    "current_scene": "S2B_20251201_T23KPQ"
  }
}
```

---

## 7. Performance Optimization

### 7.1 Caching Strategy

```python
class SatelliteDataCache:
    """Cache for satellite data and results."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.s3 = boto3.client('s3')
        self.bucket = "eudr-satellite-cache"

    async def get_or_compute_baseline(
        self,
        polygon_hash: str,
        compute_func: Callable
    ) -> BaselineResult:
        """
        Get baseline from cache or compute if not available.

        Cache TTL: 90 days (baseline is stable)
        """
        cache_key = f"baseline:{polygon_hash}"

        # Check Redis
        cached = await self.redis.get(cache_key)
        if cached:
            return BaselineResult.from_json(cached)

        # Compute baseline
        result = await compute_func()

        # Cache in Redis (90 days)
        await self.redis.setex(cache_key, 90 * 24 * 3600, result.to_json())

        return result

    async def cache_imagery(
        self,
        scene_id: str,
        imagery: np.ndarray
    ):
        """Cache downloaded imagery to S3."""
        key = f"imagery/{scene_id}.npy"
        buffer = io.BytesIO()
        np.save(buffer, imagery)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket, key)
```

### 7.2 Async Processing

```python
class AsyncSatelliteProcessor:
    """Asynchronous satellite processing with worker pool."""

    def __init__(self, num_workers: int = 4):
        self.queue = asyncio.Queue()
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(num_workers)
        ]

    async def _worker(self):
        """Worker coroutine for processing satellite requests."""
        while True:
            job = await self.queue.get()
            try:
                result = await self._process_job(job)
                await self._notify_completion(job.callback_url, result)
            except Exception as e:
                await self._notify_failure(job.callback_url, e)
            finally:
                self.queue.task_done()

    async def submit(self, job: SatelliteJob) -> str:
        """Submit job for async processing."""
        job_id = str(uuid.uuid4())
        job.id = job_id
        await self.queue.put(job)
        return job_id
```

---

## 8. Error Handling

### 8.1 Error Codes

| Code | Description | Action |
|------|-------------|--------|
| SAT-001 | No imagery available | Use GFW alerts only |
| SAT-002 | Cloud cover too high | Extend date range |
| SAT-003 | API timeout | Retry with backoff |
| SAT-004 | Rate limit exceeded | Queue for later |
| SAT-005 | Invalid polygon | Return validation error |
| SAT-006 | Model inference error | Fallback to NDVI-only |

### 8.2 Fallback Strategy

```python
async def verify_with_fallback(polygon: Polygon) -> DeforestationResult:
    """Verification with graceful degradation."""
    try:
        # Primary: Full multi-source verification
        return await full_verification(polygon)
    except Sentinel2UnavailableError:
        # Fallback 1: Landsat + GFW
        return await landsat_gfw_verification(polygon)
    except LandsatUnavailableError:
        # Fallback 2: GFW alerts only
        return await gfw_only_verification(polygon)
    except GFWUnavailableError:
        # Fallback 3: Manual review required
        return DeforestationResult(
            status="MANUAL_REVIEW_REQUIRED",
            confidence=0.0,
            reason="All satellite sources unavailable"
        )
```

---

## 9. Monitoring and Alerts

### 9.1 Metrics

```yaml
satellite_metrics:
  - name: eudr_satellite_requests_total
    type: counter
    labels: [source, status]

  - name: eudr_satellite_processing_seconds
    type: histogram
    labels: [source]
    buckets: [1, 5, 10, 30, 60, 120]

  - name: eudr_deforestation_detected_total
    type: counter
    labels: [country, commodity]

  - name: eudr_satellite_data_quality_score
    type: gauge
    labels: [source]

  - name: eudr_satellite_api_errors_total
    type: counter
    labels: [source, error_code]
```

### 9.2 Alerts

```yaml
alerts:
  - name: satellite_api_high_error_rate
    condition: rate(eudr_satellite_api_errors_total[5m]) > 0.1
    severity: critical

  - name: satellite_processing_slow
    condition: histogram_quantile(0.95, eudr_satellite_processing_seconds) > 60
    severity: warning

  - name: data_quality_degraded
    condition: avg(eudr_satellite_data_quality_score) < 50
    severity: warning
```

---

## 10. Security Considerations

### 10.1 API Key Management

- Store satellite API keys in HashiCorp Vault
- Rotate keys every 90 days
- Use separate keys for dev/staging/production

### 10.2 Data Privacy

- Do not store raw satellite imagery permanently
- Encrypt cached data at rest (AES-256)
- Anonymize polygon data in logs

### 10.3 Access Control

- Require authentication for all satellite API endpoints
- Rate limit per customer/operator
- Audit log all satellite queries

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-Satellite-ML-Specialist | Initial specification |

---

**END OF DOCUMENT**
