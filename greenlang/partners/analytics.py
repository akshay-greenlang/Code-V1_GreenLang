"""
Usage Analytics and Tracking for GreenLang Partners

This module provides comprehensive usage tracking and analytics for partners,
including request metrics, agent usage, performance monitoring, and billing data.

Features:
- Real-time usage tracking
- Hourly, daily, and monthly aggregations
- Performance metrics (response time, error rates)
- Agent usage statistics
- Time-series data storage
- Analytics API endpoints
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
from collections import defaultdict

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import FastAPI, Depends, Query
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
Base = declarative_base()


class TimeRange(str, Enum):
    """Time range for analytics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class MetricType(str, Enum):
    """Metric types"""
    REQUEST_COUNT = "request_count"
    ERROR_COUNT = "error_count"
    RESPONSE_TIME = "response_time"
    DATA_TRANSFER = "data_transfer"
    AGENT_USAGE = "agent_usage"


# Database Models
class MetricAggregationModel(Base):
    """Aggregated metrics model"""
    __tablename__ = "metric_aggregations"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    period = Column(String, nullable=False)  # hour, day, month
    metric_type = Column(String, nullable=False)

    # Request metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    error_4xx = Column(Integer, default=0)
    error_5xx = Column(Integer, default=0)

    # Performance metrics
    avg_response_time_ms = Column(Float, default=0.0)
    p50_response_time_ms = Column(Float, default=0.0)
    p95_response_time_ms = Column(Float, default=0.0)
    p99_response_time_ms = Column(Float, default=0.0)
    min_response_time_ms = Column(Float, default=0.0)
    max_response_time_ms = Column(Float, default=0.0)

    # Data transfer metrics
    total_request_bytes = Column(Integer, default=0)
    total_response_bytes = Column(Integer, default=0)
    total_data_bytes = Column(Integer, default=0)

    # Agent metrics
    agent_breakdown = Column(JSON, default=dict)  # agent_id -> count
    endpoint_breakdown = Column(JSON, default=dict)  # endpoint -> count
    status_breakdown = Column(JSON, default=dict)  # status_code -> count

    # Additional metadata
    metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_metrics_partner_time', 'partner_id', 'timestamp'),
        Index('idx_metrics_period', 'period'),
        Index('idx_metrics_type', 'metric_type'),
    )


class AgentUsageModel(Base):
    """Agent usage tracking"""
    __tablename__ = "agent_usage"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    agent_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Usage metrics
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    avg_duration_ms = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)  # in dollars

    # Performance
    avg_confidence = Column(Float, default=0.0)
    citation_count = Column(Integer, default=0)

    # Indexes
    __table_args__ = (
        Index('idx_agent_usage_partner', 'partner_id'),
        Index('idx_agent_usage_agent', 'agent_id'),
        Index('idx_agent_usage_time', 'timestamp'),
    )


# Pydantic Models
class MetricPoint(BaseModel):
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = {}


class TimeSeriesData(BaseModel):
    """Time series data"""
    metric_type: str
    period: str
    data_points: List[MetricPoint]
    summary: Dict[str, float]


class AnalyticsRequest(BaseModel):
    """Analytics query request"""
    start_date: datetime
    end_date: datetime
    metrics: List[MetricType]
    granularity: TimeRange = TimeRange.DAY
    filters: Dict[str, Any] = {}


class AnalyticsResponse(BaseModel):
    """Analytics response"""
    partner_id: str
    period_start: datetime
    period_end: datetime
    metrics: Dict[str, TimeSeriesData]
    summary: Dict[str, Any]


class AgentUsageStats(BaseModel):
    """Agent usage statistics"""
    agent_id: str
    agent_name: str
    total_executions: int
    success_rate: float
    avg_duration_ms: float
    total_cost: float
    usage_trend: List[MetricPoint]


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    success_rate: float
    uptime_percentage: float


# Dataclasses
@dataclass
class UsageEvent:
    """Usage event for tracking"""
    partner_id: str
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: int
    request_bytes: int
    response_bytes: int
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalyticsEngine:
    """
    Analytics engine for processing and aggregating metrics
    """

    def __init__(self, db: Session):
        self.db = db

    def track_usage_event(self, event: UsageEvent):
        """
        Track a usage event

        Args:
            event: Usage event to track
        """
        from .api import UsageRecordModel
        import uuid

        # Create usage record
        usage_record = UsageRecordModel(
            id=f"usage_{uuid.uuid4().hex[:16]}",
            partner_id=event.partner_id,
            timestamp=event.timestamp,
            endpoint=event.endpoint,
            method=event.method,
            status_code=event.status_code,
            response_time_ms=event.response_time_ms,
            request_size_bytes=event.request_bytes,
            response_size_bytes=event.response_bytes,
            metadata={
                'agent_id': event.agent_id,
                'workflow_id': event.workflow_id,
                **event.metadata
            }
        )

        self.db.add(usage_record)
        self.db.commit()

        logger.debug(f"Tracked usage event for partner {event.partner_id}")

    def aggregate_metrics(
        self,
        partner_id: str,
        start_time: datetime,
        end_time: datetime,
        period: str = "hour"
    ) -> List[MetricAggregationModel]:
        """
        Aggregate metrics for a time period

        Args:
            partner_id: Partner ID
            start_time: Start of period
            end_time: End of period
            period: Aggregation period (hour, day, month)

        Returns:
            List of aggregated metrics
        """
        from .api import UsageRecordModel

        # Query usage records
        usage_records = self.db.query(UsageRecordModel).filter(
            UsageRecordModel.partner_id == partner_id,
            UsageRecordModel.timestamp >= start_time,
            UsageRecordModel.timestamp < end_time
        ).all()

        if not usage_records:
            return []

        # Group by time period
        time_groups = self._group_by_period(usage_records, period)

        aggregations = []
        for timestamp, records in time_groups.items():
            agg = self._compute_aggregation(partner_id, timestamp, period, records)
            aggregations.append(agg)

        return aggregations

    def _group_by_period(
        self,
        records: List,
        period: str
    ) -> Dict[datetime, List]:
        """Group records by time period"""
        groups = defaultdict(list)

        for record in records:
            # Truncate timestamp to period
            if period == "hour":
                key = record.timestamp.replace(minute=0, second=0, microsecond=0)
            elif period == "day":
                key = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == "month":
                key = record.timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                key = record.timestamp

            groups[key].append(record)

        return dict(groups)

    def _compute_aggregation(
        self,
        partner_id: str,
        timestamp: datetime,
        period: str,
        records: List
    ) -> MetricAggregationModel:
        """Compute aggregated metrics from records"""
        import uuid

        # Request counts
        total_requests = len(records)
        successful = sum(1 for r in records if 200 <= r.status_code < 300)
        failed = total_requests - successful
        error_4xx = sum(1 for r in records if 400 <= r.status_code < 500)
        error_5xx = sum(1 for r in records if r.status_code >= 500)

        # Response times
        response_times = [r.response_time_ms for r in records]
        response_times_sorted = sorted(response_times)

        avg_response = np.mean(response_times) if response_times else 0
        p50 = np.percentile(response_times, 50) if response_times else 0
        p95 = np.percentile(response_times, 95) if response_times else 0
        p99 = np.percentile(response_times, 99) if response_times else 0
        min_response = min(response_times) if response_times else 0
        max_response = max(response_times) if response_times else 0

        # Data transfer
        total_request_bytes = sum(r.request_size_bytes for r in records)
        total_response_bytes = sum(r.response_size_bytes for r in records)
        total_data_bytes = total_request_bytes + total_response_bytes

        # Agent breakdown
        agent_breakdown = defaultdict(int)
        for r in records:
            agent_id = r.metadata.get('agent_id')
            if agent_id:
                agent_breakdown[agent_id] += 1

        # Endpoint breakdown
        endpoint_breakdown = defaultdict(int)
        for r in records:
            endpoint_breakdown[r.endpoint] += 1

        # Status breakdown
        status_breakdown = defaultdict(int)
        for r in records:
            status_breakdown[str(r.status_code)] += 1

        # Create aggregation
        agg = MetricAggregationModel(
            id=f"agg_{uuid.uuid4().hex[:16]}",
            partner_id=partner_id,
            timestamp=timestamp,
            period=period,
            metric_type=MetricType.REQUEST_COUNT.value,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            error_4xx=error_4xx,
            error_5xx=error_5xx,
            avg_response_time_ms=float(avg_response),
            p50_response_time_ms=float(p50),
            p95_response_time_ms=float(p95),
            p99_response_time_ms=float(p99),
            min_response_time_ms=float(min_response),
            max_response_time_ms=float(max_response),
            total_request_bytes=total_request_bytes,
            total_response_bytes=total_response_bytes,
            total_data_bytes=total_data_bytes,
            agent_breakdown=dict(agent_breakdown),
            endpoint_breakdown=dict(endpoint_breakdown),
            status_breakdown=dict(status_breakdown)
        )

        self.db.add(agg)
        self.db.commit()

        return agg

    def get_analytics(
        self,
        partner_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeRange = TimeRange.DAY
    ) -> AnalyticsResponse:
        """
        Get analytics for partner

        Args:
            partner_id: Partner ID
            start_date: Start date
            end_date: End date
            granularity: Data granularity

        Returns:
            Analytics response
        """
        # Query aggregations
        aggregations = self.db.query(MetricAggregationModel).filter(
            MetricAggregationModel.partner_id == partner_id,
            MetricAggregationModel.timestamp >= start_date,
            MetricAggregationModel.timestamp <= end_date,
            MetricAggregationModel.period == granularity.value
        ).order_by(MetricAggregationModel.timestamp).all()

        # Build time series data
        metrics = {}

        # Request count time series
        request_data = TimeSeriesData(
            metric_type="requests",
            period=granularity.value,
            data_points=[
                MetricPoint(
                    timestamp=agg.timestamp,
                    value=agg.total_requests,
                    metadata={
                        'successful': agg.successful_requests,
                        'failed': agg.failed_requests
                    }
                )
                for agg in aggregations
            ],
            summary={
                'total': sum(agg.total_requests for agg in aggregations),
                'avg': np.mean([agg.total_requests for agg in aggregations]) if aggregations else 0
            }
        )
        metrics['requests'] = request_data

        # Response time time series
        response_time_data = TimeSeriesData(
            metric_type="response_time",
            period=granularity.value,
            data_points=[
                MetricPoint(
                    timestamp=agg.timestamp,
                    value=agg.avg_response_time_ms,
                    metadata={
                        'p50': agg.p50_response_time_ms,
                        'p95': agg.p95_response_time_ms,
                        'p99': agg.p99_response_time_ms
                    }
                )
                for agg in aggregations
            ],
            summary={
                'avg': np.mean([agg.avg_response_time_ms for agg in aggregations]) if aggregations else 0,
                'p95': np.percentile([agg.p95_response_time_ms for agg in aggregations], 95) if aggregations else 0
            }
        )
        metrics['response_time'] = response_time_data

        # Summary statistics
        total_requests = sum(agg.total_requests for agg in aggregations)
        successful_requests = sum(agg.successful_requests for agg in aggregations)

        summary = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': sum(agg.failed_requests for agg in aggregations),
            'success_rate': (successful_requests / total_requests) if total_requests > 0 else 0,
            'avg_response_time_ms': np.mean([agg.avg_response_time_ms for agg in aggregations]) if aggregations else 0,
            'total_data_mb': sum(agg.total_data_bytes for agg in aggregations) / (1024 * 1024)
        }

        return AnalyticsResponse(
            partner_id=partner_id,
            period_start=start_date,
            period_end=end_date,
            metrics=metrics,
            summary=summary
        )

    def get_agent_usage_stats(
        self,
        partner_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[AgentUsageStats]:
        """
        Get agent usage statistics

        Args:
            partner_id: Partner ID
            start_date: Start date
            end_date: End date

        Returns:
            List of agent usage statistics
        """
        from .api import UsageRecordModel

        # Query usage by agent
        usage_records = self.db.query(UsageRecordModel).filter(
            UsageRecordModel.partner_id == partner_id,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp <= end_date
        ).all()

        # Group by agent
        agent_usage = defaultdict(list)
        for record in usage_records:
            agent_id = record.metadata.get('agent_id')
            if agent_id:
                agent_usage[agent_id].append(record)

        # Compute statistics per agent
        stats = []
        for agent_id, records in agent_usage.items():
            total = len(records)
            successful = sum(1 for r in records if 200 <= r.status_code < 300)
            avg_duration = np.mean([r.response_time_ms for r in records])

            # Usage trend (daily aggregation)
            daily_counts = defaultdict(int)
            for record in records:
                day = record.timestamp.date()
                daily_counts[day] += 1

            trend = [
                MetricPoint(timestamp=datetime.combine(day, datetime.min.time()), value=count)
                for day, count in sorted(daily_counts.items())
            ]

            stats.append(AgentUsageStats(
                agent_id=agent_id,
                agent_name=agent_id.replace('_', ' ').title(),
                total_executions=total,
                success_rate=successful / total if total > 0 else 0,
                avg_duration_ms=float(avg_duration),
                total_cost=0.0,  # Calculate based on pricing
                usage_trend=trend
            ))

        return sorted(stats, key=lambda x: x.total_executions, reverse=True)


# Background task for metric aggregation
async def aggregate_metrics_task(db: Session):
    """Background task to aggregate metrics hourly"""
    engine = AnalyticsEngine(db)

    # Get all partners
    from .api import PartnerModel
    partners = db.query(PartnerModel).all()

    # Aggregate for last hour
    end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=1)

    for partner in partners:
        try:
            engine.aggregate_metrics(
                partner.id,
                start_time,
                end_time,
                period="hour"
            )
            logger.info(f"Aggregated metrics for partner {partner.id}")
        except Exception as e:
            logger.error(f"Error aggregating metrics for {partner.id}: {e}")


if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("postgresql://localhost/greenlang_partners")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    analytics = AnalyticsEngine(db)

    # Track event
    event = UsageEvent(
        partner_id="partner_123",
        timestamp=datetime.utcnow(),
        endpoint="/api/workflows/execute",
        method="POST",
        status_code=200,
        response_time_ms=1500,
        request_bytes=1024,
        response_bytes=2048,
        agent_id="carbon_analyzer"
    )
    analytics.track_usage_event(event)

    # Get analytics
    response = analytics.get_analytics(
        "partner_123",
        datetime.utcnow() - timedelta(days=30),
        datetime.utcnow(),
        TimeRange.DAY
    )
    print(response)
