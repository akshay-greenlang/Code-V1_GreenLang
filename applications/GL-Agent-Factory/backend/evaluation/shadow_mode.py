"""
Shadow Mode Testing for GreenLang Agent Factory

Shadow mode allows running agents against production traffic without affecting
actual operations. This enables:
1. Safe validation of new agent versions
2. A/B comparison between versions
3. Performance benchmarking under real load
4. Regression detection before promotion

Example:
    >>> from evaluation.shadow_mode import ShadowModeRunner, ShadowConfig
    >>> config = ShadowConfig(mode=ShadowMode.PARALLEL, comparison_tolerance=0.001)
    >>> runner = ShadowModeRunner(config)
    >>> report = await runner.run_parallel_shadow(baseline, candidate, traffic)
    >>> print(f"Match rate: {report.match_rate:.2%}")
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ShadowMode(Enum):
    """Shadow mode operation types."""
    DISABLED = "disabled"
    RECORD = "record"
    REPLAY = "replay"
    PARALLEL = "parallel"
    COMPARE = "compare"


@dataclass
class ShadowConfig:
    """Configuration for shadow mode testing."""
    mode: ShadowMode = ShadowMode.DISABLED
    traffic_percentage: float = 100.0
    comparison_tolerance: float = 0.001
    timeout_ms: int = 5000
    max_parallel_requests: int = 100
    record_path: Optional[str] = None
    baseline_agent_id: Optional[str] = None
    candidate_agent_id: Optional[str] = None
    max_records: int = 0
    save_differences: bool = True
    difference_output_path: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.traffic_percentage <= 100.0:
            raise ValueError(f"traffic_percentage must be 0-100")
        if self.comparison_tolerance < 0:
            raise ValueError(f"comparison_tolerance must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "traffic_percentage": self.traffic_percentage,
            "comparison_tolerance": self.comparison_tolerance,
            "timeout_ms": self.timeout_ms,
            "max_parallel_requests": self.max_parallel_requests,
        }


@dataclass
class TrafficRecord:
    """A single traffic record for shadow testing."""
    request_id: str
    timestamp: datetime
    input_data: Dict[str, Any]
    input_hash: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None

    @classmethod
    def create(cls, input_data: Dict[str, Any], source: str = "production") -> "TrafficRecord":
        input_json = json.dumps(input_data, sort_keys=True, default=str)
        input_hash = hashlib.sha256(input_json.encode()).hexdigest()
        return cls(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            input_data=input_data,
            input_hash=input_hash,
            source=source,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "input_hash": self.input_hash,
            "source": self.source,
        }


@dataclass
class ShadowResult:
    """Result from a single shadow comparison."""
    request_id: str
    baseline_output: Optional[Dict[str, Any]]
    candidate_output: Optional[Dict[str, Any]]
    baseline_latency_ms: float
    candidate_latency_ms: float
    outputs_match: bool
    differences: List[Dict[str, Any]]
    baseline_error: Optional[str]
    candidate_error: Optional[str]
    input_hash: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_errors(self) -> bool:
        return self.baseline_error is not None or self.candidate_error is not None

    @property
    def latency_improvement_percent(self) -> float:
        if self.baseline_latency_ms == 0:
            return 0.0
        return ((self.baseline_latency_ms - self.candidate_latency_ms) / self.baseline_latency_ms) * 100


@dataclass
class ShadowReport:
    """Comprehensive shadow test report."""
    session_id: str
    started_at: datetime
    completed_at: datetime
    total_requests: int
    matching_outputs: int
    mismatched_outputs: int
    baseline_errors: int
    candidate_errors: int
    avg_baseline_latency_ms: float
    avg_candidate_latency_ms: float
    latency_improvement_percent: float
    match_rate: float
    differences_summary: Dict[str, int]
    recommendation: str
    config: Optional[ShadowConfig] = None
    results: List[ShadowResult] = field(default_factory=list)

    @property
    def candidate_error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.candidate_errors / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "total_requests": self.total_requests,
            "match_rate": self.match_rate,
            "recommendation": self.recommendation,
        }


class OutputComparator:
    """Compare agent outputs with configurable tolerance."""

    def __init__(self, tolerance: float = 0.001):
        self.tolerance = tolerance

    def compare(self, baseline: Optional[Dict], candidate: Optional[Dict]) -> Tuple[bool, List[Dict]]:
        differences = []
        if baseline is None and candidate is None:
            return True, []
        if baseline is None or candidate is None:
            return False, [{"type": "none_mismatch", "baseline": baseline, "candidate": candidate}]

        self._compare_values(baseline, candidate, "$", differences)
        return len(differences) == 0, differences

    def _compare_values(self, baseline: Any, candidate: Any, path: str, differences: List[Dict]):
        if type(baseline) != type(candidate):
            differences.append({"path": path, "type": "type_mismatch"})
            return

        if isinstance(baseline, dict):
            all_keys = set(baseline.keys()) | set(candidate.keys())
            for key in all_keys:
                if key not in baseline:
                    differences.append({"path": f"{path}.{key}", "type": "missing_in_baseline"})
                elif key not in candidate:
                    differences.append({"path": f"{path}.{key}", "type": "missing_in_candidate"})
                else:
                    self._compare_values(baseline[key], candidate[key], f"{path}.{key}", differences)
        elif isinstance(baseline, list):
            if len(baseline) != len(candidate):
                differences.append({"path": path, "type": "length_mismatch"})
            for i in range(min(len(baseline), len(candidate))):
                self._compare_values(baseline[i], candidate[i], f"{path}[{i}]", differences)
        elif isinstance(baseline, (int, float)):
            if not self._numbers_equal(baseline, candidate):
                differences.append({"path": path, "type": "value_mismatch", "baseline": baseline, "candidate": candidate})
        else:
            if baseline != candidate:
                differences.append({"path": path, "type": "value_mismatch"})

    def _numbers_equal(self, a, b) -> bool:
        if a == b:
            return True
        if a == 0:
            return abs(b) <= self.tolerance
        return abs(a - b) / abs(a) <= self.tolerance


class ShadowModeRunner:
    """Execute shadow mode testing for agent validation."""

    PROMOTE_MATCH_RATE = 0.99
    INVESTIGATE_MATCH_RATE = 0.95
    MAX_ERROR_RATE = 0.01

    def __init__(self, config: ShadowConfig):
        self.config = config
        self.results: List[ShadowResult] = []
        self.comparator = OutputComparator(config.comparison_tolerance)
        self._session_id = ""
        self._start_time: Optional[datetime] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def run_parallel_shadow(
        self,
        baseline_agent: Any,
        candidate_agent: Any,
        traffic_source: AsyncIterator[TrafficRecord],
    ) -> ShadowReport:
        """Run candidate agent in parallel with baseline."""
        self._session_id = str(uuid.uuid4())
        self._start_time = datetime.utcnow()
        self.results = []
        self._semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

        tasks = []
        record_count = 0

        async for record in traffic_source:
            if self.config.max_records > 0 and record_count >= self.config.max_records:
                break
            record_count += 1
            task = asyncio.create_task(
                self._process_single_request(record, baseline_agent, candidate_agent)
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return self.generate_report()

    async def _process_single_request(
        self,
        record: TrafficRecord,
        baseline_agent: Any,
        candidate_agent: Any,
    ) -> ShadowResult:
        async with self._semaphore:
            baseline_task = asyncio.create_task(self._run_agent(baseline_agent, record.input_data))
            candidate_task = asyncio.create_task(self._run_agent(candidate_agent, record.input_data))

            baseline_result, candidate_result = await asyncio.gather(
                baseline_task, candidate_task, return_exceptions=True
            )

            if isinstance(baseline_result, Exception):
                baseline_output, baseline_latency, baseline_error = None, 0.0, str(baseline_result)
            else:
                baseline_output, baseline_latency, baseline_error = baseline_result

            if isinstance(candidate_result, Exception):
                candidate_output, candidate_latency, candidate_error = None, 0.0, str(candidate_result)
            else:
                candidate_output, candidate_latency, candidate_error = candidate_result

            match, differences = self.comparator.compare(baseline_output, candidate_output)

            result = ShadowResult(
                request_id=record.request_id,
                baseline_output=baseline_output,
                candidate_output=candidate_output,
                baseline_latency_ms=baseline_latency,
                candidate_latency_ms=candidate_latency,
                outputs_match=match,
                differences=differences,
                baseline_error=baseline_error,
                candidate_error=candidate_error,
                input_hash=record.input_hash,
            )
            self.results.append(result)
            return result

    async def _run_agent(self, agent: Any, input_data: Dict) -> Tuple[Optional[Dict], float, Optional[str]]:
        start_time = time.perf_counter()
        output = None
        error = None

        try:
            timeout_seconds = self.config.timeout_ms / 1000.0
            if hasattr(agent, "arun"):
                output = await asyncio.wait_for(agent.arun(input_data), timeout=timeout_seconds)
            elif hasattr(agent, "run"):
                loop = asyncio.get_event_loop()
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, agent.run, input_data), timeout=timeout_seconds
                )
            if output is not None and not isinstance(output, dict):
                output = {"result": output}
        except asyncio.TimeoutError:
            error = f"Timeout after {self.config.timeout_ms}ms"
        except Exception as e:
            error = str(e)

        latency_ms = (time.perf_counter() - start_time) * 1000
        return output, latency_ms, error

    def generate_report(self) -> ShadowReport:
        completed_at = datetime.utcnow()

        if not self.results:
            return ShadowReport(
                session_id=self._session_id,
                started_at=self._start_time or completed_at,
                completed_at=completed_at,
                total_requests=0, matching_outputs=0, mismatched_outputs=0,
                baseline_errors=0, candidate_errors=0,
                avg_baseline_latency_ms=0.0, avg_candidate_latency_ms=0.0,
                latency_improvement_percent=0.0, match_rate=0.0,
                differences_summary={}, recommendation="NO_DATA",
            )

        total = len(self.results)
        matching = sum(1 for r in self.results if r.outputs_match)
        baseline_errors = sum(1 for r in self.results if r.baseline_error)
        candidate_errors = sum(1 for r in self.results if r.candidate_error)

        baseline_latencies = [r.baseline_latency_ms for r in self.results if not r.baseline_error]
        candidate_latencies = [r.candidate_latency_ms for r in self.results if not r.candidate_error]

        avg_baseline = sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0
        avg_candidate = sum(candidate_latencies) / len(candidate_latencies) if candidate_latencies else 0
        improvement = ((avg_baseline - avg_candidate) / avg_baseline * 100) if avg_baseline > 0 else 0

        match_rate = matching / total if total > 0 else 0

        # Generate recommendation
        error_rate = candidate_errors / total if total > 0 else 0
        if error_rate > self.MAX_ERROR_RATE:
            recommendation = "REJECT"
        elif match_rate >= self.PROMOTE_MATCH_RATE:
            recommendation = "PROMOTE"
        elif match_rate >= self.INVESTIGATE_MATCH_RATE:
            recommendation = "INVESTIGATE"
        else:
            recommendation = "REJECT"

        return ShadowReport(
            session_id=self._session_id,
            started_at=self._start_time or completed_at,
            completed_at=completed_at,
            total_requests=total,
            matching_outputs=matching,
            mismatched_outputs=total - matching,
            baseline_errors=baseline_errors,
            candidate_errors=candidate_errors,
            avg_baseline_latency_ms=round(avg_baseline, 2),
            avg_candidate_latency_ms=round(avg_candidate, 2),
            latency_improvement_percent=round(improvement, 2),
            match_rate=round(match_rate, 4),
            differences_summary={},
            recommendation=recommendation,
            config=self.config,
            results=self.results,
        )


class TrafficRecorder:
    """Record and manage traffic for shadow testing."""

    def __init__(self, base_path: str):
        """Initialize traffic recorder with storage path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._current_session: Optional[str] = None
        self._records: List[TrafficRecord] = []
        self._seen_hashes: set = set()

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new recording session."""
        self._current_session = session_id or str(uuid.uuid4())
        self._records = []
        self._seen_hashes = set()
        session_dir = self.base_path / self._current_session
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Started recording session: {self._current_session}")
        return self._current_session

    def end_session(self) -> int:
        """End current session and save records to disk."""
        if not self._current_session:
            return 0

        session_dir = self.base_path / self._current_session
        records_file = session_dir / "traffic.jsonl"

        with open(records_file, "w") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")

        metadata = {
            "session_id": self._current_session,
            "record_count": len(self._records),
            "started_at": self._records[0].timestamp.isoformat() if self._records else None,
            "ended_at": datetime.utcnow().isoformat(),
        }
        meta_file = session_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        count = len(self._records)
        logger.info(f"Ended session {self._current_session} with {count} records")
        self._current_session = None
        self._records = []
        return count

    async def record(
        self,
        record: TrafficRecord,
        deduplicate: bool = True,
    ) -> bool:
        """Record a traffic record."""
        if not self._current_session:
            logger.warning("No active session, starting new one")
            self.start_session()

        if deduplicate and record.input_hash in self._seen_hashes:
            logger.debug(f"Skipping duplicate record: {record.input_hash[:8]}")
            return False

        self._records.append(record)
        self._seen_hashes.add(record.input_hash)
        logger.debug(f"Recorded: {record.request_id}")
        return True

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get list of available recorded sessions."""
        sessions = []
        if not self.base_path.exists():
            return sessions

        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                meta_file = session_dir / "metadata.json"
                traffic_file = session_dir / "traffic.jsonl"

                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata = json.load(f)
                        metadata["path"] = str(session_dir)
                        sessions.append(metadata)
                elif traffic_file.exists():
                    # Count lines for record count
                    with open(traffic_file) as f:
                        count = sum(1 for _ in f)
                    sessions.append({
                        "session_id": session_dir.name,
                        "record_count": count,
                        "path": str(session_dir),
                    })

        return sorted(sessions, key=lambda x: x.get("ended_at", ""), reverse=True)

    async def load_session(self, session_id: str) -> AsyncIterator[TrafficRecord]:
        """Load and yield traffic records from a session."""
        session_dir = self.base_path / session_id
        traffic_file = session_dir / "traffic.jsonl"

        if not traffic_file.exists():
            logger.error(f"Traffic file not found: {traffic_file}")
            return

        with open(traffic_file) as f:
            for line in f:
                data = json.loads(line)
                yield TrafficRecord(
                    request_id=data["request_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    input_data=data["input_data"],
                    input_hash=data["input_hash"],
                    source=data["source"],
                )


class ShadowModeEvaluator:
    """Certification dimension evaluator for shadow mode results."""

    PASS_THRESHOLD = 70.0

    def evaluate(self, report: ShadowReport) -> Dict[str, Any]:
        score = 0.0

        # Match rate (40%)
        if report.match_rate >= 0.99:
            score += 40
        elif report.match_rate >= 0.95:
            score += 30
        elif report.match_rate >= 0.90:
            score += 20

        # Latency (30%)
        if report.latency_improvement_percent >= 0:
            score += 30
        elif report.latency_improvement_percent >= -10:
            score += 22.5

        # Error rate (30%)
        if report.candidate_error_rate <= 0.001:
            score += 30
        elif report.candidate_error_rate <= 0.01:
            score += 22.5

        return {
            "dimension": "shadow_mode",
            "score": score,
            "passed": score >= self.PASS_THRESHOLD,
            "recommendation": report.recommendation,
        }


# Convenience functions
def create_shadow_config(mode: str = "parallel", tolerance: float = 0.001) -> ShadowConfig:
    return ShadowConfig(mode=ShadowMode(mode.lower()), comparison_tolerance=tolerance)


async def run_shadow_test(
    baseline_agent: Any,
    candidate_agent: Any,
    traffic: List[Dict[str, Any]],
    config: Optional[ShadowConfig] = None,
) -> ShadowReport:
    config = config or ShadowConfig(mode=ShadowMode.PARALLEL)
    runner = ShadowModeRunner(config)

    async def traffic_source():
        for input_data in traffic:
            yield TrafficRecord.create(input_data)

    return await runner.run_parallel_shadow(baseline_agent, candidate_agent, traffic_source())
