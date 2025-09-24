"""
JSONL (JSON Lines) logger for structured logging of GreenLang runs
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class JSONLLogger:
    """Structured JSONL logger for workflow execution"""

    def __init__(self, output_path: Optional[Path] = None):
        """Initialize JSONL logger

        Args:
            output_path: Path to write JSONL file
        """
        self.output_path = output_path
        self.start_time = time.time()
        self.events = []
        self.file_handle = None

        if output_path:
            self.set_output(output_path)

    def set_output(self, output_path: Path):
        """Set or change the output path

        Args:
            output_path: Path to JSONL file
        """
        # Close existing file if open
        if self.file_handle:
            self.file_handle.close()

        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self.file_handle = open(self.output_path, "a", encoding="utf-8")

    def log_event(self, event_type: str, data: Dict[str, Any], level: str = "INFO"):
        """Log a structured event

        Args:
            event_type: Type of event (start, task_done, error, etc.)
            data: Event data
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": round(time.time() - self.start_time, 3),
            "event_type": event_type,
            "level": level,
            "data": data,
        }

        self.events.append(event)

        # Write to file if configured
        if self.file_handle:
            self.file_handle.write(json.dumps(event) + "\n")
            self.file_handle.flush()  # Ensure immediate write

        return event

    def log_start(self, run_id: str, workflow_name: str, **kwargs):
        """Log workflow start event"""
        return self.log_event(
            "start", {"run_id": run_id, "workflow_name": workflow_name, **kwargs}
        )

    def log_step_start(self, step_name: str, agent_id: str, **kwargs):
        """Log step start event"""
        return self.log_event(
            "step_start", {"step_name": step_name, "agent_id": agent_id, **kwargs}
        )

    def log_step_complete(
        self, step_name: str, success: bool, duration: float, **kwargs
    ):
        """Log step completion event"""
        return self.log_event(
            "step_complete",
            {
                "step_name": step_name,
                "success": success,
                "duration_seconds": round(duration, 3),
                **kwargs,
            },
        )

    def log_error(self, error_message: str, step_name: Optional[str] = None, **kwargs):
        """Log error event"""
        return self.log_event(
            "error",
            {"error_message": error_message, "step_name": step_name, **kwargs},
            level="ERROR",
        )

    def log_complete(self, run_id: str, success: bool, **kwargs):
        """Log workflow completion event"""
        return self.log_event(
            "complete",
            {
                "run_id": run_id,
                "success": success,
                "total_duration_seconds": self.get_duration(),
                "total_events": len(self.events),
                **kwargs,
            },
        )

    def log_metric(
        self, metric_name: str, value: Any, unit: Optional[str] = None, **kwargs
    ):
        """Log a metric/measurement"""
        return self.log_event(
            "metric",
            {"metric_name": metric_name, "value": value, "unit": unit, **kwargs},
        )

    def log_validation(
        self, validation_type: str, passed: bool, details: Optional[Dict] = None
    ):
        """Log validation result"""
        return self.log_event(
            "validation",
            {
                "validation_type": validation_type,
                "passed": passed,
                "details": details or {},
            },
        )

    def get_duration(self) -> float:
        """Get total duration since logger start"""
        return round(time.time() - self.start_time, 3)

    def get_events(self) -> list:
        """Get all logged events"""
        return self.events

    def save(self, output_path: Optional[Path] = None):
        """Save all events to JSONL file

        Args:
            output_path: Optional path to save to (uses configured path if not provided)
        """
        path = output_path or self.output_path
        if not path:
            raise ValueError("No output path configured")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

    def close(self):
        """Close the logger and file handle"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    @staticmethod
    def read_jsonl(file_path: Path) -> list:
        """Read and parse a JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            List of parsed events
        """
        events = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    @staticmethod
    def filter_events(
        events: list, event_type: Optional[str] = None, level: Optional[str] = None
    ) -> list:
        """Filter events by type or level

        Args:
            events: List of events
            event_type: Filter by event type
            level: Filter by log level

        Returns:
            Filtered list of events
        """
        filtered = events

        if event_type:
            filtered = [e for e in filtered if e.get("event_type") == event_type]

        if level:
            filtered = [e for e in filtered if e.get("level") == level]

        return filtered
