"""
Checkpoint Debugger - Tools for debugging and visualizing pipeline checkpoints

This module provides debugging utilities for checkpoint inspection,
validation, and troubleshooting.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass

from greenlang.pipeline.checkpointing import (
    CheckpointManager,
    CheckpointStrategy,
    CheckpointStatus,
    PipelineCheckpoint,
    CheckpointMetadata
)

logger = logging.getLogger(__name__)


@dataclass
class CheckpointAnalysis:
    """Analysis results for checkpoint health."""
    total_checkpoints: int
    healthy_checkpoints: int
    corrupted_checkpoints: int
    orphaned_checkpoints: int
    storage_size_bytes: int
    average_checkpoint_size: float
    oldest_checkpoint: Optional[datetime]
    newest_checkpoint: Optional[datetime]
    pipelines_tracked: int
    resume_success_rate: float
    common_failure_points: List[str]
    recommendations: List[str]


class CheckpointDebugger:
    """
    Debugging and visualization tools for pipeline checkpoints.
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        """Initialize debugger with checkpoint manager."""
        self.manager = checkpoint_manager
        logger.info("Initialized checkpoint debugger")

    def inspect_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Deep inspection of a checkpoint.

        Args:
            checkpoint_id: Checkpoint to inspect

        Returns:
            Detailed checkpoint information
        """
        checkpoint = self.manager.storage.load(checkpoint_id)

        if not checkpoint:
            return {"error": f"Checkpoint {checkpoint_id} not found"}

        # Basic info
        info = {
            "checkpoint_id": checkpoint_id,
            "metadata": checkpoint.metadata.to_dict(),
            "completed_stages": checkpoint.completed_stages,
            "pending_stages": checkpoint.pending_stages,
            "state_data_keys": list(checkpoint.state_data.keys()),
            "agent_outputs_keys": list(checkpoint.agent_outputs.keys()),
            "provenance_hashes": checkpoint.provenance_hashes
        }

        # Data sizes
        info["data_sizes"] = {
            "state_data_size": len(json.dumps(checkpoint.state_data, default=str)),
            "agent_outputs_size": len(json.dumps(checkpoint.agent_outputs, default=str)),
            "total_size": checkpoint.metadata.data_size
        }

        # Validation
        info["validation"] = self._validate_checkpoint(checkpoint)

        # Relationships
        if checkpoint.metadata.parent_checkpoint_id:
            info["parent_checkpoint"] = checkpoint.metadata.parent_checkpoint_id

        # Find child checkpoints
        all_checkpoints = self.manager.storage.list_checkpoints(checkpoint.metadata.pipeline_id)
        children = [
            c for c in all_checkpoints
            if c.parent_checkpoint_id == checkpoint_id
        ]
        info["child_checkpoints"] = [c.to_dict() for c in children]

        return info

    def _validate_checkpoint(self, checkpoint: PipelineCheckpoint) -> Dict[str, Any]:
        """Validate checkpoint integrity."""
        validation = {
            "is_valid": True,
            "issues": []
        }

        # Check checksum
        calculated_checksum = checkpoint.calculate_checksum()
        if checkpoint.metadata.checksum != calculated_checksum:
            validation["is_valid"] = False
            validation["issues"].append("Checksum mismatch - data may be corrupted")

        # Check data completeness
        if not checkpoint.state_data:
            validation["issues"].append("Missing state data")

        if not checkpoint.completed_stages and checkpoint.metadata.stage_index > 0:
            validation["issues"].append("No completed stages but stage index > 0")

        # Check provenance
        for stage in checkpoint.completed_stages:
            if stage not in checkpoint.provenance_hashes:
                validation["issues"].append(f"Missing provenance hash for stage {stage}")

        return validation

    def analyze_pipeline_checkpoints(self, pipeline_id: str) -> CheckpointAnalysis:
        """
        Analyze all checkpoints for a pipeline.

        Args:
            pipeline_id: Pipeline to analyze

        Returns:
            Analysis results
        """
        checkpoints = self.manager.storage.list_checkpoints(pipeline_id)

        if not checkpoints:
            return CheckpointAnalysis(
                total_checkpoints=0,
                healthy_checkpoints=0,
                corrupted_checkpoints=0,
                orphaned_checkpoints=0,
                storage_size_bytes=0,
                average_checkpoint_size=0,
                oldest_checkpoint=None,
                newest_checkpoint=None,
                pipelines_tracked=0,
                resume_success_rate=0,
                common_failure_points=[],
                recommendations=["No checkpoints found for pipeline"]
            )

        # Analyze checkpoints
        healthy = 0
        corrupted = 0
        orphaned = 0
        total_size = 0
        failure_points = {}
        resume_attempts = 0
        successful_resumes = 0

        for checkpoint_meta in checkpoints:
            # Try to load full checkpoint
            checkpoint_id = self._construct_checkpoint_id(checkpoint_meta)
            checkpoint = self.manager.storage.load(checkpoint_id)

            if checkpoint:
                validation = self._validate_checkpoint(checkpoint)
                if validation["is_valid"]:
                    healthy += 1
                else:
                    corrupted += 1

                total_size += checkpoint_meta.data_size

                # Track failures
                if checkpoint_meta.status == CheckpointStatus.FAILED:
                    stage = checkpoint_meta.stage_name
                    failure_points[stage] = failure_points.get(stage, 0) + 1

                # Track resumes
                if checkpoint_meta.resume_count > 0:
                    resume_attempts += checkpoint_meta.resume_count
                    if checkpoint_meta.status == CheckpointStatus.COMPLETED:
                        successful_resumes += 1
            else:
                orphaned += 1

        # Calculate statistics
        avg_size = total_size / len(checkpoints) if checkpoints else 0
        resume_rate = successful_resumes / resume_attempts if resume_attempts > 0 else 0

        # Sort checkpoints by time
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x.timestamp)

        # Common failure points
        common_failures = sorted(
            failure_points.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Recommendations
        recommendations = []
        if corrupted > 0:
            recommendations.append(f"Fix {corrupted} corrupted checkpoints")
        if orphaned > 0:
            recommendations.append(f"Clean up {orphaned} orphaned checkpoints")
        if resume_rate < 0.8 and resume_attempts > 0:
            recommendations.append("Low resume success rate - investigate failures")
        if avg_size > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Large checkpoint sizes - consider compression")

        return CheckpointAnalysis(
            total_checkpoints=len(checkpoints),
            healthy_checkpoints=healthy,
            corrupted_checkpoints=corrupted,
            orphaned_checkpoints=orphaned,
            storage_size_bytes=total_size,
            average_checkpoint_size=avg_size,
            oldest_checkpoint=sorted_checkpoints[0].timestamp,
            newest_checkpoint=sorted_checkpoints[-1].timestamp,
            pipelines_tracked=1,
            resume_success_rate=resume_rate,
            common_failure_points=[f[0] for f in common_failures],
            recommendations=recommendations
        )

    def _construct_checkpoint_id(self, metadata: CheckpointMetadata) -> str:
        """Construct checkpoint ID from metadata."""
        return f"{metadata.pipeline_id}_{metadata.stage_name}_{metadata.timestamp.strftime('%Y%m%d_%H%M%S')}"

    def visualize_checkpoint_graph(self, pipeline_id: str, output_path: Optional[str] = None):
        """
        Create a visual graph of checkpoint relationships.

        Args:
            pipeline_id: Pipeline to visualize
            output_path: Optional path to save graph image
        """
        checkpoints = self.manager.storage.list_checkpoints(pipeline_id)

        if not checkpoints:
            logger.warning(f"No checkpoints found for pipeline {pipeline_id}")
            return

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for checkpoint in checkpoints:
            checkpoint_id = self._construct_checkpoint_id(checkpoint)
            color = {
                CheckpointStatus.COMPLETED: 'green',
                CheckpointStatus.IN_PROGRESS: 'yellow',
                CheckpointStatus.FAILED: 'red',
                CheckpointStatus.RESUMED: 'blue',
                CheckpointStatus.EXPIRED: 'gray'
            }.get(checkpoint.status, 'white')

            G.add_node(
                checkpoint_id,
                label=checkpoint.stage_name,
                color=color,
                status=checkpoint.status.value
            )

            # Add edge from parent
            if checkpoint.parent_checkpoint_id:
                G.add_edge(checkpoint.parent_checkpoint_id, checkpoint_id)

        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        node_colors = [G.nodes[node].get('color', 'white') for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)

        # Draw labels
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title(f"Checkpoint Graph for Pipeline: {pipeline_id}")
        plt.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Completed'),
            Patch(facecolor='yellow', label='In Progress'),
            Patch(facecolor='red', label='Failed'),
            Patch(facecolor='blue', label='Resumed'),
            Patch(facecolor='gray', label='Expired')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved checkpoint graph to {output_path}")
        else:
            plt.show()

    def find_recovery_point(self, pipeline_id: str) -> Optional[str]:
        """
        Find the best checkpoint to recover from.

        Args:
            pipeline_id: Pipeline to analyze

        Returns:
            Best checkpoint ID for recovery
        """
        checkpoints = self.manager.storage.list_checkpoints(pipeline_id)

        if not checkpoints:
            return None

        # Filter to valid checkpoints
        valid_checkpoints = []
        for checkpoint_meta in checkpoints:
            if checkpoint_meta.status in [CheckpointStatus.COMPLETED, CheckpointStatus.IN_PROGRESS]:
                checkpoint_id = self._construct_checkpoint_id(checkpoint_meta)
                checkpoint = self.manager.storage.load(checkpoint_id)

                if checkpoint:
                    validation = self._validate_checkpoint(checkpoint)
                    if validation["is_valid"]:
                        valid_checkpoints.append((checkpoint_meta, checkpoint))

        if not valid_checkpoints:
            return None

        # Sort by timestamp (newest first) and stage index (highest first)
        valid_checkpoints.sort(
            key=lambda x: (x[0].timestamp, x[0].stage_index),
            reverse=True
        )

        # Return the most recent valid checkpoint
        best_checkpoint = valid_checkpoints[0]
        return self._construct_checkpoint_id(best_checkpoint[0])

    def compare_checkpoints(self, checkpoint_id1: str, checkpoint_id2: str) -> Dict[str, Any]:
        """
        Compare two checkpoints to identify differences.

        Args:
            checkpoint_id1: First checkpoint
            checkpoint_id2: Second checkpoint

        Returns:
            Comparison results
        """
        checkpoint1 = self.manager.storage.load(checkpoint_id1)
        checkpoint2 = self.manager.storage.load(checkpoint_id2)

        if not checkpoint1:
            return {"error": f"Checkpoint {checkpoint_id1} not found"}
        if not checkpoint2:
            return {"error": f"Checkpoint {checkpoint_id2} not found"}

        comparison = {
            "checkpoint1": checkpoint_id1,
            "checkpoint2": checkpoint_id2,
            "differences": {}
        }

        # Compare metadata
        if checkpoint1.metadata.stage_index != checkpoint2.metadata.stage_index:
            comparison["differences"]["stage_index"] = {
                "checkpoint1": checkpoint1.metadata.stage_index,
                "checkpoint2": checkpoint2.metadata.stage_index
            }

        # Compare completed stages
        stages1 = set(checkpoint1.completed_stages)
        stages2 = set(checkpoint2.completed_stages)

        if stages1 != stages2:
            comparison["differences"]["completed_stages"] = {
                "only_in_checkpoint1": list(stages1 - stages2),
                "only_in_checkpoint2": list(stages2 - stages1)
            }

        # Compare state data keys
        keys1 = set(checkpoint1.state_data.keys())
        keys2 = set(checkpoint2.state_data.keys())

        if keys1 != keys2:
            comparison["differences"]["state_data_keys"] = {
                "only_in_checkpoint1": list(keys1 - keys2),
                "only_in_checkpoint2": list(keys2 - keys1)
            }

        # Compare data sizes
        size_diff = checkpoint1.metadata.data_size - checkpoint2.metadata.data_size
        if abs(size_diff) > 1024:  # More than 1KB difference
            comparison["differences"]["data_size"] = {
                "checkpoint1": checkpoint1.metadata.data_size,
                "checkpoint2": checkpoint2.metadata.data_size,
                "difference": size_diff
            }

        return comparison

    def repair_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Attempt to repair a corrupted checkpoint.

        Args:
            checkpoint_id: Checkpoint to repair

        Returns:
            Success indicator
        """
        checkpoint = self.manager.storage.load(checkpoint_id)

        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        validation = self._validate_checkpoint(checkpoint)

        if validation["is_valid"]:
            logger.info(f"Checkpoint {checkpoint_id} is already valid")
            return True

        # Attempt repairs
        repaired = False

        for issue in validation["issues"]:
            if "Checksum mismatch" in issue:
                # Recalculate and update checksum
                checkpoint.metadata.checksum = checkpoint.calculate_checksum()
                logger.info(f"Repaired checksum for {checkpoint_id}")
                repaired = True

            elif "Missing provenance hash" in issue:
                # Generate placeholder provenance hashes
                for stage in checkpoint.completed_stages:
                    if stage not in checkpoint.provenance_hashes:
                        checkpoint.provenance_hashes[stage] = "repaired_" + stage
                logger.info(f"Added placeholder provenance hashes for {checkpoint_id}")
                repaired = True

        if repaired:
            # Save repaired checkpoint
            success = self.manager.storage.save(checkpoint_id, checkpoint)
            if success:
                logger.info(f"Successfully repaired checkpoint {checkpoint_id}")
            return success

        logger.warning(f"Unable to repair checkpoint {checkpoint_id}")
        return False

    def generate_recovery_report(self, pipeline_id: str) -> str:
        """
        Generate a detailed recovery report for a pipeline.

        Args:
            pipeline_id: Pipeline to analyze

        Returns:
            Formatted recovery report
        """
        analysis = self.analyze_pipeline_checkpoints(pipeline_id)
        recovery_point = self.find_recovery_point(pipeline_id)

        report = []
        report.append("=" * 60)
        report.append(f"CHECKPOINT RECOVERY REPORT")
        report.append(f"Pipeline: {pipeline_id}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 60)

        report.append("\nCHECKPOINT STATISTICS:")
        report.append(f"  Total Checkpoints: {analysis.total_checkpoints}")
        report.append(f"  Healthy: {analysis.healthy_checkpoints}")
        report.append(f"  Corrupted: {analysis.corrupted_checkpoints}")
        report.append(f"  Orphaned: {analysis.orphaned_checkpoints}")

        report.append("\nSTORAGE METRICS:")
        report.append(f"  Total Size: {analysis.storage_size_bytes / 1024 / 1024:.2f} MB")
        report.append(f"  Average Size: {analysis.average_checkpoint_size / 1024:.2f} KB")

        if analysis.oldest_checkpoint:
            report.append(f"  Oldest: {analysis.oldest_checkpoint.isoformat()}")
            report.append(f"  Newest: {analysis.newest_checkpoint.isoformat()}")

        report.append("\nRECOVERY ANALYSIS:")
        if recovery_point:
            report.append(f"  Recommended Recovery Point: {recovery_point}")

            # Load recovery checkpoint for details
            checkpoint = self.manager.storage.load(recovery_point)
            if checkpoint:
                report.append(f"  Stage: {checkpoint.metadata.stage_name}")
                report.append(f"  Completed Stages: {len(checkpoint.completed_stages)}")
                report.append(f"  Pending Stages: {len(checkpoint.pending_stages)}")
        else:
            report.append("  No valid recovery point found")

        report.append(f"\n  Resume Success Rate: {analysis.resume_success_rate:.1%}")

        if analysis.common_failure_points:
            report.append("\nCOMMON FAILURE POINTS:")
            for stage in analysis.common_failure_points[:3]:
                report.append(f"  - {stage}")

        if analysis.recommendations:
            report.append("\nRECOMMENDATIONS:")
            for rec in analysis.recommendations:
                report.append(f"  - {rec}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def monitor_checkpoints(self, refresh_seconds: int = 5):
        """
        Monitor checkpoints in real-time.

        Args:
            refresh_seconds: Refresh interval
        """
        import time
        import os

        print("Starting checkpoint monitor (Ctrl+C to stop)...")

        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')

                # Get all checkpoints
                all_checkpoints = self.manager.storage.list_checkpoints()

                # Group by pipeline
                pipelines = {}
                for checkpoint in all_checkpoints:
                    pid = checkpoint.pipeline_id
                    if pid not in pipelines:
                        pipelines[pid] = []
                    pipelines[pid].append(checkpoint)

                # Display dashboard
                print("=" * 80)
                print(f"CHECKPOINT MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)

                for pipeline_id, checkpoints in pipelines.items():
                    print(f"\nPipeline: {pipeline_id}")
                    print("-" * 40)

                    # Sort by timestamp
                    checkpoints.sort(key=lambda x: x.timestamp, reverse=True)

                    for checkpoint in checkpoints[:5]:  # Show last 5
                        status_symbol = {
                            CheckpointStatus.COMPLETED: "✓",
                            CheckpointStatus.IN_PROGRESS: "⟳",
                            CheckpointStatus.FAILED: "✗",
                            CheckpointStatus.RESUMED: "↺",
                            CheckpointStatus.EXPIRED: "⌛"
                        }.get(checkpoint.status, "?")

                        age = datetime.now() - checkpoint.timestamp
                        age_str = f"{age.total_seconds():.0f}s" if age.total_seconds() < 3600 else f"{age.total_seconds()/3600:.1f}h"

                        print(f"  {status_symbol} {checkpoint.stage_name:20} "
                             f"[{checkpoint.status.value:10}] "
                             f"Age: {age_str:8} "
                             f"Size: {checkpoint.data_size/1024:.1f}KB")

                        if checkpoint.error_message:
                            print(f"     Error: {checkpoint.error_message[:50]}...")

                print(f"\n\nRefreshing in {refresh_seconds} seconds...")
                time.sleep(refresh_seconds)

        except KeyboardInterrupt:
            print("\nMonitoring stopped")


def main():
    """Command-line interface for checkpoint debugging."""
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Debugger")
    parser.add_argument("command", choices=["inspect", "analyze", "visualize", "repair", "monitor", "report"])
    parser.add_argument("--pipeline-id", help="Pipeline ID")
    parser.add_argument("--checkpoint-id", help="Checkpoint ID")
    parser.add_argument("--strategy", default="file", help="Storage strategy")
    parser.add_argument("--base-path", default="/tmp/greenlang/checkpoints", help="Base path for file storage")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Create checkpoint manager
    manager = CheckpointManager(
        strategy=args.strategy,
        base_path=args.base_path
    )

    # Create debugger
    debugger = CheckpointDebugger(manager)

    # Execute command
    if args.command == "inspect":
        if not args.checkpoint_id:
            print("Error: --checkpoint-id required")
            return
        result = debugger.inspect_checkpoint(args.checkpoint_id)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "analyze":
        if not args.pipeline_id:
            print("Error: --pipeline-id required")
            return
        analysis = debugger.analyze_pipeline_checkpoints(args.pipeline_id)
        print(f"Analysis Results:")
        for key, value in analysis.__dict__.items():
            print(f"  {key}: {value}")

    elif args.command == "visualize":
        if not args.pipeline_id:
            print("Error: --pipeline-id required")
            return
        debugger.visualize_checkpoint_graph(args.pipeline_id, args.output)

    elif args.command == "repair":
        if not args.checkpoint_id:
            print("Error: --checkpoint-id required")
            return
        success = debugger.repair_checkpoint(args.checkpoint_id)
        print(f"Repair {'successful' if success else 'failed'}")

    elif args.command == "monitor":
        debugger.monitor_checkpoints()

    elif args.command == "report":
        if not args.pipeline_id:
            print("Error: --pipeline-id required")
            return
        report = debugger.generate_recovery_report(args.pipeline_id)
        print(report)


if __name__ == "__main__":
    main()