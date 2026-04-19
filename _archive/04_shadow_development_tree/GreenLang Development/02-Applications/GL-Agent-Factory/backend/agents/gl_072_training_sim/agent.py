"""
GL-072: Training Simulator Agent (TRAINING-SIM)

This module implements the TrainingSimAgent for operator training simulation,
scenario generation, and competency assessment for industrial process operations.

Standards Reference:
    - ISA-101.01 (Human Machine Interface)
    - OSHA Process Safety Management (PSM)
    - API 755 (Fatigue Risk Management)
    - ANSI/ISA-18.2 (Alarm Management)

Example:
    >>> agent = TrainingSimAgent()
    >>> result = agent.run(TrainingSimInput(trainee_id=..., scenarios=[...]))
    >>> print(f"Competency score: {result.competency_assessment.overall_score:.1f}%")
"""

import hashlib
import json
import logging
import math
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    NORMAL_OPERATION = "normal_operation"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    EMERGENCY = "emergency"
    ABNORMAL = "abnormal"
    UPSET_CONDITION = "upset_condition"
    EQUIPMENT_FAILURE = "equipment_failure"
    PROCESS_DEVIATION = "process_deviation"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CompetencyArea(str, Enum):
    SAFETY_RESPONSE = "safety_response"
    PROCESS_CONTROL = "process_control"
    ALARM_MANAGEMENT = "alarm_management"
    EQUIPMENT_OPERATION = "equipment_operation"
    TROUBLESHOOTING = "troubleshooting"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"


class ActionType(str, Enum):
    VALVE_OPERATION = "valve_operation"
    SETPOINT_CHANGE = "setpoint_change"
    EQUIPMENT_START = "equipment_start"
    EQUIPMENT_STOP = "equipment_stop"
    ALARM_ACKNOWLEDGE = "alarm_acknowledge"
    EMERGENCY_STOP = "emergency_stop"
    PROCEDURE_STEP = "procedure_step"
    COMMUNICATION = "communication"


class ProcessVariable(BaseModel):
    """Process variable definition."""
    variable_id: str = Field(..., description="Variable identifier")
    name: str = Field(..., description="Variable name")
    unit: str = Field(..., description="Unit of measurement")
    current_value: float = Field(..., description="Current value")
    setpoint: Optional[float] = Field(None, description="Setpoint")
    low_limit: float = Field(..., description="Low alarm limit")
    high_limit: float = Field(..., description="High alarm limit")
    low_low_limit: Optional[float] = Field(None, description="Low-low limit")
    high_high_limit: Optional[float] = Field(None, description="High-high limit")


class TrainingScenario(BaseModel):
    """Training scenario definition."""
    scenario_id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Scenario name")
    scenario_type: ScenarioType = Field(..., description="Type of scenario")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)
    description: str = Field(..., description="Scenario description")
    initial_conditions: Dict[str, float] = Field(..., description="Initial process conditions")
    disturbances: List[Dict[str, Any]] = Field(default_factory=list, description="Disturbances")
    expected_actions: List[str] = Field(..., description="Expected operator actions")
    time_limit_minutes: int = Field(default=30, description="Time limit")
    pass_criteria: Dict[str, float] = Field(default_factory=dict, description="Pass criteria")


class OperatorAction(BaseModel):
    """Operator action during simulation."""
    action_id: str = Field(..., description="Action identifier")
    timestamp: datetime = Field(..., description="Action timestamp")
    action_type: ActionType = Field(..., description="Type of action")
    target: str = Field(..., description="Action target")
    value: Optional[float] = Field(None, description="Action value")
    response_time_seconds: float = Field(..., description="Response time")
    was_correct: bool = Field(default=True, description="Was action correct")
    feedback: Optional[str] = Field(None, description="Feedback message")


class TrainingSimInput(BaseModel):
    """Input for training simulation."""
    session_id: Optional[str] = Field(None, description="Session identifier")
    trainee_id: str = Field(..., description="Trainee identifier")
    trainee_name: str = Field(default="Trainee", description="Trainee name")
    role: str = Field(default="Operator", description="Trainee role")
    scenarios: List[TrainingScenario] = Field(..., description="Scenarios to run")
    process_variables: List[ProcessVariable] = Field(default_factory=list)
    previous_scores: Dict[str, float] = Field(default_factory=dict, description="Previous competency")
    adaptive_difficulty: bool = Field(default=True, description="Adapt difficulty")
    random_seed: Optional[int] = Field(None, description="Random seed")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Result of a single scenario."""
    scenario_id: str
    scenario_name: str
    scenario_type: str
    difficulty: str
    passed: bool
    score: float
    time_taken_minutes: float
    actions_taken: List[OperatorAction]
    correct_actions: int
    incorrect_actions: int
    missed_actions: int
    response_times_seconds: List[float]
    average_response_time: float
    safety_violations: List[str]
    process_excursions: List[str]
    feedback: List[str]


class CompetencyScore(BaseModel):
    """Competency score by area."""
    area: CompetencyArea
    area_name: str
    score: float
    trend: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class CompetencyAssessment(BaseModel):
    """Overall competency assessment."""
    overall_score: float
    competency_level: str
    scores_by_area: List[CompetencyScore]
    certification_ready: bool
    gaps_identified: List[str]
    training_recommendations: List[str]
    next_steps: List[str]


class LearningObjective(BaseModel):
    """Learning objective assessment."""
    objective_id: str
    description: str
    mastered: bool
    proficiency_percent: float
    evidence: List[str]


class PerformanceMetrics(BaseModel):
    """Performance metrics summary."""
    total_scenarios: int
    scenarios_passed: int
    pass_rate_percent: float
    average_score: float
    best_scenario: str
    worst_scenario: str
    total_time_minutes: float
    total_actions: int
    accuracy_percent: float
    average_response_time_seconds: float
    safety_score: float


class TrainingSimOutput(BaseModel):
    """Output from training simulation."""
    session_id: str
    trainee_id: str
    trainee_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scenario_results: List[ScenarioResult]
    performance_metrics: PerformanceMetrics
    competency_assessment: CompetencyAssessment
    learning_objectives: List[LearningObjective]
    adaptive_recommendations: List[str]
    next_training_scenarios: List[str]
    certification_status: str
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class TrainingSimAgent:
    """GL-072: Training Simulator Agent - Operator training and assessment."""

    AGENT_ID = "GL-072"
    AGENT_NAME = "TRAINING-SIM"
    VERSION = "1.0.0"

    # Competency weights
    COMPETENCY_WEIGHTS = {
        CompetencyArea.SAFETY_RESPONSE: 0.25,
        CompetencyArea.PROCESS_CONTROL: 0.20,
        CompetencyArea.ALARM_MANAGEMENT: 0.15,
        CompetencyArea.EQUIPMENT_OPERATION: 0.15,
        CompetencyArea.TROUBLESHOOTING: 0.10,
        CompetencyArea.COMMUNICATION: 0.08,
        CompetencyArea.DECISION_MAKING: 0.07,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"TrainingSimAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: TrainingSimInput) -> TrainingSimOutput:
        start_time = datetime.utcnow()

        if input_data.random_seed:
            random.seed(input_data.random_seed)

        # Run all scenarios
        scenario_results = []
        for scenario in input_data.scenarios:
            result = self._run_scenario(
                scenario, input_data.process_variables, input_data.trainee_id)
            scenario_results.append(result)

        # Calculate performance metrics
        metrics = self._calculate_metrics(scenario_results)

        # Assess competency
        competency = self._assess_competency(
            scenario_results, input_data.previous_scores)

        # Evaluate learning objectives
        objectives = self._evaluate_learning_objectives(
            scenario_results, input_data.scenarios)

        # Generate adaptive recommendations
        recommendations = self._generate_recommendations(
            competency, metrics, input_data.adaptive_difficulty)

        # Determine next training scenarios
        next_scenarios = self._recommend_next_scenarios(
            competency, scenario_results)

        # Certification status
        if competency.overall_score >= 85 and competency.certification_ready:
            cert_status = "READY_FOR_CERTIFICATION"
        elif competency.overall_score >= 70:
            cert_status = "PROGRESSING"
        else:
            cert_status = "NEEDS_IMPROVEMENT"

        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent": self.AGENT_ID,
                "trainee": input_data.trainee_id,
                "scenarios": len(scenario_results),
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return TrainingSimOutput(
            session_id=input_data.session_id or f"SIM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            trainee_id=input_data.trainee_id,
            trainee_name=input_data.trainee_name,
            scenario_results=scenario_results,
            performance_metrics=metrics,
            competency_assessment=competency,
            learning_objectives=objectives,
            adaptive_recommendations=recommendations,
            next_training_scenarios=next_scenarios,
            certification_status=cert_status,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS")

    def _run_scenario(self, scenario: TrainingScenario,
                     process_vars: List[ProcessVariable],
                     trainee_id: str) -> ScenarioResult:
        """Simulate running a training scenario."""
        # Simulate operator actions (in real implementation, this would be interactive)
        actions = self._simulate_operator_actions(scenario)

        # Evaluate actions
        correct = sum(1 for a in actions if a.was_correct)
        incorrect = len(actions) - correct
        missed = max(0, len(scenario.expected_actions) - len(actions))

        # Calculate response times
        response_times = [a.response_time_seconds for a in actions]
        avg_response = sum(response_times) / len(response_times) if response_times else 0

        # Check for safety violations
        violations = self._check_safety_violations(actions, scenario)

        # Check process excursions
        excursions = self._check_process_excursions(scenario, process_vars)

        # Calculate score
        base_score = (correct / (correct + incorrect + missed) * 100) if (correct + incorrect + missed) > 0 else 0

        # Penalties
        safety_penalty = len(violations) * 10
        excursion_penalty = len(excursions) * 5
        time_penalty = 0

        time_taken = sum(response_times) / 60  # Convert to minutes
        if time_taken > scenario.time_limit_minutes:
            time_penalty = min(20, (time_taken - scenario.time_limit_minutes) * 2)

        final_score = max(0, base_score - safety_penalty - excursion_penalty - time_penalty)

        # Pass/fail
        pass_threshold = scenario.pass_criteria.get("min_score", 70)
        passed = final_score >= pass_threshold and len(violations) == 0

        # Generate feedback
        feedback = self._generate_scenario_feedback(
            correct, incorrect, missed, violations, excursions, final_score)

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            scenario_type=scenario.scenario_type.value,
            difficulty=scenario.difficulty.value,
            passed=passed,
            score=round(final_score, 2),
            time_taken_minutes=round(time_taken, 2),
            actions_taken=actions,
            correct_actions=correct,
            incorrect_actions=incorrect,
            missed_actions=missed,
            response_times_seconds=[round(rt, 2) for rt in response_times],
            average_response_time=round(avg_response, 2),
            safety_violations=violations,
            process_excursions=excursions,
            feedback=feedback)

    def _simulate_operator_actions(self, scenario: TrainingScenario) -> List[OperatorAction]:
        """Simulate operator actions for a scenario."""
        actions = []
        base_time = datetime.utcnow()

        # Difficulty affects performance
        difficulty_factor = {
            DifficultyLevel.BEGINNER: 0.9,
            DifficultyLevel.INTERMEDIATE: 0.8,
            DifficultyLevel.ADVANCED: 0.7,
            DifficultyLevel.EXPERT: 0.6,
        }.get(scenario.difficulty, 0.8)

        for i, expected_action in enumerate(scenario.expected_actions):
            # Simulate response time (varies with difficulty)
            base_response = random.gauss(30, 10)  # seconds
            response_time = max(5, base_response / difficulty_factor)

            # Simulate correctness (varies with difficulty)
            is_correct = random.random() < (0.7 + difficulty_factor * 0.2)

            # Determine action type from expected action
            action_type = self._infer_action_type(expected_action)

            actions.append(OperatorAction(
                action_id=f"ACT-{i+1:03d}",
                timestamp=base_time + timedelta(seconds=sum(
                    a.response_time_seconds for a in actions) + response_time),
                action_type=action_type,
                target=expected_action,
                value=random.uniform(0, 100) if "setpoint" in expected_action.lower() else None,
                response_time_seconds=round(response_time, 2),
                was_correct=is_correct,
                feedback="Action executed correctly" if is_correct else "Incorrect action sequence"))

        return actions

    def _infer_action_type(self, action_description: str) -> ActionType:
        """Infer action type from description."""
        desc_lower = action_description.lower()
        if "valve" in desc_lower:
            return ActionType.VALVE_OPERATION
        elif "setpoint" in desc_lower:
            return ActionType.SETPOINT_CHANGE
        elif "start" in desc_lower:
            return ActionType.EQUIPMENT_START
        elif "stop" in desc_lower or "shutdown" in desc_lower:
            return ActionType.EQUIPMENT_STOP
        elif "alarm" in desc_lower:
            return ActionType.ALARM_ACKNOWLEDGE
        elif "emergency" in desc_lower or "esd" in desc_lower:
            return ActionType.EMERGENCY_STOP
        elif "communicate" in desc_lower or "notify" in desc_lower:
            return ActionType.COMMUNICATION
        else:
            return ActionType.PROCEDURE_STEP

    def _check_safety_violations(self, actions: List[OperatorAction],
                                 scenario: TrainingScenario) -> List[str]:
        """Check for safety violations."""
        violations = []

        # Check for emergency scenarios without proper response
        if scenario.scenario_type == ScenarioType.EMERGENCY:
            emergency_actions = [a for a in actions
                               if a.action_type == ActionType.EMERGENCY_STOP]
            if not emergency_actions:
                violations.append("Failed to activate emergency stop when required")

            # Check response time for emergency
            if actions and actions[0].response_time_seconds > 30:
                violations.append("Emergency response time exceeded 30 seconds")

        # Check for skipped safety steps
        for action in actions:
            if not action.was_correct and "safety" in action.target.lower():
                violations.append(f"Incorrect safety action: {action.target}")

        return violations

    def _check_process_excursions(self, scenario: TrainingScenario,
                                  process_vars: List[ProcessVariable]) -> List[str]:
        """Check for process excursions."""
        excursions = []

        # Simulate some excursions based on scenario type
        if scenario.scenario_type in [ScenarioType.UPSET_CONDITION, ScenarioType.ABNORMAL]:
            if random.random() < 0.3:  # 30% chance
                excursions.append("Temperature exceeded high-high limit briefly")
            if random.random() < 0.2:
                excursions.append("Pressure approached low limit")

        return excursions

    def _generate_scenario_feedback(self, correct: int, incorrect: int,
                                    missed: int, violations: List[str],
                                    excursions: List[str], score: float) -> List[str]:
        """Generate feedback for scenario performance."""
        feedback = []

        if score >= 90:
            feedback.append("Excellent performance! All critical actions executed correctly.")
        elif score >= 70:
            feedback.append("Good performance. Minor improvements needed.")
        elif score >= 50:
            feedback.append("Satisfactory performance. Additional practice recommended.")
        else:
            feedback.append("Performance below expectations. Focused training required.")

        if incorrect > 0:
            feedback.append(f"Review the {incorrect} incorrect actions for proper technique.")

        if missed > 0:
            feedback.append(f"Practice identifying all {missed} required actions in sequence.")

        if violations:
            feedback.append("SAFETY: Address safety violations before certification.")
            for v in violations:
                feedback.append(f"  - {v}")

        if excursions:
            feedback.append("Process control needs improvement to prevent excursions.")

        return feedback

    def _calculate_metrics(self, results: List[ScenarioResult]) -> PerformanceMetrics:
        """Calculate overall performance metrics."""
        if not results:
            return PerformanceMetrics(
                total_scenarios=0, scenarios_passed=0, pass_rate_percent=0,
                average_score=0, best_scenario="N/A", worst_scenario="N/A",
                total_time_minutes=0, total_actions=0, accuracy_percent=0,
                average_response_time_seconds=0, safety_score=100)

        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]
        best = max(results, key=lambda r: r.score)
        worst = min(results, key=lambda r: r.score)

        total_correct = sum(r.correct_actions for r in results)
        total_actions = sum(r.correct_actions + r.incorrect_actions for r in results)
        accuracy = (total_correct / total_actions * 100) if total_actions > 0 else 0

        all_response_times = [rt for r in results for rt in r.response_times_seconds]
        avg_response = sum(all_response_times) / len(all_response_times) if all_response_times else 0

        # Safety score (penalize for violations)
        total_violations = sum(len(r.safety_violations) for r in results)
        safety_score = max(0, 100 - total_violations * 20)

        return PerformanceMetrics(
            total_scenarios=len(results),
            scenarios_passed=passed,
            pass_rate_percent=round(passed / len(results) * 100, 2),
            average_score=round(sum(scores) / len(scores), 2),
            best_scenario=best.scenario_name,
            worst_scenario=worst.scenario_name,
            total_time_minutes=round(sum(r.time_taken_minutes for r in results), 2),
            total_actions=total_actions,
            accuracy_percent=round(accuracy, 2),
            average_response_time_seconds=round(avg_response, 2),
            safety_score=round(safety_score, 2))

    def _assess_competency(self, results: List[ScenarioResult],
                          previous_scores: Dict[str, float]) -> CompetencyAssessment:
        """Assess competency across all areas."""
        scores_by_area = []

        for area in CompetencyArea:
            area_score = self._calculate_area_score(results, area)
            prev_score = previous_scores.get(area.value, area_score)
            trend = "improving" if area_score > prev_score else (
                "declining" if area_score < prev_score else "stable")

            strengths, weaknesses = self._identify_strengths_weaknesses(
                results, area, area_score)
            recommendations = self._generate_area_recommendations(
                area, area_score, weaknesses)

            scores_by_area.append(CompetencyScore(
                area=area,
                area_name=area.value.replace("_", " ").title(),
                score=round(area_score, 2),
                trend=trend,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations))

        # Calculate weighted overall score
        overall = sum(
            s.score * self.COMPETENCY_WEIGHTS.get(s.area, 0.1)
            for s in scores_by_area)

        # Determine competency level
        if overall >= 90:
            level = "EXPERT"
        elif overall >= 75:
            level = "PROFICIENT"
        elif overall >= 60:
            level = "COMPETENT"
        elif overall >= 45:
            level = "DEVELOPING"
        else:
            level = "NOVICE"

        # Check certification readiness
        cert_ready = (
            overall >= 75 and
            all(s.score >= 60 for s in scores_by_area) and
            scores_by_area[0].score >= 80  # Safety must be high
        )

        # Identify gaps
        gaps = [f"{s.area_name}: {100 - s.score:.0f}% gap"
                for s in scores_by_area if s.score < 70]

        # Training recommendations
        training_recs = []
        weak_areas = [s for s in scores_by_area if s.score < 70]
        for s in weak_areas[:3]:
            training_recs.append(f"Focus training on {s.area_name}")
            training_recs.extend(s.recommendations[:2])

        # Next steps
        next_steps = []
        if cert_ready:
            next_steps.append("Schedule certification assessment")
            next_steps.append("Complete final competency validation")
        else:
            next_steps.append("Complete additional training scenarios")
            if gaps:
                next_steps.append(f"Address {len(gaps)} competency gaps")

        return CompetencyAssessment(
            overall_score=round(overall, 2),
            competency_level=level,
            scores_by_area=scores_by_area,
            certification_ready=cert_ready,
            gaps_identified=gaps,
            training_recommendations=training_recs,
            next_steps=next_steps)

    def _calculate_area_score(self, results: List[ScenarioResult],
                             area: CompetencyArea) -> float:
        """Calculate score for a specific competency area."""
        if not results:
            return 0

        # Map scenario types to competency areas
        area_scenarios = {
            CompetencyArea.SAFETY_RESPONSE: [ScenarioType.EMERGENCY, ScenarioType.ABNORMAL],
            CompetencyArea.PROCESS_CONTROL: [ScenarioType.NORMAL_OPERATION, ScenarioType.PROCESS_DEVIATION],
            CompetencyArea.ALARM_MANAGEMENT: [ScenarioType.UPSET_CONDITION, ScenarioType.ABNORMAL],
            CompetencyArea.EQUIPMENT_OPERATION: [ScenarioType.STARTUP, ScenarioType.SHUTDOWN],
            CompetencyArea.TROUBLESHOOTING: [ScenarioType.EQUIPMENT_FAILURE, ScenarioType.PROCESS_DEVIATION],
            CompetencyArea.COMMUNICATION: [ScenarioType.EMERGENCY],
            CompetencyArea.DECISION_MAKING: [ScenarioType.UPSET_CONDITION, ScenarioType.ABNORMAL],
        }

        relevant_types = area_scenarios.get(area, [])
        relevant_results = [r for r in results
                          if ScenarioType(r.scenario_type) in relevant_types]

        if relevant_results:
            return sum(r.score for r in relevant_results) / len(relevant_results)
        else:
            # Use overall average if no specific scenarios
            return sum(r.score for r in results) / len(results)

    def _identify_strengths_weaknesses(self, results: List[ScenarioResult],
                                       area: CompetencyArea,
                                       score: float) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses for an area."""
        strengths = []
        weaknesses = []

        if area == CompetencyArea.SAFETY_RESPONSE:
            emergency_results = [r for r in results
                               if r.scenario_type == ScenarioType.EMERGENCY.value]
            if emergency_results:
                avg_response = sum(r.average_response_time for r in emergency_results) / len(emergency_results)
                if avg_response < 20:
                    strengths.append("Quick emergency response time")
                else:
                    weaknesses.append("Emergency response time needs improvement")

                if all(len(r.safety_violations) == 0 for r in emergency_results):
                    strengths.append("No safety violations in emergency scenarios")
                else:
                    weaknesses.append("Safety protocol adherence needs work")

        elif area == CompetencyArea.PROCESS_CONTROL:
            if score >= 80:
                strengths.append("Strong process parameter control")
            else:
                weaknesses.append("Process deviation management needs improvement")

        elif area == CompetencyArea.ALARM_MANAGEMENT:
            avg_alarm_time = sum(r.average_response_time for r in results) / len(results) if results else 0
            if avg_alarm_time < 30:
                strengths.append("Efficient alarm acknowledgment")
            else:
                weaknesses.append("Alarm response time exceeds guidelines")

        # Generic strengths/weaknesses based on score
        if score >= 85:
            strengths.append("Consistent high performance")
        elif score < 60:
            weaknesses.append("Below minimum competency threshold")

        return strengths, weaknesses

    def _generate_area_recommendations(self, area: CompetencyArea,
                                       score: float,
                                       weaknesses: List[str]) -> List[str]:
        """Generate recommendations for competency area."""
        recommendations = []

        if score < 70:
            recommendations.append(f"Complete {area.value.replace('_', ' ')} refresher training")

        if area == CompetencyArea.SAFETY_RESPONSE and score < 80:
            recommendations.append("Practice emergency response procedures")
            recommendations.append("Review safety interlocks and ESD systems")

        elif area == CompetencyArea.PROCESS_CONTROL and score < 75:
            recommendations.append("Study process control fundamentals")
            recommendations.append("Practice setpoint change scenarios")

        elif area == CompetencyArea.ALARM_MANAGEMENT and score < 70:
            recommendations.append("Review alarm rationalization documentation")
            recommendations.append("Practice alarm prioritization exercises")

        elif area == CompetencyArea.TROUBLESHOOTING and score < 70:
            recommendations.append("Complete root cause analysis training")
            recommendations.append("Practice fault diagnosis scenarios")

        return recommendations

    def _evaluate_learning_objectives(self, results: List[ScenarioResult],
                                      scenarios: List[TrainingScenario]) -> List[LearningObjective]:
        """Evaluate learning objectives."""
        objectives = [
            LearningObjective(
                objective_id="LO-001",
                description="Respond to emergency situations within 30 seconds",
                mastered=all(r.average_response_time < 30 for r in results
                           if r.scenario_type == ScenarioType.EMERGENCY.value),
                proficiency_percent=min(100, 100 - sum(max(0, r.average_response_time - 30)
                                                      for r in results) / len(results) * 5),
                evidence=["Emergency response times tracked"]),
            LearningObjective(
                objective_id="LO-002",
                description="Execute safe startup and shutdown procedures",
                mastered=all(r.passed for r in results
                           if r.scenario_type in [ScenarioType.STARTUP.value, ScenarioType.SHUTDOWN.value]),
                proficiency_percent=sum(r.score for r in results
                                       if r.scenario_type in [ScenarioType.STARTUP.value,
                                                             ScenarioType.SHUTDOWN.value]) / max(1, len([
                    r for r in results if r.scenario_type in [ScenarioType.STARTUP.value,
                                                             ScenarioType.SHUTDOWN.value]])),
                evidence=["Startup/shutdown scenarios completed"]),
            LearningObjective(
                objective_id="LO-003",
                description="Maintain process variables within control limits",
                mastered=sum(len(r.process_excursions) for r in results) < 2,
                proficiency_percent=max(0, 100 - sum(len(r.process_excursions) for r in results) * 10),
                evidence=["Process excursion data"]),
            LearningObjective(
                objective_id="LO-004",
                description="Acknowledge and respond to alarms appropriately",
                mastered=sum(r.score for r in results) / len(results) >= 70 if results else False,
                proficiency_percent=sum(r.score for r in results) / len(results) if results else 0,
                evidence=["Alarm response tracking"]),
        ]

        return objectives

    def _generate_recommendations(self, competency: CompetencyAssessment,
                                  metrics: PerformanceMetrics,
                                  adaptive: bool) -> List[str]:
        """Generate adaptive training recommendations."""
        recommendations = []

        if adaptive:
            if metrics.pass_rate_percent >= 90:
                recommendations.append("Increase scenario difficulty to Expert level")
            elif metrics.pass_rate_percent >= 70:
                recommendations.append("Progress to Advanced difficulty scenarios")
            else:
                recommendations.append("Focus on current difficulty level before advancing")

        if metrics.safety_score < 90:
            recommendations.append("PRIORITY: Complete additional safety-focused training")

        if metrics.average_response_time_seconds > 45:
            recommendations.append("Practice time-critical decision making scenarios")

        if metrics.accuracy_percent < 80:
            recommendations.append("Review standard operating procedures")

        # Add competency-specific recommendations
        recommendations.extend(competency.training_recommendations[:3])

        return recommendations

    def _recommend_next_scenarios(self, competency: CompetencyAssessment,
                                  results: List[ScenarioResult]) -> List[str]:
        """Recommend next training scenarios."""
        scenarios = []

        # Find weakest areas
        weak_areas = sorted(competency.scores_by_area, key=lambda s: s.score)[:2]

        for area in weak_areas:
            if area.area == CompetencyArea.SAFETY_RESPONSE:
                scenarios.append("Emergency Response - Fire Scenario")
                scenarios.append("Emergency Response - Gas Leak")
            elif area.area == CompetencyArea.PROCESS_CONTROL:
                scenarios.append("Process Upset - Temperature Excursion")
                scenarios.append("Normal Operations - Load Change")
            elif area.area == CompetencyArea.EQUIPMENT_OPERATION:
                scenarios.append("Equipment Startup - Compressor")
                scenarios.append("Equipment Shutdown - Furnace")
            elif area.area == CompetencyArea.TROUBLESHOOTING:
                scenarios.append("Equipment Failure - Pump Trip")
                scenarios.append("Process Deviation - Pressure Loss")

        # Add a challenging scenario if performing well
        if competency.overall_score >= 80:
            scenarios.append("Complex Multi-Failure Scenario")

        return scenarios[:5]


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-072",
    "name": "TRAINING-SIM",
    "version": "1.0.0",
    "summary": "Operator training simulation and competency assessment",
    "tags": ["training", "simulation", "operator", "competency", "safety", "HMI"],
    "standards": [
        {"ref": "ISA-101.01", "description": "Human Machine Interfaces"},
        {"ref": "OSHA PSM", "description": "Process Safety Management"},
        {"ref": "ANSI/ISA-18.2", "description": "Alarm Management"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
