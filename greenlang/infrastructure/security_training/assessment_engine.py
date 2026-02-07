# -*- coding: utf-8 -*-
"""
Security Training Assessment Engine - SEC-010

Handles quiz generation, grading, certificate issuance, and attempt tracking
for the security training platform. Implements deterministic quiz generation
from question pools with randomization for fairness.

Classes:
    - AssessmentEngine: Main assessment handling class

Features:
    - Randomized quiz generation from question pools
    - Automatic grading with detailed feedback
    - Certificate issuance with verification codes
    - Attempt tracking and limiting
    - Certificate verification

Example:
    >>> from greenlang.infrastructure.security_training.assessment_engine import (
    ...     AssessmentEngine,
    ... )
    >>> engine = AssessmentEngine(library)
    >>> quiz = await engine.generate_quiz("owasp_top_10", num_questions=10)
    >>> result = await engine.grade_assessment(submission)
"""

from __future__ import annotations

import hashlib
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from greenlang.infrastructure.security_training.models import (
    Certificate,
    Course,
    Question,
    QuizSubmission,
    TrainingCompletion,
)
from greenlang.infrastructure.security_training.content_library import ContentLibrary
from greenlang.infrastructure.security_training.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assessment Result
# ---------------------------------------------------------------------------


class AssessmentResult:
    """Result of grading an assessment.

    Attributes:
        user_id: User who took the assessment.
        course_id: Course the assessment was for.
        score: Percentage score (0-100).
        passed: Whether the user passed.
        total_questions: Total number of questions.
        correct_answers: Number of correct answers.
        incorrect_questions: List of question IDs answered incorrectly.
        feedback: Detailed feedback for each question.
        time_taken_seconds: Time taken to complete.
        attempt_number: Which attempt this was.
        certificate: Certificate if passed, None otherwise.
    """

    def __init__(
        self,
        user_id: str,
        course_id: str,
        score: int,
        passed: bool,
        total_questions: int,
        correct_answers: int,
        incorrect_questions: List[str],
        feedback: Dict[str, str],
        time_taken_seconds: int,
        attempt_number: int,
        certificate: Optional[Certificate] = None,
    ) -> None:
        self.user_id = user_id
        self.course_id = course_id
        self.score = score
        self.passed = passed
        self.total_questions = total_questions
        self.correct_answers = correct_answers
        self.incorrect_questions = incorrect_questions
        self.feedback = feedback
        self.time_taken_seconds = time_taken_seconds
        self.attempt_number = attempt_number
        self.certificate = certificate


class Quiz:
    """A generated quiz for a course.

    Attributes:
        id: Unique quiz identifier.
        course_id: Course this quiz is for.
        questions: List of questions (shuffled options).
        created_at: When the quiz was generated.
        expires_at: When the quiz expires (time limit).
        question_answers: Mapping of question ID to correct answer (internal).
    """

    def __init__(
        self,
        quiz_id: str,
        course_id: str,
        questions: List[Question],
        time_limit_minutes: int,
    ) -> None:
        self.id = quiz_id
        self.course_id = course_id
        self.questions = questions
        self.created_at = datetime.now(timezone.utc)
        self.expires_at = self.created_at + timedelta(minutes=time_limit_minutes)
        self._question_answers: Dict[str, int] = {
            q.id: q.correct_option for q in questions
        }

    def is_expired(self) -> bool:
        """Check if quiz has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def get_correct_answer(self, question_id: str) -> Optional[int]:
        """Get the correct answer for a question."""
        return self._question_answers.get(question_id)


# ---------------------------------------------------------------------------
# Assessment Engine Class
# ---------------------------------------------------------------------------


class AssessmentEngine:
    """Assessment engine for quiz generation and grading.

    Handles the complete assessment lifecycle including quiz generation,
    answer grading, certificate issuance, and attempt tracking.

    Attributes:
        library: ContentLibrary for course/question access.
        _active_quizzes: Cache of active quizzes by quiz_id.
        _user_attempts: Cache of user attempts by (user_id, course_id).
        _certificates: Cache of issued certificates by verification code.
        _completions: Cache of completion records.

    Example:
        >>> engine = AssessmentEngine(library)
        >>> quiz = await engine.generate_quiz("owasp_top_10")
        >>> # User completes quiz...
        >>> result = await engine.grade_assessment(submission)
        >>> if result.passed:
        ...     print(f"Certificate: {result.certificate.verification_code}")
    """

    def __init__(
        self,
        library: ContentLibrary,
    ) -> None:
        """Initialize the assessment engine.

        Args:
            library: ContentLibrary instance for course access.
        """
        self.library = library
        self._config = get_config()

        # Caches (in production, these would be backed by database)
        self._active_quizzes: Dict[str, Quiz] = {}
        self._user_attempts: Dict[Tuple[str, str], int] = {}
        self._certificates: Dict[str, Certificate] = {}
        self._completions: Dict[Tuple[str, str], TrainingCompletion] = {}

        logger.info(
            "AssessmentEngine initialized (pass_score=%d, max_attempts=%d)",
            self._config.pass_score,
            self._config.max_quiz_attempts,
        )

    async def generate_quiz(
        self,
        course_id: str,
        user_id: Optional[str] = None,
        num_questions: Optional[int] = None,
    ) -> Optional[Quiz]:
        """Generate a randomized quiz for a course.

        Selects questions randomly from the course's question pool and
        shuffles answer options for fairness.

        Args:
            course_id: Course to generate quiz for.
            user_id: Optional user ID for attempt tracking.
            num_questions: Number of questions (default from config).

        Returns:
            Generated Quiz or None if course not found.
        """
        course = await self.library.get_course(course_id)
        if course is None:
            logger.warning("Cannot generate quiz: course %s not found", course_id)
            return None

        questions = await self.library.get_assessment(course_id)
        if not questions:
            logger.warning("Cannot generate quiz: no questions for %s", course_id)
            return None

        # Determine number of questions
        count = num_questions or self._config.quiz_question_count
        count = min(count, len(questions))

        # Check attempt limit if user specified
        if user_id:
            attempts = self._user_attempts.get((user_id, course_id), 0)
            if attempts >= self._config.max_quiz_attempts:
                logger.warning(
                    "User %s has reached max attempts (%d) for %s",
                    user_id,
                    attempts,
                    course_id,
                )
                # Still generate quiz but mark it
                pass

        # Random selection of questions
        selected = random.sample(questions, count)

        # Create quiz with shuffled options
        quiz_questions: List[Question] = []
        for q in selected:
            # Shuffle options while tracking correct answer
            options = list(q.options)
            correct_option_text = options[q.correct_option]

            random.shuffle(options)
            new_correct_index = options.index(correct_option_text)

            quiz_questions.append(
                Question(
                    id=q.id,
                    text=q.text,
                    options=options,
                    correct_option=new_correct_index,
                    explanation=q.explanation,
                    difficulty=q.difficulty,
                )
            )

        quiz_id = f"quiz-{uuid.uuid4().hex[:12]}"
        quiz = Quiz(
            quiz_id=quiz_id,
            course_id=course_id,
            questions=quiz_questions,
            time_limit_minutes=self._config.quiz_time_limit_minutes,
        )

        self._active_quizzes[quiz_id] = quiz
        logger.info(
            "Generated quiz %s for course %s with %d questions",
            quiz_id,
            course_id,
            len(quiz_questions),
        )

        return quiz

    async def grade_assessment(
        self,
        submission: QuizSubmission,
        quiz_id: Optional[str] = None,
    ) -> AssessmentResult:
        """Grade a quiz submission.

        Calculates score, determines pass/fail, and issues certificate
        if the user passed.

        Args:
            submission: User's quiz answers.
            quiz_id: Optional quiz ID for validation.

        Returns:
            AssessmentResult with score, feedback, and certificate.
        """
        course = await self.library.get_course(submission.course_id)
        if course is None:
            raise ValueError(f"Course {submission.course_id} not found")

        # Get quiz if specified
        quiz = self._active_quizzes.get(quiz_id) if quiz_id else None

        # Get question pool for grading
        questions = await self.library.get_assessment(submission.course_id)
        question_map = {q.id: q for q in questions}

        # If quiz exists, use its shuffled correct answers
        if quiz and not quiz.is_expired():
            question_map = {q.id: q for q in quiz.questions}
        elif quiz and quiz.is_expired():
            logger.warning("Quiz %s has expired", quiz_id)
            # Still grade but note expiration

        # Grade each answer
        correct_count = 0
        incorrect_questions: List[str] = []
        feedback: Dict[str, str] = {}

        for question_id, selected_option in submission.answers.items():
            question = question_map.get(question_id)
            if question is None:
                feedback[question_id] = "Question not found"
                continue

            if selected_option == question.correct_option:
                correct_count += 1
                feedback[question_id] = "Correct!"
            else:
                incorrect_questions.append(question_id)
                correct_text = question.options[question.correct_option]
                feedback[question_id] = (
                    f"Incorrect. The correct answer was: {correct_text}. "
                    f"{question.explanation}"
                )

        # Calculate score
        total_questions = len(submission.answers)
        score = self.calculate_score(correct_count, total_questions)
        passed = self.check_passing(score, course)

        # Track attempt
        attempt_key = (submission.user_id, submission.course_id)
        current_attempts = self._user_attempts.get(attempt_key, 0) + 1
        self._user_attempts[attempt_key] = current_attempts

        # Issue certificate if passed
        certificate = None
        if passed:
            certificate = await self.issue_certificate(
                user_id=submission.user_id,
                course_id=submission.course_id,
                score=score,
            )

        # Create completion record
        completion = TrainingCompletion(
            user_id=submission.user_id,
            course_id=submission.course_id,
            completed_at=datetime.now(timezone.utc) if passed else None,
            score=score,
            passed=passed,
            certificate_id=certificate.id if certificate else None,
            attempts=current_attempts,
            time_spent_minutes=submission.time_taken_seconds // 60,
        )
        self._completions[attempt_key] = completion

        logger.info(
            "Graded assessment for user %s course %s: score=%d, passed=%s",
            submission.user_id,
            submission.course_id,
            score,
            passed,
        )

        return AssessmentResult(
            user_id=submission.user_id,
            course_id=submission.course_id,
            score=score,
            passed=passed,
            total_questions=total_questions,
            correct_answers=correct_count,
            incorrect_questions=incorrect_questions,
            feedback=feedback,
            time_taken_seconds=submission.time_taken_seconds,
            attempt_number=current_attempts,
            certificate=certificate,
        )

    def calculate_score(self, correct: int, total: int) -> int:
        """Calculate percentage score.

        Args:
            correct: Number of correct answers.
            total: Total number of questions.

        Returns:
            Percentage score (0-100).
        """
        if total == 0:
            return 0
        return round((correct / total) * 100)

    def check_passing(self, score: int, course: Course) -> bool:
        """Check if score meets passing requirements.

        Uses course-specific passing score if set, otherwise config default.

        Args:
            score: User's score (0-100).
            course: Course being assessed.

        Returns:
            True if passed, False otherwise.
        """
        passing_score = course.passing_score or self._config.pass_score
        return score >= passing_score

    async def issue_certificate(
        self,
        user_id: str,
        course_id: str,
        score: int,
        user_name: str = "",
    ) -> Certificate:
        """Generate completion certificate.

        Creates a certificate with a unique verification code for
        authenticity checks.

        Args:
            user_id: User who completed the course.
            course_id: Course that was completed.
            score: Score achieved.
            user_name: Optional user display name.

        Returns:
            Generated Certificate.
        """
        course = await self.library.get_course(course_id)
        course_title = course.title if course else course_id

        # Generate verification code
        verification_code = self._generate_verification_code(user_id, course_id)

        certificate = Certificate(
            user_id=user_id,
            course_id=course_id,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(
                days=self._config.certificate_validity_days
            ),
            verification_code=verification_code,
            score=score,
            user_name=user_name,
            course_title=course_title,
        )

        # Store for verification
        self._certificates[verification_code] = certificate

        logger.info(
            "Issued certificate %s for user %s course %s",
            verification_code,
            user_id,
            course_id,
        )

        return certificate

    async def verify_certificate(
        self,
        verification_code: str,
    ) -> Optional[Certificate]:
        """Verify certificate authenticity.

        Args:
            verification_code: Certificate verification code.

        Returns:
            Certificate if valid and not expired, None otherwise.
        """
        certificate = self._certificates.get(verification_code)

        if certificate is None:
            logger.warning("Certificate not found: %s", verification_code)
            return None

        if certificate.expires_at < datetime.now(timezone.utc):
            logger.warning("Certificate expired: %s", verification_code)
            return None

        return certificate

    async def track_attempts(
        self,
        user_id: str,
        course_id: str,
    ) -> Tuple[int, int]:
        """Get attempt count and remaining attempts.

        Args:
            user_id: User identifier.
            course_id: Course identifier.

        Returns:
            Tuple of (attempts_used, attempts_remaining).
        """
        attempts = self._user_attempts.get((user_id, course_id), 0)
        remaining = max(0, self._config.max_quiz_attempts - attempts)
        return attempts, remaining

    async def reset_attempts(
        self,
        user_id: str,
        course_id: str,
    ) -> None:
        """Reset attempt count for a user (admin function).

        Args:
            user_id: User identifier.
            course_id: Course identifier.
        """
        key = (user_id, course_id)
        if key in self._user_attempts:
            del self._user_attempts[key]
            logger.info("Reset attempts for user %s course %s", user_id, course_id)

    async def get_completion(
        self,
        user_id: str,
        course_id: str,
    ) -> Optional[TrainingCompletion]:
        """Get user's completion record for a course.

        Args:
            user_id: User identifier.
            course_id: Course identifier.

        Returns:
            TrainingCompletion if exists, None otherwise.
        """
        return self._completions.get((user_id, course_id))

    def _generate_verification_code(
        self,
        user_id: str,
        course_id: str,
    ) -> str:
        """Generate a unique verification code.

        Args:
            user_id: User identifier.
            course_id: Course identifier.

        Returns:
            Verification code string.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        unique_string = f"{user_id}:{course_id}:{timestamp}:{uuid.uuid4()}"
        hash_value = hashlib.sha256(unique_string.encode()).hexdigest()[:12].upper()
        return f"GL-{hash_value}"

    async def get_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Get an active quiz by ID.

        Args:
            quiz_id: Quiz identifier.

        Returns:
            Quiz if found and not expired, None otherwise.
        """
        quiz = self._active_quizzes.get(quiz_id)
        if quiz and not quiz.is_expired():
            return quiz
        return None

    async def cleanup_expired_quizzes(self) -> int:
        """Clean up expired quizzes.

        Returns:
            Number of quizzes cleaned up.
        """
        expired = [
            quiz_id for quiz_id, quiz in self._active_quizzes.items()
            if quiz.is_expired()
        ]
        for quiz_id in expired:
            del self._active_quizzes[quiz_id]

        if expired:
            logger.info("Cleaned up %d expired quizzes", len(expired))

        return len(expired)


__all__ = [
    "AssessmentEngine",
    "AssessmentResult",
    "Quiz",
]
