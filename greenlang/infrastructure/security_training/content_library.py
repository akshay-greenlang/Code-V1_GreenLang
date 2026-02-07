# -*- coding: utf-8 -*-
"""
Security Training Content Library - SEC-010

Manages the security training course catalog and content. Provides access to
course definitions, modules, and assessment questions for the security training
platform.

The ContentLibrary contains a comprehensive catalog of security training courses
organized by target audience:
    - All Employees: security_awareness, phishing_recognition, password_hygiene,
      data_classification
    - Developers: secure_coding_fundamentals, owasp_top_10, secure_code_review,
      dependency_security
    - DevOps: infrastructure_security, secrets_management, container_security,
      incident_response

Classes:
    - ContentLibrary: Main class for course catalog and content management

Example:
    >>> from greenlang.infrastructure.security_training.content_library import (
    ...     ContentLibrary,
    ... )
    >>> library = ContentLibrary()
    >>> courses = await library.list_courses(role_filter="developer")
    >>> content = await library.get_course("owasp_top_10")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from greenlang.infrastructure.security_training.models import (
    ContentType,
    Course,
    CourseContent,
    Module,
    Question,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Course Catalog
# ---------------------------------------------------------------------------

COURSE_CATALOG: Dict[str, Course] = {
    # All Employees Courses
    "security_awareness": Course(
        id="security_awareness",
        title="Security Awareness Fundamentals",
        description=(
            "Annual security awareness training covering essential security "
            "concepts, social engineering threats, physical security, and "
            "best practices for protecting company information."
        ),
        duration_minutes=45,
        content_type=ContentType.INTERACTIVE,
        role_required=None,  # All employees
        passing_score=80,
        is_mandatory=True,
        tags=["security", "awareness", "annual", "all-employees"],
    ),
    "phishing_recognition": Course(
        id="phishing_recognition",
        title="Phishing Recognition and Prevention",
        description=(
            "Learn to identify and respond to phishing attacks including email, "
            "SMS (smishing), voice (vishing), and social media-based attacks. "
            "Covers red flags, verification procedures, and reporting protocols."
        ),
        duration_minutes=30,
        content_type=ContentType.INTERACTIVE,
        role_required=None,
        passing_score=80,
        is_mandatory=True,
        tags=["security", "phishing", "social-engineering", "all-employees"],
    ),
    "password_hygiene": Course(
        id="password_hygiene",
        title="Password Security and MFA Best Practices",
        description=(
            "Best practices for creating strong passwords, using password managers, "
            "and implementing multi-factor authentication (MFA). Covers common "
            "password attacks and how to prevent them."
        ),
        duration_minutes=20,
        content_type=ContentType.INTERACTIVE,
        role_required=None,
        passing_score=80,
        is_mandatory=True,
        tags=["security", "passwords", "mfa", "authentication", "all-employees"],
    ),
    "data_classification": Course(
        id="data_classification",
        title="Data Classification and Handling",
        description=(
            "Understanding data classification levels (public, internal, "
            "confidential, restricted) and proper handling procedures for each. "
            "Covers data labeling, storage, transmission, and disposal requirements."
        ),
        duration_minutes=30,
        content_type=ContentType.INTERACTIVE,
        role_required=None,
        passing_score=80,
        is_mandatory=True,
        tags=["security", "data", "classification", "compliance", "all-employees"],
    ),
    # Developer Courses
    "secure_coding_fundamentals": Course(
        id="secure_coding_fundamentals",
        title="Secure Coding Fundamentals",
        description=(
            "Foundation course in secure software development practices. Covers "
            "input validation, output encoding, authentication, session management, "
            "error handling, and secure configuration."
        ),
        duration_minutes=90,
        content_type=ContentType.INTERACTIVE,
        role_required="developer",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "development", "secure-coding", "developer"],
    ),
    "owasp_top_10": Course(
        id="owasp_top_10",
        title="OWASP Top 10 Web Application Security Risks",
        description=(
            "Deep dive into the OWASP Top 10 web application security risks "
            "including injection, broken authentication, XSS, insecure design, "
            "security misconfiguration, and more. Includes code examples and "
            "remediation strategies."
        ),
        duration_minutes=120,
        content_type=ContentType.INTERACTIVE,
        role_required="developer",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "owasp", "web-security", "vulnerabilities", "developer"],
        prerequisites=["secure_coding_fundamentals"],
    ),
    "secure_code_review": Course(
        id="secure_code_review",
        title="Secure Code Review Practices",
        description=(
            "Learn effective techniques for identifying security vulnerabilities "
            "during code review. Covers common vulnerability patterns, automated "
            "tools, and review checklists."
        ),
        duration_minutes=60,
        content_type=ContentType.INTERACTIVE,
        role_required="developer",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "code-review", "vulnerabilities", "developer"],
        prerequisites=["secure_coding_fundamentals"],
    ),
    "dependency_security": Course(
        id="dependency_security",
        title="Dependency Security and Supply Chain",
        description=(
            "Managing security risks in third-party dependencies and the software "
            "supply chain. Covers SCA tools, vulnerability tracking, license "
            "compliance, and dependency update strategies."
        ),
        duration_minutes=45,
        content_type=ContentType.INTERACTIVE,
        role_required="developer",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "dependencies", "supply-chain", "sca", "developer"],
    ),
    # DevOps Courses
    "infrastructure_security": Course(
        id="infrastructure_security",
        title="Infrastructure Security Fundamentals",
        description=(
            "Security best practices for cloud and infrastructure management. "
            "Covers network security, IAM, encryption, logging, and compliance "
            "requirements for AWS, GCP, and Azure."
        ),
        duration_minutes=90,
        content_type=ContentType.INTERACTIVE,
        role_required="devops",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "infrastructure", "cloud", "devops"],
    ),
    "secrets_management": Course(
        id="secrets_management",
        title="Secrets Management Best Practices",
        description=(
            "Secure handling of secrets, credentials, and sensitive configuration. "
            "Covers secrets management tools (Vault, AWS Secrets Manager), "
            "rotation policies, and preventing secret exposure."
        ),
        duration_minutes=60,
        content_type=ContentType.INTERACTIVE,
        role_required="devops",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "secrets", "vault", "credentials", "devops"],
    ),
    "container_security": Course(
        id="container_security",
        title="Container and Kubernetes Security",
        description=(
            "Security considerations for containerized workloads and Kubernetes "
            "clusters. Covers image security, runtime protection, network policies, "
            "RBAC, and security scanning."
        ),
        duration_minutes=75,
        content_type=ContentType.INTERACTIVE,
        role_required="devops",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "containers", "kubernetes", "docker", "devops"],
    ),
    "incident_response": Course(
        id="incident_response",
        title="Security Incident Response for DevOps",
        description=(
            "Responding to security incidents in production environments. Covers "
            "incident detection, containment, eradication, recovery, and post-mortem "
            "analysis with focus on DevOps/SRE responsibilities."
        ),
        duration_minutes=60,
        content_type=ContentType.INTERACTIVE,
        role_required="devops",
        passing_score=80,
        is_mandatory=True,
        tags=["security", "incident-response", "sre", "devops"],
    ),
}


# ---------------------------------------------------------------------------
# Question Banks (Sample questions for each course)
# ---------------------------------------------------------------------------

QUESTION_BANKS: Dict[str, List[Question]] = {
    "security_awareness": [
        Question(
            id="sa-001",
            text="What is the primary goal of social engineering attacks?",
            options=[
                "To exploit software vulnerabilities",
                "To manipulate people into revealing confidential information",
                "To overload systems with traffic",
                "To encrypt files for ransom",
            ],
            correct_option=1,
            explanation=(
                "Social engineering attacks target human psychology rather than "
                "technical vulnerabilities, manipulating people into revealing "
                "information or performing actions that compromise security."
            ),
            difficulty=2,
        ),
        Question(
            id="sa-002",
            text="Which of the following is NOT a recommended practice for physical security?",
            options=[
                "Challenging unknown visitors",
                "Holding the door open for people behind you",
                "Wearing your badge visibly",
                "Reporting suspicious activity",
            ],
            correct_option=1,
            explanation=(
                "Tailgating/piggybacking is a physical security risk. Each person "
                "should badge in individually, even if it seems impolite."
            ),
            difficulty=2,
        ),
        Question(
            id="sa-003",
            text="What should you do if you receive a suspicious email?",
            options=[
                "Forward it to colleagues to warn them",
                "Click the link to verify if it's real",
                "Report it using your organization's phishing report process",
                "Reply to ask if it's legitimate",
            ],
            correct_option=2,
            explanation=(
                "Always report suspicious emails through official channels. Do not "
                "click links, download attachments, or reply to the sender."
            ),
            difficulty=1,
        ),
        Question(
            id="sa-004",
            text="What is 'pretexting' in the context of social engineering?",
            options=[
                "Sending fake text messages",
                "Creating a fabricated scenario to gain trust",
                "Texting before an attack",
                "Reading message previews",
            ],
            correct_option=1,
            explanation=(
                "Pretexting is when an attacker creates a fabricated scenario "
                "(pretext) to manipulate the victim, such as impersonating IT support "
                "or a vendor representative."
            ),
            difficulty=3,
        ),
        Question(
            id="sa-005",
            text="Which of the following passwords is the strongest?",
            options=[
                "Password123!",
                "MyD0g$N@me2024",
                "correct-horse-battery-staple",
                "Qwerty!@#$",
            ],
            correct_option=2,
            explanation=(
                "Long passphrases with random words are stronger than shorter "
                "passwords with substitutions. The passphrase has more entropy "
                "and is easier to remember."
            ),
            difficulty=2,
        ),
    ],
    "phishing_recognition": [
        Question(
            id="pr-001",
            text="Which of the following is a common indicator of a phishing email?",
            options=[
                "Email from a known colleague about a scheduled meeting",
                "Urgent request to verify your account with a deadline",
                "Company newsletter with familiar formatting",
                "Reply to a conversation you started",
            ],
            correct_option=1,
            explanation=(
                "Urgency and artificial deadlines are classic phishing tactics "
                "designed to prevent careful thinking and verification."
            ),
            difficulty=1,
        ),
        Question(
            id="pr-002",
            text="What should you check when hovering over a link in an email?",
            options=[
                "The font color of the link",
                "Whether the displayed URL matches the actual destination URL",
                "How long the link is",
                "Whether it opens in a new tab",
            ],
            correct_option=1,
            explanation=(
                "Hovering reveals the actual URL destination. Phishing emails often "
                "display legitimate-looking text that links to malicious sites."
            ),
            difficulty=1,
        ),
        Question(
            id="pr-003",
            text="What is 'spear phishing'?",
            options=[
                "Phishing using phone calls",
                "Mass phishing emails to many recipients",
                "Targeted phishing aimed at specific individuals",
                "Phishing through social media",
            ],
            correct_option=2,
            explanation=(
                "Spear phishing targets specific individuals or organizations with "
                "personalized content, making it more convincing and dangerous."
            ),
            difficulty=2,
        ),
        Question(
            id="pr-004",
            text="Which action is safest when you receive an unexpected email requesting payment?",
            options=[
                "Pay immediately if the email looks official",
                "Reply to the email to confirm the request",
                "Call the requester using a known phone number to verify",
                "Forward the email to your finance team",
            ],
            correct_option=2,
            explanation=(
                "Always verify payment requests out-of-band using contact information "
                "you already have, not information provided in the suspicious email."
            ),
            difficulty=2,
        ),
        Question(
            id="pr-005",
            text="What is 'whaling' in the context of phishing?",
            options=[
                "Phishing attacks targeting executives or high-profile individuals",
                "Using very large phishing campaigns",
                "Phishing through dating apps",
                "Attacks specifically targeting the fishing industry",
            ],
            correct_option=0,
            explanation=(
                "Whaling targets 'big fish' - executives and high-profile individuals "
                "who have access to sensitive information and authority to make "
                "financial decisions."
            ),
            difficulty=2,
        ),
    ],
    "password_hygiene": [
        Question(
            id="ph-001",
            text="What is the main benefit of using a password manager?",
            options=[
                "It remembers passwords so you can use simpler ones",
                "It generates and stores unique, strong passwords for each account",
                "It eliminates the need for multi-factor authentication",
                "It automatically changes all your passwords weekly",
            ],
            correct_option=1,
            explanation=(
                "Password managers enable unique, complex passwords for every "
                "account without requiring memorization, significantly improving security."
            ),
            difficulty=1,
        ),
        Question(
            id="ph-002",
            text="Why is password reuse dangerous?",
            options=[
                "It makes passwords easier to crack",
                "A breach at one site compromises all accounts using that password",
                "It violates most terms of service",
                "Passwords expire faster when reused",
            ],
            correct_option=1,
            explanation=(
                "When attackers breach one site, they try stolen credentials on "
                "other sites (credential stuffing). Unique passwords limit damage "
                "to a single account."
            ),
            difficulty=2,
        ),
        Question(
            id="ph-003",
            text="Which multi-factor authentication method is generally most secure?",
            options=[
                "SMS-based one-time codes",
                "Email-based one-time codes",
                "Hardware security keys (FIDO2/WebAuthn)",
                "Security questions",
            ],
            correct_option=2,
            explanation=(
                "Hardware security keys provide the strongest authentication. "
                "SMS can be intercepted via SIM swapping, and security questions "
                "are easily guessable or discoverable."
            ),
            difficulty=3,
        ),
        Question(
            id="ph-004",
            text="What is 'credential stuffing'?",
            options=[
                "Using fake credentials to gain access",
                "Automated testing of leaked username/password pairs",
                "Creating many accounts with the same email",
                "Sharing credentials with team members",
            ],
            correct_option=1,
            explanation=(
                "Credential stuffing uses automated tools to test leaked credentials "
                "from data breaches against many websites, exploiting password reuse."
            ),
            difficulty=2,
        ),
        Question(
            id="ph-005",
            text="How often should you change your passwords?",
            options=[
                "Every 30 days",
                "Every 90 days",
                "Only when there's evidence of compromise",
                "Never, if using a password manager",
            ],
            correct_option=2,
            explanation=(
                "Modern guidance (NIST) recommends against arbitrary rotation. "
                "Change passwords when compromised, not on a fixed schedule, "
                "as frequent changes often lead to weaker passwords."
            ),
            difficulty=3,
        ),
    ],
    "data_classification": [
        Question(
            id="dc-001",
            text="Which data classification typically requires the highest level of protection?",
            options=[
                "Public",
                "Internal",
                "Confidential",
                "Restricted",
            ],
            correct_option=3,
            explanation=(
                "Restricted data (e.g., PII, financial records, trade secrets) "
                "requires the highest protection level with strict access controls "
                "and encryption requirements."
            ),
            difficulty=1,
        ),
        Question(
            id="dc-002",
            text="What should you do before sharing a document with an external party?",
            options=[
                "Verify the classification level and approval requirements",
                "Remove any formatting that looks internal",
                "Convert it to PDF format",
                "Add a disclaimer to the email",
            ],
            correct_option=0,
            explanation=(
                "Always check the data classification before sharing. Some "
                "classifications require specific approvals or prohibit external sharing."
            ),
            difficulty=2,
        ),
        Question(
            id="dc-003",
            text="Which of the following is typically classified as 'Internal' data?",
            options=[
                "Customer credit card numbers",
                "Employee salaries",
                "Company org charts",
                "Published marketing materials",
            ],
            correct_option=2,
            explanation=(
                "Org charts are typically Internal - not secret but not for public "
                "distribution. Credit cards are Restricted, salaries are Confidential, "
                "and published marketing is Public."
            ),
            difficulty=2,
        ),
        Question(
            id="dc-004",
            text="What is the proper way to dispose of documents containing confidential information?",
            options=[
                "Throw them in the regular trash",
                "Recycle them in the paper recycling bin",
                "Cross-cut shred or use secure destruction bins",
                "Tear them in half before disposal",
            ],
            correct_option=2,
            explanation=(
                "Confidential documents should be cross-cut shredded or placed "
                "in secure destruction bins. Simple tearing can still allow reconstruction."
            ),
            difficulty=1,
        ),
        Question(
            id="dc-005",
            text="What is the principle of 'data minimization'?",
            options=[
                "Compressing data to use less storage",
                "Collecting only the data necessary for a specific purpose",
                "Deleting old data automatically",
                "Using shorter file names",
            ],
            correct_option=1,
            explanation=(
                "Data minimization means collecting, processing, and retaining "
                "only the data that is necessary. This reduces risk and is "
                "required by regulations like GDPR."
            ),
            difficulty=2,
        ),
    ],
    "secure_coding_fundamentals": [
        Question(
            id="scf-001",
            text="What is the most effective defense against SQL injection?",
            options=[
                "Blacklisting dangerous characters",
                "Using parameterized queries/prepared statements",
                "Escaping user input",
                "Limiting database user permissions",
            ],
            correct_option=1,
            explanation=(
                "Parameterized queries separate code from data, making SQL injection "
                "impossible. Escaping and blacklisting are prone to bypasses."
            ),
            difficulty=2,
        ),
        Question(
            id="scf-002",
            text="What is 'defense in depth'?",
            options=[
                "Using the deepest directory structure for security",
                "Multiple layers of security controls",
                "Deep inspection of all network packets",
                "Using the most complex encryption available",
            ],
            correct_option=1,
            explanation=(
                "Defense in depth uses multiple security layers so that if one "
                "control fails, others still provide protection."
            ),
            difficulty=2,
        ),
        Question(
            id="scf-003",
            text="What should error messages shown to users contain?",
            options=[
                "Full stack traces for debugging",
                "Database error messages for clarity",
                "Generic messages without implementation details",
                "The exact cause of the error",
            ],
            correct_option=2,
            explanation=(
                "User-facing error messages should be generic. Detailed errors "
                "should only be logged server-side, as they can reveal information "
                "useful to attackers."
            ),
            difficulty=2,
        ),
        Question(
            id="scf-004",
            text="What is the purpose of input validation?",
            options=[
                "To improve user experience",
                "To ensure data is in expected format and within bounds",
                "To reduce database storage size",
                "To speed up processing",
            ],
            correct_option=1,
            explanation=(
                "Input validation ensures data conforms to expected constraints, "
                "preventing many attacks including injection, overflow, and "
                "format string vulnerabilities."
            ),
            difficulty=1,
        ),
        Question(
            id="scf-005",
            text="What is the principle of 'least privilege'?",
            options=[
                "Using the least expensive security tools",
                "Granting only the minimum permissions needed",
                "Minimizing the number of admins",
                "Using the fewest lines of code",
            ],
            correct_option=1,
            explanation=(
                "Least privilege means users and systems should have only the "
                "minimum permissions necessary to perform their functions, "
                "limiting potential damage from compromise."
            ),
            difficulty=1,
        ),
    ],
    "owasp_top_10": [
        Question(
            id="owasp-001",
            text="Which OWASP Top 10 category does SQL injection fall under?",
            options=[
                "Broken Access Control",
                "Injection",
                "Security Misconfiguration",
                "Insecure Design",
            ],
            correct_option=1,
            explanation=(
                "SQL injection is a classic example of the Injection category, "
                "which also includes OS command injection, LDAP injection, etc."
            ),
            difficulty=1,
        ),
        Question(
            id="owasp-002",
            text="What is 'Broken Access Control'?",
            options=[
                "When authentication systems fail",
                "When users can access resources they shouldn't",
                "When HTTPS is not configured",
                "When passwords are stored in plaintext",
            ],
            correct_option=1,
            explanation=(
                "Broken Access Control occurs when enforcement of what authenticated "
                "users are allowed to do fails, enabling unauthorized access to data "
                "or functionality."
            ),
            difficulty=2,
        ),
        Question(
            id="owasp-003",
            text="Which of the following is an example of Cross-Site Scripting (XSS)?",
            options=[
                "Injecting SQL into a login form",
                "Storing malicious JavaScript that executes in users' browsers",
                "Brute forcing an admin password",
                "Exploiting an unpatched server vulnerability",
            ],
            correct_option=1,
            explanation=(
                "XSS occurs when attacker-controlled data is rendered as code "
                "in a user's browser, enabling script execution in the context "
                "of the victim's session."
            ),
            difficulty=2,
        ),
        Question(
            id="owasp-004",
            text="What is 'Security Misconfiguration'?",
            options=[
                "Using the wrong encryption algorithm",
                "Missing security hardening or insecure default configurations",
                "Having mismatched SSL certificates",
                "Incorrect firewall rules",
            ],
            correct_option=1,
            explanation=(
                "Security Misconfiguration includes missing hardening, default "
                "credentials, unnecessary features, and improper permissions "
                "across the application stack."
            ),
            difficulty=2,
        ),
        Question(
            id="owasp-005",
            text="What is 'Insecure Direct Object Reference' (IDOR)?",
            options=[
                "Using HTTP instead of HTTPS for object access",
                "Allowing access to internal objects by manipulating reference parameters",
                "Storing objects in an insecure location",
                "Using deprecated object serialization",
            ],
            correct_option=1,
            explanation=(
                "IDOR occurs when an application exposes internal implementation "
                "objects (like database IDs) and doesn't verify user authorization, "
                "allowing attackers to access other users' data."
            ),
            difficulty=3,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Content Library Class
# ---------------------------------------------------------------------------


class ContentLibrary:
    """Security training content library.

    Manages the course catalog and content for the security training platform.
    Provides methods to retrieve courses, content, and assessments.

    Attributes:
        _courses: Dictionary of course definitions.
        _content: Dictionary of course content (modules and questions).

    Example:
        >>> library = ContentLibrary()
        >>> courses = await library.list_courses(role_filter="developer")
        >>> len(courses)
        4
        >>> content = await library.get_course("owasp_top_10")
    """

    def __init__(self) -> None:
        """Initialize the content library with default catalog."""
        self._courses: Dict[str, Course] = dict(COURSE_CATALOG)
        self._content: Dict[str, CourseContent] = {}
        self._questions: Dict[str, List[Question]] = dict(QUESTION_BANKS)

        # Initialize content for all courses
        self._initialize_content()

        logger.info(
            "ContentLibrary initialized with %d courses, %d question banks",
            len(self._courses),
            len(self._questions),
        )

    def _initialize_content(self) -> None:
        """Initialize course content from catalog and question banks."""
        for course_id, course in self._courses.items():
            questions = self._questions.get(course_id, [])
            self._content[course_id] = CourseContent(
                course_id=course_id,
                modules=[
                    Module(
                        id=f"{course_id}-mod-1",
                        title=f"Introduction to {course.title}",
                        content_html=f"<h1>{course.title}</h1><p>{course.description}</p>",
                        duration_minutes=max(10, course.duration_minutes // 3),
                        order=0,
                    ),
                    Module(
                        id=f"{course_id}-mod-2",
                        title="Core Concepts",
                        content_html="<h1>Core Concepts</h1><p>Module content...</p>",
                        duration_minutes=max(10, course.duration_minutes // 3),
                        order=1,
                    ),
                    Module(
                        id=f"{course_id}-mod-3",
                        title="Best Practices and Assessment",
                        content_html="<h1>Best Practices</h1><p>Module content...</p>",
                        duration_minutes=max(10, course.duration_minutes // 3),
                        order=2,
                    ),
                ],
                questions=questions,
                version=1,
            )

    async def get_course(self, course_id: str) -> Optional[Course]:
        """Retrieve a course definition by ID.

        Args:
            course_id: The course identifier.

        Returns:
            The Course definition if found, None otherwise.
        """
        return self._courses.get(course_id)

    async def get_course_content(self, course_id: str) -> Optional[CourseContent]:
        """Retrieve full course content including modules and questions.

        Args:
            course_id: The course identifier.

        Returns:
            The CourseContent if found, None otherwise.
        """
        return self._content.get(course_id)

    async def get_assessment(self, course_id: str) -> List[Question]:
        """Retrieve assessment questions for a course.

        Args:
            course_id: The course identifier.

        Returns:
            List of Question objects for the course, empty if not found.
        """
        return self._questions.get(course_id, [])

    async def update_course(
        self,
        course_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Course]:
        """Update a course definition.

        Args:
            course_id: The course identifier.
            updates: Dictionary of fields to update.

        Returns:
            The updated Course if found, None otherwise.
        """
        if course_id not in self._courses:
            return None

        course = self._courses[course_id]
        course_dict = course.model_dump()
        course_dict.update(updates)
        course_dict["updated_at"] = datetime.now(timezone.utc)

        updated_course = Course(**course_dict)
        self._courses[course_id] = updated_course

        logger.info("Updated course %s", course_id)
        return updated_course

    async def update_course_content(
        self,
        course_id: str,
        content: CourseContent,
    ) -> Optional[CourseContent]:
        """Update course content.

        Args:
            course_id: The course identifier.
            content: The new CourseContent.

        Returns:
            The updated CourseContent if found, None otherwise.
        """
        if course_id not in self._courses:
            return None

        content.version = self._content.get(course_id, content).version + 1
        self._content[course_id] = content
        self._questions[course_id] = content.questions

        logger.info("Updated content for course %s (v%d)", course_id, content.version)
        return content

    async def list_courses(
        self,
        role_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        mandatory_only: bool = False,
    ) -> List[Course]:
        """List courses with optional filtering.

        Args:
            role_filter: Filter by required role (None for all-employee courses).
            tag_filter: Filter by tag.
            mandatory_only: Only return mandatory courses.

        Returns:
            List of matching Course objects.
        """
        courses = list(self._courses.values())

        if role_filter is not None:
            # Include courses for the specific role OR all-employee courses
            courses = [
                c for c in courses
                if c.role_required == role_filter or c.role_required is None
            ]

        if tag_filter is not None:
            tag_lower = tag_filter.lower()
            courses = [c for c in courses if tag_lower in c.tags]

        if mandatory_only:
            courses = [c for c in courses if c.is_mandatory]

        return courses

    async def get_courses_for_role(self, role: str) -> List[Course]:
        """Get all courses applicable to a specific role.

        This includes role-specific courses AND all-employee courses.

        Args:
            role: The user's role.

        Returns:
            List of applicable Course objects.
        """
        return await self.list_courses(role_filter=role)

    async def add_questions(
        self,
        course_id: str,
        questions: List[Question],
    ) -> bool:
        """Add questions to a course's question bank.

        Args:
            course_id: The course identifier.
            questions: List of questions to add.

        Returns:
            True if successful, False if course not found.
        """
        if course_id not in self._courses:
            return False

        existing = self._questions.get(course_id, [])
        existing_ids = {q.id for q in existing}

        for q in questions:
            if q.id not in existing_ids:
                existing.append(q)
                existing_ids.add(q.id)

        self._questions[course_id] = existing

        # Update content
        if course_id in self._content:
            self._content[course_id].questions = existing

        logger.info(
            "Added %d questions to course %s (total: %d)",
            len(questions),
            course_id,
            len(existing),
        )
        return True

    async def get_course_count(self) -> int:
        """Get the total number of courses in the catalog.

        Returns:
            Number of courses.
        """
        return len(self._courses)

    async def get_question_count(self, course_id: str) -> int:
        """Get the number of questions for a course.

        Args:
            course_id: The course identifier.

        Returns:
            Number of questions, 0 if course not found.
        """
        return len(self._questions.get(course_id, []))


# Import datetime for type hints
from datetime import datetime
from typing import Any


__all__ = [
    "COURSE_CATALOG",
    "QUESTION_BANKS",
    "ContentLibrary",
]
