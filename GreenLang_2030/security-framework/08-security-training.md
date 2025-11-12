# Security Training Program

## 1. Developer Security Training

### Secure Coding Curriculum

```yaml
# developer-security-curriculum.yaml
developer_security_training:
  onboarding_program:
    duration: "2 weeks"
    format: "Hybrid (online + hands-on)"

    week_1_fundamentals:
      day_1_security_basics:
        topics:
          - "Security mindset and threat modeling"
          - "OWASP Top 10 overview"
          - "Common vulnerability types"
          - "Security vs usability trade-offs"
        labs:
          - "Identify vulnerabilities in sample code"
          - "Basic threat modeling exercise"

      day_2_authentication_authorization:
        topics:
          - "Authentication mechanisms"
          - "Session management"
          - "OAuth 2.0 and JWT"
          - "Role-based access control"
        labs:
          - "Implement secure authentication"
          - "JWT token validation"
          - "Session hijacking prevention"

      day_3_input_validation:
        topics:
          - "Input validation strategies"
          - "SQL injection prevention"
          - "XSS prevention"
          - "Command injection"
        labs:
          - "Build parameterized queries"
          - "Implement input sanitization"
          - "XSS filter bypass challenges"

      day_4_cryptography:
        topics:
          - "Encryption fundamentals"
          - "Hashing vs encryption"
          - "Key management"
          - "TLS/SSL implementation"
        labs:
          - "Implement password hashing"
          - "Encrypt sensitive data"
          - "Certificate pinning"

      day_5_secure_apis:
        topics:
          - "API security best practices"
          - "Rate limiting"
          - "API authentication"
          - "GraphQL security"
        labs:
          - "Secure REST API design"
          - "Implement rate limiting"
          - "API penetration testing"

    week_2_advanced:
      day_6_cloud_security:
        topics:
          - "Cloud security principles"
          - "IAM best practices"
          - "Secrets management"
          - "Container security"
        labs:
          - "Configure secure S3 buckets"
          - "Implement Vault integration"
          - "Container scanning"

      day_7_devsecops:
        topics:
          - "Security in CI/CD"
          - "SAST/DAST integration"
          - "Dependency scanning"
          - "Security as code"
        labs:
          - "Setup security pipeline"
          - "Integrate Snyk/SonarQube"
          - "Automate security tests"

      day_8_incident_response:
        topics:
          - "Developer role in incidents"
          - "Evidence preservation"
          - "Security logging"
          - "Debugging vs security"
        labs:
          - "Implement security logging"
          - "Incident simulation"
          - "Log analysis"

      day_9_compliance:
        topics:
          - "GDPR for developers"
          - "Data privacy by design"
          - "Audit logging"
          - "Data retention"
        labs:
          - "Implement data anonymization"
          - "Build audit trails"
          - "Right to deletion"

      day_10_assessment:
        activities:
          - "Secure coding assessment"
          - "Vulnerability hunting exercise"
          - "Code review challenge"
          - "Certification exam"

  continuous_education:
    monthly_workshops:
      topics:
        - "Latest vulnerability trends"
        - "New attack techniques"
        - "Tool demonstrations"
        - "Case study analysis"

    quarterly_training:
      q1: "Mobile application security"
      q2: "Cloud-native security"
      q3: "Zero trust architecture"
      q4: "AI/ML security"

    annual_requirements:
      mandatory:
        - "OWASP Top 10 refresh"
        - "Secure coding assessment"
        - "Phishing awareness"
      optional:
        - "Security certifications"
        - "Conference attendance"
        - "Bug bounty participation"

  language_specific_training:
    javascript_nodejs:
      topics:
        - "Prototype pollution"
        - "NPM security"
        - "Electron security"
        - "React security patterns"
      tools:
        - "ESLint security plugin"
        - "npm audit"
        - "Snyk"

    python:
      topics:
        - "Django security"
        - "Flask security"
        - "Pickle exploitation"
        - "Type confusion"
      tools:
        - "Bandit"
        - "Safety"
        - "PyLint"

    go:
      topics:
        - "Memory safety"
        - "Goroutine security"
        - "Dependency management"
        - "CGO security"
      tools:
        - "gosec"
        - "go-audit"
        - "nancy"

    java_spring:
      topics:
        - "Spring Security"
        - "Deserialization attacks"
        - "XXE prevention"
        - "JDBC security"
      tools:
        - "SpotBugs"
        - "OWASP Dependency Check"
        - "PMD"
```

### Hands-on Security Labs

```python
# security_training_labs.py
from typing import Dict, List, Optional
import random
import hashlib

class SecurityTrainingLab:
    def __init__(self, lab_type: str, difficulty: str):
        self.lab_type = lab_type
        self.difficulty = difficulty
        self.vulnerabilities = []
        self.solutions = []

class SQLInjectionLab(SecurityTrainingLab):
    def __init__(self, difficulty: str = "beginner"):
        super().__init__("SQL Injection", difficulty)
        self.setup_lab()

    def setup_lab(self):
        """Setup SQL injection training lab"""
        self.vulnerable_code = """
        # VULNERABLE CODE - DO NOT USE IN PRODUCTION
        def get_user(username, password):
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            cursor.execute(query)
            return cursor.fetchone()
        """

        self.attack_vectors = [
            "admin' --",
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "admin'/*",
            "' UNION SELECT * FROM passwords --"
        ]

        self.secure_code = """
        # SECURE CODE
        def get_user(username, password):
            query = "SELECT * FROM users WHERE username=? AND password=?"
            cursor.execute(query, (username, password))
            return cursor.fetchone()
        """

        self.exercises = [
            {
                "title": "Basic Authentication Bypass",
                "description": "Bypass login without knowing credentials",
                "hint": "Try commenting out the password check",
                "solution": "username: admin' -- password: anything"
            },
            {
                "title": "Data Extraction",
                "description": "Extract all usernames from the database",
                "hint": "Use UNION to combine queries",
                "solution": "' UNION SELECT username, null FROM users --"
            },
            {
                "title": "Blind SQL Injection",
                "description": "Determine if user 'admin' exists",
                "hint": "Use boolean conditions",
                "solution": "' AND EXISTS(SELECT * FROM users WHERE username='admin') --"
            }
        ]

    def validate_solution(self, student_input: str, exercise_id: int) -> Dict:
        """Validate student's solution"""
        exercise = self.exercises[exercise_id]

        # Check if the student's input would successfully exploit
        is_exploit = self.is_sql_injection(student_input)

        # Check if it matches the expected solution pattern
        is_correct = self.matches_solution(student_input, exercise["solution"])

        return {
            "exploits_vulnerability": is_exploit,
            "correct_solution": is_correct,
            "feedback": self.generate_feedback(student_input, is_exploit, is_correct),
            "points_earned": self.calculate_points(is_exploit, is_correct)
        }

    def is_sql_injection(self, input_str: str) -> bool:
        """Check if input contains SQL injection patterns"""
        sql_patterns = [
            "'", '"', "--", "/*", "*/", "UNION", "SELECT",
            "DROP", "INSERT", "UPDATE", "DELETE", "OR", "AND"
        ]

        return any(pattern.lower() in input_str.lower() for pattern in sql_patterns)

    def generate_feedback(self, input_str: str, is_exploit: bool, is_correct: bool) -> str:
        """Generate educational feedback"""
        if is_correct:
            return "Excellent! You successfully exploited the SQL injection vulnerability."
        elif is_exploit:
            return "Good attempt! You found a SQL injection, but try a different approach for this specific challenge."
        else:
            return "Not quite. Remember to look for ways to manipulate the SQL query structure."

class XSSLab(SecurityTrainingLab):
    def __init__(self, difficulty: str = "intermediate"):
        super().__init__("Cross-Site Scripting", difficulty)
        self.setup_lab()

    def setup_lab(self):
        """Setup XSS training lab"""
        self.vulnerable_scenarios = [
            {
                "name": "Reflected XSS",
                "code": """
                # VULNERABLE
                @app.route('/search')
                def search():
                    query = request.args.get('q')
                    return f'<h1>Search results for: {query}</h1>'
                """,
                "payload": "<script>alert('XSS')</script>",
                "fix": "Use HTML escaping: html.escape(query)"
            },
            {
                "name": "Stored XSS",
                "code": """
                # VULNERABLE
                @app.route('/comment', methods=['POST'])
                def post_comment():
                    comment = request.form['comment']
                    db.save_comment(comment)
                    return render_template('comments.html', comment=comment)
                """,
                "payload": "<img src=x onerror=alert('XSS')>",
                "fix": "Sanitize input before storage and output encoding"
            },
            {
                "name": "DOM XSS",
                "code": """
                // VULNERABLE
                const urlParams = new URLSearchParams(window.location.search);
                const name = urlParams.get('name');
                document.getElementById('welcome').innerHTML = 'Welcome ' + name;
                """,
                "payload": "<img src=x onerror=alert('XSS')>",
                "fix": "Use textContent instead of innerHTML"
            }
        ]

        self.filter_bypass_challenges = [
            {
                "filter": "Removes <script> tags",
                "bypass": "<img src=x onerror=alert(1)>",
                "explanation": "Use event handlers instead of script tags"
            },
            {
                "filter": "Blocks 'alert'",
                "bypass": "<script>prompt('XSS')</script>",
                "explanation": "Use alternative JavaScript functions"
            },
            {
                "filter": "Removes < and >",
                "bypass": "&#60;script&#62;alert('XSS')&#60;/script&#62;",
                "explanation": "Use HTML entities"
            }
        ]

    def create_sandbox_environment(self) -> str:
        """Create safe testing environment"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>XSS Training Lab</title>
            <meta http-equiv="Content-Security-Policy"
                  content="default-src 'self'; script-src 'unsafe-inline'">
        </head>
        <body>
            <div id="sandbox">
                <!-- Student input will be rendered here -->
            </div>
            <script>
                // Safe sandbox for XSS testing
                function testXSS(input) {
                    const sandbox = document.getElementById('sandbox');
                    // This is intentionally vulnerable for training
                    sandbox.innerHTML = input;
                }
            </script>
        </body>
        </html>
        """

class AuthenticationLab(SecurityTrainingLab):
    def __init__(self, difficulty: str = "advanced"):
        super().__init__("Authentication Security", difficulty)
        self.setup_lab()

    def setup_lab(self):
        """Setup authentication security lab"""
        self.weak_implementations = [
            {
                "vulnerability": "Weak Password Storage",
                "vulnerable_code": """
                # VULNERABLE - MD5 hashing
                password_hash = hashlib.md5(password.encode()).hexdigest()
                """,
                "secure_code": """
                # SECURE - bcrypt with salt
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
                """,
                "attack": "Rainbow table attack"
            },
            {
                "vulnerability": "Session Fixation",
                "vulnerable_code": """
                # VULNERABLE - Session ID doesn't change after login
                def login(username, password):
                    if authenticate(username, password):
                        session['user'] = username
                        return redirect('/dashboard')
                """,
                "secure_code": """
                # SECURE - Regenerate session ID
                def login(username, password):
                    if authenticate(username, password):
                        session.regenerate_id()
                        session['user'] = username
                        return redirect('/dashboard')
                """,
                "attack": "Attacker sets victim's session ID"
            },
            {
                "vulnerability": "JWT None Algorithm",
                "vulnerable_code": """
                # VULNERABLE - Accepts 'none' algorithm
                def verify_jwt(token):
                    return jwt.decode(token, options={"verify_signature": False})
                """,
                "secure_code": """
                # SECURE - Verify algorithm and signature
                def verify_jwt(token):
                    return jwt.decode(
                        token,
                        secret_key,
                        algorithms=['HS256'],
                        options={"verify_signature": True}
                    )
                """,
                "attack": "JWT signature bypass"
            }
        ]

    def generate_challenge_jwt(self) -> str:
        """Generate JWT for challenge"""
        import jwt
        import time

        payload = {
            "user_id": "123",
            "username": "student",
            "role": "user",
            "exp": int(time.time()) + 3600
        }

        # Weak secret for training purposes
        weak_secret = "secret123"

        return jwt.encode(payload, weak_secret, algorithm="HS256")

    def validate_jwt_attack(self, modified_token: str) -> Dict:
        """Validate JWT manipulation attempt"""
        try:
            # Try to decode without verification (simulating vulnerable app)
            decoded = jwt.decode(modified_token, options={"verify_signature": False})

            if decoded.get("role") == "admin":
                return {
                    "success": True,
                    "message": "Successfully escalated privileges to admin!",
                    "points": 100
                }
            else:
                return {
                    "success": False,
                    "message": "Token modified but privilege escalation failed",
                    "points": 50
                }
        except:
            return {
                "success": False,
                "message": "Invalid token format",
                "points": 0
            }

class TrainingProgressTracker:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.completed_labs = []
        self.scores = {}
        self.badges = []

    def record_lab_completion(self, lab_name: str, score: int):
        """Record lab completion and score"""
        self.completed_labs.append({
            "lab": lab_name,
            "completed_at": datetime.now(),
            "score": score
        })

        self.scores[lab_name] = score
        self.check_badges()

    def check_badges(self):
        """Award badges based on achievements"""
        badges = {
            "SQL Injection Hunter": lambda: self.scores.get("SQL Injection", 0) >= 90,
            "XSS Ninja": lambda: self.scores.get("XSS", 0) >= 90,
            "Authentication Expert": lambda: self.scores.get("Authentication", 0) >= 90,
            "Security Champion": lambda: len(self.completed_labs) >= 10,
            "Perfect Score": lambda: all(score >= 100 for score in self.scores.values())
        }

        for badge_name, condition in badges.items():
            if condition() and badge_name not in self.badges:
                self.badges.append(badge_name)

    def generate_certificate(self) -> Dict:
        """Generate training certificate"""
        if len(self.completed_labs) >= 10 and sum(self.scores.values()) / len(self.scores) >= 80:
            return {
                "certificate_id": hashlib.sha256(f"{self.user_id}{datetime.now()}".encode()).hexdigest()[:12],
                "user": self.user_id,
                "completion_date": datetime.now(),
                "score": sum(self.scores.values()) / len(self.scores),
                "labs_completed": len(self.completed_labs),
                "badges": self.badges,
                "valid": True
            }
        else:
            return {
                "valid": False,
                "message": "Complete all required labs with 80% average score"
            }
```

## 2. Phishing Simulations

### Phishing Campaign Management

```yaml
# phishing-simulation.yaml
phishing_simulation_program:
  campaign_schedule:
    frequency: "Monthly"
    targeting:
      method: "Random selection with role weighting"
      high_risk_roles:
        - executives: 2x
        - finance: 2x
        - hr: 1.5x
        - it_admins: 2x
      sample_size: "33% of employees per month"

  difficulty_levels:
    level_1_obvious:
      characteristics:
        - "Obvious spelling errors"
        - "Generic greetings"
        - "Suspicious sender domain"
        - "Urgent language"
      example_subject: "Your Account Will Be Suspended!!!"
      goal: "Basic awareness"
      expected_click_rate: "<5%"

    level_2_generic:
      characteristics:
        - "Professional appearance"
        - "Common service providers"
        - "Legitimate-looking links"
        - "Standard urgency"
      example_subject: "Password Expiration Notice"
      goal: "Service impersonation awareness"
      expected_click_rate: "<10%"

    level_3_targeted:
      characteristics:
        - "Company branding"
        - "Internal system names"
        - "Colleague impersonation"
        - "Contextual relevance"
      example_subject: "Q4 Bonus Information - Action Required"
      goal: "Spear phishing awareness"
      expected_click_rate: "<15%"

    level_4_advanced:
      characteristics:
        - "Perfect spoofing"
        - "Current events tie-in"
        - "Multi-stage attack"
        - "Legitimate domain compromise"
      example_subject: "Re: Yesterday's Meeting Follow-up"
      goal: "Advanced threat awareness"
      expected_click_rate: "<20%"

  templates:
    credential_harvesting:
      - name: "Office 365 Login"
        lure: "Your mailbox is full"
        landing_page: "Fake O365 login"
        indicators:
          - "Incorrect domain"
          - "HTTP instead of HTTPS"
          - "Poor grammar"

      - name: "VPN Update"
        lure: "New VPN client required"
        landing_page: "Fake VPN portal"
        indicators:
          - "External sender"
          - "Unexpected request"
          - "Suspicious URL"

    malware_delivery:
      - name: "Invoice Attachment"
        lure: "Overdue invoice attached"
        attachment: "Invoice_2024.pdf.exe"
        indicators:
          - "Double extension"
          - "Unknown sender"
          - "Generic content"

      - name: "Resume Download"
        lure: "Candidate resume for review"
        attachment: "John_Doe_Resume.docm"
        indicators:
          - "Macro-enabled document"
          - "External sender"
          - "Unsolicited"

    business_email_compromise:
      - name: "CEO Fraud"
        lure: "Urgent wire transfer needed"
        sender_spoof: "CEO name"
        indicators:
          - "Unusual request"
          - "Bypassing process"
          - "External reply-to"

      - name: "Vendor Payment Update"
        lure: "Banking details changed"
        sender_spoof: "Known vendor"
        indicators:
          - "Payment change request"
          - "Different domain"
          - "Urgency"

  response_tracking:
    metrics:
      - opened_email
      - clicked_link
      - submitted_credentials
      - downloaded_attachment
      - reported_suspicious
      - forwarded_email
      - replied_to_sender

    user_actions:
      positive:
        - reported_to_security
        - deleted_without_opening
        - verified_with_sender

      negative:
        - clicked_link
        - entered_credentials
        - downloaded_file
        - forwarded_to_others

  training_response:
    immediate_feedback:
      clicked_link:
        message: "This was a simulated phishing attack"
        training: "5-minute video on identifying phishing"
        points_lost: 10

      reported_correctly:
        message: "Great job identifying this phishing attempt!"
        training: "Advanced tips video"
        points_earned: 20

    follow_up_training:
      failed_simulation:
        requirement: "Complete phishing awareness course"
        deadline: "Within 7 days"
        duration: "30 minutes"

      repeat_offender:
        requirement: "1-on-1 security coaching"
        deadline: "Within 3 days"
        duration: "1 hour"

  reporting:
    individual_report:
      - simulation_history
      - click_rate_trend
      - training_completed
      - risk_score

    department_report:
      - department_click_rate
      - comparison_to_company
      - high_risk_individuals
      - improvement_trends

    executive_dashboard:
      - company_click_rate
      - department_breakdown
      - risk_trends
      - training_effectiveness
```

### Phishing Simulation Platform

```python
# phishing_simulation_platform.py
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class PhishingSimulator:
    def __init__(self, employee_database, email_service):
        self.employees = employee_database
        self.email_service = email_service
        self.campaigns = []
        self.results = []

    def create_campaign(self, campaign_config: Dict) -> str:
        """Create new phishing campaign"""
        campaign = {
            "id": self.generate_campaign_id(),
            "name": campaign_config["name"],
            "created_at": datetime.now(),
            "scheduled_for": campaign_config["scheduled_for"],
            "difficulty": campaign_config["difficulty"],
            "template": campaign_config["template"],
            "target_group": self.select_targets(campaign_config["targeting"]),
            "status": "scheduled",
            "tracking_enabled": True
        }

        self.campaigns.append(campaign)
        return campaign["id"]

    def select_targets(self, targeting_config: Dict) -> List[str]:
        """Select employees for phishing campaign"""
        all_employees = self.employees.get_all()

        # Apply role-based weighting
        weighted_employees = []
        for employee in all_employees:
            weight = targeting_config["role_weights"].get(employee["role"], 1.0)

            # Add employee multiple times based on weight
            for _ in range(int(weight)):
                weighted_employees.append(employee["email"])

        # Random selection
        sample_size = int(len(all_employees) * targeting_config["percentage"] / 100)
        targets = random.sample(weighted_employees, min(sample_size, len(weighted_employees)))

        # Exclude recent targets if configured
        if targeting_config.get("exclude_recent_days"):
            recent_cutoff = datetime.now() - timedelta(days=targeting_config["exclude_recent_days"])
            recent_targets = self.get_recent_targets(recent_cutoff)
            targets = [t for t in targets if t not in recent_targets]

        return targets

    def generate_phishing_email(self, template: str, target: Dict) -> Dict:
        """Generate personalized phishing email"""
        templates = {
            "office365": self.generate_office365_phish,
            "package_delivery": self.generate_package_phish,
            "hr_update": self.generate_hr_phish,
            "it_security": self.generate_it_phish,
            "executive": self.generate_executive_phish
        }

        generator = templates.get(template, self.generate_generic_phish)
        return generator(target)

    def generate_office365_phish(self, target: Dict) -> Dict:
        """Generate Office 365 phishing email"""
        tracking_id = self.generate_tracking_id()

        return {
            "subject": "Action Required: Verify your Office 365 Account",
            "sender": "no-reply@microsft-support.com",  # Typo intentional
            "sender_name": "Microsoft Support",
            "body_html": f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <img src="https://tracking.phishtest.io/pixel/{tracking_id}" width="1" height="1">
                <h2>Account Verification Required</h2>
                <p>Dear {target['first_name']},</p>
                <p>We've detected unusual activity on your Office 365 account.
                   To ensure your account security, please verify your identity.</p>
                <p><a href="https://tracking.phishtest.io/click/{tracking_id}"
                      style="background: #0078d4; color: white; padding: 10px 20px;
                             text-decoration: none; border-radius: 4px;">
                   Verify Account
                </a></p>
                <p>This link will expire in 24 hours.</p>
                <p>Thanks,<br>The Microsoft Account Team</p>
                <small style="color: #666;">
                    This is a simulated phishing email for security training.
                    <!-- Hidden tracking: {tracking_id} -->
                </small>
            </body>
            </html>
            """,
            "indicators": [
                "Typo in sender domain (microsft)",
                "Generic greeting",
                "Sense of urgency",
                "Suspicious link destination"
            ],
            "tracking_id": tracking_id
        }

    def execute_campaign(self, campaign_id: str):
        """Execute phishing campaign"""
        campaign = self.get_campaign(campaign_id)

        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        campaign["status"] = "in_progress"
        campaign["started_at"] = datetime.now()

        results = {
            "campaign_id": campaign_id,
            "sent": 0,
            "opened": 0,
            "clicked": 0,
            "reported": 0,
            "failed": 0,
            "details": []
        }

        for target_email in campaign["target_group"]:
            target = self.employees.get_by_email(target_email)

            # Generate personalized email
            email = self.generate_phishing_email(campaign["template"], target)

            # Send email
            try:
                self.send_phishing_email(target_email, email)
                results["sent"] += 1

                # Track sending
                self.track_event(campaign_id, target_email, "sent", email["tracking_id"])

            except Exception as e:
                results["failed"] += 1
                self.log_error(f"Failed to send to {target_email}: {str(e)}")

        campaign["status"] = "completed"
        campaign["completed_at"] = datetime.now()

        self.results.append(results)
        return results

    def track_user_action(self, tracking_id: str, action: str) -> Dict:
        """Track user action on phishing email"""
        tracking_record = self.get_tracking_record(tracking_id)

        if not tracking_record:
            return {"error": "Invalid tracking ID"}

        user_email = tracking_record["user_email"]
        campaign_id = tracking_record["campaign_id"]

        # Record action
        self.track_event(campaign_id, user_email, action, tracking_id)

        # Immediate feedback based on action
        if action == "clicked":
            return self.provide_clicked_feedback(user_email, campaign_id)
        elif action == "reported":
            return self.provide_reported_feedback(user_email, campaign_id)
        elif action == "submitted_credentials":
            return self.provide_compromised_feedback(user_email, campaign_id)

        return {"status": "tracked"}

    def provide_clicked_feedback(self, user_email: str, campaign_id: str) -> Dict:
        """Provide feedback when user clicks phishing link"""
        # Redirect to training page
        training_url = f"https://training.greenlang.io/phishing-awareness?user={user_email}&campaign={campaign_id}"

        # Record training requirement
        self.assign_training(user_email, "phishing_awareness_basic", "mandatory")

        # Update user risk score
        self.update_risk_score(user_email, "increase", 10)

        return {
            "redirect": training_url,
            "message": "This was a simulated phishing test. You've been enrolled in security awareness training.",
            "training_assigned": True
        }

    def provide_reported_feedback(self, user_email: str, campaign_id: str) -> Dict:
        """Provide feedback when user reports phishing"""
        # Reward points
        self.award_points(user_email, 20)

        # Update risk score
        self.update_risk_score(user_email, "decrease", 5)

        return {
            "message": "Excellent work! You correctly identified this phishing attempt.",
            "points_awarded": 20,
            "badge": self.check_for_badge(user_email)
        }

    def generate_campaign_report(self, campaign_id: str) -> Dict:
        """Generate comprehensive campaign report"""
        campaign = self.get_campaign(campaign_id)
        tracking_data = self.get_campaign_tracking(campaign_id)

        # Calculate metrics
        total_sent = len(campaign["target_group"])
        clicked = len([t for t in tracking_data if "clicked" in t["actions"]])
        reported = len([t for t in tracking_data if "reported" in t["actions"]])
        compromised = len([t for t in tracking_data if "submitted_credentials" in t["actions"]])

        report = {
            "campaign_id": campaign_id,
            "campaign_name": campaign["name"],
            "execution_date": campaign["started_at"],
            "statistics": {
                "emails_sent": total_sent,
                "click_rate": (clicked / total_sent * 100) if total_sent > 0 else 0,
                "report_rate": (reported / total_sent * 100) if total_sent > 0 else 0,
                "compromise_rate": (compromised / total_sent * 100) if total_sent > 0 else 0
            },
            "risk_analysis": {
                "high_risk_users": self.identify_high_risk_users(tracking_data),
                "department_breakdown": self.analyze_by_department(tracking_data),
                "repeat_clickers": self.identify_repeat_clickers()
            },
            "training_impact": {
                "assigned_training": self.count_assigned_training(campaign_id),
                "completed_training": self.count_completed_training(campaign_id),
                "effectiveness": self.measure_training_effectiveness()
            },
            "recommendations": self.generate_recommendations(tracking_data)
        }

        return report

    def measure_training_effectiveness(self) -> Dict:
        """Measure effectiveness of phishing training"""
        # Compare click rates before and after training
        historical_data = self.get_historical_campaign_data()

        trained_users = self.get_trained_users()
        untrained_users = self.get_untrained_users()

        return {
            "trained_user_click_rate": self.calculate_click_rate(trained_users),
            "untrained_user_click_rate": self.calculate_click_rate(untrained_users),
            "improvement_percentage": self.calculate_improvement(),
            "training_roi": self.calculate_training_roi()
        }
```

## 3. Security Champions Program

### Champions Program Structure

```yaml
# security-champions.yaml
security_champions_program:
  program_structure:
    eligibility:
      criteria:
        - "Minimum 1 year with company"
        - "Good standing in current role"
        - "Manager approval"
        - "Interest in security"

      selection_process:
        - "Self-nomination or manager nomination"
        - "Basic security assessment"
        - "Interview with security team"
        - "Commitment agreement (4-6 hours/month)"

    roles_responsibilities:
      security_champion:
        time_commitment: "10% (4 hours/week)"
        responsibilities:
          - "Security point of contact for team"
          - "Participate in security reviews"
          - "Promote security best practices"
          - "Report security concerns"
          - "Attend monthly champions meeting"
          - "Complete quarterly training"

      senior_champion:
        time_commitment: "15% (6 hours/week)"
        additional_responsibilities:
          - "Mentor new champions"
          - "Lead security initiatives"
          - "Conduct team training"
          - "Participate in incident response"

      champion_lead:
        time_commitment: "20% (8 hours/week)"
        additional_responsibilities:
          - "Coordinate champions program"
          - "Develop training materials"
          - "Report to security leadership"
          - "Drive security culture"

    benefits:
      professional_development:
        - "Security training and certifications"
        - "Conference attendance"
        - "Access to security tools"
        - "Mentorship from security team"

      recognition:
        - "Champion badge/title"
        - "Performance review credit"
        - "Quarterly awards"
        - "Annual champion summit"

      career_advancement:
        - "Security team rotation opportunity"
        - "Priority for security roles"
        - "Leadership development"
        - "Executive visibility"

  training_curriculum:
    onboarding:
      week_1:
        - "Champions program overview"
        - "Security fundamentals"
        - "Threat landscape"
        - "Company security policies"

      week_2:
        - "Secure development lifecycle"
        - "Security tools and resources"
        - "Incident response basics"
        - "Communication skills"

      week_3:
        - "Threat modeling"
        - "Security testing basics"
        - "Vulnerability management"
        - "Risk assessment"

      week_4:
        - "Security metrics"
        - "Compliance basics"
        - "Security awareness"
        - "Final assessment"

    ongoing_training:
      monthly:
        - "Security topic deep-dive"
        - "Tool training"
        - "Case study review"
        - "Q&A with security team"

      quarterly:
        - "Advanced security workshop"
        - "Tabletop exercise"
        - "External speaker session"
        - "Certification prep"

      annual:
        - "Champions summit"
        - "Security conference"
        - "Advanced certification"
        - "Leadership training"

  activities:
    regular_activities:
      code_reviews:
        frequency: "All PRs in team"
        focus:
          - "Security vulnerabilities"
          - "Secure coding practices"
          - "Dependency risks"
          - "Secret detection"

      threat_modeling:
        frequency: "New features/quarterly"
        participation: "Required for major features"

      security_testing:
        - "Participate in pen test planning"
        - "Validate security fixes"
        - "Conduct security test cases"

      awareness_activities:
        - "Team security briefings"
        - "Lunch and learn sessions"
        - "Security tips newsletter"
        - "Phishing awareness"

    special_projects:
      - "Security tool evaluation"
      - "Process improvement"
      - "Training material development"
      - "Security automation"
      - "Incident response support"

  metrics:
    program_metrics:
      - champions_per_team
      - training_completion_rate
      - security_issues_identified
      - team_security_scores
      - incident_response_participation

    champion_performance:
      - activities_completed
      - issues_identified
      - training_delivered
      - team_improvement
      - peer_feedback

    program_impact:
      - vulnerability_reduction
      - incident_reduction
      - security_awareness_increase
      - faster_issue_resolution
      - culture_improvement

  governance:
    steering_committee:
      members:
        - "CISO"
        - "Engineering VP"
        - "HR Representative"
        - "Champion Leads"

      responsibilities:
        - "Program strategy"
        - "Resource allocation"
        - "Success metrics"
        - "Issue resolution"

    meetings:
      champions_monthly:
        agenda:
          - "Security updates"
          - "Best practices sharing"
          - "Challenge discussion"
          - "Training topic"

      leadership_quarterly:
        agenda:
          - "Program metrics review"
          - "Champion recognition"
          - "Strategic planning"
          - "Budget review"
```

### Champions Platform Implementation

```python
# security_champions_platform.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class SecurityChampion:
    employee_id: str
    name: str
    email: str
    team: str
    level: str  # champion, senior, lead
    joined_date: datetime
    training_completed: List[str]
    activities: List[Dict]
    badges: List[str]
    points: int
    active: bool

class ChampionsPlatform:
    def __init__(self):
        self.champions = {}
        self.activities = []
        self.training_modules = {}
        self.metrics = {}

    def nominate_champion(self, nominee: Dict) -> str:
        """Process champion nomination"""
        nomination = {
            "id": self.generate_nomination_id(),
            "nominee": nominee,
            "nominated_at": datetime.now(),
            "status": "pending",
            "assessment_score": None,
            "interview_notes": None,
            "decision": None
        }

        # Initial eligibility check
        if not self.check_eligibility(nominee):
            nomination["status"] = "ineligible"
            nomination["decision"] = "Does not meet eligibility criteria"
            return nomination["id"]

        # Schedule assessment
        self.schedule_assessment(nomination["id"])

        return nomination["id"]

    def check_eligibility(self, nominee: Dict) -> bool:
        """Check if nominee meets eligibility criteria"""
        criteria = {
            "tenure": nominee.get("tenure_months", 0) >= 12,
            "performance": nominee.get("performance_rating", 0) >= 3,
            "manager_approval": nominee.get("manager_approved", False),
            "availability": nominee.get("time_available", 0) >= 4
        }

        return all(criteria.values())

    def onboard_champion(self, champion_data: Dict) -> SecurityChampion:
        """Onboard new security champion"""
        champion = SecurityChampion(
            employee_id=champion_data["employee_id"],
            name=champion_data["name"],
            email=champion_data["email"],
            team=champion_data["team"],
            level="champion",
            joined_date=datetime.now(),
            training_completed=[],
            activities=[],
            badges=["New Champion"],
            points=100,
            active=True
        )

        # Assign onboarding training
        self.assign_onboarding_training(champion)

        # Add to platform
        self.champions[champion.employee_id] = champion

        # Send welcome package
        self.send_welcome_package(champion)

        # Notify team
        self.notify_team_of_champion(champion)

        return champion

    def assign_onboarding_training(self, champion: SecurityChampion):
        """Assign onboarding training modules"""
        onboarding_modules = [
            {
                "id": "SCO-001",
                "title": "Security Champions Program Overview",
                "duration": 60,
                "due_date": datetime.now() + timedelta(days=7)
            },
            {
                "id": "SCO-002",
                "title": "Security Fundamentals",
                "duration": 120,
                "due_date": datetime.now() + timedelta(days=14)
            },
            {
                "id": "SCO-003",
                "title": "Threat Modeling Basics",
                "duration": 90,
                "due_date": datetime.now() + timedelta(days=21)
            },
            {
                "id": "SCO-004",
                "title": "Secure Code Review",
                "duration": 120,
                "due_date": datetime.now() + timedelta(days=28)
            }
        ]

        for module in onboarding_modules:
            self.assign_training(champion.employee_id, module)

    def record_activity(self, champion_id: str, activity: Dict):
        """Record champion security activity"""
        champion = self.champions.get(champion_id)

        if not champion:
            raise ValueError(f"Champion {champion_id} not found")

        activity_record = {
            "id": self.generate_activity_id(),
            "champion_id": champion_id,
            "type": activity["type"],
            "description": activity["description"],
            "date": datetime.now(),
            "impact": activity.get("impact", "low"),
            "points_earned": self.calculate_activity_points(activity)
        }

        champion.activities.append(activity_record)
        champion.points += activity_record["points_earned"]

        # Check for badge eligibility
        self.check_badge_eligibility(champion)

        # Update metrics
        self.update_metrics(activity_record)

        return activity_record

    def calculate_activity_points(self, activity: Dict) -> int:
        """Calculate points for security activity"""
        point_values = {
            "code_review": 10,
            "vulnerability_found": 50,
            "training_delivered": 30,
            "threat_model": 25,
            "incident_response": 40,
            "documentation": 15,
            "mentoring": 20,
            "tool_improvement": 25
        }

        base_points = point_values.get(activity["type"], 5)

        # Apply multipliers
        if activity.get("impact") == "high":
            base_points *= 2
        elif activity.get("impact") == "critical":
            base_points *= 3

        return base_points

    def check_badge_eligibility(self, champion: SecurityChampion):
        """Check and award badges"""
        badge_criteria = {
            "Code Reviewer": lambda c: len([a for a in c.activities if a["type"] == "code_review"]) >= 10,
            "Bug Hunter": lambda c: len([a for a in c.activities if a["type"] == "vulnerability_found"]) >= 5,
            "Educator": lambda c: len([a for a in c.activities if a["type"] == "training_delivered"]) >= 3,
            "First Responder": lambda c: len([a for a in c.activities if a["type"] == "incident_response"]) >= 2,
            "Security Expert": lambda c: c.points >= 1000,
            "Team Leader": lambda c: c.level in ["senior", "lead"],
            "Annual Champion": lambda c: (datetime.now() - c.joined_date).days >= 365
        }

        for badge_name, criteria in badge_criteria.items():
            if badge_name not in champion.badges and criteria(champion):
                champion.badges.append(badge_name)
                self.notify_badge_earned(champion, badge_name)

    def promote_champion(self, champion_id: str, new_level: str):
        """Promote champion to higher level"""
        champion = self.champions.get(champion_id)

        if not champion:
            raise ValueError(f"Champion {champion_id} not found")

        promotion_requirements = {
            "senior": {
                "min_tenure_months": 6,
                "min_points": 500,
                "required_badges": ["Code Reviewer", "Bug Hunter"],
                "training_modules": ["Advanced Security"]
            },
            "lead": {
                "min_tenure_months": 12,
                "min_points": 1500,
                "required_badges": ["Security Expert", "Educator"],
                "training_modules": ["Leadership", "Advanced Security"]
            }
        }

        requirements = promotion_requirements.get(new_level)

        if self.meets_promotion_requirements(champion, requirements):
            champion.level = new_level
            champion.badges.append(f"Promoted to {new_level}")

            # Assign new responsibilities
            self.assign_level_responsibilities(champion, new_level)

            return True

        return False

    def generate_champion_report(self, champion_id: str) -> Dict:
        """Generate individual champion report"""
        champion = self.champions.get(champion_id)

        if not champion:
            raise ValueError(f"Champion {champion_id} not found")

        return {
            "champion": {
                "name": champion.name,
                "team": champion.team,
                "level": champion.level,
                "tenure": (datetime.now() - champion.joined_date).days
            },
            "achievements": {
                "total_points": champion.points,
                "badges": champion.badges,
                "activities_completed": len(champion.activities),
                "training_completed": len(champion.training_completed)
            },
            "recent_activities": champion.activities[-10:],
            "impact_metrics": self.calculate_champion_impact(champion),
            "recommendations": self.generate_champion_recommendations(champion)
        }

    def calculate_champion_impact(self, champion: SecurityChampion) -> Dict:
        """Calculate champion's security impact"""
        return {
            "vulnerabilities_prevented": len([
                a for a in champion.activities
                if a["type"] == "vulnerability_found"
            ]),
            "team_members_trained": len([
                a for a in champion.activities
                if a["type"] == "training_delivered"
            ]),
            "security_reviews_conducted": len([
                a for a in champion.activities
                if a["type"] == "code_review"
            ]),
            "estimated_risk_reduction": self.estimate_risk_reduction(champion),
            "team_security_improvement": self.measure_team_improvement(champion.team)
        }

    def run_gamification_leaderboard(self) -> List[Dict]:
        """Generate gamification leaderboard"""
        leaderboard = []

        for champion in self.champions.values():
            if champion.active:
                leaderboard.append({
                    "rank": 0,
                    "name": champion.name,
                    "team": champion.team,
                    "points": champion.points,
                    "badges": len(champion.badges),
                    "level": champion.level,
                    "recent_achievement": self.get_recent_achievement(champion)
                })

        # Sort by points and assign ranks
        leaderboard.sort(key=lambda x: x["points"], reverse=True)
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    def schedule_champion_meeting(self, meeting_type: str) -> Dict:
        """Schedule champions meeting"""
        meeting_types = {
            "monthly": {
                "duration": 60,
                "recurring": True,
                "agenda": [
                    "Security updates",
                    "Best practices sharing",
                    "Challenge discussion",
                    "Training topic"
                ]
            },
            "quarterly": {
                "duration": 120,
                "recurring": True,
                "agenda": [
                    "Program metrics",
                    "Champion recognition",
                    "Strategic planning",
                    "Advanced training"
                ]
            },
            "annual_summit": {
                "duration": 480,
                "recurring": False,
                "agenda": [
                    "Keynote presentations",
                    "Workshops",
                    "Team building",
                    "Awards ceremony",
                    "Future planning"
                ]
            }
        }

        meeting_config = meeting_types.get(meeting_type)

        if not meeting_config:
            raise ValueError(f"Unknown meeting type: {meeting_type}")

        meeting = {
            "id": self.generate_meeting_id(),
            "type": meeting_type,
            "scheduled_for": self.get_next_meeting_date(meeting_type),
            "duration": meeting_config["duration"],
            "agenda": meeting_config["agenda"],
            "invitees": self.get_meeting_invitees(meeting_type),
            "resources": self.prepare_meeting_resources(meeting_type)
        }

        # Send invitations
        self.send_meeting_invitations(meeting)

        return meeting
```

## 4. Certification Requirements

### Security Certification Program

```yaml
# certification-requirements.yaml
certification_program:
  required_certifications:
    security_team:
      entry_level:
        options:
          - "CompTIA Security+"
          - "GSEC (GIAC Security Essentials)"
        timeline: "Within 6 months of joining"
        company_support:
          - "Paid exam fees"
          - "Study materials"
          - "Study time: 2 hours/week"

      mid_level:
        options:
          - "CISSP Associate"
          - "CEH (Certified Ethical Hacker)"
          - "GCIH (GIAC Incident Handler)"
        timeline: "Within 2 years"
        company_support:
          - "Paid training course"
          - "Paid exam fees"
          - "Study time: 4 hours/week"

      senior_level:
        required:
          - "CISSP"
        optional:
          - "CCSP (Cloud Security)"
          - "OSCP (Offensive Security)"
          - "GIAC Expert"
        timeline: "Within 3 years"
        company_support:
          - "Full training sponsorship"
          - "Conference attendance"
          - "Study time: 6 hours/week"

    developers:
      recommended:
        - "Secure Coding Certification"
        - "Cloud Security Basics"
      company_support:
        - "Online course subscriptions"
        - "Internal training programs"

    devops:
      recommended:
        - "AWS Security Specialty"
        - "Kubernetes Security Specialist"
        - "DevSecOps Certification"
      company_support:
        - "Training budget: $3000/year"
        - "Lab environment access"

  certification_tracks:
    offensive_security:
      progression:
        1: "CompTIA PenTest+"
        2: "GPEN (GIAC Penetration Tester)"
        3: "OSCP"
        4: "OSCE (Advanced)"

    defensive_security:
      progression:
        1: "CompTIA CySA+"
        2: "GCIH"
        3: "GNFA (Network Forensics)"
        4: "GCFA (Advanced Forensics)"

    cloud_security:
      progression:
        1: "Cloud Security Alliance CCSK"
        2: "AWS Security Specialty"
        3: "Azure Security Engineer"
        4: "CCSP"

    management:
      progression:
        1: "CISA (IS Auditor)"
        2: "CISM (IS Manager)"
        3: "CRISC (Risk and IS Control)"
        4: "CGEIT (Governance)"

  certification_maintenance:
    continuing_education:
      requirements:
        - "Annual CPE credits"
        - "Security conference attendance"
        - "Training completion"
        - "Knowledge sharing"

      tracking:
        - "Quarterly CPE reports"
        - "Certificate renewal reminders"
        - "Budget allocation"

  study_groups:
    format:
      - "Weekly virtual sessions"
      - "Practice exams"
      - "Lab exercises"
      - "Mentorship pairing"

    resources:
      - "Online learning platforms"
      - "Books and study guides"
      - "Practice lab environments"
      - "Internal wiki/knowledge base"

  recognition:
    achievements:
      - "Certification bonuses"
      - "Public recognition"
      - "LinkedIn endorsements"
      - "Career advancement priority"

    incentives:
      first_attempt_pass:
        bonus: "$500"

      advanced_certification:
        bonus: "$1000-2000"

      knowledge_sharing:
        - "Present at team meeting: 50 points"
        - "Write blog post: 100 points"
        - "Mentor others: 200 points"
```

This comprehensive security training program provides structured learning paths for developers, robust phishing simulation capabilities, a security champions program to distribute security expertise across teams, and clear certification requirements with company support. The program emphasizes hands-on learning, continuous education, and recognition for security achievements.

The framework includes automated platforms for delivering training, tracking progress, measuring effectiveness, and maintaining engagement through gamification and recognition programs. This ensures that security becomes embedded in the company culture rather than being seen as a separate function.