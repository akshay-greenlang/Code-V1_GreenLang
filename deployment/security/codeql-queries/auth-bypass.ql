/**
 * @name GreenLang Authentication Bypass Detection
 * @description Detects potential authentication bypass vulnerabilities including
 *              hardcoded credentials, weak comparisons, and missing auth checks.
 * @kind problem
 * @problem.severity error
 * @security-severity 9.1
 * @precision high
 * @id greenlang/auth-bypass
 * @tags security
 *       external/cwe/cwe-287
 *       external/cwe/cwe-798
 *       greenlang
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.ApiGraphs

/**
 * Detects hardcoded passwords in authentication contexts.
 */
class HardcodedPassword extends DataFlow::Node {
  string value;

  HardcodedPassword() {
    exists(StringValue sv |
      this.asCfgNode() = sv.getAFlowNode() and
      value = sv.getText() and
      // Must be at least 4 characters to be a potential password
      value.length() >= 4 and
      // Exclude obvious non-passwords
      not value.regexpMatch("^(http|https|file|ftp|mailto|tel|data):.*") and
      not value.regexpMatch("^[A-Z_]+$") and // Likely constants
      not value.regexpMatch("^\\s*$") // Empty/whitespace
    )
  }

  string getValue() { result = value }
}

/**
 * A comparison that may be checking credentials.
 */
class CredentialComparison extends Compare {
  CredentialComparison() {
    exists(Name n |
      n = this.getAChildNode() and
      n.getId().toLowerCase().regexpMatch(".*(password|passwd|pwd|secret|token|key|credential|auth).*")
    )
  }
}

/**
 * Hardcoded credential in comparison.
 */
from CredentialComparison cmp, HardcodedPassword hp
where
  hp.asCfgNode().getNode() = cmp.getAChildNode() and
  hp.getValue().length() >= 4
select cmp,
  "Potential hardcoded credential in authentication comparison. Value: '" + hp.getValue().prefix(4) + "...'"


/**
 * @name GreenLang Weak Password Comparison
 * @description Detects timing-unsafe string comparisons for passwords/secrets.
 * @kind problem
 * @problem.severity warning
 * @security-severity 5.3
 * @precision medium
 * @id greenlang/weak-password-comparison
 * @tags security
 *       external/cwe/cwe-208
 *       greenlang
 */

/**
 * A direct string comparison that should use constant-time comparison.
 */
class DirectPasswordComparison extends Compare {
  DirectPasswordComparison() {
    // Comparisons with password-related variables
    exists(Name n |
      n = this.getAChildNode() and
      n.getId().toLowerCase().regexpMatch(".*(password|secret|token|hash|digest).*")
    ) and
    // Using == operator (timing unsafe)
    this.getOp(0) instanceof Eq
  }
}

from DirectPasswordComparison cmp
select cmp,
  "Timing-unsafe password comparison. Use hmac.compare_digest() or secrets.compare_digest() for constant-time comparison."


/**
 * @name GreenLang Missing Authentication Decorator
 * @description Detects API endpoints that may be missing authentication decorators.
 * @kind problem
 * @problem.severity warning
 * @security-severity 7.5
 * @precision medium
 * @id greenlang/missing-auth-decorator
 * @tags security
 *       external/cwe/cwe-306
 *       greenlang
 */

/**
 * A function that appears to be an API endpoint.
 */
class ApiEndpoint extends Function {
  string httpMethod;

  ApiEndpoint() {
    exists(Decorator d |
      d = this.getADecorator() and
      exists(Call c |
        c = d.getValue() and
        exists(Attribute a |
          a = c.getFunc() and
          a.getName() in ["get", "post", "put", "delete", "patch", "api_route", "route"]
        )
      )
    ) and
    httpMethod = this.getADecorator().getValue().(Call).getFunc().(Attribute).getName()
  }

  string getHttpMethod() { result = httpMethod }

  predicate hasAuthDecorator() {
    exists(Decorator d |
      d = this.getADecorator() and
      (
        d.getValue().(Name).getId().toLowerCase().regexpMatch(".*(auth|login|require|protect|secure).*")
        or
        d.getValue().(Call).getFunc().(Name).getId().toLowerCase().regexpMatch(".*(auth|login|require|protect|secure).*")
        or
        d.getValue().(Call).getFunc().(Attribute).getName().toLowerCase().regexpMatch(".*(auth|login|require|protect|secure).*")
        or
        // FastAPI Depends with auth
        exists(Call c |
          c = d.getValue() and
          c.getAnArg().(Call).getFunc().(Name).getId() = "Depends"
        )
      )
    )
  }
}

from ApiEndpoint endpoint
where
  not endpoint.hasAuthDecorator() and
  // Exclude health/status endpoints
  not endpoint.getName().toLowerCase().regexpMatch(".*(health|status|ping|ready|live|version|docs|openapi).*") and
  // Exclude login/auth endpoints themselves
  not endpoint.getName().toLowerCase().regexpMatch(".*(login|logout|register|signup|signin|oauth|callback).*") and
  // Require mutation endpoints to have auth
  endpoint.getHttpMethod() in ["post", "put", "delete", "patch"]
select endpoint,
  "API endpoint '" + endpoint.getName() + "' (" + endpoint.getHttpMethod().toUpperCase() + ") may be missing authentication decorator."


/**
 * @name GreenLang JWT Secret Hardcoding
 * @description Detects hardcoded JWT secrets in code.
 * @kind problem
 * @problem.severity error
 * @security-severity 9.8
 * @precision high
 * @id greenlang/hardcoded-jwt-secret
 * @tags security
 *       external/cwe/cwe-798
 *       greenlang
 */

class JwtCall extends DataFlow::CallCfgNode {
  JwtCall() {
    exists(DataFlow::AttrRead attr |
      this.getFunction() = attr and
      attr.getAttributeName() in ["encode", "decode"]
    )
  }

  DataFlow::Node getSecretArg() {
    // jwt.encode(payload, secret, algorithm)
    // jwt.decode(token, secret, algorithms)
    result = this.getArg(1)
  }
}

from JwtCall jwtCall, HardcodedPassword secret
where jwtCall.getSecretArg() = secret
select jwtCall,
  "Hardcoded JWT secret detected. Use environment variables or secure secrets management."


/**
 * @name GreenLang Session Fixation Risk
 * @description Detects potential session fixation vulnerabilities.
 * @kind problem
 * @problem.severity warning
 * @security-severity 6.5
 * @precision medium
 * @id greenlang/session-fixation
 * @tags security
 *       external/cwe/cwe-384
 *       greenlang
 */

class LoginFunction extends Function {
  LoginFunction() {
    this.getName().toLowerCase().regexpMatch(".*(login|authenticate|signin).*")
  }

  predicate regeneratesSession() {
    exists(Call c |
      c.getEnclosingModule() = this.getEnclosingModule() and
      c.getFunc().(Attribute).getName() in [
        "regenerate", "clear", "flush", "invalidate", "cycle_key"
      ]
    )
  }
}

from LoginFunction login
where not login.regeneratesSession()
select login,
  "Login function '" + login.getName() + "' may not regenerate session after authentication, risking session fixation."
