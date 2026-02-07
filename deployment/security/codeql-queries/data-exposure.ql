/**
 * @name GreenLang Sensitive Data Exposure Detection
 * @description Detects potential exposure of sensitive data through logging,
 *              API responses, error messages, and insecure storage.
 * @kind problem
 * @problem.severity error
 * @security-severity 7.5
 * @precision high
 * @id greenlang/data-exposure
 * @tags security
 *       external/cwe/cwe-200
 *       external/cwe/cwe-532
 *       greenlang
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import semmle.python.ApiGraphs

/**
 * A variable or attribute that may contain sensitive data.
 */
class SensitiveData extends DataFlow::Node {
  string dataType;

  SensitiveData() {
    exists(string name |
      (
        this.asCfgNode().(NameNode).getId() = name
        or
        this.asCfgNode().(AttrNode).getName() = name
      ) and
      (
        name.toLowerCase().regexpMatch(".*(password|passwd|pwd|secret|token|api_key|apikey|access_key|private_key).*") and dataType = "credential"
        or
        name.toLowerCase().regexpMatch(".*(ssn|social_security|tax_id|national_id).*") and dataType = "PII-SSN"
        or
        name.toLowerCase().regexpMatch(".*(credit_card|card_number|cvv|ccn|pan).*") and dataType = "PCI"
        or
        name.toLowerCase().regexpMatch(".*(bank_account|routing_number|iban|swift).*") and dataType = "financial"
        or
        name.toLowerCase().regexpMatch(".*(dob|date_of_birth|birthdate).*") and dataType = "PII"
        or
        name.toLowerCase().regexpMatch(".*(medical|diagnosis|prescription|health).*") and dataType = "PHI"
      )
    )
  }

  string getDataType() { result = dataType }
}

/**
 * A logging call that may expose sensitive data.
 */
class LoggingCall extends DataFlow::CallCfgNode {
  string level;

  LoggingCall() {
    exists(DataFlow::AttrRead attr |
      this.getFunction() = attr and
      attr.getAttributeName() in ["debug", "info", "warning", "error", "critical", "exception"] and
      level = attr.getAttributeName()
    )
    or
    exists(DataFlow::Node func |
      this.getFunction() = func and
      func.asCfgNode().(NameNode).getId() = "print" and
      level = "print"
    )
  }

  string getLevel() { result = level }
}

from LoggingCall log, SensitiveData sensitive
where
  sensitive = log.getAnArg() or
  exists(DataFlow::Node fmtArg |
    fmtArg = log.getArg(0) and
    // Check if the format string contains the sensitive variable
    exists(BinaryExpr binop |
      binop = fmtArg.asCfgNode().getNode() and
      binop.getOp() instanceof Mod and
      sensitive.asCfgNode().getNode() = binop.getRight()
    )
  )
select log,
  "Potential exposure of " + sensitive.getDataType() + " data in log statement."


/**
 * @name GreenLang PII in API Response
 * @description Detects potential PII exposure in API responses.
 * @kind problem
 * @problem.severity error
 * @security-severity 7.5
 * @precision high
 * @id greenlang/pii-in-response
 * @tags security
 *       external/cwe/cwe-359
 *       external/cwe/cwe-200
 *       greenlang
 *       gdpr
 */

/**
 * A return statement in an API endpoint.
 */
class ApiReturnStatement extends Return {
  ApiReturnStatement() {
    exists(Function f |
      this.getEnclosingScope() = f and
      exists(Decorator d |
        d = f.getADecorator() and
        d.getValue().(Call).getFunc().(Attribute).getName() in [
          "get", "post", "put", "delete", "patch", "route", "api_route"
        ]
      )
    )
  }
}

/**
 * A dictionary literal that may contain sensitive keys.
 */
class SensitiveResponseDict extends Dict {
  string sensitiveKey;

  SensitiveResponseDict() {
    exists(DictItem item |
      item = this.getAnItem() and
      item.getKey().(StrConst).getText() = sensitiveKey and
      sensitiveKey.toLowerCase().regexpMatch(
        ".*(ssn|social_security|password|credit_card|card_number|cvv|bank_account|" +
        "private_key|secret|api_key|access_token|refresh_token|" +
        "date_of_birth|dob|medical_record|diagnosis|tax_id).*"
      )
    )
  }

  string getSensitiveKey() { result = sensitiveKey }
}

from ApiReturnStatement ret, SensitiveResponseDict dict
where ret.getValue() = dict
select ret,
  "API response may expose sensitive data through key '" + dict.getSensitiveKey() + "'."


/**
 * @name GreenLang Stack Trace Exposure
 * @description Detects exposure of stack traces to end users.
 * @kind problem
 * @problem.severity warning
 * @security-severity 5.3
 * @precision high
 * @id greenlang/stack-trace-exposure
 * @tags security
 *       external/cwe/cwe-209
 *       greenlang
 */

class TracebackCall extends DataFlow::CallCfgNode {
  TracebackCall() {
    exists(DataFlow::AttrRead attr |
      this.getFunction() = attr and
      attr.getAttributeName() in ["format_exc", "format_exception", "print_exc"]
    )
    or
    exists(DataFlow::Node func |
      this.getFunction() = func and
      func.asCfgNode().(AttrNode).getName() = "format_tb"
    )
  }
}

class ExceptionHandler extends ExceptStmt {
  ExceptionHandler() { any() }

  predicate returnsTraceback() {
    exists(Return r, TracebackCall tb |
      r.getEnclosingScope() = this.getEnclosingScope() and
      r.getLocation().getStartLine() > this.getLocation().getStartLine() and
      tb.asCfgNode().getNode().getLocation().getStartLine() < r.getLocation().getStartLine()
    )
  }
}

from ExceptionHandler handler
where handler.returnsTraceback()
select handler,
  "Exception handler may expose stack trace to end users. Use structured error responses in production."


/**
 * @name GreenLang Verbose Error Messages
 * @description Detects error messages that may reveal internal implementation details.
 * @kind problem
 * @problem.severity warning
 * @security-severity 5.3
 * @precision medium
 * @id greenlang/verbose-error
 * @tags security
 *       external/cwe/cwe-209
 *       greenlang
 */

class VerboseExceptionMessage extends Raise {
  VerboseExceptionMessage() {
    exists(Call c |
      c = this.getRaised() and
      exists(StrConst s |
        s = c.getAnArg() and
        (
          s.getText().regexpMatch(".*\\b(table|column|database|query|SQL|SELECT|INSERT|UPDATE|DELETE)\\b.*")
          or
          s.getText().regexpMatch(".*\\b(file|path|directory|/home/|/var/|/etc/)\\b.*")
          or
          s.getText().regexpMatch(".*\\b(internal|server|backend|implementation)\\b.*")
        )
      )
    )
  }
}

from VerboseExceptionMessage exc
select exc,
  "Exception message may reveal internal implementation details. Use generic error messages for end users."


/**
 * @name GreenLang Unencrypted Sensitive Data Storage
 * @description Detects storage of sensitive data without encryption.
 * @kind problem
 * @problem.severity error
 * @security-severity 7.5
 * @precision medium
 * @id greenlang/unencrypted-storage
 * @tags security
 *       external/cwe/cwe-312
 *       greenlang
 */

class FileWrite extends DataFlow::CallCfgNode {
  FileWrite() {
    exists(DataFlow::AttrRead attr |
      this.getFunction() = attr and
      attr.getAttributeName() in ["write", "writelines", "dump", "dumps"]
    )
  }
}

class SensitiveDataWrite extends FileWrite {
  SensitiveData sensitive;

  SensitiveDataWrite() {
    sensitive = this.getAnArg()
  }

  SensitiveData getSensitiveData() { result = sensitive }
}

from SensitiveDataWrite write
where
  not exists(DataFlow::CallCfgNode encrypt |
    encrypt.getFunction().(DataFlow::AttrRead).getAttributeName() in [
      "encrypt", "encode", "fernet", "cipher"
    ] and
    encrypt.getLocation().getStartLine() < write.getLocation().getStartLine()
  )
select write,
  "Sensitive data (" + write.getSensitiveData().getDataType() + ") may be written without encryption."


/**
 * @name GreenLang Insecure Temporary File
 * @description Detects insecure temporary file usage that may expose sensitive data.
 * @kind problem
 * @problem.severity warning
 * @security-severity 5.3
 * @precision high
 * @id greenlang/insecure-tempfile
 * @tags security
 *       external/cwe/cwe-377
 *       greenlang
 */

class InsecureTempFile extends DataFlow::CallCfgNode {
  InsecureTempFile() {
    exists(DataFlow::AttrRead attr |
      this.getFunction() = attr and
      attr.getAttributeName() = "mktemp"
    )
  }
}

from InsecureTempFile tempfile
select tempfile,
  "Use of tempfile.mktemp() is insecure due to race condition. Use tempfile.mkstemp() or tempfile.NamedTemporaryFile() instead."


/**
 * @name GreenLang Debug Mode in Production
 * @description Detects debug mode settings that may expose sensitive information.
 * @kind problem
 * @problem.severity warning
 * @security-severity 5.3
 * @precision high
 * @id greenlang/debug-mode
 * @tags security
 *       external/cwe/cwe-489
 *       greenlang
 */

class DebugSetting extends Assign {
  DebugSetting() {
    exists(Name target |
      target = this.getATarget() and
      target.getId().toUpperCase() in ["DEBUG", "FLASK_DEBUG", "DJANGO_DEBUG"] and
      this.getValue().(NameConstant).getValue().(BooleanLiteral).booleanValue() = true
    )
  }
}

from DebugSetting debug
select debug,
  "Debug mode enabled. Ensure this is disabled in production to prevent information disclosure."
