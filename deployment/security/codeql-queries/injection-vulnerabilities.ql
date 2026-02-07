/**
 * @name GreenLang SQL Injection Detection
 * @description Detects SQL injection vulnerabilities in GreenLang codebase,
 *              including async database operations with psycopg and asyncpg.
 * @kind path-problem
 * @problem.severity error
 * @security-severity 9.8
 * @precision high
 * @id greenlang/sql-injection
 * @tags security
 *       external/cwe/cwe-089
 *       greenlang
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import semmle.python.dataflow.new.RemoteFlowSources
import semmle.python.ApiGraphs

/**
 * A sink that represents SQL query execution.
 */
class SqlQuerySink extends DataFlow::Node {
  SqlQuerySink() {
    // psycopg2/psycopg execute
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() = "execute" and
      this = call.getArg(0)
    )
    or
    // asyncpg execute/fetch
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() in [
        "execute", "fetch", "fetchrow", "fetchval"
      ] and
      this = call.getArg(0)
    )
    or
    // SQLAlchemy text()
    exists(DataFlow::CallCfgNode call |
      call.getFunction().asCfgNode().(NameNode).getId() = "text" and
      this = call.getArg(0)
    )
    or
    // Raw SQL in ORM
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() in [
        "raw", "execute_sql", "run_sql"
      ] and
      this = call.getArg(0)
    )
  }
}

/**
 * A source of user-controlled data.
 */
class UserInputSource extends DataFlow::Node {
  UserInputSource() {
    // FastAPI request parameters
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() in [
        "query_params", "path_params", "body", "json"
      ] and
      this = call
    )
    or
    // Flask/Starlette request
    exists(DataFlow::AttrRead attr |
      attr.getAttributeName() in ["args", "form", "data", "json", "values"] and
      this = attr
    )
    or
    // Standard remote flow sources
    this instanceof RemoteFlowSource
  }
}

/**
 * Configuration for SQL injection taint tracking.
 */
module SqlInjectionConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    source instanceof UserInputSource
  }

  predicate isSink(DataFlow::Node sink) {
    sink instanceof SqlQuerySink
  }

  predicate isBarrier(DataFlow::Node node) {
    // Parameterized query - tuple/list as second arg
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() = "execute" and
      call.getArg(1).asCfgNode().(TupleNode).getAnElement() = node.asCfgNode()
    )
    or
    // Sanitization functions
    exists(DataFlow::CallCfgNode call |
      call.getFunction().asCfgNode().(NameNode).getId() in [
        "escape", "quote", "sanitize", "escape_string"
      ] and
      call.getArg(0) = node
    )
  }
}

module SqlInjectionFlow = TaintTracking::Global<SqlInjectionConfig>;
import SqlInjectionFlow::PathGraph

from SqlInjectionFlow::PathNode source, SqlInjectionFlow::PathNode sink
where SqlInjectionFlow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "SQL injection vulnerability: user input from $@ flows to SQL query.",
  source.getNode(), "user input"


/**
 * @name GreenLang Command Injection Detection
 * @description Detects command injection vulnerabilities where user input
 *              flows to shell command execution.
 * @kind path-problem
 * @problem.severity error
 * @security-severity 9.8
 * @precision high
 * @id greenlang/command-injection
 * @tags security
 *       external/cwe/cwe-078
 *       greenlang
 */

/**
 * A sink that represents command execution.
 */
class CommandExecutionSink extends DataFlow::Node {
  CommandExecutionSink() {
    // os.system
    exists(DataFlow::CallCfgNode call |
      call.getFunction().asCfgNode().(AttrNode).getName() = "system" and
      this = call.getArg(0)
    )
    or
    // subprocess with shell=True
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() in [
        "run", "call", "Popen", "check_output", "check_call"
      ] and
      exists(DataFlow::Node shellArg |
        shellArg = call.getArgByName("shell") and
        shellArg.asCfgNode().(NameNode).getId() = "True"
      ) and
      this = call.getArg(0)
    )
    or
    // eval/exec
    exists(DataFlow::CallCfgNode call |
      call.getFunction().asCfgNode().(NameNode).getId() in ["eval", "exec"] and
      this = call.getArg(0)
    )
  }
}

module CommandInjectionConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    source instanceof UserInputSource
  }

  predicate isSink(DataFlow::Node sink) {
    sink instanceof CommandExecutionSink
  }

  predicate isBarrier(DataFlow::Node node) {
    // shlex.quote sanitization
    exists(DataFlow::CallCfgNode call |
      call.getFunction().asCfgNode().(AttrNode).getName() = "quote" and
      call.getArg(0) = node
    )
  }
}

module CommandInjectionFlow = TaintTracking::Global<CommandInjectionConfig>;

from CommandInjectionFlow::PathNode source, CommandInjectionFlow::PathNode sink
where CommandInjectionFlow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "Command injection vulnerability: user input from $@ flows to shell command.",
  source.getNode(), "user input"


/**
 * @name GreenLang LDAP Injection Detection
 * @description Detects LDAP injection vulnerabilities in directory service queries.
 * @kind path-problem
 * @problem.severity error
 * @security-severity 8.1
 * @precision high
 * @id greenlang/ldap-injection
 * @tags security
 *       external/cwe/cwe-090
 *       greenlang
 */

class LdapQuerySink extends DataFlow::Node {
  LdapQuerySink() {
    exists(DataFlow::CallCfgNode call |
      call.getFunction().(DataFlow::AttrRead).getAttributeName() in [
        "search", "search_s", "search_st", "search_ext", "search_ext_s"
      ] and
      // Filter is typically the 3rd argument
      this = call.getArg(2)
    )
  }
}

module LdapInjectionConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    source instanceof UserInputSource
  }

  predicate isSink(DataFlow::Node sink) {
    sink instanceof LdapQuerySink
  }

  predicate isBarrier(DataFlow::Node node) {
    // ldap.filter.escape_filter_chars
    exists(DataFlow::CallCfgNode call |
      call.getFunction().asCfgNode().(AttrNode).getName() = "escape_filter_chars" and
      call.getArg(0) = node
    )
  }
}

module LdapInjectionFlow = TaintTracking::Global<LdapInjectionConfig>;

from LdapInjectionFlow::PathNode source, LdapInjectionFlow::PathNode sink
where LdapInjectionFlow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "LDAP injection vulnerability: user input from $@ flows to LDAP filter.",
  source.getNode(), "user input"
