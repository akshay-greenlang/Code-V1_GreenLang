# GreenLang IDE Extensions Specification
## VSCode & JetBrains Plugin Architecture

### 1. VSCode Extension Architecture

#### Core Technologies
```yaml
Extension Framework: VSCode Extension API
Language: TypeScript 5.3+
Build: ESBuild + Webpack
Testing: Jest + VSCode Test
LSP: Language Server Protocol
Debug: Debug Adapter Protocol (DAP)

Key Dependencies:
  - vscode-languageclient: LSP integration
  - vscode-debugadapter: Debugging support
  - monaco-editor: Code editing
  - @greenlang/sdk: GreenLang SDK
  - axios: API communication
```

#### Extension Structure
```
greenlang-vscode/
├── src/
│   ├── extension.ts          # Extension entry point
│   ├── languageServer/
│   │   ├── server.ts
│   │   ├── parser.ts
│   │   └── validator.ts
│   ├── features/
│   │   ├── completion/
│   │   ├── hover/
│   │   ├── diagnostics/
│   │   ├── formatting/
│   │   └── codeActions/
│   ├── debugger/
│   │   ├── adapter.ts
│   │   └── session.ts
│   ├── views/
│   │   ├── agentExplorer.ts
│   │   ├── traceViewer.ts
│   │   └── testRunner.ts
│   ├── commands/
│   │   ├── deploy.ts
│   │   ├── test.ts
│   │   └── generate.ts
│   └── providers/
│       ├── hoverProvider.ts
│       ├── completionProvider.ts
│       └── definitionProvider.ts
├── syntaxes/
│   └── gcel.tmLanguage.json   # GCEL syntax highlighting
├── snippets/
│   └── gcel.json              # Code snippets
├── icons/
├── package.json
└── tsconfig.json
```

### 2. GCEL Language Support

```typescript
// Language Server Implementation
import {
  createConnection,
  TextDocuments,
  ProposedFeatures,
  InitializeParams,
  CompletionItem,
  CompletionItemKind,
  TextDocumentPositionParams,
  Hover,
  Diagnostic,
  DiagnosticSeverity,
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';

export class GCELLanguageServer {
  private connection = createConnection(ProposedFeatures.all);
  private documents = new TextDocuments(TextDocument);
  private parser: GCELParser;
  private validator: GCELValidator;

  constructor() {
    this.parser = new GCELParser();
    this.validator = new GCELValidator();
    this.setupHandlers();
  }

  private setupHandlers() {
    // Initialize
    this.connection.onInitialize((params: InitializeParams) => {
      return {
        capabilities: {
          textDocumentSync: 1,
          completionProvider: {
            resolveProvider: true,
            triggerCharacters: ['.', '>', '|', ' '],
          },
          hoverProvider: true,
          definitionProvider: true,
          referencesProvider: true,
          documentFormattingProvider: true,
          codeActionProvider: true,
          diagnosticProvider: true,
        },
      };
    });

    // Completion
    this.connection.onCompletion(
      this.handleCompletion.bind(this)
    );

    // Hover
    this.connection.onHover(
      this.handleHover.bind(this)
    );

    // Diagnostics
    this.documents.onDidChangeContent(change => {
      this.validateDocument(change.document);
    });

    // Listen
    this.documents.listen(this.connection);
    this.connection.listen();
  }

  private async handleCompletion(
    params: TextDocumentPositionParams
  ): Promise<CompletionItem[]> {
    const document = this.documents.get(params.textDocument.uri);
    if (!document) return [];

    const context = this.parser.getContext(document, params.position);

    // Agent completions
    if (context.type === 'agent') {
      return await this.getAgentCompletions(context);
    }

    // Operator completions
    if (context.type === 'operator') {
      return this.getOperatorCompletions();
    }

    // Property completions
    if (context.type === 'property') {
      return this.getPropertyCompletions(context.parent);
    }

    return [];
  }

  private async getAgentCompletions(
    context: CompletionContext
  ): Promise<CompletionItem[]> {
    // Fetch available agents from GreenLang API
    const agents = await fetchAvailableAgents();

    return agents.map(agent => ({
      label: agent.name,
      kind: CompletionItemKind.Class,
      detail: agent.type,
      documentation: {
        kind: 'markdown',
        value: agent.documentation,
      },
      insertText: `${agent.name}(${agent.parameters.map(p => `${p.name}=$\{${p.name}\}`).join(', ')})`,
      insertTextFormat: 2, // Snippet
    }));
  }

  private getOperatorCompletions(): CompletionItem[] {
    return [
      {
        label: '>>',
        kind: CompletionItemKind.Operator,
        detail: 'Sequential operator',
        documentation: 'Chain operations sequentially',
        insertText: '>> ',
      },
      {
        label: '|',
        kind: CompletionItemKind.Operator,
        detail: 'Pipe operator',
        documentation: 'Pipe data through operations',
        insertText: '| ',
      },
      {
        label: 'parallel',
        kind: CompletionItemKind.Function,
        detail: 'Parallel execution',
        documentation: 'Execute multiple operations in parallel',
        insertText: 'parallel([\n\t$1\n])',
        insertTextFormat: 2,
      },
      {
        label: 'conditional',
        kind: CompletionItemKind.Function,
        detail: 'Conditional branching',
        documentation: 'Branch based on conditions',
        insertText: 'conditional({\n\t$1\n})',
        insertTextFormat: 2,
      },
    ];
  }

  private async handleHover(
    params: TextDocumentPositionParams
  ): Promise<Hover | null> {
    const document = this.documents.get(params.textDocument.uri);
    if (!document) return null;

    const word = this.parser.getWordAtPosition(document, params.position);
    if (!word) return null;

    // Check if it's an agent
    const agent = await fetchAgentInfo(word);
    if (agent) {
      return {
        contents: {
          kind: 'markdown',
          value: this.formatAgentDocumentation(agent),
        },
      };
    }

    // Check if it's a built-in function
    const builtIn = BUILTIN_FUNCTIONS[word];
    if (builtIn) {
      return {
        contents: {
          kind: 'markdown',
          value: builtIn.documentation,
        },
      };
    }

    return null;
  }

  private async validateDocument(document: TextDocument): Promise<void> {
    const diagnostics: Diagnostic[] = [];

    try {
      const ast = this.parser.parse(document.getText());
      const errors = this.validator.validate(ast);

      for (const error of errors) {
        diagnostics.push({
          severity: DiagnosticSeverity.Error,
          range: error.range,
          message: error.message,
          source: 'gcel',
        });
      }
    } catch (error: any) {
      diagnostics.push({
        severity: DiagnosticSeverity.Error,
        range: {
          start: { line: 0, character: 0 },
          end: { line: 0, character: Number.MAX_VALUE },
        },
        message: error.message,
        source: 'gcel',
      });
    }

    this.connection.sendDiagnostics({
      uri: document.uri,
      diagnostics,
    });
  }

  private formatAgentDocumentation(agent: AgentInfo): string {
    return `
# ${agent.name}

**Type:** ${agent.type}
**Version:** ${agent.version}

${agent.description}

## Parameters

${agent.parameters.map(p => `- **${p.name}** (${p.type})${p.required ? ' *required*' : ''}: ${p.description}`).join('\n')}

## Returns

${agent.returns.description}

## Example

\`\`\`gcel
${agent.example}
\`\`\`
`;
  }
}
```

### 3. Syntax Highlighting (TextMate Grammar)

```json
// gcel.tmLanguage.json
{
  "name": "GreenLang Climate Expression Language",
  "scopeName": "source.gcel",
  "fileTypes": ["gcel", "gl"],
  "patterns": [
    {
      "include": "#comments"
    },
    {
      "include": "#keywords"
    },
    {
      "include": "#operators"
    },
    {
      "include": "#agents"
    },
    {
      "include": "#strings"
    },
    {
      "include": "#numbers"
    },
    {
      "include": "#functions"
    }
  ],
  "repository": {
    "comments": {
      "patterns": [
        {
          "name": "comment.line.number-sign.gcel",
          "match": "#.*$"
        },
        {
          "name": "comment.block.gcel",
          "begin": "/\\*",
          "end": "\\*/"
        }
      ]
    },
    "keywords": {
      "patterns": [
        {
          "name": "keyword.control.gcel",
          "match": "\\b(if|else|for|while|return|import|from|as)\\b"
        },
        {
          "name": "keyword.operator.gcel",
          "match": "\\b(parallel|sequential|conditional|route|aggregate)\\b"
        }
      ]
    },
    "operators": {
      "patterns": [
        {
          "name": "keyword.operator.chain.gcel",
          "match": "(>>|\\||=>)"
        },
        {
          "name": "keyword.operator.logical.gcel",
          "match": "(and|or|not|==|!=|<|>|<=|>=)"
        }
      ]
    },
    "agents": {
      "patterns": [
        {
          "name": "entity.name.class.agent.gcel",
          "match": "\\b([A-Z][a-zA-Z0-9_]*)(?=\\()"
        }
      ]
    },
    "strings": {
      "patterns": [
        {
          "name": "string.quoted.double.gcel",
          "begin": "\"",
          "end": "\"",
          "patterns": [
            {
              "name": "constant.character.escape.gcel",
              "match": "\\\\."
            }
          ]
        },
        {
          "name": "string.quoted.single.gcel",
          "begin": "'",
          "end": "'",
          "patterns": [
            {
              "name": "constant.character.escape.gcel",
              "match": "\\\\."
            }
          ]
        }
      ]
    },
    "numbers": {
      "patterns": [
        {
          "name": "constant.numeric.gcel",
          "match": "\\b\\d+(\\.\\d+)?\\b"
        }
      ]
    },
    "functions": {
      "patterns": [
        {
          "name": "entity.name.function.gcel",
          "match": "\\b([a-z_][a-zA-Z0-9_]*)(?=\\()"
        }
      ]
    }
  }
}
```

### 4. Code Snippets

```json
// gcel.json snippets
{
  "CSRD Chain": {
    "prefix": "csrd-chain",
    "body": [
      "# CSRD Reporting Chain",
      "chain = (",
      "    DataIntakeAgent(source=\"${1:erp}\") >>",
      "    ValidationAgent(schema=\"csrd\") >>",
      "    EmissionsCalculatorAgent(scope=\"${2:all}\") >>",
      "    CSRDReportingAgent(framework=\"esrs\") >>",
      "    OutputAgent(format=\"${3:pdf}\")",
      ")",
      "",
      "result = await chain.run(${4:data})",
      "$0"
    ],
    "description": "Create a complete CSRD reporting chain"
  },
  "CBAM Chain": {
    "prefix": "cbam-chain",
    "body": [
      "# CBAM Compliance Chain",
      "chain = (",
      "    ImportDataAgent(source=\"${1:customs}\") >>",
      "    ProductCategoryAgent() >>",
      "    EmbeddedEmissionsAgent(method=\"${2:tier1}\") >>",
      "    CBAMReportingAgent(quarter=\"${3:Q1-2024}\") >>",
      "    ValidationAgent(rules=\"cbam\") >>",
      "    SubmissionAgent(authority=\"${4:eu}\")",
      ")",
      "",
      "result = await chain.run(${5:import_data})",
      "$0"
    ],
    "description": "Create a CBAM compliance chain"
  },
  "Parallel Processing": {
    "prefix": "parallel",
    "body": [
      "result = await parallel(",
      "    ${1:agent1} >> ${2:agent2},",
      "    ${3:agent3} >> ${4:agent4}",
      ").aggregate(${5:sum})",
      "$0"
    ],
    "description": "Parallel agent execution"
  },
  "Conditional Route": {
    "prefix": "conditional",
    "body": [
      "result = await conditional({",
      "    '${1:condition1}': ${2:agent1},",
      "    '${3:condition2}': ${4:agent2},",
      "    'default': ${5:agent3}",
      "})",
      "$0"
    ],
    "description": "Conditional routing"
  },
  "Custom Agent": {
    "prefix": "agent",
    "body": [
      "@agent(name=\"${1:AgentName}\")",
      "class ${1:AgentName}:",
      "    \"\"\"${2:Agent description}\"\"\"",
      "",
      "    def __init__(self, ${3:params}):",
      "        self.${3:params} = ${3:params}",
      "",
      "    async def process(self, input_data):",
      "        # Agent logic here",
      "        ${4:pass}",
      "        return result",
      "$0"
    ],
    "description": "Create a custom agent"
  }
}
```

### 5. Inline Documentation Provider

```typescript
// Hover Provider Implementation
export class GCELHoverProvider implements vscode.HoverProvider {
  private cache: Map<string, AgentInfo> = new Map();

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    token: vscode.CancellationToken
  ): Promise<vscode.Hover | null> {
    const wordRange = document.getWordRangeAtPosition(position);
    if (!wordRange) return null;

    const word = document.getText(wordRange);

    // Check if it's an agent
    const agentInfo = await this.getAgentInfo(word);
    if (agentInfo) {
      const markdown = new vscode.MarkdownString();
      markdown.isTrusted = true;
      markdown.supportHtml = true;

      // Agent name and type
      markdown.appendMarkdown(`### ${agentInfo.name}\n\n`);
      markdown.appendMarkdown(`**Type:** ${agentInfo.type}  \n`);
      markdown.appendMarkdown(`**Category:** ${agentInfo.category}  \n\n`);

      // Description
      markdown.appendMarkdown(`${agentInfo.description}\n\n`);

      // Parameters
      if (agentInfo.parameters.length > 0) {
        markdown.appendMarkdown(`**Parameters:**\n\n`);
        for (const param of agentInfo.parameters) {
          const required = param.required ? '*(required)*' : '*(optional)*';
          markdown.appendMarkdown(
            `- \`${param.name}\` (${param.type}) ${required}: ${param.description}\n`
          );
        }
        markdown.appendMarkdown('\n');
      }

      // Example
      if (agentInfo.example) {
        markdown.appendMarkdown('**Example:**\n\n');
        markdown.appendCodeblock(agentInfo.example, 'gcel');
      }

      // Links
      markdown.appendMarkdown('\n---\n\n');
      markdown.appendMarkdown(
        `[📖 Documentation](${agentInfo.docsUrl}) | `
      );
      markdown.appendMarkdown(
        `[🔗 View in Hub](${agentInfo.hubUrl})`
      );

      return new vscode.Hover(markdown, wordRange);
    }

    // Check for emission factors
    const emissionFactor = await this.getEmissionFactor(word);
    if (emissionFactor) {
      const markdown = new vscode.MarkdownString();
      markdown.appendMarkdown(`### Emission Factor\n\n`);
      markdown.appendMarkdown(`**Value:** ${emissionFactor.value} ${emissionFactor.unit}\n\n`);
      markdown.appendMarkdown(`**Source:** ${emissionFactor.source}\n\n`);
      markdown.appendMarkdown(`**Region:** ${emissionFactor.region}\n\n`);
      markdown.appendMarkdown(`**Last Updated:** ${emissionFactor.lastUpdated}\n\n`);
      return new vscode.Hover(markdown, wordRange);
    }

    return null;
  }

  private async getAgentInfo(agentName: string): Promise<AgentInfo | null> {
    // Check cache
    if (this.cache.has(agentName)) {
      return this.cache.get(agentName)!;
    }

    try {
      // Fetch from API
      const response = await axios.get(
        `${API_BASE_URL}/agents/${agentName}/info`
      );
      const agentInfo = response.data;

      // Cache result
      this.cache.set(agentName, agentInfo);

      return agentInfo;
    } catch (error) {
      return null;
    }
  }

  private async getEmissionFactor(name: string): Promise<EmissionFactor | null> {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/emission-factors/search?q=${name}`
      );
      return response.data.items[0] || null;
    } catch (error) {
      return null;
    }
  }
}
```

### 6. Debugging Support

```typescript
// Debug Adapter Implementation
export class GCELDebugAdapter implements vscode.DebugAdapter {
  private session: DebugSession;
  private breakpoints: Map<string, vscode.SourceBreakpoint[]> = new Map();

  async launchRequest(
    response: DebugProtocol.LaunchResponse,
    args: LaunchRequestArguments
  ): Promise<void> {
    // Start debug session
    this.session = await this.startDebugSession(args);

    // Send initialized event
    this.sendEvent(new InitializedEvent());
    this.sendResponse(response);
  }

  async setBreakPointsRequest(
    response: DebugProtocol.SetBreakpointsResponse,
    args: DebugProtocol.SetBreakpointsArguments
  ): Promise<void> {
    const path = args.source.path!;
    const breakpoints: vscode.Breakpoint[] = [];

    for (const bp of args.breakpoints || []) {
      const verified = await this.session.setBreakpoint(path, bp.line);
      breakpoints.push({
        verified,
        line: bp.line,
        id: this.generateBreakpointId(),
      });
    }

    this.breakpoints.set(path, args.breakpoints || []);

    response.body = {
      breakpoints,
    };
    this.sendResponse(response);
  }

  async continueRequest(
    response: DebugProtocol.ContinueResponse,
    args: DebugProtocol.ContinueArguments
  ): Promise<void> {
    await this.session.continue();
    this.sendResponse(response);
  }

  async nextRequest(
    response: DebugProtocol.NextResponse,
    args: DebugProtocol.NextArguments
  ): Promise<void> {
    await this.session.stepOver();
    this.sendResponse(response);
  }

  async stepInRequest(
    response: DebugProtocol.StepInResponse,
    args: DebugProtocol.StepInArguments
  ): Promise<void> {
    await this.session.stepInto();
    this.sendResponse(response);
  }

  async stepOutRequest(
    response: DebugProtocol.StepOutResponse,
    args: DebugProtocol.StepOutArguments
  ): Promise<void> {
    await this.session.stepOut();
    this.sendResponse(response);
  }

  async stackTraceRequest(
    response: DebugProtocol.StackTraceResponse,
    args: DebugProtocol.StackTraceArguments
  ): Promise<void> {
    const stackFrames = await this.session.getStackTrace();

    response.body = {
      stackFrames: stackFrames.map((frame, index) => ({
        id: index,
        name: frame.name,
        source: {
          path: frame.source,
        },
        line: frame.line,
        column: frame.column,
      })),
      totalFrames: stackFrames.length,
    };

    this.sendResponse(response);
  }

  async scopesRequest(
    response: DebugProtocol.ScopesResponse,
    args: DebugProtocol.ScopesArguments
  ): Promise<void> {
    const scopes = await this.session.getScopes(args.frameId);

    response.body = {
      scopes: scopes.map((scope) => ({
        name: scope.name,
        variablesReference: scope.variablesReference,
        expensive: scope.expensive || false,
      })),
    };

    this.sendResponse(response);
  }

  async variablesRequest(
    response: DebugProtocol.VariablesResponse,
    args: DebugProtocol.VariablesArguments
  ): Promise<void> {
    const variables = await this.session.getVariables(args.variablesReference);

    response.body = {
      variables: variables.map((v) => ({
        name: v.name,
        value: v.value,
        type: v.type,
        variablesReference: v.variablesReference || 0,
      })),
    };

    this.sendResponse(response);
  }

  async evaluateRequest(
    response: DebugProtocol.EvaluateResponse,
    args: DebugProtocol.EvaluateArguments
  ): Promise<void> {
    try {
      const result = await this.session.evaluate(
        args.expression,
        args.frameId
      );

      response.body = {
        result: result.value,
        type: result.type,
        variablesReference: result.variablesReference || 0,
      };
    } catch (error: any) {
      response.body = {
        result: `Error: ${error.message}`,
        type: 'error',
        variablesReference: 0,
      };
    }

    this.sendResponse(response);
  }

  private async startDebugSession(
    args: LaunchRequestArguments
  ): Promise<DebugSession> {
    // Connect to GreenLang debug server
    const session = new DebugSession({
      host: args.debugServer || 'localhost',
      port: args.debugPort || 5678,
    });

    await session.connect();

    // Set up event handlers
    session.on('stopped', (event) => {
      this.sendEvent(new StoppedEvent(event.reason, event.threadId));
    });

    session.on('output', (event) => {
      this.sendEvent(
        new OutputEvent(event.output, event.category)
      );
    });

    return session;
  }
}
```

### 7. Custom Views & Panels

```typescript
// Agent Explorer View
export class AgentExplorerProvider
  implements vscode.TreeDataProvider<AgentTreeItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    AgentTreeItem | undefined | null | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(private context: vscode.ExtensionContext) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: AgentTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: AgentTreeItem): Promise<AgentTreeItem[]> {
    if (!element) {
      // Root level - show categories
      return [
        new AgentTreeItem('Installed', 'category', [], {
          command: 'greenlang.showInstalledAgents',
        }),
        new AgentTreeItem('Hub', 'category', [], {
          command: 'greenlang.browseHub',
        }),
        new AgentTreeItem('Custom', 'category', [], {
          command: 'greenlang.showCustomAgents',
        }),
      ];
    }

    // Child level - show agents in category
    const agents = await this.getAgentsForCategory(element.label);
    return agents.map(
      (agent) =>
        new AgentTreeItem(agent.name, 'agent', [], {
          command: 'greenlang.openAgent',
          arguments: [agent.id],
        })
    );
  }

  private async getAgentsForCategory(
    category: string
  ): Promise<AgentSummary[]> {
    // Fetch agents from API or local cache
    const response = await axios.get(
      `${API_BASE_URL}/agents?category=${category}`
    );
    return response.data.items;
  }
}

// Trace Viewer Panel
export class TraceViewerPanel {
  public static currentPanel: TraceViewerPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private _disposables: vscode.Disposable[] = [];

  public static createOrShow(
    extensionUri: vscode.Uri,
    traceId: string
  ): void {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    if (TraceViewerPanel.currentPanel) {
      TraceViewerPanel.currentPanel._panel.reveal(column);
      TraceViewerPanel.currentPanel.loadTrace(traceId);
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'greenlangTraceViewer',
      'Trace Viewer',
      column || vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.joinPath(extensionUri, 'media'),
        ],
      }
    );

    TraceViewerPanel.currentPanel = new TraceViewerPanel(
      panel,
      extensionUri,
      traceId
    );
  }

  private constructor(
    panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri,
    traceId: string
  ) {
    this._panel = panel;

    // Set HTML content
    this._update(traceId);

    // Handle messages from the webview
    this._panel.webview.onDidReceiveMessage(
      (message) => {
        switch (message.command) {
          case 'export':
            this.exportTrace(message.traceId);
            break;
          case 'share':
            this.shareTrace(message.traceId);
            break;
        }
      },
      null,
      this._disposables
    );

    // Clean up
    this._panel.onDidDispose(
      () => this.dispose(),
      null,
      this._disposables
    );
  }

  private async _update(traceId: string): Promise<void> {
    const webview = this._panel.webview;
    this._panel.title = `Trace: ${traceId.substring(0, 8)}`;
    this._panel.webview.html = await this._getHtmlForWebview(
      webview,
      traceId
    );
  }

  private async _getHtmlForWebview(
    webview: vscode.Webview,
    traceId: string
  ): Promise<string> {
    // Fetch trace data
    const trace = await fetchTraceData(traceId);

    // Generate HTML with trace visualization
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trace Viewer</title>
        <style>
          body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
          }
          /* Add more styles */
        </style>
      </head>
      <body>
        <h1>Trace Visualization</h1>
        <div id="trace-viewer"></div>
        <script>
          const vscode = acquireVsCodeApi();
          const trace = ${JSON.stringify(trace)};

          // Render trace visualization
          renderTrace(trace);

          function renderTrace(data) {
            // Visualization code
          }
        </script>
      </body>
      </html>
    `;
  }

  private async loadTrace(traceId: string): Promise<void> {
    await this._update(traceId);
  }

  private async exportTrace(traceId: string): Promise<void> {
    const trace = await fetchTraceData(traceId);
    const uri = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(`trace-${traceId}.json`),
      filters: {
        JSON: ['json'],
      },
    });

    if (uri) {
      await vscode.workspace.fs.writeFile(
        uri,
        Buffer.from(JSON.stringify(trace, null, 2))
      );
      vscode.window.showInformationMessage(
        `Trace exported to ${uri.fsPath}`
      );
    }
  }

  private async shareTrace(traceId: string): Promise<void> {
    // Generate shareable link
    const link = await generateShareLink(traceId);
    await vscode.env.clipboard.writeText(link);
    vscode.window.showInformationMessage(
      'Share link copied to clipboard!'
    );
  }

  public dispose(): void {
    TraceViewerPanel.currentPanel = undefined;

    this._panel.dispose();

    while (this._disposables.length) {
      const disposable = this._disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }
}
```

### 8. Commands Implementation

```typescript
// Extension Commands
export function registerCommands(context: vscode.ExtensionContext): void {
  // Initialize GreenLang Project
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'greenlang.initProject',
      async () => {
        const projectName = await vscode.window.showInputBox({
          prompt: 'Enter project name',
          placeHolder: 'my-climate-project',
        });

        if (projectName) {
          await initializeProject(projectName);
          vscode.window.showInformationMessage(
            `GreenLang project '${projectName}' initialized!`
          );
        }
      }
    )
  );

  // Deploy Agent
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'greenlang.deployAgent',
      async (agentPath: string) => {
        const environment = await vscode.window.showQuickPick(
          ['development', 'staging', 'production'],
          {
            placeHolder: 'Select deployment environment',
          }
        );

        if (environment) {
          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Deploying agent...',
              cancellable: false,
            },
            async (progress) => {
              progress.report({ increment: 0 });

              const result = await deployAgent(agentPath, environment);

              progress.report({ increment: 100 });

              vscode.window.showInformationMessage(
                `Agent deployed successfully! URL: ${result.url}`
              );
            }
          );
        }
      }
    )
  );

  // Run Tests
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'greenlang.runTests',
      async () => {
        const terminal = vscode.window.createTerminal('GreenLang Tests');
        terminal.show();
        terminal.sendText('greenlang test');
      }
    )
  );

  // Generate Agent
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'greenlang.generateAgent',
      async () => {
        const agentType = await vscode.window.showQuickPick(
          [
            'Data Intake',
            'Calculation',
            'Validation',
            'Reporting',
            'Integration',
          ],
          {
            placeHolder: 'Select agent type',
          }
        );

        if (agentType) {
          const agentName = await vscode.window.showInputBox({
            prompt: 'Enter agent name',
            placeHolder: 'MyCustomAgent',
          });

          if (agentName) {
            await generateAgentScaffold(agentType, agentName);
            vscode.window.showInformationMessage(
              `Agent '${agentName}' generated!`
            );
          }
        }
      }
    )
  );

  // View Trace
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'greenlang.viewTrace',
      async (traceId?: string) => {
        if (!traceId) {
          traceId = await vscode.window.showInputBox({
            prompt: 'Enter trace ID',
            placeHolder: 'abc123def456',
          });
        }

        if (traceId) {
          TraceViewerPanel.createOrShow(context.extensionUri, traceId);
        }
      }
    )
  );

  // Browse Hub
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'greenlang.browseHub',
      async () => {
        const agents = await fetchHubAgents();
        const selected = await vscode.window.showQuickPick(
          agents.map((a) => ({
            label: a.name,
            description: a.description,
            detail: `${a.downloads} downloads`,
            agent: a,
          })),
          {
            placeHolder: 'Search GreenLang Hub',
          }
        );

        if (selected) {
          await installAgent(selected.agent.id);
          vscode.window.showInformationMessage(
            `Agent '${selected.label}' installed!`
          );
        }
      }
    )
  );
}
```

### 9. JetBrains Plugin (IntelliJ/PyCharm)

```kotlin
// Plugin.kt - JetBrains Plugin Entry Point
class GreenLangPlugin : DumbAware, ApplicationComponent {
    override fun initComponent() {
        // Initialize plugin components
        EditorFactory.getInstance().addEditorFactoryListener(
            GreenLangEditorListener(),
            ApplicationManager.getApplication()
        )
    }

    override fun getComponentName(): String = "GreenLang.Plugin"
}

// GreenLangFileType.kt
class GreenLangFileType private constructor() : LanguageFileType(GreenLangLanguage.INSTANCE) {
    override fun getName(): String = "GreenLang"
    override fun getDescription(): String = "GreenLang Climate Expression Language"
    override fun getDefaultExtension(): String = "gcel"
    override fun getIcon(): Icon = GreenLangIcons.FILE

    companion object {
        val INSTANCE = GreenLangFileType()
    }
}

// GreenLangSyntaxHighlighter.kt
class GreenLangSyntaxHighlighter : SyntaxHighlighterBase() {
    override fun getHighlightingLexer(): Lexer = GreenLangLexerAdapter()

    override fun getTokenHighlights(tokenType: IElementType): Array<TextAttributesKey> {
        return when (tokenType) {
            GreenLangTypes.KEYWORD -> KEYWORD_KEYS
            GreenLangTypes.OPERATOR -> OPERATOR_KEYS
            GreenLangTypes.AGENT -> AGENT_KEYS
            GreenLangTypes.STRING -> STRING_KEYS
            GreenLangTypes.NUMBER -> NUMBER_KEYS
            GreenLangTypes.COMMENT -> COMMENT_KEYS
            else -> EMPTY_KEYS
        }
    }

    companion object {
        val KEYWORD = TextAttributesKey.createTextAttributesKey(
            "GCEL_KEYWORD",
            DefaultLanguageHighlighterColors.KEYWORD
        )

        val OPERATOR = TextAttributesKey.createTextAttributesKey(
            "GCEL_OPERATOR",
            DefaultLanguageHighlighterColors.OPERATION_SIGN
        )

        val AGENT = TextAttributesKey.createTextAttributesKey(
            "GCEL_AGENT",
            DefaultLanguageHighlighterColors.CLASS_NAME
        )

        private val KEYWORD_KEYS = arrayOf(KEYWORD)
        private val OPERATOR_KEYS = arrayOf(OPERATOR)
        private val AGENT_KEYS = arrayOf(AGENT)
        private val STRING_KEYS = arrayOf(DefaultLanguageHighlighterColors.STRING)
        private val NUMBER_KEYS = arrayOf(DefaultLanguageHighlighterColors.NUMBER)
        private val COMMENT_KEYS = arrayOf(DefaultLanguageHighlighterColors.LINE_COMMENT)
        private val EMPTY_KEYS = emptyArray<TextAttributesKey>()
    }
}

// GreenLangCompletionContributor.kt
class GreenLangCompletionContributor : CompletionContributor() {
    init {
        extend(
            CompletionType.BASIC,
            PlatformPatterns.psiElement(),
            AgentCompletionProvider()
        )
    }
}

class AgentCompletionProvider : CompletionProvider<CompletionParameters>() {
    override fun addCompletions(
        parameters: CompletionParameters,
        context: ProcessingContext,
        result: CompletionResultSet
    ) {
        // Fetch available agents
        val agents = GreenLangService.getInstance().getAvailableAgents()

        agents.forEach { agent ->
            val lookup = LookupElementBuilder
                .create(agent.name)
                .withIcon(GreenLangIcons.AGENT)
                .withTypeText(agent.type)
                .withTailText(" (${agent.category})")
                .withInsertHandler { context, item ->
                    // Auto-insert parameters
                    val document = context.document
                    val offset = context.tailOffset
                    document.insertString(offset, "(${agent.parametersTemplate})")
                    context.editor.caretModel.moveToOffset(offset + 1)
                }

            result.addElement(lookup)
        }
    }
}
```

### 10. Performance & Distribution

```yaml
Performance:
  - Extension bundle size: <5MB
  - Activation time: <500ms
  - Completion latency: <100ms
  - Syntax highlighting: Real-time
  - Memory footprint: <50MB

Distribution:
  VSCode:
    - Marketplace: Visual Studio Marketplace
    - Auto-updates: Built-in
    - Telemetry: Optional

  JetBrains:
    - Marketplace: JetBrains Plugin Repository
    - Compatible: IntelliJ IDEA, PyCharm, WebStorm
    - Versions: 2023.1+

Installation:
  VSCode:
    - Command: ext install greenlang.greenlang
    - Or search "GreenLang" in Extensions

  JetBrains:
    - Settings → Plugins → Search "GreenLang"
```

### 11. Timeline & Milestones

```yaml
Q3 2026: VSCode MVP
  Month 1:
    - Language server
    - Syntax highlighting
    - Basic completion
    - Documentation hover

  Month 2:
    - Debugging support
    - Agent explorer
    - Code snippets
    - Commands

  Month 3:
    - Trace viewer
    - Test runner
    - Marketplace publish
    - Beta testing

Q4 2026: JetBrains Plugin
  Month 1:
    - Plugin framework
    - Syntax support
    - Completion provider

  Month 2:
    - Debugging integration
    - Tool windows
    - Beta release

  Month 3:
    - Polish & testing
    - Documentation
    - Public release

2027: Advanced Features
  - AI-powered code suggestions
  - Performance profiling
  - Team collaboration
  - 100K+ installs target
```