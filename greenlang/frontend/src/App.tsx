/**
 * Sample App component demonstrating the Visual Workflow Builder
 */

import React from 'react';
import { WorkflowCanvas } from './components/WorkflowBuilder/WorkflowCanvas';
import { AgentPalette } from './components/WorkflowBuilder/AgentPalette';
import { DAGEditor } from './components/WorkflowBuilder/DAGEditor';
import './index.css';

function App() {
  return (
    <div className="flex h-screen bg-gray-100">
      {/* Agent Palette - Left Sidebar */}
      <AgentPalette />

      {/* Main Canvas - Center */}
      <div className="flex-1">
        <WorkflowCanvas />
      </div>

      {/* DAG Editor - Right Sidebar */}
      <DAGEditor />
    </div>
  );
}

export default App;
