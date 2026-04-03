import { Component, type ErrorInfo, type ReactNode } from "react";
import Alert from "@mui/material/Alert";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    void error;
    void errorInfo;
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return <Alert severity="error">Shell crashed. Reload to recover.</Alert>;
    }
    return this.props.children;
  }
}
