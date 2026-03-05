/**
 * GL-CDP-APP v1.0 - Application Entry Point
 *
 * Bootstraps the React application with:
 *   - React.StrictMode for development diagnostics
 *   - Redux Provider wiring the global store
 *   - BrowserRouter for client-side routing
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { store } from './store';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </Provider>
  </React.StrictMode>,
);
