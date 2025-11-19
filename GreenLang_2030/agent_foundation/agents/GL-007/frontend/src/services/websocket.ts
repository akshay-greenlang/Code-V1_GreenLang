/**
 * GL-007 Furnace Performance Monitor - WebSocket Service
 *
 * Real-time data streaming via WebSocket
 */

import { io, Socket } from 'socket.io-client';
import type {
  WebSocketMessage,
  WebSocketMessageType,
  FurnacePerformance,
  Alert,
  SensorReading,
} from '../types';

// ============================================================================
// WEBSOCKET EVENT TYPES
// ============================================================================

type EventHandler<T = any> = (data: T) => void;

interface WebSocketEvents {
  performance_update: EventHandler<FurnacePerformance>;
  alert: EventHandler<Alert>;
  sensor_reading: EventHandler<SensorReading>;
  status_change: EventHandler<{ furnaceId: string; status: string }>;
  maintenance_update: EventHandler<any>;
  configuration_change: EventHandler<any>;
  connected: EventHandler<void>;
  disconnected: EventHandler<void>;
  error: EventHandler<Error>;
}

// ============================================================================
// WEBSOCKET SERVICE CLASS
// ============================================================================

export class WebSocketService {
  private socket: Socket | null = null;
  private eventHandlers: Map<string, Set<EventHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;

  constructor(private url: string) {}

  /**
   * Connect to WebSocket server
   */
  connect(furnaceId: string, authToken?: string): void {
    if (this.socket?.connected || this.isConnecting) {
      console.warn('WebSocket already connected or connecting');
      return;
    }

    this.isConnecting = true;

    this.socket = io(this.url, {
      transports: ['websocket'],
      auth: {
        token: authToken,
        furnaceId,
      },
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
      reconnectionDelayMax: 5000,
    });

    this.setupSocketListeners();
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnecting = false;
      this.reconnectAttempts = 0;
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Subscribe to furnace updates
   */
  subscribe(furnaceId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', { furnaceId });
    }
  }

  /**
   * Unsubscribe from furnace updates
   */
  unsubscribe(furnaceId: string): void {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { furnaceId });
    }
  }

  /**
   * Register event handler
   */
  on<K extends keyof WebSocketEvents>(
    event: K,
    handler: WebSocketEvents[K]
  ): () => void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }

    const handlers = this.eventHandlers.get(event)!;
    handlers.add(handler);

    // Return unsubscribe function
    return () => {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.eventHandlers.delete(event);
      }
    };
  }

  /**
   * Remove event handler
   */
  off<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.eventHandlers.delete(event);
      }
    }
  }

  /**
   * Remove all event handlers for an event
   */
  removeAllListeners(event?: keyof WebSocketEvents): void {
    if (event) {
      this.eventHandlers.delete(event);
    } else {
      this.eventHandlers.clear();
    }
  }

  /**
   * Send message to server
   */
  send(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected. Message not sent:', event, data);
    }
  }

  /**
   * Request performance snapshot
   */
  requestPerformanceSnapshot(furnaceId: string): void {
    this.send('request_performance', { furnaceId });
  }

  /**
   * Request thermal profile
   */
  requestThermalProfile(furnaceId: string): void {
    this.send('request_thermal_profile', { furnaceId });
  }

  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId: string, userId: string): void {
    this.send('acknowledge_alert', { alertId, userId });
  }

  /**
   * Setup socket event listeners
   */
  private setupSocketListeners(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnecting = false;
      this.reconnectAttempts = 0;
      this.emit('connected', undefined);
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnecting = false;
      this.emit('disconnected', undefined);
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.isConnecting = false;
      this.reconnectAttempts++;
      this.emit('error', error);

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        this.disconnect();
      }
    });

    // Data events
    this.socket.on('message', (message: WebSocketMessage) => {
      this.handleMessage(message);
    });

    // Specific event types
    this.socket.on('performance_update', (data: FurnacePerformance) => {
      this.emit('performance_update', data);
    });

    this.socket.on('alert', (data: Alert) => {
      this.emit('alert', data);
    });

    this.socket.on('sensor_reading', (data: SensorReading) => {
      this.emit('sensor_reading', data);
    });

    this.socket.on('status_change', (data: any) => {
      this.emit('status_change', data);
    });

    this.socket.on('maintenance_update', (data: any) => {
      this.emit('maintenance_update', data);
    });

    this.socket.on('configuration_change', (data: any) => {
      this.emit('configuration_change', data);
    });

    // Ping/pong for keepalive
    this.socket.on('ping', () => {
      this.socket?.emit('pong');
    });
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(message: WebSocketMessage): void {
    const { type, data } = message;

    switch (type) {
      case 'performance_update':
        this.emit('performance_update', data);
        break;
      case 'alert':
        this.emit('alert', data);
        break;
      case 'sensor_reading':
        this.emit('sensor_reading', data);
        break;
      case 'status_change':
        this.emit('status_change', data);
        break;
      case 'maintenance_update':
        this.emit('maintenance_update', data);
        break;
      case 'configuration_change':
        this.emit('configuration_change', data);
        break;
      default:
        console.warn('Unknown message type:', type);
    }
  }

  /**
   * Emit event to registered handlers
   */
  private emit<K extends keyof WebSocketEvents>(
    event: K,
    data: Parameters<WebSocketEvents[K]>[0]
  ): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in ${event} handler:`, error);
        }
      });
    }
  }

  /**
   * Get connection status
   */
  getStatus(): {
    connected: boolean;
    reconnectAttempts: number;
    isConnecting: boolean;
  } {
    return {
      connected: this.socket?.connected || false,
      reconnectAttempts: this.reconnectAttempts,
      isConnecting: this.isConnecting,
    };
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const WS_URL = import.meta.env.VITE_WS_URL || 'wss://ws.greenlang.io';

export const websocketService = new WebSocketService(WS_URL);

export default websocketService;

// ============================================================================
// REACT HOOK
// ============================================================================

import { useEffect, useRef, useState } from 'react';

export interface UseWebSocketOptions {
  furnaceId: string;
  authToken?: string;
  autoConnect?: boolean;
  autoSubscribe?: boolean;
}

export function useWebSocket(options: UseWebSocketOptions) {
  const { furnaceId, authToken, autoConnect = true, autoSubscribe = true } = options;
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const serviceRef = useRef(websocketService);

  useEffect(() => {
    const service = serviceRef.current;

    if (autoConnect) {
      service.connect(furnaceId, authToken);
    }

    const unsubscribeConnected = service.on('connected', () => {
      setIsConnected(true);
      setError(null);
      if (autoSubscribe) {
        service.subscribe(furnaceId);
      }
    });

    const unsubscribeDisconnected = service.on('disconnected', () => {
      setIsConnected(false);
    });

    const unsubscribeError = service.on('error', (err) => {
      setError(err);
    });

    return () => {
      unsubscribeConnected();
      unsubscribeDisconnected();
      unsubscribeError();
      if (autoSubscribe) {
        service.unsubscribe(furnaceId);
      }
    };
  }, [furnaceId, authToken, autoConnect, autoSubscribe]);

  return {
    isConnected,
    error,
    service: serviceRef.current,
    subscribe: (id: string) => serviceRef.current.subscribe(id),
    unsubscribe: (id: string) => serviceRef.current.unsubscribe(id),
    on: serviceRef.current.on.bind(serviceRef.current),
    off: serviceRef.current.off.bind(serviceRef.current),
    send: serviceRef.current.send.bind(serviceRef.current),
  };
}

// ============================================================================
// SPECIALIZED HOOKS
// ============================================================================

export function usePerformanceUpdates(
  furnaceId: string,
  callback: (performance: FurnacePerformance) => void
) {
  const { service } = useWebSocket({ furnaceId });

  useEffect(() => {
    const unsubscribe = service.on('performance_update', callback);
    return unsubscribe;
  }, [service, callback]);
}

export function useAlertUpdates(furnaceId: string, callback: (alert: Alert) => void) {
  const { service } = useWebSocket({ furnaceId });

  useEffect(() => {
    const unsubscribe = service.on('alert', callback);
    return unsubscribe;
  }, [service, callback]);
}

export function useSensorReadings(
  furnaceId: string,
  callback: (reading: SensorReading) => void
) {
  const { service } = useWebSocket({ furnaceId });

  useEffect(() => {
    const unsubscribe = service.on('sensor_reading', callback);
    return unsubscribe;
  }, [service, callback]);
}
