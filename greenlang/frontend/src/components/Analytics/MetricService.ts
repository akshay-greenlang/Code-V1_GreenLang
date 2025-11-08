/**
 * WebSocket client for real-time metric streaming.
 *
 * Provides reconnection logic, metric buffering, transformation,
 * and alert evaluation for dashboard components.
 */

import msgpack from 'msgpack-lite';

export interface MetricSubscription {
  channels: string[];
  tags?: Record<string, string>;
  aggregation_interval?: string;
  compression?: boolean;
}

export interface Metric {
  name: string;
  type: 'counter' | 'gauge' | 'histogram';
  value: number | number[];
  timestamp: string;
  tags?: Record<string, string>;
  channel?: string;
  [key: string]: any;
}

export interface HistoricalRequest {
  channel: string;
  start_time: string;
  end_time?: string;
}

export type MetricTransform = 'rate' | 'derivative' | 'moving_average';

interface MetricCallback {
  (metric: Metric): void;
}

interface ErrorCallback {
  (error: Error): void;
}

interface ConnectionCallback {
  (connected: boolean): void;
}

export class MetricService {
  private ws: WebSocket | null = null;
  private url: string;
  private token: string | null;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectDelay: number = 1000; // Start with 1 second
  private maxReconnectDelay: number = 30000; // Max 30 seconds
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private heartbeatInterval: number = 30000; // 30 seconds

  private metricCallbacks: Map<string, Set<MetricCallback>> = new Map();
  private errorCallbacks: Set<ErrorCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();

  private buffer: Metric[] = [];
  private maxBufferSize: number = 1000;
  private offline: boolean = false;

  private subscriptions: Set<string> = new Set();
  private clientId: string | null = null;

  constructor(url: string, token: string | null = null) {
    this.url = url;
    this.token = token;
  }

  /**
   * Connect to WebSocket server.
   */
  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    try {
      // Add token to URL if available
      const wsUrl = this.token ? `${this.url}?token=${this.token}` : this.url;
      this.ws = new WebSocket(wsUrl);

      // Set binary type for MessagePack
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
    } catch (error) {
      this.handleError(error as Event);
    }
  }

  /**
   * Disconnect from WebSocket server.
   */
  public disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.offline = false;
    this.notifyConnectionCallbacks(false);
  }

  /**
   * Subscribe to metric channels.
   */
  public subscribe(subscription: MetricSubscription): void {
    subscription.channels.forEach(channel => {
      this.subscriptions.add(channel);
    });

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: 'subscribe',
        data: subscription
      });
    }
  }

  /**
   * Unsubscribe from metric channels.
   */
  public unsubscribe(channels: string[]): void {
    channels.forEach(channel => {
      this.subscriptions.delete(channel);
    });

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: 'unsubscribe',
        channels
      });
    }
  }

  /**
   * Request historical metrics.
   */
  public getHistory(request: HistoricalRequest): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: 'get_history',
        data: request
      });
    }
  }

  /**
   * Add metric callback for specific channel.
   */
  public onMetric(channel: string, callback: MetricCallback): void {
    if (!this.metricCallbacks.has(channel)) {
      this.metricCallbacks.set(channel, new Set());
    }
    this.metricCallbacks.get(channel)!.add(callback);
  }

  /**
   * Remove metric callback.
   */
  public offMetric(channel: string, callback: MetricCallback): void {
    const callbacks = this.metricCallbacks.get(channel);
    if (callbacks) {
      callbacks.delete(callback);
      if (callbacks.size === 0) {
        this.metricCallbacks.delete(channel);
      }
    }
  }

  /**
   * Add error callback.
   */
  public onError(callback: ErrorCallback): void {
    this.errorCallbacks.add(callback);
  }

  /**
   * Remove error callback.
   */
  public offError(callback: ErrorCallback): void {
    this.errorCallbacks.delete(callback);
  }

  /**
   * Add connection status callback.
   */
  public onConnectionChange(callback: ConnectionCallback): void {
    this.connectionCallbacks.add(callback);
  }

  /**
   * Remove connection status callback.
   */
  public offConnectionChange(callback: ConnectionCallback): void {
    this.connectionCallbacks.delete(callback);
  }

  /**
   * Transform metric using specified transformation.
   */
  public transformMetric(
    metrics: Metric[],
    transform: MetricTransform,
    windowSize: number = 10
  ): Metric[] {
    switch (transform) {
      case 'rate':
        return this.calculateRate(metrics);
      case 'derivative':
        return this.calculateDerivative(metrics);
      case 'moving_average':
        return this.calculateMovingAverage(metrics, windowSize);
      default:
        return metrics;
    }
  }

  /**
   * Get buffered metrics (for offline mode).
   */
  public getBufferedMetrics(): Metric[] {
    return [...this.buffer];
  }

  /**
   * Clear metric buffer.
   */
  public clearBuffer(): void {
    this.buffer = [];
  }

  /**
   * Check if connected.
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get client ID.
   */
  public getClientId(): string | null {
    return this.clientId;
  }

  // Private methods

  private handleOpen(event: Event): void {
    console.log('WebSocket connected');
    this.offline = false;
    this.reconnectAttempts = 0;
    this.reconnectDelay = 1000;

    this.notifyConnectionCallbacks(true);
    this.startHeartbeat();

    // Resubscribe to channels
    if (this.subscriptions.size > 0) {
      this.subscribe({
        channels: Array.from(this.subscriptions),
        compression: true
      });
    }

    // Flush buffered metrics
    this.flushBuffer();
  }

  private handleMessage(event: MessageEvent): void {
    try {
      let data: any;

      // Check if binary (MessagePack) or text (JSON)
      if (event.data instanceof ArrayBuffer) {
        data = msgpack.decode(new Uint8Array(event.data));
      } else {
        data = JSON.parse(event.data);
      }

      const messageType = data.type;

      switch (messageType) {
        case 'welcome':
          this.clientId = data.client_id;
          console.log(`WebSocket client ID: ${this.clientId}`);
          break;

        case 'subscribed':
          console.log(`Subscribed to channels: ${data.channels.join(', ')}`);
          break;

        case 'unsubscribed':
          console.log(`Unsubscribed from channels: ${data.channels.join(', ')}`);
          break;

        case 'ping':
          // Respond to heartbeat
          this.send({ type: 'pong', timestamp: Date.now() / 1000 });
          break;

        case 'error':
          this.notifyErrorCallbacks(new Error(data.message));
          break;

        case 'history':
          // Handle historical metrics
          this.notifyMetricCallbacks(data.channel, data.metrics);
          break;

        default:
          // Assume it's a metric
          if (data.channel) {
            this.handleMetric(data);
          }
          break;
      }
    } catch (error) {
      console.error('Error processing message:', error);
      this.notifyErrorCallbacks(error as Error);
    }
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.notifyErrorCallbacks(new Error('WebSocket connection error'));
  }

  private handleClose(event: CloseEvent): void {
    console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
    this.offline = true;
    this.notifyConnectionCallbacks(false);

    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    // Attempt reconnection with exponential backoff
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = Math.min(
        this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
        this.maxReconnectDelay
      );

      console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);

      this.reconnectTimer = setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
      this.notifyErrorCallbacks(new Error('Failed to reconnect to WebSocket server'));
    }
  }

  private handleMetric(metric: Metric): void {
    // Buffer metric if offline
    if (this.offline) {
      this.bufferMetric(metric);
      return;
    }

    // Interpolate missing data if needed
    const interpolated = this.interpolateMetric(metric);

    // Notify callbacks
    this.notifyMetricCallbacks(metric.channel || '*', [interpolated]);
  }

  private send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch (error) {
        console.error('Error sending message:', error);
        this.notifyErrorCallbacks(error as Error);
      }
    }
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'pong', timestamp: Date.now() / 1000 });
      }
    }, this.heartbeatInterval) as any;
  }

  private bufferMetric(metric: Metric): void {
    this.buffer.push(metric);

    // Limit buffer size
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer.shift();
    }
  }

  private flushBuffer(): void {
    if (this.buffer.length === 0) {
      return;
    }

    console.log(`Flushing ${this.buffer.length} buffered metrics`);

    for (const metric of this.buffer) {
      this.handleMetric(metric);
    }

    this.buffer = [];
  }

  private interpolateMetric(metric: Metric): Metric {
    // Implement metric interpolation for missing data points
    // For now, just return the metric as-is
    return metric;
  }

  private calculateRate(metrics: Metric[]): Metric[] {
    const result: Metric[] = [];

    for (let i = 1; i < metrics.length; i++) {
      const prev = metrics[i - 1];
      const curr = metrics[i];

      if (typeof prev.value === 'number' && typeof curr.value === 'number') {
        const timeDiff = (new Date(curr.timestamp).getTime() - new Date(prev.timestamp).getTime()) / 1000;
        const valueDiff = curr.value - prev.value;

        result.push({
          ...curr,
          value: timeDiff > 0 ? valueDiff / timeDiff : 0
        });
      }
    }

    return result;
  }

  private calculateDerivative(metrics: Metric[]): Metric[] {
    const result: Metric[] = [];

    for (let i = 1; i < metrics.length; i++) {
      const prev = metrics[i - 1];
      const curr = metrics[i];

      if (typeof prev.value === 'number' && typeof curr.value === 'number') {
        result.push({
          ...curr,
          value: curr.value - prev.value
        });
      }
    }

    return result;
  }

  private calculateMovingAverage(metrics: Metric[], windowSize: number): Metric[] {
    const result: Metric[] = [];

    for (let i = 0; i < metrics.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = metrics.slice(start, i + 1);

      const sum = window.reduce((acc, m) => {
        return acc + (typeof m.value === 'number' ? m.value : 0);
      }, 0);

      result.push({
        ...metrics[i],
        value: sum / window.length
      });
    }

    return result;
  }

  private notifyMetricCallbacks(channel: string, metrics: Metric[]): void {
    const callbacks = this.metricCallbacks.get(channel);
    if (callbacks) {
      metrics.forEach(metric => {
        callbacks.forEach(callback => {
          try {
            callback(metric);
          } catch (error) {
            console.error('Error in metric callback:', error);
          }
        });
      });
    }

    // Also notify wildcard callbacks
    const wildcardCallbacks = this.metricCallbacks.get('*');
    if (wildcardCallbacks) {
      metrics.forEach(metric => {
        wildcardCallbacks.forEach(callback => {
          try {
            callback(metric);
          } catch (error) {
            console.error('Error in metric callback:', error);
          }
        });
      });
    }
  }

  private notifyErrorCallbacks(error: Error): void {
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (err) {
        console.error('Error in error callback:', err);
      }
    });
  }

  private notifyConnectionCallbacks(connected: boolean): void {
    this.connectionCallbacks.forEach(callback => {
      try {
        callback(connected);
      } catch (error) {
        console.error('Error in connection callback:', error);
      }
    });
  }
}

// Singleton instance
let metricServiceInstance: MetricService | null = null;

export function getMetricService(url?: string, token?: string): MetricService {
  if (!metricServiceInstance && url) {
    metricServiceInstance = new MetricService(url, token);
  }

  if (!metricServiceInstance) {
    throw new Error('MetricService not initialized. Provide URL on first call.');
  }

  return metricServiceInstance;
}

export function resetMetricService(): void {
  if (metricServiceInstance) {
    metricServiceInstance.disconnect();
    metricServiceInstance = null;
  }
}
