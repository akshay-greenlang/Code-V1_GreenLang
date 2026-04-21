/**
 * Async pagination helpers, parity with the Python sync/async paginators.
 *
 * Both `OffsetPaginator` and `CursorPaginator` implement the async
 * iterator protocol so callers can simply `for await ... of` them.
 */

export interface PageInfo {
  page: number;
  pageSize: number;
  totalCount: number | null;
  nextCursor: string | null;
  itemsYielded: number;
}

export type OffsetFetcher<T> = (
  offset: number,
  limit: number,
) => Promise<{ items: T[]; totalCount: number | null }>;

export type CursorFetcher<T> = (
  cursor: string | null,
) => Promise<{ items: T[]; nextCursor: string | null }>;

/**
 * Async offset/limit iterator. Fetcher receives `(offset, limit)` and
 * returns `{ items, totalCount }`. Iteration stops when:
 *   - `items` is empty, OR
 *   - offset + yielded >= totalCount (if known), OR
 *   - `maxItems` has been reached.
 */
export class OffsetPaginator<T> implements AsyncIterable<T>, AsyncIterator<T> {
  private readonly fetcher: OffsetFetcher<T>;
  private readonly pageSize: number;
  private readonly maxItems: number | undefined;
  private offset: number;
  private buffer: T[] = [];
  private totalCount: number | null = null;
  private exhausted = false;
  private yielded = 0;

  constructor(
    fetcher: OffsetFetcher<T>,
    opts: {
      pageSize?: number;
      startOffset?: number;
      maxItems?: number;
    } = {},
  ) {
    this.fetcher = fetcher;
    this.pageSize = Math.max(1, opts.pageSize ?? 100);
    this.offset = opts.startOffset ?? 0;
    this.maxItems = opts.maxItems;
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return this;
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.maxItems !== undefined && this.yielded >= this.maxItems) {
      return { done: true, value: undefined };
    }
    if (this.buffer.length === 0 && !this.exhausted) {
      await this.fill();
    }
    if (this.buffer.length === 0) {
      return { done: true, value: undefined };
    }
    const item = this.buffer.shift() as T;
    this.yielded += 1;
    return { done: false, value: item };
  }

  get total(): number | null {
    return this.totalCount;
  }

  /** Collect *all* remaining items into an array. */
  async toArray(): Promise<T[]> {
    const out: T[] = [];
    for (;;) {
      const r = await this.next();
      if (r.done) break;
      out.push(r.value);
    }
    return out;
  }

  private async fill(): Promise<void> {
    let limit = this.pageSize;
    if (this.maxItems !== undefined) {
      limit = Math.min(limit, this.maxItems - this.yielded);
      if (limit <= 0) {
        this.exhausted = true;
        return;
      }
    }
    const { items, totalCount } = await this.fetcher(this.offset, limit);
    this.buffer.push(...items);
    this.offset += items.length;
    if (totalCount !== null && totalCount !== undefined) {
      this.totalCount = totalCount;
    }
    if (
      items.length === 0 ||
      (this.totalCount !== null && this.offset >= this.totalCount)
    ) {
      this.exhausted = true;
    }
  }
}

/**
 * Async cursor iterator. Fetcher receives `cursor` (null on the first
 * call) and returns `{ items, nextCursor }`. Iteration stops when
 * `nextCursor` is null or `maxItems` is reached.
 */
export class CursorPaginator<T> implements AsyncIterable<T>, AsyncIterator<T> {
  private readonly fetcher: CursorFetcher<T>;
  private readonly maxItems: number | undefined;
  private buffer: T[] = [];
  private cursor: string | null = null;
  private started = false;
  private exhausted = false;
  private yielded = 0;

  constructor(fetcher: CursorFetcher<T>, opts: { maxItems?: number } = {}) {
    this.fetcher = fetcher;
    this.maxItems = opts.maxItems;
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return this;
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.maxItems !== undefined && this.yielded >= this.maxItems) {
      return { done: true, value: undefined };
    }
    if (this.buffer.length === 0 && !this.exhausted) {
      await this.fill();
    }
    if (this.buffer.length === 0) {
      return { done: true, value: undefined };
    }
    const item = this.buffer.shift() as T;
    this.yielded += 1;
    return { done: false, value: item };
  }

  get nextCursor(): string | null {
    return this.cursor;
  }

  async toArray(): Promise<T[]> {
    const out: T[] = [];
    for (;;) {
      const r = await this.next();
      if (r.done) break;
      out.push(r.value);
    }
    return out;
  }

  private async fill(): Promise<void> {
    if (this.started && this.cursor === null) {
      this.exhausted = true;
      return;
    }
    const { items, nextCursor } = await this.fetcher(this.cursor);
    this.started = true;
    this.buffer.push(...items);
    this.cursor = nextCursor;
    if (items.length === 0 || nextCursor === null) {
      this.exhausted = true;
    }
  }
}

/** Pull `key` out of a response payload regardless of wrapper shape. */
export function extractItems<T = unknown>(
  payload: unknown,
  key = 'factors',
): T[] {
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    const inner = (payload as Record<string, unknown>)[key];
    if (Array.isArray(inner)) return inner as T[];
  }
  if (Array.isArray(payload)) return payload as T[];
  return [];
}
