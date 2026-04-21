import { CursorPaginator, OffsetPaginator, extractItems } from '../src';

describe('OffsetPaginator', () => {
  it('iterates across multiple pages and stops at totalCount', async () => {
    const pages = [
      { items: ['a', 'b'], totalCount: 5 },
      { items: ['c', 'd'], totalCount: 5 },
      { items: ['e'], totalCount: 5 },
    ];
    let idx = 0;
    const pager = new OffsetPaginator<string>(async () => pages[idx++], {
      pageSize: 2,
    });
    const out = await pager.toArray();
    expect(out).toEqual(['a', 'b', 'c', 'd', 'e']);
    expect(idx).toBe(3);
  });

  it('respects maxItems', async () => {
    const pager = new OffsetPaginator<number>(
      async (_, limit) => ({
        items: Array.from({ length: limit }, (_, i) => i),
        totalCount: 1000,
      }),
      { pageSize: 10, maxItems: 5 },
    );
    const out = await pager.toArray();
    expect(out).toEqual([0, 1, 2, 3, 4]);
  });
});

describe('CursorPaginator', () => {
  it('walks cursors until null', async () => {
    const pager = new CursorPaginator<string>(async (cursor) => {
      if (cursor === null) return { items: ['a', 'b'], nextCursor: 'c2' };
      if (cursor === 'c2') return { items: ['c'], nextCursor: null };
      return { items: [], nextCursor: null };
    });
    const out = await pager.toArray();
    expect(out).toEqual(['a', 'b', 'c']);
  });
});

describe('extractItems', () => {
  it('extracts from {factors:[...]}', () => {
    expect(extractItems({ factors: [1, 2] })).toEqual([1, 2]);
  });

  it('returns array payloads unchanged', () => {
    expect(extractItems([1, 2])).toEqual([1, 2]);
  });

  it('returns [] on other shapes', () => {
    expect(extractItems({})).toEqual([]);
    expect(extractItems(null)).toEqual([]);
    expect(extractItems(42)).toEqual([]);
  });
});
