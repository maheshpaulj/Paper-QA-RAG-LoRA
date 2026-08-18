// Standardized color coding for RAG chunks across Chat, PDF Viewer, and Inspector

export const CHUNK_COLORS = [
  {
    key: 'c0',
    name: 'Amber',
    text: 'var(--chunk-c0-text)',
    bg: 'var(--chunk-c0-bg)',
    border: 'var(--chunk-c0-border)',
    mark: 'var(--chunk-c0-mark)',
  },
  {
    key: 'c1',
    name: 'Teal',
    text: 'var(--chunk-c1-text)',
    bg: 'var(--chunk-c1-bg)',
    border: 'var(--chunk-c1-border)',
    mark: 'var(--chunk-c1-mark)',
  },
  {
    key: 'c2',
    name: 'Sky',
    text: 'var(--chunk-c2-text)',
    bg: 'var(--chunk-c2-bg)',
    border: 'var(--chunk-c2-border)',
    mark: 'var(--chunk-c2-mark)',
  },
  {
    key: 'c3',
    name: 'Rose',
    text: 'var(--chunk-c3-text)',
    bg: 'var(--chunk-c3-bg)',
    border: 'var(--chunk-c3-border)',
    mark: 'var(--chunk-c3-mark)',
  },
  {
    key: 'c4',
    name: 'Emerald',
    text: 'var(--chunk-c4-text)',
    bg: 'var(--chunk-c4-bg)',
    border: 'var(--chunk-c4-border)',
    mark: 'var(--chunk-c4-mark)',
  },
];

export const FIGURE_COLOR = {
  key: 'fig',
  name: 'Orange',
  text: 'var(--chunk-fig-text)',
  bg: 'var(--chunk-fig-bg)',
  border: 'var(--chunk-fig-border)',
  mark: 'var(--chunk-fig-mark)',
};

export const getChunkColor = (chunkOrId, index = 0) => {
  if (typeof chunkOrId === 'object' && chunkOrId?.type === 'figure') {
    return FIGURE_COLOR;
  }
  if (typeof chunkOrId === 'string' && chunkOrId.startsWith('F')) {
    return FIGURE_COLOR;
  }
  return CHUNK_COLORS[index % CHUNK_COLORS.length];
};
