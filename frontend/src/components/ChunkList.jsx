import React from 'react';
import { getChunkColor } from '../utils/chunkColors';
import './ChunkList.css';

const ChunkList = ({ chunks = [], onNavigateToPage }) => {
  if (!chunks || chunks.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '30px 10px', color: 'var(--text-tertiary)', fontSize: '13px' }}>
        No retrieved context. Ask a question to inspect retrieved paper passages.
      </div>
    );
  }

  return (
    <div className="chunk-list-container">
      <div className="chunk-list-header">
        <span>Grounded Chunks ({chunks.length})</span>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Click to jump in PDF</span>
      </div>

      <div className="chunks-grid">
        {chunks.map((chunk, idx) => {
          const color = getChunkColor(chunk, idx);
          return (
            <div
              key={chunk.id || idx}
              className="chunk-card-item"
              onClick={() => onNavigateToPage(chunk.page, chunk.id)}
              title="Jump to passage in PDF"
            >
              <div className="chunk-card-header">
                <span
                  className="chunk-id-tag"
                  style={{
                    color: color.text,
                    backgroundColor: color.bg,
                    border: `1px solid ${color.border}`,
                  }}
                >
                  {chunk.id || `C${idx + 1}`}
                </span>
                <div className="chunk-card-tags">
                  <span>Page {chunk.page}</span>
                  {chunk.section && <span>• {chunk.section}</span>}
                  {chunk.type === 'figure' && <span>• Figure</span>}
                </div>
              </div>

              {chunk.image_url && (
                <img
                  src={chunk.image_url}
                  alt={chunk.id || 'Figure'}
                  className="chunk-image-preview-thumb"
                />
              )}

              <div className="chunk-text-snippet">
                {chunk.text}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ChunkList;
