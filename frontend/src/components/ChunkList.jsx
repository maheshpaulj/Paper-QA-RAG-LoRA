import React from 'react';
import './ChunkList.css';

const ChunkList = ({ chunks, onNavigateToPage }) => {
  if (!chunks || chunks.length === 0) return null;

  return (
    <div className="chunk-list-container">
      <div className="chunk-list-title">Retrieved Context</div>
      <div className="chunks">
        {chunks.map((chunk, idx) => (
          <div 
            key={chunk.id || idx} 
            className="chunk-card"
            onClick={() => onNavigateToPage(chunk.page)}
            title="Click to view in PDF"
          >
            <div className="chunk-header">
              <span className="chunk-badge">[C{idx + 1}]</span>
              <span className="chunk-meta">Page {chunk.page} {chunk.section ? `• ${chunk.section}` : ''}</span>
              {chunk.type === 'figure' && <span className="chunk-type-icon">🖼️</span>}
            </div>
            <div className="chunk-preview">
              {chunk.type === 'figure' ? (
                <span className="figure-text">Figure / Table referenced</span>
              ) : (
                chunk.text.substring(0, 100) + (chunk.text.length > 100 ? '...' : '')
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChunkList;
