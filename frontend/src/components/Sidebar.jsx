import React, { useState } from 'react';
import { fetchArxiv, deletePaper } from '../api/client';
import './Sidebar.css';

const Sidebar = ({ papers, selectedPaper, onSelectPaper, onOpenUpload, onRefresh }) => {
  const [arxivId, setArxivId] = useState('');
  const [isFetching, setIsFetching] = useState(false);

  const handleArxivFetch = async () => {
    if (!arxivId) return;
    setIsFetching(true);
    try {
      await fetchArxiv(arxivId);
      setArxivId('');
      onRefresh();
    } catch (err) {
      console.error(err);
      alert('Failed to fetch arXiv paper');
    } finally {
      setIsFetching(false);
    }
  };

  const handleDelete = async (e, name) => {
    e.stopPropagation();
    try {
      await deletePaper(name);
      onRefresh();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <aside className="sidebar glass-panel flex-col">
      <div className="sidebar-header">
        <h1 className="logo-text">RAG<span className="accent">Mind</span></h1>
        <div className="stats-row">
          <span className="paper-count">{papers.length} Papers</span>
          <button className="icon-btn" onClick={onRefresh} title="Refresh">🔄</button>
        </div>
      </div>

      <div className="sidebar-actions gap-2 flex-col">
        <button className="btn btn-primary" onClick={onOpenUpload}>
          + Upload PDF
        </button>
        <div className="arxiv-input-group">
          <input 
            type="text" 
            className="input-field" 
            placeholder="arXiv ID (e.g. 1706.03762)"
            value={arxivId}
            onChange={(e) => setArxivId(e.target.value)}
          />
          <button className="btn" onClick={handleArxivFetch} disabled={isFetching}>
            {isFetching ? '...' : 'Fetch'}
          </button>
        </div>
      </div>

      <div className="paper-list">
        {papers.map((p) => (
          <div 
            key={p.index_name} 
            className={`paper-item ${selectedPaper?.index_name === p.index_name ? 'active' : ''}`}
            onClick={() => onSelectPaper(p)}
          >
            <div className="paper-info">
              <h3 className="paper-title" title={p.title || p.index_name}>
                {p.title || p.index_name}
              </h3>
              <span className="paper-meta">{p.page_count} pages • {p.index_name}</span>
            </div>
            <button className="delete-btn" onClick={(e) => handleDelete(e, p.index_name)}>×</button>
          </div>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
