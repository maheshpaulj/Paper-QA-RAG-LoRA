import React, { useState } from 'react';
import { fetchArxiv, deletePaper } from '../api/client';
import './Sidebar.css';

const Sidebar = ({
  papers,
  selectedPaper,
  onSelectPaper,
  onOpenUpload,
  onRefresh,
  isOpen,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [arxivId, setArxivId] = useState('');
  const [isFetchingArxiv, setIsFetchingArxiv] = useState(false);
  const [showArxivBox, setShowArxivBox] = useState(false);

  const handleArxivFetch = async (e) => {
    e.preventDefault();
    if (!arxivId.trim() || isFetchingArxiv) return;
    setIsFetchingArxiv(true);
    try {
      await fetchArxiv(arxivId.trim());
      setArxivId('');
      setShowArxivBox(false);
      await onRefresh();
    } catch (err) {
      console.error(err);
      alert('Failed to download & index arXiv paper. Please check the arXiv ID.');
    } finally {
      setIsFetchingArxiv(false);
    }
  };

  const handleDelete = async (e, name, title) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to remove "${title || name}" from the index?`)) return;
    try {
      await deletePaper(name);
      await onRefresh();
    } catch (err) {
      console.error(err);
      alert('Failed to delete paper index.');
    }
  };

  const filteredPapers = papers.filter((p) => {
    const q = searchTerm.toLowerCase();
    return (
      (p.title && p.title.toLowerCase().includes(q)) ||
      (p.index_name && p.index_name.toLowerCase().includes(q))
    );
  });

  return (
    <aside className={`sidebar-container ${isOpen ? '' : 'collapsed'}`}>
      <div className="sidebar-inner">
        {/* Brand Header */}
        <div className="sidebar-header">
          <div className="brand-wrapper">
            <div className="brand-logo-badge">RAG</div>
            <span className="brand-text">PaperQA</span>
            <span className="brand-version">v6</span>
          </div>
          <button className="icon-btn" onClick={onRefresh} title="Refresh library">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
            </svg>
          </button>
        </div>

        {/* Primary Action Buttons */}
        <div className="sidebar-actions">
          <button className="action-card-btn" onClick={onOpenUpload}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14M5 12h14"/>
            </svg>
            <span>Upload PDF</span>
          </button>

          <button 
            type="button" 
            className="action-card-btn"
            onClick={() => setShowArxivBox(!showArxivBox)}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
            </svg>
            <span>Fetch from arXiv</span>
          </button>

          {showArxivBox && (
            <form className="arxiv-box" onSubmit={handleArxivFetch}>
              <div className="arxiv-form-row">
                <input
                  type="text"
                  className="input-field"
                  placeholder="e.g. 1706.03762"
                  value={arxivId}
                  onChange={(e) => setArxivId(e.target.value)}
                  disabled={isFetchingArxiv}
                  autoFocus
                />
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!arxivId.trim() || isFetchingArxiv}
                  style={{ padding: '4px 10px', fontSize: '12px' }}
                >
                  {isFetchingArxiv ? '...' : 'Fetch'}
                </button>
              </div>
            </form>
          )}
        </div>

        {/* Search Field */}
        <div className="search-container">
          <svg className="search-icon-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.3-4.3"/>
          </svg>
          <input
            type="text"
            className="input-field"
            placeholder="Search papers..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {/* Paper List Section */}
        <div className="papers-nav-header">
          <span>Corpus ({filteredPapers.length})</span>
        </div>

        <div className="papers-list">
          {filteredPapers.map((paper) => {
            const isSelected = selectedPaper?.index_name === paper.index_name;
            return (
              <div
                key={paper.index_name}
                className={`paper-nav-item ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectPaper(paper)}
              >
                <div className="paper-item-row">
                  <svg className="paper-item-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                  <div className="paper-item-details">
                    <div className="paper-item-title">
                      {paper.title || paper.index_name}
                    </div>
                    <div className="paper-item-meta">
                      {paper.page_count ? <span>{paper.page_count} pages</span> : null}
                      {paper.reference_count ? <span>• {paper.reference_count} refs</span> : null}
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px' }}>• {paper.index_name}</span>
                    </div>
                  </div>
                </div>

                <button
                  className="paper-delete-trigger"
                  onClick={(e) => handleDelete(e, paper.index_name, paper.title)}
                  title="Delete from index"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
            );
          })}

          {filteredPapers.length === 0 && (
            <div style={{ textAlign: 'center', padding: '24px 10px', color: 'var(--text-tertiary)', fontSize: '12px' }}>
              No matching papers
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sidebar-footer">
          <div className="system-status-indicator">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div className="online-dot"></div>
              <span>LoRA MiniLM • FAISS</span>
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px' }}>LCEL</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
