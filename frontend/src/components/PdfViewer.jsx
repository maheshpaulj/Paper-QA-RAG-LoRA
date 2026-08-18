import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { getPdfUrl } from '../api/client';
import ChunkList from './ChunkList';
import { getChunkColor } from '../utils/chunkColors';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import './PdfViewer.css';

// Configure pdfjs worker for version 3.11.174
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version || '3.11.174'}/build/pdf.worker.min.js`;

const PdfViewer = ({
  paper,
  activeChunks = [],
  targetPage,
  highlightChunkId,
  onClose,
}) => {
  const [numPages, setNumPages] = useState(null);
  const [scale, setScale] = useState(1.1);
  const [activeTab, setActiveTab] = useState('pdf'); // 'pdf' | 'chunks'
  const [currentPageInput, setCurrentPageInput] = useState(1);
  const [loadError, setLoadError] = useState(null);
  const containerRef = useRef(null);
  const pageRefs = useRef({});

  // Reset when selected paper changes
  useEffect(() => {
    setLoadError(null);
    setNumPages(null);
    setCurrentPageInput(1);
  }, [paper?.index_name]);

  // Smooth scroll to page when targetPage updates
  useEffect(() => {
    if (targetPage && pageRefs.current[targetPage]) {
      setActiveTab('pdf');
      pageRefs.current[targetPage].scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
      setCurrentPageInput(targetPage);
    }
  }, [targetPage, highlightChunkId]);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setLoadError(null);
  };

  const onDocumentLoadError = (error) => {
    console.error('react-pdf document load error:', error);
    setLoadError(error.message || 'Failed to load PDF document.');
  };

  // Text layer highlight matcher
  const textRenderer = useCallback(
    (textItem, pageNumber) => {
      const text = textItem.str;
      if (!activeChunks || activeChunks.length === 0 || !text || !text.trim()) {
        return text;
      }

      const pageChunks = activeChunks.filter((c) => c.page === pageNumber);
      if (pageChunks.length === 0) return text;

      const textLower = text.toLowerCase().trim();
      if (textLower.length < 3) return text;

      for (let i = 0; i < pageChunks.length; i++) {
        const chunk = pageChunks[i];
        if (!chunk.text) continue;
        const chunkLower = chunk.text.toLowerCase();

        if (chunkLower.includes(textLower)) {
          const color = getChunkColor(chunk, i);
          return `<mark class="chunk-highlight-mark" style="background-color: ${color.mark}; border-bottom: 2px solid ${color.border};" title="${chunk.id || `Chunk ${i + 1}`}">${text}</mark>`;
        }
      }

      return text;
    },
    [activeChunks]
  );

  const handlePageJump = (e) => {
    e.preventDefault();
    const page = parseInt(currentPageInput, 10);
    if (page >= 1 && numPages && page <= numPages && pageRefs.current[page]) {
      pageRefs.current[page].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const pdfUrl = paper ? getPdfUrl(paper.index_name) : null;

  return (
    <div className="pdf-panel-wrapper">
      {/* Top Header & Tabs */}
      <div className="pdf-panel-header">
        <div className="pdf-tabs-group">
          <button
            className={`pdf-tab-trigger ${activeTab === 'pdf' ? 'active' : ''}`}
            onClick={() => setActiveTab('pdf')}
          >
            PDF Document
          </button>
          <button
            className={`pdf-tab-trigger ${activeTab === 'chunks' ? 'active' : ''}`}
            onClick={() => setActiveTab('chunks')}
          >
            Chunks ({activeChunks.length})
          </button>
        </div>

        <button className="icon-btn" onClick={onClose} title="Close PDF viewer">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {activeTab === 'pdf' && (
        <>
          {/* Sub Toolbar */}
          <div className="pdf-sub-toolbar">
            <div className="zoom-controls-row">
              <button
                className="icon-btn"
                onClick={() => setScale((s) => Math.max(0.6, s - 0.15))}
                title="Zoom Out"
              >
                -
              </button>
              <span className="zoom-pct-display">{Math.round(scale * 100)}%</span>
              <button
                className="icon-btn"
                onClick={() => setScale((s) => Math.min(2.2, s + 0.15))}
                title="Zoom In"
              >
                +
              </button>
              <button
                className="btn btn-ghost"
                style={{ fontSize: '11px', padding: '2px 6px' }}
                onClick={() => setScale(1.1)}
                title="Reset zoom"
              >
                Reset
              </button>
            </div>

            {numPages && (
              <form className="page-jump-form" onSubmit={handlePageJump}>
                <span>Page</span>
                <input
                  type="number"
                  className="page-input-box"
                  min="1"
                  max={numPages}
                  value={currentPageInput}
                  onChange={(e) => setCurrentPageInput(e.target.value)}
                />
                <span>/ {numPages}</span>
              </form>
            )}
          </div>

          {/* PDF Canvas */}
          <div className="pdf-canvas-feed" ref={containerRef}>
            {pdfUrl ? (
              <Document
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={
                  <div className="pdf-state-container">
                    <p>Loading document...</p>
                  </div>
                }
                error={
                  <div className="pdf-state-container">
                    <div className="pdf-error-dialog">
                      <p><strong>Failed to load PDF</strong></p>
                      <p style={{ fontSize: '12px', marginTop: '6px', color: 'var(--text-secondary)' }}>
                        {loadError || 'Could not fetch file from server.'}
                      </p>
                      <a
                        href={pdfUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="btn btn-primary"
                        style={{ marginTop: '12px', display: 'inline-flex' }}
                      >
                        Open PDF In Tab ↗
                      </a>
                    </div>
                  </div>
                }
              >
                {numPages &&
                  Array.from(new Array(numPages), (el, index) => {
                    const pageNum = index + 1;
                    const isTarget = targetPage === pageNum;
                    const hasChunks = activeChunks.some((c) => c.page === pageNum);

                    return (
                      <div
                        key={`page_${pageNum}`}
                        className={`pdf-page-container ${isTarget ? 'targeted-page' : ''}`}
                        ref={(el) => (pageRefs.current[pageNum] = el)}
                        data-page={pageNum}
                      >
                        <div className="page-corner-badge">
                          p.{pageNum} {hasChunks && '• Highlighted'}
                        </div>
                        <Page
                          pageNumber={pageNum}
                          scale={scale}
                          customTextRenderer={({ str }) =>
                            textRenderer({ str }, pageNum)
                          }
                          renderAnnotationLayer={true}
                          renderTextLayer={true}
                        />
                      </div>
                    );
                  })}
              </Document>
            ) : (
              <div className="pdf-state-container">
                <p>No paper selected.</p>
              </div>
            )}
          </div>
        </>
      )}

      {activeTab === 'chunks' && (
        <div className="chunks-tab-content">
          <ChunkList
            chunks={activeChunks}
            onNavigateToPage={(page, id) => {
              setActiveTab('pdf');
              setTimeout(() => {
                if (pageRefs.current[page]) {
                  pageRefs.current[page].scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                  });
                }
              }, 100);
            }}
          />
        </div>
      )}
    </div>
  );
};

export default PdfViewer;
