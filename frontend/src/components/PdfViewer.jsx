import React, { useState, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { getPdfUrl } from '../api/client';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import './PdfViewer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const PdfViewer = ({ paper, activeChunks = [], targetPage }) => {
  const [numPages, setNumPages] = useState(null);
  const [scale, setScale] = useState(1.2);
  const containerRef = useRef(null);
  const pageRefs = useRef({});

  useEffect(() => {
    if (targetPage && pageRefs.current[targetPage]) {
      pageRefs.current[targetPage].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [targetPage]);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  const getHighlightColor = (chunkId) => {
    // Map chunk to a color based on its index in activeChunks
    const idx = activeChunks.findIndex(c => c.id === chunkId);
    if (idx === -1) return 'var(--accent-purple)';
    const colors = [
      'var(--color-chunk-1)', 
      'var(--color-chunk-2)', 
      'var(--color-chunk-3)', 
      'var(--color-chunk-4)', 
      'var(--color-chunk-5)'
    ];
    return colors[idx % colors.length];
  };

  const textRenderer = (textItem, pageNumber) => {
    const text = textItem.str;
    if (!activeChunks || activeChunks.length === 0 || !text.trim()) return text;

    const chunksOnPage = activeChunks.filter(c => c.page === pageNumber);
    if (chunksOnPage.length === 0) return text;

    // Check if this text fragment is part of any retrieved chunk.
    // PDF text layer gives us small fragments (words/phrases), so we check
    // if the chunk text contains this fragment.
    for (let i = 0; i < chunksOnPage.length; i++) {
      const chunk = chunksOnPage[i];
      if (!chunk.text) continue;
      const chunkLower = chunk.text.toLowerCase();
      const textLower = text.toLowerCase();

      if (textLower.length > 2 && chunkLower.includes(textLower)) {
        const colors = [
          'rgba(108, 99, 255, 0.35)',
          'rgba(0, 212, 170, 0.35)',
          'rgba(255, 107, 107, 0.35)',
          'rgba(254, 202, 87, 0.35)',
          'rgba(72, 219, 251, 0.35)',
        ];
        const color = colors[i % colors.length];
        return `<mark style="background-color: ${color}; border-radius: 2px; padding: 0 1px;">${text}</mark>`;
      }
    }

    return text;
  };

  return (
    <div className="pdf-viewer-container">
      <div className="pdf-toolbar glass-panel">
        <h3>{paper.title || paper.index_name}</h3>
        <div className="pdf-controls">
          <button className="btn" onClick={() => setScale(s => Math.max(0.5, s - 0.1))}>-</button>
          <span>{Math.round(scale * 100)}%</span>
          <button className="btn" onClick={() => setScale(s => Math.min(2.5, s + 0.1))}>+</button>
        </div>
      </div>

      <div className="pdf-scroll-area" ref={containerRef}>
        <Document
          file={getPdfUrl(paper.index_name)}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={<div className="pdf-loading"><div className="spinner"></div><p>Loading PDF...</p></div>}
          error={<div className="pdf-error">Failed to load PDF.</div>}
        >
          {numPages && Array.from(new Array(numPages), (el, index) => (
            <div 
              key={`page_${index + 1}`} 
              className="pdf-page-wrapper"
              ref={el => pageRefs.current[index + 1] = el}
              data-page={index + 1}
            >
              <div className="page-number-indicator">Page {index + 1}</div>
              <Page 
                pageNumber={index + 1} 
                scale={scale} 
                customTextRenderer={({ str, itemIndex }) => textRenderer({str}, index + 1)}
                renderAnnotationLayer={true}
                renderTextLayer={true}
              />
            </div>
          ))}
        </Document>
      </div>
    </div>
  );
};

export default PdfViewer;
