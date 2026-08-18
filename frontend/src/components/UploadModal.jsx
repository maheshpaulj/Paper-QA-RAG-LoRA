import React, { useState, useRef } from 'react';
import { ingestPaper } from '../api/client';
import './UploadModal.css';

const UploadModal = ({ onClose, onSuccess }) => {
  const [file, setFile] = useState(null);
  const [indexName, setIndexName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type === 'application/pdf') {
      handleFileSelect(dropped);
    }
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    const name = selectedFile.name
      .replace('.pdf', '')
      .replace(/[^a-zA-Z0-9_-]/g, '_')
      .toLowerCase();
    setIndexName(name);
  };

  const handleUpload = async () => {
    if (!file || !indexName.trim()) return;
    setIsUploading(true);
    setError('');

    try {
      await ingestPaper(file, indexName.trim());
      await onSuccess();
      onClose();
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Document ingestion failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="upload-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="upload-modal-header">
          <h2 className="upload-modal-title">Ingest Research Paper</h2>
          <button className="modal-close-icon" onClick={onClose}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          hidden
          onChange={(e) => handleFileSelect(e.target.files[0])}
        />

        {!file ? (
          <div
            className="drop-area-box"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="drop-icon-wrap">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
              </svg>
            </div>
            <p className="drop-main-text">Drag and drop PDF here</p>
            <p className="drop-sub-text">or click to browse local files</p>
          </div>
        ) : (
          <div className="file-selected-pill">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <span className="file-selected-name">{file.name}</span>
            <button
              className="btn btn-ghost"
              style={{ padding: '2px 6px', fontSize: '11px' }}
              onClick={() => {
                setFile(null);
                setIndexName('');
              }}
            >
              Change
            </button>
          </div>
        )}

        <div className="form-group-block">
          <label>Index Key (unique identifier)</label>
          <input
            type="text"
            className="input-field"
            value={indexName}
            onChange={(e) => setIndexName(e.target.value)}
            placeholder="e.g. attention_paper"
            disabled={isUploading}
          />
          <span className="form-hint">
            Alphanumeric characters, underscores, and dashes only.
          </span>
        </div>

        {error && (
          <p style={{ color: '#f87171', fontSize: '12px', marginBottom: '12px' }}>
            {error}
          </p>
        )}

        <button
          className="btn btn-primary modal-action-btn"
          onClick={handleUpload}
          disabled={!file || !indexName.trim() || isUploading}
        >
          {isUploading ? 'Chunking & Embedding Document...' : 'Start Ingestion'}
        </button>
      </div>
    </div>
  );
};

export default UploadModal;
