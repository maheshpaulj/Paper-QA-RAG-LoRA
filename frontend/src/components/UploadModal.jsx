import React, { useState } from 'react';
import { ingestPaper } from '../api/client';
import './UploadModal.css';

const UploadModal = ({ onClose, onSuccess }) => {
  const [file, setFile] = useState(null);
  const [indexName, setIndexName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type === 'application/pdf') {
      handleFileSelect(dropped);
    }
  };

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    // Auto-fill index name from filename
    const name = selectedFile.name.replace('.pdf', '').replace(/[^a-zA-Z0-9_-]/g, '_');
    setIndexName(name);
  };

  const handleUpload = async () => {
    if (!file || !indexName) return;
    setIsUploading(true);
    setError('');
    
    try {
      await ingestPaper(file, indexName);
      onSuccess();
      onClose();
    } catch (err) {
      console.error(err);
      setError('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content glass-panel">
        <button className="modal-close" onClick={onClose}>×</button>
        <h2>Upload Document</h2>
        
        <div 
          className="drop-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {file ? (
            <div className="file-selected">
              <span className="file-icon">📄</span>
              <p>{file.name}</p>
            </div>
          ) : (
            <div className="drop-prompt">
              <span className="upload-icon">☁️</span>
              <p>Drag & drop a PDF here</p>
              <span className="or">or</span>
              <label className="btn btn-primary">
                Browse Files
                <input 
                  type="file" 
                  accept=".pdf" 
                  hidden 
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                />
              </label>
            </div>
          )}
        </div>

        <div className="input-group">
          <label>Index Name (Alphanumeric only)</label>
          <input 
            type="text" 
            className="input-field" 
            value={indexName}
            onChange={(e) => setIndexName(e.target.value)}
            placeholder="e.g. attention_is_all_you_need"
          />
        </div>

        {error && <p className="error-msg">{error}</p>}

        <button 
          className="btn btn-primary w-full mt-4" 
          onClick={handleUpload}
          disabled={!file || !indexName || isUploading}
        >
          {isUploading ? 'Uploading...' : 'Ingest Paper'}
        </button>
      </div>
    </div>
  );
};

export default UploadModal;
