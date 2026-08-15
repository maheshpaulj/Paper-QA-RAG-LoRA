import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import PdfViewer from './components/PdfViewer';
import ChatPanel from './components/ChatPanel';
import UploadModal from './components/UploadModal';
import { listPapers } from './api/client';
import './index.css';

function App() {
  const [papers, setPapers] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [activeChunks, setActiveChunks] = useState([]);
  const [targetPage, setTargetPage] = useState(null);

  const fetchPapers = async () => {
    try {
      const data = await listPapers();
      setPapers(data);
    } catch (error) {
      console.error('Failed to fetch papers:', error);
    }
  };

  useEffect(() => {
    fetchPapers();
  }, []);

  const handleSelectPaper = (paper) => {
    setSelectedPaper(paper);
    setActiveChunks([]);
    setTargetPage(null);
  };

  return (
    <div className="app-container">
      <Sidebar 
        papers={papers} 
        selectedPaper={selectedPaper} 
        onSelectPaper={handleSelectPaper} 
        onOpenUpload={() => setIsUploadModalOpen(true)}
        onRefresh={fetchPapers}
      />
      
      <main className="main-content">
        {selectedPaper ? (
          <PdfViewer 
            paper={selectedPaper} 
            activeChunks={activeChunks}
            targetPage={targetPage}
          />
        ) : (
          <div className="empty-state">
            <div className="empty-state-icon">📄</div>
            <h2>No Paper Selected</h2>
            <p>Select a paper from the sidebar or upload a new one to get started.</p>
          </div>
        )}
      </main>

      {selectedPaper && (
        <ChatPanel 
          paper={selectedPaper} 
          onChunksUpdate={setActiveChunks}
          onNavigateToPage={setTargetPage}
        />
      )}

      {isUploadModalOpen && (
        <UploadModal 
          onClose={() => setIsUploadModalOpen(false)} 
          onSuccess={fetchPapers}
        />
      )}
    </div>
  );
}

export default App;
