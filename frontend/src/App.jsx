import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import PdfViewer from './components/PdfViewer';
import UploadModal from './components/UploadModal';
import { listPapers } from './api/client';
import './index.css';

function App() {
  const [papers, setPapers] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [activeChunks, setActiveChunks] = useState([]);
  const [targetPage, setTargetPage] = useState(null);
  const [highlightChunkId, setHighlightChunkId] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isPdfOpen, setIsPdfOpen] = useState(true);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isLoadingPapers, setIsLoadingPapers] = useState(true);

  const fetchPapers = async () => {
    try {
      setIsLoadingPapers(true);
      const data = await listPapers();
      setPapers(data);
      // Auto-select first paper if none selected
      if (!selectedPaper && data.length > 0) {
        setSelectedPaper(data[0]);
      }
    } catch (error) {
      console.error('Failed to fetch papers:', error);
    } finally {
      setIsLoadingPapers(false);
    }
  };

  useEffect(() => {
    fetchPapers();
  }, []);

  const handleSelectPaper = (paper) => {
    setSelectedPaper(paper);
    setActiveChunks([]);
    setTargetPage(null);
    setHighlightChunkId(null);
  };

  const handleNavigateToPage = (pageNumber, chunkId) => {
    setTargetPage(pageNumber);
    setHighlightChunkId(chunkId || null);
    if (!isPdfOpen) {
      setIsPdfOpen(true);
    }
  };

  return (
    <div className="app-container">
      {/* 1. Left Sidebar: Library & Corpus */}
      <Sidebar
        papers={papers}
        selectedPaper={selectedPaper}
        onSelectPaper={handleSelectPaper}
        onOpenUpload={() => setIsUploadModalOpen(true)}
        onRefresh={fetchPapers}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      {/* 2. Center Panel: ChatGPT / Claude Conversation */}
      <ChatPanel
        paper={selectedPaper}
        onChunksUpdate={setActiveChunks}
        onNavigateToPage={handleNavigateToPage}
        onTogglePdf={() => setIsPdfOpen(!isPdfOpen)}
        isPdfOpen={isPdfOpen}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        isSidebarOpen={isSidebarOpen}
      />

      {/* 3. Right Panel: PDF Viewer with Chunk Highlights */}
      <div className={`pdf-panel-container ${isPdfOpen && selectedPaper ? '' : 'collapsed'}`}>
        {selectedPaper && isPdfOpen && (
          <PdfViewer
            paper={selectedPaper}
            activeChunks={activeChunks}
            targetPage={targetPage}
            highlightChunkId={highlightChunkId}
            onClose={() => setIsPdfOpen(false)}
          />
        )}
      </div>

      {/* Upload Modal */}
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
