import React, { useState, useRef, useEffect } from 'react';
import { askQuestion } from '../api/client';
import ChunkList from './ChunkList';
import './ChatPanel.css';

const ChatPanel = ({ paper, onChunksUpdate, onNavigateToPage }) => {
  const [question, setQuestion] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history, isLoading]);

  // Clear history when paper changes
  useEffect(() => {
    setHistory([]);
    setQuestion('');
  }, [paper]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    const q = question;
    setQuestion('');
    
    // Add user message immediately
    setHistory(prev => [...prev, { role: 'user', text: q }]);
    setIsLoading(true);
    onChunksUpdate([]); // Clear chunks

    try {
      const result = await askQuestion(q, paper.index_name);
      setHistory(prev => [...prev, { 
        role: 'assistant', 
        text: result.answer,
        route: result.route,
        chunks: result.chunks
      }]);
      onChunksUpdate(result.chunks || []);
    } catch (err) {
      console.error(err);
      setHistory(prev => [...prev, { 
        role: 'error', 
        text: 'An error occurred while getting the answer. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to format citations in text like [C1], [C2]
  const renderFormattedText = (text, chunks) => {
    if (!text) return null;
    
    // Simple regex to match [C1], [C2], etc.
    const parts = text.split(/(\[C\d+\])/g);
    
    return parts.map((part, i) => {
      const match = part.match(/\[C(\d+)\]/);
      if (match) {
        const chunkIndex = parseInt(match[1]) - 1;
        const chunkId = chunks && chunks[chunkIndex] ? chunks[chunkIndex].id : null;
        
        return (
          <span 
            key={i} 
            className="citation-badge"
            title={chunkId ? `View Citation ${match[1]}` : 'Citation'}
            onClick={() => {
              if (chunks && chunks[chunkIndex]) {
                onNavigateToPage(chunks[chunkIndex].page);
              }
            }}
          >
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <aside className="chat-panel glass-panel">
      <div className="chat-header">
        <h2>Research Assistant</h2>
        <span className="badge">AI Powered</span>
      </div>

      <div className="chat-history">
        {history.length === 0 ? (
          <div className="chat-empty">
            <span className="chat-icon">✨</span>
            <p>Ask a question about this paper.</p>
          </div>
        ) : (
          history.map((msg, idx) => (
            <div key={idx} className={`chat-message ${msg.role}`}>
              {msg.role === 'assistant' && msg.route && (
                <div className="route-indicator">
                  Route: <span>{msg.route}</span>
                </div>
              )}
              
              <div className="message-content">
                {msg.role === 'assistant' ? renderFormattedText(msg.text, msg.chunks) : msg.text}
              </div>
              
              {msg.role === 'assistant' && msg.chunks && msg.chunks.length > 0 && (
                <div className="message-chunks">
                  <ChunkList 
                    chunks={msg.chunks} 
                    onNavigateToPage={onNavigateToPage} 
                  />
                </div>
              )}
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="chat-message assistant">
            <div className="loading-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-area" onSubmit={handleSubmit}>
        <input
          type="text"
          className="input-field"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="send-btn"
          disabled={!question.trim() || isLoading}
        >
          {isLoading ? '...' : '↗'}
        </button>
      </form>
    </aside>
  );
};

export default ChatPanel;
