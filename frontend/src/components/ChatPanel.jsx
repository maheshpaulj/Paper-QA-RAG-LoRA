import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { askQuestion } from '../api/client';
import { getChunkColor } from '../utils/chunkColors';
import './ChatPanel.css';

const STARTER_PROMPTS = [
  {
    title: 'Summarize core findings',
    sub: 'Extract main hypothesis, approach, and key takeaways',
    prompt: 'Summarize the core hypothesis, methodology, and primary results of this paper.',
  },
  {
    title: 'Explain architecture & methods',
    sub: 'Detail the model design, formulation, and innovations',
    prompt: 'Explain the proposed model architecture, algorithmic flow, and mathematical formulation.',
  },
  {
    title: 'Experimental benchmarks',
    sub: 'Review metrics, datasets, and baseline comparisons',
    prompt: 'What are the main experimental benchmarks, baseline comparisons, and quantitative metrics reported?',
  },
  {
    title: 'Dataset & implementation',
    sub: 'Inspect data collection, annotations, and setup',
    prompt: 'How was the dataset collected, annotated, and prepared for experiments?',
  },
];

const ROUTE_INFO = {
  qa: { label: 'Semantic QA (Top-5 Reranked)', kind: 'route-badge' },
  section: { label: 'Section Extraction', kind: 'route-badge' },
  summary: { label: 'Paper Summary', kind: 'route-badge' },
  figure: { label: 'Figure Grounding', kind: 'route-badge' },
  metadata: { label: 'Document Metadata', kind: 'route-badge' },
  'metadata+text': { label: 'Front Matter & Text', kind: 'route-badge' },
};

const THINKING_MESSAGES = [
  'Searching indexed paper chunks in FAISS...',
  'Reranking top candidate passages with cross-encoder...',
  'Synthesizing answer with verified citations...',
];

const ChatPanel = ({
  paper,
  onChunksUpdate,
  onNavigateToPage,
  onTogglePdf,
  isPdfOpen,
  onToggleSidebar,
  isSidebarOpen,
}) => {
  const [question, setQuestion] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [thinkingStep, setThinkingStep] = useState(0);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history, isLoading]);

  // Reset chat when active paper changes
  useEffect(() => {
    setHistory([]);
    setQuestion('');
    onChunksUpdate([]);
  }, [paper?.index_name]);

  // Cycle thinking phase text while loading
  useEffect(() => {
    if (!isLoading) {
      setThinkingStep(0);
      return;
    }
    const interval = setInterval(() => {
      setThinkingStep((prev) => (prev + 1) % THINKING_MESSAGES.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [isLoading]);

  const handleAsk = async (promptText) => {
    const q = (promptText || question).trim();
    if (!q || isLoading || !paper) return;

    setQuestion('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setHistory((prev) => [...prev, { role: 'user', text: q }]);
    setIsLoading(true);

    try {
      const chatHistory = history
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .slice(-8)
        .map(m => ({ role: m.role, text: m.text }));
        
      const result = await askQuestion(q, paper.index_name, chatHistory);
      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: result.answer,
          route: result.route,
          chunks: result.chunks || [],
        },
      ]);
      onChunksUpdate(result.chunks || []);
    } catch (err) {
      console.error(err);
      setHistory((prev) => [
        ...prev,
        {
          role: 'error',
          text: `Error connecting to RAG backend: ${err.response?.data?.detail || err.message || 'Please check API server connection.'}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const handleTextareaInput = (e) => {
    setQuestion(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 150)}px`;
  };

  // Preprocess text to turn [C1], [F0] into markdown links without breaking markdown lists/structure
  const formatMarkdownText = (rawText) => {
    if (!rawText) return '';
    // Replace [C0], [C1], [F0], [C33] with markdown link [C1](cite:C1)
    return rawText.replace(/\[([CF]\d+)\]/g, '[$1](cite:$1)');
  };

  return (
    <main className="chat-workspace">
      {/* Navbar Header */}
      <header className="chat-navbar">
        <div className="chat-navbar-left">
          <button
            className="icon-btn"
            onClick={onToggleSidebar}
            title={isSidebarOpen ? 'Hide Sidebar' : 'Show Sidebar'}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>

          <div className="chat-paper-heading">
            <span className="chat-paper-title">
              {paper ? paper.title || paper.index_name : 'No Paper Selected'}
            </span>
            {paper && (
              <div className="chat-paper-meta-pills">
                {paper.page_count ? (
                  <span className="tag-pill">{paper.page_count}p</span>
                ) : null}
                {paper.reference_count ? (
                  <span className="tag-pill">{paper.reference_count} refs</span>
                ) : null}
              </div>
            )}
          </div>
        </div>

        <div className="chat-navbar-right">
          {history.length > 0 && (
            <button
              className="btn btn-ghost"
              onClick={() => {
                setHistory([]);
                onChunksUpdate([]);
              }}
              title="Clear conversation"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
              <span>Clear</span>
            </button>
          )}

          <button
            className={`btn ${isPdfOpen ? 'btn-primary' : ''}`}
            onClick={onTogglePdf}
            title={isPdfOpen ? 'Hide PDF' : 'View PDF Document'}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <span>{isPdfOpen ? 'PDF Open' : 'View PDF'}</span>
          </button>
        </div>
      </header>

      {/* Message Area */}
      <div className="chat-feed">
        {!paper ? (
          <div className="empty-chat-hero">
            <div className="hero-symbol">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
            </div>
            <h2 className="hero-title">Select a Research Paper</h2>
            <p className="hero-desc">
              Choose a paper from the left sidebar or upload a PDF to ask grounded questions.
            </p>
          </div>
        ) : history.length === 0 ? (
          <div className="empty-chat-hero">
            <div className="hero-symbol">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 16v-4M12 8h.01" />
              </svg>
            </div>
            <h2 className="hero-title">Ask about {paper.title || paper.index_name}</h2>
            <p className="hero-desc">
              Questions are answered using LoRA bi-encoder retrieval and cross-encoder reranking.
            </p>

            <div className="starter-cards-grid">
              {STARTER_PROMPTS.map((item, idx) => (
                <div
                  key={idx}
                  className="starter-card"
                  onClick={() => handleAsk(item.prompt)}
                >
                  <span className="starter-card-label">{item.title}</span>
                  <span className="starter-card-sub">{item.sub}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          history.map((msg, idx) => {
            // Build map of chunk id -> index for color consistency
            const chunkMap = {};
            const figureChunks = [];
            (msg.chunks || []).forEach((c, cIdx) => {
              chunkMap[c.id] = cIdx;
              if (c.type === 'figure' || c.image_url) {
                figureChunks.push(c);
              }
            });

            return (
              <div key={idx} className={`message-entry ${msg.role}`}>
                {msg.role === 'assistant' && (
                  <div className="message-avatar-box assistant">AI</div>
                )}

                <div className="message-body-container">
                  {msg.role === 'assistant' && msg.route && (
                    <div className="route-badge-row">
                      <span className="tag-pill route-badge">
                        {ROUTE_INFO[msg.route]?.label || msg.route}
                      </span>
                    </div>
                  )}

                  <div className={`message-card ${msg.role}`}>
                    {msg.role === 'assistant' ? (
                      <div className="prose-content">
                        <ReactMarkdown
                          components={{
                            a: ({ href, children }) => {
                              if (href && href.startsWith('cite:')) {
                                const citeId = href.replace('cite:', '');
                                const chunkIndex = chunkMap[citeId] ?? 0;
                                const color = getChunkColor(citeId, chunkIndex);
                                const targetChunk = (msg.chunks || []).find((c) => c.id === citeId);

                                return (
                                  <button
                                    type="button"
                                    className="inline-citation-chip"
                                    style={{
                                      color: color.text,
                                      backgroundColor: color.bg,
                                      borderColor: color.border,
                                    }}
                                    title={targetChunk ? `View page ${targetChunk.page} in PDF` : `Citation ${citeId}`}
                                    onClick={() => {
                                      if (targetChunk) {
                                        onChunksUpdate(msg.chunks || []);
                                        onNavigateToPage(targetChunk.page, targetChunk.id);
                                      }
                                    }}
                                  >
                                    {children}
                                  </button>
                                );
                              }
                              return (
                                <a href={href} target="_blank" rel="noreferrer">
                                  {children}
                                </a>
                              );
                            },
                          }}
                        >
                          {formatMarkdownText(msg.text)}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div>{msg.text}</div>
                    )}

                    {/* In-chat Extracted Figures Strip */}
                    {figureChunks.length > 0 && (
                      <div className="figures-preview-strip">
                        {figureChunks.map((fig, fIdx) => (
                          <div
                            key={fIdx}
                            className="figure-preview-card"
                            onClick={() => onNavigateToPage(fig.page, fig.id)}
                            title="Click to view figure in PDF"
                          >
                            {fig.image_url && (
                              <img
                                src={fig.image_url}
                                alt={fig.id || 'Figure'}
                                className="figure-thumb-img"
                              />
                            )}
                            <div className="figure-caption-text">
                              {fig.text || 'Figure from paper'}
                            </div>
                            <span className="figure-tag">{fig.id} • p.{fig.page}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Bottom Citation Badges Footer */}
                    {msg.role === 'assistant' && msg.chunks && msg.chunks.length > 0 && (
                      <div className="citations-footer-bar">
                        <span style={{ marginRight: '4px' }}>Citations:</span>
                        {msg.chunks.map((chunk, cIdx) => {
                          const color = getChunkColor(chunk, cIdx);
                          return (
                            <button
                              key={chunk.id || cIdx}
                              type="button"
                              className="citation-footer-chip"
                              style={{
                                color: color.text,
                                backgroundColor: color.bg,
                                border: `1px solid ${color.border}`,
                              }}
                              onClick={() => onNavigateToPage(chunk.page, chunk.id)}
                            >
                              <span>{chunk.id}</span>
                              <span style={{ opacity: 0.75 }}>p.{chunk.page}</span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>

                {msg.role === 'user' && (
                  <div className="message-avatar-box user">U</div>
                )}
              </div>
            );
          })
        )}

        {/* Claude-Style Thinking / Loading State */}
        {isLoading && (
          <div className="message-entry assistant">
            <div className="message-avatar-box assistant">AI</div>
            <div className="message-body-container">
              <div className="claude-thinking-box">
                <div className="thinking-pulse-dot"></div>
                <span className="thinking-phase-text">
                  {THINKING_MESSAGES[thinkingStep]}
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Floating Prompt Input Bar */}
      <footer className="chat-input-wrapper">
        <form
          className="chat-input-container"
          onSubmit={(e) => {
            e.preventDefault();
            handleAsk();
          }}
        >
          <textarea
            ref={textareaRef}
            className="chat-input-textarea"
            placeholder={paper ? `Ask a question about ${paper.title || paper.index_name}...` : 'Select a paper first...'}
            value={question}
            onChange={handleTextareaInput}
            onKeyDown={handleKeyDown}
            disabled={!paper || isLoading}
            rows={1}
          />
          <div className="chat-input-bar-footer">
            <span className="input-instructions">
              <strong>Enter</strong> to send • <strong>Shift + Enter</strong> for newline
            </span>
            <button
              type="submit"
              className="send-trigger-btn"
              disabled={!question.trim() || isLoading || !paper}
              title="Send message"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          </div>
        </form>
      </footer>
    </main>
  );
};

export default ChatPanel;
