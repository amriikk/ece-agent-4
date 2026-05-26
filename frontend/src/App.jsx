import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const IconSend = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
  </svg>
);
const IconUpload = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
  </svg>
);
const IconReset = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8M3 3v5h5"/>
  </svg>
);
const IconChevron = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="m6 9 6 6 6-6"/>
  </svg>
);
const IconAlert = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

function PlotlyChart({ figure }) {
  const divRef = useRef(null);
  useEffect(() => {
    if (!divRef.current || !figure?.data || !window.Plotly) return;
    window.Plotly.newPlot(divRef.current, figure.data, {
      ...figure.layout,
      autosize: true, height: 340,
      margin: { l: 48, r: 14, t: 36, b: 48 },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: "'JetBrains Mono', monospace", color: '#8ba3c7', size: 11 },
      title: { ...figure.layout?.title, font: { family: "'Bricolage Grotesque', sans-serif", color: '#eef4ff', size: 13 } },
      legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#8ba3c7', size: 10 }, bordercolor: 'rgba(59,130,246,0.15)', borderwidth: 1 },
      xaxis: { ...figure.layout?.xaxis, gridcolor: 'rgba(59,130,246,0.08)', linecolor: 'rgba(59,130,246,0.15)', tickfont: { color: '#3d5a80', size: 10 } },
      yaxis: { ...figure.layout?.yaxis, gridcolor: 'rgba(59,130,246,0.08)', linecolor: 'rgba(59,130,246,0.15)', tickfont: { color: '#3d5a80', size: 10 } },
      colorway: ['#3b82f6','#22d3ee','#818cf8','#10b981','#f59e0b','#ef4444','#60a5fa'],
    }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d','lasso2d'] });
    return () => { if (divRef.current) window.Plotly.purge(divRef.current); };
  }, [figure]);
  return <div ref={divRef} style={{ width: '100%', height: '340px' }} />;
}

function TracePanel({ traces, validatorVerdict, open, onToggle }) {
  if (!traces.length && !validatorVerdict) return null;
  const verdictClass = !validatorVerdict ? ''
    : validatorVerdict.startsWith('APPROVED') ? 'approved'
    : validatorVerdict.startsWith('RETRY') ? 'retry'
    : 'suspicious';
  return (
    <div className={`trace-panel ${open ? 'open' : ''}`}>
      <button className="trace-toggle" onClick={onToggle}>
        <span style={{ color: 'var(--blue-400)', fontSize: '10px' }}>◆</span>
        Agent Trace
        {traces.length > 0 && <span className="trace-count">{traces.length} steps</span>}
        <span className="trace-chevron"><IconChevron /></span>
      </button>
      {open && (
        <div className="trace-body">
          {traces.map((t, i) => (
            <div key={i} className="trace-step">
              <div className={`trace-step-dot ${t.n === 0 ? 'inspect' : i === traces.length - 1 ? 'done' : ''}`} />
              <div className="trace-step-label">{t.n === 0 ? 'Auto-inspect' : `Iteration ${t.n}`}</div>
              {t.reasoning && (
                <div className="trace-field">
                  <span className="trace-field-label">Reasoning</span>
                  <p className="trace-field-text">{t.reasoning}</p>
                </div>
              )}
              {t.code && (
                <div className="trace-field">
                  <span className="trace-field-label">Code</span>
                  <pre className="trace-field-code">{t.code}</pre>
                </div>
              )}
              {t.observation && (
                <div className="trace-field">
                  <span className="trace-field-label">Observation</span>
                  <pre className="trace-field-code" style={{ color: 'var(--cyan-300)' }}>{t.observation}</pre>
                </div>
              )}
            </div>
          ))}
          {validatorVerdict && (
            <div className={`trace-verdict ${verdictClass}`}>
              <span>◆</span><strong>Validator:</strong> {validatorVerdict}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CitationStrip({ citations }) {
  if (!citations?.length) return null;
  return (
    <div className="citation-strip">
      <span className="citation-label">Sources</span>
      {citations.map((c, i) => (
        <a key={i} href={c.url} target="_blank" rel="noopener noreferrer" className="citation-chip">
          [{i + 1}] {c.title?.slice(0, 38) || c.url}
        </a>
      ))}
    </div>
  );
}

function StatusStream({ messages }) {
  if (!messages.length) return null;
  return (
    <div className="status-wrap">
      <span className="status-label">Processing</span>
      <div className="status-card">
        {messages.map((m, i) => {
          const isActive = i === messages.length - 1;
          return (
            <div key={i} className={`status-line ${isActive ? 'active' : 'done'}`}>
              {isActive ? <span className="status-icon-spin" /> : <span className="status-icon-done">✓</span>}
              {m}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DatasetBar({ datasets, onUpload }) {
  return (
    <div className="dataset-bar">
      <span className="dataset-label">Data</span>
      {datasets.length === 0
        ? <span className="dataset-none">none loaded</span>
        : datasets.map(y => <span key={y} className="dataset-chip">{y}</span>)
      }
      <input type="file" id="csv-upload" accept=".csv" style={{ display: 'none' }} onChange={onUpload} multiple />
      <label htmlFor="csv-upload" className="upload-btn"><IconUpload /> Load CSV</label>
    </div>
  );
}

const SUGGESTIONS = [
  { icon: '📊', text: 'What does this dataset contain?' },
  { icon: '📈', text: 'Compare ROE of Technology vs Healthcare 2015–2018' },
  { icon: '🔍', text: 'Find undervalued Tech stocks in 2017 (PE ratio < 15)' },
  { icon: '🌐', text: 'What caused the 2016 oil price crash?' },
  { icon: '💹', text: 'Which sectors had highest revenue growth in 2016?' },
  { icon: '📉', text: 'Show free cash flow trends across all years' },
];

export default function App() {
  const [messages, setMessages]       = useState([]);
  const [input, setInput]             = useState('');
  const [isLoading, setIsLoading]     = useState(false);
  const [statusLog, setStatusLog]     = useState([]);
  const [error, setError]             = useState(null);
  const [loadedYears, setLoadedYears] = useState([]);
  const [traceOpen, setTraceOpen]     = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/datasets')
      .then(r => r.json())
      .then(d => setLoadedYears(d.datasets.map(x => x.year)))
      .catch(() => {});
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, statusLog]);

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    setIsLoading(true); setError(null);
    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res  = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Upload failed');
        const data = await res.json();
        setLoadedYears(data.loaded_years || []);
        setMessages(prev => [...prev, { role: 'system', content: `Loaded **${data.filename}** — ${data.rows?.toLocaleString()} rows · year ${data.year}` }]);
      } catch (err) { setError(err.message); }
    }
    setIsLoading(false); e.target.value = '';
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;
    const userQuery = input.trim();
    setInput(''); setError(null); setIsLoading(true); setStatusLog([]);
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    let accumulatedTraces = [], accumulatedVerdict = '';
    try {
      const res = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith(': ') || !line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let event; try { event = JSON.parse(raw); } catch { continue; }
          if (event.type === 'progress') {
            setStatusLog(prev => [...prev, event.message]);
          } else if (event.type === 'iteration') {
            accumulatedTraces = [...accumulatedTraces, { n: event.n, reasoning: event.reasoning, code: event.code, observation: event.observation }];
          } else if (event.type === 'validator') {
            accumulatedVerdict = event.verdict + (event.reason ? `: ${event.reason}` : '');
            setStatusLog(prev => [...prev, event.verdict.startsWith('APPROVED') ? '✓ Validator approved' : `Validator: ${event.verdict}`]);
          } else if (event.type === 'done') {
            setStatusLog([]);
            setMessages(prev => [...prev, { role: 'ai', content: event.answer, plots: event.plots || [], citations: event.citations || [], traces: accumulatedTraces, verdict: accumulatedVerdict }]);
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }
        }
      }
    } catch (err) {
      setStatusLog([]); setError(err.message || 'Request failed.');
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); } };
  const handleReset = async () => {
    await fetch('http://localhost:8000/api/reset', { method: 'POST' });
    setMessages([]); setStatusLog([]); setError(null);
  };

  return (
    <div className="root">
      <header className="header">
        <div className="header-left">
          <div className="logo-mark">
            <div className="logo-icon">D</div>
            <span className="logo-name">Datum</span>
          </div>
          <span className="logo-badge">HW4</span>
        </div>
        <div className="header-right">
          <DatasetBar datasets={loadedYears} onUpload={handleFileUpload} />
          <button className="reset-btn" onClick={handleReset} title="Reset conversation"><IconReset /></button>
        </div>
      </header>

      <main className="chat-pane">
        {messages.length === 0 && !isLoading && (
          <div className="empty-state">
            <div className="empty-icon">◈</div>
            <h2 className="empty-title">What would you like to analyze?</h2>
            <p className="empty-hint">Ask about financial metrics, sector trends, and stock comparisons — or any general question answered by web search.</p>
            <div className="suggestion-grid">
              {SUGGESTIONS.map(s => (
                <button key={s.text} className="suggestion-card" onClick={() => { setInput(s.text); inputRef.current?.focus(); }}>
                  <span className="suggestion-icon">{s.icon}</span>
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === 'system') return (
            <div key={i} className="system-notice">
              <span className="system-dot">●</span>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          );
          if (msg.role === 'user') return (
            <div key={i} className="message-row user">
              <div className="bubble-user">{msg.content}</div>
            </div>
          );
          return (
            <div key={i} className="message-row ai">
              <div className="bubble-ai-wrap">
                <div className="ai-header">
                  <div className="ai-avatar">D</div>
                  <span className="ai-label">Datum AI</span>
                </div>
                <div className="bubble-ai">
                  <div className="ai-content"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                  <CitationStrip citations={msg.citations} />
                  {msg.plots?.map((p, pi) => (
                    <div key={pi} className="chart-container">
                      <div className="chart-header">
                        <span className="chart-dot" />
                        <span className="chart-title">{p.title || 'Visualization'}</span>
                      </div>
                      <PlotlyChart figure={p.plotly_json} />
                    </div>
                  ))}
                  {(msg.traces?.length > 0 || msg.verdict) && (
                    <TracePanel traces={msg.traces || []} validatorVerdict={msg.verdict} open={traceOpen} onToggle={() => setTraceOpen(o => !o)} />
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {isLoading && <StatusStream messages={statusLog} />}
        {error && <div className="error-bar"><IconAlert />{error}</div>}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-footer">
        <div className={`input-border-wrap ${isLoading ? 'processing' : ''}`}>
          <div className="input-inner">
            <textarea
              ref={inputRef} className="query-textarea" value={input}
              onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
              placeholder="Ask about financial data or any general question..."
              disabled={isLoading} rows={1}
            />
            <button className="send-btn" onClick={handleSubmit} disabled={isLoading || !input.trim()} title="Send">
              {isLoading ? <span className="send-spinner" /> : <IconSend />}
            </button>
          </div>
        </div>
        <div className="footer-meta">
          <span>Shift+Enter for new line · Enter to send</span>
          <span className="status-pip">
            <span className={`pip-dot ${isLoading ? 'busy' : ''}`} />
            {isLoading ? 'Processing' : 'Ready'}
          </span>
        </div>
      </footer>
    </div>
  );
}
