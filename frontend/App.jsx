import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

// ── Plotly: CSP-safe direct API call ────────────────────────────────────────
function PlotlyChart({ figure }) {
  const divRef = useRef(null);
  useEffect(() => {
    if (!divRef.current || !figure?.data || !window.Plotly) return;
    window.Plotly.newPlot(divRef.current, figure.data, {
      ...figure.layout,
      autosize: true, height: 360,
      margin: { l: 52, r: 16, t: 48, b: 56 },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: "'Special Elite', monospace", color: '#7eb8e8' },
      legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#4a7fab', size: 11 } },
      xaxis: { ...figure.layout?.xaxis, gridcolor: 'rgba(100,160,220,0.1)', tickfont: { color: '#4a7fab', size: 10 } },
      yaxis: { ...figure.layout?.yaxis, gridcolor: 'rgba(100,160,220,0.1)', tickfont: { color: '#4a7fab', size: 10 } },
      colorway: ['#7eb8e8','#6ea87e','#e87e7e','#c8a96e','#9e7ec8','#7ec8c8'],
    }, { responsive: true, displaylogo: false });
    return () => { if (divRef.current) window.Plotly.purge(divRef.current); };
  }, [figure]);
  return <div ref={divRef} style={{ width: '100%', height: '360px' }} />;
}

// ── Agent Trace Panel ────────────────────────────────────────────────────────
function TracePanel({ traces, validatorVerdict, open, onToggle }) {
  if (!traces.length && !validatorVerdict) return null;
  return (
    <div className={`trace-panel ${open ? 'open' : ''}`}>
      <button className="trace-toggle" onClick={onToggle}>
        {open ? '◂ HIDE TRACE' : '▸ AGENT TRACE'}
      </button>
      {open && (
        <div className="trace-content">
          <div className="trace-header">EXECUTION TRACE</div>
          {traces.map((t, i) => (
            <div key={i} className="trace-iteration">
              <div className="trace-iter-label">
                {t.n === 0 ? '⚙ AUTO-INSPECT' : `✍ ITERATION ${t.n}`}
              </div>
              {t.reasoning && (
                <div className="trace-section">
                  <span className="trace-key">REASONING</span>
                  <p className="trace-value">{t.reasoning}</p>
                </div>
              )}
              {t.code && (
                <div className="trace-section">
                  <span className="trace-key">CODE</span>
                  <pre className="trace-code">{t.code}</pre>
                </div>
              )}
              {t.observation && (
                <div className="trace-section">
                  <span className="trace-key">OBSERVATION</span>
                  <pre className="trace-obs">{t.observation}</pre>
                </div>
              )}
            </div>
          ))}
          {validatorVerdict && (
            <div className={`trace-verdict ${validatorVerdict.startsWith('APPROVED') ? 'approved' : validatorVerdict.startsWith('RETRY') ? 'retry' : 'suspicious'}`}>
              <span className="trace-key">VALIDATOR</span>
              <span className="verdict-text">{validatorVerdict}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Citation Strip ────────────────────────────────────────────────────────────
function CitationStrip({ citations }) {
  if (!citations?.length) return null;
  return (
    <div className="citation-strip">
      <span className="citation-label">SOURCES</span>
      {citations.map((c, i) => (
        <a key={i} href={c.url} target="_blank" rel="noopener noreferrer" className="citation-chip">
          [{i+1}] {c.title?.slice(0, 40) || c.url}
        </a>
      ))}
    </div>
  );
}

// ── Status Stream ─────────────────────────────────────────────────────────────
function StatusStream({ messages }) {
  if (!messages.length) return null;
  return (
    <div className="message-row ai">
      <div className="msg-stamp">PROCESSING</div>
      <div className="bubble ai status-bubble">
        {messages.map((m, i) => (
          <div key={i} className={`status-line ${i === messages.length - 1 ? 'active' : 'done'}`}>
            <span className="status-tick">{i === messages.length - 1 ? '▸' : '✓'}</span>
            {m}
          </div>
        ))}
        <span className="status-cursor">▌</span>
      </div>
    </div>
  );
}

// ── Dataset Badge Bar ─────────────────────────────────────────────────────────
function DatasetBar({ datasets, onUpload }) {
  return (
    <div className="dataset-bar">
      <span className="dataset-label">LOADED</span>
      {datasets.length === 0
        ? <span className="dataset-none">no datasets</span>
        : datasets.map(y => <span key={y} className="dataset-badge">{y}</span>)
      }
      <input type="file" id="csv-upload" accept=".csv" style={{ display: 'none' }} onChange={onUpload} multiple />
      <label htmlFor="csv-upload" className="upload-btn-small">⊕ LOAD CSV</label>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
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
    setIsLoading(true);
    setError(null);
    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res  = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Upload failed');
        const data = await res.json();
        setLoadedYears(data.loaded_years || []);
        setMessages(prev => [...prev, {
          role: 'system',
          content: `✓ Loaded **${data.filename}** (${data.rows?.toLocaleString()} rows, year ${data.year})`,
        }]);
      } catch (err) {
        setError(err.message);
      }
    }
    setIsLoading(false);
    e.target.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setError(null);
    setIsLoading(true);
    setStatusLog([]);
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);

    let accumulatedTraces  = [];
    let accumulatedVerdict = '';

    try {
      const res = await fetch('http://localhost:8000/api/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ query: userQuery }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith(': ')) continue;
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let event;
          try { event = JSON.parse(raw); } catch { continue; }

          if (event.type === 'progress') {
            setStatusLog(prev => [...prev, event.message]);
          } else if (event.type === 'iteration') {
            accumulatedTraces = [...accumulatedTraces, {
              n: event.n, reasoning: event.reasoning,
              code: event.code, observation: event.observation,
            }];
          } else if (event.type === 'validator') {
            accumulatedVerdict = event.verdict + (event.reason ? `: ${event.reason}` : '');
            setStatusLog(prev => [...prev,
              event.verdict.startsWith('APPROVED') ? '✅ Validator: APPROVED' : `🔄 Validator: ${event.verdict}`
            ]);
          } else if (event.type === 'done') {
            setStatusLog([]);
            setMessages(prev => [...prev, {
              role: 'ai', content: event.answer,
              plots: event.plots || [], citations: event.citations || [],
              traces: accumulatedTraces, verdict: accumulatedVerdict,
            }]);
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }
        }
      }
    } catch (err) {
      setStatusLog([]);
      setError(err.message || 'Request failed.');
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e); }
  };

  const handleReset = async () => {
    await fetch('http://localhost:8000/api/reset', { method: 'POST' });
    setMessages([]); setStatusLog([]); setError(null);
  };

  return (
    <div className="root">
      <div className="scanlines" aria-hidden="true" />

      <header className="header">
        <div className="header-left">
          <span className="logo-bracket">[</span>
          <h1 className="logo-text">DATUM</h1>
          <span className="logo-bracket">]</span>
          <span className="logo-sub">orchestrated analytics · hw4</span>
        </div>
        <div className="header-right">
          <DatasetBar datasets={loadedYears} onUpload={handleFileUpload} />
          <button className="reset-btn" onClick={handleReset}>↺ RESET</button>
        </div>
      </header>

      <main className="chat-pane">
        {messages.length === 0 && !isLoading && (
          <div className="empty-state">
            <div className="empty-glyph">◈</div>
            <p className="empty-title">Awaiting query.</p>
            <p className="empty-hint">Ask about financial metrics, sector trends, stock comparisons — or any general question answered by web search.</p>
            <div className="example-queries">
              {[
                'What does this dataset contain?',
                'Compare average ROE of Technology vs Healthcare from 2015 to 2018',
                'Find undervalued Technology stocks in 2017 with PE ratio below 15',
                'What caused the 2016 oil price crash?',
                'Which sectors had the highest revenue growth in 2016?',
              ].map(q => (
                <button key={q} className="example-chip" onClick={() => { setInput(q); inputRef.current?.focus(); }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === 'system') return (
            <div key={i} className="system-notice">
              <span className="sys-icon">▸</span>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          );
          if (msg.role === 'user') return (
            <div key={i} className="message-row user">
              <div className="bubble user">
                <span className="prompt-caret">&gt;&gt;</span>{msg.content}
              </div>
            </div>
          );
          return (
            <div key={i} className="message-row ai">
              <div className="msg-stamp">FILED</div>
              <div className="bubble ai">
                <div className="ai-content"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                <CitationStrip citations={msg.citations} />
                {msg.plots?.map((p, pi) => (
                  <div key={pi} className="chart-container">
                    <div className="chart-header"><span className="chart-tag">◈ {p.title || 'VISUALISATION'}</span></div>
                    <PlotlyChart figure={p.plotly_json} />
                  </div>
                ))}
                {(msg.traces?.length > 0 || msg.verdict) && (
                  <TracePanel
                    traces={msg.traces || []} validatorVerdict={msg.verdict}
                    open={traceOpen} onToggle={() => setTraceOpen(o => !o)}
                  />
                )}
              </div>
            </div>
          );
        })}

        {isLoading && <StatusStream messages={statusLog} />}
        {error && <div className="error-strip"><span className="error-icon">✕</span> {error}</div>}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-footer">
        <div className="input-row">
          <span className="input-caret">▸</span>
          <textarea
            ref={inputRef} className="query-textarea" value={input}
            onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
            placeholder="Ask about financial data or any general question..."
            disabled={isLoading} rows={1}
          />
          <button className="send-btn" onClick={handleSubmit} disabled={isLoading || !input.trim()}>
            {isLoading ? '◌' : 'RUN ▸'}
          </button>
        </div>
        <div className="footer-meta">
          <span>SHIFT+ENTER for new line · ENTER to submit</span>
          <span className={`status-dot ${isLoading ? 'busy' : 'ready'}`}>
            {isLoading ? '● PROCESSING' : '● READY'}
          </span>
        </div>
      </footer>
    </div>
  );
}
