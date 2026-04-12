import React, { useEffect, useRef, useState } from 'react'

const starterPrompts = [
  'Summarize the latest uploaded document.',
  'What are the key steps described in this procedure?',
  'Help me troubleshoot an issue using the available knowledge base.',
]

const operatingSignals = [
  { label: 'Stack', value: 'React + FastAPI' },
  { label: 'Engine', value: 'CAG v0.2' },
  { label: 'Scope', value: 'Knowledge Workspace' },
]

function getApiBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, '')
  }

  if (typeof window === 'undefined') {
    return 'http://localhost:8000'
  }

  const { protocol, hostname, port } = window.location
  // Vite dev server uses ports in the 5173–5179 range; always proxy to FastAPI on 8000
  const isViteDev = port >= '5173' && port <= '5179'
  if (isViteDev) {
    return `${protocol}//${hostname}:8000`
  }

  return `${protocol}//${hostname}:${port || '8000'}`
}

const apiBaseUrl = getApiBaseUrl()

function getPreviewApiKey() {
  const configuredApiKey = import.meta.env.VITE_CAG_API_KEY?.trim()
  if (configuredApiKey) {
    return configuredApiKey
  }

  if (typeof window === 'undefined') {
    return ''
  }

  try {
    return window.localStorage.getItem('CAG_API_KEY')?.trim() || ''
  } catch {
    return ''
  }
}

function buildApiHeaders(extraHeaders = {}) {
  const apiKey = getPreviewApiKey()
  return apiKey
    ? { ...extraHeaders, 'X-API-Key': apiKey }
    : extraHeaders
}

function formatPercentage(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '--'
  }

  return `${Math.round(value * 100)}%`
}

function basename(path) {
  if (!path) {
    return ''
  }

  return String(path).split(/[\\/]/).pop()
}

function UploadPanel({ lastUpload, onUpload }) {
  const inputRef = useRef(null)
  const [uploadMsg, setUploadMsg] = useState('')
  const [accruedFiles, setAccruedFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)

  const handleSelection = (event) => {
    const incoming = Array.from(event.target.files || [])
    setAccruedFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.name))
      const fresh = incoming.filter((f) => !existingNames.has(f.name))
      return [...prev, ...fresh]
    })
    // Reset native input so the same file can be re-added later if removed
    event.target.value = ''
  }

  const removeFile = (name) => {
    setAccruedFiles((prev) => prev.filter((f) => f.name !== name))
  }

  const handleUpload = async () => {
    if (accruedFiles.length === 0 || isUploading) return

    const form = new FormData()
    for (const file of accruedFiles) {
      form.append('files', file)
    }

    setIsUploading(true)
    setUploadMsg('Indexing is starting...')

    try {
      const response = await fetch(`${apiBaseUrl}/upload?ingest=true`, {
        method: 'POST',
        body: form,
        headers: buildApiHeaders(),
      })
      const data = await response.json()
      setUploadMsg(JSON.stringify(data, null, 2))
      setAccruedFiles([])
      onUpload?.(data)
    } catch (error) {
      setUploadMsg(`Upload failed: ${String(error)}`)
    } finally {
      setIsUploading(false)
    }
  }

  const savedFiles = Array.isArray(lastUpload?.saved) ? lastUpload.saved : []

  return (
    <section className="surface upload-panel">
      <div className="panel-heading">
        <p className="eyebrow">Knowledge Intake</p>
        <h2>Add new documents to the knowledge base</h2>
        <p className="panel-copy">
          Select files, internal notes, or reference material and start
          background ingestion right away.
        </p>
      </div>

      <label className="file-picker">
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md"
          onChange={handleSelection}
        />
        <span className="file-picker__button">Add documents</span>
        <span className="file-picker__meta">
          {accruedFiles.length > 0
            ? `${accruedFiles.length} file${accruedFiles.length > 1 ? 's' : ''} queued — add more or upload`
            : 'PDF, TXT, MD — hold Ctrl (or ⌘) to select multiple'}
        </span>
      </label>

      {accruedFiles.length > 0 && (
        <div className="file-tags">
          {accruedFiles.map((file) => (
            <span key={file.name} className="file-tag">
              {file.name}
              <button
                className="file-tag__remove"
                onClick={() => removeFile(file.name)}
                title="Remove"
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      )}

      <button
        className="button button--primary"
        onClick={handleUpload}
        disabled={isUploading || accruedFiles.length === 0}
      >
        {isUploading
          ? 'Indexing...'
          : `Upload and ingest${accruedFiles.length > 0 ? ` (${accruedFiles.length})` : ''}`}
      </button>

      <div className="mini-stats">
        <div className="mini-stat">
          <span>Latest batch</span>
          <strong>{savedFiles.length || 0} files</strong>
        </div>
        <div className="mini-stat">
          <span>Pipeline</span>
          <strong>{lastUpload?.ingest_started ? 'Running' : 'Ready'}</strong>
        </div>
      </div>

      {savedFiles.length > 0 && (
        <div className="upload-list">
          {savedFiles.slice(0, 4).map((filePath) => (
            <div key={filePath} className="upload-list__item">
              <span className="upload-list__dot" />
              <span>{basename(filePath)}</span>
            </div>
          ))}
        </div>
      )}

      {uploadMsg && <pre className="output-panel">{uploadMsg}</pre>}
    </section>
  )
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function IntelligencePanel({ lastResult, kbFiles, onDeleteFile }) {
  const citations = Array.isArray(lastResult?.citations) ? lastResult.citations : []

  return (
    <section className="surface intelligence-panel">
      <div className="panel-heading">
        <p className="eyebrow">Session Pulse</p>
        <h2>Operational signals</h2>
      </div>

      <div className="insight-grid">
        <div className="insight-card">
          <span>Documents indexed</span>
          <strong>{kbFiles.length}</strong>
        </div>
        <div className="insight-card">
          <span>Confidence</span>
          <strong>{formatPercentage(lastResult?.confidence)}</strong>
        </div>
        <div className="insight-card">
          <span>Hallucination risk</span>
          <strong>{formatPercentage(lastResult?.hallucination_risk)}</strong>
        </div>
        <div className="insight-card">
          <span>Query type</span>
          <strong>{lastResult?.query_type || 'Waiting'}</strong>
        </div>
      </div>

      {kbFiles.length > 0 && (
        <div className="kb-file-list">
          <span className="kb-file-list__label">Knowledge base</span>
          {kbFiles.map((file) => (
            <div key={file.name} className="kb-file-item">
              <span className="upload-list__dot" />
              <span className="kb-file-item__name" title={file.name}>{file.name}</span>
              <span className="kb-file-item__size">{formatSize(file.size_bytes)}</span>
              <button
                className="kb-file-item__delete"
                title="Delete document"
                onClick={() => onDeleteFile(file.name)}
              >
                🗑
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="insight-note">
        <span className="insight-note__label">Response status</span>
        <p>
          {lastResult?.should_escalate
            ? 'The pipeline suggests escalation to a specialist.'
            : 'The session is ready for questions, exploration, and troubleshooting.'}
        </p>
      </div>

      {citations.length > 0 && (
        <div className="reference-stack">
          <span className="reference-stack__label">Recent sources</span>
          {citations.slice(0, 3).map((citation, index) => (
            <div key={`${citation.source || 'source'}-${index}`} className="reference-chip">
              {citation.source || 'Internal source'}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

function ChatPanel({ onResult }) {
  const listRef = useRef(null)
  const [chatMessages, setChatMessages] = useState([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [latestMeta, setLatestMeta] = useState(null)

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [chatMessages, isSending])

  const sendMessage = async (presetText) => {
    const draft = typeof presetText === 'string' ? presetText : input
    const trimmed = draft.trim()

    if (!trimmed || isSending) {
      return
    }

    const userMsg = { role: 'user', content: trimmed }
    const nextConversation = [...chatMessages, userMsg]

    setChatMessages(nextConversation)
    setInput('')
    setIsSending(true)

    try {
      const response = await fetch(`${apiBaseUrl}/query`, {
        method: 'POST',
        headers: buildApiHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          query: trimmed,
          conversation_history: nextConversation.map((message) => ({
            role: message.role,
            content: message.content,
          })),
        }),
      })

      const data = await response.json()
      const assistantText =
        data.answer || data.error || 'No answer available.'
      const assistantMessage = {
        role: 'assistant',
        content: assistantText,
        meta: data,
      }

      setChatMessages((previous) => [...previous, assistantMessage])
      setLatestMeta(data)
      onResult?.(data)
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${String(error)}`,
      }

      setChatMessages((previous) => [...previous, errorMessage])
      setLatestMeta(null)
      onResult?.(null)
    } finally {
      setIsSending(false)
    }
  }

  const resetConversation = () => {
    setChatMessages([])
    setLatestMeta(null)
    onResult?.(null)
  }

  const citations = Array.isArray(latestMeta?.citations) ? latestMeta.citations : []
  const nodeTrace = Array.isArray(latestMeta?.node_trace) ? latestMeta.node_trace : []

  return (
    <section className="surface chat-panel">
      <div className="chat-panel__header">
        <div>
          <p className="eyebrow">Conversation Studio</p>
          <h2>Talk with your knowledge assistant</h2>
          <p className="panel-copy">
            Multi-turn chat with reliability signals and document-aware context.
          </p>
        </div>

        <button className="button button--secondary" onClick={resetConversation}>
          New session
        </button>
      </div>

      <div ref={listRef} className="message-list">
        {chatMessages.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state__badge">Ready</span>
            <h3>Start with a clear request.</h3>
            <p>
              Search your uploaded content, extract the key steps, or investigate
              a problem using the available context.
            </p>
            <div className="prompt-grid">
              {starterPrompts.map((prompt) => (
                <button
                  key={prompt}
                  className="prompt-chip"
                  onClick={() => setInput(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          chatMessages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`message-card message-card--${message.role}`}
            >
              <span className="message-card__role">
                {message.role === 'user' ? 'Operator' : 'Assistant'}
              </span>
              <p>{message.content}</p>
            </article>
          ))
        )}

        {isSending && (
          <article className="message-card message-card--assistant">
            <span className="message-card__role">Assistant</span>
            <p>I am analyzing the context, retrieval path, and best response...</p>
          </article>
        )}
      </div>

      <div className="composer">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              sendMessage()
            }
          }}
          placeholder="Ask a question about your documents, workflow, or knowledge base..."
          rows={1}
        />
        <button
          className="button button--primary composer__action"
          onClick={() => sendMessage()}
          disabled={isSending}
        >
          {isSending ? 'Sending...' : 'Send'}
        </button>
      </div>

      <div className="meta-strip">
        <div className="metric-pill">
          <span>Confidence</span>
          <strong>{formatPercentage(latestMeta?.confidence)}</strong>
        </div>
        <div className="metric-pill">
          <span>Hallucination risk</span>
          <strong>{formatPercentage(latestMeta?.hallucination_risk)}</strong>
        </div>
        <div className="metric-pill">
          <span>Query type</span>
          <strong>{latestMeta?.query_type || 'Waiting'}</strong>
        </div>
        <div className="metric-pill">
          <span>Escalation</span>
          <strong>{latestMeta?.should_escalate ? 'Suggested' : 'No'}</strong>
        </div>
      </div>

      {(nodeTrace.length > 0 || citations.length > 0) && (
        <div className="meta-panel">
          {nodeTrace.length > 0 && (
            <div>
              <span className="meta-panel__label">Node trace</span>
              <div className="trace-list">
                {nodeTrace.map((node) => (
                  <span key={node} className="trace-chip">
                    {node}
                  </span>
                ))}
              </div>
            </div>
          )}

          {citations.length > 0 && (
            <div>
              <span className="meta-panel__label">Citations</span>
              <div className="citation-list">
                {citations.slice(0, 4).map((citation, index) => (
                  <div
                    key={`${citation.source || 'citation'}-${index}`}
                    className="citation-card"
                  >
                    <strong>{citation.source || 'Internal source'}</strong>
                    <span>{citation.text || 'Excerpt not available.'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default function App() {
  const [lastUpload, setLastUpload] = useState(null)
  const [lastResult, setLastResult] = useState(null)
  const [kbFiles, setKbFiles] = useState([])

  const refreshKbFiles = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/files`, {
        headers: buildApiHeaders(),
      })
      if (response.ok) {
        const data = await response.json()
        setKbFiles(data.files || [])
      }
    } catch {
      // API not yet available, silently ignore
    }
  }

  useEffect(() => {
    refreshKbFiles()
  }, [])

  const handleUpload = (data) => {
    setLastUpload(data)
    refreshKbFiles()
  }

  const handleDeleteFile = async (name) => {
    if (!window.confirm(`Delete "${name}" from the knowledge base?\nThis will re-index all remaining documents.`)) return
    try {
      const response = await fetch(`${apiBaseUrl}/files/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        headers: buildApiHeaders(),
      })
      if (response.ok) {
        await refreshKbFiles()
      } else {
        const err = await response.json()
        window.alert(`Delete failed: ${err.detail || response.status}`)
      }
    } catch (error) {
      window.alert(`Delete failed: ${String(error)}`)
    }
  }

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand-lockup">
          <span className="brand-mark">CAG</span>
          <div>
            <p className="eyebrow">Knowledge Intelligence Desk</p>
            <h1>A more elegant control room for documents, research, and answers</h1>
          </div>
        </div>
      </header>

      <section className="hero surface">
        <div className="hero-copy">
          <p className="eyebrow">Editorial Ops Interface</p>
          <h2>
            Upload source material, query the CAG pipeline, and read answers in a
            workspace with stronger visual hierarchy.
          </h2>
          <p>
            The interface now follows a brighter editorial direction with warm
            paper tones, teal accents, expressive serif typography, and panels
            designed for long reading sessions.
          </p>
        </div>

        <div className="hero-side">
          <div className="signal-grid">
            {operatingSignals.map((signal) => (
              <div key={signal.label} className="signal-card">
                <span>{signal.label}</span>
                <strong>{signal.value}</strong>
              </div>
            ))}
          </div>

          <div className="hero-note">
            <span className="hero-note__label">Latest session</span>
            <p>
              {lastResult?.answer
                ? 'A response is available with telemetry and consulted sources.'
                : 'No query has been sent yet. The console is ready for a first test.'}
            </p>
          </div>
        </div>
      </section>

      <main className="layout">
        <aside className="sidebar-stack">
          <UploadPanel lastUpload={lastUpload} onUpload={handleUpload} />
          <IntelligencePanel lastResult={lastResult} kbFiles={kbFiles} onDeleteFile={handleDeleteFile} />
        </aside>

        <ChatPanel onResult={setLastResult} />
      </main>
    </div>
  )
}
