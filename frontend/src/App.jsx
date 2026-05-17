import React, { useEffect, useRef, useState } from 'react'

const starterPrompts = [
  'Summarize the latest uploaded document.',
  'What are the key steps described in this procedure?',
  'Help me troubleshoot an issue using the available knowledge base.',
]

const graphNodes = ['ENTRY', 'RETRIEVE', 'SELECT', 'REASON', 'POST', 'VALIDATE']

function normalizeTraceNode(node) {
  const value = String(node || '').toUpperCase()
  if (value.startsWith('SELECT_CONTEXT')) return 'SELECT'
  if (value.startsWith('REASON')) return 'REASON'
  if (value.startsWith('POST_GROUNDING')) return 'POST'
  if (value.startsWith('CONVERSATION_TRANSFORM')) return 'TRANSFORM'
  return value
}

function buildGraphFlow(result, progress) {
  if (progress?.status === 'running') {
    const activeIndex = Math.max(0, graphNodes.indexOf(progress.activeStage))
    return graphNodes.map((stage, index) => ({
      label: stage,
      state: index < activeIndex ? 'done' : index === activeIndex ? 'active' : 'pending',
    }))
  }

  const trace = Array.isArray(result?.node_trace)
    ? result.node_trace.map(normalizeTraceNode)
    : []
  const stages = trace.includes('TRANSFORM') ? ['TRANSFORM'] : graphNodes
  const visited = new Set(trace)
  const landedStage = [...trace].reverse().find((node) => stages.includes(node))

  return stages.map((stage) => {
    let state = 'pending'
    if (visited.has(stage)) {
      state = stage === landedStage ? 'active' : 'done'
    }
    return { label: stage, state }
  })
}

function getApiBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, '')
  }

  if (typeof window === 'undefined') {
    return 'http://localhost:8000'
  }

  const { protocol, hostname, port } = window.location
  // Vite dev server uses ports in the 5173-5179 range; proxy to the current FastAPI backend on 8010.
  const isViteDev = port >= '5173' && port <= '5179'
  if (isViteDev) {
    return `${protocol}//${hostname}:8010`
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

function percentageValue(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 0
  }
  return Math.max(0, Math.min(100, Math.round(value * 100)))
}

function getRunVerdict(result, progress) {
  if (progress?.status === 'running') {
    return {
      label: 'Running',
      tone: 'running',
      detail: 'CAG is moving through retrieval, selection, reasoning, and validation.',
    }
  }
  if (!result?.answer) {
    return {
      label: 'Ready',
      tone: 'ready',
      detail: 'Load evidence or ask a question to inspect the graph path.',
    }
  }
  if (result.should_escalate) {
    return {
      label: 'Escalation',
      tone: 'danger',
      detail: 'The answer needs human review or stronger source material.',
    }
  }
  if (result.query_type === 'CONVERSATION_TRANSFORM') {
    return {
      label: 'Transformed',
      tone: 'transform',
      detail: 'CAG handled the request as conversation work, without document retrieval.',
    }
  }
  return {
    label: 'Answered',
    tone: 'success',
    detail: 'The answer is available with telemetry and evidence inspection.',
  }
}

function basename(path) {
  if (!path) {
    return ''
  }

  return String(path).split(/[\\/]/).pop()
}

function isIngestActive(status) {
  return status?.status === 'queued' || status?.status === 'running'
}

function ingestPercent(status) {
  return Math.round(Math.max(0, Math.min(1, Number(status?.progress) || 0)) * 100)
}

function truncateText(text, limit = 180) {
  const value = String(text || '').replace(/\s+/g, ' ').trim()
  if (value.length <= limit) {
    return value
  }
  return `${value.slice(0, limit - 1)}...`
}

function normalizeChunks(chunks) {
  return Array.isArray(chunks) ? chunks : []
}

function RunConsole({ result, progress, graphFlow }) {
  const verdict = getRunVerdict(result, progress)
  const confidence = percentageValue(result?.confidence)
  const risk = percentageValue(result?.hallucination_risk)
  const trace = Array.isArray(result?.node_trace) ? result.node_trace : []
  const plan = result?.retrieval_plan && typeof result.retrieval_plan === 'object'
    ? result.retrieval_plan
    : null
  const intent = result?.intent && typeof result.intent === 'object'
    ? result.intent
    : null
  const activeStage = graphFlow.find((stage) => stage.state === 'active')?.label
    || graphFlow.filter((stage) => stage.state === 'done').at(-1)?.label
    || 'Waiting'

  return (
    <aside className="run-console" aria-label="Current run status">
      <div className="run-console__topline">
        <span className={`run-verdict run-verdict--${verdict.tone}`}>
          <span className="run-verdict__dot" aria-hidden="true" />
          {verdict.label}
        </span>
        <span className="run-console__api">{apiBaseUrl.replace(/^https?:\/\//, '')}</span>
      </div>

      <div>
        <span className="run-console__label">Current stage</span>
        <strong className="run-console__stage">{activeStage}</strong>
        <p>{verdict.detail}</p>
      </div>

      <div className="run-bars">
        <div className="run-bar">
          <div className="run-bar__label">
            <span>Confidence</span>
            <strong>{formatPercentage(result?.confidence)}</strong>
          </div>
          <span className="run-bar__track">
            <span className="run-bar__fill run-bar__fill--confidence" style={{ width: `${confidence}%` }} />
          </span>
        </div>
        <div className="run-bar">
          <div className="run-bar__label">
            <span>Risk</span>
            <strong>{formatPercentage(result?.hallucination_risk)}</strong>
          </div>
          <span className="run-bar__track">
            <span className="run-bar__fill run-bar__fill--risk" style={{ width: `${risk}%` }} />
          </span>
        </div>
      </div>

      <div className="run-console__facts">
        <div>
          <span>Type</span>
          <strong>{result?.query_type || 'GENERAL'}</strong>
        </div>
        <div>
          <span>Trace</span>
          <strong>{trace.length || 0} nodes</strong>
        </div>
        <div>
          <span>Sources</span>
          <strong>{Array.isArray(result?.citations) ? result.citations.length : 0}</strong>
        </div>
      </div>

      <div className="run-console__plan">
        <div>
          <span>Plan</span>
          <strong>{plan?.strategy || 'semantic'}</strong>
        </div>
        <div>
          <span>Grounding</span>
          <strong>{result?.post_grounding_status || 'pending'}</strong>
        </div>
        <div>
          <span>Scope</span>
          <strong>{intent?.question_scope || 'domain'}</strong>
        </div>
      </div>
    </aside>
  )
}

function UploadPanel({ lastUpload, ingestStatus, onUpload, onResetAll }) {
  const inputRef = useRef(null)
  const [uploadMsg, setUploadMsg] = useState('')
  const [accruedFiles, setAccruedFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isLoadingDemo, setIsLoadingDemo] = useState(false)
  const [isResetting, setIsResetting] = useState(false)

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
      setUploadMsg(`${data.saved?.length || 0} file(s) saved. Ingestion queued.`)
      setAccruedFiles([])
      onUpload?.(data)
    } catch (error) {
      setUploadMsg(`Upload failed: ${String(error)}`)
    } finally {
      setIsUploading(false)
    }
  }

  const handleDemoReset = async () => {
    if (isLoadingDemo) return

    setIsLoadingDemo(true)
    setUploadMsg('Loading demo corpus...')

    try {
      const response = await fetch(`${apiBaseUrl}/demo/reset?ingest=true`, {
        method: 'POST',
        headers: buildApiHeaders(),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`)
      }
      setUploadMsg(`${data.copied?.length || 0} demo file(s) loaded. Ingestion queued.`)
      onUpload?.({ saved: data.copied || [], ingest_started: data.ingest_started })
    } catch (error) {
      setUploadMsg(`Demo load failed: ${String(error)}`)
    } finally {
      setIsLoadingDemo(false)
    }
  }

  const handleResetAll = async () => {
    if (isResetting || isUploading || isLoadingDemo) return
    const confirmed = window.confirm(
      'Reset everything?\n\nThis deletes uploaded documents, document profiles, the knowledge graph, and vector index data.'
    )
    if (!confirmed) return

    setIsResetting(true)
    setUploadMsg('Resetting knowledge base...')

    try {
      const data = await onResetAll?.()
      setAccruedFiles([])
      setUploadMsg(
        `Workspace reset. Deleted ${data.deleted_files?.length || 0} file(s); knowledge ${
          data.knowledge_deleted ? 'cleared' : 'already empty'
        }.`
      )
    } catch (error) {
      setUploadMsg(`Reset failed: ${String(error)}`)
    } finally {
      setIsResetting(false)
    }
  }

  const savedFiles = Array.isArray(lastUpload?.saved) ? lastUpload.saved : []

  return (
    <section className="surface upload-panel">
      <div className="panel-heading">
        <p className="eyebrow">Intake</p>
        <h2>Source documents</h2>
        <p className="panel-copy">
          Add PDFs, markdown, or text files and rebuild the evidence index for this session.
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
        <span className="file-picker__button">Choose files</span>
        <span className="file-picker__meta">
          {accruedFiles.length > 0
            ? `${accruedFiles.length} file${accruedFiles.length > 1 ? 's' : ''} queued`
            : 'PDF, TXT, MD up to the API limits'}
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
                x
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
          : `Upload${accruedFiles.length > 0 ? ` ${accruedFiles.length}` : ''}`}
      </button>

      <button
        className="button button--secondary demo-loader"
        onClick={handleDemoReset}
        disabled={isLoadingDemo || isUploading || isResetting}
      >
        {isLoadingDemo ? 'Loading demo...' : 'Load demo'}
      </button>

      <button
        className="button button--danger reset-all"
        onClick={handleResetAll}
        disabled={isResetting || isUploading || isLoadingDemo}
      >
        {isResetting ? 'Resetting...' : 'Reset workspace'}
      </button>

      <div className="mini-stats">
        <div className="mini-stat">
          <span>Batch</span>
          <strong>{savedFiles.length || 0} files</strong>
        </div>
        <div className="mini-stat">
          <span>Pipeline</span>
          <strong>{isIngestActive(ingestStatus) ? 'Running' : ingestStatus?.status || 'Ready'}</strong>
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

function IntelligencePanel({ lastResult, kbFiles, documentProfiles, onDeleteFile }) {
  const citations = Array.isArray(lastResult?.citations) ? lastResult.citations : []
  const llmProfiles = documentProfiles.filter((profile) => profile.generator === 'llm').length

  return (
    <section className="surface intelligence-panel">
      <div className="panel-heading">
        <p className="eyebrow">Telemetry</p>
        <h2>Run signals</h2>
      </div>

      <div className="insight-grid">
        <div className="insight-card">
          <span>Profiles</span>
          <strong>{documentProfiles.length || kbFiles.length}</strong>
        </div>
        <div className="insight-card">
          <span>Confidence</span>
          <strong>{formatPercentage(lastResult?.confidence)}</strong>
        </div>
        <div className="insight-card">
          <span>Risk</span>
          <strong>{formatPercentage(lastResult?.hallucination_risk)}</strong>
        </div>
        <div className="insight-card">
          <span>Type</span>
          <strong>{lastResult?.query_type || 'Waiting'}</strong>
        </div>
      </div>

      <div className="document-dashboard">
        <div className="document-dashboard__header">
          <span className="kb-file-list__label">Document intelligence</span>
          <span>{llmProfiles}/{documentProfiles.length || 0} LLM</span>
        </div>
        {documentProfiles.length === 0 ? (
          <p className="evidence-muted">
            Upload documents to inspect summaries, topics, keywords, and covered intents.
          </p>
        ) : (
          <div className="document-profile-list">
            {documentProfiles.slice(0, 6).map((profile) => (
              <article key={profile.profile_id} className="document-profile-card">
                <div className="document-profile-card__topline">
                  <strong title={profile.filename}>{profile.filename}</strong>
                  <span>{profile.generator}</span>
                </div>
                <p>{truncateText(profile.summary, 180)}</p>
                <div className="document-profile-card__meta">
                  <span>{profile.chunk_count || 0} chunks</span>
                  {(profile.topics || []).slice(0, 3).map((topic) => (
                    <span key={topic}>{topic}</span>
                  ))}
                </div>
                {(profile.keywords || []).length > 0 && (
                  <div className="document-profile-card__keywords">
                    {(profile.keywords || []).slice(0, 5).map((keyword) => (
                      <span key={keyword}>{keyword}</span>
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>
        )}
      </div>

      {kbFiles.length > 0 && (
        <div className="kb-file-list">
          <span className="kb-file-list__label">Indexed files</span>
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
                Delete
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="insight-note">
        <span className="insight-note__label">Response status</span>
        <p>
          {lastResult?.should_escalate
            ? 'CAG recommends escalation because the evidence is not strong enough.'
            : 'Ready for grounded questions, trace inspection, and retrieval checks.'}
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

function IngestStatusBanner({ status }) {
  if (!status || status.status === 'idle') {
    return null
  }

  const isProblem = status.status === 'failed'
  const steps = Array.isArray(status.steps) ? status.steps : []
  const percent = ingestPercent(status)
  const stageLabel = String(status.stage || status.status || '').replace(/_/g, ' ')
  const counters = [
    ['Files', status.files_total],
    ['Docs', status.documents_loaded],
    ['Chunks', status.chunks_created || status.chunks_indexed],
    ['Claims', status.claims_created],
    ['Vectors', status.vectors_indexed],
  ]

  return (
    <div className={`ingest-banner ${isProblem ? 'ingest-banner--problem' : ''}`}>
      <div className="ingest-banner__header">
        <div>
          <strong>{isProblem ? 'Ingestion failed' : `Ingestion ${status.status}`}</strong>
          <span className="ingest-banner__stage">{stageLabel}</span>
        </div>
        <span className="ingest-banner__percent">{percent}%</span>
      </div>

      <div className="ingest-banner__progress" aria-hidden="true">
        <span style={{ width: `${percent}%` }} />
      </div>

      <div className="ingest-banner__meta">
        {counters.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{Number(value) || 0}</strong>
          </div>
        ))}
      </div>

      {steps.length > 0 && (
        <div className="ingest-banner__steps">
          {steps.map((step) => (
            <div key={step.id} className={`ingest-step ingest-step--${step.status}`}>
              <span className="ingest-step__dot" aria-hidden="true" />
              <span>{step.label}</span>
            </div>
          ))}
        </div>
      )}

      <span className="ingest-banner__message">
        {status.message || `${status.chunks_indexed || 0} chunks indexed`}
      </span>
    </div>
  )
}

function EvidencePanel({ result }) {
  const retrieved = normalizeChunks(result?.chunks)
  const selected = normalizeChunks(result?.ranked_chunks).slice(0, 6)
  const documentCandidates = Array.isArray(result?.document_candidates) ? result.document_candidates : []
  const gaps = Array.isArray(result?.gaps) ? result.gaps : []
  const hasEvidence = retrieved.length > 0 || selected.length > 0 || documentCandidates.length > 0 || gaps.length > 0

  return (
    <section className="evidence-panel">
      <div className="evidence-panel__header">
        <div>
          <span className="meta-panel__label">Evidence workbench</span>
          <h3>Selected evidence</h3>
        </div>
        <div className="evidence-counters">
          <span>{retrieved.length} retrieved</span>
          <span>{selected.length} selected</span>
          <span>{documentCandidates.length} docs</span>
          <span>{gaps.length} gaps</span>
        </div>
      </div>

      {!hasEvidence ? (
        <div className="evidence-empty">
          Ask a question to inspect retrieved chunks, selected context, and evidence gaps.
        </div>
      ) : (
        <div className="evidence-grid">
          <div className="evidence-column">
            <span className="evidence-column__label">Selected context</span>
            {selected.length === 0 ? (
              <p className="evidence-muted">No selected context yet.</p>
            ) : (
              selected.map((chunk, index) => (
                <article key={`${chunk.source || 'selected'}-${chunk.chunk_index ?? index}`} className="evidence-card">
                  <div className="evidence-card__topline">
                    <strong>{basename(chunk.source) || 'Unknown source'}</strong>
                    <span>{formatPercentage(chunk.relevance_score)}</span>
                  </div>
                  <p>{truncateText(chunk.content)}</p>
                  <div className="evidence-card__meta">
                    <span>{chunk.selection_category || 'general'}</span>
                    {chunk.compiled_knowledge && <span>compiled</span>}
                  </div>
                </article>
              ))
            )}
          </div>

          <div className="evidence-column">
            <span className="evidence-column__label">Document Map</span>
            {documentCandidates.length === 0 ? (
              <p className="evidence-muted">No document candidates reported.</p>
            ) : (
              <div className="document-candidate-list">
                {documentCandidates.slice(0, 5).map((candidate, index) => (
                  <article key={`${candidate.profile_id || candidate.filename || 'candidate'}-${index}`} className="document-candidate-card">
                    <div className="evidence-card__topline">
                      <strong>{basename(candidate.filename || candidate.source) || 'Unknown document'}</strong>
                      <span>{Number(candidate.score || 0).toFixed(1)}</span>
                    </div>
                    <p>{candidate.match_reason || truncateText(candidate.summary, 140)}</p>
                    <div className="evidence-card__meta">
                      <span>{candidate.generator || 'profile'}</span>
                      {(candidate.topics || []).slice(0, 2).map((topic) => (
                        <span key={topic}>{topic}</span>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            )}

            <span className="evidence-column__label">Gaps</span>
            {gaps.length === 0 ? (
              <p className="evidence-muted">No evidence gaps reported.</p>
            ) : (
              <ul className="gap-list">
                {gaps.slice(0, 6).map((gap, index) => (
                  <li key={`${gap}-${index}`}>{gap}</li>
                ))}
              </ul>
            )}

            <span className="evidence-column__label evidence-column__label--spaced">Retrieved pool</span>
            <div className="retrieved-list">
              {retrieved.slice(0, 6).map((chunk, index) => (
                <div key={`${chunk.source || 'retrieved'}-${chunk.chunk_index ?? index}`} className="retrieved-row">
                  <span>{basename(chunk.source) || 'Unknown source'}</span>
                  <small>#{chunk.chunk_index ?? index}</small>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

function ChatPanel({ onResult, onGraphProgress }) {
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
    onGraphProgress?.({ status: 'running', activeStage: 'ENTRY' })
    const progressTimers = [
      window.setTimeout(() => onGraphProgress?.({ status: 'running', activeStage: 'RETRIEVE' }), 350),
      window.setTimeout(() => onGraphProgress?.({ status: 'running', activeStage: 'SELECT' }), 950),
      window.setTimeout(() => onGraphProgress?.({ status: 'running', activeStage: 'REASON' }), 1650),
      window.setTimeout(() => onGraphProgress?.({ status: 'running', activeStage: 'POST' }), 2300),
      window.setTimeout(() => onGraphProgress?.({ status: 'running', activeStage: 'VALIDATE' }), 3000),
    ]

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
      progressTimers.forEach((timer) => window.clearTimeout(timer))
      onGraphProgress?.(null)
      setIsSending(false)
    }
  }

  const resetConversation = () => {
    setChatMessages([])
    setLatestMeta(null)
    onResult?.(null)
    onGraphProgress?.(null)
  }

  const citations = Array.isArray(latestMeta?.citations) ? latestMeta.citations : []
  const nodeTrace = Array.isArray(latestMeta?.node_trace) ? latestMeta.node_trace : []
  const suggestedActions = Array.isArray(latestMeta?.suggested_actions) ? latestMeta.suggested_actions : []
  const groundingChecks = Array.isArray(latestMeta?.grounding_checks) ? latestMeta.grounding_checks : []
  const retrievalPlan = latestMeta?.retrieval_plan && typeof latestMeta.retrieval_plan === 'object'
    ? latestMeta.retrieval_plan
    : null

  return (
    <section className="surface chat-panel">
      <div className="chat-panel__header">
        <div>
          <p className="eyebrow">Query</p>
          <h2>Ask CAG</h2>
          <p className="panel-copy">
            Send questions through the graph and inspect the answer, confidence, risk, and sources.
          </p>
        </div>

        <button className="button button--secondary" onClick={resetConversation}>
          Clear
        </button>
      </div>

      <div ref={listRef} className="message-list">
        {chatMessages.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state__badge">Ready</span>
            <h3>Start with a grounded request.</h3>
            <p>
              Search uploaded content, extract procedures, or investigate a problem using the indexed evidence.
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
                {message.role === 'user' ? 'You' : 'CAG'}
              </span>
              <p>{message.content}</p>
            </article>
          ))
        )}

        {isSending && (
          <article className="message-card message-card--assistant message-card--thinking">
            <span className="message-card__role">CAG</span>
            <p>Analyzing query type, retrieval path, selected context, and validation state...</p>
            <span className="thinking-meter" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
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
          placeholder="Ask about your documents, procedures, incidents, or configuration..."
          rows={1}
        />
        <button
          className="button button--primary composer__action"
          onClick={() => sendMessage()}
          disabled={isSending}
        >
          {isSending ? 'Running...' : 'Run'}
        </button>
      </div>

      <div className="meta-strip">
        <div className="metric-pill">
          <span>Confidence</span>
          <strong>{formatPercentage(latestMeta?.confidence)}</strong>
        </div>
        <div className="metric-pill">
          <span>Risk</span>
          <strong>{formatPercentage(latestMeta?.hallucination_risk)}</strong>
        </div>
        <div className="metric-pill">
          <span>Type</span>
          <strong>{latestMeta?.query_type || 'Waiting'}</strong>
        </div>
        <div className="metric-pill">
          <span>Escalate</span>
          <strong>{latestMeta?.should_escalate ? 'Suggested' : 'No'}</strong>
        </div>
        <div className="metric-pill">
          <span>Grounding</span>
          <strong>{latestMeta?.post_grounding_status || 'Waiting'}</strong>
        </div>
        <div className="metric-pill">
          <span>Plan</span>
          <strong>{retrievalPlan?.strategy || 'Waiting'}</strong>
        </div>
      </div>

      {suggestedActions.length > 0 && (
        <div className="action-strip" aria-label="Suggested next actions">
          {suggestedActions.map((action) => (
            <button
              key={action.id || action.label}
              className="action-chip"
              type="button"
              title={action.reason || ''}
              onClick={() => {
                if (action.prompt) {
                  setInput(action.prompt)
                }
              }}
            >
              <span>{action.type || 'action'}</span>
              {action.label}
            </button>
          ))}
        </div>
      )}

      {(nodeTrace.length > 0 || citations.length > 0 || retrievalPlan || groundingChecks.length > 0) && (
        <div className="meta-panel">
          {retrievalPlan && (
            <div>
              <span className="meta-panel__label">Modified prompt and retrieval plan</span>
              <div className="plan-grid">
                <div>
                  <span>Modified</span>
                  <strong>{latestMeta?.modified_query || latestMeta?.original_query || 'Not available'}</strong>
                </div>
                <div>
                  <span>Sources</span>
                  <strong>{(retrievalPlan.sources || []).join(', ') || 'document index'}</strong>
                </div>
                <div>
                  <span>Variants</span>
                  <strong>{(retrievalPlan.query_variants || []).length || 0}</strong>
                </div>
                <div>
                  <span>Access</span>
                  <strong>{retrievalPlan.access_filter_applied ? 'Filtered' : 'Open'}</strong>
                </div>
              </div>
            </div>
          )}

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

          {groundingChecks.length > 0 && (
            <div>
              <span className="meta-panel__label">Post grounding</span>
              <div className="grounding-list">
                {groundingChecks.slice(0, 3).map((check, index) => (
                  <div
                    key={`${check.claim || 'claim'}-${index}`}
                    className={`grounding-row ${check.supported ? 'grounding-row--supported' : 'grounding-row--weak'}`}
                  >
                    <span>{check.supported ? 'Supported' : 'Weak'}</span>
                    <strong>{truncateText(check.claim, 140)}</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <EvidencePanel result={latestMeta} />
    </section>
  )
}

export default function App() {
  const [lastUpload, setLastUpload] = useState(null)
  const [lastResult, setLastResult] = useState(null)
  const [kbFiles, setKbFiles] = useState([])
  const [documentProfiles, setDocumentProfiles] = useState([])
  const [ingestStatus, setIngestStatus] = useState(null)
  const [graphProgress, setGraphProgress] = useState(null)
  const graphFlow = buildGraphFlow(lastResult, graphProgress)

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

  const refreshIngestStatus = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/ingest/status`, {
        headers: buildApiHeaders(),
      })
      if (response.ok) {
        setIngestStatus(await response.json())
      }
    } catch {
      // API not yet available, silently ignore
    }
  }

  const refreshDocumentProfiles = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/document-profiles`, {
        headers: buildApiHeaders(),
      })
      if (response.ok) {
        const data = await response.json()
        setDocumentProfiles(data.profiles || [])
      }
    } catch {
      // API not yet available, silently ignore
    }
  }

  useEffect(() => {
    refreshKbFiles()
    refreshIngestStatus()
    refreshDocumentProfiles()
  }, [])

  useEffect(() => {
    if (!isIngestActive(ingestStatus)) {
      return undefined
    }

    const intervalId = window.setInterval(() => {
      refreshIngestStatus()
      refreshKbFiles()
      refreshDocumentProfiles()
    }, 1200)

    return () => window.clearInterval(intervalId)
  }, [ingestStatus?.status])

  const handleUpload = (data) => {
    setLastUpload(data)
    refreshKbFiles()
    refreshIngestStatus()
    refreshDocumentProfiles()
    window.setTimeout(refreshIngestStatus, 1500)
    window.setTimeout(refreshDocumentProfiles, 2500)
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
        await refreshDocumentProfiles()
      } else {
        const err = await response.json()
        window.alert(`Delete failed: ${err.detail || response.status}`)
      }
    } catch (error) {
      window.alert(`Delete failed: ${String(error)}`)
    }
  }

  const handleResetAll = async () => {
    const response = await fetch(`${apiBaseUrl}/reset/all`, {
      method: 'DELETE',
      headers: buildApiHeaders(),
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`)
    }
    setLastUpload(null)
    setLastResult(null)
    setKbFiles([])
    setDocumentProfiles([])
    setIngestStatus({ status: 'idle', stage: 'idle', message: '', chunks_indexed: 0 })
    return data
  }

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand-lockup">
          <span className="brand-mark">CAG</span>
          <div>
            <p className="eyebrow">Cognitive Augmented Generation</p>
            <h1>Evidence workbench</h1>
          </div>
        </div>
        <div className="masthead-actions">
          <span className="status-light" aria-hidden="true" />
          <span>Local preview</span>
        </div>
      </header>

      <section className="workspace-summary">
        <div className="workspace-summary__copy">
          <p className="eyebrow">Graph-driven document QA</p>
          <h2>Load evidence, run a query, verify the reasoning path.</h2>
          <p>
            CAG makes retrieval decisions explicit: query type, selected context,
            validation, sources, and escalation state stay visible while you test.
          </p>
          <div className="graph-rail" aria-label="CAG graph stages">
            {graphFlow.map((node, index) => (
              <React.Fragment key={node.label}>
                <span className={`graph-node graph-node--${node.state}`}>
                  <span className="graph-node__status" aria-hidden="true" />
                  {node.label}
                </span>
                {index < graphFlow.length - 1 && (
                  <span
                    className={`graph-link graph-link--${node.state === 'done' ? 'done' : 'pending'}`}
                    aria-hidden="true"
                  />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        <RunConsole result={lastResult} progress={graphProgress} graphFlow={graphFlow} />
      </section>

      <main className="layout">
        <aside className="sidebar-stack">
          <IngestStatusBanner status={ingestStatus} />
          <UploadPanel
            lastUpload={lastUpload}
            ingestStatus={ingestStatus}
            onUpload={handleUpload}
            onResetAll={handleResetAll}
          />
          <IntelligencePanel
            lastResult={lastResult}
            kbFiles={kbFiles}
            documentProfiles={documentProfiles}
            onDeleteFile={handleDeleteFile}
          />
        </aside>

        <ChatPanel onResult={setLastResult} onGraphProgress={setGraphProgress} />
      </main>
    </div>
  )
}
