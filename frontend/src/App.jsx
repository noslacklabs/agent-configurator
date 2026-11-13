import { useCallback, useEffect, useState } from 'react'

// Served by the hand-off service in production (same origin). In `npm run dev` the page is on 5175.
const API = import.meta.env.VITE_HANDOFF_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8003' : '')

async function request(path, options) {
  let response
  try {
    response = await fetch(`${API}${path}`, options)
  } catch {
    throw new Error('Could not reach the hand-off service. Is it running on port 8003?')
  }
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`)
  return body
}

function formatDate(value) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString()
}

function ServiceStatus({ health }) {
  if (!health) return null
  const items = [
    ['Langflow', health.langflow],
    ['Agent Evaluator', health.evaluator],
    ['LLM Configurator', health.llm_configurator],
  ]
  return (
    <ul className="services" aria-label="Service status">
      {items.map(([label, state]) => (
        <li key={label} className={state.ok ? 'up' : 'down'} title={state.error || state.url}>
          <span className="dot" aria-hidden="true" />
          {label}: {state.ok ? 'running' : 'not reachable'}
        </li>
      ))}
      {!health.langflow_api_key_set && <li className="down"><span className="dot" aria-hidden="true" />Langflow API key missing in .env</li>}
    </ul>
  )
}

function Result({ result }) {
  return (
    <div className="result" role="status">
      <p className="result-title">
        {result.all_ok ? 'Registered in the evaluator.' : 'Registered, but a connection test failed.'}
      </p>
      <ul>
        {result.registered.map((item) => (
          <li key={item.agent_id} className={item.test.ok === false ? 'failed' : 'passed'}>
            <strong>{item.name}</strong>
            <span className="tag">{item.scope === 'agent' ? 'one agent' : 'whole flow'}</span>
            <span className="tag">{item.created ? 'new' : 'already registered'}</span>
            <span className="outcome">{item.test.ok === false ? `Test failed: ${item.test.message}` : 'Connection test passed'}</span>
          </li>
        ))}
      </ul>
      <p>
        <a className="button primary" href={result.evaluator_url} target="_blank" rel="noreferrer">Open the evaluator</a>
        <span className="hint">Pick the agent by name in step 1 of the wizard.</span>
      </p>
    </div>
  )
}

function FlowCard({ flow, onRegistered }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [includeAgents, setIncludeAgents] = useState(true)
  const orchestrator = flow.orchestrator

  const evaluate = async () => {
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const body = await request(`/api/flows/${encodeURIComponent(flow.id)}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_agents: includeAgents }),
      })
      setResult(body)
      onRegistered()
    } catch (problem) {
      setError(problem.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className="flow">
      <header>
        <div>
          <h2>{flow.name}</h2>
          <p className="meta">
            {formatDate(flow.updated_at) && <>Updated {formatDate(flow.updated_at)} · </>}
            <a href={flow.langflow_url} target="_blank" rel="noreferrer">Open in Langflow</a>
          </p>
        </div>
        <button className="button primary" onClick={evaluate} disabled={busy}>
          {busy ? 'Registering and testing…' : 'Evaluate this agent'}
        </button>
      </header>

      {flow.description && <p className="description">{flow.description}</p>}

      <dl>
        {orchestrator && (<><dt>Pattern</dt><dd>{orchestrator.pattern}{orchestrator.name && orchestrator.name !== orchestrator.pattern ? ` (${orchestrator.name})` : ''}</dd></>)}
        {flow.agents.length > 0 && (
          <>
            <dt>{flow.agents.length === 1 ? 'Agent' : 'Agents'}</dt>
            <dd>{flow.agents.map((agent) => <span className="chip" key={agent.id} title={agent.role}>{agent.name}</span>)}</dd>
          </>
        )}
        {flow.llm_configs.length > 0 && (<><dt>LLM config</dt><dd>{flow.llm_configs.join(', ')}</dd></>)}
        {flow.registered.length > 0 && (
          <>
            <dt>In the evaluator</dt>
            <dd>{flow.registered.map((item) => <span className="chip registered" key={item.agent_id}>{item.name}</span>)}</dd>
          </>
        )}
      </dl>

      {!flow.uses_coworkai_blocks && <p className="note">This flow has no CoworkAI blocks. It can still be evaluated as a whole.</p>}
      {flow.problems.length > 0 && <p className="warning">Check before evaluating: {flow.problems.join('; ')}.</p>}

      {orchestrator && orchestrator.agents.length > 0 && (
        <label className="option">
          <input type="checkbox" checked={includeAgents} onChange={(event) => setIncludeAgents(event.target.checked)} />
          Also register each agent on its own ({orchestrator.agents.join(', ')}), so it can be scored separately
        </label>
      )}

      {error && <p className="error" role="alert">{error}</p>}
      {result && <Result result={result} />}
    </article>
  )
}

export default function App() {
  const [listing, setListing] = useState(null)
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    request('/api/health').then(setHealth).catch(() => setHealth(null))
    try {
      setListing(await request('/api/flows'))
      setError('')
    } catch (problem) {
      setError(problem.message)
    } finally {
      setLoading(false)
    }
  }, [])

  // Refresh the "In the evaluator" badges without clearing the card's result.
  const refreshQuietly = useCallback(() => {
    request('/api/flows').then(setListing).catch(() => {})
  }, [])

  useEffect(() => { load() }, [load])

  const flows = listing?.flows || []

  return (
    <main>
      <header className="page-header">
        <div>
          <h1>Agent Configurator</h1>
          <p>Send a flow you built in Langflow to the Agent Evaluator.</p>
        </div>
        <button className="button" onClick={load} disabled={loading}>{loading ? 'Loading…' : 'Refresh'}</button>
      </header>

      <ServiceStatus health={health} />

      {error && <p className="error" role="alert">{error}</p>}
      {listing?.evaluator_error && <p className="warning">{listing.evaluator_error}</p>}

      {!loading && !error && flows.length === 0 && (
        <p className="empty">No flows yet. Build one in Langflow, or import a starter template from the templates folder, then refresh.</p>
      )}

      <section className="flows">
        {flows.map((flow) => <FlowCard key={flow.id} flow={flow} onRegistered={refreshQuietly} />)}
      </section>
    </main>
  )
}
