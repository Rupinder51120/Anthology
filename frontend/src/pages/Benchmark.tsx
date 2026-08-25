import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Activity, Play, CheckCircle, AlertCircle, Clock, Zap, Target, BarChart2, BookOpen } from 'lucide-react'
import { Button, Spinner } from '../components/ui'
import {
  getBenchmarkScores,
  runBenchmarkEval as runEval,
  getBenchmarkStatus as getStatus,
  getBenchmarkResults as getResults,
} from '../api/client'

// ── Metric Card ───────────────────────────────────────────────

function MetricCard({ label, value, sub, color = '#7c5cfc', icon: Icon }: {
  label: string; value: string | number; sub?: string
  color?: string; icon?: any
}) {
  return (
    <div style={{
      background: 'var(--color-surface)', border: '1px solid var(--color-border)',
      borderRadius: 12, padding: '16px 18px', borderTop: `3px solid ${color}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        {Icon && <Icon size={14} color={color} />}
        <span style={{ fontSize: 11, color: 'var(--color-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--color-muted)', marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

// ── Score Bar ─────────────────────────────────────────────────

function ScoreBar({ label, value, max = 1, color = '#7c5cfc' }: {
  label: string; value: number; max?: number; color?: string
}) {
  const pct = Math.min((value / max) * 100, 100)
  const display = max === 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(4)
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 12, color: 'var(--color-text)', fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 12, color, fontWeight: 700 }}>{display}</span>
      </div>
      <div style={{ height: 6, background: 'var(--color-border)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 3,
          background: `linear-gradient(90deg, ${color}aa, ${color})`,
          transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────

export default function BenchmarkPage() {
  const [sampleSize, setSampleSize] = useState(20)
  const [useJudge, setUseJudge] = useState(false)
  const [polling, setPolling] = useState(false)
  const qc = useQueryClient()

  const { data: scores, isLoading } = useQuery({
    queryKey: ['benchmark-scores'],
    queryFn: getBenchmarkScores,
  })

  const { data: status } = useQuery({
    queryKey: ['benchmark-status'],
    queryFn: getStatus,
    refetchInterval: polling ? 2000 : false,
  })

  const { data: resultsData } = useQuery({
    queryKey: ['benchmark-results'],
    queryFn: () => getResults(30),
  })

  // Stop polling when job finishes
  useEffect(() => {
    if (status?.status === 'done' || status?.status === 'error') {
      setPolling(false)
      qc.invalidateQueries({ queryKey: ['benchmark-scores'] })
      qc.invalidateQueries({ queryKey: ['benchmark-results'] })
    }
  }, [status?.status])

  const runMutation = useMutation({
    mutationFn: () => runEval(sampleSize, useJudge),
    onSuccess: () => setPolling(true),
  })

  const isRunning = status?.status === 'running' || status?.status === 'starting'

  // Latest eval run
  const runs = scores?.runs || {}
  const runKeys = Object.keys(runs).sort().reverse()
  const latest = runKeys.length > 0 ? runs[runKeys[0]] : null
  const retrieval = scores?.quick_retrieval || latest?.retrieval || {}
  const judge = latest?.judge || {}

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1000, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: 'linear-gradient(135deg, #7c5cfc22, #06b6d422)', border: '1px solid var(--color-accent)33', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <BarChart2 size={18} color="var(--color-accent)" />
            </div>
            <div style={{ fontWeight: 800, fontSize: 20, letterSpacing: '-0.02em' }}>RAG Evaluation</div>
          </div>
          <div style={{ fontSize: 13, color: 'var(--color-muted)' }}>
            {scores?.qa_count ? `${scores.qa_count} QA pairs · ${scores.result_count || 0} evaluated` : 'No QA dataset yet'}
          </div>
        </div>

        {/* Run controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 10, padding: '6px 12px' }}>
            <span style={{ fontSize: 12, color: 'var(--color-muted)' }}>Sample:</span>
            <select
              value={sampleSize}
              onChange={e => setSampleSize(Number(e.target.value))}
              style={{ background: 'transparent', border: 'none', color: 'var(--color-text)', fontSize: 12, outline: 'none', cursor: 'pointer' }}
            >
              {[10, 20, 50, 100].map(n => <option key={n} value={n}>{n} Q</option>)}
            </select>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--color-muted)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={useJudge}
              onChange={e => setUseJudge(e.target.checked)}
              style={{ accentColor: 'var(--color-accent)' }}
            />
            LLM judge
          </label>

          <Button
            onClick={() => runMutation.mutate()}
            disabled={isRunning || runMutation.isPending}
          >
            {isRunning ? <><Spinner size={13} /> Running…</> : <><Play size={13} /> Run Eval</>}
          </Button>
        </div>
      </div>

      {/* Job status banner */}
      {isRunning && (
        <div style={{
          background: 'var(--color-accent)11', border: '1px solid var(--color-accent)33',
          borderRadius: 10, padding: '12px 16px', marginBottom: 20,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <Spinner size={14} />
          <span style={{ fontSize: 13, color: 'var(--color-accent)', fontWeight: 600 }}>
            Evaluating… {status?.progress || 0} / {status?.total || sampleSize} questions
            {status?.elapsed_s ? ` · ${status.elapsed_s}s` : ''}
          </span>
          <div style={{ flex: 1, height: 4, background: 'var(--color-border)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${Math.round(((status?.progress || 0) / (status?.total || 1)) * 100)}%`,
              background: 'var(--color-accent)', borderRadius: 2, transition: 'width 0.3s',
            }} />
          </div>
        </div>
      )}

      {status?.status === 'error' && (
        <div style={{ background: '#ef444411', border: '1px solid #ef444433', borderRadius: 10, padding: '12px 16px', marginBottom: 20, display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <AlertCircle size={14} color="#ef4444" style={{ flexShrink: 0, marginTop: 1 }} />
          <span style={{ fontSize: 13, color: '#ef4444' }}>{status.error}</span>
        </div>
      )}

      {status?.status === 'done' && (
        <div style={{ background: '#22c55e11', border: '1px solid #22c55e33', borderRadius: 10, padding: '12px 16px', marginBottom: 20, display: 'flex', gap: 8, alignItems: 'center' }}>
          <CheckCircle size={14} color="#22c55e" />
          <span style={{ fontSize: 13, color: '#22c55e', fontWeight: 600 }}>Eval complete — scores updated</span>
        </div>
      )}

      {/* Top metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 24 }}>
        <MetricCard
          label="Hit@5"
          value={retrieval['hit@5'] != null ? `${(retrieval['hit@5'] * 100).toFixed(1)}%` : '—'}
          sub="Source paper in top-5"
          color="#7c5cfc"
          icon={Target}
        />
        <MetricCard
          label="MRR"
          value={retrieval.mrr != null ? retrieval.mrr.toFixed(3) : '—'}
          sub="Mean Reciprocal Rank"
          color="#06b6d4"
          icon={Activity}
        />
        <MetricCard
          label="nDCG@5"
          value={retrieval['ndcg@5'] != null ? retrieval['ndcg@5'].toFixed(3) : '—'}
          sub="Ranked retrieval quality"
          color="#22c55e"
          icon={BarChart2}
        />
        <MetricCard
          label="Faithfulness"
          value={judge.faithfulness != null ? `${(judge.faithfulness / 5 * 100).toFixed(0)}%` : '—'}
          sub={judge.faithfulness != null ? `${judge.faithfulness.toFixed(2)}/5 — grounded in context` : 'Run with LLM judge'}
          color="#f59e0b"
          icon={CheckCircle}
        />
      </div>

      {/* Two-panel: retrieval + judge */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Retrieval metrics */}
        <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 14, padding: '20px 22px' }}>
          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Target size={14} color="#7c5cfc" /> Retrieval Metrics
          </div>
          {Object.keys(retrieval).length === 0 ? (
            <div style={{ color: 'var(--color-muted)', fontSize: 13 }}>No retrieval data yet. Run an eval.</div>
          ) : (
            <>
              <ScoreBar label="Hit@1" value={retrieval['hit@1'] || 0} color="#7c5cfc" />
              <ScoreBar label="Hit@3" value={retrieval['hit@3'] || 0} color="#7c5cfc" />
              <ScoreBar label="Hit@5" value={retrieval['hit@5'] || 0} color="#7c5cfc" />
              <ScoreBar label="MRR" value={retrieval.mrr || 0} color="#06b6d4" />
              <ScoreBar label="nDCG@5" value={retrieval['ndcg@5'] || 0} color="#22c55e" />
              <div style={{ fontSize: 11, color: 'var(--color-muted)', marginTop: 8 }}>
                n = {retrieval.n_eval || '?'} questions
              </div>
            </>
          )}
        </div>

        {/* Judge metrics */}
        <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 14, padding: '20px 22px' }}>
          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Zap size={14} color="#f59e0b" /> Generation Metrics
            <span style={{ fontSize: 10, color: 'var(--color-muted)', fontWeight: 400 }}>(qwen2.5:7b judge)</span>
          </div>
          {Object.keys(judge).length === 0 ? (
            <div style={{ color: 'var(--color-muted)', fontSize: 13 }}>
              Enable "LLM judge" and re-run to see generation quality scores.
              <div style={{ marginTop: 8, fontSize: 11 }}>Measures: faithfulness, relevance, completeness</div>
            </div>
          ) : (
            <>
              <ScoreBar label="Faithfulness" value={judge.faithfulness / 5} color="#f59e0b" />
              <ScoreBar label="Relevance" value={judge.relevance / 5} color="#f59e0b" />
              <ScoreBar label="Completeness" value={judge.completeness / 5} color="#f59e0b" />
              <div style={{ marginTop: 16, padding: '12px 14px', background: 'var(--color-surface2)', borderRadius: 10, border: '1px solid var(--color-border)' }}>
                <div style={{ fontSize: 11, color: 'var(--color-muted)', marginBottom: 4 }}>MEAN SCORE</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#f59e0b' }}>{(judge.mean_score / 5 * 100).toFixed(1)}%</div>
              </div>
              <div style={{ fontSize: 11, color: 'var(--color-muted)', marginTop: 8 }}>
                n = {judge.n_eval} · failed = {judge.n_failed || 0}
              </div>
            </>
          )}
        </div>
      </div>

      {/* History table */}
      {runKeys.length > 1 && (
        <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 14, padding: '20px 22px', marginBottom: 24 }}>
          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 14 }}>Eval History</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                {['Run', 'Hit@1', 'Hit@5', 'MRR', 'nDCG@5', 'Faithfulness', 'Mean'].map(h => (
                  <th key={h} style={{ padding: '6px 10px', textAlign: h === 'Run' ? 'left' : 'right', color: 'var(--color-muted)', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runKeys.map(key => {
                const r = runs[key]
                const ret = r.retrieval || {}
                const j = r.judge || {}
                return (
                  <tr key={key} style={{ borderBottom: '1px solid var(--color-border)' }}>
                    <td style={{ padding: '8px 10px', color: 'var(--color-muted)', fontSize: 11 }}>{key}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right' }}>{ret['hit@1'] != null ? (ret['hit@1'] * 100).toFixed(1) + '%' : '—'}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right' }}>{ret['hit@5'] != null ? (ret['hit@5'] * 100).toFixed(1) + '%' : '—'}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right' }}>{ret.mrr?.toFixed(3) ?? '—'}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right' }}>{ret['ndcg@5']?.toFixed(3) ?? '—'}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right' }}>{j.faithfulness != null ? (j.faithfulness / 5 * 100).toFixed(0) + '%' : '—'}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 700, color: 'var(--color-accent)' }}>
                      {r.mean_score != null ? (r.mean_score / (j.faithfulness != null ? 5 : 1) * 100).toFixed(1) + '%' : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Per-question results */}
      {resultsData?.results?.length > 0 && (
        <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 14, padding: '20px 22px' }}>
          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
            <BookOpen size={14} color="var(--color-accent)" /> Per-Question Results
            <span style={{ fontSize: 11, color: 'var(--color-muted)', fontWeight: 400 }}>({resultsData.total} total)</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {resultsData.results.map((r: any, i: number) => {
              const hit = r.sources?.some((s: string) => {
                const stem = (p: string) => p.split('/').pop()?.replace('.pdf', '') || p
                return stem(s) === stem(r.source_chunk || '')
              })
              return (
                <div key={i} style={{
                  background: 'var(--color-surface2)', border: '1px solid var(--color-border)',
                  borderLeft: `3px solid ${r.answer === 'ERROR' ? '#ef4444' : hit ? '#22c55e' : '#f59e0b'}`,
                  borderRadius: 8, padding: '10px 14px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-text)', flex: 1 }}>{r.question}</div>
                    <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                      {r.elapsed_s && (
                        <span style={{ fontSize: 10, color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
                          <Clock size={9} />{r.elapsed_s}s
                        </span>
                      )}
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 20,
                        background: r.answer === 'ERROR' ? '#ef444422' : hit ? '#22c55e22' : '#f59e0b22',
                        color: r.answer === 'ERROR' ? '#ef4444' : hit ? '#22c55e' : '#f59e0b',
                      }}>
                        {r.answer === 'ERROR' ? 'ERROR' : hit ? '✓ HIT' : '✗ MISS'}
                      </span>
                    </div>
                  </div>
                  {r.answer !== 'ERROR' && (
                    <div style={{ fontSize: 11, color: 'var(--color-muted)', marginTop: 5, lineHeight: 1.5 }}>
                      {r.answer?.slice(0, 150)}{r.answer?.length > 150 ? '…' : ''}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && Object.keys(retrieval).length === 0 && !isRunning && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--color-muted)' }}>
          <BarChart2 size={40} color="var(--color-border)" style={{ margin: '0 auto 16px' }} />
          <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>No eval data yet</div>
          <div style={{ fontSize: 13 }}>
            {scores?.qa_count ? `${scores.qa_count} QA pairs ready — click Run Eval above` : 'Build a QA dataset first with benchmarker.py'}
          </div>
        </div>
      )}
    </div>
  )
}
