import { useState, useEffect, useCallback } from 'react'

const API = 'http://127.0.0.1:8765/api'

interface TrackInfo {
  track_id: string
  title: string
  artist: string
}

interface Status {
  playing: boolean
  current_track: TrackInfo | null
  song_k: number
  set_distance: number
  mood_text: string | null
  mood_descriptors: string[]
  played_count: number
  total_tracks: number
  history: TrackInfo[]
}

function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [mood, setMood] = useState('')
  const [songK, setSongK] = useState(10)
  const [setDist, setSetDist] = useState(5)
  const [seed, setSeed] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<TrackInfo[]>([])
  const [moodPending, setMoodPending] = useState(false)
  const [error, setError] = useState('')

  // Poll status
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API}/status`)
        const data: Status = await res.json()
        setStatus(data)
      } catch { /* server not running */ }
    }, 2000)
    return () => clearInterval(poll)
  }, [])

  const startSession = useCallback(async () => {
    setError('')
    try {
      await fetch(`${API}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seed: seed || undefined,
          song_k: songK,
          set_distance: setDist,
          mood: mood || undefined,
        }),
      })
    } catch {
      setError('Could not connect to DJ server')
    }
  }, [seed, songK, setDist, mood])

  const stopSession = useCallback(async () => {
    await fetch(`${API}/stop`, { method: 'POST' })
  }, [])

  const skip = useCallback(async () => {
    await fetch(`${API}/skip`, { method: 'POST' })
  }, [])

  const updateSongK = useCallback(async (k: number) => {
    setSongK(k)
    await fetch(`${API}/song_k`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ song_k: k }),
    })
  }, [])

  const updateSetDist = useCallback(async (d: number) => {
    setSetDist(d)
    await fetch(`${API}/set_distance`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ set_distance: d }),
    })
  }, [])

  const updateMood = useCallback(async () => {
    if (!mood.trim()) return
    setMoodPending(true)
    setError('')
    try {
      const res = await fetch(`${API}/mood`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mood }),
      })
      const data = await res.json()
      if (data.error) setError(data.error)
    } catch {
      setError('Mood update failed')
    }
    setMoodPending(false)
  }, [mood])

  const playNow = useCallback(async (trackId: string) => {
    await fetch(`${API}/play/${trackId}`, { method: 'POST' })
    setSearchQuery('')
    setSearchResults([])
  }, [])

  const queueNext = useCallback(async (trackId: string) => {
    await fetch(`${API}/queue/${trackId}`, { method: 'POST' })
    setSearchQuery('')
    setSearchResults([])
  }, [])

  // Search on Enter
  const doSearch = useCallback(async () => {
    if (searchQuery.length < 2) return
    try {
      const res = await fetch(`${API}/search?q=${encodeURIComponent(searchQuery)}`)
      setSearchResults(await res.json())
    } catch { /* ignore */ }
  }, [searchQuery])

  const isPlaying = status?.playing ?? false

  return (
    <div style={{
      maxWidth: 900, margin: '30px auto',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      color: '#e0e0e0', padding: '24px',
    }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 20, color: '#fff' }}>
        ALBart DJ
      </h1>

      <div style={{ display: 'flex', gap: 20 }}>
        {/* ── Left column: Mood ── */}
        <div style={{ flex: '0 0 320px' }}>
          <Card>
            <Label>Mood</Label>
            <textarea
              rows={3}
              placeholder="chill dinner party, jazz, downtempo, no opera..."
              value={mood} onChange={e => setMood(e.target.value)}
              style={{ ...inputStyle, width: '100%', resize: 'vertical' }}
            />
            <button
              onClick={updateMood} disabled={moodPending}
              style={{ ...btnStyle(moodPending ? '#555' : '#2ecc71'), marginTop: 8, width: '100%' }}
            >
              {moodPending ? 'Processing...' : 'Apply Mood'}
            </button>

            {/* Mood descriptors from Claude */}
            {status && status.mood_descriptors.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <Label>Active Descriptors</Label>
                <div style={{
                  fontSize: 12, color: '#8890a8', lineHeight: 1.8,
                  padding: '8px 0',
                }}>
                  {status.mood_descriptors.map((d, i) => (
                    <span key={i} style={{
                      display: 'inline-block', background: d.toUpperCase().startsWith('NOT:') ? '#3d1f1f' : '#1f2d3d',
                      borderRadius: 4, padding: '2px 8px', margin: '2px 4px 2px 0',
                      color: d.toUpperCase().startsWith('NOT:') ? '#e88' : '#8ab4f8',
                      fontSize: 11,
                    }}>
                      {d}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>

        {/* ── Right column: Controls + Status ── */}
        <div style={{ flex: 1 }}>
          {/* Now Playing */}
          {status?.current_track && (
            <Card accent>
              <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>Now Playing</div>
              <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
                {status.current_track.title}
              </div>
              <div style={{ fontSize: 14, color: '#aaa', marginTop: 2 }}>
                {status.current_track.artist}
              </div>
              <div style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
                {status.played_count} / {status.total_tracks} played
              </div>
            </Card>
          )}

          {/* Start / Stop / Skip */}
          <Card>
            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
              {!isPlaying ? (
                <>
                  <input
                    type="text" placeholder="Seed track (optional)"
                    value={seed} onChange={e => setSeed(e.target.value)}
                    style={{ ...inputStyle, flex: 1 }}
                  />
                  <button onClick={startSession} style={btnStyle('#4a6cf7')}>Start</button>
                </>
              ) : (
                <>
                  <button onClick={skip} style={btnStyle('#4a6cf7')}>Skip</button>
                  <button onClick={stopSession} style={btnStyle('#e74c3c')}>Stop</button>
                </>
              )}
            </div>

            {/* Next Song Temperature */}
            <Slider
              label="Next Song Temperature"
              value={songK} min={1} max={50} step={1}
              leftLabel="1 (nearest only)"
              rightLabel="50 (wide exploration)"
              onChange={updateSongK}
            />

            {/* Next Set Temperature */}
            <Slider
              label="Next Set Temperature"
              value={setDist} min={1} max={20} step={0.5}
              leftLabel="1× (adjacent)"
              rightLabel="20× (big jump)"
              onChange={updateSetDist}
            />
          </Card>

          {/* Search */}
          <Card>
            <Label>Search & Play</Label>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                type="text" placeholder="Search tracks..."
                value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && doSearch()}
                style={{ ...inputStyle, flex: 1 }}
              />
              <button onClick={doSearch} style={btnStyle('#4a6cf7')}>Search</button>
            </div>
            {searchResults.length > 0 && (
              <div style={{ marginTop: 8 }}>
                {searchResults.map(t => (
                  <div key={t.track_id} style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '6px 0', borderBottom: '1px solid #222', fontSize: 13,
                  }}>
                    <span style={{ flex: 1 }}>
                      {t.title} — <span style={{ color: '#888' }}>{t.artist}</span>
                    </span>
                    <button onClick={() => queueNext(t.track_id)}
                      style={smallBtn('#2ecc71')}>Next</button>
                    <button onClick={() => playNow(t.track_id)}
                      style={smallBtn('#4a6cf7')}>Now</button>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* History */}
          {status && status.history.length > 1 && (
            <Card>
              <Label>Recent History</Label>
              {status.history.slice(1, 12).map((t, i) => (
                <div key={`${t.track_id}-${i}`} style={{
                  padding: '3px 0', fontSize: 13, color: '#999',
                  borderBottom: '1px solid #1a1a2e',
                }}>
                  {t.title} — {t.artist}
                </div>
              ))}
            </Card>
          )}
        </div>
      </div>

      {error && (
        <div style={{
          marginTop: 16, padding: '10px 16px', borderRadius: 8,
          background: '#e74c3c22', color: '#e74c3c', fontSize: 13,
        }}>
          {error}
        </div>
      )}
    </div>
  )
}

// ── Reusable components ─────────────────────────────────────────────────

function Card({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <div style={{
      background: '#16213e', borderRadius: 12, padding: 20, marginBottom: 16,
      borderLeft: accent ? '4px solid #4a6cf7' : 'none',
    }}>
      {children}
    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>{children}</div>
}

function Slider({ label, value, min, max, step, leftLabel, rightLabel, onChange }: {
  label: string; value: number; min: number; max: number; step: number
  leftLabel: string; rightLabel: string; onChange: (v: number) => void
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <Label>{label}</Label>
        <span style={{ fontSize: 13, color: '#aaa', fontWeight: 600 }}>
          {Number.isInteger(step) ? value : value.toFixed(1)}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: '#4a6cf7' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#555' }}>
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  )
}

// ── Styles ───────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = {
  padding: '8px 12px', borderRadius: 8,
  border: '1px solid #333', background: '#0f1729',
  color: '#fff', fontSize: 14,
}

function btnStyle(color: string): React.CSSProperties {
  return {
    padding: '8px 16px', borderRadius: 8, border: 'none',
    background: color, color: '#fff', fontSize: 14,
    fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
  }
}

function smallBtn(color: string): React.CSSProperties {
  return {
    padding: '3px 10px', borderRadius: 6, border: 'none',
    background: color, color: '#fff', fontSize: 11,
    fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
  }
}

export default App
