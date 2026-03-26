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
  temperature: number
  mood: string | null
  played_count: number
  total_tracks: number
  history: TrackInfo[]
}

function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [mood, setMood] = useState('')
  const [temperature, setTemperature] = useState(0.5)
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
        const data = await res.json()
        setStatus(data)
      } catch {
        // server not running
      }
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
          temperature,
          mood: mood || undefined,
        }),
      })
    } catch {
      setError('Could not connect to DJ server')
    }
  }, [seed, temperature, mood])

  const stopSession = useCallback(async () => {
    await fetch(`${API}/stop`, { method: 'POST' })
  }, [])

  const skip = useCallback(async () => {
    await fetch(`${API}/skip`, { method: 'POST' })
  }, [])

  const updateTemperature = useCallback(async (t: number) => {
    setTemperature(t)
    await fetch(`${API}/temperature`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ temperature: t }),
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

  const playTrack = useCallback(async (trackId: string) => {
    await fetch(`${API}/play/${trackId}`, { method: 'POST' })
    setSearchQuery('')
    setSearchResults([])
  }, [])

  // Search
  useEffect(() => {
    if (searchQuery.length < 2) {
      setSearchResults([])
      return
    }
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/search?q=${encodeURIComponent(searchQuery)}`)
        setSearchResults(await res.json())
      } catch { /* ignore */ }
    }, 300)
    return () => clearTimeout(timer)
  }, [searchQuery])

  const isPlaying = status?.playing ?? false

  return (
    <div style={{
      maxWidth: 520, margin: '40px auto',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      color: '#e0e0e0', background: '#1a1a2e',
      minHeight: '100vh', padding: '24px',
    }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8, color: '#fff' }}>
        ALBart DJ
      </h1>

      {/* Now Playing */}
      {status?.current_track && (
        <div style={{
          background: '#16213e', borderRadius: 12, padding: 20,
          marginBottom: 20, borderLeft: '4px solid #4a6cf7',
        }}>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>Now Playing</div>
          <div style={{ fontSize: 20, fontWeight: 600, color: '#fff' }}>
            {status.current_track.title}
          </div>
          <div style={{ fontSize: 14, color: '#aaa', marginTop: 2 }}>
            {status.current_track.artist}
          </div>
          <div style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
            {status.played_count} / {status.total_tracks} played
          </div>
        </div>
      )}

      {/* Controls */}
      <div style={{ background: '#16213e', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          {!isPlaying ? (
            <>
              <input
                type="text" placeholder="Seed track (optional)"
                value={seed} onChange={e => setSeed(e.target.value)}
                style={inputStyle}
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

        {/* Temperature */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <label style={{ fontSize: 13, color: '#888' }}>Temperature</label>
            <span style={{ fontSize: 13, color: '#aaa' }}>{temperature.toFixed(2)}</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.05}
            value={temperature}
            onChange={e => updateTemperature(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: '#4a6cf7' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#555' }}>
            <span>deterministic</span>
            <span>exploratory</span>
          </div>
        </div>

        {/* Mood */}
        <div>
          <label style={{ fontSize: 13, color: '#888', display: 'block', marginBottom: 4 }}>Mood</label>
          <div style={{ display: 'flex', gap: 8 }}>
            <textarea
              rows={2}
              placeholder="chill dinner party, jazz, downtempo, no opera..."
              value={mood} onChange={e => setMood(e.target.value)}
              style={{ ...inputStyle, resize: 'vertical' as const }}
            />
            <button
              onClick={updateMood} disabled={moodPending}
              style={btnStyle(moodPending ? '#555' : '#2ecc71')}
            >
              {moodPending ? '...' : 'Apply'}
            </button>
          </div>
        </div>
      </div>

      {/* Search */}
      <div style={{ background: '#16213e', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <label style={{ fontSize: 13, color: '#888', display: 'block', marginBottom: 4 }}>
          Search & Play
        </label>
        <input
          type="text" placeholder="Search tracks..."
          value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
          style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
        />
        {searchResults.length > 0 && (
          <div style={{ marginTop: 8 }}>
            {searchResults.map(t => (
              <div
                key={t.track_id} onClick={() => playTrack(t.track_id)}
                style={{
                  padding: '6px 0', cursor: 'pointer',
                  borderBottom: '1px solid #222', fontSize: 13,
                }}
              >
                {t.title} — <span style={{ color: '#888' }}>{t.artist}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* History */}
      {status && status.history.length > 1 && (
        <div style={{ background: '#16213e', borderRadius: 12, padding: 20 }}>
          <div style={{ fontSize: 13, color: '#888', marginBottom: 8 }}>Recent History</div>
          {status.history.slice(1, 15).map((t, i) => (
            <div key={`${t.track_id}-${i}`} style={{
              padding: '4px 0', fontSize: 13, color: '#999',
              borderBottom: i < 13 ? '1px solid #1a1a2e' : 'none',
            }}>
              {t.title} — {t.artist}
            </div>
          ))}
        </div>
      )}

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

const inputStyle: React.CSSProperties = {
  flex: 1, padding: '8px 12px', borderRadius: 8,
  border: '1px solid #333', background: '#0f1729',
  color: '#fff', fontSize: 14,
}

function btnStyle(color: string): React.CSSProperties {
  return {
    padding: '8px 16px', borderRadius: 8, border: 'none',
    background: color, color: '#fff', fontSize: 14,
    fontWeight: 600, cursor: 'pointer',
  }
}

export default App
