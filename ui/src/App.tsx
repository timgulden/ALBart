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
  const [moodDescriptors, setMoodDescriptors] = useState<string[]>([])
  const [moodApplied, setMoodApplied] = useState(false)
  const [moodThreshold, setMoodThreshold] = useState(0.35)
  const [volume, setVolume] = useState(50)
  const [devices, setDevices] = useState<{id:string,name:string,type:string,is_active:boolean,volume:number}[]>([])
  const [error, setError] = useState('')

  // Poll status
  useEffect(() => {
    let active = true
    const poll = async () => {
      while (active) {
        try {
          const res = await fetch(`${API}/status`)
          if (active && res.ok) {
            const data: Status = await res.json()
            setStatus(data)
          }
        } catch { /* server not running */ }
        await new Promise(r => setTimeout(r, 2000))
      }
    }
    poll()
    return () => { active = false }
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

  const interpretMood = useCallback(async () => {
    if (!mood.trim()) return
    setMoodPending(true)
    setMoodApplied(false)
    setError('')
    try {
      const res = await fetch(`${API}/interpret`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mood }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else if (data.descriptors) {
        setMoodDescriptors(data.descriptors)
      }
    } catch {
      setError('Interpret failed — is ANTHROPIC_API_KEY set?')
    }
    setMoodPending(false)
  }, [mood])

  const applyMood = useCallback(async () => {
    setError('')
    try {
      const res = await fetch(`${API}/mood`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ descriptors: moodDescriptors }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else {
        setMoodApplied(true)
      }
    } catch {
      setError('Apply mood failed — is the DJ running?')
    }
  }, [moodDescriptors])

  const updateMoodThreshold = useCallback(async (t: number) => {
    setMoodThreshold(t)
    await fetch(`${API}/mood_threshold`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold: t }),
    })
  }, [])

  // Fetch devices periodically
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API}/devices`)
        if (res.ok) {
          const data = await res.json()
          setDevices(data)
          const active = data.find((d: any) => d.is_active)
          if (active) setVolume(active.volume)
        }
      } catch { /* ignore */ }
    }, 5000)
    return () => clearInterval(poll)
  }, [])

  const updateVolume = useCallback(async (v: number) => {
    setVolume(v)
    await fetch(`${API}/volume`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volume: v }),
    })
  }, [])

  const switchDevice = useCallback(async (deviceId: string) => {
    await fetch(`${API}/device/${deviceId}`, { method: 'POST' })
  }, [])

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
      maxWidth: 960, margin: '30px auto',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      color: '#e0e0e0', padding: '24px',
    }}>
      <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 24, color: '#fff' }}>
        ALBart DJ
      </h1>

      <div style={{ display: 'flex', gap: 24 }}>
        {/* ── Left column: Controls + Status ── */}
        <div style={{ flex: 1 }}>
          {/* Now Playing */}
          {status?.current_track && (
            <Card accent>
              <div style={{ fontSize: 13, color: '#888', marginBottom: 4 }}>Now Playing</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: '#fff' }}>
                {status.current_track.title}
              </div>
              <div style={{ fontSize: 17, color: '#aaa', marginTop: 4 }}>
                {status.current_track.artist}
              </div>
              <div style={{ fontSize: 14, color: '#666', marginTop: 10 }}>
                {status.played_count} / {status.total_tracks} played
              </div>
            </Card>
          )}

          {/* Start / Stop / Skip */}
          <Card>
            <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
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

            <Slider
              label="Next Song Temperature"
              value={songK} min={1} max={50} step={1}
              leftLabel="1 (nearest only)"
              rightLabel="50 (wide exploration)"
              onChange={updateSongK}
            />

            <Slider
              label="Next Set Temperature"
              value={setDist} min={1} max={20} step={0.5}
              leftLabel="1× (adjacent)"
              rightLabel="20× (big jump)"
              onChange={updateSetDist}
            />
          </Card>

          {/* Volume & Device */}
          <Card>
            <Slider
              label="Volume"
              value={volume} min={0} max={100} step={1}
              leftLabel="0%" rightLabel="100%"
              onChange={updateVolume}
            />
            {devices.length > 0 && (
              <div>
                <Label>Output Device</Label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {devices.map(d => (
                    <button
                      key={d.id}
                      onClick={() => switchDevice(d.id)}
                      style={{
                        padding: '8px 12px', borderRadius: 8, border: 'none',
                        background: d.is_active ? '#2a3f5f' : '#0f1729',
                        color: d.is_active ? '#8ab4f8' : '#888',
                        fontSize: 14, cursor: 'pointer', textAlign: 'left',
                        borderLeft: d.is_active ? '3px solid #4a6cf7' : '3px solid transparent',
                      }}
                    >
                      {d.name} <span style={{ fontSize: 11, color: '#555' }}>({d.type})</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
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
              <div style={{ marginTop: 10 }}>
                {searchResults.map(t => (
                  <div key={t.track_id} style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '8px 0', borderBottom: '1px solid #222', fontSize: 15,
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
          {status?.history && status.history.length > 1 && (
            <Card>
              <Label>Recent History</Label>
              {status.history.slice(1, 12).map((t, i) => (
                <div key={`${t.track_id}-${i}`} style={{
                  padding: '4px 0', fontSize: 15, color: '#999',
                  borderBottom: '1px solid #1a1a2e',
                }}>
                  {t.title} — {t.artist}
                </div>
              ))}
            </Card>
          )}
        </div>

        {/* ── Right column: Mood ── */}
        <div style={{ flex: '0 0 340px' }}>
          <Card>
            <Label>Mood</Label>
            <textarea
              rows={3}
              placeholder="chill dinner party, jazz, downtempo, no opera..."
              value={mood} onChange={e => { setMood(e.target.value); setMoodApplied(false) }}
              style={{ ...inputStyle, width: '100%', resize: 'vertical' }}
            />
            <button
              onClick={interpretMood} disabled={moodPending}
              style={{
                ...btnStyle(moodPending ? '#555' : '#4a6cf7'),
                marginTop: 10, width: '100%',
              }}
            >
              {moodPending ? 'Interpreting...' : 'Interpret'}
            </button>

            {/* Claude's interpretation */}
            <div style={{
              marginTop: 14, padding: 14,
              background: '#0f1729', borderRadius: 8,
              border: '1px solid #2a2f45',
              minHeight: 60,
            }}>
              {moodDescriptors.length > 0 ? (
                <>
                  <div style={{ fontSize: 13, color: '#667', marginBottom: 6 }}>
                    Descriptors (editable):
                  </div>
                  <textarea
                    rows={8}
                    value={moodDescriptors.join('\n')}
                    onChange={e => {
                      setMoodDescriptors(e.target.value.split('\n'))
                      setMoodApplied(false)
                    }}
                    style={{
                      ...inputStyle, width: '100%', fontSize: 13,
                      lineHeight: 1.6, resize: 'vertical',
                    }}
                  />
                </>
              ) : (
                <div style={{ fontSize: 14, color: '#444', fontStyle: 'italic' }}>
                  Click Interpret to see how Claude breaks down your mood description
                </div>
              )}
            </div>

            <button
              onClick={applyMood}
              disabled={moodDescriptors.length === 0 || moodApplied}
              style={{
                ...btnStyle(
                  moodDescriptors.length === 0 ? '#333'
                    : moodApplied ? '#555'
                    : '#2ecc71'
                ),
                marginTop: 10, width: '100%',
              }}
            >
              {moodApplied ? 'Applied ✓' : 'Apply Mood'}
            </button>

            <div style={{ marginTop: 16 }}>
              <Slider
                label="Mood Strictness"
                value={moodThreshold} min={0.15} max={0.55} step={0.05}
                leftLabel="loose (most tracks)"
                rightLabel="strict (few tracks)"
                onChange={updateMoodThreshold}
              />
            </div>
          </Card>
        </div>
      </div>

      {error && (
        <div style={{
          marginTop: 16, padding: '12px 18px', borderRadius: 8,
          background: '#e74c3c22', color: '#e74c3c', fontSize: 15,
        }}>
          {error}
        </div>
      )}
    </div>
  )
}

// ── Components ──────────────────────────────────────────────────────────

function Card({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <div style={{
      background: '#16213e', borderRadius: 12, padding: 22, marginBottom: 18,
      borderLeft: accent ? '4px solid #4a6cf7' : 'none',
    }}>
      {children}
    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 15, color: '#888', marginBottom: 8 }}>{children}</div>
}

function Slider({ label, value, min, max, step, leftLabel, rightLabel, onChange }: {
  label: string; value: number; min: number; max: number; step: number
  leftLabel: string; rightLabel: string; onChange: (v: number) => void
}) {
  // Position the value indicator over the slider thumb
  const pct = ((value - min) / (max - min)) * 100
  const displayVal = Number.isInteger(step) ? String(value) : value.toFixed(1)

  return (
    <div style={{ marginBottom: 18 }}>
      <Label>{label}</Label>
      <div style={{ position: 'relative', marginBottom: 4 }}>
        <input
          type="range" min={min} max={max} step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{ width: '100%', accentColor: '#4a6cf7' }}
        />
        {/* Value bubble centered over the thumb */}
        <div style={{
          position: 'absolute',
          top: -22,
          left: `calc(${pct}% - 16px)`,
          fontSize: 14,
          fontWeight: 700,
          color: '#fff',
          background: '#4a6cf7',
          borderRadius: 4,
          padding: '1px 8px',
          pointerEvents: 'none',
        }}>
          {displayVal}
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#555' }}>
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  )
}

// ── Styles ───────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = {
  padding: '10px 14px', borderRadius: 8,
  border: '1px solid #333', background: '#0f1729',
  color: '#fff', fontSize: 16,
}

function btnStyle(color: string): React.CSSProperties {
  return {
    padding: '10px 20px', borderRadius: 8, border: 'none',
    background: color, color: '#fff', fontSize: 16,
    fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
  }
}

function smallBtn(color: string): React.CSSProperties {
  return {
    padding: '5px 12px', borderRadius: 6, border: 'none',
    background: color, color: '#fff', fontSize: 13,
    fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
  }
}

export default App
