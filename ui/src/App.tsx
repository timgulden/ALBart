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
  progress_ms: number
  duration_ms: number
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
  const [progress, setProgress] = useState(0)   // ms, ticks locally
  const [duration, setDuration] = useState(0)    // ms, from status poll
  const [lastPollTime, setLastPollTime] = useState(0)
  const [seeking, setSeeking] = useState(false)  // true while user drags
  const [seekPos, setSeekPos] = useState(0)      // position while dragging
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
            setProgress(data.progress_ms)
            setDuration(data.duration_ms)
            setLastPollTime(Date.now())
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

  // Local progress ticker (pauses while user is dragging)
  useEffect(() => {
    if (!status?.playing || duration <= 0 || seeking) return
    const tick = setInterval(() => {
      const elapsed = Date.now() - lastPollTime
      const estimated = (status?.progress_ms ?? 0) + elapsed
      setProgress(Math.min(estimated, duration))
    }, 250)
    return () => clearInterval(tick)
  }, [status?.playing, status?.progress_ms, duration, lastPollTime, seeking])

  const onSeekStart = useCallback(() => {
    setSeeking(true)
  }, [])

  const onSeekMove = useCallback((ms: number) => {
    setSeekPos(ms)
  }, [])

  const onSeekEnd = useCallback(async () => {
    // Set progress and poll baseline BEFORE resuming ticker
    // so it ticks from the new position, not the old one
    setProgress(seekPos)
    setLastPollTime(Date.now())
    if (status) {
      status.progress_ms = seekPos
    }
    setSeeking(false)
    await fetch(`${API}/seek`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position_ms: Math.round(seekPos) }),
    })
  }, [seekPos, status])

  // No device polling — volume is set locally, not synced from Spotify
  // (avoids extra API calls that contribute to rate limiting)

  const updateVolume = useCallback(async (v: number) => {
    setVolume(v)
    await fetch(`${API}/volume`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volume: v }),
    })
  }, [])

  const newSet = useCallback(async () => {
    await fetch(`${API}/new_set`, { method: 'POST' })
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
          {/* Now Playing + controls */}
          <Card accent>
            {status?.current_track ? (
              <>
                <div style={{ fontSize: 13, color: '#888', marginBottom: 4 }}>Now Playing</div>
                <div style={{ fontSize: 24, fontWeight: 600, color: '#fff' }}>
                  {status.current_track.title}
                </div>
                <div style={{ fontSize: 17, color: '#aaa', marginTop: 4 }}>
                  {status.current_track.artist}
                </div>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8, marginTop: 12,
                }}>
                  <span style={{ fontSize: 14, color: '#666' }}>
                    {status.played_count} / {status.total_tracks} played
                  </span>
                  <span style={{ flex: 1 }} />
                  {isPlaying && (
                    <>
                      <button onClick={skip} style={btnStyle('#4a6cf7')}>Skip</button>
                      <button onClick={stopSession} style={btnStyle('#e74c3c')}>Stop</button>
                    </>
                  )}
                </div>
                {isPlaying && (
                  <div style={{ marginTop: 6, textAlign: 'right' }}>
                    <button onClick={newSet} style={btnStyle('#7c3aed')}>New Set</button>
                  </div>
                )}
                {/* Progress bar */}
                {duration > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ position: 'relative', height: 20 }}>
                      <input
                        type="range" min={0} max={duration} step={1000}
                        value={seeking ? seekPos : progress}
                        onMouseDown={onSeekStart}
                        onTouchStart={onSeekStart}
                        onChange={e => onSeekMove(parseFloat(e.target.value))}
                        onMouseUp={onSeekEnd}
                        onTouchEnd={onSeekEnd}
                        style={{
                          position: 'absolute', top: 0, left: 0,
                          width: '100%', height: '100%',
                          opacity: 0, cursor: 'pointer', zIndex: 2,
                        }}
                      />
                      <div style={{
                        position: 'absolute', top: 6, left: 0, right: 0, height: 6,
                        background: '#0f1729', borderRadius: 3,
                      }}>
                        <div style={{
                          height: '100%', borderRadius: 3, background: '#4a6cf7',
                          width: `${((seeking ? seekPos : progress) / duration) * 100}%`,
                          transition: seeking ? 'none' : 'width 0.25s linear',
                        }} />
                      </div>
                    </div>
                    <div style={{
                      display: 'flex', justifyContent: 'space-between',
                      fontSize: 12, color: '#555', marginTop: 2,
                    }}>
                      <span>{formatTime(seeking ? seekPos : progress)}</span>
                      <span>{formatTime(duration)}</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  type="text" placeholder="Seed track (optional)"
                  value={seed} onChange={e => setSeed(e.target.value)}
                  style={{ ...inputStyle, flex: 1 }}
                />
                <button onClick={startSession} style={btnStyle('#4a6cf7')}>Start</button>
              </div>
            )}</Card>

          {/* Volume */}
          <Card>
            <Slider
              label="Volume"
              value={volume} min={0} max={100} step={1}
              leftLabel="0%" rightLabel="100%"
              onChange={updateVolume}
            />
          </Card>

          {/* Search & Play */}
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
              <Label>Session History</Label>
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                {status.history.slice(1).map((t, i) => (
                  <div key={`${t.track_id}-${i}`}>
                    {t.set_start && (
                      <div style={{
                        padding: '6px 0', fontSize: 12, color: '#4a6cf7',
                        fontWeight: 600, letterSpacing: 1,
                        borderTop: i > 0 ? '1px solid #2a3050' : 'none',
                        marginTop: i > 0 ? 6 : 0,
                      }}>
                        ── NEW SET ──
                      </div>
                    )}
                    <div style={{
                      padding: '3px 0', fontSize: 14, color: '#999',
                    }}>
                      {t.title} — {t.artist}
                    </div>
                  </div>
                ))}
              </div>
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
                value={moodThreshold} min={0} max={0.60} step={0.05}
                leftLabel="loose (most tracks)"
                rightLabel="strict (few tracks)"
                onChange={updateMoodThreshold}
              />
              {status && status.total_tracks > 0 && (
                <div style={{ fontSize: 13, color: '#888', textAlign: 'center', marginTop: -8 }}>
                  {status.mood_in_count}/{status.total_tracks} tracks in-mood
                  ({Math.round(100 * status.mood_in_count / status.total_tracks)}%)
                </div>
              )}
            </div>
          </Card>

          {/* Temperature */}
          <Card>
            <Label>Temperature</Label>
            <Slider
              label="Next Song"
              value={songK} min={1} max={50} step={1}
              leftLabel="1 (nearest only)"
              rightLabel="50 (wide exploration)"
              onChange={updateSongK}
            />
            <Slider
              label="Next Set"
              value={setDist} min={1} max={20} step={0.5}
              leftLabel="1× (adjacent)"
              rightLabel="20× (big jump)"
              onChange={updateSetDist}
            />
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
  const displayVal = Number.isInteger(step) ? String(value) : value.toFixed(2)

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
        <Label>{label}</Label>
      </div>
      <div style={{ position: 'relative', height: 32 }}>
        {/* Invisible range input covers the full area for dragging */}
        <input
          type="range" min={min} max={max} step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%', height: '100%',
            opacity: 0, cursor: 'grab', zIndex: 2,
          }}
        />
        {/* Visual track bar */}
        <div style={{
          position: 'absolute', top: 12, left: 0, right: 0, height: 8,
          background: '#0f1729', borderRadius: 4,
        }}>
          <div style={{
            height: '100%', borderRadius: 4, background: '#4a6cf7',
            width: `${((value - min) / (max - min)) * 100}%`,
          }} />
        </div>
        {/* Value label as the draggable thumb */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: `calc(${((value - min) / (max - min)) * 100}% - 20px)`,
          fontSize: 14, fontWeight: 700,
          color: '#fff', background: '#4a6cf7',
          borderRadius: 6, padding: '2px 10px',
          pointerEvents: 'none', zIndex: 1,
          minWidth: 24, textAlign: 'center',
        }}>
          {displayVal}
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#555', marginTop: 2 }}>
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  )
}

function formatTime(ms: number): string {
  const s = Math.floor(ms / 1000)
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}:${sec.toString().padStart(2, '0')}`
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
