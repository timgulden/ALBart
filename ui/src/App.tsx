import { useState, useEffect, useCallback, useRef } from 'react'

const API = 'http://127.0.0.1:8765/api'

interface TrackInfo {
  track_id: string
  title: string
  artist: string
  set_start?: string | null  // "NEW SET", "DWELL", "TRANSIT", or null
}

interface OrbitAnchorInfo {
  description: string
  track_id: string
  title: string
  artist: string
  art_url: string
  active: boolean
}

interface OrbitProgress {
  phase: string  // "dwell" or "transit"
  current_index: number
  prev_index: number
  segment_progress: number
  completed_segments: number[]  // from-anchor indices of completed segments
  dwell_elapsed: number
  dwell_duration: number
  transit_remaining: number
  transit_total: number
}

interface Status {
  playing: boolean
  current_track: TrackInfo | null
  progress_ms: number
  duration_ms: number
  song_k: number
  set_distance: number
  mode: string
  dj_active: boolean
  mood_text: string | null
  mood_descriptors: string[]
  mood_threshold: number
  mood_in_count: number
  played_count: number
  total_tracks: number
  history: TrackInfo[]
  volume: number
  orbit_active: boolean
  orbit_anchors: OrbitAnchorInfo[]
  orbit_progress: OrbitProgress | null
}

interface ProcessStatus {
  running: boolean
  pid: number | null
}

interface SystemStatus {
  mapview: ProcessStatus
  listener: ProcessStatus
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
  const volumeSentAt = useRef(0)      // timestamp: suppress poll volume until elapsed
  const volumeSentVal = useRef(0)    // the volume value we sent (suppress poll snap-back)
  const seekSentAt = useRef(0)        // timestamp: suppress poll progress until elapsed
  const seekSentPos = useRef(0)       // the position we seeked to
  const [error, setError] = useState('')
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [orbitViewerOpen, setOrbitViewerOpen] = useState(false)
  const [orbitJourney, setOrbitJourney] = useState('')
  const [orbitDescriptions, setOrbitDescriptions] = useState<string[]>([])
  const [orbitPending, setOrbitPending] = useState(false)
  const [orbitApplied, setOrbitApplied] = useState(false)
  const [orbitAllowSameArtist, setOrbitAllowSameArtist] = useState(false)

  // Poll status
  useEffect(() => {
    let active = true
    const poll = async () => {
      while (active) {
        try {
          const [statusRes, systemRes] = await Promise.all([
            fetch(`${API}/status`),
            fetch(`${API}/system`),
          ])
          if (active && statusRes.ok) {
            const data: Status = await statusRes.json()
            const seekLocked = Date.now() <= seekSentAt.current + 5000
            if (seekLocked) {
              // Override progress_ms in the status object so the ticker
              // doesn't snap back to the pre-seek position.
              data.progress_ms = seekSentPos.current
            }
            setStatus(data)
            if (!seekLocked) {
              setProgress(data.progress_ms)
              setLastPollTime(Date.now())
            }
            setDuration(data.duration_ms)
            if (data.volume >= 0) {
              // Suppress poll volume for 5s after a user drag to prevent
              // snap-back from stale Spotify values. External changes
              // (e.g. phone) will sync once the lock expires.
              if (Date.now() > volumeSentAt.current + 5000) {
                setVolume(data.volume)
              }
            }
            // Sync local mood/orbit "applied" flags with server
            // (clears UI indicators after a server restart, but only
            // if the user had previously applied — don't wipe descriptors
            // the user is still reviewing)
            if (data.mood_descriptors.length === 0 && moodApplied) {
              setMoodApplied(false)
            }
            if (!data.orbit_active && orbitApplied) {
              setOrbitApplied(false)
            }
          }
          if (active && systemRes.ok) {
            setSystem(await systemRes.json())
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
      const seekLocked = Date.now() <= seekSentAt.current + 5000
      const base = seekLocked ? seekSentPos.current : (status?.progress_ms ?? 0)
      const baseTime = seekLocked ? seekSentAt.current : lastPollTime
      const elapsed = Date.now() - baseTime
      setProgress(Math.min(base + elapsed, duration))
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
    seekSentAt.current = Date.now()
    seekSentPos.current = seekPos
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
    volumeSentAt.current = Date.now()
    volumeSentVal.current = v
    await fetch(`${API}/volume`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volume: v }),
    })
  }, [])

  const newSet = useCallback(async () => {
    await fetch(`${API}/new_set`, { method: 'POST' })
  }, [])

  const toggleMode = useCallback(async () => {
    const newMode = status?.mode === 'roomear' ? 'exact' : 'roomear'
    await fetch(`${API}/mode`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode }),
    })
  }, [status?.mode])

  const toggleMapView = useCallback(async () => {
    const running = system?.mapview.running
    await fetch(`${API}/mapview/${running ? 'stop' : 'start'}`, { method: 'POST' })
  }, [system])

  const toggleDjActive = useCallback(async () => {
    const newActive = !(status?.dj_active ?? true)
    await fetch(`${API}/dj_active`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dj_active: newActive }),
    })
  }, [status?.dj_active])

  const interpretOrbit = useCallback(async () => {
    if (!orbitJourney.trim()) return
    setOrbitPending(true)
    setOrbitApplied(false)
    setError('')
    try {
      const res = await fetch(`${API}/orbit/interpret`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: orbitJourney }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else if (data.descriptions) {
        setOrbitDescriptions(data.descriptions)
        setOrbitAllowSameArtist(data.allow_same_artist ?? false)
      }
    } catch {
      setError('Interpret failed — is ANTHROPIC_API_KEY set?')
    }
    setOrbitPending(false)
  }, [orbitJourney])

  const applyOrbit = useCallback(async () => {
    setError('')
    try {
      const res = await fetch(`${API}/orbit/apply`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ descriptions: orbitDescriptions, allow_same_artist: orbitAllowSameArtist }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else {
        setOrbitApplied(true)
      }
    } catch {
      setError('Apply orbit failed — is the DJ running?')
    }
  }, [orbitDescriptions, orbitAllowSameArtist])

  const clearOrbit = useCallback(async () => {
    await fetch(`${API}/orbit`, { method: 'DELETE' })
    setOrbitApplied(false)
    setOrbitDescriptions([])
    setOrbitJourney('')
  }, [])

  const clearMood = useCallback(async () => {
    await fetch(`${API}/mood`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ descriptors: [] }),
    })
    setMoodApplied(false)
    setMoodDescriptors([])
    setMood('')
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
        ALBart Control Center
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
          {status?.history && (status.history.length > 1 || status.history[0]?.set_start) && (
            <Card>
              <Label>Session History</Label>
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                {status.history.map((t, i) => (
                  <div key={`${t.track_id}-${i}`}>
                    {i > 0 && (
                      <div style={{
                        padding: '3px 0', fontSize: 14, color: '#999',
                      }}>
                        {t.title} — {t.artist}
                      </div>
                    )}
                    {t.set_start && (
                      <div style={{
                        padding: '6px 0', fontSize: 12,
                        color: t.set_start === 'ORBIT TRANSIT' ? '#f0c040'
                             : t.set_start === 'ORBIT DWELL' ? '#2ecc71'
                             : '#7c3aed',  // purple — matches New Set button
                        fontWeight: 600, letterSpacing: 1,
                        borderBottom: '1px solid #2a3050',
                        marginBottom: 6,
                      }}>
                        ── {t.set_start} ──
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>

        {/* ── Right column: System + Mood ── */}
        <div style={{ flex: '0 0 340px' }}>
          {/* System */}
          <Card>
            <Label>System</Label>
            <div style={{ display: 'flex', gap: 10 }}>
              <button
                onClick={toggleMapView}
                style={{
                  ...btnStyle(system?.mapview.running ? '#e74c3c' : '#2ecc71'),
                  flex: 1, fontSize: 14, padding: '8px 12px',
                }}
              >
                {system?.mapview.running ? 'Stop MapView' : 'Start MapView'}
              </button>
              <button
                onClick={toggleDjActive}
                style={{
                  ...btnStyle(status?.dj_active !== false ? '#2ecc71' : '#e74c3c'),
                  flex: 1, fontSize: 14, padding: '8px 12px',
                }}
              >
                {status?.dj_active !== false ? 'DJ Control' : 'Spotify Control'}
              </button>
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 8, fontSize: 12, color: '#666' }}>
              <span style={{ flex: 1, textAlign: 'center' }}>
                {system?.mapview.running
                  ? <span style={{ color: '#2ecc71' }}>running (pid {system.mapview.pid})</span>
                  : 'stopped'}
              </span>
              <span style={{ flex: 1, textAlign: 'center' }}>
                {status?.dj_active !== false
                  ? <span style={{ color: '#2ecc71' }}>DJ picks tracks</span>
                  : <span style={{ color: '#e74c3c' }}>following Spotify</span>}
              </span>
            </div>
          </Card>

          {/* Orbit */}
          <Card>
            <Label>Orbit</Label>
            <textarea
              rows={2}
              placeholder="Describe a 3-hour musical journey..."
              value={orbitJourney}
              onChange={e => { setOrbitJourney(e.target.value); setOrbitApplied(false) }}
              style={{ ...inputStyle, width: '100%', resize: 'vertical' }}
            />
            <button
              onClick={interpretOrbit} disabled={orbitPending}
              style={{
                ...btnStyle(orbitPending ? '#555' : '#4a6cf7'),
                marginTop: 10, width: '100%',
              }}
            >
              {orbitPending ? 'Interpreting...' : 'Interpret'}
            </button>

            {/* Anchor descriptions (editable) */}
            <div style={{
              marginTop: 14, padding: 14,
              background: '#0f1729', borderRadius: 8,
              border: '1px solid #2a2f45',
              minHeight: 60,
            }}>
              {orbitDescriptions.length > 0 ? (
                <>
                  <div style={{ fontSize: 13, color: '#667', marginBottom: 6 }}>
                    Anchors (editable, one per line):
                  </div>
                  <textarea
                    rows={6}
                    value={orbitDescriptions.join('\n')}
                    onChange={e => {
                      setOrbitDescriptions(e.target.value.split('\n'))
                      setOrbitApplied(false)
                    }}
                    style={{
                      ...inputStyle, width: '100%', fontSize: 13,
                      lineHeight: 1.6, resize: 'vertical',
                    }}
                  />
                </>
              ) : (
                <div style={{ fontSize: 14, color: '#444', fontStyle: 'italic' }}>
                  Describe a journey and click Interpret to generate waypoints
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
              <button
                onClick={applyOrbit}
                disabled={orbitDescriptions.length === 0 || orbitApplied}
                style={{
                  ...btnStyle(
                    orbitDescriptions.length === 0 ? '#333'
                      : orbitApplied ? '#555'
                      : '#2ecc71'
                  ),
                  flex: 1,
                }}
              >
                {orbitApplied ? 'Active' : 'Apply Orbit'}
              </button>
              {(status?.orbit_active || orbitApplied) && (
                <button onClick={clearOrbit} style={btnStyle('#e74c3c')}>
                  Clear
                </button>
              )}
            </div>

            {/* Orbit viewer button */}
            {status?.orbit_active && status.orbit_anchors.length > 0 && (
              <button
                onClick={() => setOrbitViewerOpen(true)}
                style={{ ...btnStyle('#4a6cf7'), marginTop: 12, width: '100%' }}
              >
                View Orbit
              </button>
            )}
          </Card>

          {/* Mood */}
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
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
              {status && status.mood_in_count > 0 && status.total_tracks > 0 && moodApplied && (
                <span style={{ fontSize: 13, color: '#888', flex: 1, textAlign: 'center' }}>
                  {status.mood_in_count}/{status.total_tracks} tracks in-mood
                  ({Math.round(100 * status.mood_in_count / status.total_tracks)}%)
                </span>
              )}
              {moodApplied && (
                <button onClick={clearMood} style={{ ...btnStyle('#e74c3c'), fontSize: 13, padding: '4px 12px' }}>
                  Clear
                </button>
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

      {orbitViewerOpen && status?.orbit_active && (
        <OrbitViewer
          anchors={status.orbit_anchors}
          progress={status.orbit_progress}
          onClose={() => setOrbitViewerOpen(false)}
        />
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

// ── Orbit Viewer ────────────────────────────────────────────────────────

function OrbitViewer({ anchors, progress, onClose }: {
  anchors: OrbitAnchorInfo[]
  progress: OrbitProgress | null
  onClose: () => void
}) {
  const size = 420
  const cx = size / 2
  const cy = size / 2
  const radius = 150
  const coverSize = 56

  const [pos, setPos] = useState({ x: window.innerWidth - size - 60, y: 30 })
  const [dragging, setDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    setDragging(true)
    setDragOffset({ x: e.clientX - pos.x, y: e.clientY - pos.y })
  }, [pos])

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: MouseEvent) => {
      setPos({ x: e.clientX - dragOffset.x, y: e.clientY - dragOffset.y })
    }
    const onUp = () => setDragging(false)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [dragging, dragOffset])

  // Arrange anchors in a circle
  const positions = anchors.map((_, i) => {
    const angle = (i / anchors.length) * Math.PI * 2 - Math.PI / 2
    return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius }
  })

  const prevIdx = progress?.prev_index ?? 0
  const curIdx = progress?.current_index ?? 0
  const segProgress = progress?.segment_progress ?? 0
  const phase = progress?.phase ?? 'dwell'
  const completedSegments = new Set(progress?.completed_segments ?? [])

  const phaseLabel = phase === 'dwell'
    ? `Dwelling (${Math.floor((progress?.dwell_elapsed ?? 0) / 60)}/${Math.floor((progress?.dwell_duration ?? 1800) / 60)}m)`
    : `Transit ${(progress?.transit_total ?? 10) - (progress?.transit_remaining ?? 0)}/${progress?.transit_total ?? 10}`

  return (
    <div style={{
      position: 'fixed', left: pos.x, top: pos.y, zIndex: 1000,
      background: '#16213e', borderRadius: 12,
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      border: '1px solid #2a3050',
      userSelect: 'none',
    }}>
      {/* Title bar — draggable */}
      <div
        onMouseDown={onMouseDown}
        style={{
          padding: '8px 14px', cursor: 'grab',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          borderBottom: '1px solid #2a3050',
        }}
      >
        <span style={{ fontSize: 13, color: '#888', fontWeight: 600 }}>
          Orbit — <span style={{ color: phase === 'transit' ? '#f0c040' : '#2ecc71' }}>{phaseLabel}</span>
        </span>
        <button
          onClick={onClose}
          style={{
            background: 'none', border: 'none', color: '#666',
            fontSize: 18, cursor: 'pointer', padding: '0 4px',
          }}
        >
          x
        </button>
      </div>

      <div style={{ padding: 16, position: 'relative' }}>
        <svg width={size} height={size}>
          {anchors.map((_, i) => {
            const j = (i + 1) % anchors.length
            const from = positions[i]
            const to = positions[j]

            const isActiveSegment = (i === prevIdx && j === curIdx)
            const isCompleted = completedSegments.has(i)

            if (isActiveSegment && segProgress > 0 && segProgress < 1) {
              // Partially traversed — show progress bar
              const mx = from.x + (to.x - from.x) * segProgress
              const my = from.y + (to.y - from.y) * segProgress
              return (
                <g key={`seg-${i}`}>
                  <line
                    x1={from.x} y1={from.y} x2={mx} y2={my}
                    stroke="#f0c040" strokeWidth={4} strokeLinecap="round"
                  />
                  <line
                    x1={mx} y1={my} x2={to.x} y2={to.y}
                    stroke="#334" strokeWidth={1.5}
                  />
                </g>
              )
            }

            return (
              <line
                key={`seg-${i}`}
                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke={isCompleted || (isActiveSegment && segProgress >= 1) ? '#f0c040' : '#334'}
                strokeWidth={isCompleted || (isActiveSegment && segProgress >= 1) ? 4 : 1.5}
                strokeLinecap="round"
              />
            )
          })}
        </svg>

        {anchors.map((a, i) => {
          const p = positions[i]
          const isDwelling = (i === curIdx && phase === 'dwell')
          return (
            <div key={`cover-${i}`} style={{
              position: 'absolute',
              left: 16 + p.x - coverSize / 2,
              top: 16 + p.y - coverSize / 2,
              width: coverSize, height: coverSize,
              borderRadius: 6,
              border: isDwelling ? '3px solid #2ecc71' : '2px solid #334',
              overflow: 'hidden',
              boxShadow: isDwelling ? '0 0 12px #2ecc7166' : 'none',
              transition: 'border-color 0.3s, box-shadow 0.3s',
            }}>
              <img
                src={`http://127.0.0.1:8765${a.art_url}`}
                alt={a.title}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                draggable={false}
              />
              <div style={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                background: 'rgba(0,0,0,0.75)', padding: '2px 4px',
                fontSize: 8, color: '#ccc', textAlign: 'center',
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                {a.artist}
              </div>
            </div>
          )
        })}
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
