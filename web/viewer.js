const $ = id => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const state = {
  parents: [], ws: null, mode: 'live', hasLiveFrame: false, retry: 0, retryTimer: null,
  inputMode: 'file', serviceState: 'waiting_for_input', poseUsable: false,
  lastFrameAt: null,
  fps: 30, frame: null, replay: [], replayIndex: 0, playhead: 0, paused: false,
  lastTick: performance.now(), yaw: -.42, dragging: false, pointerX: 0,
  telemetryAt: performance.now(), renderedFrames: 0, visibleStalls: 0,
  arrivalAnchorAt: null, arrivalAnchorFrame: null, viewerDelayMs: null,
};

function status(text, tone = 'waiting') {
  $('connection').className = `status-pill ${tone}`;
  $('connection').innerHTML = `<i></i>${text}`;
}

function mode(value) {
  state.mode = value;
  updateModePill();
  $('play').disabled = value === 'live' || !state.replay.length;
}

function updateModePill() {
  if (state.mode === 'replay') $('mode').textContent = 'NDJSON · REPLAY';
  else $('mode').textContent = state.inputMode === 'mediapipe' ? 'MEDIAPIPE' : 'FILE';
}

function resetArrivalTelemetry() {
  state.arrivalAnchorAt = null;
  state.arrivalAnchorFrame = null;
  state.viewerDelayMs = null;
}

function resetLiveExperience() {
  state.serviceState = 'waiting_for_input';
  state.hasLiveFrame = false;
  state.poseUsable = false;
  state.lastFrameAt = null;
  resetArrivalTelemetry();
}

const viewerCopy = {
  file: {
    waiting: ['Waiting for file input', 'Start a file run, or open an NDJSON recording.'],
    preparing: ['Preparing file playback', 'Building the first model window.'],
    live: ['Playing file stream', 'Duet-EDGE output is live.'],
    completed: ['File playback completed', 'The full recording has been delivered.'],
    error: ['File stream failed', 'Check the service log for details.'],
  },
  mediapipe: {
    waiting: ['Waiting for a usable pose', 'Step back and keep shoulders, hips, knees and ankles visible.'],
    preparing: ['Preparing live duet', 'Usable pose received. Waiting for fresh model output.'],
    live: ['Live duet', 'MediaPipe is driving Duet-EDGE in realtime.'],
    paused: ['Live duet paused', 'Pose unavailable. Check camera framing or connection.'],
    error: ['Live pipeline failed', 'Check camera, pose input and service status.'],
  },
};

function experienceState() {
  if (state.serviceState === 'failed') return 'error';
  if (state.inputMode === 'file') {
    if (state.serviceState === 'finished') return 'completed';
    if (state.hasLiveFrame) return 'live';
    return ['starting', 'buffering'].includes(state.serviceState) ? 'preparing' : 'waiting';
  }
  if (!state.poseUsable) return state.hasLiveFrame ? 'paused' : 'waiting';
  const outputFresh = state.lastFrameAt != null
    && performance.now() - state.lastFrameAt < 1000;
  return outputFresh ? 'live' : 'preparing';
}

function applyExperience() {
  if (state.mode !== 'live') return;
  const value = experienceState();
  updateModePill();
  const copy = (viewerCopy[state.inputMode] || viewerCopy.file)[value] || [value, ''];
  $('experience-title').textContent = copy[0];
  $('experience-detail').textContent = copy[1];

  const viewer = document.querySelector('.viewer');
  viewer.classList.toggle('viewer-state-paused', value === 'paused');
  viewer.classList.toggle('viewer-state-error', value === 'error');
  const showOverlay = ['waiting', 'preparing', 'paused', 'error'].includes(value)
    || (value === 'completed' && !state.hasLiveFrame);
  $('empty-state').classList.toggle('hidden', !showOverlay);

  const statusText = {
    waiting: state.inputMode === 'mediapipe' ? 'Waiting for pose' : 'Waiting for file',
    preparing: 'Preparing', live: 'Live', paused: 'Paused', completed: 'Completed', error: 'Failed',
  }[value] || value;
  const tone = value === 'error' ? 'offline'
    : ['waiting', 'preparing', 'paused'].includes(value) ? 'waiting' : 'online';
  status(statusText, tone);
}

function clearViewer() {
  state.frame = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $('empty-state').classList.remove('hidden');
  ['frame', 'stream-time', 'age', 'e2e', 'p95', 'delivery'].forEach(id => $(id).textContent = '—');
}

function project(point, centerX) {
  const c = Math.cos(state.yaw), s = Math.sin(state.yaw);
  const x = point[0] * c - point[1] * s;
  const depth = point[0] * s + point[1] * c;
  const scale = 260 / Math.max(.72, 1 + depth * .07);
  return [centerX + x * scale, canvas.height * .94 - point[2] * scale];
}

function drawSkeleton(joints, centerX, color) {
  const root = joints[0];
  const floor = Math.min(...joints.map(point => point[2]));
  const points = joints.map(point => project([
    point[0] - root[0], point[1] - root[1], point[2] - floor + .04,
  ], centerX));

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 5;
  ctx.strokeStyle = color;
  state.parents.forEach((parent, index) => {
    if (parent < 0 || !points[parent]) return;
    ctx.beginPath();
    ctx.moveTo(...points[parent]);
    ctx.lineTo(...points[index]);
    ctx.stroke();
  });
  points.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(...point, index ? 3.4 : 6.5, 0, Math.PI * 2);
    ctx.fillStyle = index ? '#edf3f4' : color;
    ctx.fill();
  });
}

function draw(frame) {
  const lead = frame?.lead_joints;
  const companion = frame?.companion_joints || frame?.joints;
  if (!lead || !companion) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawSkeleton(lead, canvas.width * .3, '#6ca9ff');
  drawSkeleton(companion, canvas.width * .7, '#61e6c8');
  state.frame = frame;
  if (state.mode === 'replay') $('empty-state').classList.add('hidden');
  $('frame').textContent = `${frame.frame_id ?? frame.seq ?? '—'} · ${frame.clip_id || 'timeline'}`;
  $('stream-time').textContent = formatStreamTime(frame);
  $('age').textContent = state.mode === 'live' ? frameAge(frame) : 'Local replay';
  $('e2e').textContent = frame.end_to_end_latency_ms == null ? '—' : `${frame.end_to_end_latency_ms.toFixed(1)} ms`;
}

function frameAge() {
  return Number.isFinite(state.viewerDelayMs)
    ? `${state.viewerDelayMs.toFixed(0)} ms`
    : '—';
}

function recordFrameArrival(frame) {
  const frameId = frame.frame_id ?? frame.seq;
  if (!Number.isFinite(frameId)) return;
  const arrivedAt = performance.now();
  if (state.arrivalAnchorAt == null || state.arrivalAnchorFrame == null) {
    state.arrivalAnchorAt = arrivedAt;
    state.arrivalAnchorFrame = frameId;
  }
  const expectedAt = state.arrivalAnchorAt
    + (frameId - state.arrivalAnchorFrame) * 1000 / state.fps;
  state.viewerDelayMs = Math.max(0, arrivedAt - expectedAt);
}

function formatStreamTime(frame) {
  const frameId = frame.frame_id ?? frame.seq;
  const seconds = frame.source_time_s ?? frame.motion_time_s ??
    (frameId == null ? null : frameId / state.fps);
  if (!Number.isFinite(seconds)) return '—';
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor(whole % 3600 / 60);
  const secs = whole % 60;
  return [hours, minutes, secs].map(value => String(value).padStart(2, '0')).join(':');
}

function handle(message) {
  if (message.type === 'hello') {
    state.parents = message.parents || [];
    state.fps = message.fps || 30;
    state.inputMode = message.input_mode || 'file';
    resetLiveExperience();
    applyExperience();
  } else if (message.type === 'input_status') {
    state.inputMode = message.input_mode || state.inputMode;
    if (message.pose_usable === true && !state.poseUsable && state.hasLiveFrame) {
      state.lastFrameAt = null;
      resetArrivalTelemetry();
    }
    state.poseUsable = message.pose_usable === true;
    applyExperience();
  } else if (message.type === 'state') {
    state.serviceState = message.state;
    applyExperience();
  } else if (message.type === 'frame') {
    if (state.mode === 'live') {
      state.hasLiveFrame = true;
      state.lastFrameAt = performance.now();
    }
    if (state.mode === 'live') recordFrameArrival(message);
    draw(message);
    applyExperience();
  } else if (message.type === 'metrics') {
    if (state.mode === 'live' && !state.hasLiveFrame) return;
    $('p95').textContent = message.inference_p95_ms == null ? '—' : `${message.inference_p95_ms.toFixed(1)} ms`;
    const jitter = message.jitter_p95_ms == null ? '—' : `${message.jitter_p95_ms.toFixed(1)} ms`;
    $('delivery').textContent = `${jitter} · ${message.dropped_view_frames ?? 0}`;
  } else if (message.type === 'eos') {
    state.serviceState = 'finished';
    applyExperience();
  } else if (message.type === 'error') {
    state.serviceState = 'failed';
    applyExperience();
  }
}

function retryLater(silent = false) {
  const delay = Math.min(10000, 700 * 2 ** state.retry++);
  if (!silent) status('Waiting for stream');
  state.retryTimer = setTimeout(() => connect(), delay);
}

function connect(manual = false) {
  clearTimeout(state.retryTimer);
  if (manual) state.retry = 0;
  const leavingReplay = state.mode === 'replay';
  mode('live');
  resetLiveExperience();
  if (leavingReplay) clearViewer();
  if (state.ws) {
    const old = state.ws;
    state.ws = null;
    old.close();
  }
  if (manual) status('Connecting…');

  const ws = new WebSocket($('ws-url').value.trim());
  state.ws = ws;
  ws.onopen = () => {
    state.retry = 0;
    $('p95').textContent = '—';
    $('delivery').textContent = '—';
    status('Connected · waiting for frames', 'online');
  };
  ws.onmessage = event => {
    try { handle(JSON.parse(event.data)); }
    catch (error) { console.warn('Invalid stream message', error); }
  };
  ws.onclose = () => {
    state.poseUsable = false;
    state.lastFrameAt = null;
    applyExperience();
    if (state.ws === ws && state.mode === 'live') {
      retryLater(state.inputMode === 'file' && state.serviceState === 'finished');
    }
  };
}

async function openFile(file) {
  mode('replay');
  state.hasLiveFrame = false;
  state.inputMode = 'file';
  clearTimeout(state.retryTimer);
  if (state.ws) state.ws.close();
  try {
    const messages = (await file.text()).split(/\r?\n/).filter(Boolean).map(JSON.parse);
    messages.filter(item => item.type === 'hello').forEach(handle);
    state.replay = messages.filter(item => item.type === 'frame');
    state.replayIndex = 0;
    state.playhead = 0;
    state.paused = false;
    state.lastTick = performance.now();
    mode('replay');
    $('play').textContent = 'Pause';
    status(state.replay.length ? 'Playing recording' : 'No frames in file', state.replay.length ? 'online' : 'offline');
    if (state.replay.length) draw(state.replay[0]);
  } catch (error) {
    status('Invalid NDJSON file', 'offline');
  }
}

$('connect').onclick = () => connect(true);
$('file').onchange = event => {
  if (event.target.files[0]) openFile(event.target.files[0]);
  event.target.value = '';
};
$('play').onclick = () => {
  if (state.replayIndex === state.replay.length - 1) {
    state.replayIndex = state.playhead = 0;
    draw(state.replay[0]);
  }
  state.paused = !state.paused;
  state.lastTick = performance.now();
  $('play').textContent = state.paused ? 'Play' : 'Pause';
  status(state.paused ? 'Recording paused' : 'Playing recording', 'online');
};

canvas.onpointerdown = event => {
  state.dragging = true;
  state.pointerX = event.clientX;
  canvas.classList.add('dragging');
};
canvas.onpointermove = event => {
  if (!state.dragging) return;
  state.yaw += (event.clientX - state.pointerX) * .009;
  state.pointerX = event.clientX;
  draw(state.frame);
};
window.onpointerup = () => {
  state.dragging = false;
  canvas.classList.remove('dragging');
};
document.addEventListener('visibilitychange', () => {
  state.lastTick = performance.now();
});

function tick(now) {
  const rawElapsed = (now - state.lastTick) / 1000;
  const elapsed = Math.min(.1, rawElapsed);
  state.lastTick = now;
  state.renderedFrames += 1;
  const liveExperience = state.mode === 'live' && experienceState() === 'live';
  if (
    liveExperience && !document.hidden && rawElapsed > .1
  ) state.visibleStalls += 1;
  if (
    state.mode === 'live' && state.inputMode === 'mediapipe'
    && state.poseUsable && state.lastFrameAt != null
    && now - state.lastFrameAt >= 1000
  ) {
    resetArrivalTelemetry();
    applyExperience();
  }
  const telemetryElapsed = now - state.telemetryAt;
  if (
    state.mode === 'live' && state.frame && telemetryElapsed >= 1000
    && state.ws?.readyState === WebSocket.OPEN
  ) {
    state.ws.send(JSON.stringify({
      type: 'client_metrics',
      render_fps: state.renderedFrames * 1000 / telemetryElapsed,
      frame_age_ms: state.viewerDelayMs,
      visible_stalls: state.visibleStalls,
    }));
    state.telemetryAt = now;
    state.renderedFrames = 0;
    state.visibleStalls = 0;
  }
  if (state.mode === 'replay' && !state.paused && state.replay.length) {
    state.playhead += elapsed * state.fps;
    const index = Math.min(state.replay.length - 1, Math.floor(state.playhead));
    if (index !== state.replayIndex) draw(state.replay[state.replayIndex = index]);
    if (index === state.replay.length - 1) {
      state.paused = true;
      $('play').textContent = 'Play';
      status('Recording completed', 'online');
    }
  }
  requestAnimationFrame(tick);
}

clearViewer();
mode('live');
requestAnimationFrame(tick);
setTimeout(connect, 250);
