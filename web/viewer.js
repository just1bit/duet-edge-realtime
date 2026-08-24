const $ = id => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const state = {
  parents: [], ws: null, mode: 'live', ended: false, retry: 0, retryTimer: null,
  fps: 30, frame: null, replay: [], replayIndex: 0, playhead: 0, paused: false,
  lastTick: performance.now(), yaw: -.42, dragging: false, pointerX: 0,
};

function status(text, tone = 'waiting') {
  $('connection').className = `status-pill ${tone}`;
  $('connection').innerHTML = `<i></i>${text}`;
}

function mode(value) {
  state.mode = value;
  $('mode').textContent = value === 'live' ? 'LIVE' : 'NDJSON';
  $('play').disabled = value === 'live' || !state.replay.length;
}

function clearViewer() {
  state.frame = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $('empty-state').classList.remove('hidden');
  ['frame', 'age', 'e2e', 'p95', 'delivery'].forEach(id => $(id).textContent = '—');
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
  drawSkeleton(lead, canvas.width * .25, '#6ca9ff');
  drawSkeleton(companion, canvas.width * .75, '#61e6c8');
  state.frame = frame;
  $('empty-state').classList.add('hidden');
  $('frame').textContent = `${frame.frame_id ?? frame.seq ?? '—'} · ${frame.clip_id || 'timeline'}`;
  $('age').textContent = state.mode === 'live' ? frameAge(frame) : 'Local replay';
  $('e2e').textContent = frame.end_to_end_latency_ms == null ? '—' : `${frame.end_to_end_latency_ms.toFixed(1)} ms`;
}

function frameAge(frame) {
  return frame.emitted_wall_time_s
    ? `${Math.max(0, Date.now() - frame.emitted_wall_time_s * 1000).toFixed(0)} ms`
    : '—';
}

function handle(message) {
  if (message.type === 'hello') {
    state.parents = message.parents || [];
    state.fps = message.fps || 30;
  } else if (message.type === 'state') {
    const label = {starting:'Starting', buffering:'Buffering', playing:'Live', draining:'Finishing', finished:'Completed', failed:'Failed'}[message.state];
    status(label || message.state, message.state === 'failed' ? 'offline' : 'online');
  } else if (message.type === 'frame') {
    draw(message);
  } else if (message.type === 'metrics') {
    $('p95').textContent = message.inference_p95_ms == null ? '—' : `${message.inference_p95_ms.toFixed(1)} ms`;
    const jitter = message.jitter_p95_ms == null ? '—' : `${message.jitter_p95_ms.toFixed(1)} ms`;
    $('delivery').textContent = `${jitter} · ${message.dropped_view_frames ?? 0}`;
  } else if (message.type === 'eos') {
    state.ended = true;
    status('Stream completed', 'online');
  } else if (message.type === 'error') {
    status('Stream failed', 'offline');
  }
}

function retryLater() {
  const delay = Math.min(10000, 700 * 2 ** state.retry++);
  status('Waiting for stream');
  state.retryTimer = setTimeout(() => connect(), delay);
}

function connect(manual = false) {
  clearTimeout(state.retryTimer);
  if (manual) state.retry = 0;
  const leavingReplay = state.mode === 'replay';
  mode('live');
  state.ended = false;
  if (leavingReplay) clearViewer();
  if (state.ws) {
    const old = state.ws;
    state.ws = null;
    old.close();
  }
  if (manual) status('Connecting…');

  const ws = new WebSocket($('ws-url').value.trim());
  state.ws = ws;
  ws.onopen = () => { state.retry = 0; status('Connected · waiting for frames', 'online'); };
  ws.onmessage = event => {
    try { handle(JSON.parse(event.data)); }
    catch (error) { console.warn('Invalid stream message', error); }
  };
  ws.onclose = () => {
    if (state.ws === ws && state.mode === 'live' && !state.ended) retryLater();
  };
}

async function openFile(file) {
  mode('replay');
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

function tick(now) {
  const elapsed = Math.min(.1, (now - state.lastTick) / 1000);
  state.lastTick = now;
  if (state.mode === 'live' && state.frame) $('age').textContent = frameAge(state.frame);
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
