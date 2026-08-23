const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const state = {
  parents: [], ws: null, mode: 'live', view: 'world', paused: false,
  liveFrames: [], replayFrames: [], replayIndex: 0, replayStart: 0,
  streamFps: 30, reconnectAttempt: 0, reconnects: 0, reconnectTimer: null,
  manualClose: false, lastFrame: null, yaw: -.55, zoom: 1, trails: [[], []],
  renderCount: 0, renderFps: 0, renderSecond: performance.now(), frameAge: 0,
  visibleStalls: 0, lastRenderedId: null, lastNewFrameAt: performance.now(),
  serviceState: 'disconnected',
};

function setConnection(text, online=false) {
  $('connection').textContent=text;$('connection').classList.toggle('offline',!online);
}
function setStatus(value) {
  const labels={starting:'Starting',buffering:'Buffering',playing:'Live',draining:'Completing',finished:'Completed',failed:'Failed',replaying:'Replaying',disconnected:'Disconnected'};
  $('status').textContent=labels[value]||value;
}
function rotate(p) {const c=Math.cos(state.yaw),s=Math.sin(state.yaw);return[p[0]*c-p[1]*s,p[0]*s+p[1]*c,p[2]]}
function project(p,centerX=canvas.width/2) {const q=rotate(p),d=Math.max(.7,1+q[1]*.045),scale=220*state.zoom/d;return[centerX+q[0]*scale,canvas.height*.82-q[2]*scale]}
function ground() {
  ctx.strokeStyle='#26343d';ctx.lineWidth=1;
  for(let x=-8;x<=8;x++){const a=project([x,-6,0]),b=project([x,6,0]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
  for(let y=-6;y<=6;y++){const a=project([-8,y,0]),b=project([8,y,0]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
}
function drawTrail(points,color){if(points.length<2)return;ctx.strokeStyle=color;ctx.globalAlpha=.35;ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.globalAlpha=1}
function drawSkeleton(joints,color,label,centerX=null) {
  let display=joints;
  if(state.view==='diagnostic'){const root=joints[0],floor=Math.min(...joints.map(p=>p[2]));display=joints.map(p=>[p[0]-root[0],p[1]-root[1],p[2]-floor])}
  const points=display.map(p=>project(p,centerX===null?canvas.width/2:centerX));ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=5;ctx.strokeStyle=color;
  state.parents.forEach((parent,i)=>{if(parent<0)return;ctx.beginPath();ctx.moveTo(...points[parent]);ctx.lineTo(...points[i]);ctx.stroke()});
  points.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],i===0?7:3.5,0,Math.PI*2);ctx.fillStyle=i===0?'#ffc268':'#f5f0e8';ctx.fill()});
  ctx.font='700 13px system-ui,sans-serif';ctx.textAlign='center';ctx.fillStyle=color;ctx.fillText(label,centerX===null?points[0][0]:centerX,30);
}
function interpolate(a,b,t) {
  if(!a||!b)return a||b;const mix=(left,right)=>left.map((joint,i)=>joint.map((v,k)=>v+(right[i][k]-v)*t));
  return {...a,lead_joints:mix(a.lead_joints,b.lead_joints),companion_joints:mix(a.companion_joints,b.companion_joints)};
}
function draw(frame) {
  if(!frame)return;ctx.clearRect(0,0,canvas.width,canvas.height);ground();const lead=frame.lead_joints,companion=frame.companion_joints||frame.joints;
  if(state.view==='world'){
    if(state.lastRenderedId!==frame.frame_id){state.trails[0].push(lead[0]);state.trails[1].push(companion[0]);state.trails.forEach(t=>{if(t.length>90)t.shift()});state.lastRenderedId=frame.frame_id;state.lastNewFrameAt=performance.now()}
    drawTrail(state.trails[0],'#62a8ff');drawTrail(state.trails[1],'#5ee8d7');drawSkeleton(lead,'#62a8ff','LEAD');drawSkeleton(companion,'#5ee8d7','COMPANION');
  }else{drawSkeleton(lead,'#62a8ff','LEAD',canvas.width*.28);drawSkeleton(companion,'#5ee8d7','COMPANION',canvas.width*.72)}
  state.lastFrame=frame;$('frame').textContent=`${frame.frame_id} / ${frame.clip_id||'timeline'}`;$('seek').value=frame.frame_id;
  state.frameAge=frame.emitted_wall_time_s?Math.max(0,Date.now()-frame.emitted_wall_time_s*1000):0;$('age').textContent=`${state.frameAge.toFixed(0)} ms`;
}
function handle(message) {
  if(message.type==='hello'){
    state.parents=message.parents;state.streamFps=message.fps||30;$('latency').textContent=`${message.fixed_latency_s.toFixed(3)} s`;
    $('backend').textContent=message.backend_badge||message.backend||'UNKNOWN';$('backend').dataset.backend=message.backend;
    $('model').textContent=`${message.checkpoint||'fixture'} / ${message.sampling_steps} steps`;
    $('guidance').textContent=`${message.model_mode} · ${message.source_timeline?.identity||'source timeline'}`;
  }
  if(message.type==='state'){state.serviceState=message.state;setStatus(message.state)}
  if(message.type==='frame'){state.liveFrames.push({...message,_arrival:Date.now()});if(state.liveFrames.length>4)state.liveFrames.shift();$('seek').max=Math.max(Number($('seek').max),message.frame_id)}
  if(message.type==='metrics'){$('p95').textContent=message.inference_p95_ms==null?'—':`${message.inference_p95_ms.toFixed(1)} ms`;$('delivery').textContent=`${message.jitter_p95_ms==null?'—':message.jitter_p95_ms.toFixed(1)+' ms'} / ${message.dropped_view_frames}`}
  if(message.type==='degraded')setStatus('Deadline event');if(message.type==='backpressure')setStatus('Capacity balancing');if(message.type==='eos')setStatus('finished');if(message.type==='error')setStatus('failed');
}
function connect() {
  clearTimeout(state.reconnectTimer);state.manualClose=false;if(state.ws){state.manualClose=true;state.ws.close()}state.manualClose=false;state.mode='live';setConnection('Connecting');
  const ws=new WebSocket($('ws-url').value);state.ws=ws;
  ws.onopen=()=>{state.reconnectAttempt=0;setConnection('Live',true);setStatus('buffering');sendTelemetry(state.reconnects?'reconnect':'connect')};
  ws.onmessage=e=>handle(JSON.parse(e.data));ws.onerror=()=>setConnection('Reconnecting');
  ws.onclose=()=>{if(state.ws!==ws||state.manualClose)return;setConnection('Disconnected');setStatus('disconnected');scheduleReconnect()};
}
function scheduleReconnect(){const delay=Math.min(10000,500*2**state.reconnectAttempt++);state.reconnects++;$('reconnects').textContent=state.reconnects;state.reconnectTimer=setTimeout(connect,delay);$('guidance').textContent=`Live stream reconnect in ${(delay/1000).toFixed(1)} s`}
function sendTelemetry(event){if(state.ws?.readyState!==WebSocket.OPEN)return;state.ws.send(JSON.stringify({type:'client_metrics',event,render_fps:state.renderFps,frame_age_ms:state.frameAge,visible_stalls:state.visibleStalls}));state.visibleStalls=0}
function render(now) {
  if(!state.paused){let frame=null;
    if(state.mode==='live'&&state.liveFrames.length){const target=Date.now()-1000/state.streamFps*1.5;let a=state.liveFrames[0],b=state.liveFrames[state.liveFrames.length-1];for(let i=1;i<state.liveFrames.length;i++)if(state.liveFrames[i]._arrival>=target){a=state.liveFrames[i-1];b=state.liveFrames[i];break}const span=Math.max(1,b._arrival-a._arrival);frame=interpolate(a,b,Math.max(0,Math.min(1,(target-a._arrival)/span)))}
    else if(state.mode==='replay'&&state.replayFrames.length){const elapsed=(now-state.replayStart)/1000*Number($('speed').value);state.replayIndex=Math.min(state.replayFrames.length-1,Math.floor(elapsed*state.streamFps));frame=interpolate(state.replayFrames[state.replayIndex],state.replayFrames[state.replayIndex+1],elapsed*state.streamFps-state.replayIndex);if(state.replayIndex===state.replayFrames.length-1)setStatus('finished')}
    if(frame)draw(frame)}
  state.renderCount++;if(now-state.renderSecond>=1000){state.renderFps=state.renderCount*1000/(now-state.renderSecond);state.renderCount=0;state.renderSecond=now;$('render-fps').textContent=state.renderFps.toFixed(1);if(now-state.lastNewFrameAt>100&&state.mode==='live'&&state.serviceState==='playing'&&!state.paused)state.visibleStalls++}requestAnimationFrame(render);
}
$('connect').onclick=connect;$('pause').onclick=()=>{state.paused=!state.paused;$('pause').textContent=state.paused?'Resume':'Pause'};
$('restart').onclick=()=>{state.trails=[[],[]];if(state.mode==='replay'){state.replayStart=performance.now();state.replayIndex=0;setStatus('replaying')}else connect()};
$('file').onchange=async event=>{state.manualClose=true;if(state.ws)state.ws.close();clearTimeout(state.reconnectTimer);const messages=(await event.target.files[0].text()).split(/\r?\n/).filter(Boolean).map(JSON.parse);messages.filter(m=>m.type==='hello').forEach(handle);state.replayFrames=messages.filter(m=>m.type==='frame');state.mode='replay';state.replayIndex=0;state.replayStart=performance.now();state.trails=[[],[]];$('seek').max=Math.max(0,state.replayFrames.length-1);setConnection('Local Replay',true);setStatus('replaying')};
$('seek').oninput=event=>{if(state.mode!=='replay')return;state.replayIndex=Number(event.target.value);state.replayStart=performance.now()-state.replayIndex/state.streamFps*1000/Number($('speed').value);draw(state.replayFrames[state.replayIndex])};
$('world').onclick=()=>{state.view='world';$('world').classList.add('active');$('diagnostic').classList.remove('active')};$('diagnostic').onclick=()=>{state.view='diagnostic';$('diagnostic').classList.add('active');$('world').classList.remove('active')};
$('camera-left').onclick=()=>state.yaw-=.18;$('camera-right').onclick=()=>state.yaw+=.18;$('zoom').onclick=()=>{state.zoom=state.zoom>=1.3?.75:state.zoom+.15};
setInterval(()=>sendTelemetry('sample'),2000);requestAnimationFrame(render);setTimeout(connect,250);
