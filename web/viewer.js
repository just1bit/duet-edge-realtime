const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
let parents = [], ws = null, view = 0, playbackTimer = null;

function setConnection(text, online=false) {
  $('connection').textContent = text;
  $('connection').classList.toggle('offline', !online);
}

function project(p) {
  const views = [
    q => [q[0], q[2], q[1]],
    q => [q[1], q[2], -q[0]],
    q => [(q[0]-q[1])*.72, q[2]+(q[0]+q[1])*.18, q[0]+q[1]],
  ];
  const [x,y,d] = views[view](p);
  const scale = 230 / Math.max(.65, 1 + d*.08);
  return [canvas.width/2 + x*scale, canvas.height*.83 - y*scale];
}

function draw(joints) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#182127'; ctx.lineWidth=1;
  for(let x=0;x<canvas.width;x+=50){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}
  for(let y=0;y<canvas.height;y+=50){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}
  const pts = joints.map(project);
  ctx.lineCap='round'; ctx.lineJoin='round'; ctx.lineWidth=5; ctx.strokeStyle='#5ee8d7';
  parents.forEach((parent,i) => { if(parent<0)return; ctx.beginPath();ctx.moveTo(...pts[parent]);ctx.lineTo(...pts[i]);ctx.stroke(); });
  pts.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],i===0?7:4,0,Math.PI*2);ctx.fillStyle=i===0?'#ffc268':'#f5f0e8';ctx.fill()});
}

function handle(message) {
  if(message.type==='hello') { parents=message.parents; $('latency').textContent=`${message.fixed_latency_s.toFixed(2)} s`; $('status').textContent='流已就绪'; }
  if(message.type==='frame') { draw(message.joints); $('frame').textContent=message.seq; $('motion-time').textContent=`${message.motion_time_s.toFixed(2)} s`; }
  if(message.type==='metrics') { $('p95').textContent=message.inference_p95_ms==null?'—':`${message.inference_p95_ms.toFixed(1)} ms`; $('queue').textContent=`${message.input_backlog}/${message.output_backlog}`; $('dropped').textContent=message.dropped_view_frames; }
  if(message.type==='degraded') $('status').textContent='降级：推理超时';
  if(message.type==='eos') $('status').textContent=`完成 / ${message.frames} 帧`;
  if(message.type==='error') $('status').textContent='运行错误';
}

$('connect').onclick = () => {
  if(ws) ws.close();
  ws = new WebSocket($('ws-url').value);
  setConnection('连接中');
  ws.onopen=()=>setConnection('实时流',true);
  ws.onmessage=e=>handle(JSON.parse(e.data));
  ws.onerror=()=>setConnection('连接错误');
  ws.onclose=()=>setConnection('已断开');
};

$('view').onclick=()=>{view=(view+1)%3};
$('file').onchange = async event => {
  clearInterval(playbackTimer); if(ws) ws.close();
  const lines=(await event.target.files[0].text()).split(/\r?\n/).filter(Boolean);
  const messages=lines.map(JSON.parse); messages.filter(m=>m.type==='hello').forEach(handle);
  const frames=messages.filter(m=>m.type==='frame'); let i=0;
  setConnection('NDJSON 回放',true); $('status').textContent='本地回放';
  playbackTimer=setInterval(()=>{if(i>=frames.length){clearInterval(playbackTimer);$('status').textContent='回放完成';return}handle(frames[i++])},1000/30);
};
