'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const PYFILES = ['py/mobile.py', 'py/roomeq/__init__.py', 'py/roomeq/dsp.py', 'py/roomeq/sweep.py', 'py/roomeq/calfile.py'];
const TAIL_S = 9;            // extra recording after the playback ends: AirPlay/Bluetooth delay the sound by seconds
const LEAD_S = 0.6;          // recording starts this long before the playback
const $ = id => document.getElementById(id);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

let mobile = null, ctx = null, stream = null, node = null, chunks = null, busy = false, level = -20, lvlOk = false;
let nPos = 4, nVer = 0, lastResult = null, lastAnalysis = null, calLabel = '';

// ---------- the analysis engine (same dsp/sweep code as the PC tool, in Pyodide) --------------------------------
async function loadEngine() {
  try {
    await new Promise((res, rej) => { const s = document.createElement('script'); s.src = PYODIDE + 'pyodide.js'; s.onload = res; s.onerror = () => rej(new Error('Pyodide could not be loaded (internet needed the first time)')); document.head.appendChild(s); });
    const py = await loadPyodide({ indexURL: PYODIDE });
    await py.loadPackage(['numpy', 'scipy']);
    py.FS.mkdirTree('/py/roomeq');
    for (const f of PYFILES) py.FS.writeFile('/' + f, await (await fetch(f)).text());
    py.runPython("import sys; sys.path.insert(0, '/py')");
    mobile = py.pyimport('mobile');
    window.py = py;
    $('engine').textContent = 'Analysis engine ready.';
    const cal = store.get('cal', '');
    if (cal) { const r = callPy('set_cal', cal, store.get('calname', 'cal.txt')); showCal(r); }
  } catch (e) { $('engine').innerHTML = '<span class="bad">' + e.message + '</span>'; }
}
function callPy(fn, ...args) { return JSON.parse(mobile.call(fn, ...args)); }
function needEngine() { if (!mobile) { alert('The analysis engine is still loading.'); return false; } return true; }

// ---------- setup ------------------------------------------------------------------------------------------------
function showCal(r) {
  if (!r || r.ok === false) { $('calinfo').innerHTML = '<span class="bad">' + (r && r.error) + '</span>'; return; }
  calLabel = r.label;
  const deg = r.deg != null ? ` · ${r.deg}°` : '';
  $('calinfo').innerHTML = `<span class="ok">${r.label}</span> · serial ${r.serial || '?'}${deg}`;
}
$('calbtn').onclick = () => $('calfile').click();
$('calfile').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  if (!needEngine()) return;
  const text = await f.text();
  const r = callPy('set_cal', text, f.name);
  showCal(r);
  if (r.ok !== false) { store.set('cal', text); store.set('calname', f.name); }
};
$('room').value = store.get('room', ''); $('room').oninput = e => store.set('room', e.target.value);
$('layout').value = store.get('layout', '2'); $('layout').onchange = e => { store.set('layout', e.target.value); resetAll(); };
$('npos').value = store.get('npos', '4'); $('npos').onchange = e => { store.set('npos', e.target.value); nPos = +e.target.value; updatePos(); };
nPos = +$('npos').value;

function resetAll() {
  if (mobile) callPy('reset', $('layout').value);
  lastAnalysis = null; $('result').innerHTML = ''; $('verres').innerHTML = '';
  nVer = 0; $('vermeas').textContent = 'Measure verification position 1'; $('verscore').hidden = true;
  $('c4').classList.add('off'); $('c5').classList.add('off');
  updatePos();
}

// ---------- microphone -------------------------------------------------------------------------------------------
async function startMic(deviceId) {
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    await ctx.audioWorklet.addModule('worklet.js');
  }
  await ctx.resume();
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  if (node) node.disconnect();
  const src = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, 'rec');
  node.port.onmessage = e => { meter(e.data); if (chunks) chunks.push(e.data); };
  const sink = ctx.createGain(); sink.gain.value = 0;
  src.connect(node); node.connect(sink); sink.connect(ctx.destination);
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
  const sel = $('dev'); sel.innerHTML = '';
  const cur = stream.getAudioTracks()[0].getSettings().deviceId;
  devs.forEach(d => { const o = document.createElement('option'); o.value = d.deviceId; o.textContent = d.label || 'Input'; if (d.deviceId === cur) o.selected = true; sel.appendChild(o); });
  const show = devs.length > 1; sel.hidden = !show; $('devlbl').hidden = !show;
  const umik = devs.find(d => /umik/i.test(d.label));
  if (umik && umik.deviceId !== cur) return startMic(umik.deviceId);
  $('micbtn').textContent = 'Microphone on';
  $('lvlbtn').disabled = false;
  const label = (devs.find(d => d.deviceId === cur) || {}).label || '';
  if (!/umik/i.test(label)) $('meterTxt').innerHTML = '<span class="warn">The UMIK-1 is not the selected input (' + (label || 'unknown') + '). Plug it in and choose it above.</span>';
}
$('micbtn').onclick = async () => {
  if (!needEngine()) return;
  try { await startMic(); } catch (e) { $('meterTxt').innerHTML = '<span class="bad">Microphone: ' + e.message + '</span>'; }
};
$('dev').onchange = e => startMic(e.target.value);

let meterAt = 0;
function meter(buf) {
  const now = performance.now(); if (now - meterAt < 120) return; meterAt = now;
  let s = 0, p = 0; for (let i = 0; i < buf.length; i++) { const v = buf[i]; s += v * v; if (Math.abs(v) > p) p = Math.abs(v); }
  const db = 20 * Math.log10(Math.sqrt(s / buf.length) + 1e-9);
  $('meter').style.width = Math.max(0, Math.min(100, (db + 100) / 100 * 100)) + '%';
  $('meter').style.background = p > 0.98 ? 'var(--bad)' : 'var(--ok)';
  if (!busy) $('meterTxt').textContent = `Microphone ${db.toFixed(0)} dBFS` + (p > 0.98 ? ' · CLIPPING' : '');
}

// ---------- play + record ----------------------------------------------------------------------------------------
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
async function playRecord(short, onProgress) {
  const [flat, nch, seconds] = (() => { const p = mobile.plan_audio(level, short); const j = p.toJs(); p.destroy(); return j; })();
  if (window.__fakeRecord) return window.__fakeRecord(short);
  const frames = flat.length / nch;
  const buf = ctx.createBuffer(nch, frames, 48000);
  for (let c = 0; c < nch; c++) buf.copyToChannel(flat.subarray(c * frames, (c + 1) * frames), c);
  let lock = null; try { lock = await navigator.wakeLock.request('screen'); } catch (e) { /* not available */ }
  await ctx.resume();
  chunks = [];
  await sleep(LEAD_S * 1000);
  const t0 = performance.now();
  const srcNode = ctx.createBufferSource(); srcNode.buffer = buf; srcNode.connect(ctx.destination); srcNode.start();
  const total = seconds + TAIL_S;
  while ((performance.now() - t0) / 1000 < total) { onProgress(Math.min(1, (performance.now() - t0) / 1000 / total)); await sleep(150); }
  const got = chunks; chunks = null;
  if (lock) lock.release();
  const n = got.reduce((a, c) => a + c.length, 0);
  const rec = new Float32Array(n); let o = 0; for (const c of got) { rec.set(c, o); o += c.length; }
  return rec;
}
function lock(b) { busy = b; document.querySelectorAll('button').forEach(x => { if (b) { x.dataset.was = x.disabled ? '1' : ''; x.disabled = true; } else x.disabled = x.dataset.was === '1'; }); }

// ---------- level check ------------------------------------------------------------------------------------------
$('lvlbtn').onclick = async () => {
  if (!needEngine()) return;
  lock(true); $('lvlres').innerHTML = '<div class="msg">Playing… keep the room quiet.</div>';
  try {
    const rec = await playRecord(true, p => { $('meterTxt').textContent = `Measuring… ${(p * 100).toFixed(0)} %`; });
    const r = callPy('check_level', rec, (ctx ? ctx.sampleRate : 48000));
    let h = '';
    if (r.error) h = `<div class="note bad">${r.error}</div>`;
    (r.channels || []).forEach(c => { h += `<div class="msg"><b>${c.name}</b> <span class="${c.status}">${c.text}</span></div>`; });
    if (r.noise_dba != null) h += `<div class="msg">Room noise ${r.noise_dba.toFixed(0)} dBFS(A)</div>`;
    $('lvlres').innerHTML = h;
    lvlOk = !!r.ok;
    if (lvlOk) { $('c3').classList.remove('off'); callPy('new_session', $('layout').value, level); updatePos(); }
  } catch (e) { $('lvlres').innerHTML = '<div class="note bad">' + e.message + '</div>'; }
  lock(false);
};

// ---------- positions --------------------------------------------------------------------------------------------
let nDone = 0;
function updatePos() {
  nDone = 0; $('measbtn').textContent = 'Measure position 1'; $('undobtn').hidden = true;
  $('posinfo').textContent = `Position 1 is the main listening seat; then ${nPos - 1} more around it (seated ear height, about 0.5–1 m apart).`;
}
async function runMeasure(verify, barId, msgId) {
  if (!needEngine()) return null;
  lock(true); $(msgId).textContent = 'Measuring…';
  try {
    const rec = await playRecord(false, p => { $(barId).style.width = (p * 100) + '%'; });
    $(msgId).textContent = 'Analysing…'; await sleep(30);
    const r = callPy('measure', rec, (ctx ? ctx.sampleRate : 48000), verify);
    $(barId).style.width = '0';
    if (!r.ok) { $(msgId).innerHTML = '<span class="bad">' + r.error + '</span>'; lock(false); return null; }
    $(msgId).textContent = '';
    lock(false); return r;
  } catch (e) { $(msgId).innerHTML = '<span class="bad">' + e.message + '</span>'; lock(false); return null; }
}
$('measbtn').onclick = async () => {
  const r = await runMeasure(false, 'bar', 'measmsg'); if (!r) return;
  nDone = r.n; $('measwarn').innerHTML = r.warnings.map(w => `<div class="note">${w}</div>`).join('');
  $('undobtn').hidden = false;
  $('measbtn').textContent = nDone >= nPos ? 'Measure another position' : `Measure position ${nDone + 1}`;
  $('measmsg').innerHTML = `<span class="ok">Position ${nDone} done.</span>` + (nDone >= nPos ? ' Enough for the EQ.' : '');
  if (nDone >= Math.min(nPos, 3) && nDone >= 1) $('c4').classList.remove('off');
};
$('undobtn').onclick = () => { nDone = callPy('undo', false); $('measbtn').textContent = `Measure position ${nDone + 1}`; $('undobtn').hidden = nDone === 0; $('measmsg').textContent = `${nDone} position(s) kept.`; };

// ---------- results ----------------------------------------------------------------------------------------------
function plot(c, p, fmin = 20, fmax = 20000) {
  const cv = document.createElement('canvas'); cv.width = 640; cv.height = 380;
  const g = cv.getContext('2d'), W = cv.width, H = cv.height, L = 38, B = 22;
  const css = getComputedStyle(document.body);
  const x = f => L + (Math.log10(f) - Math.log10(fmin)) / (Math.log10(fmax) - Math.log10(fmin)) * (W - L - 6);
  const y = d => (H - B) - (d + 15) / 30 * (H - B - 6);
  g.strokeStyle = css.getPropertyValue('--line'); g.fillStyle = css.getPropertyValue('--mut'); g.font = '18px sans-serif'; g.lineWidth = 1;
  [-12, -6, 0, 6, 12].forEach(d => { g.beginPath(); g.moveTo(L, y(d)); g.lineTo(W - 6, y(d)); g.stroke(); g.fillText(d, 4, y(d) + 5); });
  [50, 100, 200, 500, 1000, 2000, 5000, 10000].forEach(f => { g.beginPath(); g.moveTo(x(f), 4); g.lineTo(x(f), H - B); g.stroke(); g.fillText(f >= 1000 ? f / 1000 + 'k' : f, x(f) - 10, H - 4); });
  const line = (arr, col, w, ref) => {
    g.strokeStyle = col; g.lineWidth = w; g.beginPath(); let on = false;
    p.f.forEach((f, i) => { const v = arr[i]; if (v == null || f < fmin || f > fmax) return; const yy = y(Math.max(-15, Math.min(15, v - (ref || 0)))); on ? g.lineTo(x(f), yy) : g.moveTo(x(f), yy); on = true; });
    g.stroke();
  };
  line(p.target, css.getPropertyValue('--mut'), 2);
  line(p.meas, '#ff9f43', 3);
  if (p.pred) line(p.pred, css.getPropertyValue('--ok'), 3);
  return cv;
}
function legend(pred) { return `<span class="pill" style="color:#ff9f43">measured</span>` + (pred ? `<span class="pill ok">after EQ (predicted)</span>` : '') + `<span class="pill">target</span>`; }

$('anabtn').onclick = async () => {
  if (!needEngine()) return;
  $('result').innerHTML = '<div class="msg">Calculating…</div>'; await sleep(30);
  const r = callPy('analyze'); lastAnalysis = r;
  const box = $('result'); box.innerHTML = '';
  if (!r.ok) { box.innerHTML = `<div class="note bad">${r.error}</div>`; return; }
  const room = $('room').value || 'Room';
  let text = `Room EQ - ${room} - ${new Date().toISOString().slice(0, 10)}\nMic: ${calLabel || 'no calibration file'}\n`;
  const ex = r.extra || {};
  if (ex.delay_ms != null) {
    const d = ex.delay_ms < 0.05 ? 'none needed' : `${ex.delay_ms.toFixed(2)} ms on ${ex.delay_on}`;
    const l = ex.level_db < 0.3 ? 'none needed' : `${ex.level_db.toFixed(1)} dB lower on ${ex.level_on}`;
    box.insertAdjacentHTML('beforeend', `<div class="msg">Enter first:</div><div class="big">Delay: ${d}</div><div class="big">Level: ${l}</div>` +
      (ex.delay_spread_ms > 1 ? `<div class="note">The L/R delay moves ${ex.delay_spread_ms.toFixed(1)} ms between positions, so no single delay suits every seat. Prefer leaving it at 0.</div>` : ''));
    text += `Delay: ${d}\nLevel trim: ${l}\n`;
  }
  r.order.forEach(c => {
    const ch = r.channels[c];
    let rows = ch.filters.map((f, i) => `<tr><td>${i + 1}</td><td>${f.freq} Hz</td><td>${f.gain > 0 ? '+' : ''}${f.gain.toFixed(1)} dB</td><td>${f.q}</td></tr>`).join('');
    if (!rows) rows = '<tr><td colspan="4">No filters needed</td></tr>';
    const wrap = document.createElement('div');
    wrap.innerHTML = `<h2 style="margin-top:16px">${c}</h2><table><tr><th>#</th><th>Freq</th><th>Gain</th><th>Q</th></tr>${rows}</table>` +
      `<div class="msg">Error vs target ${ch.rms_before.toFixed(1)} → ${ch.rms_after.toFixed(1)} dB rms (predicted)</div>` + legend(true);
    wrap.appendChild(plot(c, ch.plot));
    box.appendChild(wrap);
    text += `\n${c} (peaking filters, Q-based)\n` + ch.filters.map((f, i) => `  ${i + 1}. ${f.freq} Hz  ${f.gain > 0 ? '+' : ''}${f.gain.toFixed(1)} dB  Q ${f.q}`).join('\n') + `\n  rms ${ch.rms_before.toFixed(1)} -> ${ch.rms_after.toFixed(1)} dB\n`;
  });
  r.notes.forEach(n => box.insertAdjacentHTML('beforeend', `<div class="note">${n}</div>`));
  lastResult = text;
  const b = document.createElement('button'); b.className = 'sec'; b.textContent = 'Copy settings as text';
  b.onclick = async () => { try { await navigator.clipboard.writeText(lastResult); b.textContent = 'Copied'; } catch (e) { prompt('Copy:', lastResult); } };
  box.appendChild(b);
  $('c5').classList.remove('off'); nVer = 0; $('vermeas').textContent = 'Measure verification position 1'; $('verscore').hidden = true;
};

// ---------- verify -----------------------------------------------------------------------------------------------
$('vermeas').onclick = async () => {
  const r = await runMeasure(true, 'verbar', 'vermsg'); if (!r) return;
  nVer = r.n; $('vermeas').textContent = `Measure verification position ${nVer + 1}`;
  $('vermsg').innerHTML = `<span class="ok">Verification position ${nVer} done.</span>`;
  $('verscore').hidden = false;
};
$('verscore').onclick = () => {
  const r = callPy('verify_score'); const box = $('verres'); box.innerHTML = '';
  if (!r.ok) { box.innerHTML = `<div class="note bad">${r.error}</div>`; return; }
  box.innerHTML = `<div class="big ${r.pass ? 'ok' : 'warn'}">${r.pass ? 'PASS' : 'Not yet within 2 dB'}</div>`;
  Object.entries(r.channels).forEach(([c, v]) => {
    const w = document.createElement('div');
    w.innerHTML = `<div class="msg"><b>${c}</b> ${v.rms.toFixed(1)} dB rms (was ${v.rms_before.toFixed(1)}) · worst ${v.max.toFixed(1)} dB</div>` + legend(false);
    w.appendChild(plot(c, v.plot)); box.appendChild(w);
  });
};

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
loadEngine();

// test hook (tests/test_mobile_ui.py): lets a script replace the audio hardware with recorded files
if (location.hash === '#selftest') { const s = document.createElement('script'); s.src = 'selftest.js'; document.head.appendChild(s); }
