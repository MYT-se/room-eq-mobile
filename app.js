'use strict';
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const PYFILES = ['py/mobile.py', 'py/roomeq/__init__.py', 'py/roomeq/dsp.py', 'py/roomeq/sweep.py', 'py/roomeq/calfile.py'];
const TAIL_S = 9;            // extra recording after the playback ends: AirPlay/Bluetooth delay the sound by seconds
const EXT_WAIT_S = 40;      // other-device mode: time to press play on the other device, after the signal's own length
const LEAD_S = 0.6;          // recording starts this long before the playback
const $ = id => document.getElementById(id);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

// Inside the iOS app (ios/RoomEQ) the audio runs natively: AirPlay output and the USB microphone at the same time.
const NATIVE = !!(window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.roomeq);
const nat = (cmd, args) => window.webkit.messageHandlers.roomeq.postMessage(Object.assign({ cmd }, args || {}));
let nativeRate = 48000;
const recRate = () => NATIVE ? nativeRate : (ctx ? ctx.sampleRate : 48000);
function f32ToB64(f) {
  const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = '';
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}
function b64ToF32(b) {
  const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
window.__nativeMeter = (db, peak) => showMeter(db, peak);

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
  if (NATIVE) {
    const r = await nat('startMic');
    nativeRate = r.rate || 48000;
    $('micbtn').textContent = 'Microphone on'; $('lvlbtn').disabled = false; $('tonebtn').disabled = false;
    if (!/umik/i.test(r.input || '')) $('meterTxt').innerHTML = '<span class="warn">Input: ' + (r.input || 'unknown') + ' - the UMIK-1 is not the active input. Plug it in and tap again.</span>';
    else $('meterTxt').textContent = 'Input: ' + r.input + ' (' + nativeRate + ' Hz)';
    return;
  }
  applySession();
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
  $('lvlbtn').disabled = false; $('tonebtn').disabled = false;
  const label = (devs.find(d => d.deviceId === cur) || {}).label || '';
  if (!/umik/i.test(label)) $('meterTxt').innerHTML = '<span class="warn">The UMIK-1 is not the selected input (' + (label || 'unknown') + '). Plug it in and choose it above.</span>';
}
function applySession() {
  const t = store.get('asess', 'auto');
  try { if (navigator.audioSession) navigator.audioSession.type = t; } catch (e) { /* not supported */ }
}
$('asess').value = store.get('asess', 'auto');
$('asess').onchange = async e => { store.set('asess', e.target.value); applySession(); $('toneres').textContent = 'Session set to ' + e.target.value + (navigator.audioSession ? '' : ' (this browser has no audioSession API)') + '. Tap the mic button again, then play the tone.'; };
$('tonebtn').onclick = async () => {
  if (NATIVE) { $('toneres').textContent = (await nat('tone')).text; return; }
  await ctx.resume();
  const o = ctx.createOscillator(), g = ctx.createGain(); o.frequency.value = 1000; g.gain.value = 0.1; o.connect(g); g.connect(ctx.destination);
  let peak = -120; const t0 = performance.now(); const iv = setInterval(() => { const m = $('meterTxt').textContent.match(/(-?\d+) dBFS/); if (m) peak = Math.max(peak, +m[1]); }, 100);
  o.start(); await sleep(2000); o.stop(); clearInterval(iv);
  $('toneres').textContent = `Tone played. Mic peak during the tone: ${peak} dBFS (room noise is about -80 to -60; a tone you can hear should read -50 or higher). Session: ${navigator.audioSession ? navigator.audioSession.type : 'API missing'}.`;
};
$('micbtn').onclick = async () => {
  if (!needEngine()) return;
  try { await startMic(); } catch (e) { $('meterTxt').innerHTML = '<span class="bad">Microphone: ' + e.message + '</span>'; }
};
$('dev').onchange = e => startMic(e.target.value);

let meterAt = 0;
function meter(buf) {
  const now = performance.now(); if (now - meterAt < 120) return; meterAt = now;
  let s = 0, p = 0; for (let i = 0; i < buf.length; i++) { const v = buf[i]; s += v * v; if (Math.abs(v) > p) p = Math.abs(v); }
  showMeter(20 * Math.log10(Math.sqrt(s / buf.length) + 1e-9), p);
}
function showMeter(db, p) {
  $('meter').style.width = Math.max(0, Math.min(100, (db + 100) / 100 * 100)) + '%';
  $('meter').style.background = p > 0.98 ? 'var(--bad)' : 'var(--ok)';
  if (!busy) $('meterTxt').textContent = `Microphone ${db.toFixed(0)} dBFS` + (p > 0.98 ? ' · CLIPPING' : '');
}

// ---------- test signal as a WAV file (to play from another device) -----------------------------------------------
function wavBlob(flat, nch) {
  const frames = flat.length / nch, bytes = frames * nch * 2, v = new DataView(new ArrayBuffer(44 + bytes));
  const w = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + bytes, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, nch, true);
  v.setUint32(24, 48000, true); v.setUint32(28, 48000 * nch * 2, true); v.setUint16(32, nch * 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, bytes, true);
  let o = 44;
  for (let i = 0; i < frames; i++) for (let c = 0; c < nch; c++) { const x = Math.max(-1, Math.min(1, flat[c * frames + i])); v.setInt16(o, Math.round(x * 32767), true); o += 2; }
  return new Blob([v], { type: 'audio/wav' });
}
async function saveWav(short) {
  if (!needEngine()) return;
  const p = mobile.plan_audio(level, short); const [flat, nch] = p.toJs(); p.destroy();
  const name = `roomeq-${short ? 'levelcheck' : 'measure'}-${nch === 1 ? 'mono' : 'stereo'}.wav`;
  const file = new File([wavBlob(flat, nch)], name, { type: 'audio/wav' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) { try { await navigator.share({ files: [file] }); return; } catch (e) { if (e.name === 'AbortError') return; } }
  const a = document.createElement('a'); a.href = URL.createObjectURL(file); a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
$('wavshort').onclick = () => saveWav(true);
$('wavfull').onclick = () => saveWav(false);
$('src').value = store.get('src', 'self');
$('src').onchange = e => { store.set('src', e.target.value); $('extbox').hidden = e.target.value !== 'ext'; };
$('extbox').hidden = $('src').value !== 'ext';

// ---------- play + record ----------------------------------------------------------------------------------------
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
async function playRecord(short, onProgress) {
  const [flat, nch, seconds] = (() => { const p = mobile.plan_audio(level, short); const j = p.toJs(); p.destroy(); return j; })();
  if (window.__fakeRecord) return window.__fakeRecord(short);
  if (NATIVE) {
    const total = LEAD_S + seconds + TAIL_S, t0 = performance.now();
    const iv = setInterval(() => onProgress(Math.min(1, (performance.now() - t0) / 1000 / total)), 150);
    try {
      const r = await nat('playRecord', { nch, seconds, lead: LEAD_S, tail: TAIL_S, data: f32ToB64(flat) });
      nativeRate = r.rate || nativeRate;
      return b64ToF32(r.data);
    } finally { clearInterval(iv); }
  }
  const ext = $('src').value === 'ext';
  mobile.set_search(ext ? 45 : 9);
  const frames = flat.length / nch;
  const buf = ctx.createBuffer(nch, frames, 48000);
  for (let c = 0; c < nch; c++) buf.copyToChannel(flat.subarray(c * frames, (c + 1) * frames), c);
  let lock = null; try { lock = await navigator.wakeLock.request('screen'); } catch (e) { /* not available */ }
  await ctx.resume();
  chunks = [];
  await sleep(LEAD_S * 1000);
  const t0 = performance.now();
  if (!ext) { const srcNode = ctx.createBufferSource(); srcNode.buffer = buf; srcNode.connect(ctx.destination); srcNode.start(); }
  const total = ext ? seconds + EXT_WAIT_S : seconds + TAIL_S;
  if (ext) $('meterTxt').textContent = `Recording - START THE ${short ? 'LEVEL-CHECK' : 'MEASUREMENT'} SIGNAL on the other device now (within ${EXT_WAIT_S - 5} s).`;
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
    const r = callPy('check_level', rec, recRate());
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
    const r = callPy('measure', rec, recRate(), verify);
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

if (NATIVE) {
  ['src', 'extbox', 'routebox', 'asess', 'devlbl', 'dev'].forEach(i => { const e = $(i); if (e) e.hidden = true; });
  const l = $('src').previousElementSibling; if (l) l.hidden = true;
  $('routebox').hidden = false; $('routebox').querySelector('summary').textContent = 'Play a test tone';
  $('asess').previousElementSibling.hidden = true;
}
if ('serviceWorker' in navigator && !NATIVE) navigator.serviceWorker.register('sw.js').catch(() => {});
loadEngine();

// test hook (tests/test_mobile_ui.py): lets a script replace the audio hardware with recorded files
if (location.hash === '#selftest') { const s = document.createElement('script'); s.src = 'selftest.js'; document.head.appendChild(s); }
