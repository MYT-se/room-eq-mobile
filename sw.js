// Offline cache: the page, the Python sources and (after the first load) the Pyodide runtime from the CDN.
const V = 'roomeq-m5';
const OWN = ['./', 'index.html', 'app.js', 'worklet.js', 'manifest.json', 'icon.png', 'py/mobile.py', 'py/roomeq/__init__.py', 'py/roomeq/dsp.py', 'py/roomeq/sweep.py', 'py/roomeq/calfile.py'];
self.addEventListener('install', e => { e.waitUntil(caches.open(V).then(c => c.addAll(OWN)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const cdn = e.request.url.startsWith('https://cdn.jsdelivr.net/pyodide/');
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(r => {
    if (r.ok && (cdn || e.request.url.startsWith(self.location.origin))) { const copy = r.clone(); caches.open(V).then(c => c.put(e.request, copy)); }
    return r;
  })));
});
