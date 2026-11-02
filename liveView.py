'''
Live learning curves in the browser (for watching runs from the Mac). A small web server on the PC reads the CSVs in models/
(the same ones plotRun.py reads) and serves one page in the style of `plotRun.py --live --running --chain`: a panel per running run with
the population 10-90% band, raw and smoothed mean, best of generation, best ever with its records, end labels, elapsed time on top,
seconds per generation below, and the header facts (progress, ETA, speed, trend, recipe, network). Zoom by dragging, double-click resets,
hover shows the values. A second tab overlays several runs (e.g. a mirrored run against its plain twin) by generation or hours.
Refreshes every 10 s. Runs in its own process and only reads files, so it never slows the training down.

    python liveView.py                 serves port 8050 on all interfaces (tailnet: http://<machine name>:8050)
    python liveView.py --host 127.0.0.1   only this machine (then use VS Code's Ports tab or ssh -L 8050:localhost:8050)

WSL in mirrored networking mode: the tailnet works directly (Tailscale runs inside WSL). The LAN also needs an inbound rule in the
Hyper-V firewall, which blocks WSL by default (admin PowerShell on Windows, once):
    New-NetFirewallHyperVRule -Name WSL-liveView -DisplayName "WSL liveView 8050" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8050 -Action Allow
nightRun.py starts it in tmux `liveview` when nothing answers on the port, so it outlives the run and the ssh session.
'''
import argparse
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
NAMES = ["gen", "elapsed_s", "mean", "median", "p10", "p90", "best", "chosen"]
RUNNING_SECONDS = 180          # a CSV written in the last 3 minutes counts as running (same rule as plotRun --running)


class Run:
    '''One CSV, read incrementally: each refresh only parses the lines appended since the last one.'''
    def __init__(self, path):
        self.path, self.offset, self.meta, self.rows, self._array = path, 0, {}, [], None

    def update(self):
        size = os.path.getsize(self.path)
        if size < self.offset:                        # file was rewritten: start over
            self.offset, self.rows, self._array = 0, [], None
        if size == self.offset:
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1                  # keep a half-written last line for the next refresh
        self.offset += end
        for line in chunk[:end].decode(errors="replace").splitlines():
            if line.startswith("#"):
                self.meta = dict(kv.split("=", 1) for kv in line[1:].split() if "=" in kv)
            elif line and not line[0].isalpha() and line.count(",") == 7:
                try:
                    self.rows.append([float(v) for v in line.split(",")])
                except ValueError:
                    pass

    def array(self):
        if self._array is None or len(self._array) != len(self.rows):
            self._array = np.array(self.rows, dtype=float).reshape(-1, 8)
        return self._array

    def summary(self):
        a, mtime = self.array(), os.path.getmtime(self.path)
        info = dict(file=os.path.basename(self.path), name=self.meta.get("name", os.path.basename(self.path)),
                    running=time.time() - mtime < RUNNING_SECONDS, mtime=mtime, gens=len(self.rows),
                    device=self.meta.get("device", "?"), mirrored=self.meta.get("mirrored", "False"),
                    population=self.meta.get("population", "?"), games=self.meta.get("games", "?"),
                    cap=int(self.meta["generations"]) if self.meta.get("generations", "").isdigit() else None)
        if len(a):
            i0 = int(np.searchsorted(a[:, 1], a[-1, 1] - 3600))          # speed over the last hour
            info.update(gen=int(a[-1, 0]), hours=a[-1, 1] / 3600, recentMean=float(a[-200:, 2].mean()),   # recent level: last 200 generations
                        bestEver=float(a[:, 6].max()))
            if len(a) - i0 > 1 and a[-1, 1] > a[i0, 1]:
                info["gensPerHour"] = float((a[-1, 0] - a[i0, 0]) / ((a[-1, 1] - a[i0, 1]) / 3600))
        return info

    def curve(self, buckets):
        '''Bucketed curve for the compare tab (mean of each column per bucket, max for best).'''
        a = self.array()
        if len(a) == 0:
            return {n: [] for n in NAMES[:1] + NAMES[2:] + ["hours"]}
        starts = np.linspace(0, len(a), min(len(a), buckets) + 1).astype(int)[:-1]
        sizes = np.diff(np.append(starts, len(a)))
        out = {n: (np.maximum.reduceat(a[:, k], starts) if n == "best" else np.add.reduceat(a[:, k], starts) / sizes).round(2).tolist()
               for k, n in enumerate(NAMES) if n != "elapsed_s"}
        out["hours"] = (np.add.reduceat(a[:, 1], starts) / sizes / 3600).round(4).tolist()
        return out


runs = {}
lock = threading.Lock()          # the server answers requests in threads; one reader of the shared Run state at a time


def getRun(path):
    if path not in runs:
        runs[path] = Run(path)
    runs[path].update()
    return runs[path]


def scan():
    '''All CSVs in models/ (newest first).'''
    paths = sorted(glob.glob(os.path.join(MODELS, "*.csv")), key=os.path.getmtime, reverse=True)
    for p in paths:
        if p not in runs:
            runs[p] = Run(p)
    return [runs[p] for p in paths]


def resolveChain(path):
    '''Follows the 'startFrom' links in the CSV headers back to the first run (as plotRun --chain). CSV paths, oldest first.'''
    chain, seen = [path], {os.path.abspath(path)}
    while True:
        src = getRun(chain[0]).meta.get("startFrom", "None")
        if src in ("None", ""):
            break
        csv = src[:-8] + ".csv" if src.endswith("_pool.pt") else src[:-3] + ".csv" if src.endswith(".pt") else src
        csv = csv if os.path.isabs(csv) else os.path.join(BASE, csv)
        if not os.path.exists(csv) or os.path.abspath(csv) in seen:
            break
        seen.add(os.path.abspath(csv))
        chain.insert(0, csv)
    return chain


def rolling(x, window):
    '''Trailing moving average that is defined from the first point on (plotRun's trend line).'''
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    hi = np.arange(1, len(x) + 1)
    lo = np.maximum(0, hi - window)
    return (c[hi] - c[lo]) / (hi - lo)


def fmtTime(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m {seconds % 60:02d}s"


_panelCache = {}


def panel(path, buckets):
    '''Everything one run panel shows: bucketed series (min / max per bucket so spikes survive) and the header facts.'''
    paths = resolveChain(path)
    key = (buckets, tuple(getRun(p).offset for p in paths))
    if _panelCache.get(path, (None,))[0] == key:          # nothing new since the last request: no work at all
        return _panelCache[path][1]
    parts, bounds, genOff, timeOff, meta, total = [], [], 0, 0.0, {}, 0
    for p in paths:
        r = getRun(p)
        a = r.array().copy()
        if len(a) == 0:
            continue
        dt = np.diff(a[:, 1], prepend=0.0)
        if parts:
            m = r.meta
            bounds.append([genOff + 1, f"run {len(bounds) + 2}: sigma {m.get('sigma', '?')}, lr {m.get('learningRate', '?')}"])
        prevGen = genOff
        a[:, 0] += genOff
        a[:, 1] += timeOff
        parts.append(np.column_stack([a, dt]))
        genOff, timeOff, meta = int(a[-1, 0]), a[-1, 1], r.meta
        total = prevGen + int(meta.get("generations", 0) or 0)
    own = getRun(path)
    if not parts:
        return dict(empty=True, name=own.meta.get("name", os.path.basename(path)), file=os.path.basename(path))
    d = np.concatenate(parts)
    gen, el, mean, med, p10, p90, best, dt = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 8]
    n = len(gen)
    bestEver = np.maximum.accumulate(best)
    trend = rolling(mean, max(5, n // 12))
    rec = np.flatnonzero(np.diff(bestEver, prepend=-np.inf) > 0)

    edges = np.linspace(0, n, min(n, buckets) + 1).astype(int)
    st, la = edges[:-1], edges[1:] - 1
    sizes = la - st + 1
    lo = lambda y: np.minimum.reduceat(y, st)
    hi = lambda y: np.maximum.reduceat(y, st)
    avg = lambda y: np.add.reduceat(y, st) / sizes
    r0 = lambda y: np.round(y, 1).tolist()
    series = dict(x=gen[la].tolist(), hours=np.round(el[la] / 3600, 4).tolist(), p10=r0(lo(p10)), p90=r0(hi(p90)),
                  meanLo=r0(lo(mean)), meanHi=r0(hi(mean)), meanAvg=r0(avg(mean)), median=r0(avg(med)),
                  bestLo=r0(lo(best)), bestHi=r0(hi(best)), trend=r0(trend[la]), bestEver=r0(bestEver[la]), dt=np.round(hi(dt), 3).tolist(),
                  records=np.unique(np.searchsorted(la, rec)).tolist())

    # header facts (same wording as plotRun)
    ownRows = own.array()
    ownElapsed = ownRows[-1, 1] if len(ownRows) else 0.0
    perGen = float(dt[-20:].mean())
    live = time.time() - os.path.getmtime(path) < max(90.0, 8 * perGen)
    cap = None if meta.get("maxHours", "None") in ("None", "") else float(meta["maxHours"])
    timeLeft = max(0.0, cap * 3600 - ownElapsed) if cap else None
    done = int(gen[-1])
    if cap:
        eta = (f"time cap {cap:g} h reached" if timeLeft <= 0 else
               f"{fmtTime(timeLeft)} left of {cap:g} h, ends ~{time.strftime('%a %H:%M', time.localtime(time.time() + timeLeft))}" if live else
               f"stopped at {fmtTime(ownElapsed)} of {cap:g} h")
        frac = min(1.0, ownElapsed / (cap * 3600))
    elif total and done < total:
        left = (total - done) * perGen
        eta = f"ETA {fmtTime(left)}" + (f", ends ~{time.strftime('%a %H:%M', time.localtime(time.time() + left))}" if live else ", stopped")
        frac = done / total
    else:
        eta, frac = ("finished" if total else ""), (1.0 if total else None)
    recent = min(1000, n - 1)
    gain = float(trend[-1] - trend[-1 - recent]) if recent > 0 else 0.0
    sizesNet = [int(s) for s in meta["shape"].split("-")] if "shape" in meta else None
    netLine = []
    if sizesNet:
        netLine.append(f"network {' → '.join(map(str, sizesNet))}   {sum(a * b + b for a, b in zip(sizesNet, sizesNet[1:])):,} weights")
    if meta.get("steerSmoothing") not in (None, "1.0", "1"):
        netLine.append(f"steer smoothing {meta['steerSmoothing']}")
    if meta.get("actionRepeat"):
        netLine.append(f"action repeat {meta['actionRepeat']}")
    src = getRun(paths[0]).meta.get("startFrom", "None")
    netLine.append("from scratch" if src in ("None", "") else "continues " + os.path.basename(src).replace(".pt", "").replace("_pool", "")[:34])
    if len(paths) > 1:
        netLine.append(f"{len(paths)} runs chained")
    sched = lambda a, b: f"{meta.get(a, '?')}" + (f" → {meta[b]}" if meta.get(b) not in (None, "None", "") else "")
    cfgLine = (f"population {meta.get('population', '?')}  games/network {meta.get('games', '?')}  method {meta.get('method', '?')}"
               f"  sigma {sched('sigma', 'sigmaEnd')}  lr {sched('learningRate', 'learningRateEnd')}"
               + ("  mirrored" if meta.get("mirrored") == "True" else "")
               + (f"  start heights {meta.get('minStartHeight', 0)}-{meta['maxStartHeight']}" if meta.get("maxStartHeight", "None") != "None" else "  normal start"))
    out = dict(file=os.path.basename(path), name=meta.get("name", os.path.basename(path)), device=(meta.get("device") or "").upper(),
               mirrored=meta.get("mirrored") == "True", live=live, frac=frac, bounds=bounds,
               progress=f"generation {done:,}" + (f" / {total:,}" if total and not cap else ""),
               line1=f"elapsed {fmtTime(el[-1])}   |   {eta}   |   {perGen:.2f} s/gen ({3600 / max(perGen, 1e-9):,.0f} gens/h)",
               line2=(f"trend now {trend[-1]:,.0f}   ({gain:+,.0f} / {recent} gens, {100 * gain / max(trend[-1 - recent], 1e-9):+.0f}%)   |   "
                      f"best ever {bestEver[-1]:,.0f}, set {done - int(gen[rec[-1]])} gens ago   |   {len(rec)} records"),
               cfg=cfgLine, net="   |   ".join(netLine),
               ends=dict(bestEver=float(bestEver[-1]), best=float(best[-1]), trend=float(trend[-1])),
               unit="minutes" if el[-1] < 5400 else "hours", series=series)
    _panelCache[path] = (key, out)
    return out


class Handler(BaseHTTPRequestHandler):
    def send(self, body, kind):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with lock:
            self.route()

    def route(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        buckets = max(100, min(5000, int(q.get("buckets", ["1500"])[0])))
        files = [os.path.join(MODELS, os.path.basename(f)) for f in q.get("file", [])]
        files = [f for f in files if os.path.exists(f)]
        if url.path == "/":
            self.send(PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/runs":
            listed = scan()[:200]                       # every run in models/ (each CSV is read once, then only its new lines)
            for r in listed:
                if r.offset == 0 or time.time() - os.path.getmtime(r.path) < RUNNING_SECONDS:
                    r.update()
            out = [r.summary() for r in listed]
            continued = {os.path.basename(p) for r in listed if r.path and time.time() - os.path.getmtime(r.path) < RUNNING_SECONDS
                         for p in resolveChain(r.path)[:-1]}
            for o in out:
                o["continued"] = o["file"] in continued   # a run that a running run continues: shown inside that run's panel
            self.send(json.dumps(out), "application/json")
        elif url.path == "/api/panel":
            self.send(json.dumps({os.path.basename(f): panel(f, buckets) for f in files}), "application/json")
        elif url.path == "/api/curve":
            self.send(json.dumps({os.path.basename(f): getRun(f).curve(buckets) for f in files}), "application/json")
        else:
            self.send_error(404)

    def log_message(self, *args):                     # no line per request in the console
        pass


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sprudler Live</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js"></script>
<style>
/* plotRun's themes: validated categorical slots 1-3 (blue, orange, aqua), stepped per theme */
:root { --surface:#1a1a19; --panel:#1f1f1d; --ink:#f2f1ec; --ink2:#c3c2b7; --muted:#8a8980; --grid:#2e2d2a;
        --blue:#3987e5; --orange:#d95926; --aqua:#199e70; }
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) { --surface:#fcfcfb; --panel:#f6f5f1; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8980;
        --grid:#e8e7e2; --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; } }
:root[data-theme="light"] { --surface:#fcfcfb; --panel:#f6f5f1; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8980; --grid:#e8e7e2;
        --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--surface); color:var(--ink); font:14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif; }
.top { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
.top h1 { font-size:18px; margin:0; }
.top small { color:var(--muted); font-size:12px; }
.tabs { display:flex; gap:4px; }
.tabs button, .controls button { background:var(--panel); color:var(--ink2); border:1px solid var(--grid); border-radius:6px; padding:4px 12px; font:inherit; cursor:pointer; }
.tabs button.on { color:var(--ink); border-color:var(--muted); }
.controls { display:flex; gap:14px; flex-wrap:wrap; align-items:center; color:var(--ink2); }
select, input[type=number], input[type=search] { background:var(--panel); color:var(--ink); border:1px solid var(--grid); border-radius:6px; padding:3px 6px; font:inherit; }
.panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(640px, 100%), 1fr)); gap:16px; }
.card { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:14px 14px 8px; min-width:0; }
.head { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
.name { font-size:19px; font-weight:700; }
.badges { display:flex; gap:6px; }
.badge { font-size:12px; font-weight:700; padding:2px 8px; border-radius:5px; color:#fff; }
.bar { height:4px; background:var(--grid); border-radius:2px; margin:8px 0 6px; overflow:hidden; }
.bar > div { height:100%; background:var(--blue); }
.facts { color:var(--ink2); font-size:13px; margin:2px 0; font-variant-numeric:tabular-nums; }
.facts.m { color:var(--muted); font-size:12px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--ink2); font-size:12px; margin:10px 0 2px; }
.key { display:inline-block; width:14px; height:3px; margin-right:6px; vertical-align:middle; border-radius:2px; }
.key.box { height:10px; opacity:.45; }
.readout { height:36px; overflow:hidden; color:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }
.readout span { color:var(--muted); }
.wait { color:var(--ink2); padding:60px 0; text-align:center; }
.wrap { overflow-x:auto; margin-top:18px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { color:var(--muted); font-weight:500; font-size:12px; }
td.num, th.num { text-align:right; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.live { color:var(--aqua); font-weight:600; }
.old { color:var(--muted); }
#compareChart { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:8px; margin-top:12px; }
.u-legend { color:var(--ink2); }
.hidden { display:none !important; }
</style></head><body>
<div class="top">
  <h1>Sprudler Live</h1>
  <div class="tabs"><button id="tabRuns" class="on">runs</button><button id="tabCompare">compare</button></div>
  <div class="controls">
    <label>points <select id="buckets"><option>600</option><option selected>1500</option><option>3000</option></select></label>
    <label><input type="checkbox" id="logy"> log y</label>
    <label><input type="checkbox" id="showOld"> finished runs in the table</label>
    <label>filter <input type="search" id="filter" placeholder="run name" style="width:120px"></label>
    <button id="theme">theme</button>
  </div>
  <small id="stamp"></small>
</div>

<div id="viewRuns"><div class="panels" id="panels"></div></div>

<div id="viewCompare" class="hidden">
  <div class="controls">
    <label>y <select id="metric"><option value="mean">mean</option><option value="median">median</option><option value="p90">p90</option><option value="p10">p10</option><option value="best">best</option></select></label>
    <label>x <select id="xaxis"><option value="gen">generation</option><option value="hours">hours</option></select></label>
    <label>smooth <input type="number" id="smooth" value="1" min="1" max="200" style="width:64px"> points</label>
  </div>
  <div id="compareChart"></div>
</div>

<div class="wrap"><table><thead><tr><th>show</th><th>run</th><th>status</th><th class="num">gen</th><th class="num">cap</th><th class="num">hours</th>
<th class="num">mean (last 200)</th><th class="num">best ever</th><th class="num">gens/h (last hour)</th><th class="num">ETA</th><th>setup</th></tr></thead>
<tbody id="runs"></tbody></table></div>

<script>
const PALETTE = ["#3987e5", "#d95926", "#199e70", "#c548a8", "#d9a400", "#7d5fd6", "#3fa8c9", "#8a8980"];
const $ = id => document.getElementById(id);
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
                set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} } };
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const num = v => v == null || isNaN(v) ? "" : Math.round(v).toLocaleString("en-US");
const short = v => v == null ? "" : Math.abs(v) >= 1000 ? (v / 1000).toFixed(Math.abs(v) >= 1e5 ? 0 : 1) + "k" : v.toFixed(0);
const dpr = () => uPlot.pxRatio;

let view = store.get("view", "runs"), extra = store.get("extra", []), hiddenRuns = store.get("hiddenRuns", []), cmpPicked = store.get("cmpPicked", null);
let listing = [], panels = {}, colorOf = {}, cmp = null, cmpKey = "";
const theme = store.get("theme", null); if (theme) document.documentElement.dataset.theme = theme;
$("buckets").value = store.get("buckets", "1500"); $("logy").checked = store.get("logy", false);
$("metric").value = store.get("metric", "mean"); $("xaxis").value = store.get("xaxis", "gen"); $("smooth").value = store.get("smooth", 1);

function fmtTime(h) { const m = Math.round(h * 60); return m >= 60 ? `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m` : `${m}m`; }
function interp(xs, ys, x) {                       // piecewise linear (top axis: generation -> elapsed time)
  if (!xs.length) return 0; if (x <= xs[0]) return ys[0]; if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / ((xs[hi] - xs[lo]) || 1);
}
function spread(ys, minGap, top, bottom) {        // end labels: nudge apart, keep order (plotRun.spread)
  const order = ys.map((y, i) => i).sort((a, b) => ys[b] - ys[a]), out = ys.slice();   // screen y: larger = lower
  for (let k = 1; k < order.length; k++) { const a = order[k - 1], b = order[k]; if (out[a] - out[b] < minGap) out[b] = out[a] - minGap; }
  const highest = Math.min(...out); if (highest < top) out.forEach((v, i) => out[i] = v + (top - highest));
  return out;
}

// ---------- one run panel ----------
function makePanel(file) {
  const el = document.createElement("div"); el.className = "card";
  el.innerHTML = `<div class="head"><div class="name"></div><div class="badges"></div></div><div class="bar"><div></div></div>
    <div class="facts p"></div><div class="facts l2"></div><div class="facts m cfg"></div><div class="facts m net"></div>
    <div class="legend"><span><i class="key" style="background:var(--blue)"></i>population mean (trend)</span>
      <span><i class="key box" style="background:var(--blue)"></i>population 10-90%</span>
      <span><i class="key" style="background:var(--orange)"></i>best of generation</span>
      <span><i class="key" style="background:var(--aqua)"></i>best ever</span></div>
    <div class="readout"></div><div class="main"></div><div class="dt"></div>`;
  $("panels").appendChild(el);
  const p = { file, el, main: null, dt: null, data: null, zoomed: false };
  new ResizeObserver(() => { if (p.main) { const w = el.querySelector(".main").clientWidth; p.main.setSize({ width: w, height: mainHeight() }); p.dt.setSize({ width: w, height: 130 }); } }).observe(el);
  return p;
}
const mainHeight = () => Math.max(300, Math.min(520, Math.round(window.innerHeight * 0.48)));

function buildCharts(p) {
  const s = p.data.series, width = p.el.querySelector(".main").clientWidth, log = $("logy").checked;
  const C = { blue: css("--blue"), orange: css("--orange"), aqua: css("--aqua"), grid: css("--grid"), ink: css("--ink"), ink2: css("--ink2"), muted: css("--muted"), surface: css("--panel") };
  const hidden = { stroke: "transparent", points: { show: false }, spanGaps: true };
  const axis = { stroke: C.ink2, grid: { stroke: C.grid, width: 1 }, ticks: { show: false }, font: "11px system-ui", labelFont: "12px system-ui" };
  const syncKey = "sync-" + p.file;
  const opts = {
    width, height: mainHeight(), padding: [8, 150, 0, 4],
    cursor: { sync: { key: syncKey, setSeries: false }, drag: { x: true, y: true, uni: 40 }, points: { show: false } },
    legend: { show: false },
    scales: { x: { time: false }, y: { distr: log ? 3 : 1, range: (u, lo, hi) => {
      const top = Math.max(...p.data.series.bestEver.slice(-1), hi || 1);
      if (p.zoomed) return [lo, hi];
      return log ? [Math.max(1, lo * 0.8), top * 1.35] : [0, top * 1.10]; } } },
    axes: [
      { ...axis, scale: "x", side: 2, values: (u, v) => v.map(x => x.toLocaleString("en-US")), size: 30 },
      { ...axis, scale: "y", side: 3, size: 64, label: "score per game (points)", values: (u, v) => v.map(num) },
      { ...axis, scale: "x", side: 0, size: 34, grid: { show: false }, label: `elapsed ${p.data.unit}`, labelSize: 14, labelFont: "11px system-ui",
        splits: u => timeSplits(p, u),
        values: (u, v) => v.map(x => { const h = interp(p.data.series.x, p.data.series.hours, x) * (p.data.unit === "hours" ? 1 : 60); return +h.toFixed(2) + ""; }) },
    ],
    series: [{}, { ...hidden }, { ...hidden }, { ...hidden }, { ...hidden }, { ...hidden }, { ...hidden },
      { stroke: C.blue, width: 2, points: { show: false }, spanGaps: true },
      { stroke: C.aqua, width: 2, paths: uPlot.paths.stepped({ align: 1 }), spanGaps: true,
        points: { show: true, size: 8, width: 2, fill: C.aqua, stroke: C.surface, filter: () => p.data.series.records } }],
    bands: [{ series: [2, 1], fill: C.blue + "2e" }, { series: [4, 3], fill: C.blue + "59" }, { series: [6, 5], fill: C.orange + "b0" }],
    hooks: {
      setSelect: [u => { if (u.select.width > 0 || u.select.height > 0) p.zoomed = true; }],
      setScale: [(u, key) => { if (key === "x" && p.dt) { const { min, max } = u.scales.x; if (p.dt.scales.x.min !== min || p.dt.scales.x.max !== max) p.dt.setScale("x", { min, max }); } }],
      setCursor: [u => readout(p, u.cursor.idx)],
      draw: [u => drawOverlay(p, u, C)],
      ready: [u => u.over.addEventListener("dblclick", () => { p.zoomed = false; u.setData(seriesData(p), true); })],
    },
  };
  p.el.querySelector(".main").innerHTML = ""; p.el.querySelector(".dt").innerHTML = "";
  p.main = new uPlot(opts, seriesData(p), p.el.querySelector(".main"));
  p.dt = new uPlot({
    width, height: 130, padding: [6, 150, 0, 4], legend: { show: false },
    cursor: { sync: { key: syncKey }, drag: { x: false, y: false }, points: { show: false } },
    scales: { x: { time: false }, y: { range: () => [0, dtTop(p.data.series.dt)] } },
    axes: [{ ...axis, label: "generation", size: 44, values: (u, v) => v.map(x => x.toLocaleString("en-US")) },
           { ...axis, side: 3, size: 64, space: 18, label: "sec / gen", values: (u, v) => v.map(x => x < 1 ? x.toFixed(2) : x.toFixed(1)) }],
    series: [{}, { stroke: "transparent", fill: C.muted + "8c", paths: uPlot.paths.stepped({ align: 1 }), points: { show: false } }],
  }, [s.x, s.dt], p.el.querySelector(".dt"));
}
function dtTop(dt) {                              // the first generation includes compiling: a percentile, not the max, sets the scale
  const v = dt.filter(x => x != null).sort((a, b) => a - b); if (!v.length) return 1;
  return Math.max(v[Math.floor(0.995 * (v.length - 1))], 1e-3) * 1.25;
}
function niceStep(range, count) { const raw = range / count, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag; }
function timeSplits(p, u) {                       // top axis: round elapsed hours (or minutes), placed at their generation
  const s = p.data.series, k = p.data.unit === "hours" ? 1 : 60, xmin = u.scales.x.min, xmax = u.scales.x.max;
  const t0 = interp(s.x, s.hours, xmin) * k, t1 = interp(s.x, s.hours, xmax) * k; if (!(t1 > t0)) return [];
  const step = niceStep(t1 - t0, Math.max(2, Math.floor(u.bbox.width / uPlot.pxRatio / 90))), out = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1 + 1e-9; t += step) out.push(interp(s.hours, s.x, t / k));
  return out;
}
const seriesData = p => { const s = p.data.series; return [s.x, s.p10, s.p90, s.meanLo, s.meanHi, s.bestLo, s.bestHi, s.trend, s.bestEver]; };

function drawOverlay(p, u, C) {                   // chain boundaries + end labels in the right gutter (plotRun style)
  const ctx = u.ctx, r = dpr(), s = p.data.series, bb = u.bbox;
  ctx.save();
  for (const [g, label] of p.data.bounds) {
    const x = u.valToPos(g - 0.5, "x", true); if (x < bb.left || x > bb.left + bb.width) continue;
    ctx.strokeStyle = C.muted; ctx.setLineDash([4 * r, 3 * r]); ctx.lineWidth = r;
    ctx.beginPath(); ctx.moveTo(x, bb.top); ctx.lineTo(x, bb.top + bb.height); ctx.stroke(); ctx.setLineDash([]);
    ctx.save(); ctx.translate(x + 12 * r, bb.top + bb.height - 6 * r); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = C.muted; ctx.font = `${11 * r}px system-ui`; ctx.textAlign = "left"; ctx.fillText(label, 0, 0); ctx.restore();
  }
  const lastX = s.x[s.x.length - 1], xPx = u.valToPos(lastX, "x", true);
  if (xPx >= bb.left - 1 && xPx <= bb.left + bb.width + 1) {
    const ends = [["best ever", p.data.ends.bestEver, C.aqua], ["best of generation", p.data.ends.best, C.orange], ["population mean", p.data.ends.trend, C.blue]];
    const ys = ends.map(e => u.valToPos(e[1], "y", true));
    const placed = spread(ys, 36 * r, bb.top + 14 * r, bb.top + bb.height);
    const tx = bb.left + bb.width + 18 * r;
    ends.forEach(([label, v, color], i) => {
      ctx.strokeStyle = C.muted; ctx.lineWidth = 0.8 * r;
      ctx.beginPath(); ctx.moveTo(xPx, ys[i]); ctx.lineTo(tx - 4 * r, placed[i]); ctx.stroke();
      ctx.beginPath(); ctx.arc(xPx, ys[i], 5 * r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2 * r; ctx.strokeStyle = C.surface; ctx.stroke();
      ctx.fillStyle = C.ink; ctx.font = `${12 * r}px system-ui`; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      ctx.fillText(label, tx, placed[i] - 8 * r); ctx.fillText(num(v), tx, placed[i] + 8 * r);
    });
  }
  ctx.restore();
}

function readout(p, i) {
  const el = p.el.querySelector(".readout"), s = p.data && p.data.series;
  if (i == null || !s) { el.innerHTML = ""; return; }
  el.innerHTML = `<span>gen</span> ${num(s.x[i])} &nbsp; <span>${p.data.unit === "hours" ? "h" : "min"}</span> ${p.data.unit === "hours" ? s.hours[i].toFixed(2) : (s.hours[i] * 60).toFixed(1)}
    &nbsp; <span>mean</span> ${num(s.meanAvg[i])} &nbsp; <span>trend</span> ${num(s.trend[i])} &nbsp; <span>median</span> ${num(s.median[i])}
    &nbsp; <span>p10-p90</span> ${num(s.p10[i])}-${num(s.p90[i])} &nbsp; <span>best</span> ${num(s.bestHi[i])} &nbsp; <span>best ever</span> ${num(s.bestEver[i])}
    &nbsp; <span>s/gen</span> ${s.dt[i].toFixed(2)}`;
}

function fillHeader(p) {
  const d = p.data, el = p.el;
  el.querySelector(".name").textContent = d.name;
  el.querySelector(".badges").innerHTML = (d.mirrored ? `<span class="badge" style="background:#7d5fd6">MIRRORED</span>` : "")
    + (d.device ? `<span class="badge" style="background:${d.device === "GPU" ? "var(--aqua)" : "var(--blue)"}">${d.device}</span>` : "")
    + (d.live ? "" : `<span class="badge" style="background:var(--muted)">DONE</span>`);
  el.querySelector(".bar").classList.toggle("hidden", d.frac == null);
  el.querySelector(".bar > div").style.width = ((d.frac || 0) * 100).toFixed(2) + "%";
  el.querySelector(".p").textContent = `${d.progress}   |   ${d.line1}`;
  el.querySelector(".l2").textContent = d.line2;
  el.querySelector(".cfg").textContent = d.cfg;
  el.querySelector(".net").textContent = d.net;
}

async function refreshPanels() {
  const want = listing.filter(r => (r.running && !r.continued && !hiddenRuns.includes(r.file)) || extra.includes(r.file)).map(r => r.file);
  for (const f of Object.keys(panels)) if (!want.includes(f)) { panels[f].el.remove(); delete panels[f]; }
  if (!want.length) { $("panels").innerHTML = `<div class="wait">no run is going right now - tick one in the table below</div>`; return; }
  $("panels").querySelector(".wait")?.remove();
  const qs = want.map(f => "file=" + encodeURIComponent(f)).join("&") + "&buckets=" + $("buckets").value;
  const data = await (await fetch("api/panel?" + qs)).json();
  for (const f of want) {
    const d = data[f]; if (!d) continue;
    const p = panels[f] || (panels[f] = makePanel(f));
    if (d.empty) { p.el.querySelector(".name").textContent = d.name; p.el.querySelector(".main").innerHTML = `<div class="wait">waiting for the first generation...</div>`; continue; }
    p.data = d; fillHeader(p);
    if (!p.main) buildCharts(p);
    else { p.main.setData(seriesData(p), !p.zoomed); p.dt.setData([d.series.x, d.series.dt], !p.zoomed); if (p.zoomed) p.dt.setScale("x", { min: p.main.scales.x.min, max: p.main.scales.x.max }); }
  }
}
function rebuildPanels() { for (const p of Object.values(panels)) if (p.data) buildCharts(p); }

// ---------- compare tab: several runs overlaid ----------
async function refreshCompare() {
  if (cmpPicked === null) cmpPicked = listing.filter(r => r.running).map(r => r.file);
  const files = cmpPicked.filter(f => f in colorOf);
  const metric = $("metric").value, xaxis = $("xaxis").value, k = Math.max(1, +$("smooth").value | 0);
  const data = files.length ? await (await fetch("api/curve?" + files.map(f => "file=" + encodeURIComponent(f)).join("&") + "&buckets=" + $("buckets").value)).json() : {};
  const tables = [], series = [{ label: xaxis === "gen" ? "generation" : "hours", value: (u, v) => v == null ? "" : xaxis === "gen" ? num(v) : v.toFixed(2) }];
  for (const f of files) {
    const c = data[f]; if (!c || !c.gen.length) continue;
    const y = c[metric].map((v, i, a) => { const lo = Math.max(0, i - k + 1); let s = 0; for (let j = lo; j <= i; j++) s += a[j]; return s / (i - lo + 1); });
    tables.push([c[xaxis], y]);
    series.push({ label: f.replace(/_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.csv$/, ""), stroke: colorOf[f], width: 2, spanGaps: true, points: { show: false }, value: (u, v) => num(v) });
  }
  const joined = tables.length ? uPlot.join(tables) : [[]];
  const key = [files.join(), metric, xaxis, $("logy").checked, document.documentElement.dataset.theme].join("|");
  const width = $("compareChart").clientWidth - 16;
  if (cmp && key === cmpKey) { cmp.setData(joined, !cmp._zoomed); return; }
  if (cmp) cmp.destroy();
  cmpKey = key;
  const axis = { stroke: css("--ink2"), grid: { stroke: css("--grid") }, ticks: { show: false } };
  cmp = new uPlot({ width, height: Math.max(320, Math.min(620, window.innerHeight - 320)), series,
    scales: { x: { time: false }, y: { distr: $("logy").checked ? 3 : 1 } },
    axes: [{ ...axis, label: xaxis === "gen" ? "generation" : "elapsed hours", values: (u, v) => v.map(x => xaxis === "gen" ? x.toLocaleString("en-US") : x.toFixed(1)) },
           { ...axis, size: 64, label: metric + " score", values: (u, v) => v.map(num) }],
    cursor: { drag: { x: true, y: true, uni: 40 } },
    hooks: { setSelect: [u => { if (u.select.width > 0) u._zoomed = true; }], ready: [u => u.over.addEventListener("dblclick", () => u._zoomed = false)] },
  }, joined, $("compareChart"));
}

// ---------- table + loop ----------
function fillTable() {
  const q = $("filter").value.trim().toLowerCase();       // compare tab: every run in models/, historic ones too
  const shown = listing.filter(r => (view === "compare" || r.running || $("showOld").checked || extra.includes(r.file)) && (!q || r.name.toLowerCase().includes(q)));
  $("runs").innerHTML = shown.map(r => {
    const on = view === "runs" ? (r.running && !r.continued ? !hiddenRuns.includes(r.file) : extra.includes(r.file)) : (cmpPicked || []).includes(r.file);
    const left = r.cap && r.gensPerHour ? (r.cap - r.gen) / r.gensPerHour : null;
    return `<tr><td><input type="checkbox" data-file="${r.file}" ${on ? "checked" : ""}></td>
      <td><span class="dot" style="background:${colorOf[r.file]}"></span>${r.name}</td>
      <td class="${r.running ? "live" : "old"}">${r.running ? (r.continued ? "continued" : "running") : "done"}</td>
      <td class="num">${num(r.gen)}</td><td class="num">${num(r.cap)}</td><td class="num">${r.hours != null ? r.hours.toFixed(1) : ""}</td>
      <td class="num">${num(r.recentMean)}</td><td class="num">${num(r.bestEver)}</td><td class="num">${r.gensPerHour ? num(r.gensPerHour) : ""}</td>
      <td class="num">${r.running && left != null ? fmtTime(left) : ""}</td>
      <td class="old">${r.population}x${r.games} ${r.device}${r.mirrored === "True" ? " mirrored" : ""}</td></tr>`; }).join("");
  document.querySelectorAll("#runs input").forEach(cb => cb.onchange = () => {
    const f = cb.dataset.file, r = listing.find(x => x.file === f);
    if (view === "compare") { cmpPicked = cb.checked ? [...cmpPicked, f] : cmpPicked.filter(x => x !== f); store.set("cmpPicked", cmpPicked); }
    else if (r && r.running && !r.continued) { hiddenRuns = cb.checked ? hiddenRuns.filter(x => x !== f) : [...hiddenRuns, f]; store.set("hiddenRuns", hiddenRuns); }
    else { extra = cb.checked ? [...extra, f] : extra.filter(x => x !== f); store.set("extra", extra); }
    refresh();
  });
}

let busy = false;
async function refresh() {
  if (busy) return; busy = true;
  try {
    listing = await (await fetch("api/runs")).json();
    listing.forEach(r => { if (!(r.file in colorOf)) colorOf[r.file] = PALETTE[Object.keys(colorOf).length % PALETTE.length]; });
    if (view === "runs") await refreshPanels(); else await refreshCompare();
    fillTable();
    $("stamp").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) { $("stamp").textContent = "connection lost - retrying (" + new Date().toLocaleTimeString() + ")"; }
  busy = false;
}
function setView(v) {
  view = v; store.set("view", v);
  $("viewRuns").classList.toggle("hidden", v !== "runs"); $("viewCompare").classList.toggle("hidden", v !== "compare");
  $("tabRuns").classList.toggle("on", v === "runs"); $("tabCompare").classList.toggle("on", v === "compare");
  if (cmp) { cmp.destroy(); cmp = null; }
  refresh();
}
$("tabRuns").onclick = () => setView("runs"); $("tabCompare").onclick = () => setView("compare");
$("buckets").onchange = () => { store.set("buckets", $("buckets").value); refresh(); };
$("logy").onchange = () => { store.set("logy", $("logy").checked); rebuildPanels(); if (cmp) { cmp.destroy(); cmp = null; } refresh(); };
for (const id of ["metric", "xaxis", "smooth"]) $(id).onchange = () => { store.set(id, $(id).value); if (cmp) { cmp.destroy(); cmp = null; } refresh(); };
$("showOld").onchange = fillTable; $("filter").oninput = fillTable;
$("theme").onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark"; store.set("theme", document.documentElement.dataset.theme);
  rebuildPanels(); if (cmp) { cmp.destroy(); cmp = null; } refresh();
};
window.addEventListener("resize", () => { if (cmp) { cmp.setSize({ width: $("compareChart").clientWidth - 16, height: cmp.height }); } });
setView(view);
setInterval(() => { if (!document.hidden) refresh(); }, 10000);      // a hidden tab asks the PC for nothing
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
</script></body></html>
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="live learning curves in the browser")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = LAN + tailnet (default), 127.0.0.1 = only via port forward")
    a = ap.parse_args()
    os.nice(19)                                       # lowest priority: the training always comes first
    print(f"liveView on http://{a.host}:{a.port}  (Ctrl+C stops)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
