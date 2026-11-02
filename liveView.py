'''
Live learning curves in the browser (for watching runs from the Mac). A small web server on the PC reads the CSVs in models/
(the same ones plotRun.py reads) and serves one page with an interactive chart: zoom by dragging, double-click to reset, hover for
values, x axis by generation or by hours, several runs overlaid (e.g. a mirrored run against its plain twin). Redraws every 10 s.
Runs in its own process and only reads files, so it never slows the training down.

    python liveView.py                 serves port 8050 on all interfaces (tailnet: http://<machine name>:8050)
    python liveView.py --host 127.0.0.1   only this machine (then use VS Code's Ports tab or ssh -L 8050:localhost:8050)

WSL in mirrored networking mode: the tailnet works directly (Tailscale runs inside WSL). The LAN also needs an inbound rule in the
Hyper-V firewall, which blocks WSL by default (admin PowerShell on Windows, once):
    New-NetFirewallHyperVRule -Name WSL-liveView -DisplayName "WSL liveView 8050" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8050 -Action Allow
Start it in tmux (`tmux new -d -s liveview ./venv/bin/python liveView.py`) so it outlives the ssh session.
'''
import argparse
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
NAMES = ["gen", "elapsed_s", "mean", "median", "p10", "p90", "best", "chosen"]
RUNNING_SECONDS = 180          # a CSV written in the last 3 minutes counts as running (same rule as plotRun --running)


class Run:
    '''One CSV, read incrementally: each refresh only parses the lines appended since the last one.'''
    def __init__(self, path):
        self.path, self.offset, self.meta, self.rows = path, 0, {}, []

    def update(self):
        size = os.path.getsize(self.path)
        if size < self.offset:                        # file was rewritten: start over
            self.offset, self.rows = 0, []
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

    def summary(self):
        rows, mtime = self.rows, os.path.getmtime(self.path)
        info = dict(file=os.path.basename(self.path), name=self.meta.get("name", os.path.basename(self.path)),
                    running=time.time() - mtime < RUNNING_SECONDS, mtime=mtime, gens=len(rows),
                    device=self.meta.get("device", "?"), mirrored=self.meta.get("mirrored", "False"),
                    population=self.meta.get("population", "?"), games=self.meta.get("games", "?"),
                    cap=int(self.meta["generations"]) if self.meta.get("generations", "").isdigit() else None)
        if rows:
            last = rows[-200:]                         # recent level: mean of the last 200 generations
            info.update(gen=int(rows[-1][0]), hours=rows[-1][1] / 3600, recentMean=sum(r[2] for r in last) / len(last),
                        bestEver=max(r[6] for r in rows))
            window = [r for r in rows if r[1] > rows[-1][1] - 3600]     # speed over the last hour
            if len(window) > 1 and window[-1][1] > window[0][1]:
                info["gensPerHour"] = (window[-1][0] - window[0][0]) / ((window[-1][1] - window[0][1]) / 3600)
        return info

    def curve(self, buckets):
        '''Bucketed curve (mean of each column per bucket, max for best): ~buckets points however long the run is.'''
        rows = self.rows
        step = max(1, len(rows) // buckets)
        out = {n: [] for n in NAMES}
        for i in range(0, len(rows), step):
            part = rows[i:i + step]
            for k, n in enumerate(NAMES):
                vals = [r[k] for r in part]
                out[n].append(max(vals) if n == "best" else sum(vals) / len(vals))
        out["hours"] = [s / 3600 for s in out.pop("elapsed_s")]
        return out


runs = {}
lock = threading.Lock()          # the server answers requests in threads; one reader of the shared Run state at a time


def scan():
    '''All CSVs in models/ (newest first), each read up to date.'''
    paths = sorted(glob.glob(os.path.join(MODELS, "*.csv")), key=os.path.getmtime, reverse=True)
    for p in paths:
        if p not in runs:
            runs[p] = Run(p)
    return [runs[p] for p in paths]


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
        if url.path == "/":
            self.send(PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/runs":
            listed = scan()
            for r in listed[:40]:                       # only the 40 newest get read (old runs are only read when picked)
                if r.offset == 0 or time.time() - os.path.getmtime(r.path) < RUNNING_SECONDS:
                    r.update()
            self.send(json.dumps([r.summary() for r in listed[:40]]), "application/json")
        elif url.path == "/api/curve":
            buckets = int(q.get("buckets", ["1500"])[0])
            out = {}
            for f in q.get("file", []):
                path = os.path.join(MODELS, os.path.basename(f))
                if os.path.exists(path):
                    run = runs.setdefault(path, Run(path))
                    run.update()
                    out[f] = run.curve(buckets)
            self.send(json.dumps(out), "application/json")
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
:root { --surface:#fcfcfb; --panel:#f3f2ee; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8980; --grid:#e8e7e2; }
@media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --panel:#232321; --ink:#f2f1ec; --ink2:#c3c2b7; --muted:#8a8980; --grid:#2e2d2a; } }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--surface); color:var(--ink); font:14px/1.4 -apple-system, system-ui, sans-serif; }
h1 { font-size:18px; margin:0 0 12px; display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
h1 small { color:var(--muted); font-weight:400; font-size:12px; }
.controls { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:12px; align-items:center; }
.controls label { color:var(--ink2); }
select, input { background:var(--panel); color:var(--ink); border:1px solid var(--grid); border-radius:6px; padding:4px 6px; font:inherit; }
#chart { background:var(--panel); border-radius:10px; padding:8px; margin-bottom:16px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { color:var(--muted); font-weight:500; font-size:12px; }
td.num, th.num { text-align:right; }
.wrap { overflow-x:auto; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.live { color:#199e70; font-weight:600; }
.old { color:var(--muted); }
.u-legend { color:var(--ink2); }
.u-axis { color:var(--ink2); }
</style></head><body>
<h1>Sprudler Live <small id="stamp"></small></h1>
<div class="controls">
  <label>y <select id="metric"><option value="mean">mean</option><option value="median">median</option><option value="p90">p90</option><option value="p10">p10</option><option value="best">best</option></select></label>
  <label>x <select id="xaxis"><option value="gen">generation</option><option value="hours">hours</option></select></label>
  <label>points <select id="buckets"><option>500</option><option selected>1500</option><option>4000</option></select></label>
  <label><input type="checkbox" id="band" checked> p10-p90 band</label>
  <label><input type="checkbox" id="logy"> log y</label>
  <label><input type="checkbox" id="showOld"> show finished runs</label>
</div>
<div id="chart"></div>
<div class="wrap"><table><thead><tr><th></th><th>run</th><th>status</th><th class="num">gen</th><th class="num">cap</th><th class="num">hours</th>
<th class="num">mean (last 200)</th><th class="num">best ever</th><th class="num">gens/h</th><th class="num">ETA</th><th>setup</th></tr></thead>
<tbody id="runs"></tbody></table></div>
<script>
const COLORS = ["#3987e5", "#d95926", "#199e70", "#c548a8", "#d9a400", "#7d5fd6", "#3fa8c9", "#8a8980"];
const $ = id => document.getElementById(id);
let picked = null, plot = null, lastKey = "", colorOf = {};
try { picked = JSON.parse(localStorage.getItem("picked")); } catch (e) {}
for (const id of ["metric", "xaxis", "buckets"]) { try { const v = localStorage.getItem(id); if (v) $(id).value = v; } catch (e) {} }
const fmt = v => v == null ? "" : Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

async function refresh() {
  const runs = await (await fetch("api/runs")).json();
  if (picked === null) picked = runs.filter(r => r.running).map(r => r.file);
  const shown = runs.filter(r => r.running || $("showOld").checked || picked.includes(r.file));
  runs.forEach((r, i) => { if (!(r.file in colorOf)) colorOf[r.file] = COLORS[Object.keys(colorOf).length % COLORS.length]; });
  $("runs").innerHTML = shown.map(r => {
    const left = r.cap && r.gensPerHour ? (r.cap - r.gen) / r.gensPerHour : null;
    return `<tr><td><input type="checkbox" data-file="${r.file}" ${picked.includes(r.file) ? "checked" : ""}></td>
      <td><span class="dot" style="background:${colorOf[r.file]}"></span>${r.name}</td>
      <td class="${r.running ? "live" : "old"}">${r.running ? "running" : "done"}</td>
      <td class="num">${r.gen ?? ""}</td><td class="num">${r.cap ?? ""}</td><td class="num">${r.hours != null ? r.hours.toFixed(1) : ""}</td>
      <td class="num">${fmt(r.recentMean)}</td><td class="num">${fmt(r.bestEver)}</td><td class="num">${r.gensPerHour ? r.gensPerHour.toFixed(0) : ""}</td>
      <td class="num">${r.running && left != null ? left.toFixed(1) + " h" : ""}</td>
      <td class="old">${r.population}x${r.games} ${r.device}${r.mirrored === "True" ? " mirrored" : ""}</td></tr>`; }).join("");
  document.querySelectorAll("#runs input").forEach(cb => cb.onchange = () => {
    picked = cb.checked ? [...picked, cb.dataset.file] : picked.filter(f => f !== cb.dataset.file);
    try { localStorage.setItem("picked", JSON.stringify(picked)); } catch (e) {}
    draw(true);
  });
  await draw(false);
  $("stamp").textContent = "updated " + new Date().toLocaleTimeString();
}

async function draw(force) {
  const files = picked.filter(f => f in colorOf);
  const metric = $("metric").value, xaxis = $("xaxis").value, band = $("band").checked && metric === "mean";
  const qs = files.map(f => "file=" + encodeURIComponent(f)).join("&") + "&buckets=" + $("buckets").value;
  const data = files.length ? await (await fetch("api/curve?" + qs)).json() : {};
  const tables = [], series = [{ label: xaxis === "gen" ? "generation" : "hours", value: (u, v) => v == null ? "" : xaxis === "gen" ? v.toFixed(0) : v.toFixed(2) }];
  const bands = [];
  for (const f of files) {
    const c = data[f]; if (!c || !c.gen.length) continue;
    const name = f.replace(/_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.csv$/, "");
    tables.push([c[xaxis], c[metric]]);
    series.push({ label: name, stroke: colorOf[f], width: 2, spanGaps: true, value: (u, v) => fmt(v) });
    if (band) {
      tables.push([c[xaxis], c.p10]); tables.push([c[xaxis], c.p90]);
      const lo = series.length, hi = lo + 1;
      series.push({ label: name + " p10", stroke: colorOf[f] + "00", spanGaps: true, show: true, value: (u, v) => fmt(v) });
      series.push({ label: name + " p90", stroke: colorOf[f] + "00", spanGaps: true, value: (u, v) => fmt(v) });
      bands.push({ series: [hi, lo], fill: colorOf[f] + "22" });
    }
  }
  const joined = tables.length ? uPlot.join(tables) : [[]];
  const key = [files.join(), metric, xaxis, band, $("logy").checked, $("buckets").value].join("|");
  const width = $("chart").clientWidth - 16, height = Math.max(320, Math.min(620, window.innerHeight - 300));
  if (plot && key === lastKey && !force) {            // same view: only new data, keep the zoom
    plot.setData(joined, false);
    return;
  }
  if (plot) plot.destroy();
  lastKey = key;
  const axis = { stroke: css("--ink2"), grid: { stroke: css("--grid") }, ticks: { stroke: css("--grid") } };
  plot = new uPlot({
    width, height, series, bands,
    scales: { x: { time: false }, y: { distr: $("logy").checked ? 3 : 1 } },
    axes: [{ ...axis, label: xaxis === "gen" ? "generation" : "hours" }, { ...axis, size: 60, values: (u, vals) => vals.map(fmt), label: metric }],
    cursor: { drag: { x: true, y: true, uni: 30 } },
    legend: { live: true },
  }, joined, $("chart"));
}

for (const id of ["metric", "xaxis", "buckets"]) $(id).onchange = () => { try { localStorage.setItem(id, $(id).value); } catch (e) {} draw(true); };
for (const id of ["band", "logy"]) $(id).onchange = () => draw(true);
$("showOld").onchange = refresh;
window.addEventListener("resize", () => draw(true));
refresh(); setInterval(refresh, 10000);
</script></body></html>
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="live learning curves in the browser")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = LAN + tailnet (default), 127.0.0.1 = only via port forward")
    a = ap.parse_args()
    print(f"liveView on http://{a.host}:{a.port}  (Ctrl+C stops)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
