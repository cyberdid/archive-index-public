"""Search results as a picture of neurons (interactive HTML, opens in any browser).

The query is the bright central neuron, each hit a neuron coloured by collection and sized by its score,
a glowing process to the query shows similarity (dashed when found by words only), and a process between
two hits means their vectors are close. The layout starts from a PCA projection of the vectors; the browser
settles it with d3-force. Cosine between hits runs on the GPU when torch sees one.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


D3_URL = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"
COLORS = {"wiki": "#4cc9f0", "docs": "#f72585", "tables": "#b5e48c", "git": "#ffd166", "sessions": "#c77dff",
          "run_logs": "#adb5bd", "images": "#ff9f1c", "audio": "#2ec4b6", "video": "#e76f51", "code": "#90be6d"}


def fetch_vectors(store: Store, ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not ids:
        return out
    marks = ",".join("?" for _ in ids)
    for cid, dim, blob in store.conn.execute(
            f"SELECT id, dimension, vector FROM chunks WHERE id IN ({marks})", ids):
        if dim and dim > 0:
            out[cid] = np.frombuffer(blob, dtype=np.float32).copy()
    return out


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine. torch on the GPU when present; numpy otherwise (identical result)."""
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.from_numpy(vectors).to(device)
        t = torch.nn.functional.normalize(t, dim=1)
        return (t @ t.T).cpu().numpy()
    except Exception:
        norm = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return norm @ norm.T


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    centred = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vt[:2].T
    span = np.maximum(np.abs(coords).max(axis=0), 1e-9)
    return coords / span  # scaled to [-1, 1]


def build_graph(query: str, hits, vectors: dict[int, np.ndarray], query_vec, near: float, per_node: int = 3):
    nodes = [{"id": "q", "kind": "query", "label": query, "x": 0.0, "y": 0.0}]
    links = []
    with_vec = [h for h in hits if h.chunk_id in vectors]
    if with_vec:
        matrix = np.stack([vectors[h.chunk_id] for h in with_vec])
        seed = pca_2d(matrix)
        cos = cosine_matrix(matrix)
    else:
        seed, cos = np.zeros((0, 2)), np.zeros((0, 0))
    seed_of = {h.chunk_id: seed[i] for i, h in enumerate(with_vec)}
    for h in hits:
        x, y = seed_of.get(h.chunk_id, (0.0, 0.0))
        sid = f"h{h.chunk_id}"
        nodes.append({"id": sid, "kind": "hit", "collection": h.collection, "title": h.title,
                      "locator": h.locator, "snippet": h.snippet, "project": h.project or "-",
                      "path": h.path, "via": h.via,
                      "score": float(h.similarity) if h.similarity is not None else None,
                      "x": float(x) * 300, "y": float(y) * 300})
        links.append({"source": "q", "target": sid, "kind": "query",
                      "w": float(h.similarity) if h.similarity is not None else 0.4,
                      "word_only": h.similarity is None})
    index = {h.chunk_id: i for i, h in enumerate(with_vec)}
    seen = set()
    for h in with_vec:
        i = index[h.chunk_id]
        order = np.argsort(-cos[i])
        added = 0
        for j in order:
            if j == i or cos[i, j] < near or added >= per_node:
                continue
            a, b = sorted((h.chunk_id, with_vec[j].chunk_id))
            if (a, b) not in seen:
                seen.add((a, b))
                links.append({"source": f"h{a}", "target": f"h{b}", "kind": "near", "w": float(cos[i, j])})
            added += 1
    return nodes, links


PAGE = """<!doctype html>
<html lang="uk"><head><meta charset="utf-8"><title>archive: __TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#000;color:#e8ecf5;font-family:system-ui,sans-serif;overflow:hidden}
 canvas{display:block;width:100vw;height:100vh}
 #info{position:fixed;left:14px;top:12px;max-width:50vw;font-size:14px;line-height:1.45;pointer-events:none;
       text-shadow:0 0 6px #000,0 0 3px #000}
 #info h1{font-size:18px;margin:0 0 4px}
 #legend{position:fixed;right:14px;top:12px;font-size:13px;background:#000a;padding:8px 10px;border-radius:8px;
         border:1px solid #223}
 #legend div{margin:2px 0}
 #tip{position:fixed;pointer-events:none;background:#05070fee;border:1px solid #334;border-radius:8px;padding:8px 10px;
      max-width:460px;font-size:13px;display:none;line-height:1.35}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="info"><h1>__TITLE__</h1><div id="stats"></div><div style="opacity:.65;margin-top:6px">
наведи на нейрон · перетягуй · колесо — масштаб · пунктир — знайдено лише словами</div></div>
<div id="legend"></div><div id="tip"></div>
<script src="__D3__"></script>
<script>
const DATA = __DATA__;
const COL = __COLORS__;
const canvas = document.getElementById("c"), ctx = canvas.getContext("2d");
const tip = document.getElementById("tip");
let W = 0, H = 0, DPR = Math.max(1, window.devicePixelRatio || 1);
let view = {x: 0, y: 0, k: 1};

function resize() {
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W * DPR; canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener("resize", () => { resize(); draw(); });
resize();

// ---- deterministic randomness: the same fragment always grows the same neuron -----------------
function hash(str) { let h = 2166136261; for (const c of String(str)) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function rng(seed) { let t = seed >>> 0; return () => { t += 0x6D2B79F5; let r = Math.imul(t ^ t >>> 15, 1 | t);
  r ^= r + Math.imul(r ^ r >>> 7, 61 | r); return ((r ^ r >>> 14) >>> 0) / 4294967296; }; }

// dendrite tree in local coordinates: [x1, y1, x2, y2, width, depth]
function dendrites(seed, radius, branches, depthMax) {
  const R = rng(seed), segs = [];
  const grow = (x, y, ang, len, width, depth) => {
    const nx = x + Math.cos(ang) * len, ny = y + Math.sin(ang) * len;
    segs.push([x, y, nx, ny, width, depth]);
    if (depth >= depthMax || len < 1.5) return;
    const kids = depth < 2 ? 2 + Math.floor(R() * 2) : (R() < 0.5 ? 1 : 2);
    for (let k = 0; k < kids; k++) {
      const a = ang + (R() - 0.5) * (depth < 1 ? 1.5 : 1.0);
      grow(nx, ny, a, len * (0.6 + R() * 0.22), Math.max(0.25, width * 0.7), depth + 1);
    }
  };
  for (let b = 0; b < branches; b++) grow(0, 0, (b / branches) * Math.PI * 2 + R() * 0.5, radius * (0.75 + R() * 0.5), 1.8, 0);
  return segs;
}

// ---- nodes: the query is the big central neuron, each hit a neuron sized by its score ---------
const nodes = DATA.nodes.map(d => Object.assign({}, d));
const links = DATA.links.map(d => Object.assign({}, d));
const byId = new Map(nodes.map(n => [n.id, n]));
for (const n of nodes) {
  n.seed = hash(n.id);
  const s = n.score == null ? 0.4 : Math.max(0, Math.min(1, n.score));
  if (n.kind === "query") { n.r = 16; n.trees = dendrites(n.seed, 150, 16, 6); n.color = "#fff2a8"; }
  else {
    n.r = 5 + 5 * s;
    n.trees = dendrites(n.seed, 14 + 60 * s, 4 + Math.round(6 * s), 4 + Math.round(s * 2));
    n.color = COL[n.collection] || "#cccccc";
  }
}
const stroke = (c, a) => c.startsWith("#") ? `rgba(${parseInt(c.slice(1,3),16)},${parseInt(c.slice(3,5),16)},${parseInt(c.slice(5,7),16)},${a})` : c;

function drawTrees(n, x, y) {
  for (const [x1, y1, x2, y2, width, depth] of n.trees) {
    const a = Math.max(0.18, 0.9 - depth * 0.12);
    ctx.strokeStyle = stroke(n.color, a);
    ctx.lineWidth = width * (n.kind === "query" ? 1.1 : 0.8);
    ctx.beginPath(); ctx.moveTo(x + x1, y + y1); ctx.lineTo(x + x2, y + y2); ctx.stroke();
  }
}

function drawSoma(n, x, y) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, n.r * 3.2);
  g.addColorStop(0, stroke(n.color, 0.95)); g.addColorStop(0.35, stroke(n.color, 0.35)); g.addColorStop(1, stroke(n.color, 0));
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, n.r * 3.2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#ffffffee"; ctx.beginPath(); ctx.arc(x, y, n.r * 0.45, 0, Math.PI * 2); ctx.fill();
}

// a synapse between two neurons: a curved glowing process, bent by a fixed seed for each pair
function drawLink(l) {
  const a = byId.get(typeof l.source === "object" ? l.source.id : l.source);
  const b = byId.get(typeof l.target === "object" ? l.target.id : l.target);
  if (!a || !b || a.x == null || b.x == null) return;
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, dx = b.x - a.x, dy = b.y - a.y;
  const bend = ((hash(a.id + b.id) % 100) / 100 - 0.5) * 0.35;
  const cx = mx - dy * bend, cy = my + dx * bend;
  const w = l.kind === "near" ? l.w : (l.w || 0.4);
  if (l.kind === "near") {
    ctx.strokeStyle = stroke(a.color, 0.10 + 0.45 * Math.max(0, (w - 0.75) / 0.25));
    ctx.lineWidth = 3 + 3 * Math.max(0, (w - 0.75) / 0.25);
  } else {
    ctx.strokeStyle = l.word_only ? "rgba(255,242,168,0.35)" : "rgba(255,242,168,0.55)";
    ctx.lineWidth = 0.7 + 1.5 * w;
    if (l.word_only) ctx.setLineDash([4, 6]); else ctx.setLineDash([]);
  }
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(cx, cy, b.x, b.y); ctx.stroke();
  ctx.setLineDash([]);
}

function draw() {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "lighter";
  ctx.save(); ctx.translate(view.x, view.y); ctx.scale(view.k, view.k);
  // glow pass for the processes, then the bright pass
  for (const l of links) { ctx.globalAlpha = 0.6; drawLink(l); }
  for (const n of nodes) if (n.x != null) { ctx.globalAlpha = 1; drawTrees(n, n.x, n.y); }
  for (const n of nodes) if (n.x != null) drawSoma(n, n.x, n.y);
  ctx.restore();
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
}

// ---- layout: d3-force starts from the PCA seed and settles in the browser -----------------------
const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.kind === "near" ? 70 : 170)
     .strength(d => d.kind === "near" ? 0.35 * (d.w || 0.5) : 0.12))
  .force("charge", d3.forceManyBody().strength(-90))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(d => d.r * 3 + 6))
  .alphaDecay(0.02)
  .on("tick", draw);
const q = nodes.find(d => d.kind === "query"); if (q) { q.fx = W / 2; q.fy = H / 2; }

// pan and zoom with the canvas transform
let drag = null;
canvas.addEventListener("wheel", e => { e.preventDefault(); const f = Math.exp(-e.deltaY * 0.001);
  view.x = e.offsetX - (e.offsetX - view.x) * f; view.y = e.offsetY - (e.offsetY - view.y) * f; view.k *= f; draw(); }, {passive: false});
canvas.addEventListener("mousedown", e => { drag = {x: e.clientX, y: e.clientY, vx: view.x, vy: view.y}; });
window.addEventListener("mouseup", () => drag = null);
canvas.addEventListener("mousemove", e => {
  if (drag) { view.x = drag.vx + (e.clientX - drag.x); view.y = drag.vy + (e.clientY - drag.y); draw(); return; }
  const mx = (e.offsetX - view.x) / view.k, my = (e.offsetY - view.y) / view.k;
  let best = null, bd = 1e9;
  for (const n of nodes) { if (n.x == null) continue; const d = Math.hypot(n.x - mx, n.y - my) - n.r * 2.5;
    if (d < bd) { bd = d; best = n; } }
  if (best && bd < 12) {
    tip.style.display = "block"; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px";
    if (best.kind === "query") tip.innerHTML = "<b>запит</b><br>" + esc(best.label);
    else { const s = best.score == null ? "лише словами" : "схожість " + best.score.toFixed(3);
      tip.innerHTML = "<b>" + esc(best.title) + "</b><br>" + esc(best.collection) + " · " + esc(best.project) + " · " + esc(best.locator)
        + "<br>" + s + " · " + esc(best.via) + "<br><span style='opacity:.8'>" + esc((best.snippet || "").slice(0, 220)) + "</span>"; }
  } else tip.style.display = "none";
});

const counts = {}; nodes.filter(d => d.kind === "hit").forEach(d => counts[d.collection] = (counts[d.collection] || 0) + 1);
document.getElementById("legend").innerHTML = Object.entries(counts).map(([k, v]) =>
  "<div><span style='color:" + (COL[k] || "#ccc") + "'>●</span> " + esc(k) + " " + v + "</div>").join("");
document.getElementById("stats").textContent = "нейронів (фрагментів): " + (nodes.length - 1) + " · синапсів до запиту: "
  + links.filter(l => l.kind === "query").length + " · зв'язків між фрагментами: " + links.filter(l => l.kind === "near").length;
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }
</script></body></html>
"""

def render(query: str, nodes, links, out: Path) -> None:
    data = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False).replace("</", "<\\/")
    page = (PAGE.replace("__TITLE__", html.escape(query))
            .replace("__D3__", D3_URL)
            .replace("__COLORS__", json.dumps(COLORS))
            .replace("__DATA__", data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


def clusters(nodes: list[dict], links: list[dict]) -> list[list[str]]:
    """Groups of hits that are connected by 'near' links (close meaning), biggest first. Union-find."""
    parent: dict[str, str] = {n["id"]: n["id"] for n in nodes if n.get("kind") == "hit"}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for link in links:
        if link.get("kind") == "near" and link["source"] in parent and link["target"] in parent:
            parent[find(link["source"])] = find(link["target"])
    groups: dict[str, list[str]] = {}
    for node_id in parent:
        groups.setdefault(find(node_id), []).append(node_id)
    return sorted(groups.values(), key=len, reverse=True)


def text_graph(query: str, nodes: list[dict], links: list[dict], width: int = 96) -> "Text":
    """The search as text: rows are hits by score, bars show similarity, links show close neighbours, and
    groups show which hits are close to each other. Colours follow the collection colours of the picture."""
    from rich.text import Text

    hits = sorted((n for n in nodes if n.get("kind") == "hit"),
                  key=lambda n: -(n["score"] if n["score"] is not None else 0.4))
    number = {n["id"]: i + 1 for i, n in enumerate(hits)}
    near: dict[str, list[tuple[str, float]]] = {}
    for link in links:
        if link.get("kind") == "near":
            near.setdefault(link["source"], []).append((link["target"], link["w"]))
            near.setdefault(link["target"], []).append((link["source"], link["w"]))
    by_id = {n["id"]: n for n in hits}
    out = Text()
    out.append("◉ ", style="bold #ffe66d")
    out.append(f"запит «{query}»", style="bold")
    out.append(f"   {len(hits)} фрагментів · {sum(1 for l in links if l.get('kind') == 'near')} близьких пар\n",
               style="dim")
    groups = clusters(nodes, links)
    out.append(f"\nГрупи близьких за змістом ({len(groups)}):\n", style="bold")
    for g_index, members in enumerate(groups[:12], 1):
        kinds: dict[str, int] = {}
        for m in members:
            kinds[by_id[m]["collection"]] = kinds.get(by_id[m]["collection"], 0) + 1
        mix = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        out.append(f"  група {g_index} · {len(members)} · {mix}\n", style="bold #9fd3ff")
        for m in sorted(members, key=lambda m: -(by_id[m]["score"] or 0.4))[:4]:
            n = by_id[m]
            out.append("     ● ", style=COLORS.get(n["collection"], "#ccc"))
            out.append(f"#{number[m]:<3}", style="dim")
            out.append(f"{n['title'][:50]} · {n['locator'][:22]}\n")
    out.append("\nФрагменти (за схожістю з запитом):\n", style="bold")
    for n in hits:
        score = n["score"]
        bar_len = 0 if score is None else max(1, round(score * 20))
        word_only = score is None
        out.append("  ", style="")
        out.append(f"#{number[n['id']]:<3}", style="dim")
        out.append("● ", style=COLORS.get(n["collection"], "#ccc"))
        out.append(f"{n['collection']:<9}", style=COLORS.get(n["collection"], "#ccc"))
        out.append("█" * bar_len + "░" * (20 - bar_len), style="#ffe66d" if not word_only else "dim")
        label = "слова" if word_only else f"{score:.3f}"
        out.append(f" {label:>5}  ", style="dim")
        out.append(f"{n['title'][:width - 52]}", style="bold" if not word_only else "")
        out.append(f" · {n['locator'][:20]}", style="dim")
        peers = sorted(near.get(n["id"], []), key=lambda p: -p[1])[:3]
        if peers:
            out.append("  ↔ " + " ".join(f"#{number[p]}({w:.2f})" for p, w in peers), style="#9fd3ff")
        out.append("\n")
    return out
