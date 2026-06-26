// IMPaCT IMDP visualizer — self-contained SVG renderer (no external deps).
// POSTs the model to /api/solve, draws the IMDP on a circular layout, and
// heat-maps each state by its robust/optimistic satisfaction probability.
const $ = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";

const EXAMPLES = {
  fork: { format: "imdp", prop: "reach", label: "target", bound: "both",
    model: `# interval fork: nature splits [0.4,0.6] between a reaching and a dead state
states 4
init 0
label target 3
tran 0 0  1:0.4:0.6 2:0.4:0.6
tran 1 0  3:1:1
tran 2 0  2:1:1
tran 3 0  3:1:1` },
  patrol: { format: "prism", prop: "patrol", label: "r0,r2", bound: "pess",
    model: `// patrol: visit r0 and r2 infinitely often on a forced 3-cycle
mdp
module cyc
  x : [0..2] init 0;
  [] x=0 -> 1:(x'=1);
  [] x=1 -> 1:(x'=2);
  [] x=2 -> 1:(x'=0);
endmodule
label "r0" = x=0;
label "r2" = x=2;` },
  chain: { format: "imdp", prop: "reach", label: "target", bound: "both",
    model: `# point chain: P(reach 3) = 0.25
states 4
init 0
label target 3
tran 0 0  1:0.5:0.5 2:0.5:0.5
tran 1 0  3:0.5:0.5 2:0.5:0.5
tran 2 0  2:1:1
tran 3 0  3:1:1` },
};

function loadExample(name) {
  const e = EXAMPLES[name]; if (!e) return;
  $("model").value = e.model; $("format").value = e.format;
  $("prop").value = e.prop; $("label").value = e.label; $("bound").value = e.bound;
}

// value (0..1) -> red->amber->green
function heat(v) {
  v = Math.max(0, Math.min(1, v));
  const stops = [[209,73,91],[237,174,73],[102,161,130]];
  const t = v * 2, i = Math.min(1, Math.floor(t)), f = t - i;
  const c = k => Math.round(stops[i][k] + (stops[i+1][k]-stops[i][k]) * f);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}

function showMsg(text) { const m = $("msg"); if (!text) { m.style.display = "none"; return; } m.textContent = text; m.style.display = "block"; }

function render(data, colorBy) {
  const svg = $("graph"); svg.innerHTML = "";
  const n = data.nStates;
  const W = svg.clientWidth || 800, H = svg.clientHeight || 600;
  const cx = W/2, cy = H/2, R = Math.min(W, H)/2 - 70;
  const pos = [];
  for (let i = 0; i < n; i++) { const a = -Math.PI/2 + 2*Math.PI*i/n; pos.push([cx + R*Math.cos(a), cy + R*Math.sin(a)]); }

  const labelOf = {}; // state -> label names
  for (const [name, sts] of Object.entries(data.labels || {})) for (const s of sts) (labelOf[s] = labelOf[s] || []).push(name);
  const vals = (data.values[colorBy] || data.values.pess || data.values.opt);

  const mk = (tag, attrs) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  // arrow marker
  const defs = mk("defs", {});
  const mr = mk("marker", { id: "arr", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
  mr.appendChild(mk("path", { d: "M0,0 L10,5 L0,10 z", fill: "#6e7681" })); defs.appendChild(mr); svg.appendChild(defs);

  const r = Math.max(10, Math.min(26, 320/n));
  // edges (aggregate per from->to across actions; show widest interval)
  const agg = {};
  for (const e of data.edges) { const k = e.from+"-"+e.to; const a = agg[k] || (agg[k] = {from:e.from,to:e.to,lo:1,hi:0}); a.lo=Math.min(a.lo,e.lo); a.hi=Math.max(a.hi,e.hi); }
  for (const e of Object.values(agg)) {
    if (e.from === e.to) {            // self-loop
      const [x,y] = pos[e.from], ox = (x-cx)/R||0.01, oy = (y-cy)/R||0.01;
      svg.appendChild(mk("circle", { cx: x+ox*r*1.6, cy: y+oy*r*1.6, r: r*0.7, fill:"none", stroke:"#6e7681", "stroke-width":1 }));
      continue;
    }
    const [x1,y1] = pos[e.from], [x2,y2] = pos[e.to];
    const dx=x2-x1, dy=y2-y1, L=Math.hypot(dx,dy)||1, ux=dx/L, uy=dy/L;
    const path = mk("line", { x1:x1+ux*r, y1:y1+uy*r, x2:x2-ux*r, y2:y2-uy*r, stroke:"#6e7681", "stroke-width":1.2, "marker-end":"url(#arr)" });
    const t = mk("title", {}); t.textContent = `${e.from}→${e.to}  [${(+e.lo).toFixed(3)}, ${(+e.hi).toFixed(3)}]`; path.appendChild(t);
    svg.appendChild(path);
  }
  // nodes
  for (let i = 0; i < n; i++) {
    const [x,y] = pos[i], v = vals[i] ? 0.5*(vals[i].lower+vals[i].upper) : 0;
    const g = mk("g", {});
    const isInit = (i === data.init), lab = labelOf[i];
    const circ = mk("circle", { cx:x, cy:y, r:r, fill:heat(v), stroke: isInit ? "#e6edf3" : "#0f1419", "stroke-width": isInit ? 3.5 : 1.5 });
    const tt = mk("title", {});
    const vinfo = Object.keys(data.values).map(k => `${k} [${vals===data.values[k]?"●":" "}${data.values[k][i].lower.toFixed(3)}, ${data.values[k][i].upper.toFixed(3)}]`).join("\n");
    tt.textContent = `state ${i}${lab?" {"+lab.join(",")+"}":""}${isInit?" (init)":""}\n${vinfo}`;
    circ.appendChild(tt); g.appendChild(circ);
    g.appendChild(mk("text", { x:x, y:y+4, "text-anchor":"middle", fill:"#0f1419", "font-size":Math.max(9,r*0.6), "font-weight":"700" })).textContent = i;
    if (lab) { const star = mk("text", { x:x, y:y-r-4, "text-anchor":"middle", fill:"#edae49", "font-size":13 }); star.textContent = "★"; g.appendChild(star); }
    const vt = mk("text", { x:x, y:y+r+14, "text-anchor":"middle", fill:"#8b949e", "font-size":11 }); vt.textContent = v.toFixed(2); g.appendChild(vt);
    svg.appendChild(g);
  }
}

async function solve() {
  const cap = parseInt($("cap").value) || 60;
  const body = { model: $("model").value, format: $("format").value, prop: $("prop").value,
                 label: $("label").value, bound: $("bound").value };
  $("go").disabled = true; $("status").textContent = "solving…"; showMsg("");
  try {
    const res = await fetch("/api/solve", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) { showMsg("Error: " + (data.error||res.status)); $("status").textContent=""; return; }
    $("status").textContent = `${data.nStates} states · ${data.edges.length} edges · property ${data.prop}`;
    if (data.nStates > cap) { showMsg(`Model has ${data.nStates} states (> cap ${cap}). Increase the cap to render, or use a smaller model.`); $("graph").innerHTML=""; return; }
    let colorBy = $("colorby").value;
    if (!data.values[colorBy]) colorBy = Object.keys(data.values)[0];
    render(data, colorBy);
  } catch (e) { showMsg("Request failed: " + e.message); }
  finally { $("go").disabled = false; }
}

$("go").addEventListener("click", solve);
document.querySelectorAll("[data-ex]").forEach(a => a.addEventListener("click", () => loadExample(a.dataset.ex)));
$("colorby").addEventListener("change", solve);
loadExample("fork");
window.addEventListener("load", solve);
