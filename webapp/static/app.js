// IMPaCT visualizers — mode-based front end (self-contained SVG; no external deps).
const $ = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const mk = (tag, a) => { const e = document.createElementNS(SVGNS, tag); for (const k in a) e.setAttribute(k, a[k]); return e; };
function heat(v) {
  v = Math.max(0, Math.min(1, v));
  const st = [[209,73,91],[237,174,73],[102,161,130]], t = v*2, i = Math.min(1, Math.floor(t)), f = t-i;
  const c = k => Math.round(st[i][k] + (st[i+1][k]-st[i][k])*f);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}
function showMsg(t){ const m=$("msg"); if(!t){m.style.display="none";return;} m.textContent=t; m.style.display="block"; }
function clearView(){ $("graph").innerHTML=""; }

// ===================== models reused across examples =========================
const M_FORK=`# interval fork: nature splits [0.4,0.6] between reaching (1->3) and dead (2)
states 4
init 0
label target 3
tran 0 0  1:0.4:0.6 2:0.4:0.6
tran 1 0  3:1:1
tran 2 0  2:1:1
tran 3 0  3:1:1`;
const M_CHAIN=`# point chain: P(reach 3)=0.25
states 4
init 0
label target 3
tran 0 0  1:0.5:0.5 2:0.5:0.5
tran 1 0  3:0.5:0.5 2:0.5:0.5
tran 2 0  2:1:1
tran 3 0  3:1:1`;
const M_SAFE=`# safety: avoid state 2 (Pmax never-reach from 0 = 0.72)
states 4
init 0
label avoid 2
tran 0 0  1:0.7:0.7 3:0.3:0.3
tran 1 0  2:0.4:0.4 3:0.6:0.6
tran 2 0  2:1:1
tran 3 0  3:1:1`;
const M_CYC2=`# forced 2-cycle 0<->1; acc/r at state 1
states 2
init 0
label acc 1
label r 1
tran 0 0  1:1:1
tran 1 0  0:1:1`;
const M_PERSIST=`# F G safe: reach safe sink {1}; nature can leak via 0
states 3
init 0
label safe 0 1
tran 0 0  1:0.5:1.0 2:0.0:0.5
tran 1 0  1:1:1
tran 2 0  2:1:1`;
const M_PATROL=`// patrol: visit r0 and r2 infinitely often on a 3-cycle
mdp
module cyc
  x : [0..2] init 0;
  [] x=0 -> 1:(x'=1);
  [] x=1 -> 1:(x'=2);
  [] x=2 -> 1:(x'=0);
endmodule
label "r0" = x=0;
label "r2" = x=2;`;
const M_UNTIL=`# a U b : reach b={2} staying in a={0,1}
states 3
init 0
label a 0 1
label b 2
tran 0 0  1:1:1
tran 1 0  2:1:1
tran 2 0  2:1:1`;
const M_PRISM=`// PRISM-language input
mdp
const int N = 2;
module choice
  x : [0..N] init 0;
  [] x=0 -> [0.5,0.7]:(x'=1) + [0.3,0.5]:(x'=2);
  [] x=0 -> 0.6:(x'=1) + 0.4:(x'=2);
  [] x=1 -> 1:(x'=1);
  [] x=2 -> 1:(x'=2);
endmodule
label "target" = x=2;`;

const SYS_REACH=`# 2D robot: x' = 0.9 x + 0.5 u + N(0,0.3); reach the box [1.5,3]^2
xlb -3 -3
xub 3 3
eta 0.5 0.5
ulb -1 -1
uub 1 1
ueta 1 1
A 0.9 0 0 0.9
B 0.5 0 0 0.5
sigma 0.3 0.3
region 1.5 3 1.5 3
prop reach
prune 1e-5`;
const SYS_SAFE=`# safety: avoid a central obstacle; cells near it have lower safety
xlb -3 -3
xub 3 3
eta 0.4 0.4
ulb -1 -1
uub 1 1
ueta 1 1
A 1 0 0 1
B 0.5 0 0 0.5
sigma 0.4 0.4
region -0.5 0.5 -0.5 0.5
prop safety
prune 1e-5`;

// ===================== IMDP examples ========================================
const IMDP_EX = [
  { name:"reach F (interval)", format:"imdp", prop:"reach", label:"target", bound:"both", model:M_FORK },
  { name:"reach F (=0.25)",   format:"imdp", prop:"reach", label:"target", bound:"both", model:M_CHAIN },
  { name:"safety G (=0.72)",  format:"imdp", prop:"safety", label:"avoid", bound:"both", model:M_SAFE },
  { name:"until a U b",       format:"imdp", prop:"until", label:"a,b", bound:"pess", model:M_UNTIL },
  { name:"next X b",          format:"imdp", prop:"next", label:"b", bound:"pess", model:M_UNTIL },
  { name:"recurrence G F r",  format:"imdp", prop:"buchi", label:"acc", bound:"both", model:M_CYC2 },
  { name:"persistence F G",   format:"imdp", prop:"persist", label:"safe", bound:"both", model:M_PERSIST },
  { name:"patrol (PRISM)",    format:"prism", prop:"patrol", label:"r0,r2", bound:"pess", model:M_PATROL },
  { name:"LTL: F target",     format:"imdp", prop:"ltl", label:"F target", bound:"both", model:M_FORK },
  { name:"LTL: G F r",        format:"imdp", prop:"ltl", label:"G F r", bound:"both", model:M_CYC2 },
  { name:"PRISM input",       format:"prism", prop:"reach", label:"target", bound:"both", model:M_PRISM },
];
const GRID_EX = [
  { name:"reach a target box",      model:SYS_REACH },
  { name:"safety around obstacle",  model:SYS_SAFE },
];

// ===================== node-link graph (IMDP) ===============================
function renderGraph(data, colorBy) {
  const svg=$("graph"); svg.innerHTML="";
  const n=data.nStates, W=svg.clientWidth||800, H=svg.clientHeight||600, cx=W/2, cy=H/2, R=Math.min(W,H)/2-70;
  const pos=[]; for(let i=0;i<n;i++){const a=-Math.PI/2+2*Math.PI*i/n; pos.push([cx+R*Math.cos(a), cy+R*Math.sin(a)]);}
  const labelOf={}; for(const [nm,sts] of Object.entries(data.labels||{})) for(const s of sts)(labelOf[s]=labelOf[s]||[]).push(nm);
  const vals=data.values[colorBy]||data.values.pess||data.values.opt;
  const defs=mk("defs",{}); const mr=mk("marker",{id:"arr",viewBox:"0 0 10 10",refX:"9",refY:"5",markerWidth:"7",markerHeight:"7",orient:"auto-start-reverse"});
  mr.appendChild(mk("path",{d:"M0,0 L10,5 L0,10 z",fill:"#6e7681"})); defs.appendChild(mr); svg.appendChild(defs);
  const r=Math.max(10,Math.min(26,320/n));
  const agg={}; for(const e of data.edges){const k=e.from+"-"+e.to; const a=agg[k]||(agg[k]={from:e.from,to:e.to,lo:1,hi:0}); a.lo=Math.min(a.lo,e.lo);a.hi=Math.max(a.hi,e.hi);}
  for(const e of Object.values(agg)){
    if(e.from===e.to){const [x,y]=pos[e.from],ox=(x-cx)/R||0.01,oy=(y-cy)/R||0.01; svg.appendChild(mk("circle",{cx:x+ox*r*1.6,cy:y+oy*r*1.6,r:r*0.7,fill:"none",stroke:"#6e7681","stroke-width":1})); continue;}
    const [x1,y1]=pos[e.from],[x2,y2]=pos[e.to]; const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L;
    const ln=mk("line",{x1:x1+ux*r,y1:y1+uy*r,x2:x2-ux*r,y2:y2-uy*r,stroke:"#6e7681","stroke-width":1.2,"marker-end":"url(#arr)"});
    const t=mk("title",{});t.textContent=`${e.from}→${e.to} [${(+e.lo).toFixed(3)}, ${(+e.hi).toFixed(3)}]`;ln.appendChild(t);svg.appendChild(ln);
  }
  for(let i=0;i<n;i++){
    const [x,y]=pos[i], v=vals[i]?0.5*(vals[i].lower+vals[i].upper):0, g=mk("g",{}), isInit=(i===data.init), lab=labelOf[i];
    const c=mk("circle",{cx:x,cy:y,r:r,fill:heat(v),stroke:isInit?"#e6edf3":"#0f1419","stroke-width":isInit?3.5:1.5});
    const tt=mk("title",{}); tt.textContent=`state ${i}${lab?" {"+lab.join(",")+"}":""}${isInit?" (init)":""}\n`+Object.keys(data.values).map(k=>`${k} [${data.values[k][i].lower.toFixed(3)}, ${data.values[k][i].upper.toFixed(3)}]`).join("\n");
    c.appendChild(tt); g.appendChild(c);
    g.appendChild(mk("text",{x:x,y:y+4,"text-anchor":"middle",fill:"#0f1419","font-size":Math.max(9,r*0.6),"font-weight":"700"})).textContent=i;
    if(lab){const s=mk("text",{x:x,y:y-r-4,"text-anchor":"middle",fill:"#edae49","font-size":13});s.textContent="★";g.appendChild(s);}
    g.appendChild(mk("text",{x:x,y:y+r+14,"text-anchor":"middle",fill:"#8b949e","font-size":11})).textContent=v.toFixed(2);
    svg.appendChild(g);
  }
}

// ===================== grid heatmap (abstraction) ===========================
function renderHeatmap(data) {
  const svg=$("graph"); svg.innerHTML="";
  const W=svg.clientWidth||800, H=svg.clientHeight||600;
  const nx=data.nx, ny=data.ny;
  const panels=[["robust (min)",data.min],["optimistic (max)",data.max]];
  const pad=40, gap=50, pw=(W-2*pad-gap)/2, ph=Math.min(H-2*pad-20, pw);
  const cw=pw/nx, ch=ph/ny, top=(H-ph)/2;
  panels.forEach((pn,pi)=>{
    const [title,grid]=pn, x0=pad+pi*(pw+gap);
    svg.appendChild(mk("text",{x:x0+pw/2,y:top-14,"text-anchor":"middle",fill:"#e6edf3","font-size":14,"font-weight":"600"})).textContent=title;
    for(let j1=0;j1<ny;j1++) for(let j0=0;j0<nx;j0++){
      const v=grid[j1][j0], px=x0+j0*cw, py=top+(ny-1-j1)*ch;   // flip y so row 0 is bottom
      const rc=mk("rect",{x:px,y:py,width:Math.ceil(cw)+0.5,height:Math.ceil(ch)+0.5,fill:heat(v)});
      const xc=(data.xlb+(j0+0.5)*data.etax).toFixed(2), yc=(data.ylb+(j1+0.5)*data.etay).toFixed(2);
      const tt=mk("title",{}); tt.textContent=`cell (${xc}, ${yc}) = ${v.toFixed(3)}`; rc.appendChild(tt);
      svg.appendChild(rc);
    }
    svg.appendChild(mk("rect",{x:x0,y:top,width:pw,height:ph,fill:"none",stroke:"#30363d"}));
    svg.appendChild(mk("text",{x:x0,y:top+ph+16,fill:"#8b949e","font-size":11})).textContent=`x∈[${data.xlb}, ${(data.xlb+nx*data.etax).toFixed(1)}]`;
    svg.appendChild(mk("text",{x:x0,y:top-14,fill:"#8b949e","font-size":11,"text-anchor":"start","transform":`rotate(-90 ${x0-14} ${top+ph/2})`})).textContent="";
  });
  $("status").textContent=`${nx}×${ny} cells · ${data.prop} · nnz=${data.nnz}`;
}

const ABOUT=`<p>The visualizers cover IMPaCT's verified stochastic-synthesis stack:</p>
<ul>
<li><b>IMDP graph</b> — reach/safety/until/next, ω-regular (GF/FG/patrol) and LTL formulas
  on Interval-MDPs (.imdp / PRISM), heat-mapped by robust/optimistic probability.</li>
<li><b>Grid heatmap</b> — per-cell min (robust) &amp; max (optimistic) satisfaction
  probability for a 2-D continuous system via the sparse abstraction.</li>
<li><b>Zone graph</b> — symbolic states of a (probabilistic) timed automaton.</li>
<li><b>Belief tree</b> — the POMDP optimal-policy belief tree.</li>
</ul>
<p>Each engine is differential/brute-force-oracle tested with verified literature
references. Backend shells to the C++ CLIs; the C++ core is untouched.</p>`;

// ===================== modes ================================================
const MODES = {
  imdp:   { label:"IMDP graph",   modelLabel:"Model (.imdp / PRISM)",   endpoint:"/api/solve",      controls:["format","label","bound","cap"], examples:IMDP_EX,
            body:()=>({model:$("model").value, format:$("format").value, prop:$("prop").value, label:$("label").value, bound:$("bound").value}),
            render:(d)=>{ if(d.nStates>(parseInt($("cap").value)||60)){showMsg(`${d.nStates} states > cap — increase "Max states".`);clearView();return;}
                          let cb=$("colorby").value; if(!d.values[cb]) cb=Object.keys(d.values)[0]; renderGraph(d,cb);
                          $("status").textContent=`${d.nStates} states · ${d.edges.length} edges · ${d.prop}`; } },
  grid:   { label:"Grid heatmap", modelLabel:"System (.sys, 2-D affine)", endpoint:"/api/grid", controls:["eps"], examples:GRID_EX,
            body:()=>({model:$("model").value, eps:$("eps").value}),
            render:renderHeatmap },
  about:  { label:"About", modelLabel:"", endpoint:null, controls:[], examples:[], render:()=>{} },
};
let mode = "imdp";

function showControls() {
  const want = new Set(MODES[mode].controls);
  $("c_format").style.display = want.has("format") ? "" : "none";
  $("c_label").style.display  = want.has("label")  ? "" : "none";
  $("c_bound").style.display  = want.has("bound")  ? "" : "none";
  $("c_cap").style.display    = want.has("cap")    ? "" : "none";
  $("c_eps").style.display    = want.has("eps")    ? "" : "none";
  $("c_horizon").style.display= want.has("horizon")? "" : "none";
  const isAbout = (mode === "about");
  $("modePanel").style.display = isAbout ? "none" : "";
  $("aboutPanel").style.display = isAbout ? "" : "none";
  $("legend").style.display = isAbout ? "none" : "";
  if (isAbout) { $("aboutPanel").innerHTML = ABOUT; clearView(); }
  $("modelLabel").textContent = MODES[mode].modelLabel;
}
function renderModes() {
  const bar=$("modes"); bar.innerHTML="";
  for (const k of Object.keys(MODES)) {
    const b=document.createElement("button"); b.textContent=MODES[k].label; b.className=(k===mode)?"active":"";
    b.addEventListener("click",()=>{ mode=k; renderModes(); renderExamples(); showControls(); if(MODES[mode].examples.length) loadExample(0); });
    bar.appendChild(b);
  }
}
function renderExamples() {
  const box=$("examples"); box.innerHTML="";
  MODES[mode].examples.forEach((e,i)=>{ const b=document.createElement("button"); b.textContent=e.name;
    b.addEventListener("click",()=>{ loadExample(i); run(); }); box.appendChild(b); });
}
function loadExample(i) {
  const e=MODES[mode].examples[i]; if(!e) return;
  $("model").value=e.model;
  if(mode==="imdp"){ $("format").value=e.format; $("prop").value=e.prop; $("label").value=e.label; $("bound").value=e.bound; }
}

async function run() {
  const M=MODES[mode]; if(!M.endpoint) return;
  $("go").disabled=true; $("status").textContent="solving…"; showMsg("");
  try {
    const res=await fetch(M.endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(M.body())});
    const data=await res.json();
    if(!res.ok){ showMsg("Error: "+(data.error||res.status)); $("status").textContent=""; clearView(); return; }
    M.render(data);
  } catch(e){ showMsg("Request failed: "+e.message); }
  finally{ $("go").disabled=false; }
}

$("go").addEventListener("click", run);
$("colorby").addEventListener("change", ()=>{ if(mode==="imdp") run(); });
renderModes(); renderExamples(); showControls(); loadExample(0);
window.addEventListener("load", run);
