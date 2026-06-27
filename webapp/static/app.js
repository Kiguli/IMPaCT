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

const PTA_PROB=`# PTA: at clock x>=2 a probabilistic edge -> 0.7 target L1 / 0.3 dead L2
clocks 1
init 0
kmax 1 3
target 1
edge 0 | 1>=2 | 0.7 1 ; 0.3 2
edge 2 | 1>=0 | 1.0 2`;
const PTA_SEQ=`# PTA: L0 --(x>=1,reset)--> L1 --(x>=1)--> {0.5: target L2 / 0.5: L3}
clocks 1
init 0
kmax 1 2
target 2
edge 0 | 1>=1 | 1.0 1 r1
edge 1 | 1>=1 | 0.5 2 ; 0.5 3
edge 3 | 1>=0 | 1.0 3`;
const PTA_INV=`# PTA: invariant x<=1 blocks the x>=2 guard -> target unreachable
clocks 1
init 0
kmax 1 2
target 1
inv 0 1<=1
edge 0 | 1>=2 | 1.0 1`;
const PTA_EX = [
  { name:"probabilistic edge",   model:PTA_PROB },
  { name:"sequential + reset",   model:PTA_SEQ },
  { name:"invariant blocks goal",model:PTA_INV },
];

const POM_LINEAR=`# per-step 0.5 to target, no information; horizon 3 -> 0.875
states 2
actions 1
obs 1
init 0:1
target 1
horizon 3
T 0 0 : 0:0.5 1:0.5
T 0 1 : 1:1`;
const POM_BRANCH=`# 2 observations -> branching belief tree. obs reveals state 1.
states 3
actions 2
obs 2
init 0:1
target 2
horizon 3
T 0 0 : 0:0.5 1:0.5
T 1 0 : 0:0.7 2:0.3
T 0 1 : 2:1
T 1 1 : 2:1
T 0 2 : 2:1
T 1 2 : 2:1
O 0 1 : 1:1
O 1 1 : 1:1`;
const POM_EX = [
  { name:"no info (linear)",  model:POM_LINEAR, horizon:3 },
  { name:"branching (2 obs)", model:POM_BRANCH, horizon:3 },
];

// ===================== node-link graph (IMDP) ===============================
// `frame` (optional): per-state values from a value-iteration step (animation).
function renderGraph(data, colorBy, frame) {
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
    const [x,y]=pos[i], v=frame?(frame[i]||0):(vals[i]?0.5*(vals[i].lower+vals[i].upper):0), g=mk("g",{}), isInit=(i===data.init), lab=labelOf[i];
    const c=mk("circle",{cx:x,cy:y,r:r,fill:heat(v),stroke:isInit?"#e6edf3":"#0f1419","stroke-width":isInit?3.5:1.5});
    const tt=mk("title",{}); tt.textContent=`state ${i}${lab?" {"+lab.join(",")+"}":""}${isInit?" (init)":""}\n`+(frame?`iteration value = ${v.toFixed(3)}`:Object.keys(data.values).map(k=>`${k} [${data.values[k][i].lower.toFixed(3)}, ${data.values[k][i].upper.toFixed(3)}]`).join("\n"));
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

// ===================== zone graph (timed automata / PTA) ====================
function renderZoneGraph(data) {
  const svg=$("graph"); svg.innerHTML="";
  const nodes=data.nodes, n=nodes.length, W=svg.clientWidth||800, H=svg.clientHeight||600, cx=W/2, cy=H/2, R=Math.min(W,H)/2-80;
  const pos={}; nodes.forEach((nd,i)=>{ const a=-Math.PI/2+2*Math.PI*i/Math.max(1,n); pos[nd.id]=[cx+R*Math.cos(a), cy+R*Math.sin(a)]; });
  const defs=mk("defs",{}); const m=mk("marker",{id:"arrz",viewBox:"0 0 10 10",refX:"9",refY:"5",markerWidth:"7",markerHeight:"7",orient:"auto-start-reverse"});
  m.appendChild(mk("path",{d:"M0,0 L10,5 L0,10 z",fill:"#6e7681"})); defs.appendChild(m); svg.appendChild(defs);
  const r=Math.max(13,Math.min(30,360/Math.max(1,n)));
  for(const e of data.edges){
    if(!(e.from in pos)||!(e.to in pos))continue;
    const [x1,y1]=pos[e.from],[x2,y2]=pos[e.to],dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L;
    const ln=mk("line",{x1:x1+ux*r,y1:y1+uy*r,x2:x2-ux*r,y2:y2-uy*r,stroke:"#6e7681","stroke-width":1.2,"marker-end":"url(#arrz)"});
    const t=mk("title",{}); t.textContent=`p=${e.prob}`; ln.appendChild(t); svg.appendChild(ln);
    const mx=(x1+x2)/2,my=(y1+y2)/2; svg.appendChild(mk("text",{x:mx,y:my-2,"text-anchor":"middle",fill:"#8b949e","font-size":10})).textContent=(+e.prob).toFixed(2);
  }
  nodes.forEach(nd=>{
    const [x,y]=pos[nd.id], g=mk("g",{}), isInit=(nd.id===data.init);
    const c=mk("circle",{cx:x,cy:y,r:r,fill:heat(nd.value),stroke:isInit?"#e6edf3":"#0f1419","stroke-width":isInit?3.5:1.5});
    const tt=mk("title",{}); tt.textContent=`${nd.descr}\nreach value = ${nd.value.toFixed(3)}${nd.target?" (target)":""}${isInit?" (init)":""}`; c.appendChild(tt); g.appendChild(c);
    g.appendChild(mk("text",{x:x,y:y+4,"text-anchor":"middle",fill:"#0f1419","font-size":Math.max(9,r*0.5),"font-weight":"700"})).textContent="L"+nd.loc;
    if(nd.target){const s=mk("text",{x:x,y:y-r-4,"text-anchor":"middle",fill:"#edae49","font-size":13});s.textContent="★";g.appendChild(s);}
    g.appendChild(mk("text",{x:x,y:y+r+14,"text-anchor":"middle",fill:"#8b949e","font-size":11})).textContent=nd.value.toFixed(2);
    svg.appendChild(g);
  });
  $("status").textContent=`${n} symbolic states · target L${data.target}`;
}

// ===================== belief tree (POMDP) ==================================
const BELCOLORS = ["#4493f8","#edae49","#66a182","#d1495b","#a78bfa","#8b949e"];
function renderBeliefTree(data) {
  const svg=$("graph"); svg.innerHTML="";
  const nodes=data.nodes; if(!nodes.length){ $("status").textContent="empty"; return; }
  const byId={}; nodes.forEach(n=>byId[n.id]=n);
  const ch={}; nodes.forEach(n=>{ if(n.parent>=0)(ch[n.parent]=ch[n.parent]||[]).push(n.id); });
  const root=nodes.find(n=>n.parent<0)||nodes[0];
  const X={}; let leaf=0;
  (function assign(id){ const c=ch[id]||[]; if(!c.length){X[id]=leaf++;return X[id];} const xs=c.map(assign); X[id]=(xs[0]+xs[xs.length-1])/2; return X[id]; })(root.id);
  const maxLeaf=Math.max(1,leaf-1), H0=data.horizon, W=svg.clientWidth||800, Hh=svg.clientHeight||600, m=60;
  const PX=id=>m+(maxLeaf?X[id]/maxLeaf:0.5)*(W-2*m);
  const PY=d=>m+((H0-d)/Math.max(1,H0))*(Hh-2*m-20);
  // edges
  for(const n of nodes){ if(n.parent<0)continue; const x1=PX(n.parent),y1=PY(byId[n.parent].depth),x2=PX(n.id),y2=PY(n.depth);
    svg.appendChild(mk("line",{x1,y1,x2,y2,stroke:"#6e7681","stroke-width":1.2}));
    svg.appendChild(mk("text",{x:(x1+x2)/2+4,y:(y1+y2)/2,fill:"#8b949e","font-size":10})).textContent=`o${n.obs}:${n.prob.toFixed(2)}`;
  }
  // nodes: belief mini-bar + value
  const bw=46, bh=12;
  for(const n of nodes){
    const x=PX(n.id), y=PY(n.depth), g=mk("g",{});
    // value disc
    const c=mk("circle",{cx:x,cy:y,r:7,fill:heat(n.value),stroke:(n.parent<0)?"#e6edf3":"#0f1419","stroke-width":(n.parent<0)?2.5:1});
    const tt=mk("title",{}); tt.textContent=`belief [${n.belief.map(v=>v.toFixed(2)).join(", ")}]\nvalue ${n.value.toFixed(3)}`+(n.action>=0?`\nbest action ${n.action}`:"");
    c.appendChild(tt); g.appendChild(c);
    // belief stacked bar
    let bx=x-bw/2; const by=y+10;
    n.belief.forEach((pv,si)=>{ const w=pv*bw; if(w>0.5){ g.appendChild(mk("rect",{x:bx,y:by,width:w,height:bh,fill:BELCOLORS[si%BELCOLORS.length]})); bx+=w; } });
    g.appendChild(mk("rect",{x:x-bw/2,y:by,width:bw,height:bh,fill:"none",stroke:"#30363d"}));
    g.appendChild(mk("text",{x:x,y:by+bh+11,"text-anchor":"middle",fill:"#8b949e","font-size":10})).textContent=n.value.toFixed(2);
    svg.appendChild(g);
  }
  // legend for states
  let lx=14, ly=Hh-26;
  for(let s=0;s<data.nStates && s<BELCOLORS.length;s++){ svg.appendChild(mk("rect",{x:lx,y:ly,width:10,height:10,fill:BELCOLORS[s]})); svg.appendChild(mk("text",{x:lx+14,y:ly+9,fill:"#8b949e","font-size":11})).textContent="s"+s; lx+=40; }
  $("status").textContent=`${nodes.length} belief nodes · horizon ${data.horizon} · root value ${root.value.toFixed(3)}`;
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
  zone:   { label:"Zone graph", modelLabel:"Timed automaton (.pta)", endpoint:"/api/zonegraph", controls:["bound"], examples:PTA_EX,
            body:()=>({model:$("model").value, bound:($("bound").value==="opt"?"opt":"pess"), engine:"zone"}),
            render:renderZoneGraph },
  belief: { label:"Belief tree", modelLabel:"POMDP (.pomdp)", endpoint:"/api/belieftree", controls:["horizon"], examples:POM_EX,
            body:()=>({model:$("model").value, horizon:$("horizon").value}),
            render:renderBeliefTree },
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
  $("animate").style.display = (mode === "imdp") ? "" : "none";
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
  if(mode==="belief" && e.horizon){ $("horizon").value=e.horizon; }
}

let LAST=null;
async function run() {
  const M=MODES[mode]; if(!M.endpoint) return;
  stopAnim(); $("anim").style.display="none";
  $("go").disabled=true; $("status").textContent="solving…"; showMsg("");
  try {
    const res=await fetch(M.endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(M.body())});
    const data=await res.json();
    if(!res.ok){ showMsg("Error: "+(data.error||res.status)); $("status").textContent=""; clearView(); return; }
    LAST=data; M.render(data);
  } catch(e){ showMsg("Request failed: "+e.message); }
  finally{ $("go").disabled=false; }
}

// ---- value-iteration animation (IMDP graph, reach/safety) -------------------
const ANIM={ data:null, frames:null, cur:0, timer:null };
function stopAnim(){ if(ANIM.timer){clearInterval(ANIM.timer); ANIM.timer=null;} const b=$("animPlay"); if(b) b.textContent="▶"; }
function setFrame(k){ if(!ANIM.frames) return; k=Math.max(0,Math.min(ANIM.frames.length-1,k)); ANIM.cur=k;
  renderGraph(ANIM.data,null,ANIM.frames[k]); $("animSlider").value=k;
  $("animLabel").textContent=`iteration ${k} / ${ANIM.frames.length-1}`; }
function playAnim(){ if(!ANIM.frames) return; if(ANIM.timer){ stopAnim(); return; }
  if(ANIM.cur>=ANIM.frames.length-1) setFrame(0);
  $("animPlay").textContent="⏸";
  ANIM.timer=setInterval(()=>{ if(ANIM.cur>=ANIM.frames.length-1){ stopAnim(); return; } setFrame(ANIM.cur+1); }, 650); }
async function animateVI(){
  if(mode!=="imdp"){ showMsg("Value-iteration animation is for the IMDP graph (reach / safety)."); return; }
  const prop=$("prop").value;
  if(prop!=="reach"&&prop!=="safety"){ showMsg("VI animation is available for reach and safety properties."); return; }
  stopAnim(); $("animate").disabled=true; $("status").textContent="computing trace…"; showMsg("");
  try{
    const body=Object.assign(MODES.imdp.body(),{trace:true});
    const res=await fetch("/api/solve",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const data=await res.json();
    if(!res.ok){ showMsg("Error: "+(data.error||res.status)); return; }
    if(data.nStates>(parseInt($("cap").value)||60)){ showMsg("Too many states to animate; increase cap or use a smaller model."); return; }
    if(!data.trace){ showMsg("No trace available for this property."); return; }
    ANIM.data=data; ANIM.frames=data.trace.frames; ANIM.cur=0;
    $("anim").style.display="flex"; $("animSlider").max=ANIM.frames.length-1; $("animTitle").textContent=data.trace.prop;
    $("status").textContent=`${data.trace.frames.length-1} iterations · ${data.trace.prop}`;
    setFrame(0);
  }catch(e){ showMsg("Request failed: "+e.message); }
  finally{ $("animate").disabled=false; }
}

// ---- upload / download ------------------------------------------------------
function downloadText(name,text,type){ const b=new Blob([text],{type:type||"text/plain"}); const u=URL.createObjectURL(b);
  const a=document.createElement("a"); a.href=u; a.download=name; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(u); }
const EXT={imdp:"imdp",grid:"sys",zone:"pta",belief:"pomdp"};

$("go").addEventListener("click", run);
$("animate").addEventListener("click", animateVI);
$("colorby").addEventListener("change", ()=>{ if(mode==="imdp" && $("anim").style.display==="none") run(); });
$("animPlay").addEventListener("click", playAnim);
$("animStepF").addEventListener("click", ()=>{ stopAnim(); setFrame(ANIM.cur+1); });
$("animStepB").addEventListener("click", ()=>{ stopAnim(); setFrame(ANIM.cur-1); });
$("animSlider").addEventListener("input", e=>{ stopAnim(); setFrame(parseInt(e.target.value)); });
$("animClose").addEventListener("click", ()=>{ stopAnim(); $("anim").style.display="none"; if(LAST&&mode==="imdp"){let cb=$("colorby").value; if(!LAST.values||!LAST.values[cb])cb=LAST.values?Object.keys(LAST.values)[0]:"pess"; renderGraph(LAST,cb);} });
$("upload").addEventListener("change", e=>{ const f=e.target.files[0]; if(!f)return; const rd=new FileReader(); rd.onload=()=>{ $("model").value=rd.result; run(); }; rd.readAsText(f); e.target.value=""; });
$("download").addEventListener("click", ()=>downloadText("model."+(EXT[mode]||"txt"), $("model").value));
$("downloadSvg").addEventListener("click", ()=>{ const s=$("graph"); downloadText("impact-figure.svg", '<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(s), "image/svg+xml"); });
$("downloadJson").addEventListener("click", ()=>{ if(!LAST){showMsg("Run something first.");return;} downloadText("impact-result.json", JSON.stringify(LAST,null,2), "application/json"); });

renderModes(); renderExamples(); showControls(); loadExample(0);
window.addEventListener("load", run);
