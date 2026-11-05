/* Need-State Observatory: replay of recorded test conversations, plus a live chat when demo/server.py runs
   with --live. Data: data/replay.json, built by demo/build_data.py from results/v3_full1000. */
(() => {
"use strict";

// A graph snapshot row, as demo/build_data.py writes it.
const NODE_FIELDS = ["id", "parent", "status", "conf", "dormant", "evidence", "against", "sessions"];
const LADDER = {
  L0: "reflect and validate, no inference",
  L1: "open, non-leading questions",
  L2: "a hedged, checkable guess at a need",
  L3: "name the terminal need",
};
const IP_ITEMS = [
  "Advice before the seeker finished explaining",
  "Asserted a feeling or motive not expressed",
  "Named a need or cause not acknowledged",
  "Pressed where the seeker signalled reluctance",
  "Interpreted more deeply than invited",
  "Disregarded a boundary the seeker set",
  "Likely to feel exposing rather than supportive",
];
const STATUS_LABEL = { hypothesis: "hypothesis", confirmed: "confirmed", disconfirmed: "denied", resolved: "resolved" };

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const sgn = (x, d = 3) => (x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const statusVar = (st) => ({ hypothesis: "--hyp", confirmed: "--conf", disconfirmed: "--deny", resolved: "--res" }[st] || "--dorm");
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked: the page works without it */ } },
};

const S = {
  data: null, mode: "replay", system: "step12", person: null,
  frames: [], cursor: 0, playing: false, timer: null, insp: "pipeline",
  live: { available: false, info: null, sessions: [], nodes: {}, busy: false },
};

// ---------------------------------------------------------------------------- boot
fetch("data/replay.json").then((r) => r.json()).then((data) => {
  S.data = data;
  initTabs();
  initConversation();
  renderResults();
  renderSystem();
  checkLive();
}).catch((e) => {
  $("#chat").innerHTML = `<p class="muted">Could not load data/replay.json (${esc(e.message)}). Run <code>python demo/build_data.py</code>.</p>`;
});

function initTabs() {
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  const fromHash = location.hash.slice(1);
  showView(["conv", "results", "system"].includes(fromHash) ? fromHash : "conv");
  window.addEventListener("hashchange", () => showView(location.hash.slice(1) || "conv"));
}
function showView(v) {
  if (!["conv", "results", "system"].includes(v)) v = "conv";
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === v)));
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${v}`; });
  if (location.hash.slice(1) !== v) history.replaceState(null, "", `#${v}`);
  if (v === "results") renderResults();
  if (v === "conv") { drawGraph(true); drawMiniCharts(); }
}

// ---------------------------------------------------------------------------- conversation: replay
function initConversation() {
  const sysSel = $("#sel-system");
  sysSel.innerHTML = Object.entries(S.data.systems).map(([k, s]) => `<option value="${k}">${esc(s.label)}</option>`).join("");
  S.system = store.get("nso.system") in S.data.systems ? store.get("nso.system") : "step12";
  sysSel.value = S.system;
  sysSel.addEventListener("change", () => { S.system = sysSel.value; store.set("nso.system", S.system); fillPeople(); });
  $("#sel-person").addEventListener("change", (e) => loadPerson(e.target.value));

  $("#b-play").addEventListener("click", togglePlay);
  $("#b-prev").addEventListener("click", () => step(-1));
  $("#b-next").addEventListener("click", () => step(1));
  $("#b-first").addEventListener("click", () => seek(0));
  $("#b-last").addEventListener("click", () => seek(S.frames.length - 1));
  $("#scrub").addEventListener("input", (e) => { stop(); seek(+e.target.value); });
  $("#speed").addEventListener("change", () => { if (S.playing) { stop(); play(); } });
  document.addEventListener("keydown", (e) => {
    if (S.mode !== "replay" || $("#view-conv").hidden || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") step(1);
    if (e.key === "ArrowLeft") step(-1);
  });
  document.querySelectorAll(".insp-tabs [role=tab]").forEach((b) => b.addEventListener("click", () => {
    S.insp = b.dataset.insp;
    document.querySelectorAll(".insp-tabs [role=tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    renderInspector();
  }));
  $("#mode-replay").addEventListener("click", () => setMode("replay"));
  $("#mode-live").addEventListener("click", () => setMode("live"));
  new ResizeObserver(() => { drawGraph(true); drawMiniCharts(); }).observe($("#graph"));
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderResults, 150); });
  fillPeople();
}

function fillPeople() {
  const sys = S.data.systems[S.system];
  const people = S.data.people.filter((p) => p.system === S.system);
  $("#system-note").textContent = `${sys.detail}. ${people.length} of ${sys.people} test people included, chosen for graphs that change. ` +
    `Graphs are rebuilt by replaying the recorded Analyzer outputs through the memory code (${sys.replay_match.nodes}/${sys.replay_match.saved} final nodes match the run's saved memory).`;
  $("#sel-person").innerHTML = people.map((p) =>
    `<option value="${p.id}">${p.id} · ${esc(p.profile.problem || "")} · ${p.sessions.length} sessions</option>`).join("");
  // people come sorted by how much their graph changes; open on the liveliest unless one was remembered
  const keep = people.find((p) => p.id === store.get("nso.person")) || people[0];
  $("#sel-person").value = keep.id;
  loadPerson(keep.id);
}

function loadPerson(id) {
  stop();
  const p = S.data.people.find((x) => x.id === id && x.system === S.system);
  S.person = p;
  store.set("nso.person", id);
  G.nodes = new Map();
  S.frames = [];
  p.sessions.forEach((s, si) => s.turns.forEach((t, ti) => S.frames.push({ si, ti })));
  $("#scrub").max = S.frames.length - 1;
  renderPerson(p.profile, p.id, p.sessions[0]?.change);
  renderChat();
  // open on the first supporter turn that wrote memory, so the first frame already shows a graph
  const first = S.frames.findIndex((f) => p.sessions[f.si].turns[f.ti].graph?.nodes.length);
  seek(Math.max(0, first));
}

function renderPerson(pr, id, change) {
  $("#p-eyebrow").textContent = id ? `Seeker ${id}` : "Seeker";
  $("#p-title").textContent = pr ? cap(pr.problem || "Open conversation") : "You";
  if (!pr) {
    $("#p-kv").innerHTML = `<dt>Role</dt><dd>You type as the seeker</dd>`;
    $("#p-truth").hidden = true;
    return;
  }
  const persona = pr.persona || {};
  $("#p-kv").innerHTML = [
    ["Feels", pr.emotion], ["Resistance", pr.resistance], ["Style", [persona.style, persona.tone, persona.verbosity].filter(Boolean).join(", ")],
    ["Opens with", pr.feeling],
  ].filter(([, v]) => v).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  $("#p-truth").hidden = false;
  $("#p-chain").innerHTML = (pr.chain || []).map((c, i) => `<li><span class="muted">depth ${i}</span> ${esc(c)}</li>`).join("") +
    (pr.memories?.length ? `<li style="list-style:none;margin-left:-18px;margin-top:6px" class="muted">Memories: ${pr.memories.map(esc).join(" · ")}</li>` : "");
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function renderSessions() {
  const p = S.mode === "live" ? null : S.person;
  const cur = currentFrame();
  const sessions = p ? p.sessions : S.live.sessions;
  $("#sessions").innerHTML = sessions.map((s, i) => {
    const succ = s.success;
    const label = i === 0 ? "first contact" : `${Math.round(s.gap)} days later`;
    return `<button class="sess" data-si="${i}" aria-current="${cur && cur.si === i}">
      <b>S${s.index}</b><span><span style="font-size:12px" class="muted">${label}</span>
      ${succ != null ? `<div class="bar" title="judge Success ${fmt(succ, 2)}"><i style="width:${succ * 100}%"></i></div>` : ""}</span>
      <span class="mono" style="font-size:12px">${succ != null ? fmt(succ, 2) : ""}</span></button>`;
  }).join("");
  $("#sessions").querySelectorAll(".sess").forEach((b) => b.addEventListener("click", () => {
    if (S.mode === "live") return;
    stop();
    seek(S.frames.findIndex((f) => f.si === +b.dataset.si));
  }));
}

function renderChat() {
  const sessions = S.mode === "live" ? S.live.sessions : S.person.sessions;
  const html = [];
  sessions.forEach((s, si) => {
    const decayed = (s.start?.events || []).length;
    html.push(`<div class="divider"><b>Session ${s.index}</b>${si ? ` · ${Math.round(s.gap)} days later` : ""}${decayed ? ` · ${decayed} belief${decayed > 1 ? "s" : ""} faded` : ""}</div>`);
    if (s.change && si) html.push(`<div class="divider" style="font-style:italic">${esc(trimText(s.change, 140))}</div>`);
    s.turns.forEach((t, ti) => html.push(msgHTML(t, si, ti)));
  });
  $("#chat").innerHTML = html.join("");
  $("#chat").querySelectorAll(".msg").forEach((el) => el.addEventListener("click", () => {
    if (S.mode === "live") { S.cursor = S.frames.findIndex((f) => f.si === +el.dataset.si && f.ti === +el.dataset.ti); syncCursor(false); return; }
    stop();
    seek(S.frames.findIndex((f) => f.si === +el.dataset.si && f.ti === +el.dataset.ti));
  }));
  $("#chat-chip").textContent = S.mode === "live" ? "live" : `${S.frames.length} turns`;
}

function msgHTML(t, si, ti) {
  const chips = [];
  if (t.role === "supporter") {
    if (t.source === "opener_pool") chips.push(`<span class="chip">opener</span>`);
    if (t.step) {
      const st = t.step;
      chips.push(`<span class="chip accent" title="${esc(LADDER[st.critic?.rung || st.strategist.requested] || "")}">${st.critic?.rung || st.strategist.requested || st.permitted}</span>`);
      if (st.path === "revised") chips.push(`<span class="chip hyp">revised ×${st.revisions}</span>`);
      if (st.path === "fallback") chips.push(`<span class="chip deny">fallback</span>`);
      if (st.critic?.ip_pred != null) chips.push(`<span class="chip" title="Critic's predicted IP">critic ${fmt(st.critic.ip_pred, 2)}</span>`);
      const w = (t.events || []).length;
      if (w) chips.push(`<span class="chip conf" title="belief changes">${w} belief Δ</span>`);
    }
    if (t.ip != null) chips.push(`<span class="chip deny" title="Judge IP for this turn (0 to 1, lower is better)">judge IP ${fmt(t.ip, 2)}</span>`);
  } else if (t.pri != null) {
    chips.push(`<span class="chip" title="Judge PRI: resistance in this reply (0 to 1)">PRI ${fmt(t.pri, 2)}</span>`);
  }
  const who = t.role === "seeker" ? "Seeker" : "Supporter";
  return `<div class="msg ${t.role}" data-si="${si}" data-ti="${ti}"><div class="who">${who}</div>${esc(t.text)}${chips.length ? `<div class="meta">${chips.join("")}</div>` : ""}</div>`;
}
const trimText = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s);

function currentFrame() { return S.frames[S.cursor]; }
function sessionsNow() { return S.mode === "live" ? S.live.sessions : S.person.sessions; }

function seek(i) {
  S.cursor = Math.max(0, Math.min(S.frames.length - 1, i));
  syncCursor(true);
}
function step(d) { stop(); seek(S.cursor + d); }
function togglePlay() { S.playing ? stop() : play(); }
function play() {
  if (S.cursor >= S.frames.length - 1) seek(0);
  S.playing = true;
  $("#b-play").textContent = "Pause";
  const tick = () => {
    if (S.cursor >= S.frames.length - 1) return stop();
    seek(S.cursor + 1);
    const f = currentFrame();
    const t = sessionsNow()[f.si].turns[f.ti];
    S.timer = setTimeout(tick, +$("#speed").value * (t.role === "supporter" ? 1.4 : 0.8));
  };
  S.timer = setTimeout(tick, 200);
}
function stop() {
  S.playing = false;
  clearTimeout(S.timer);
  $("#b-play").textContent = "Play";
}

function syncCursor(scroll) {
  const f = currentFrame();
  if (!f) return;
  $("#scrub").value = S.cursor;
  $("#pos").textContent = `S${sessionsNow()[f.si].index} · turn ${f.ti} · ${S.cursor + 1}/${S.frames.length}`;
  const msgs = $("#chat").querySelectorAll(".msg");
  let cursorEl = null;
  msgs.forEach((el) => {
    const si = +el.dataset.si, ti = +el.dataset.ti;
    const future = S.mode === "replay" && (si > f.si || (si === f.si && ti > f.ti));
    el.classList.toggle("future", future);
    const isCur = si === f.si && ti === f.ti;
    el.classList.toggle("cursor", isCur);
    if (isCur) cursorEl = el;
  });
  if (scroll && cursorEl) {
    const box = $("#chat");
    const top = cursorEl.offsetTop - box.offsetTop - box.clientHeight * 0.55;
    box.scrollTo({ top: Math.max(0, top) });
  }
  renderSessions();
  drawGraph(false);
  renderInspector();
  drawMiniCharts();
}

// The graph and the step shown for a frame: the latest supporter turn with a step at or before it.
function stateAt(frameIdx) {
  const sessions = sessionsNow();
  const f = S.frames[frameIdx];
  if (!f) return { graph: { nodes: [] }, events: [], step: null, brief: null, turn: null };
  for (let i = frameIdx; i >= 0; i--) {
    const g = S.frames[i];
    const t = sessions[g.si].turns[g.ti];
    if (g.si !== f.si) break;
    if (t.graph) return { graph: t.graph, events: i === frameIdx ? t.events : [], step: t.step, brief: t.brief, turn: t, at: i };
  }
  const s = sessions[f.si];
  return { graph: s.start?.graph || { nodes: [] }, events: f.ti === 0 ? s.start?.events || [] : [], step: null, brief: null, turn: null };
}
function nodeTable() { return S.mode === "live" ? S.live.nodes : S.person.nodes; }

// ---------------------------------------------------------------------------- need-state graph
const G = { sim: null, nodes: new Map(), w: 0, h: 0, prevStatus: new Map() };

function drawGraph(resized) {
  const svgEl = $("#graph");
  if (!svgEl || !S.frames.length || $("#view-conv").hidden) return;
  const st = stateAt(S.cursor);
  const table = nodeTable();
  const w = svgEl.clientWidth || 600, h = svgEl.clientHeight || 420;
  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${w} ${h}`);
  const cols = [0, 1, 2, 3];
  const padX = 70, colX = (d) => padX + (d / 3) * (w - padX * 2);
  if (resized || G.w !== w) {
    svg.selectAll("*").remove();
    const bg = svg.append("g").attr("class", "cols");
    const names = ["surface feeling", "need", "deeper need", "terminal need"];
    cols.forEach((c) => {
      bg.append("line").attr("class", "col-line").attr("x1", colX(c)).attr("x2", colX(c)).attr("y1", 26).attr("y2", h - 8);
      bg.append("text").attr("class", "col-label").attr("x", colX(c)).attr("y", 14).attr("text-anchor", "middle").text(`${c} · ${names[c]}`);
    });
    svg.append("g").attr("class", "edges");
    svg.append("g").attr("class", "nodes");
    G.w = w; G.h = h;
  }
  const rows = (st.graph.nodes || []).map((r) => Object.fromEntries(NODE_FIELDS.map((k, i) => [k, r[i]])));
  $("#graph-empty").hidden = rows.length > 0;
  const chain = new Set(st.graph.chain || []);
  const target = st.step?.strategist?.target;
  const radius = (d) => 7 + 13 * Math.max(0, Math.min(1, d.conf || 0));

  // keep positions across frames; new nodes start beside their parent
  const next = new Map();
  rows.forEach((r) => {
    const [text, depth] = table[r.id] || [r.id, 1];
    const old = G.nodes.get(r.id);
    const parent = G.nodes.get(r.parent);
    const n = old || { x: colX(depth) + (Math.random() - 0.5) * 10, y: parent ? parent.y + (Math.random() - 0.5) * 40 : h / 2 + (Math.random() - 0.5) * 120, fresh: true };
    Object.assign(n, r, { text, depth, fresh: !old });
    next.set(r.id, n);
  });
  G.nodes = next;
  const nodes = [...next.values()];
  const links = nodes.filter((n) => n.parent && next.has(n.parent)).map((n) => ({ source: next.get(n.parent), target: n }));

  if (!G.sim) {
    G.sim = d3.forceSimulation().alphaDecay(0.06).on("tick", ticked);
  }
  G.sim.nodes(nodes)
    .force("x", d3.forceX((d) => colX(d.depth)).strength(0.9))
    .force("y", d3.forceY((d) => (d.parent && next.has(d.parent) ? next.get(d.parent).y : h / 2)).strength(0.05))
    .force("center", d3.forceY(h / 2).strength(0.02))
    .force("collide", d3.forceCollide((d) => radius(d) + 16).strength(0.9))
    .force("link", d3.forceLink(links).distance(60).strength(0.15));
  G.sim.alpha(resized ? 0.8 : 0.5).restart();

  const edgeSel = svg.select("g.edges").selectAll("path").data(links, (d) => d.target.id);
  edgeSel.exit().remove();
  edgeSel.enter().append("path").attr("class", "edge").merge(edgeSel)
    .classed("chain", (d) => chain.has(d.source.id) && chain.has(d.target.id));

  const sel = svg.select("g.nodes").selectAll("g.node").data(nodes, (d) => d.id);
  sel.exit().transition().duration(300).style("opacity", 0).remove();
  const enter = sel.enter().append("g").attr("class", "node").style("cursor", "pointer");
  enter.append("circle").attr("class", "halo");
  enter.append("circle").attr("class", "pulse").attr("r", 0);
  enter.append("circle").attr("class", "body").attr("r", 0);
  enter.append("text").attr("class", "cf").attr("text-anchor", "middle").attr("dy", "0.35em");
  enter.append("text").attr("class", "lbl").attr("dy", "0.35em");
  enter.on("mousemove", (ev, d) => showTip(ev, d)).on("mouseleave", () => { $("#tip").hidden = true; });
  const all = enter.merge(sel);
  all.classed("dormant", (d) => !!d.dormant).classed("target", (d) => d.id === target);
  all.select("circle.body").transition().duration(450).attr("r", radius).style("fill", (d) => css(d.dormant ? "--dorm" : statusVar(d.status)));
  all.select("circle.halo").attr("r", (d) => radius(d) + 5).style("opacity", (d) => (chain.has(d.id) ? 1 : 0));
  all.select("text.cf").text((d) => (radius(d) > 12 ? Math.round((d.conf || 0) * 100) : ""));
  all.select("text.lbl").text((d) => trimText(d.text, 30) + (d.status === "disconfirmed" ? " ✕" : ""))
    .attr("x", (d) => (d.depth >= 3 ? -(radius(d) + 6) : radius(d) + 6)).attr("text-anchor", (d) => (d.depth >= 3 ? "end" : "start"))
    .style("font-weight", (d) => (chain.has(d.id) ? 600 : 400)).style("opacity", (d) => (d.dormant ? 0.6 : 1));
  // pulse nodes whose status changed or that are new at this frame
  const changed = new Set((st.events || []).map((e) => e.node));
  all.filter((d) => changed.has(d.id)).select("circle.pulse")
    .style("stroke", (d) => css(statusVar(d.status))).attr("r", radius).style("opacity", 0.9)
    .transition().duration(900).ease(d3.easeCubicOut).attr("r", (d) => radius(d) + 22).style("opacity", 0);

  function ticked() {
    const m = 30;
    nodes.forEach((d) => { d.y = Math.max(m, Math.min(h - 14, d.y)); });
    svg.select("g.edges").selectAll("path").attr("d", (d) => {
      const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y, mx = (sx + tx) / 2;
      return `M${sx},${sy} C${mx},${sy} ${mx},${ty} ${tx},${ty}`;
    });
    svg.select("g.nodes").selectAll("g.node").attr("transform", (d) => `translate(${d.x},${d.y})`);
  }

  // best guess and the events list
  const best = st.graph.best ? next.get(st.graph.best) : null;
  $("#best").innerHTML = best
    ? `${esc(best.text)} <span class="chip ${chipClass(best.status)}">${STATUS_LABEL[best.status]} · ${fmt(best.conf, 2)}</span>`
    : `<span class="muted">none yet</span>`;
  const evs = st.events || [];
  $("#events").innerHTML = evs.length
    ? evs.map((e) => `<div><span class="chip ${chipClass(e.to)}">${esc(e.to)}</span><span><b>${esc((table[e.node] || [e.node])[0])}</b> <span class="muted">${esc(e.reason)}</span>${e.quote ? ` <span class="quote" style="display:inline;border:0;padding:0">“${esc(trimText(e.quote, 120))}”</span>` : ""}</span></div>`).join("")
    : `<span class="muted">No change at this turn.</span>`;
}
const chipClass = (st) => ({ hypothesis: "hyp", proposed: "hyp", confirmed: "conf", disconfirmed: "deny", resolved: "res" }[st] || "");

function showTip(ev, d) {
  const tip = $("#tip");
  const box = $("#graph").getBoundingClientRect();
  tip.innerHTML = `<b>${esc(d.text)}</b><br>${STATUS_LABEL[d.status] || d.status}${d.dormant ? ", dormant" : ""} · confidence ${fmt(d.conf, 2)}<br>
    ${d.evidence} supporting quote${d.evidence === 1 ? "" : "s"}${d.against ? `, ${d.against} against` : ""} · seen in ${d.sessions} session${d.sessions === 1 ? "" : "s"}<br><span style="opacity:.7">${d.id}${d.parent ? ` under ${d.parent}` : " (root)"}</span>`;
  tip.hidden = false;
  const x = ev.clientX - box.left + 14, y = ev.clientY - box.top + 10;
  tip.style.left = `${Math.min(x, box.width - 290)}px`;
  tip.style.top = `${y}px`;
}

// ---------------------------------------------------------------------------- inspector
function renderInspector() {
  const st = stateAt(S.cursor);
  const el = $("#insp");
  const f = currentFrame();
  const turn = f ? sessionsNow()[f.si].turns[f.ti] : null;
  $("#insp-title").textContent = st.turn ? `Turn ${st.turn.i} inspector` : "Turn inspector";
  if (!st.step) {
    el.innerHTML = `<p class="muted" style="margin:0">${turn?.source === "opener_pool" ? "The session opens with a fixed check-in line; the pipeline runs from the supporter's first reply." : "Step to a supporter reply to see how it was made."}</p>`;
    return;
  }
  if (S.insp === "brief") {
    el.innerHTML = `<p class="note" style="margin:0 0 8px">What the memory looked like after this turn. This text goes into the next turn's prompts, always next to the raw transcript.</p><pre class="brief">${esc(st.brief || "")}</pre>`;
    return;
  }
  if (S.insp === "writes") {
    const w = st.step.writes || [];
    el.innerHTML = w.length ? `<table><thead><tr><th>Claim</th><th>Action</th><th>Node</th></tr></thead><tbody>${w.map((x) => `<tr><td style="white-space:normal">${esc(x.claim)}</td><td><span class="chip ${chipClass(x.action)}">${esc(x.action)}</span></td><td class="mono">${esc(x.node || "–")}</td></tr>`).join("")}</tbody></table>
      <p class="note">“refused” means the Analyzer gave no quoted evidence or zero confidence. Nothing enters memory ungrounded.</p>`
      : `<p class="muted">No memory writes at this turn.</p>`;
    return;
  }
  el.innerHTML = pipelineHTML(st.step, st.turn);
  el.querySelectorAll("[data-gauge]").forEach(drawGauge);
}

function pipelineHTML(s, turn) {
  const a = s.analyzer, p = s.strategist, c = s.critic;
  const needs = (a.needs || []).map((n) => `<div class="need"><span>${esc(n.text)} <span class="chip">d${n.depth}</span> ${n.hint && n.hint !== "hypothesis" ? `<span class="chip ${chipClass(n.hint)}">${esc(n.hint)}</span>` : ""}</span>
      <span class="meter" title="confidence ${fmt(n.conf, 2)}"><i style="width:${(n.conf || 0) * 100}%"></i></span></div>`).join("");
  const allowed = new Set(p.allowed || []);
  const used = c?.rung || p.requested;
  const ladder = ["L0", "L1", "L2", "L3"].map((r) => `<div class="rung ${allowed.has(r) ? "allowed" : "locked"} ${r === used ? "req" : ""}"><b>${r}</b>${LADDER[r]}</div>`).join("");
  const cal = S.mode === "live" ? { lambda: S.live.info?.lambda } : S.data.systems[S.system].calibration;
  const gateOn = S.mode === "live" ? S.live.info?.gate : !!cal;
  const decisionChip = { release: "conf", revise: "hyp", fallback: "deny" }[c?.decision] || "";
  const pathText = { released: "released as drafted", revised: `released after ${s.revisions} revision${s.revisions > 1 ? "s" : ""}`, fallback: "replaced by the reflective fallback" }[s.path] || s.path;
  return `<div class="flow">
    <div class="agent"><h4>Analyzer <span class="role">reads turns + memory brief</span>
      ${a.grounding ? `<span class="chip ${a.grounding.invalid ? "deny" : "conf"}" title="evidence spans checked against the transcript">${a.grounding.valid} grounded / ${a.grounding.invalid} invalid</span>` : ""}</h4>
      <p><b>${esc(a.emotion || "–")}</b>${a.gloss ? ` · ${esc(a.gloss)}` : ""} ${a.conf != null ? `<span class="chip">conf ${fmt(a.conf, 2)}</span>` : ""} ${a.resistance ? `<span class="chip">resistance ${esc(a.resistance)}</span>` : ""}</p>
      ${a.quote ? `<div class="quote">“${esc(trimText(a.quote, 180))}”</div>` : ""}
      ${needs ? `<div class="needs">${needs}</div>` : `<p class="muted">No inspectable needs (one-call arm).</p>`}
      ${(a.questions || []).length ? `<p class="muted" style="font-size:12.5px">Open question: ${esc(a.questions[0])}</p>` : ""}
    </div>
    <div class="agent"><h4>Strategist <span class="role">plans within the ladder</span>${p.target ? `<span class="chip accent">targets ${esc(p.target)}</span>` : ""}</h4>
      <p>${esc(p.plan || "–")}</p>
      ${(p.why || []).length ? `<ol style="margin:0;padding-left:18px;color:var(--ink-2);font-size:12.5px">${p.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ol>` : ""}
      <div class="ladder">${ladder}</div>
      ${p.restricted ? `<p class="muted" style="font-size:12.5px">Cap: ${esc(p.restricted)}</p>` : ""}
    </div>
    <div class="agent"><h4>Generator <span class="role">writes the reply</span><span class="chip">${esc(S.system === "step12" && S.mode === "replay" ? "fine-tuned 7B" : S.mode === "live" && S.live.info?.adapter ? "fine-tuned" : "Qwen2.5-7B")}</span></h4>
      <p>${esc(turn?.text || "")}</p></div>
    <div class="agent"><h4>Critic ${gateOn ? "+ conformal gate" : "(gate off in this arm)"} <span class="role">predicts intrusiveness</span>
      ${c ? `<span class="chip ${decisionChip}">${esc(c.decision)}</span>` : ""}<span class="chip">${esc(pathText)}</span></h4>
      ${c ? `<div class="gauge"><svg data-gauge data-score="${c.score}" data-lambda="${gateOn ? cal?.lambda ?? "" : ""}"></svg>
        <span class="muted" style="font-size:12px">score ${fmt(c.score, 3)}${gateOn && cal?.lambda != null ? ` · released only if ≤ λ̂ = ${fmt(cal.lambda, 3)}` : ""} · predicted IP ${fmt(c.ip_pred, 3)}</span></div>
        <div class="items">${IP_ITEMS.map((t, i) => { const v = c.items?.[String(i + 1)] ?? null; const k = v == null ? 0 : (v - 1) / 4;
          return `<div title="${esc(t)}: ${v ?? "–"} of 5" style="background:color-mix(in srgb, var(--deny) ${Math.round(k * 70)}%, var(--sunk))">${v ?? "–"}</div>`; }).join("")}</div>
        <span class="muted" style="font-size:11.5px">IP items 1–7, each scored 1–5. Hover for the item.</span>
        ${c.violations?.length ? `<p><span class="chip deny">grounding: ${esc(c.violations.join(", "))}</span></p>` : ""}
        ${c.feedback ? `<p class="muted" style="font-size:12.5px">Critic note: ${esc(trimText(c.feedback, 260))}</p>` : ""}` : `<p class="muted">No critic output stored.</p>`}
    </div>
  </div>`;
}

function drawGauge(el) {
  const score = +el.dataset.score, lam = el.dataset.lambda === "" ? null : +el.dataset.lambda;
  const w = el.clientWidth || 300, h = 34, pad = 8;
  const x = d3.scaleLinear().domain([0, 1]).range([pad, w - pad]);
  const svg = d3.select(el).attr("viewBox", `0 0 ${w} ${h}`);
  svg.append("rect").attr("x", pad).attr("y", 10).attr("width", w - 2 * pad).attr("height", 8).attr("rx", 4).style("fill", css("--sunk"));
  if (lam != null) {
    svg.append("rect").attr("x", pad).attr("y", 10).attr("width", x(lam) - pad).attr("height", 8).attr("rx", 4).style("fill", css("--conf")).style("opacity", 0.35);
    svg.append("line").attr("x1", x(lam)).attr("x2", x(lam)).attr("y1", 4).attr("y2", 24).style("stroke", css("--ink-2")).style("stroke-width", 1.5);
    svg.append("text").attr("x", x(lam)).attr("y", 33).attr("text-anchor", "middle").text("λ̂");
  }
  svg.append("circle").attr("cx", x(Math.min(1, score))).attr("cy", 14).attr("r", 6)
    .style("fill", css(lam != null && score > lam ? "--deny" : "--accent")).style("stroke", css("--panel")).style("stroke-width", 2);
  [0, 0.5, 1].forEach((t) => svg.append("text").attr("x", x(t)).attr("y", 33).attr("text-anchor", t === 0 ? "start" : t === 1 ? "end" : "middle").text(t));
}

// ---------------------------------------------------------------------------- mini charts
function drawMiniCharts() {
  if (!S.frames.length || $("#view-conv").hidden) return;
  const sessions = sessionsNow();
  const pts = [];
  S.frames.forEach((f, i) => {
    const t = sessions[f.si].turns[f.ti];
    if (t.role !== "supporter" || !t.graph) return;
    const rows = t.graph.nodes.map((r) => Object.fromEntries(NODE_FIELDS.map((k, j) => [k, r[j]])));
    pts.push({ i, si: f.si, critic: t.step?.critic?.ip_pred, judge: t.ip,
      hyp: rows.filter((r) => r.status === "hypothesis" && !r.dormant).length,
      conf: rows.filter((r) => (r.status === "confirmed" || r.status === "resolved") && !r.dormant).length,
      dorm: rows.filter((r) => r.dormant || r.status === "disconfirmed").length });
  });
  miniChart("#ch-ip", pts, (svg, x, h, m) => {
    const y = d3.scaleLinear().domain([0, Math.max(0.4, d3.max(pts, (p) => Math.max(p.critic ?? 0, p.judge ?? 0)) || 0.4)]).nice().range([h - m.b, m.t]);
    axisY(svg, y, m, x.range()[1]);
    const line = d3.line().defined((p) => p.critic != null).x((p) => x(p.i)).y((p) => y(p.critic)).curve(d3.curveMonotoneX);
    svg.append("path").datum(pts).attr("d", line).style("fill", "none").style("stroke", css("--accent")).style("stroke-width", 1.8);
    svg.selectAll(null).data(pts.filter((p) => p.judge != null)).enter().append("circle").attr("cx", (p) => x(p.i)).attr("cy", (p) => y(p.judge)).attr("r", 3.5).style("fill", css("--deny"));
  });
  miniChart("#ch-bel", pts, (svg, x, h, m) => {
    const y = d3.scaleLinear().domain([0, d3.max(pts, (p) => p.hyp + p.conf + p.dorm) || 4]).nice().range([h - m.b, m.t]);
    axisY(svg, y, m, x.range()[1], "d");
    const stack = d3.stack().keys(["conf", "hyp", "dorm"])(pts);
    const colors = { conf: "--conf", hyp: "--hyp", dorm: "--dorm" };
    stack.forEach((layer) => svg.append("path").datum(layer)
      .attr("d", d3.area().x((d) => x(d.data.i)).y0((d) => y(d[0])).y1((d) => y(d[1])).curve(d3.curveStepAfter))
      .style("fill", css(colors[layer.key])).style("opacity", 0.75));
  });
}
function miniChart(sel, pts, draw) {
  const el = $(sel);
  const w = el.clientWidth || 300, h = 150, m = { t: 8, r: 8, b: 18, l: 30 };
  const svg = d3.select(el).attr("viewBox", `0 0 ${w} ${h}`);
  svg.selectAll("*").remove();
  if (!pts.length) { svg.append("text").attr("x", w / 2).attr("y", h / 2).attr("text-anchor", "middle").text("no supporter turns yet"); return; }
  const x = d3.scaleLinear().domain([0, Math.max(1, S.frames.length - 1)]).range([m.l, w - m.r]);
  // session bands
  const sessions = sessionsNow();
  sessions.forEach((s, si) => {
    const idx = S.frames.map((f, i) => (f.si === si ? i : null)).filter((v) => v != null);
    if (!idx.length) return;
    if (si % 2) svg.append("rect").attr("x", x(idx[0])).attr("width", Math.max(0, x(idx[idx.length - 1]) - x(idx[0]))).attr("y", m.t).attr("height", h - m.t - m.b).style("fill", css("--sunk")).style("opacity", 0.6);
    svg.append("text").attr("x", x(idx[0]) + 3).attr("y", h - 5).text(`S${s.index}`);
  });
  draw(svg, x, h, m);
  if (S.mode === "replay") {
    svg.append("rect").attr("x", x(S.cursor)).attr("width", Math.max(0, x(S.frames.length - 1) - x(S.cursor))).attr("y", m.t).attr("height", h - m.t - m.b).style("fill", css("--panel")).style("opacity", 0.55);
  }
  svg.append("line").attr("x1", x(S.cursor)).attr("x2", x(S.cursor)).attr("y1", m.t).attr("y2", h - m.b).style("stroke", css("--ink-2")).style("stroke-dasharray", "3 3");
}
function axisY(svg, y, m, right, f = ".1f") {
  const g = svg.append("g").attr("class", "grid");
  y.ticks(4).forEach((t) => {
    g.append("line").attr("x1", m.l).attr("x2", right).attr("y1", y(t)).attr("y2", y(t));
    svg.append("text").attr("x", m.l - 5).attr("y", y(t) + 3).attr("text-anchor", "end").text(d3.format(f)(t));
  });
}

// ---------------------------------------------------------------------------- live mode
function checkLive() {
  fetch("api/status").then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status)))).then((st) => {
    S.live.available = !!st.live;
    S.live.info = st;
    $("#live-dot").classList.toggle("on", S.live.available);
    $("#live-text").textContent = S.live.available ? `Live · ${st.backend}${st.adapter ? " · fine-tuned" : ""}` : "Replay data";
    $("#sel-live-person").innerHTML = `<option value="">You play the seeker</option>` +
      (st.people || []).map((p) => `<option value="${p.id}">${p.id} · ${esc(p.problem || "")} · ${esc(p.resistance || "")}</option>`).join("");
    if (st.error) $("#live-note").textContent = st.error;
  }).catch(() => { S.live.available = false; });
}

function setMode(m) {
  if (m === "live" && !S.live.available) {
    S.mode = "live";
    $("#mode-replay").setAttribute("aria-pressed", "false");
    $("#mode-live").setAttribute("aria-pressed", "true");
    $("#replay-controls").hidden = true;
    $("#live-controls").hidden = false;
    $("#live-note").innerHTML = `Live chat runs the real pipeline, so it needs the model. On a GPU machine with Ollama run:<br><code class="mono" style="font-size:12px">python demo/server.py --live --backend ollama</code><br>and open the address it prints. This copy only has the recorded test runs.`;
    ["#live-reset", "#live-next", "#sel-live-person"].forEach((s) => { $(s).disabled = true; });
    stop();
    S.live.sessions = []; S.frames = []; S.live.nodes = {};
    $("#chat").innerHTML = `<p class="muted">Live mode is off on this page. Switch back to Replay to watch recorded test conversations.</p>`;
    $("#composer").hidden = true;
    $("#transport-panel").hidden = true;
    renderPerson(null);
    renderSessions();
    d3.select("#graph").selectAll("*").remove(); G.nodes = new Map(); G.w = 0;
    $("#graph-empty").hidden = false; $("#insp").innerHTML = ""; $("#events").innerHTML = ""; $("#best").textContent = "";
    ["#ch-ip", "#ch-bel"].forEach((s) => d3.select(s).selectAll("*").remove());
    return;
  }
  S.mode = m;
  $("#mode-replay").setAttribute("aria-pressed", String(m === "replay"));
  $("#mode-live").setAttribute("aria-pressed", String(m === "live"));
  $("#replay-controls").hidden = m !== "replay";
  $("#live-controls").hidden = m !== "live";
  $("#transport-panel").hidden = m !== "replay";
  $("#composer").hidden = m !== "live";
  G.nodes = new Map(); G.w = 0;
  if (m === "replay") { fillPeople(); return; }
  ["#live-reset", "#live-next", "#sel-live-person"].forEach((s) => { $(s).disabled = false; });
  $("#live-note").textContent = `Live: ${S.live.info.backend}${S.live.info.adapter ? ", fine-tuned Generator" : ", untuned Generator"}, gate ${S.live.info.gate ? `on (λ̂ ${fmt(S.live.info.lambda, 3)})` : "off"}. Pick a test person to let the simulated seeker talk, or type yourself.`;
  if (!S.live.bound) bindLive();
  liveReset();
}

function bindLive() {
  S.live.bound = true;
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#msg").value.trim(); if (t) liveTurn(t); });
  $("#auto").addEventListener("click", () => liveTurn(""));
  $("#live-reset").addEventListener("click", liveReset);
  $("#live-next").addEventListener("click", () => livePost("api/live/session", { gap: +$("#live-gap").value || 7 }).then((r) => {
    S.live.nodes = r.nodes;
    S.live.sessions.push({ index: r.session, gap: r.gap, start: { graph: r.graph, events: r.events || [] }, turns: r.turns });
    liveRefresh();
  }));
}

function livePost(url, body) {
  S.live.busy = true;
  ["#send", "#auto", "#live-next", "#live-reset"].forEach((s) => { $(s).disabled = true; });
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then(async (r) => { const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j; })
    .catch((e) => { $("#chat").insertAdjacentHTML("beforeend", `<p class="chip deny" style="align-self:center">${esc(e.message)}</p>`); throw e; })
    .finally(() => {
      S.live.busy = false;
      ["#send", "#auto", "#live-next", "#live-reset"].forEach((s) => { $(s).disabled = false; });
      $("#auto").disabled = !$("#sel-live-person").value;
    });
}

function liveReset() {
  const pid = $("#sel-live-person").value;
  livePost("api/live/reset", { profile: pid || null }).then((r) => {
    S.live.nodes = r.nodes;
    S.live.sessions = [{ index: 1, gap: 0, start: { graph: r.graph, events: [] }, turns: r.turns }];
    renderPerson(r.profile ? { ...r.profile, persona: {} } : null, r.profile?.id);
    d3.select("#graph").selectAll("*").remove(); G.nodes = new Map(); G.w = 0;
    liveRefresh();
  });
}

function liveTurn(text) {
  $("#msg").value = "";
  if (text) $("#chat").insertAdjacentHTML("beforeend", `<div class="msg seeker fresh"><div class="who">Seeker</div>${esc(text)}</div>`);
  $("#chat").insertAdjacentHTML("beforeend", `<div class="typing" id="typing">${text ? "" : "Simulated seeker is writing… then "}Analyzer → Strategist → Generator → Critic…</div>`);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  livePost("api/live/turn", { text }).then((r) => {
    S.live.nodes = r.nodes;
    const s = S.live.sessions[S.live.sessions.length - 1];
    s.turns.push(r.seeker, r.supporter);
    liveRefresh(true);
  }).catch(() => {}).finally(() => $("#typing")?.remove());
}

function liveRefresh(fresh) {
  S.frames = [];
  S.live.sessions.forEach((s, si) => s.turns.forEach((t, ti) => S.frames.push({ si, ti })));
  renderChat();
  if (fresh) [...$("#chat").querySelectorAll(".msg")].slice(-2).forEach((el) => el.classList.add("fresh"));
  S.cursor = S.frames.length - 1;
  syncCursor(true);
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

// ---------------------------------------------------------------------------- results
const METRICS = [
  { key: "success", label: "Success", better: "higher", note: "the judge's rating of whether the supporter found the underlying need" },
  { key: "ip", label: "Intrusiveness (IP)", better: "lower", note: "turn-level intrusiveness, 4 sampled supporter turns per session" },
  { key: "pri", label: "Resistance (PRI)", better: "lower", note: "resistance in the seeker's next turn" },
  { key: "pri_cf", label: "Counterfactual PRI", better: "lower", note: "resistance the turn caused, against a neutral reflective control replay" },
  { key: "aels", label: "AELS listening", better: "higher", note: "Active-Empathic Listening Scale, session level" },
];
let resultsState = { metric: store.get("nso.metric") || "ip", ct: "all", cal: "step12", drawn: false };

function renderResults() {
  if (!S.data || $("#view-results").hidden) return;
  const d = S.data, c = d.corpus;
  $("#kpis").innerHTML = [
    [c.profiles.toLocaleString(), "seeker profiles seeded from EmpatheticDialogues"],
    [c.sessions_kept.toLocaleString(), "linked multi-session dialogues kept after filtering"],
    [c.annotations.toLocaleString(), "supporter turns annotated with Analysis and Strategy"],
    [(c.turns_per_session.mean).toFixed(1), "turns per session on average"],
    ["100", "held-out test people, 2–4 sessions each"],
    [`${d.contrasts.filter((x) => x.survives).length}/${d.contrasts.length}`, "headline contrasts that survive Holm"],
  ].map(([v, l]) => `<div class="panel kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

  $("#metric-seg").innerHTML = METRICS.map((m) => `<button aria-pressed="${m.key === resultsState.metric}" data-m="${m.key}">${m.label}</button>`).join("");
  $("#metric-seg").querySelectorAll("button").forEach((b) => b.onclick = () => { resultsState.metric = b.dataset.m; store.set("nso.metric", b.dataset.m); renderResults(); });
  $("#ct-seg").innerHTML = ["all", "ip", "success", "pri"].map((k) => `<button aria-pressed="${k === resultsState.ct}" data-k="${k}">${k === "all" ? "All" : k === "ip" ? "IP" : k === "pri" ? "PRI" : "Success"}</button>`).join("");
  $("#ct-seg").querySelectorAll("button").forEach((b) => b.onclick = () => { resultsState.ct = b.dataset.k; renderResults(); });
  const calSystems = Object.entries(d.systems).filter(([, s]) => s.calibration);
  $("#cal-seg").innerHTML = calSystems.map(([k]) => `<button aria-pressed="${k === resultsState.cal}" data-k="${k}">${k === "step12" ? "Fine-tuned Generator" : k}</button>`).join("");

  drawArms(); drawScatter(); drawCalibration(); drawRecovery(); drawContrasts(); drawCorpus(); drawTable();
}

function chartFrame(sel, height, m) {
  const el = $(sel);
  const w = Math.max(280, el.clientWidth || el.parentElement.clientWidth || 600);
  const svg = d3.select(el).attr("viewBox", `0 0 ${w} ${height}`).attr("height", height);
  svg.selectAll("*").remove();
  return { svg, w, h: height, m };
}

function drawArms() {
  const metric = METRICS.find((m) => m.key === resultsState.metric);
  const rows = S.data.arms.filter((a) => a[metric.key]?.mean != null);
  const groups = [...new Set(rows.map((r) => r.group))];
  const rowH = 22, groupGap = 18;
  const height = rows.length * rowH + groups.length * groupGap + 40;
  const narrow = ($("#ch-arms").clientWidth || 600) < 560;
  const { svg, w, h } = chartFrame("#ch-arms", height, null);
  const m = { l: narrow ? 150 : 270, r: 60, t: 10, b: 26 };
  const lo = d3.min(rows, (r) => r[metric.key].lo ?? r[metric.key].mean), hi = d3.max(rows, (r) => r[metric.key].hi ?? r[metric.key].mean);
  const pad = (hi - lo) * 0.15 || 0.01;
  const x = d3.scaleLinear().domain([Math.min(0, lo - pad) < 0 && lo < 0 ? lo - pad : Math.max(0, lo - pad), hi + pad]).nice().range([m.l, w - m.r]);
  x.ticks(6).forEach((t) => {
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", h - m.b).style("stroke", css("--line")).style("stroke-opacity", t === 0 ? 1 : 0.5);
    svg.append("text").attr("x", x(t)).attr("y", h - 8).attr("text-anchor", "middle").text(d3.format(".2f")(t));
  });
  let y = m.t;
  const ref = rows.find((r) => r.key === "cellA");
  groups.forEach((g) => {
    svg.append("text").attr("x", 0).attr("y", y + 12).style("font-weight", 600).style("fill", css("--ink-2")).style("text-transform", "uppercase").style("letter-spacing", ".06em").text(g);
    y += groupGap;
    rows.filter((r) => r.group === g).forEach((r) => {
      const v = r[metric.key];
      const cy = y + rowH / 2;
      const hl = r.key === "step12";
      svg.append("text").attr("x", 10).attr("y", cy + 4).style("fill", css(hl ? "--accent" : "--ink")).style("font-size", "12px").style("font-weight", hl ? 600 : 400)
        .text(trimText(r.label, narrow ? 22 : 44));
      if (v.lo != null) svg.append("line").attr("x1", x(v.lo)).attr("x2", x(v.hi)).attr("y1", cy).attr("y2", cy).style("stroke", css(hl ? "--accent" : "--ink-3")).style("stroke-width", 2);
      svg.append("circle").attr("cx", x(v.mean)).attr("cy", cy).attr("r", 4.5).style("fill", css(hl ? "--accent" : "--ink")).append("title")
        .text(`${r.label}: ${fmt(v.mean)} [${fmt(v.lo)}, ${fmt(v.hi)}], ${v.n ?? r.sessions} items`);
      svg.append("text").attr("x", w - m.r + 6).attr("y", cy + 4).attr("class", "mono").style("font-size", "11px").style("fill", css("--ink-2")).text(fmt(v.mean));
      y += rowH;
    });
  });
  if (ref && ref[metric.key]) {
    svg.append("line").attr("x1", x(ref[metric.key].mean)).attr("x2", x(ref[metric.key].mean)).attr("y1", m.t).attr("y2", h - m.b)
      .style("stroke", css("--accent")).style("stroke-dasharray", "4 4").style("opacity", 0.6);
  }
  const best = metric.better === "lower" ? d3.least(rows, (r) => r[metric.key].mean) : d3.greatest(rows, (r) => r[metric.key].mean);
  $("#metric-finding").textContent = `${metric.label}: ${metric.note}; ${metric.better} is better. Best here: ${best.label} (${fmt(best[metric.key].mean)}). The dashed line is cell A, the one-call untuned listener.`;
}

function drawScatter() {
  const rows = S.data.arms.filter((a) => a.ip?.mean != null && a.success?.mean != null && a.group !== "Corpus comparison");
  const { svg, w, h } = chartFrame("#ch-scatter", 300, null);
  const m = { l: 58, r: 16, t: 12, b: 34 };
  const x = d3.scaleLinear().domain(d3.extent(rows, (r) => r.ip.mean)).nice().range([m.l, w - m.r]);
  const y = d3.scaleLinear().domain(d3.extent(rows, (r) => r.success.mean)).nice().range([h - m.b, m.t]);
  x.ticks(5).forEach((t) => { svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", h - m.b).style("stroke", css("--line")).style("stroke-opacity", 0.5); svg.append("text").attr("x", x(t)).attr("y", h - m.b + 14).attr("text-anchor", "middle").text(d3.format(".2f")(t)); });
  y.ticks(5).forEach((t) => { svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(t)).attr("y2", y(t)).style("stroke", css("--line")).style("stroke-opacity", 0.5); svg.append("text").attr("x", m.l - 6).attr("y", y(t) + 3).attr("text-anchor", "end").text(d3.format(".2f")(t)); });
  svg.append("text").attr("x", (m.l + w - m.r) / 2).attr("y", h - 4).attr("text-anchor", "middle").text("intrusiveness (IP) → worse");
  svg.append("text").attr("transform", `translate(12,${(m.t + h - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("Success → better");
  const named = { cellA: "A", cellB: "B", cellC: "C", cellD: "D", step12: "step 12", dpo: "DPO", steer: "steering", reactive_baseline: "reactive", mem_needstate: "need-state" };
  const groupColor = { "Baselines": "--ink-3", "2 x 2 architecture": "--accent", "Full system": "--accent", "Fine-tuning": "--res", "Memory": "--hyp", "Alternatives": "--conf" };
  rows.forEach((r) => {
    const g = svg.append("g");
    g.append("circle").attr("cx", x(r.ip.mean)).attr("cy", y(r.success.mean)).attr("r", r.key === "step12" ? 7 : 5)
      .style("fill", css(groupColor[r.group] || "--ink-3")).style("stroke", css("--panel")).style("stroke-width", 1.5)
      .append("title").text(`${r.label}\nIP ${fmt(r.ip.mean)} · Success ${fmt(r.success.mean)}${r.calls ? ` · ${r.calls} model calls per turn` : ""}`);
    if (named[r.key]) g.append("text").attr("x", x(r.ip.mean) + 8).attr("y", y(r.success.mean) + (r.key === "reactive_baseline" ? 14 : 3)).style("fill", css("--ink-2")).style("font-size", "11px").text(named[r.key]);
  });
}

function drawCalibration() {
  const sys = S.data.systems[resultsState.cal];
  const cal = sys?.calibration;
  const { svg, w, h } = chartFrame("#ch-cal", 240, null);
  if (!cal) return;
  const m = { l: 40, r: 14, t: 10, b: 32 };
  const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
  [0, .25, .5, .75, 1].forEach((t) => {
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(t)).attr("y2", y(t)).style("stroke", css("--line")).style("stroke-opacity", 0.5);
    svg.append("text").attr("x", m.l - 6).attr("y", y(t) + 3).attr("text-anchor", "end").text(t);
    svg.append("text").attr("x", x(t)).attr("y", h - m.b + 14).attr("text-anchor", "middle").text(t);
  });
  svg.append("text").attr("x", (m.l + w) / 2).attr("y", h - 3).attr("text-anchor", "middle").text("threshold λ on the Critic's score");
  svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(cal.alpha)).attr("y2", y(cal.alpha)).style("stroke", css("--ink-3")).style("stroke-dasharray", "5 4");
  const line = (k, color) => svg.append("path").datum(cal.curve.filter((p) => p[k] != null))
    .attr("d", d3.line().x((p) => x(p.lambda)).y((p) => y(p[k])).curve(d3.curveStepAfter)).style("fill", "none").style("stroke", css(color)).style("stroke-width", 2);
  line("release", "--accent"); line("ucb", "--deny");
  svg.append("line").attr("x1", x(cal.lambda)).attr("x2", x(cal.lambda)).attr("y1", m.t).attr("y2", h - m.b).style("stroke", css("--ink"));
  svg.append("text").attr("x", x(cal.lambda) + 4).attr("y", m.t + 10).style("fill", css("--ink")).text(`λ̂ = ${cal.lambda}`);
  const at = cal.curve.find((p) => p.lambda === cal.lambda);
  $("#cal-finding").textContent = `Calibrated on ${cal.n} held-out drafts at α = ${cal.alpha}, τ = ${cal.tau}, δ = ${cal.delta}. λ̂ = ${cal.lambda} is the largest threshold whose risk upper bound stays under α; it releases ${Math.round((at?.release ?? 0) * 100)}% of drafts as written.`;
}

function drawRecovery() {
  const rec = S.data.recovery;
  const { svg, w, h } = chartFrame("#ch-rec", 240, null);
  const m = { l: 40, r: 14, t: 10, b: 32 };
  const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
  [0, .25, .5, .75, 1].forEach((t) => {
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(t)).attr("y2", y(t)).style("stroke", css("--line")).style("stroke-opacity", 0.5);
    svg.append("text").attr("x", m.l - 6).attr("y", y(t) + 3).attr("text-anchor", "end").text(t);
    svg.append("text").attr("x", x(t)).attr("y", h - m.b + 14).attr("text-anchor", "middle").text(t);
  });
  svg.append("text").attr("x", (m.l + w) / 2).attr("y", h - 3).attr("text-anchor", "middle").text("cost: share of all drafts stopped");
  svg.append("text").attr("transform", `translate(10,${(m.t + h - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("intrusive drafts caught");
  svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).style("stroke", css("--ink-3")).style("stroke-dasharray", "4 4");
  const colors = ["--accent", "--hyp"];
  Object.entries(rec).forEach(([name, pts], i) => {
    const sorted = [...pts].sort((a, b) => a.cost - b.cost);
    svg.append("path").datum(sorted).attr("d", d3.line().x((p) => x(p.cost)).y((p) => y(p.recovery))).style("fill", "none").style("stroke", css(colors[i])).style("stroke-width", 2);
    svg.selectAll(null).data(sorted).enter().append("circle").attr("cx", (p) => x(p.cost)).attr("cy", (p) => y(p.recovery)).attr("r", 2.5).style("fill", css(colors[i]))
      .append("title").text((p) => `${name}, λ ${p.lambda}: stops ${Math.round(p.cost * 100)}% of drafts, catches ${Math.round(p.recovery * 100)}% of intrusive ones`);
  });
}

function drawContrasts() {
  const rows = S.data.contrasts.filter((c) => resultsState.ct === "all" || c.scale === resultsState.ct);
  const narrow = ($("#ch-ct").clientWidth || 600) < 560;
  const rowH = 17, m = { l: narrow ? 150 : 250, r: 12, t: 6, b: 24 };
  const { svg, w, h } = chartFrame("#ch-ct", rows.length * rowH + m.t + m.b, null);
  const ext = Math.max(0.08, d3.max(rows, (r) => Math.max(Math.abs(r.lo), Math.abs(r.hi))));
  const x = d3.scaleLinear().domain([-ext, ext]).nice().range([m.l, w - m.r]);
  x.ticks(5).forEach((t) => {
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", h - m.b).style("stroke", css("--line")).style("stroke-opacity", t === 0 ? 1 : 0.4);
    svg.append("text").attr("x", x(t)).attr("y", h - 8).attr("text-anchor", "middle").text(d3.format("+.2f")(t).replace("+0.00", "0"));
  });
  const scaleColor = { ip: "--deny", success: "--conf", pri: "--hyp" };
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    const g = svg.append("g");
    g.append("text").attr("x", 0).attr("y", cy + 3.5).style("font-size", "11px").style("fill", css(r.survives ? "--ink" : "--ink-3")).style("font-weight", r.survives ? 600 : 400)
      .text(trimText(`${r.scale.toUpperCase()} · ${r.contrast}`, narrow ? 26 : 46));
    g.append("line").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", cy).attr("y2", cy).style("stroke", css(scaleColor[r.scale])).style("stroke-width", 1.6).style("opacity", r.survives ? 1 : 0.5);
    g.append("circle").attr("cx", x(r.diff)).attr("cy", cy).attr("r", 4).style("fill", r.survives ? css(scaleColor[r.scale]) : css("--panel")).style("stroke", css(scaleColor[r.scale])).style("stroke-width", 1.5);
    g.append("title").text(`${r.part}\n${r.contrast} (${r.scale})\ndifference ${sgn(r.diff)} [${sgn(r.lo)}, ${sgn(r.hi)}]${r.paired ? ", paired" : ""}\np = ${r.p}, Holm p = ${r.holm} → ${r.survives ? "survives" : "does not survive"}`);
  });
  const surv = S.data.contrasts.filter((c) => c.survives);
  const by = d3.rollup(surv, (v) => v.length, (c) => c.scale);
  $("#ch-ct").closest(".panel-b").querySelector(".finding").textContent =
    `${surv.length} of ${S.data.contrasts.length} headline contrasts survive Holm correction at 0.05: ${[...by].map(([k, n]) => `${n} on ${k === "ip" ? "IP" : k === "pri" ? "PRI" : "Success"}`).join(", ")}.`;
}

function drawCorpus() {
  const c = S.data.corpus;
  const pt = Object.entries(c.problem_types || {}).sort((a, b) => b[1] - a[1]);
  const rowH = 18, m = { l: 150, r: 40, t: 4, b: 8 };
  const { svg, w, h } = chartFrame("#ch-corpus", pt.length * rowH + m.t + m.b, null);
  const x = d3.scaleLinear().domain([0, d3.max(pt, (d) => d[1])]).range([m.l, w - m.r]);
  pt.forEach(([k, v], i) => {
    const cy = m.t + i * rowH;
    svg.append("text").attr("x", m.l - 8).attr("y", cy + 12).attr("text-anchor", "end").style("font-size", "11.5px").style("fill", css("--ink-2")).text(k);
    svg.append("rect").attr("x", m.l).attr("y", cy + 3).attr("height", rowH - 6).attr("width", Math.max(1, x(v) - m.l)).attr("rx", 2).style("fill", css("--accent")).style("opacity", 0.8);
    svg.append("text").attr("x", x(v) + 5).attr("y", cy + 12).attr("class", "mono").style("font-size", "11px").text(v);
  });
  const r = c.resistance;
  $("#corpus-finding").textContent = `${c.profiles.toLocaleString()} profiles by problem type. Resistance is set by design: ${r.low} low, ${r.medium} medium, ${r.high} high. ` +
    `${c.sessions_kept.toLocaleString()} of ${c.sessions_generated.toLocaleString()} sessions passed the AELS quality filter; the median gap between sessions is ${c.gap_days.median.toFixed(1)} days.`;
}

function drawTable() {
  const cols = [["success", "Success"], ["ip", "IP"], ["pri", "PRI"], ["pri_cf", "PRI cf"], ["aels", "AELS"]];
  $("#arm-table").innerHTML = `<thead><tr><th>Group</th><th>System</th><th>Sessions</th>${cols.map(([, l]) => `<th style="text-align:right">${l}</th>`).join("")}<th style="text-align:right">Calls/turn</th><th>Source</th></tr></thead><tbody>` +
    S.data.arms.map((a) => `<tr class="${a.key === "step12" ? "hl" : ""}"><td class="muted">${esc(a.group)}</td><td>${esc(a.label)}</td><td class="num">${a.sessions}</td>` +
      cols.map(([k]) => `<td class="num" title="${a[k]?.lo != null ? `[${fmt(a[k].lo)}, ${fmt(a[k].hi)}]` : ""}">${a[k] ? fmt(a[k].mean) : "–"}</td>`).join("") +
      `<td class="num">${a.calls != null ? a.calls.toFixed(2) : "–"}</td><td class="mono muted" style="font-size:11px">${esc(a.source.replace("results/v3_full1000/", ""))}</td></tr>`).join("") + "</tbody>";
}

// ---------------------------------------------------------------------------- system
function renderSystem() {
  const sys = S.data.systems;
  $("#sys-ladder").innerHTML = `<div class="ladder" style="grid-template-columns:1fr">${Object.entries(LADDER).map(([k, v]) => `<div class="rung allowed"><b>${k}</b>${v}</div>`).join("")}</div>
    <p class="note">The Strategist may pick any rung up to the cap. The Analyzer's confidence below 0.45 caps the turn at L1. L3 also needs the terminal need confirmed in memory.</p>`;
  $("#sys-items").innerHTML = IP_ITEMS.map((t) => `<li>${esc(t)}</li>`).join("");
  $("#sys-prose").innerHTML = `
    <h3 style="margin-top:0">What you are watching</h3>
    <p>A supporter that tries to find the need underneath what someone says, without getting ahead of them. Each reply goes through four agents. Between replies a belief memory keeps the candidate needs as a graph: feelings on the left, deeper needs to the right, each with quoted evidence and a confidence.</p>
    <h3>How the memory changes</h3>
    <ul>
      <li><b>Proposed:</b> the Analyzer names a need with a quote from the seeker. A need without a quote is refused.</li>
      <li><b>Confirmed:</b> the seeker agrees with it in their latest turn. Rival needs under the same parent drop to 0.7× their confidence.</li>
      <li><b>Denied:</b> the seeker rejects it. The denial is kept with their own words, and the need cannot be proposed again.</li>
      <li><b>Fading:</b> between sessions every belief except a denial halves over 14 days (3× slower once confirmed, 1.5× slower for each later session that brings it back). Under 0.15 it goes dormant; under 0.05 it is pruned.</li>
    </ul>
    <h3>Where the replay comes from</h3>
    <p>The ${Object.values(sys).map((s) => s.label.toLowerCase()).join(" and ")} conversations are the recorded test runs on 100 held-out simulated people. The page rebuilds every graph by feeding each turn's stored Analyzer output back through the project's memory code. The rebuilt final graphs match what the runs saved: ${Object.values(sys).map((s) => `${s.replay_match.nodes}/${s.replay_match.saved}`).join(" and ")} nodes.</p>
    <h3>Read the numbers with care</h3>
    <ul>
      <li>The seekers are simulated by Qwen2.5-7B, and the scores come from one LLM judge (Mistral-Nemo-12B). No human ratings have been collected yet.</li>
      <li>The full system ran on the transformers backend and cells A–D on Ollama. The backend alone adds +0.017 IP.</li>
      <li>Outside the training distribution (ExTES people) the gate stays under α only by falling back on 65% of turns.</li>
    </ul>
    <p class="note">Built from <code>${esc(S.data.built_from)}</code> by <code>demo/build_data.py</code>.</p>`;
}

})();
