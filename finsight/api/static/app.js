// FinSight dashboard. All model and filing text is inserted with textContent (never as HTML).
"use strict";

const state = { key: null, jobsBefore: null, poll: null };
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
};

function getKey() {
  try { state.key = sessionStorage.getItem("finsight-key"); } catch { state.key = null; }
  if (state.key) return Promise.resolve(state.key);
  const dialog = $("#key-dialog");
  dialog.showModal();
  return new Promise((resolve) => {
    $("#key-form").addEventListener("submit", () => {
      state.key = $("#key-form").key.value.trim();
      try { sessionStorage.setItem("finsight-key", state.key); } catch { /* private mode: keep in memory */ }
      resolve(state.key);
    }, { once: true });
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "X-API-Key": state.key, "Content-Type": "application/json", ...(options.headers || {}) } });
  if (response.status === 401) {
    try { sessionStorage.removeItem("finsight-key"); } catch { /* ignore */ }
    state.key = null; await getKey(); return api(path, options);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("json") ? response.json() : response.text();
}

const fmtDate = (s) => (s ? new Date(s).toLocaleString() : "—");
const money = (v) => {
  const n = Number(v);
  if (!isFinite(n)) return String(v);
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)} bn`;
  if (abs >= 1e6) return `${(n / 1e6).toLocaleString(undefined, { maximumFractionDigits: 0 })} m`;
  return n.toLocaleString();
};
const value = (v, unit) => (v === null || v === undefined ? "—" : unit === "ratio" ? `${(Number(v) * 100).toFixed(1)}%` : `${money(v)} ${unit || ""}`.trim());
const badge = (text) => el("span", { class: `badge ${text}`, text: String(text).replaceAll("_", " ") });

// ------------------------------------------------------------------ status strip
async function refreshStatus() {
  const box = $("#status");
  try {
    const h = await api("/health");
    const pills = [
      ["Database", "ok", "ok"],
      ["Jobs", `${h.jobs.running} running · ${h.jobs.queued} queued`, h.jobs.running || h.jobs.queued ? "warn" : "ok"],
      ["Model server", h.model_server.reachable ? `${h.model_server.models.length} models` : "unreachable", h.model_server.reachable ? "ok" : "bad"],
      ["SEC contact", h.sec_contact_configured ? "set" : "missing", h.sec_contact_configured ? "ok" : "bad"],
      ["Claude spend", `$${Number(h.spend.committed_usd).toFixed(4)} of $${h.spend.limit_usd}${h.spend.claude_enabled ? "" : " · off"}`, "ok"],
      ["Watchlist", `${h.watchlist.tickers} tickers`, "ok"],
    ];
    box.replaceChildren(...pills.map(([k, v, cls]) => el("span", { class: `pill ${cls}`, text: `${k}: ${v}` })));
  } catch (e) {
    box.replaceChildren(el("span", { class: "pill bad", text: `API: ${e.message}` }));
  }
}

async function loadModels() {
  const select = $("#model-select");
  try {
    const { options, note } = await api("/models");
    select.replaceChildren(...options.map((o) => el("option", { value: `${o.provider}|${o.model}`, text: `${o.model}${o.paid ? " (Claude, paid)" : ""}` })));
    $("#model-hint").textContent = options.length ? note : "No model server reachable: start vLLM and the SSH tunnel, then reload.";
  } catch (e) { $("#model-hint").textContent = e.message; }
}

// ------------------------------------------------------------------ research
$("#research-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const metrics = [...form.querySelectorAll("input[name=metrics]:checked")].map((i) => i.value);
  const [provider, model] = form.model.value.split("|");
  if (!metrics.length) { alertBox("Choose at least one metric."); return; }
  const company = form.company.value.trim().toUpperCase();
  const period = `FY${form.year.value.trim()}`;
  const body = { company, period, metrics, question: `Report ${metrics.join(", ")} for ${company} ${period}.`,
                 model: { provider, model, max_tokens: 2048 } };
  form.querySelector("button[type=submit]").disabled = true;
  try {
    const job = await api("/research-jobs", { method: "POST", body: JSON.stringify(body) });
    watchJob(job.id);
  } catch (e) { alertBox(e.message); } finally { form.querySelector("button[type=submit]").disabled = false; }
});

function alertBox(message) {
  $("#report").replaceChildren(el("p", { class: "error", text: message }));
  $("#report").classList.remove("empty");
}

function watchJob(id) {
  clearInterval(state.poll);
  showTab("research");
  const box = $("#report");
  box.classList.remove("empty");
  const tick = async () => {
    try {
      const job = await api(`/research-jobs/${id}`);
      if (["queued", "running"].includes(job.status)) {
        box.replaceChildren(el("p", {}, badge(job.status), ` ${job.request.company} ${job.request.period} · stage: ${job.stage || "waiting"} · job ${id}`));
        return;
      }
      clearInterval(state.poll);
      await renderReport(id, job);
      refreshStatus();
    } catch (e) { clearInterval(state.poll); alertBox(e.message); }
  };
  tick();
  state.poll = setInterval(tick, 2000);
}

async function renderReport(id, job) {
  const box = $("#report");
  let data;
  try { data = await api(`/research-jobs/${id}/report`); }
  catch (e) { box.replaceChildren(el("p", {}, badge(job.status), ` ${job.outcome_reason || e.message}`)); return; }
  const r = data.report;
  const f = r.filing || {};
  const head = el("div", { class: "report-head" },
    el("h2", { text: `${r.company}: ${r.period}` }),
    el("div", {}, badge(r.outcome), " ",
      el("button", { class: "ghost", onclick: () => downloadMarkdown(id, r) , text: "Download Markdown" })));
  const source = f.accession
    ? el("p", { class: "source" }, `${f.form} · accession ${f.accession} · period ending ${f.period_end} · filed ${f.filing_date} · `,
        el("a", { href: f.source_uri, target: "_blank", rel: "noopener noreferrer", text: "original filing" }),
        ` · sha256 ${String(f.sha256).slice(0, 12)} · amounts as reported in this filing`)
    : el("p", { class: "source", text: "No filing was used (see unresolved information)." });
  const rows = (r.findings || []).map((x) => el("tr", {},
    el("td", { text: x.metric.replaceAll("_", " ") }), el("td", { text: x.period }), el("td", { class: "num", text: value(x.value, x.unit) }),
    el("td", {}, badge(x.status)), el("td", {}, ...(x.checks || []).map((c) => badge(c.includes("(agrees)") ? "agrees" : c.includes("(differs)") ? "differs" : "check")))));
  const table = el("table", { class: "table" }, el("thead", {}, el("tr", {}, ...["Metric", "Period", "Value", "Status", "XBRL"].map((t) => el("th", { text: t })))), el("tbody", {}, ...rows));
  const details = (r.findings || []).filter((x) => x.status === "accepted" || (x.issues || []).length).map((x) => {
    const calc = [
      ...(x.operands || []).map((o) => `${o.name} = ${o.original_text} (${o.scale}, source period ${o.source_period || "?"}) [${o.passage_id}]`),
      ...(x.calculation || []).map((s) => `${s.op}(${s.args.join(", ")}) = ${s.value}`),
      ...(x.checks || []).map((c) => `check: ${c}`), ...(x.issues || []).map((i) => `issue: ${i}`)].join("\n");
    const sources = el("div");
    const ids = [...new Set((x.operands || []).map((o) => o.passage_id))];
    const toggle = el("button", { class: "ghost", text: `Show ${ids.length} cited passage${ids.length === 1 ? "" : "s"}`, onclick: async () => {
      toggle.disabled = true;
      try {
        const { passages } = await api(`/passages?ids=${encodeURIComponent(ids.join(","))}`);
        sources.replaceChildren(...passages.map((p) => el("div", { class: "passage", text: `${p.id} (${p.section}${p.location && p.location.page ? `, page ${p.location.page}` : ""})\n${p.text}` })));
      } catch (e) { sources.replaceChildren(el("p", { class: "error", text: e.message })); }
    } });
    return el("div", { class: "finding" }, el("h3", { text: `${x.metric.replaceAll("_", " ")} (${x.period})` }), el("div", { class: "calc", text: calc }), ids.length ? toggle : null, sources);
  });
  const unresolved = el("ul", {}, ...((r.unresolved || []).length ? r.unresolved : ["none"]).map((u) => el("li", { text: u })));
  const run = el("p", { class: "muted", text: `Model ${r.model.provider} ${r.model.model} · contract ${r.model.contract} · generated ${fmtDate(r.generated_at)} · ${r.usage.attempts} model call(s), cost $${Number(r.usage.cost_usd).toFixed(4)}` });
  box.replaceChildren(head, source, table, ...details, el("h3", { text: "Unresolved and missing information" }), unresolved, run);
}

async function downloadMarkdown(id, r) {
  const text = await api(`/research-jobs/${id}/report?format=markdown`);
  const link = el("a", { href: URL.createObjectURL(new Blob([text], { type: "text/markdown" })), download: `finsight-${r.company}-${r.period}.md` });
  document.body.append(link); link.click(); link.remove();
}

// ------------------------------------------------------------------ company trends
$("#trend-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const ticker = event.target.ticker.value.trim().toUpperCase();
  const box = $("#trends");
  box.replaceChildren(el("p", { class: "muted", text: `Loading ${ticker} from SEC XBRL…` }));
  try {
    const t = await api(`/companies/${encodeURIComponent(ticker)}/trends`);
    const titles = { revenue: "Revenue", operating_income: "Operating income", net_income: "Net income", capex: "Capital expenditure" };
    const charts = Object.entries(t.series).map(([key, s]) => chartCard(titles[key], s));
    const margin = marginSeries(t.series.revenue, t.series.operating_income);
    if (margin.points.length) charts.push(chartCard("Operating margin (derived)", margin, true));
    box.replaceChildren(
      el("div", { class: "card" }, el("h2", { text: `${t.entity || ticker} (${t.ticker})` }),
        el("p", { class: "muted", text: `${t.basis}. SEC companyfacts snapshot ${fmtDate(t.snapshot_retrieved_at)}. Orange points were later restated.` })),
      el("div", { class: "charts" }, ...charts), tableCard(t.series));
  } catch (e) { box.replaceChildren(el("p", { class: "error", text: e.message })); }
});

function marginSeries(rev, op) {
  const byEnd = Object.fromEntries((rev.points || []).map((p) => [p.period_end, p.value]));
  return { concept: "OperatingIncomeLoss / revenue", unit: "ratio",
           points: (op.points || []).filter((p) => byEnd[p.period_end]).map((p) => ({ period_end: p.period_end, value: p.value / byEnd[p.period_end], restated_values: [] })) };
}

function chartCard(title, series, ratio = false) {
  const pts = series.points || [];
  const card = el("div", { class: "card chart" }, el("h2", { text: title }), el("p", { class: "muted", text: series.concept ? `XBRL ${series.concept}` : "No XBRL concept found" }));
  if (!pts.length) return card;
  const W = 360, H = 190, L = 54, R = 10, T = 12, B = 26;
  const vals = pts.map((p) => p.value);
  let lo = Math.min(0, ...vals), hi = Math.max(...vals);
  if (hi === lo) hi = lo + 1;
  const x = (i) => L + (pts.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (pts.length - 1));
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `${title} by fiscal year`);
  const add = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text) n.textContent = text; svg.append(n); return n; };
  add("line", { x1: L, y1: y(0), x2: W - R, y2: y(0), class: "axis" });
  for (const v of [lo, hi]) add("text", { x: 4, y: y(v) + 4 }, ratio ? `${(v * 100).toFixed(0)}%` : money(v));
  add("polyline", { points: pts.map((p, i) => `${x(i)},${y(p.value)}`).join(" "), class: "line" });
  pts.forEach((p, i) => {
    const dot = add("circle", { cx: x(i), cy: y(p.value), r: 4, class: `dot${(p.restated_values || []).length ? " restated" : ""}` });
    const tip = document.createElementNS(ns, "title");
    tip.textContent = `${p.period_end}: ${ratio ? (p.value * 100).toFixed(1) + "%" : money(p.value)}${p.accession ? ` · ${p.form} ${p.accession}` : ""}${(p.restated_values || []).length ? ` · later reported as ${p.restated_values.map(money).join(", ")}` : ""}`;
    dot.append(tip);
    if (pts.length <= 10) add("text", { x: x(i) - 12, y: H - 8 }, String(p.period_end).slice(0, 4));
  });
  card.append(svg);
  return card;
}

function tableCard(series) {
  const rows = [];
  for (const [metric, s] of Object.entries(series)) for (const p of s.points || []) {
    rows.push(el("tr", {}, el("td", { text: metric.replaceAll("_", " ") }), el("td", { text: p.period_end }), el("td", { class: "num", text: money(p.value) }),
      el("td", { text: `${p.form} ${p.accession}` }), el("td", { text: p.filed }), el("td", { text: (p.restated_values || []).map(money).join(", ") || "—" })));
  }
  return el("div", { class: "card" }, el("h2", { text: "As-reported values" }),
    el("table", { class: "table" }, el("thead", {}, el("tr", {}, ...["Metric", "Period end", "Value (USD)", "Filing", "Filed", "Later reported as"].map((t) => el("th", { text: t })))), el("tbody", {}, ...rows)));
}

// ------------------------------------------------------------------ watchlist and filings
$("#watch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  try {
    await api("/watchlist", { method: "POST", body: JSON.stringify({ ticker: form.ticker.value.trim(), auto_analyze: form.auto.checked }) });
    form.reset(); loadWatchlist();
  } catch (e) { alert(e.message); }
});
$("#show-baseline").addEventListener("change", loadFilings);

async function loadWatchlist() {
  const { items } = await api("/watchlist");
  $("#watch-table tbody").replaceChildren(...items.map((w) => el("tr", {},
    el("td", { text: w.ticker + (w.auto_analyze ? " · auto" : "") }), el("td", { text: w.name || "—" }),
    el("td", { text: w.last_error ? `error: ${w.last_error}` : fmtDate(w.last_checked_at), class: w.last_error ? "error" : "" }),
    el("td", { class: "num", text: w.new_filings }),
    el("td", {}, el("button", { class: "ghost", text: "Remove", onclick: async () => { await api(`/watchlist/${w.ticker}`, { method: "DELETE" }); loadWatchlist(); loadFilings(); } })))));
  loadFilings();
}

async function loadFilings() {
  const { items } = await api(`/filings?include_baseline=${$("#show-baseline").checked}`);
  const model = () => { const v = $("#model-select").value; return v && v.startsWith("qwen|") ? v.split("|")[1] : null; };
  $("#filings-table tbody").replaceChildren(...(items.length ? items.map((f) => el("tr", {},
    el("td", { text: f.filing_date }), el("td", { text: f.ticker }), el("td", { text: f.form + (f.baseline ? "" : " · new") }),
    el("td", { text: f.report_date || "—" }), el("td", {}, el("a", { href: f.url, target: "_blank", rel: "noopener noreferrer", text: "SEC" })),
    el("td", {}, f.form === "10-K" ? el("button", { text: (f.job_ids || []).length ? "Analyzed" : "Analyze", onclick: async (ev) => {
      const m = model();
      if (!m) { alert("Choose a local model on the Research tab first."); return; }
      ev.target.disabled = true;
      try { const job = await api(`/filings/${encodeURIComponent(f.accession)}/analyze`, { method: "POST", body: JSON.stringify({ model: m }) }); watchJob(job.id); }
      catch (e) { alert(e.message); ev.target.disabled = false; }
    } }) : null))) : [el("tr", {}, el("td", { colspan: 6, class: "muted", text: "No new filings yet. The watcher records filings that appear after a ticker is added." }))]));
}

// ------------------------------------------------------------------ jobs
async function loadJobs(more = false) {
  const query = more && state.jobsBefore ? `?before=${encodeURIComponent(state.jobsBefore)}` : "";
  const { items, next_before } = await api(`/research-jobs${query}`);
  state.jobsBefore = next_before;
  const rows = items.map((j) => el("tr", { class: "click", onclick: () => watchJob(j.id) },
    el("td", { text: fmtDate(j.created_at) }), el("td", { text: j.company || "—" }), el("td", { text: j.period || "—" }),
    el("td", { text: (j.metrics || []).join(", ") || "—" }), el("td", { text: j.model }), el("td", {}, badge(j.status))));
  const body = $("#jobs-table tbody");
  if (more) body.append(...rows); else body.replaceChildren(...rows);
  $("#jobs-more").hidden = !next_before;
}
$("#jobs-more").addEventListener("click", () => loadJobs(true));

// ------------------------------------------------------------------ tabs and start
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (name === "jobs") loadJobs();
  if (name === "watchlist") loadWatchlist();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

getKey().then(() => { refreshStatus(); loadModels(); setInterval(refreshStatus, 15000); });
