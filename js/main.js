/* EO-Jev project page: the results explorer, the three charts with tooltips, the example player, tabs,
   section tracking and the BibTeX copy button. All numbers come from js/data.js (scripts/make_project_page.py). */
(function () {
  "use strict";

  const D = window.EOJEV_DATA;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const signed = (v, d = 1) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(d);
  const fmt1 = (v) => (v == null ? "—" : v.toFixed(1));
  const times = (v) => `${v >= 20 ? v.toFixed(0) : v.toFixed(1)}×`;
  const GROUP_STYLE = {
    "Scene & Land Cover": "g-blue", "Objects & Attributes": "g-blue", "Counting": "g-blue",
    "Change & Temporal": "g-violet", "Multi-Sensor": "g-violet",
    "Disaster & Damage": "g-peach", "Complex Reasoning": "g-peach", "Geo-Localisation": "g-peach",
    "Honesty & Hallucination": "g-rose", "General (non-EO)": "g-grey",
  };
  // names in the calibration and speed charts, which do not use strict LHRS accuracy or the abstention metric
  const chartName = (r) => (r.domain === "G" ? "Decision Index (general)" : r.name.replace(" (strict)", "").replace(" (abstain)", ""));
  const letter = (i) => {
    const L = (k) => String.fromCharCode(65 + k);
    return i < 26 ? L(i) : L(Math.floor(i / 26) - 1) + L(i % 26);
  };

  /* Header, section tracking, tabs ---------------------------------------- */
  function header() {
    const h = $("[data-header]");
    const onScroll = () => h.classList.toggle("is-scrolled", window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    if (!("IntersectionObserver" in window)) return;
    const sections = $$(".side-nav a").map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        $$(".side-nav a, .site-nav a").forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + e.target.id));
      }
    }, { rootMargin: "-40% 0px -55% 0px" });
    sections.forEach((s) => io.observe(s));
  }

  function tablists() {
    $$('[role="tablist"]').forEach((list) => {
      const tabs = $$('[role="tab"]', list);
      const select = (t, focus) => {
        tabs.forEach((x) => {
          const on = x === t;
          x.setAttribute("aria-selected", String(on));
          x.tabIndex = on ? 0 : -1;
          document.getElementById(x.getAttribute("aria-controls")).hidden = !on;
        });
        if (focus) t.focus();
      };
      tabs.forEach((t) => t.addEventListener("click", () => select(t)));
      list.addEventListener("keydown", (e) => {
        const i = tabs.indexOf(document.activeElement);
        if (i < 0 || (e.key !== "ArrowRight" && e.key !== "ArrowLeft")) return;
        e.preventDefault();
        select(tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length], true);
      });
      select(tabs.find((t) => t.getAttribute("aria-selected") === "true") || tabs[0]);
    });
  }

  function findings() {
    $$(".finding .more-btn").forEach((b) => b.addEventListener("click", () => {
      const open = b.getAttribute("aria-expanded") !== "true";
      b.setAttribute("aria-expanded", String(open));
      b.textContent = open ? "Less" : "Details";
      b.nextElementSibling.hidden = !open;
    }));
  }

  function stats() {
    const s = D.summary;
    const set = (k, v) => { const el = $(`[data-stat="${k}"]`); if (el) el.textContent = v; };
    set("up", `${s.all.up}/${s.all.n}`);
    set("spec", `${s.all.ge_spec}/${s.all.with_spec}`);
    set("ood", signed(s.out.mean_gain));
    set("all-gain", signed(s.all.mean_gain));
    set("in-gain", signed(s.in.mean_gain));
    set("out-gain", signed(s.out.mean_gain));
    set("speed", times(s.speed_median));
  }

  /* Results explorer ------------------------------------------------------ */
  function heat(v, vals) {
    const hi = Math.max(...vals), lo = Math.min(Math.min(...vals), hi - 10);  // at least a 10-point range
    return Math.round(((v - lo) / (hi - lo)) * 5);
  }

  function facts(r) {
    const f = [["Domain", r.domain === "I" ? "In-domain: other images of this dataset were used to adapt EO-Jev"
      : r.domain === "O" ? "Out-of-domain: held out from adaptation (no image of this dataset was used to adapt EO-Jev)" : "General (non-EO) decision skills, a retention check"]];
    if (r.spec != null) f.push(["Specialist level", `EO-Jev scores <b>${r.pct_label}</b> of ${esc(r.spec_name)}'s ${r.spec.toFixed(1)}`]);
    if (r.ece_eo != null) {
      f.push(["Calibration error (ECE)", `d3-flash ${r.ece_d3.toFixed(3)} → EO-Jev <b>${r.ece_eo.toFixed(3)}</b> `
        + `(${r.ece_sig ? "significant" : "no significant change"})${r.ece_spec != null ? `; ${esc(r.spec_name)} ${r.ece_spec.toFixed(3)}` : ""}`]);
    }
    if (r.t_ratio != null) {
      f.push(["Inference cost", `EO-Jev has <b>${times(r.t_ratio)}</b> the throughput of ${esc(r.spec_name)} at each model's `
        + `best batch size${r.t1_ratio != null ? `; one question per call, ${esc(r.spec_name)} takes <b>${times(r.t1_ratio)}</b> `
        + "EO-Jev's time per question" : ""} (${r.t_mode === "W" ? "generated answers" : "option probabilities"})`]);
    }
    return `<div class="facts">${f.map(([k, v]) => `<div><span class="k">${k}</span>${v}</div>`).join("")}</div>`;
  }

  function results() {
    D.rows.forEach((r, i) => { r._i = i; });
    const groups = [...new Set(D.rows.map((r) => r.group))];
    const state = { group: "all", domain: "all", sort: "order", asc: false, open: new Set() };
    const inDomain = (r) => state.domain === "all" || r.domain === state.domain;
    const visible = () => D.rows.filter((r) => (state.group === "all" || r.group === state.group) && inDomain(r));

    const renderCats = () => {
      const n = (g) => D.rows.filter((r) => g === "all" || r.group === g).length;
      $("#cat-filter").innerHTML = [["all", "All capabilities"], ...groups.map((g) => [g, g])].map(([g, label]) => {
        const dot = g === "all" ? "" : `<span class="dot ${GROUP_STYLE[g] || "g-grey"}"></span>`;
        return `<button type="button" data-group="${esc(g)}" aria-pressed="${g === state.group}">${dot}${esc(label)}<span class="count">${n(g)}</span>${g === "all" ? '<sup class="fn-ref" title="30 EO settings plus the general Decision Index; see note 1 under the hero numbers">1</sup>' : ""}</button>`;
      }).join("");
    };

    // a domain button shows only when the chosen capability has settings of that domain; otherwise back to "All"
    const domainButtons = $$("#domain-filter button");
    const renderDomains = () => {
      const inGroup = D.rows.filter((r) => state.group === "all" || r.group === state.group);
      const has = (d) => inGroup.some((r) => r.domain === d);
      if (state.domain !== "all" && !has(state.domain)) state.domain = "all";
      domainButtons.forEach((b) => {
        const d = b.dataset.domain;
        b.hidden = d !== "all" && !has(d);
        b.setAttribute("aria-pressed", String(d === state.domain));
      });
    };

    const renderTiles = (rows) => {
      const tile = (v, label, cls = "") => `<div class="tile"><b class="${cls}">${v}</b><span>${label}</span></div>`;
      const eo = rows.filter((r) => r.domain !== "G");
      if (!eo.length) {
        const r = rows[0];
        $("#result-tiles").innerHTML = tile(`${r.d3.toFixed(1)} → ${r.eo.toFixed(1)}`, "Decision Index, d3-flash → EO-Jev")
          + tile(signed(r.d_d3), "points of general decision skill", "down")
          + tile("1", "retention check, not an EO setting") + tile("—", "no specialist reference");
        return;
      }
      const mean = eo.reduce((a, r) => a + r.d_d3, 0) / eo.length;
      const up = eo.filter((r) => r.sig && r.d_d3 > 0).length;
      const spec = eo.filter((r) => r.spec != null);
      const ge = spec.filter((r) => r.d_spec >= 0).length;
      $("#result-tiles").innerHTML = tile(eo.length, eo.length === 1 ? "EO setting shown" : "EO settings shown")
        + tile(signed(mean), "points: mean gain of EO-Jev over d3-flash", mean >= 0 ? "up" : "down")
        + tile(`${up}/${eo.length}`, "significant gains over d3-flash", up ? "up" : "")
        + tile(spec.length ? `${ge}/${spec.length}` : "—", "at or above the best open EO specialist", ge ? "up" : "");
    };

    const rowHtml = (r) => {
      const vals = ["d3", "qwen", "spec", "eo"].map((k) => r[k]).filter((v) => v != null);
      const best = Math.max(...vals);
      const cell = (k, sub) => {
        const v = r[k];
        if (v == null) return `<td class="${k}"><span class="cell na">—</span></td>`;
        return `<td class="${k}"><span class="cell${v === best ? " best" : ""}" style="background: var(--h${heat(v, vals)})">${fmt1(v)}${sub ? `<small>${esc(sub)}</small>` : ""}</span></td>`;
      };
      const open = state.open.has(r._i);
      const dCls = r.sig ? (r.d_d3 > 0 ? "up" : "down") : "ns";
      const spec = r.d_spec == null ? `<td class="d ns">—</td>` : `<td class="d ${r.d_spec >= 0 ? "up" : "down"}">${signed(r.d_spec)}</td>`;
      return `<tr class="row-main" data-i="${r._i}"><td class="bench"><button type="button" aria-expanded="${open}">${esc(r.name)}</button>`
        + ` <span class="badge ${r.domain}" title="${r.domain === "I" ? "in-domain" : r.domain === "O" ? "out-of-domain" : "retention check"}">${r.domain}</span></td>`
        + cell("d3") + cell("qwen") + cell("spec", r.spec_name) + cell("eo")
        + `<td class="d sep ${dCls}">${signed(r.d_d3)}</td>${spec}</tr>`
        + (open ? `<tr class="detail"><td colspan="7">${facts(r)}</td></tr>` : "");
    };

    const renderTable = () => {
      let rows = visible();
      renderTiles(rows);
      if (state.sort !== "order") {
        const k = state.sort, dir = state.asc ? 1 : -1;
        rows = [...rows].sort((a, b) => (a[k] == null) - (b[k] == null) || dir * (a[k] - b[k]));
      }
      const headers = state.sort === "order" && state.group === "all";
      let html = "", group = null;
      for (const r of rows) {
        if (headers && r.group !== group) {
          group = r.group;
          html += `<tr class="grp ${GROUP_STYLE[group] || "g-grey"}"><th colspan="7" scope="rowgroup"><span>${esc(group)}</span></th></tr>`;
        }
        html += rowHtml(r);
      }
      $("#results-table tbody").innerHTML = html;
      $$("#results-table thead button").forEach((b) => {
        const on = b.dataset.sort === state.sort && state.sort !== "order";
        b.classList.toggle("sorted", on);
        b.classList.toggle("asc", on && state.asc);
        b.closest("th").setAttribute("aria-sort", on ? (state.asc ? "ascending" : "descending") : "none");
      });
    };

    $("#cat-filter").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-group]");
      if (!b) return;
      state.group = b.dataset.group;
      renderCats();
      renderDomains();
      renderTable();
    });
    domainButtons.forEach((b) => b.addEventListener("click", () => {
      state.domain = b.dataset.domain;
      renderDomains();
      renderTable();
    }));
    $("#results-table thead").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-sort]");
      if (!b) return;
      const k = b.dataset.sort;
      if (k === "order") { state.sort = "order"; state.asc = false; } else if (state.sort === k) state.asc = !state.asc;
      else { state.sort = k; state.asc = false; }
      renderTable();
    });
    $("#results-table tbody").addEventListener("click", (e) => {
      const tr = e.target.closest("tr.row-main");
      if (!tr) return;
      const i = Number(tr.dataset.i);
      if (state.open.has(i)) state.open.delete(i); else state.open.add(i);
      renderTable();
      const btn = $(`tr.row-main[data-i="${i}"] .bench button`);
      if (btn && e.target.closest("button")) btn.focus();
    });
    renderCats();
    renderDomains();
    renderTable();
  }

  /* Charts --------------------------------------------------------------- */
  function axis(ticks, x, title) {
    return `<div class="axis"><span></span><div class="ticks">${ticks.map(([v, t]) => `<span style="left:${x(v)}%">${t}</span>`).join("")}</div><span></span></div>`
      + `<div class="axis-title">${title}</div>`;
  }
  const grid = (ticks, x, base) => ticks.map(([v]) => `<span class="gridline${v === base ? " base" : ""}" style="left:${x(v)}%"></span>`).join("");
  const tiers = (rows, defs) => defs.map((d) => ({ ...d, rows: rows.filter((r) => d.test(r)).sort(d.sort) })).filter((t) => t.rows.length);
  const tipRow = (name, tip, track, tail, style = "") =>
    `<div class="row" tabindex="0" data-tip="${esc(tip)}"${style}><span class="lbl">${esc(name)}</span><div class="track">${track}</div><span class="tail">${tail}</span></div>`;

  function maturity() {
    const lo = 75, hi = 165;
    const x = (v) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * 100;
    const ticks = [[75, "75%"], [100, "100%"], [125, "125%"], [150, "150%"]];
    const desc = (a, b) => b.pct_spec - a.pct_spec;
    const ts = tiers(D.rows.filter((r) => r.pct_spec != null), [
      { cls: "t1", label: "Exceeds specialist level (above 110%)", test: (r) => r.pct_spec > 110, sort: desc },
      { cls: "t2", label: "At specialist level (100–110%)", test: (r) => r.pct_spec >= 100 && r.pct_spec <= 110, sort: desc },
      { cls: "t3", label: "Approaching (85–100%)", test: (r) => r.pct_spec < 100, sort: desc },
    ]);
    let html = "";
    for (const t of ts) {
      html += `<div class="tier ${t.cls}"><div class="tier-label">${t.label}</div>`;
      for (const r of t.rows) {
        const p = r.pct_spec, b = x(100), e = x(p);
        const bar = p > hi
          ? `<span class="bar" style="left:${b}%;width:${e - b}%"><span class="brk"></span></span><span class="val inside" style="left:calc(${e}% - 24px)">${r.pct_label}</span>`
          : p >= 100
            ? `<span class="bar" style="left:${b}%;width:${e - b}%"></span><span class="val" style="left:calc(${e}% + 6px)">${r.pct_label}</span>`
            : `<span class="bar" style="left:${e}%;width:${b - e}%"></span><span class="val neg" style="left:calc(${e}% - 6px);--base:${b}%">${r.pct_label}</span>`;
        const tip = `<b>${esc(r.name)}</b><span class="tip-sub">EO-Jev ${r.eo.toFixed(1)} vs ${esc(r.spec_name)} ${r.spec.toFixed(1)}: ${r.pct_label} of specialist level</span>`;
        html += tipRow(r.name, tip, grid(ticks, x, 100) + bar, `<span class="badge ${r.domain}">${r.domain}</span>`);
      }
      html += "</div>";
    }
    $("#maturity-chart .chart-body").innerHTML = html + axis(ticks, x, "Score as a percentage of the best open EO specialist (100% = specialist level)");
  }

  function calibration() {
    const hi = 0.27;
    const x = (v) => (Math.min(Math.max(v, 0), hi) / hi) * 100;
    const ticks = [[0, "0"], [0.05, ".05"], [0.1, ".10"], [0.15, ".15"], [0.2, ".20"], [0.25, ".25"]];
    const asc = (a, b) => a.ece_delta - b.ece_delta;
    const ts = tiers(D.rows.filter((r) => r.ece_eo != null && r.ece_d3 != null), [
      { cls: "t1", label: "Much better calibrated (ΔECE ≤ −0.05)", test: (r) => r.ece_sig && r.ece_delta <= -0.05, sort: asc },
      { cls: "t2", label: "Better calibrated", test: (r) => r.ece_sig && r.ece_delta < 0 && r.ece_delta > -0.05, sort: asc },
      { cls: "t3", label: "No significant change", test: (r) => !r.ece_sig, sort: asc },
      { cls: "t4", label: "Worse calibrated (over-confident)", test: (r) => r.ece_sig && r.ece_delta > 0, sort: (a, b) => -asc(a, b) },
    ]);
    let html = "";
    for (const t of ts) {
      html += `<div class="tier ${t.cls}"><div class="tier-label">${t.label}</div>`;
      for (const r of t.rows) {
        const a = x(r.ece_d3), b = x(r.ece_eo);
        const head = Math.abs(r.ece_eo - r.ece_d3) < 0.012 ? ""
          : b < a ? `<span class="db-head left" style="left:calc(${b}% + 6px)"></span>` : `<span class="db-head right" style="left:calc(${b}% - 15px)"></span>`;
        const track = grid(ticks, x, -1) + `<span class="db-line" style="left:${Math.min(a, b)}%;width:${Math.abs(a - b)}%"></span>${head}`
          + `<span class="db-dot open" style="left:${a}%"></span><span class="db-dot fill" style="left:${b}%"></span>`;
        const tip = `<b>${esc(chartName(r))}</b><span class="tip-sub">ECE: d3-flash ${r.ece_d3.toFixed(3)} → EO-Jev ${r.ece_eo.toFixed(3)}`
          + `${r.ece_spec != null ? `; ${esc(r.spec_name)} ${r.ece_spec.toFixed(3)}` : ""}</span>`;
        html += tipRow(chartName(r), tip, track,
          `<span>${signed(r.ece_delta, 3)}${r.ece_sig ? "*" : ""}</span><span class="badge ${r.domain}">${r.domain}</span>`, ' style="--tail:5.6rem"');
      }
      html += "</div>";
    }
    $("#calibration-chart .chart-body").innerHTML = html + axis(ticks, x, "Expected calibration error (15 bins; lower is better)")
      + `<div class="chart-legend"><span><i class="open"></i>d3-flash (zero-shot)</span><span><i class="fill"></i>EO-Jev (ours)</span></div>`;
  }

  // the paper's Figure 5: (a) one question per call, (b) each model at its best batch size
  const SPEED = {
    a: { key: "t1_ratio", hi: 150, top: "EO-Jev ≥ 10× faster",
         sub: "The best specialist's median latency per question as a multiple of EO-Jev's, one question per call, log scale (paper, Figure 5a).",
         axis: "Specialist's median latency per question as a multiple of EO-Jev's (one question per call; log scale)",
         tip: (r, v, mode) => `With one question per call, ${esc(r.spec_name)} takes ${times(v)} EO-Jev's time per question (${mode})` },
    b: { key: "t_ratio", hi: 60, top: "EO-Jev ≥ 10× the throughput",
         sub: "EO-Jev's throughput as a multiple of the best specialist's, each at its best batch size, log scale (paper, Figure 5b).",
         axis: "EO-Jev throughput as a multiple of the best specialist's (best batch size for each; log scale)",
         tip: (r, v, mode) => `EO-Jev answers ${times(v)} as many questions per second as ${esc(r.spec_name)} (${mode})` },
  };

  function speed(m = "b") {
    const S = SPEED[m], val = (r) => r[S.key];
    const lo = Math.log(0.4), hi = Math.log(S.hi);
    const x = (v) => ((Math.log(v) - lo) / (hi - lo)) * 100;
    const ticks = [[0.5, "0.5×"], [1, "1×"], [2, "2×"], [5, "5×"], [10, "10×"], [20, "20×"], [40, "40×"]]
      .concat(S.hi > 100 ? [[100, "100×"]] : []);
    const desc = (a, b) => val(b) - val(a);
    const ts = tiers(D.rows.filter((r) => val(r) != null), [
      { cls: "t1", label: S.top, test: (r) => val(r) >= 10, sort: desc },
      { cls: "t2", label: "3–10×", test: (r) => val(r) >= 3 && val(r) < 10, sort: desc },
      { cls: "t3", label: "1–3×", test: (r) => val(r) >= 1 && val(r) < 3, sort: desc },
      { cls: "t4", label: "Specialist faster", test: (r) => val(r) < 1, sort: desc },
    ]);
    let html = "";
    for (const t of ts) {
      html += `<div class="tier ${t.cls}"><div class="tier-label">${t.label}</div>`;
      for (const r of t.rows) {
        const v = val(r), b = x(1), e = x(v);
        const bar = v >= 1
          ? `<span class="bar" style="left:${b}%;width:${e - b}%"></span><span class="val" style="left:calc(${e}% + 6px)">${times(v)}</span>`
          : `<span class="bar" style="left:${e}%;width:${b - e}%"></span><span class="val neg" style="left:calc(${e}% - 6px);--base:${b}%">${times(v)}</span>`;
        const mode = r.t_mode === "W" ? "generated answers" : "option probabilities, one pass";
        const tip = `<b>${esc(chartName(r))}</b><span class="tip-sub">${S.tip(r, v, mode)}</span>`;
        html += tipRow(chartName(r), tip, grid(ticks, x, 1) + bar, `<span class="badge ${r.t_mode}">${r.t_mode}</span>`);
      }
      html += "</div>";
    }
    $("#time-chart .chart-body").innerHTML = html + axis(ticks, x, S.axis);
    $("#time-sub").textContent = S.sub;
    $$("#time-mode button").forEach((btn) => btn.setAttribute("aria-pressed", String(btn.dataset.mode === m)));
  }

  function tooltips(panel) {
    const tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    tip.setAttribute("role", "status");
    panel.appendChild(tip);
    let current = null;
    const show = (row) => {
      if (current) current.classList.remove("is-tip");
      current = row;
      row.classList.add("is-tip");
      tip.innerHTML = row.dataset.tip;
      tip.hidden = false;
      const pr = panel.getBoundingClientRect(), rr = row.getBoundingClientRect(), track = $(".track", row).getBoundingClientRect();
      let top = rr.top - pr.top - tip.offsetHeight - 6;
      if (top < 4) top = rr.bottom - pr.top + 6;
      const left = Math.min(Math.max(track.left - pr.left + track.width / 2 - tip.offsetWidth / 2, 8), pr.width - tip.offsetWidth - 8);
      tip.style.top = `${top}px`;
      tip.style.left = `${left}px`;
    };
    const hide = () => {
      if (current) current.classList.remove("is-tip");
      current = null;
      tip.hidden = true;
    };
    panel.addEventListener("pointerover", (e) => { const row = e.target.closest(".row"); if (row) show(row); });
    panel.addEventListener("pointerleave", hide);
    panel.addEventListener("focusin", (e) => { const row = e.target.closest(".row"); if (row) show(row); });
    panel.addEventListener("focusout", hide);
    panel.addEventListener("click", (e) => { const row = e.target.closest(".row"); if (row) show(row); else hide(); });
  }

  /* Examples player -------------------------------------------------------- */
  const SHOW_ORDER = ["d3", "qwen", "spec", "eo"];  // EO-Jev is revealed last
  const CARD_MS = 1800, SLIDE_MS = 9000;
  const STEPS = [0, 600, 1400, 2000, 2600, 3300, 4300];  // .on1 .. .on7, counted from the end of the chapter card
  const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // two probabilities written with enough decimals to read as different
  function pair(a, b) {
    for (const d of [2, 3, 4]) if (a.toFixed(d) !== b.toFixed(d)) return [a.toFixed(d), b.toFixed(d)];
    return [a.toFixed(4), b.toFixed(4)];
  }

  function verdict(key, p) {
    const eo = p.models.find((m) => m.key === "eo");
    const others = p.models.filter((m) => m.key !== "eo");
    const withP = others.filter((m) => m.p != null);
    const names = (ms) => ms.map((m) => m.label).join(", ").replace(/, ([^,]*)$/, " and $1");
    const spec = others.find((m) => m.written);
    const note = spec ? ` ${spec.label} gave a written answer, which has no probability.` : "";
    if (key === "only_ours") return { cls: "good", text: `Only EO-Jev picks the right answer (p = ${eo.p.toFixed(2)}); ${names(others)} are all wrong.` };
    if (key === "surest_right") {
      const top = withP.reduce((a, b) => (b.p > a.p ? b : a));
      const [x, y] = pair(eo.p, top.p);
      const right = others.filter((m) => m.correct), wrong = others.filter((m) => !m.correct);
      return { cls: "good", text: `EO-Jev is right with p = ${x}, ${(eo.p - top.p).toFixed(2)} above the next most confident model `
        + `(${top.label}, ${y}). ` + (wrong.length
        ? `${names(right)} ${right.length > 1 ? "are" : "is"} also right; ${names(wrong)} ${wrong.length > 1 ? "are" : "is"} not.`
        : "The other three are also right.") + note };
    }
    const low = withP.reduce((a, b) => (b.p < a.p ? b : a));
    const [x, y] = pair(eo.p, low.p);
    if (key === "least_sure_wrong") {
      const right = others.filter((m) => m.correct);
      return { cls: "honest", text: `EO-Jev is wrong, but unsure: p = ${x}, below 0.5 and under every other model (at least ${y}), `
        + `so a 0.5 threshold would send it to an analyst. ${names(right)} ${right.length > 1 ? "get" : "gets"} it right.${note}` };
    }
    return { cls: "honest", text: `All four are wrong, and EO-Jev is the least confident: p = ${x}, against at least ${y} for the others.${note}` };
  }

  // long option lists: the gold option, every model's choice, then the first options up to `max`
  function optionSubset(p, max = 12) {
    const n = p.options.length;
    if (n <= max) return { idx: [...Array(n).keys()], more: 0 };
    const keep = new Set([p.gold, ...p.models.map((m) => m.pred)].map((o) => p.options.indexOf(o)).filter((i) => i >= 0));
    for (let i = 0; keep.size < max && i < n; i++) keep.add(i);
    const idx = [...keep].sort((a, b) => a - b);
    return { idx, more: n - idx.length };
  }

  function slideHtml(s) {
    const { c, ci, p } = s;
    const n = p.frames.length;
    const media = `<div class="s-media r1${n === 2 ? " multi" : n > 2 ? " many" : ""}">`
      + p.frames.map((f) => `<figure><img src="${f.src}" alt="${esc(p.benchmark)} input image${f.label ? " " + esc(f.label) : ""}" width="${f.w}" height="${f.h}">`
        + (f.label ? `<figcaption>${esc(f.label)}</figcaption>` : "") + `</figure>`).join("") + `</div>`;
    const { idx, more } = optionSubset(p);
    const opts = idx.map((i) => `<span class="s-opt${p.options[i] === p.gold ? " gold" : ""}"><b>${letter(i)}</b>${esc(p.options[i])}</span>`).join("")
      + (more ? `<span class="s-more">+ ${more} more options</span>` : "");
    const coded = (o) => { const i = p.options.indexOf(o); return (i >= 0 ? `<b>${letter(i)}</b>` : "") + esc(o); };
    const rows = SHOW_ORDER.map((k) => p.models.find((m) => m.key === k)).filter(Boolean).map((m, j) => {
      const label = m.key === "eo" ? "EO-Jev (ours)" : m.label;
      const conf = m.written ? `<span class="pv written">written answer</span>`
        : `<span class="pbar-track" style="--p:${m.p}"><i></i></span><span class="pv">${m.p.toFixed(2)}</span>`;
      return `<li class="${m.key} r${3 + j}"><span class="who ${m.key}" title="${esc(label)}">${esc(label)}</span><span class="pick">${coded(m.pred)}</span>${conf}`
        + `<span class="mark ${m.correct ? "ok" : "bad"}" aria-label="${m.correct ? "correct" : "wrong"}">${m.correct ? "✓" : "✗"}</span></li>`;
    }).join("");
    const v = verdict(c.key, p);
    const card = s.first ? `<div class="chapter-card"><span class="n">Chapter ${ci + 1} of ${D.video.length}</span><h3>${esc(c.title)}</h3><p>${esc(c.definition)}.</p></div>` : "";
    return `${card}<div class="slide">${media}<div class="s-body">`
      + `<div class="s-head r1"><span class="s-chip">${ci + 1} · ${esc(c.title)}</span><span class="s-grp">${esc(p.group_title)}</span>`
      + `<h3>${esc(p.benchmark)} <span class="badge ${p.domain}" title="${p.domain === "I" ? "in-domain" : "out-of-domain"} for EO-Jev">${p.domain}</span></h3></div>`
      + `<p class="s-q r1">${esc(p.question)}</p><div class="s-opts r2">${opts}</div><ul class="s-models">${rows}</ul>`
      + `<p class="s-verdict ${v.cls} r7">${esc(v.text)}</p></div></div>`;
  }

  function player() {
    if (!D.video || !D.video.length) return;
    const slides = [];
    D.video.forEach((c, ci) => c.panels.forEach((p, pi) => slides.push({ c, ci, p, first: pi === 0 })));
    const screen = $("#screen"), playBtn = $("#p-play");
    const st = { i: 0, t: 0, playing: false, held: false, started: false, last: 0 };
    const dur = (s) => (s.first ? CARD_MS : 0) + SLIDE_MS;
    const revealed = (s) => (s.first ? CARD_MS : 0) + STEPS[STEPS.length - 1];

    $("#timeline").innerHTML = D.video.map((c, ci) => `<div class="tl-ch" style="flex:${c.panels.length}">`
      + slides.map((s, i) => [s, i]).filter(([s]) => s.ci === ci)
        .map(([s, i]) => `<button type="button" class="tl-tick" data-i="${i}" aria-label="Example ${i + 1}: ${esc(s.p.benchmark)}"><span><i></i></span></button>`).join("")
      + `</div>`).join("");
    $("#chapters").innerHTML = D.video.map((c, ci) => `<button type="button" data-ch="${ci}"><span class="k">${ci + 1}</span>${esc(c.title)}</button>`).join("");
    const fills = $$("#timeline .tl-tick i");

    const apply = (t) => {
      const s = slides[st.i], off = s.first ? CARD_MS : 0;
      const slide = $(".slide", screen), card = $(".chapter-card", screen);
      if (card) card.classList.toggle("off", t >= off);
      STEPS.forEach((o, k) => slide.classList.toggle(`on${k + 1}`, t >= off + o));
      fills.forEach((el, i) => { el.style.width = i < st.i ? "100%" : i > st.i ? "0%" : `${Math.min(100, (t / dur(s)) * 100)}%`; });
    };
    const show = (i) => {
      st.i = (i + slides.length) % slides.length;
      st.t = 0;
      screen.innerHTML = slideHtml(slides[st.i]);
      apply(st.playing ? 0 : Infinity);  // paused: the whole example at once
      $("#p-count").textContent = `${st.i + 1} / ${slides.length}`;
      $$("#chapters button").forEach((b) => b.setAttribute("aria-current", String(Number(b.dataset.ch) === slides[st.i].ci)));
    };
    const frame = (now) => {
      if (!st.playing) return;
      st.t += Math.min(now - st.last, 100);
      st.last = now;
      if (st.t >= dur(slides[st.i])) show(st.i + 1); else apply(st.t);
      requestAnimationFrame(frame);
    };
    const setPlaying = (on) => {
      if (on === st.playing) return;
      st.playing = on;
      playBtn.classList.toggle("playing", on);
      playBtn.setAttribute("aria-label", on ? "Pause" : "Play");
      if (on) {
        // the first start plays the example from its beginning; a resume keeps what is already on screen
        st.t = st.started ? Math.max(st.t, revealed(slides[st.i])) : 0;
        st.started = true;
        st.last = performance.now();
        requestAnimationFrame(frame);
      } else {
        apply(Infinity);
      }
    };

    playBtn.addEventListener("click", () => { st.held = st.playing; setPlaying(!st.playing); });
    $("#p-prev").addEventListener("click", () => show(st.i - 1));
    $("#p-next").addEventListener("click", () => show(st.i + 1));
    $("#timeline").addEventListener("click", (e) => { const b = e.target.closest("[data-i]"); if (b) show(Number(b.dataset.i)); });
    $("#chapters").addEventListener("click", (e) => {
      const b = e.target.closest("[data-ch]");
      if (b) show(slides.findIndex((s) => s.ci === Number(b.dataset.ch)));
    });
    $("#player").addEventListener("keydown", (e) => {
      if (e.target.closest("button") && e.key === " ") return;
      if (e.key === " " || e.key === "k") { e.preventDefault(); st.held = st.playing; setPlaying(!st.playing); }
      if (e.key === "ArrowRight") { e.preventDefault(); show(st.i + 1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); show(st.i - 1); }
    });
    // plays while on screen (unless the viewer paused it, or prefers reduced motion)
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        for (const e of entries) {
          if (e.isIntersecting && !st.held && !REDUCED) setPlaying(true);
          else if (!e.isIntersecting) setPlaying(false);
        }
      }, { threshold: 0.35 }).observe($("#player"));
    }
    document.addEventListener("visibilitychange", () => { if (document.hidden) setPlaying(false); });
    show(0);
  }

  /* BibTeX --------------------------------------------------------------- */
  function bibtex() {
    const btn = $("#copy-bib"), pre = $("#bibtex");
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.textContent);
        btn.textContent = "Copied";
      } catch (err) {
        const range = document.createRange();
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = "Selected: press Ctrl+C";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 2000);
    });
  }

  /* Rotating Earth behind the hero ------------------------------------------------------------------
     One full-canvas WebGL quad: each pixel inside the disc is mapped back to latitude and longitude (orthographic
     view, the centre at 24 S so the visible upper part runs from about 66 N to the tropics) and coloured from the
     NASA Blue Marble map (public domain), lit from the upper left with an atmosphere rim. One turn per two minutes,
     only while the hero is on screen; a still frame for reduced motion; nothing at all without WebGL. */
  function globe() {
    const canvas = $(".globe");
    const hero = canvas && canvas.closest(".hero");
    const gl = canvas && canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false });
    if (!gl) { if (canvas) canvas.remove(); return; }
    const VS = "attribute vec2 a; varying vec2 v; void main() { v = a; gl_Position = vec4(a, 0.0, 1.0); }";
    const FS = `precision mediump float;
varying vec2 v;
uniform sampler2D tex;
uniform float rot;
const float PI = 3.14159265;
const float TILT = -0.42;
const float HALO = 1.07;
void main() {
  vec2 p = v * HALO;
  float r = length(p);
  vec3 sky = vec3(0.56, 0.76, 1.0);
  if (r > 1.0) {
    float g = pow(smoothstep(HALO, 1.0, r), 2.0) * 0.5;
    gl_FragColor = vec4(sky * g, g);
    return;
  }
  float z = sqrt(1.0 - r * r);
  vec3 n = vec3(p, z);
  float ct = cos(TILT), st = sin(TILT);
  vec3 w = vec3(n.x, ct * n.y + st * n.z, ct * n.z - st * n.y);
  float lon = atan(w.x, w.z) + rot;
  float lat = asin(clamp(w.y, -1.0, 1.0));
  vec3 c = texture2D(tex, vec2(fract(lon / (2.0 * PI) + 0.5), 0.5 - lat / PI)).rgb;
  float d = max(dot(n, normalize(vec3(-0.5, 0.55, 0.68))), 0.0);
  vec3 col = c * (0.32 + 0.85 * d);
  col = mix(col, sky, pow(1.0 - z, 3.0) * 0.6);
  float a = smoothstep(1.0, 0.993, r);
  gl_FragColor = vec4(col * a, a);
}`;
    const shader = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
    };
    const vs = shader(gl.VERTEX_SHADER, VS), fs = shader(gl.FRAGMENT_SHADER, FS);
    const prog = vs && fs && gl.createProgram();
    if (prog) { gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog); }
    if (!prog || !gl.getProgramParameter(prog, gl.LINK_STATUS)) { canvas.remove(); return; }
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const aLoc = gl.getAttribLocation(prog, "a");
    gl.enableVertexAttribArray(aLoc);
    gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0);
    const uRot = gl.getUniformLocation(prog, "rot");
    gl.uniform1i(gl.getUniformLocation(prog, "tex"), 0);
    gl.clearColor(0, 0, 0, 0);

    // size and position: about 80% of the viewport wide (130% on phones), its top just above the stats row
    let px = 0;
    const place = () => {
      const vw = document.documentElement.clientWidth;
      const d = vw < 720 ? Math.round(vw * 1.3) : Math.round(Math.min(1120, Math.max(640, vw * 0.8)));
      const stats = $("#stats");
      const top = stats ? stats.getBoundingClientRect().top - hero.getBoundingClientRect().top - d * 0.1 : hero.clientHeight / 2;
      canvas.style.setProperty("--globe", `${d}px`);
      canvas.style.top = `${Math.round(top)}px`;
      const want = Math.min(1024, Math.round(d * Math.min(window.devicePixelRatio || 1, 1.5)));
      if (want !== px) { px = want; canvas.width = canvas.height = px; gl.viewport(0, 0, px, px); }
    };
    let angle = 54 * Math.PI / 180;  // starts over the Arabian Peninsula
    const OMEGA = 2 * Math.PI / 120000;  // radians per ms: one turn per two minutes, eastward
    let ready = false;
    const draw = () => {
      if (!ready) return;
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(uRot, angle);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    let raf = 0, last = 0, onScreen = true;
    const frame = (now) => {
      raf = 0;
      if (!onScreen || document.hidden) return;
      const dt = last ? Math.min(now - last, 100) : 0;
      if (!last || dt >= 33) {  // about 30 frames a second is plenty for this speed
        angle -= dt * OMEGA;
        last = now;
        draw();
      }
      raf = requestAnimationFrame(frame);
    };
    const play = () => { if (ready && !REDUCED && !raf) { last = 0; raf = requestAnimationFrame(frame); } };
    const texture = gl.createTexture();
    const load = (cors) => {
      const img = new Image();
      if (cors) img.crossOrigin = "anonymous";
      img.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        try {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
        } catch (e) {  // a texture served from another origin needs CORS; without it, no globe
          if (!cors) load(true); else canvas.remove();
          return;
        }
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        ready = true;
        place();
        draw();
        canvas.classList.add("is-on");
        play();
      };
      img.onerror = () => { if (!cors) load(true); else canvas.remove(); };
      img.src = canvas.dataset.texture;
    };
    place();
    load(false);
    window.addEventListener("resize", () => { place(); draw(); });
    document.addEventListener("visibilitychange", play);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es) => { onScreen = es[0].isIntersecting; play(); }).observe(hero);
    }
  }

  /* Demo video: browsers can only seek in a video whose server answers byte-range requests, which some static hosts
     (and `python -m http.server`) do not. A local copy of the file (6 MB) can always be seeked, so it is fetched once
     the section comes near; until then, or if fetching fails, the video plays from its normal source. */
  function demoVideo() {
    const v = $("#demo video");
    const src = v && v.querySelector("source") && v.querySelector("source").getAttribute("src");
    if (!src || !window.fetch || !window.URL || !URL.createObjectURL) return;
    let started = false;
    const load = () => {
      if (started) return;
      started = true;
      fetch(src)
        .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(r.status))))
        .then((blob) => {
          const at = v.currentTime, playing = !v.paused;
          v.src = URL.createObjectURL(blob.type ? blob : new Blob([blob], { type: "video/mp4" }));
          v.addEventListener("loadedmetadata", () => {
            if (at) v.currentTime = at;
            if (playing) v.play().catch(() => {});
          }, { once: true });
        })
        .catch(() => {});
    };
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { o.disconnect(); load(); } }, { rootMargin: "800px" }).observe(v);
    } else {
      load();
    }
  }

  header();
  tablists();
  findings();
  bibtex();
  globe();
  demoVideo();
  if (!D) return;
  stats();
  results();
  maturity();
  calibration();
  speed();
  $("#time-mode").addEventListener("click", (e) => { const b = e.target.closest("button[data-mode]"); if (b) speed(b.dataset.mode); });
  $$(".chart").forEach(tooltips);
  player();
})();
