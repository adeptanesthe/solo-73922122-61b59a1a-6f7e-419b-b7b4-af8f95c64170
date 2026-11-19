/* 机构简称核查前端（原生 JS）。
 *
 * 过期保护：每次提交携带本地自增 token，只有响应里的 request_id 与当前
 * token 一致才允许渲染；输入框任何变更都会立刻撤下旧判断并作废轮询，
 * 因此迟到结果（含慢核查）不可能覆盖新输入。
 */
"use strict";

const SAMPLE_MANUSCRIPT = (
  "近日，国家发展和改革委员会召开年度工作会议，部署下一阶段重点任务。" +
  "国家发展改革委在会上强调，要继续推进重大基础设施建设，并加强区域协调发展。" +
  "与此同时，国家卫生健康委员会发布了基层医疗服务能力提升方案。" +
  "国家卫健委要求各地在年内完成社区医院的标准化改造，重点补齐儿科与老年医学短板。" +
  "记者还从科学技术部获悉，新一轮重点研发计划已经启动。" +
  "科技部表示，本年度计划将向人工智能、量子信息等前沿方向倾斜。" +
  "工业和信息化部随后公布了制造业数字化转型的配套政策。" +
  "工信部提出，三年内要培育一批具备全球竞争力的智能制造示范工厂。" +
  "以上部门将在国务院的统一协调下建立月度会商机制，确保各项政策落地见效。" +
  "业内专家指出，国家发展改革委与工信部在产业项目上的分工需要进一步明确，" +
  "避免地方在申报时把两个部门的支持方向混为一谈。"
);
const SAMPLE_FULL = [
  "国家发展和改革委员会",
  "国家卫生健康委员会",
  "科学技术部",
  "工业和信息化部",
];
const SAMPLE_ABBR = ["国家发展改革委", "国家卫健委", "科技部", "工信部"];

const abbrState = {
  token: 0,          // 本地世代：提交 +1，输入变更作废当前结果
  pollTimer: null,
  lastResult: null,  // 最近一次已渲染的结果
};

const abbr$ = (sel) => document.querySelector(sel);

function abbrApi(path, options = {}) {
  return fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  }).then(async (resp) => {
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `请求失败 (${resp.status})`);
    return body;
  });
}

function readNameList(el) {
  return el.value.split("\n").map((s) => s.trim()).filter(Boolean);
}

function showAbbrMsg(text, ok = false) {
  const el = abbr$("#abbr-msg");
  el.textContent = text;
  el.className = `msg ${ok ? "ok" : ""}`;
}

function clearAbbrPoll() {
  if (abbrState.pollTimer) {
    clearTimeout(abbrState.pollTimer);
    abbrState.pollTimer = null;
  }
}

/** 撤下旧判断：任何输入变更都调用，立即清空结果并推进本地世代。 */
function invalidateAbbr(keepText = true) {
  abbrState.token += 1;
  clearAbbrPoll();
  abbrState.lastResult = null;
  abbr$("#abbr-job-status").textContent = "输入已变更，旧判断已撤下";
  abbr$("#abbr-job-status").className = "job-status stale";
  abbr$("#abbr-pairs").innerHTML = "";
  abbr$("#abbr-warnings").innerHTML = "";
  abbr$("#abbr-context").innerHTML = "";
  abbr$("#abbr-article").textContent = keepText
    ? "核查结果将在这里标出每处原文位置。"
    : "";
}

function updateLength() {
  const len = abbr$("#abbr-manuscript").value.trim().length;
  const el = abbr$("#abbr-length");
  el.textContent = `${len} 字`;
  el.className = `length-hint ${len >= 300 && len <= 1200 ? "ok" : "bad"}`;
}

function renderPairs(job) {
  const result = job.result;
  abbrState.lastResult = result;
  const wrap = abbr$("#abbr-pairs");
  wrap.innerHTML = "";

  const warnEl = abbr$("#abbr-warnings");
  warnEl.innerHTML = "";
  (result.warnings || []).forEach((w) => {
    const div = document.createElement("div");
    div.className = "abbr-warning";
    div.textContent = `⚠ ${w}`;
    warnEl.appendChild(div);
  });

  result.pairs.forEach((p) => {
    const card = document.createElement("div");
    card.className = `pair-card ${p.status}`;
    const reasons = p.reasons.length
      ? `<ul class="reason-list">${p.reasons.map((r) => `<li>${r}</li>`).join("")}</ul>`
      : `<div class="reason-ok">判断依据：语境与字形证据均达标，且无接近的替代对应。</div>`;
    card.innerHTML = `
      <div class="pair-head">
        <span class="pair-names">
          <b class="name-abbr">${p.abbreviation}</b>
          <span class="arrow">→</span>
          <b class="name-full">${p.full_name}</b>
        </span>
        <span class="status-pill ${p.status}">${p.status === "confirmed" ? "可确认" : "待核对"}</span>
      </div>
      <div class="pair-scores">
        <span>综合得分 <b>${p.score.toFixed(3)}</b></span>
        <span>语境 TF-IDF 余弦 <b>${p.context_similarity.toFixed(3)}</b></span>
        <span>字形相似度 <b>${p.glyph_similarity.toFixed(3)}</b></span>
        <span>独立出现 <b>${p.standalone_count}</b> 处（共 ${p.abbr_occurrences.length} 处）</span>
      </div>
      <div class="pair-runner">
        次优对应：${p.runner_up_full}（得分 ${p.runner_up_score.toFixed(3)}），
        差距 ${p.score_gap.toFixed(3)}
      </div>
      ${reasons}`;
    wrap.appendChild(card);
  });

  renderAnnotatedArticle(result);
  renderContextDetails(result);
}

function renderAnnotatedArticle(result) {
  const text = abbr$("#abbr-manuscript").value;
  // 按 start 排序；同一起点更长的名称优先，保证嵌套标注可正确包裹
  const spans = result.spans.slice().sort((a, b) =>
    a.start - b.start || b.end - a.end
  );
  const host = abbr$("#abbr-article");
  host.innerHTML = "";
  let cursor = 0;
  spans.forEach((sp) => {
    if (sp.start < cursor) return; // 已被更长名称覆盖
    if (sp.start > cursor) host.appendChild(document.createTextNode(text.slice(cursor, sp.start)));
    const mark = document.createElement("mark");
    const cls = sp.kind === "full"
      ? "mk-full"
      : (sp.embedded_in ? "mk-embed" : "mk-abbr");
    mark.className = cls;
    mark.textContent = text.slice(sp.start, sp.end);
    const title = sp.kind === "full"
      ? `全称：${sp.name}`
      : sp.embedded_in
        ? `简称“${sp.name}”嵌在全称“${sp.embedded_in}”内，不重复计数`
        : `简称：${sp.name}`;
    mark.title = title;
    host.appendChild(mark);
    cursor = sp.end;
  });
  if (cursor < text.length) host.appendChild(document.createTextNode(text.slice(cursor)));
}

function renderContextDetails(result) {
  const t = result.thresholds;
  const host = abbr$("#abbr-context");
  host.innerHTML = "";
  const meta = document.createElement("div");
  meta.className = "threshold-meta";
  meta.textContent =
    `本次拟合特征 ${result.feature_count} 个（全部为字符 2~3 元）；` +
    `权重：语境 ${t.context_weight} / 字形 ${t.glyph_weight}；` +
    `确认阈值 ${t.confirm_min_score}，语境相似度下限 ${t.context_min_sim}，` +
    `歧义间距 ${t.ambiguity_margin}，单侧有效语境下限 ${t.min_context_chars} 字，` +
    `语境窗口 ±${t.context_window} 字。`;
  host.appendChild(meta);

  const names = [...result.full_names, ...result.abbreviations];
  names.forEach((name) => {
    const block = document.createElement("div");
    block.className = "context-block";
    const doc = result.context_docs[name] || "";
    block.innerHTML = `<div class="context-name">${name}</div>
      <div class="context-doc">${doc || "（无独立语境）"}</div>`;
    host.appendChild(block);
  });
}

/** 响应是否仍属于当前本地世代：优先比对回显的 client_token。 */
function isFresh(job, token) {
  if (job.client_token != null) return job.client_token === token;
  return job.request_id === token; // 兼容旧后端
}

function renderJobStatus(job, token) {
  const el = abbr$("#abbr-job-status");
  const label = {
    pending: "核查排队中……",
    running: "正在拟合本稿语境并计算相似度……",
    done: `核查完成（请求 #${job.request_id}）`,
    failed: `核查失败：${job.error || "未知错误"}`,
    stale: "该结果已过期（输入已变更，迟到结果不会覆盖新输入）",
  }[job.status] || job.status;
  el.textContent = label;
  el.className = `job-status ${job.status}`;

  if (job.status === "done" && isFresh(job, token)) {
    clearAbbrPoll();
    renderPairs(job);
    showAbbrMsg("核查完成", true);
  } else if (job.status === "stale" || !isFresh(job, token)) {
    clearAbbrPoll();
    el.textContent = "该结果已过期，已按最新输入撤下";
    el.className = "job-status stale";
  } else if (job.status === "failed") {
    clearAbbrPoll();
    showAbbrMsg(job.error || "核查失败");
  }
}

async function pollJob(requestId, token) {
  try {
    const data = await abbrApi(`/api/abbr/jobs/${requestId}`);
    if (token !== abbrState.token) return; // 已被新输入取代
    renderJobStatus(data.job, token);
    if (["pending", "running"].includes(data.job.status)) {
      abbrState.pollTimer = setTimeout(() => pollJob(requestId, token), 500);
    }
  } catch (err) {
    if (token !== abbrState.token) return;
    showAbbrMsg(err.message);
  }
}

async function runAbbrCheck() {
  const manuscript = abbr$("#abbr-manuscript").value;
  const fullNames = readNameList(abbr$("#abbr-full"));
  const abbreviations = readNameList(abbr$("#abbr-short"));
  const delay = parseInt(abbr$("#abbr-delay").value, 10) || 0;

  abbrState.token += 1;
  const token = abbrState.token;
  clearAbbrPoll();
  abbrState.lastResult = null;
  abbr$("#abbr-pairs").innerHTML = "";
  abbr$("#abbr-warnings").innerHTML = "";
  abbr$("#abbr-context").innerHTML = "";
  abbr$("#abbr-job-status").className = "job-status running";
  abbr$("#abbr-job-status").textContent = "正在提交核查……";

  try {
    const data = await abbrApi("/api/abbr/check", {
      method: "POST",
      body: JSON.stringify({
        manuscript,
        full_names: fullNames,
        abbreviations,
        compute_delay_ms: delay,
        client_token: token,
      }),
    });
    if (token !== abbrState.token) return;
    renderJobStatus(data.job, token);
    if (["pending", "running"].includes(data.job.status)) {
      abbrState.pollTimer = setTimeout(
        () => pollJob(data.job.request_id, token), 500
      );
    }
  } catch (err) {
    if (token !== abbrState.token) return;
    clearAbbrPoll();
    abbr$("#abbr-job-status").textContent = "输入校验未通过";
    abbr$("#abbr-job-status").className = "job-status failed";
    showAbbrMsg(err.message);
  }
}

function fillSample() {
  abbr$("#abbr-manuscript").value = SAMPLE_MANUSCRIPT;
  abbr$("#abbr-full").value = SAMPLE_FULL.join("\n");
  abbr$("#abbr-short").value = SAMPLE_ABBR.join("\n");
  updateLength();
  invalidateAbbr();
}

function clearAll() {
  abbr$("#abbr-manuscript").value = "";
  abbr$("#abbr-full").value = "";
  abbr$("#abbr-short").value = "";
  updateLength();
  invalidateAbbr(false);
  showAbbrMsg("");
}

function initAbbrView() {
  abbr$("#btn-abbr-check").addEventListener("click", runAbbrCheck);
  abbr$("#btn-abbr-sample").addEventListener("click", fillSample);
  abbr$("#btn-abbr-clear").addEventListener("click", clearAll);
  ["#abbr-manuscript", "#abbr-full", "#abbr-short", "#abbr-delay"].forEach((sel) => {
    abbr$(sel).addEventListener("input", () => {
      updateLength();
      invalidateAbbr();
    });
  });
  updateLength();
}

// 标签页切换（两个视图互不影响状态）
function initTabs() {
  const tabs = document.querySelectorAll("#view-tabs .tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      const view = tab.dataset.view;
      abbr$("#view-recommend").hidden = view !== "recommend";
      abbr$("#view-abbr").hidden = view !== "abbr";
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initAbbrView();
  initTabs();
});
