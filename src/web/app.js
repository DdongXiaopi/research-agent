"use strict";

const $ = (id) => document.getElementById(id);
const stageNames = {parse:"解析问题",route_skill:"选择研究方法",retrieve:"检索文献",organize_evidence:"整理证据",answer:"生成回答",verify:"核验引用"};
let token = "";
let currentJob = "";

async function post(path, body) {
  const response = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json", "X-Research-Token":token}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function sourceMode() {
  const search = document.querySelector('input[name="source-mode"]:checked').value === "search";
  $("provided-section").hidden = search;
  $("search-section").hidden = !search;
}
document.querySelectorAll('input[name="source-mode"]').forEach((input) => input.addEventListener("change", sourceMode));

function showStages(stage, completed) {
  document.querySelectorAll("#stages li").forEach((item, index) => {
    item.classList.toggle("done", index < completed);
    item.classList.toggle("active", item.dataset.stage === stage && index >= completed);
  });
}

function renderResult(result) {
  $("empty-result").hidden = true;
  $("result").hidden = false;
  $("result-status").textContent = result.response.status === "abstain" ? "材料不足" : "已完成";
  $("answer").textContent = result.response.answer;
  const citations = $("citations");
  citations.replaceChildren();
  $("citation-status").textContent = result.citation_verification.citation_count ? "原文已核验" : "无引用";
  for (const citation of result.response.citations) {
    const item = document.createElement("div"); item.className = "citation";
    const heading = document.createElement("strong"); heading.textContent = citation.source_id;
    const quote = document.createElement("p"); quote.textContent = "“" + citation.evidence_excerpt + "”";
    const detail = document.createElement("small"); detail.textContent = citation.locator || "来源材料";
    item.append(heading, quote, detail);
    try {
      const url = new URL(citation.url);
      if (["https:", "http:"].includes(url.protocol)) {
        const link = document.createElement("a"); link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = "查看来源";
        item.append(document.createTextNode("  "), link);
      }
    } catch (_) { /* Supplied materials need not have a URL. */ }
    citations.append(item);
  }
  if (!result.response.citations.length) citations.textContent = "当前回答没有可展示的引用。";
  const trace = $("trace-list"); trace.replaceChildren();
  result.stages.forEach((stage) => {
    const li = document.createElement("li");
    let detail = stage.stage === "route_skill" ? (stage.selected_skill || "未匹配专门方法") : "已完成";
    if (stage.stage === "retrieve") detail = stage.mode === "supplied" ? "使用自备材料" : `网络请求 ${stage.physical_requests} 次`;
    if (stage.stage === "organize_evidence") detail = `${stage.source_count} 条材料`;
    li.textContent = `${stageNames[stage.stage]}：${detail}`; trace.append(li);
  });
  showStages("", 6);
}

async function poll(jobId) {
  try {
    const response = await fetch("/api/jobs/" + encodeURIComponent(jobId));
    if (!response.ok) throw new Error("无法读取任务状态");
    const job = await response.json();
    showStages(job.stage, job.stages_completed);
    if (job.status === "running") {
      $("task-message").textContent = `正在${stageNames[job.stage] || "处理任务"}…`;
      $("result-status").textContent = "运行中";
      setTimeout(() => poll(jobId), 800); return;
    }
    $("run").disabled = false;
    if (job.status === "failed") {
      $("task-message").textContent = `任务在${stageNames[job.stage] || "执行"}阶段失败（${job.error_type}）。请检查配置与材料后重试。`;
      $("result-status").textContent = "任务失败"; return;
    }
    $("task-message").textContent = "研究完成。";
    renderResult(job.result);
  } catch (error) {
    $("run").disabled = false;
    $("task-message").textContent = error.message;
  }
}

$("task-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const mode = document.querySelector('input[name="source-mode"]:checked').value;
  const providers = mode === "search" ? [...document.querySelectorAll('input[name="provider"]:checked')].map((x) => x.value) : [];
  const sourceText = mode === "provided" ? $("source-text").value : "";
  $("run").disabled = true; $("result").hidden = true; $("empty-result").hidden = false;
  $("feedback-message").textContent = ""; $("task-message").textContent = "正在准备任务…"; showStages("parse", 0);
  try {
    const result = await post("/api/tasks", {query:$("query").value, source_text:sourceText, providers});
    currentJob = result.job_id; poll(currentJob);
  } catch (error) { $("run").disabled = false; $("task-message").textContent = error.message; }
});

$("category").addEventListener("change", () => {
  const category = $("category").value;
  $("quote-fields").hidden = category !== "evidence_fact";
  $("consent").disabled = category === "preference";
  if (category === "preference") $("consent").checked = false;
});
$("feedback-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const category = $("category").value;
  try {
    await post("/api/feedback", {job_id:currentJob, category, correction:$("correction").value,
      reviewer:$("reviewer").value, consent:$("consent").checked,
      source_id:category === "evidence_fact" ? $("source-id").value : "",
      quote:category === "evidence_fact" ? $("quote").value : ""});
    $("feedback-message").textContent = "反馈已保存。";
    $("correction").value = "";
  } catch (error) { $("feedback-message").textContent = error.message; }
});

fetch("/api/session").then((response) => { if (!response.ok) throw new Error(); return response.json(); })
  .then((session) => { token = session.token; })
  .catch(() => { $("run").disabled = true; $("task-message").textContent = "无法连接本地服务，请重新启动工作台。"; });
