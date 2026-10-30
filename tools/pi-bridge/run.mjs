import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, closeSync, fsyncSync, openSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  AgentHarness,
  BACKGROUND_CONTEXT,
  JsonlSessionRepo,
  MemorySessionRepo,
  formatSkillsForSystemPrompt,
} from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/harness/env/nodejs";
import { createModels, createProvider, envApiKeyAuth, fauxAssistantMessage, fauxProvider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";

function requiredString(value, label) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} is required`);
  return value;
}

export function compatibilityFor(baseUrl) {
  let hostname = "";
  try {
    hostname = new URL(baseUrl).hostname.toLowerCase();
  } catch {
    return {};
  }
  // SiliconFlow's OpenAI-compatible chat endpoint accepts max_tokens. The
  // generic Pi detector selects max_completion_tokens for unknown hosts; the
  // service silently ignores that field and falls back to its 8192-token cap.
  if (hostname === "api.siliconflow.cn") {
    return { maxTokensField: "max_tokens", supportsStore: false };
  }
  return {};
}

/**
 * Build the provider sampling payload without relying on Pi's generic
 * OpenAI-compatible defaults.  DeepSeek exposes two wire-level differences
 * that matter for the self-evolution harness: thinking is controlled by a
 * top-level `thinking` object, and JSON Schema response formats are not
 * accepted (the provider supports JSON mode instead).  Keep these adaptations
 * scoped to the official DeepSeek host so arbitrary OpenAI-compatible
 * endpoints retain their configured payload unchanged.
 */
export function samplingParamsFor(input, baseUrl) {
  const samplingParams = {
    temperature: Number(input.temperature ?? 0),
    enable_thinking: Boolean(input.enableThinking ?? false),
    ...(input.extraBody || {}),
    ...(input.responseFormat ? { response_format: input.responseFormat } : {}),
  };
  let hostname = "";
  try {
    hostname = new URL(baseUrl).hostname.toLowerCase();
  } catch {
    return samplingParams;
  }
  if (hostname === "api.deepseek.com") {
    samplingParams.thinking = {
      type: samplingParams.enable_thinking ? "enabled" : "disabled",
    };
    if (samplingParams.response_format?.type === "json_schema") {
      samplingParams.response_format = { type: "json_object" };
    }
  }
  return samplingParams;
}

const sha256 = (value) => createHash("sha256").update(value, "utf8").digest("hex");

const AUDITED_RESPONSE_HEADERS = [
  "retry-after", "retry-after-ms", "x-request-id", "request-id",
  "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
  "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
  "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
];

function redactProviderErrorText(value) {
  return String(value)
    .replace(/\bBearer\s+[^\s,;]+/gi, "Bearer [REDACTED]")
    .replace(/\b(?:sk|sf)-[A-Za-z0-9._-]{8,}\b/g, "[REDACTED]")
    .replace(/\b(api[_ -]?key|token)\s*[:=]\s*[^\s,;]+/gi, "$1=[REDACTED]")
    .slice(0, 512);
}

function appendCallLedgerEvent(event) {
  const path = process.env.RSI_CALL_LEDGER_PATH;
  if (!path) return;
  const fd = openSync(path, "a");
  try {
    appendFileSync(fd, JSON.stringify({ ...event, timestamp: new Date().toISOString() }) + "\n",
      "utf8");
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function ledgerPhysicalState(ledgerPath) {
  const actualByOperation = new Map();
  const reservedByOperation = new Map();
  const planByOperation = new Map();
  const endedOperations = new Set();
  let logicalStarts = 0;
  let unreservedActual = 0;
  let unscopedReservations = 0;
  try {
    for (const line of readFileSync(ledgerPath, "utf8").split(/\r?\n/)) {
      if (!line) continue;
      try {
        const event = JSON.parse(line);
        if (event.event === "logical_start") {
          const operationId = String(event.operation_id || "");
          if (operationId) {
            logicalStarts += 1;
            planByOperation.set(operationId, String(event.plan_sha256 || ""));
          }
        } else if (event.event === "logical_end") {
          const operationId = String(event.operation_id || "");
          if (operationId) endedOperations.add(operationId);
        } else if (event.event === "http_attempt_start") {
          const operationId = String(event.operation_id || "");
          if (operationId) {
            actualByOperation.set(operationId,
              (actualByOperation.get(operationId) || 0) + 1);
          } else {
            unreservedActual += 1;
          }
        } else if (event.event === "physical_reservation") {
          const count = Math.max(0, Number(event.count) || 0);
          const operationId = String(event.operation_id || "");
          if (operationId) {
            reservedByOperation.set(operationId,
              (reservedByOperation.get(operationId) || 0) + count);
          } else {
            unscopedReservations += count;
          }
        }
      } catch {
        // Ignore a torn final record; all preceding starts remain counted.
      }
    }
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  let consumed = unreservedActual + unscopedReservations;
  for (const [operationId, count] of reservedByOperation) {
    consumed += Math.max(count, actualByOperation.get(operationId) || 0);
  }
  for (const [operationId, count] of actualByOperation) {
    if (!reservedByOperation.has(operationId)) consumed += count;
  }
  return { actualByOperation, reservedByOperation, planByOperation,
    endedOperations, logicalStarts, consumed };
}

function reserveLedgerHttpAttempt(operationId, attempt) {
  const planSha256 = process.env.RSI_ACTIVE_EXECUTION_PLAN_SHA256 || "";
  const ledgerPath = process.env.RSI_CALL_LEDGER_PATH || "";
  const logicalLimit = Number.parseInt(process.env.RSI_CALL_LEDGER_MAX_LOGICAL || "0", 10);
  const physicalLimit = Number.parseInt(process.env.RSI_CALL_LEDGER_MAX_PHYSICAL || "0", 10);
  if (!operationId || !ledgerPath || !/^[0-9a-f]{64}$/.test(planSha256)
      || !Number.isSafeInteger(logicalLimit) || logicalLimit <= 0
      || !Number.isSafeInteger(physicalLimit) || physicalLimit <= 0) {
    throw new Error("Pi provider dispatch requires a bounded request ledger and active plan hash");
  }
  const ledger = ledgerPhysicalState(ledgerPath);
  if (ledger.planByOperation.get(operationId) !== planSha256
      || ledger.logicalStarts > logicalLimit
      || ledger.endedOperations.has(operationId)) {
    throw new Error("Pi provider dispatch requires a matching active plan-bound logical ledger operation");
  }
  const actual = ledger.actualByOperation.get(operationId) || 0;
  const reserved = ledger.reservedByOperation.get(operationId) || 0;
  const consumesReservedCapacity = actual < reserved;
  if (ledger.consumed >= physicalLimit && !consumesReservedCapacity) {
    throw new Error("run physical HTTP-attempt budget exhausted before dispatch");
  }
  appendCallLedgerEvent({ event: "http_attempt_start", operation_id: operationId, attempt });
}

function nextLedgerAttemptNumber(operationId, attempts) {
  if (!operationId || !process.env.RSI_CALL_LEDGER_PATH) return attempts.length + 1;
  return (ledgerPhysicalState(process.env.RSI_CALL_LEDGER_PATH)
    .actualByOperation.get(operationId) || 0) + 1;
}

async function boundedProviderError(response, maxBytes = 4096) {
  let text = "";
  try {
    const reader = response.clone().body?.getReader();
    if (!reader) return undefined;
    const chunks = [];
    let size = 0;
    while (size < maxBytes) {
      const { done, value } = await reader.read();
      if (done) break;
      const remaining = maxBytes - size;
      const chunk = value.subarray(0, remaining);
      chunks.push(chunk);
      size += chunk.byteLength;
      if (chunk.byteLength < value.byteLength) break;
    }
    await reader.cancel().catch(() => {});
    text = Buffer.concat(chunks).toString("utf8");
  } catch {
    return undefined;
  }
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    return undefined;
  }
  const source = payload?.error && typeof payload.error === "object"
    ? payload.error : payload;
  const result = {};
  for (const key of ["code", "type", "message"]) {
    const value = source?.[key];
    if (typeof value === "string" || typeof value === "number") {
      result[key] = redactProviderErrorText(value);
    }
  }
  return Object.keys(result).length ? result : undefined;
}

/** Capture bounded, non-secret metadata for every provider HTTP attempt. */
export function createProviderAuditFetch(fetchImpl, attempts, ledgerOperationId = "", retryPolicy = {}) {
  if (typeof fetchImpl !== "function" || !Array.isArray(attempts)) {
    throw new TypeError("fetchImpl and attempts are required");
  }
  return async (input, init) => {
    const startedAt = Date.now();
    let url = "";
    try {
      const rawUrl = typeof input === "string" || input instanceof URL
        ? String(input) : input?.url;
      const parsed = new URL(rawUrl);
      url = `${parsed.origin}${parsed.pathname}`;
    } catch {
      url = "<invalid-url>";
    }
    const attemptNumber = nextLedgerAttemptNumber(ledgerOperationId, attempts);
    let reserved = false;
    let response;
    try {
      reserveLedgerHttpAttempt(ledgerOperationId, attemptNumber);
      reserved = true;
      response = await fetchImpl(input, init);
      const headers = {};
      for (const name of AUDITED_RESPONSE_HEADERS) {
        const value = response.headers?.get(name);
        if (value !== null && value !== undefined) headers[name] = String(value).slice(0, 256);
      }
      const providerError = response.status >= 400
        ? await boundedProviderError(response) : undefined;
      attempts.push({
        status: response.status,
        url,
        durationMs: Math.max(0, Date.now() - startedAt),
        headers,
        ...(providerError ? { providerError } : {}),
      });
      appendCallLedgerEvent({
        event: "http_attempt_end", operation_id: ledgerOperationId,
        attempt: attemptNumber, status: Number(response.status),
      });
      const retriesRemaining = Number(retryPolicy.maxRetries || 0);
      const baseDelayMs = Number(retryPolicy.rateLimitBackoffBaseMs || 0);
      const hasRetryAfter = response.headers?.get("retry-after")
        || response.headers?.get("retry-after-ms");
      if (response.status === 429 && attemptNumber <= retriesRemaining
          && baseDelayMs > 0 && !hasRetryAfter) {
        const delayMs = Math.min(60_000, baseDelayMs * (2 ** (attemptNumber - 1)));
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
      return response;
    } catch (error) {
      // Failed budget reservations happen before dispatch and are not HTTP attempts.
      if (!reserved) throw error;
      appendCallLedgerEvent({
        event: "http_attempt_end", operation_id: ledgerOperationId,
        attempt: attemptNumber,
        status: Number.isInteger(response?.status) ? Number(response.status) : null,
        error_type: String(error?.name || "Error").slice(0, 100),
      });
      attempts.push({
        status: Number.isInteger(response?.status) ? Number(response.status) : null,
        url,
        durationMs: Math.max(0, Date.now() - startedAt),
        errorName: String(error?.name || "Error").slice(0, 80),
      });
      throw error;
    }
  };
}

function modelFor(input, providerTraceId) {
  const models = createModels();
  if (input.fakeResponse !== undefined) {
    if (typeof input.fakeResponse !== "string") throw new Error("fakeResponse must be a string");
    const faux = fauxProvider();
    faux.setResponses([fauxAssistantMessage(input.fakeResponse)]);
    models.setProvider(faux.provider);
    return { models, model: faux.getModel(), provider: "faux" };
  }
  const modelId = requiredString(input.modelId, "modelId");
  const baseUrl = requiredString(input.baseUrl, "baseUrl");
  const apiKeyEnv = requiredString(input.apiKeyEnv || "SILICONFLOW_API_KEY", "apiKeyEnv");
  if (!process.env[apiKeyEnv]) throw new Error(`${apiKeyEnv} is not set`);
  const model = {
    id: modelId,
    name: modelId,
    api: "openai-completions",
    provider: "rsi-task",
    baseUrl,
    compat: compatibilityFor(baseUrl),
    headers: { "X-Trace-Id": providerTraceId },
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: Number(input.contextWindow || 32768),
    maxTokens: Number(input.maxTokens || 512),
    samplingParams: samplingParamsFor(input, baseUrl),
  };
  const provider = createProvider({
    id: "rsi-task",
    name: "RSI task model",
    baseUrl,
    auth: { apiKey: envApiKeyAuth("RSI task model key", [apiKeyEnv]) },
    models: [model],
    api: openAICompletionsApi(),
  });
  models.setProvider(provider);
  return { models, model, provider: "rsi-task" };
}

/** Execute one already formatted task through Pi's durable harness and JSONL session. */
export async function runOneTask(input) {
  if (!input || typeof input !== "object") throw new Error("input must be an object");
  const cwd = requiredString(input.cwd, "cwd");
  const sessionsRoot = requiredString(input.sessionsRoot, "sessionsRoot");
  const systemPrompt = requiredString(input.systemPrompt, "systemPrompt");
  const userPrompt = requiredString(input.userPrompt, "userPrompt");
  const skills = Array.isArray(input.skills) ? input.skills : [];
  for (const skill of skills) {
    for (const key of ["name", "description", "content", "filePath"]) requiredString(skill[key], `skill.${key}`);
  }
  const selectedSkill = input.selectedSkill || "";
  const catalog = formatSkillsForSystemPrompt(skills);
  const catalogBudget = input.catalogCharBudget ?? 16000;
  const procedureBudget = input.selectedSkillCharBudget ?? 20000;
  if (!Number.isSafeInteger(catalogBudget) || catalogBudget < 1 || catalog.length > catalogBudget) {
    throw new Error("Skill catalog exceeds disclosure budget");
  }
  if (!Number.isSafeInteger(procedureBudget) || procedureBudget < 1 ||
      skills.some((skill) => skill.name === selectedSkill && skill.content.length > procedureBudget)) {
    throw new Error("selected Skill exceeds disclosure budget");
  }
  const taskSplit = requiredString(input.taskSplit, "taskSplit");
  const providerTraceId = requiredString(input.providerTraceId, "providerTraceId");
  const providerAttempts = [];
  const originalFetch = globalThis.fetch;
  if (!["dev_optimize", "selection_gate", "final_holdout",
        "independent_confirmation"].includes(taskSplit)) {
    throw new Error("unsupported taskSplit");
  }
  if ((taskSplit !== "dev_optimize") !== (input.ephemeral === true)) {
    throw new Error("gate/final tasks require an ephemeral Pi session");
  }
  if (selectedSkill && !skills.some((skill) => skill.name === selectedSkill)) {
    throw new Error("selectedSkill is not in skills");
  }
  const { models, model, provider } = modelFor(input, providerTraceId);
  const ephemeral = input.ephemeral === true;
  const env = ephemeral ? null : new NodeExecutionEnv({ cwd });
  const repo = ephemeral
    ? new MemorySessionRepo()
    : new JsonlSessionRepo({ fileSystem: env, sessionsRoot });
  const session = ephemeral
    ? await repo.create({ id: input.sessionId || randomUUID() }, BACKGROUND_CONTEXT)
    : await repo.create({ id: input.sessionId || randomUUID(), cwd }, BACKGROUND_CONTEXT);
  const sessionPath = ephemeral ? "" : session.metadata.path;
  let harness;
  try {
    globalThis.fetch = createProviderAuditFetch(
      originalFetch, providerAttempts, String(input.ledgerOperationId || ""), {
        maxRetries: Number.isSafeInteger(input.providerMaxRetries)
          ? Math.max(0, Math.min(4, input.providerMaxRetries)) : 4,
        rateLimitBackoffBaseMs: Number.isSafeInteger(input.provider429BackoffBaseMs)
          ? Math.max(0, Math.min(60_000, input.provider429BackoffBaseMs)) : 0,
      });
    ({ harness } = await AgentHarness.create({
      session,
      models,
      model,
      systemPrompt: [systemPrompt, catalog].filter(Boolean).join("\n\n"),
      resources: { skills },
      activeToolNames: [],
      tools: [],
      thinkingLevel: "off",
      // Use the provider SDK's bounded retry loop so Retry-After is honored.
      // Disable the outer harness retry loop to avoid multiplying requests.
      retry: { enabled: false, maxRetries: 0, baseDelayMs: 0 },
      streamOptions: {
        maxRetries: Number.isSafeInteger(input.providerMaxRetries)
          ? Math.max(0, Math.min(4, input.providerMaxRetries)) : 4,
        maxRetryDelayMs: 60_000,
      },
    }, BACKGROUND_CONTEXT));
    const lane = await harness.lane("main", BACKGROUND_CONTEXT);
    const taskManifest = {
      protocol: "pi-harness-task-input-v3",
      taskId: String(input.taskId || ""),
      taskSplit,
      releaseId: String(input.releaseId || ""),
      providerTraceId,
      inputMode: "task_input",
      systemPrompt,
      systemPromptSha256: sha256(systemPrompt),
      userPromptSha256: sha256(userPrompt),
      selectedSkill,
      demoTaskIds: Array.isArray(input.demoTaskIds) ? input.demoTaskIds : [],
      candidateDemoTaskIds: Array.isArray(input.candidateDemoTaskIds)
        ? input.candidateDemoTaskIds : [],
      demosBlockSha256: String(input.demosBlockSha256 || ""),
      skills: skills.map((skill) => ({
        name: skill.name,
        description: skill.description,
        bodySha256: sha256(skill.content),
        filePath: skill.filePath,
      })),
      sources: Array.isArray(input.sources) ? input.sources : [],
      responseFormatSha256: String(input.responseFormatSha256 || ""),
    };
    await lane.appendCustomEntry("rsi.task_manifest", taskManifest, BACKGROUND_CONTEXT);
    const result = selectedSkill
      ? await lane.skill(selectedSkill, userPrompt, BACKGROUND_CONTEXT)
      : await lane.prompt(userPrompt, undefined, BACKGROUND_CONTEXT);
    if (!result.ok) throw new Error(`Pi run admission failed: ${JSON.stringify(result.error)}`);
    if (result.value.status !== "completed") {
      throw new Error(`Pi run ${result.value.status}: ${JSON.stringify(result.value.error || {})}`);
    }
    const entries = await lane.findEntries({ type: "message", order: "oldestFirst" }, BACKGROUND_CONTEXT);
    const assistants = entries.filter((entry) => entry.type === "message" && entry.message.role === "assistant");
    const last = assistants.at(-1)?.message;
    const assistantText = last?.content?.filter((item) => item.type === "text").map((item) => item.text).join("") || "";
    if (!assistantText) throw new Error("Pi completed without an assistant text response");
    return {
      assistantText,
      sessionPath,
      sessionId: session.metadata.id,
      operationId: result.value.operationId,
      messageCount: entries.length,
      // Memory sessions have no JSONL to parse in Python. Keep the complete
      // role/ID manifest: some providers emit more than one assistant message.
      messageEntries: entries.map((entry) => ({ entryId: entry.id, role: entry.message.role })),
      latencyMs: Math.max(0, result.value.endedAt - result.value.startedAt),
      selectedSkill,
      provider,
      modelId: model.id,
      providerTraceId,
      usage: last?.usage || null,
      providerAttempts,
      taskManifest,
      ephemeral,
      userEntryIds: entries.filter((entry) => entry.message.role === "user").map((entry) => entry.id),
      userMessages: entries.filter((entry) => entry.message.role === "user")
        .map((entry) => entry.message.content?.filter((item) => item.type === "text")
          .map((item) => item.text).join("") || ""),
      assistantEntryId: assistants.at(-1)?.id || "",
    };
  } catch (error) {
    error.providerAttempts = providerAttempts;
    error.sessionPath = sessionPath;
    error.sessionId = session.metadata.id;
    throw error;
  } finally {
    globalThis.fetch = originalFetch;
    if (harness) await harness.close(BACKGROUND_CONTEXT);
    await session.close(BACKGROUND_CONTEXT);
    await repo.close(BACKGROUND_CONTEXT);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  let raw = "";
  for await (const chunk of process.stdin) raw += chunk;
  try {
    process.stdout.write(JSON.stringify(await runOneTask(JSON.parse(raw))) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({
      error: {
        name: String(error?.name || "Error"),
        message: String(error?.message || error),
      },
      providerAttempts: Array.isArray(error?.providerAttempts) ? error.providerAttempts : [],
      sessionPath: String(error?.sessionPath || ""),
      sessionId: String(error?.sessionId || ""),
    }) + "\n");
    process.stderr.write(`${error?.stack || error}\n`);
    process.exitCode = 1;
  }
}
