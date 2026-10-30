import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  compatibilityFor,
  createProviderAuditFetch,
  runOneTask,
  samplingParamsFor,
} from "../run.mjs";

const PLAN_SHA256 = "a".repeat(64);

async function withLedger(operationIds, callback, events = []) {
  const root = await mkdtemp(join(tmpdir(), "rsi-call-ledger-"));
  const path = join(root, "calls.jsonl");
  const starts = operationIds.map((operationId) => ({
    event: "logical_start", operation_id: operationId, plan_sha256: PLAN_SHA256,
  }));
  await writeFile(path, [...starts, ...events].map((event) => JSON.stringify(event)).join("\n") + "\n");
  const names = [
    "RSI_CALL_LEDGER_PATH",
    "RSI_CALL_LEDGER_MAX_LOGICAL",
    "RSI_CALL_LEDGER_MAX_PHYSICAL",
    "RSI_ACTIVE_EXECUTION_PLAN_SHA256",
  ];
  const previous = Object.fromEntries(names.map((name) => [name, process.env[name]]));
  Object.assign(process.env, {
    RSI_CALL_LEDGER_PATH: path,
    RSI_CALL_LEDGER_MAX_LOGICAL: "10",
    RSI_CALL_LEDGER_MAX_PHYSICAL: "10",
    RSI_ACTIVE_EXECUTION_PLAN_SHA256: PLAN_SHA256,
  });
  try {
    return await callback(path);
  } finally {
    for (const name of names) {
      if (previous[name] === undefined) delete process.env[name];
      else process.env[name] = previous[name];
    }
  }
}

test("SiliconFlow compatibility uses its documented max_tokens field", () => {
  assert.deepEqual(compatibilityFor("https://api.siliconflow.cn/v1"), {
    maxTokensField: "max_tokens",
    supportsStore: false,
  });
  assert.deepEqual(compatibilityFor("https://example.test/v1"), {});
});

test("DeepSeek sampling payload disables thinking and downgrades JSON Schema", () => {
  const responseFormat = {
    type: "json_schema",
    json_schema: { name: "answer", strict: true, schema: { type: "object" } },
  };
  assert.deepEqual(samplingParamsFor({
    temperature: 0,
    enableThinking: false,
    responseFormat,
  }, "https://api.deepseek.com/v1"), {
    temperature: 0,
    enable_thinking: false,
    response_format: { type: "json_object" },
    thinking: { type: "disabled" },
  });
});

test("non-DeepSeek endpoints keep their configured sampling payload", () => {
  const extraBody = { top_p: 0.2, response_format: { type: "json_schema" } };
  assert.deepEqual(samplingParamsFor({
    temperature: 0.1,
    enableThinking: true,
    extraBody,
  }, "https://api.example.test/v1"), {
    temperature: 0.1,
    enable_thinking: true,
    top_p: 0.2,
    response_format: { type: "json_schema" },
  });
});

test("provider audit captures bounded safe error fields without consuming the response", async () => {
  const body = JSON.stringify({
    error: {
      code: "rate_limit_exceeded",
      type: "rate_limit_error",
      message: "Request rejected; bearer Bearer sf-secret-token-value, token=sf-another-secret-value",
      prompt: "must not be recorded",
    },
  });
  await withLedger(["audit-error"], async () => {
    const attempts = [];
    const auditedFetch = createProviderAuditFetch(async () => new Response(body, {
      status: 429,
      headers: { "content-type": "application/json", "retry-after": "7" },
    }), attempts, "audit-error");

    const response = await auditedFetch("https://api.example.test/v1/chat/completions", {});
    assert.equal(response.status, 429);
    assert.equal(await response.text(), body);
    assert.equal(attempts[0].providerError.code, "rate_limit_exceeded");
    assert.equal(attempts[0].providerError.type, "rate_limit_error");
    assert.match(attempts[0].providerError.message, /Request rejected/);
    assert.match(attempts[0].providerError.message, /REDACTED/);
    assert.equal(attempts[0].headers["retry-after"], "7");
    assert.doesNotMatch(JSON.stringify(attempts), /sf-secret-token-value|prompt/);
  });
});

test("Pi provider applies bounded backoff to 429 without Retry-After", async () => {
  const started = Date.now();
  await withLedger(["backoff"], async () => {
    const auditedFetch = createProviderAuditFetch(async () => new Response("busy", {
      status: 429,
    }), [], "backoff", { maxRetries: 1, rateLimitBackoffBaseMs: 5 });
    await auditedFetch("https://api.example.test/v1/chat/completions", {});
  });
  assert.ok(Date.now() - started >= 5);
});

test("Pi bridge refuses to dispatch when the plan-bound request ledger is absent", async () => {
  const names = [
    "RSI_CALL_LEDGER_PATH",
    "RSI_CALL_LEDGER_MAX_LOGICAL",
    "RSI_CALL_LEDGER_MAX_PHYSICAL",
    "RSI_ACTIVE_EXECUTION_PLAN_SHA256",
  ];
  const previous = Object.fromEntries(names.map((name) => [name, process.env[name]]));
  for (const name of names) delete process.env[name];
  let dispatched = 0;
  try {
    await assert.rejects(
      createProviderAuditFetch(async () => {
        dispatched += 1;
        return new Response("must not dispatch", { status: 200 });
      }, [], "")("https://api.example.test/v1/chat/completions", {}),
      /bounded request ledger/,
    );
    assert.equal(dispatched, 0);
  } finally {
    for (const name of names) {
      if (previous[name] === undefined) delete process.env[name];
      else process.env[name] = previous[name];
    }
  }
});

test("Pi bridge refuses to dispatch when either request ceiling is zero", async () => {
  await withLedger(["zero-cap-operation"], async () => {
    let dispatched = 0;
    for (const name of ["RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL"]) {
      process.env[name] = "0";
      await assert.rejects(
        createProviderAuditFetch(async () => {
          dispatched += 1;
          return new Response("must not dispatch", { status: 200 });
        }, [], "zero-cap-operation")("https://api.example.test/v1/chat/completions", {}),
        /bounded request ledger/,
      );
      process.env[name] = "10";
    }
    assert.equal(dispatched, 0);
  });
});

test("Pi bridge refuses an unscoped request even with an otherwise valid ledger", async () => {
  await withLedger(["valid-but-unused-operation"], async () => {
    let dispatched = 0;
    await assert.rejects(
      createProviderAuditFetch(async () => {
        dispatched += 1;
        return new Response("must not dispatch", { status: 200 });
      }, [], "")("https://api.example.test/v1/chat/completions", {}),
      /bounded request ledger/,
    );
    assert.equal(dispatched, 0);
  });
});

test("Pi provider attempts are reserved against the shared physical request ceiling", async () => {
  await withLedger(["ledger-operation-1", "ledger-operation-2"], async (path) => {
    await writeFile(path, JSON.stringify({
      event: "logical_start", operation_id: "ledger-operation-1", plan_sha256: PLAN_SHA256,
    }) + "\n" + JSON.stringify({
      event: "logical_start", operation_id: "ledger-operation-2", plan_sha256: PLAN_SHA256,
    }) + "\n" + JSON.stringify({
      event: "physical_reservation", count: 1, operation_id: "ledger-operation-1",
    }) + "\n");
    let dispatched = 0;
    process.env.RSI_CALL_LEDGER_MAX_PHYSICAL = "2";
    const attempts = [];
    const audited = createProviderAuditFetch(async () => {
      dispatched += 1;
      return new Response("ok", { status: 200 });
    }, attempts, "ledger-operation-1");
    await audited("https://api.example.test/v1/chat/completions", {});
    assert.equal(dispatched, 1);
    assert.equal(attempts.length, 1);

    process.env.RSI_CALL_LEDGER_MAX_PHYSICAL = "1";
    await assert.rejects(
      createProviderAuditFetch(async () => {
        dispatched += 1;
        return new Response("should not send", { status: 200 });
      }, [], "ledger-operation-2")("https://api.example.test/v1/chat/completions", {}),
      /physical HTTP-attempt budget exhausted/,
    );
    assert.equal(dispatched, 1);
    const ledger = (await readFile(path, "utf8")).split(/\r?\n/).filter(Boolean)
      .map((line) => JSON.parse(line));
    assert.equal(ledger.filter((event) => event.event === "http_attempt_start").length, 1);
  });
});

test("Pi bridge rejects an operation recorded under another plan hash", async () => {
  await withLedger(["other-plan-operation"], async () => {
    process.env.RSI_ACTIVE_EXECUTION_PLAN_SHA256 = "b".repeat(64);
    let dispatched = 0;
    await assert.rejects(
      createProviderAuditFetch(async () => {
        dispatched += 1;
        return new Response("must not dispatch", { status: 200 });
      }, [], "other-plan-operation")("https://api.example.test/v1/chat/completions", {}),
      /matching active plan-bound logical ledger operation/,
    );
    assert.equal(dispatched, 0);
  });
});

test("Pi bridge rejects reuse of a completed logical operation", async () => {
  await withLedger(["completed-operation"], async (path) => {
    const ledger = (await readFile(path, "utf8")).trim();
    await writeFile(path, ledger + "\n" + JSON.stringify({
      event: "logical_end", operation_id: "completed-operation", status: "success",
    }) + "\n");
    let dispatched = 0;
    await assert.rejects(
      createProviderAuditFetch(async () => {
        dispatched += 1;
        return new Response("must not dispatch", { status: 200 });
      }, [], "completed-operation")("https://api.example.test/v1/chat/completions", {}),
      /matching active plan-bound logical ledger operation/,
    );
    assert.equal(dispatched, 0);
  });
});

test("Pi runs one task, persists JSONL, and only invokes the selected Skill", async () => {
  const root = await mkdtemp(join(tmpdir(), "rsi-pi-contract-"));
  const output = await runOneTask({
    cwd: root,
    sessionsRoot: join(root, "sessions"),
    sessionId: "single-task-contract",
    taskId: "visible-dev-1",
    taskSplit: "dev_optimize",
    releaseId: "baseline-v1",
    providerTraceId: "provider-trace-visible-dev-1",
    systemPrompt: "Answer from the supplied evidence.",
    userPrompt: "Evidence: treatment did not improve survival. Question: Did it improve survival?",
    skills: [
      { name: "interpret", description: "Interpret study findings", content: "Follow the outcome statement.", filePath: join(root, "interpret", "SKILL.md") },
      { name: "unused", description: "Other tasks", content: "UNUSED_SKILL_SENTINEL", filePath: join(root, "unused", "SKILL.md") },
    ],
    selectedSkill: "interpret",
    fakeResponse: "ANSWER: 1\nCITED: task-evidence",
  });
  assert.equal(output.assistantText, "ANSWER: 1\nCITED: task-evidence");
  assert.equal(output.provider, "faux");
  assert.equal(output.selectedSkill, "interpret");
  assert.ok(output.messageCount >= 2);
  assert.equal(output.messageEntries.length, output.messageCount);
  assert.equal(output.messageEntries.at(-1).entryId, output.assistantEntryId);
  const jsonl = await readFile(output.sessionPath, "utf8");
  assert.match(jsonl, /"kind":"header"/);
  assert.match(jsonl, /"customType":"rsi.task_manifest"/);
  assert.match(jsonl, /"taskId":"visible-dev-1"/);
  assert.match(jsonl, /Follow the outcome statement/);
  assert.doesNotMatch(jsonl, /UNUSED_SKILL_SENTINEL/);
  assert.match(jsonl, /treatment did not improve survival/);
});

test("Pi rejects unknown selected Skill before making a session", async () => {
  const root = await mkdtemp(join(tmpdir(), "rsi-pi-contract-"));
  await assert.rejects(runOneTask({
    cwd: root,
    sessionsRoot: join(root, "sessions"),
    systemPrompt: "Answer from evidence.",
    taskSplit: "dev_optimize",
    providerTraceId: "provider-trace-unknown-skill",
    userPrompt: "Question",
    skills: [],
    selectedSkill: "unknown",
    fakeResponse: "ANSWER: 0",
  }), /selectedSkill is not in skills/);
});

test("gate task uses Pi memory session and writes no JSONL", async () => {
  const root = await mkdtemp(join(tmpdir(), "rsi-pi-gate-"));
  const result = await runOneTask({
    cwd: root,
    sessionsRoot: join(root, "sessions"),
    taskId: "gate-1",
    taskSplit: "selection_gate",
    releaseId: "baseline-v1",
    providerTraceId: "provider-trace-gate-1",
    ephemeral: true,
    systemPrompt: "Answer from evidence.",
    userPrompt: "Question: Which one?",
    fakeResponse: "ANSWER: 0",
  });
  assert.equal(result.ephemeral, true);
  assert.equal(result.sessionPath, "");
  assert.equal(result.assistantText, "ANSWER: 0");
  assert.equal(result.messageEntries.length, result.messageCount);
  await assert.rejects(readFile(join(root, "sessions"), "utf8"), { code: "ENOENT" });
});

test("Pi bounds the rendered catalog and selected procedure before opening a session", async () => {
  const root = await mkdtemp(join(tmpdir(), "rsi-pi-budget-"));
  const input = {
    cwd: root, sessionsRoot: join(root, "sessions"),
    systemPrompt: "Answer from evidence.", userPrompt: "Question",
    taskSplit: "dev_optimize", providerTraceId: "provider-trace-budget",
    skills: [{ name: "interpret", description: "Interpret study findings",
      content: "Follow the outcome statement.", filePath: join(root, "interpret", "SKILL.md") }],
    selectedSkill: "interpret", fakeResponse: "ANSWER: 0",
  };
  await assert.rejects(runOneTask({ ...input, catalogCharBudget: 1 }), /catalog.*budget/i);
  await assert.rejects(runOneTask({ ...input, selectedSkillCharBudget: 1 }), /Skill.*budget/i);
  await assert.rejects(readFile(join(root, "sessions"), "utf8"), { code: "ENOENT" });
});
