const sourceEl = document.getElementById("source");
const stdoutEl = document.getElementById("stdout");
const stderrEl = document.getElementById("stderr");
const statusEl = document.getElementById("status");
const runBtn = document.getElementById("run-btn");
const validateBtn = document.getElementById("validate-btn");
const emitBtn = document.getElementById("emit-btn");
const emitKind = document.getElementById("emit-kind");
const ffiToggle = document.getElementById("ffi-toggle");
const clearToggle = document.getElementById("clear-toggle");
const clearBtn = document.getElementById("clear-btn");
const insertBtn = document.getElementById("insert-btn");
const snippetSelect = document.getElementById("snippet-select");

const SNIPPETS = {
  arith: "let x = 2 + 3;\nx * 4",
  noetic: "3^1",
  ordinal: "omega + 2",
  quantum: "measure(superpose { 1: |10>, 2: |20> })",
  rpm: "return 5",
  effect:
    "effect Log {\n  op log(msg: Int): Int;\n}\n\nhandle let x = perform log(2) in x with {\n  return v -> v;\n  log(msg) k -> resume(msg);\n}\n",
  foundation: "1a",
};

function setStatus(state, text) {
  statusEl.classList.remove("idle", "ok", "error", "busy");
  statusEl.classList.add(state);
  statusEl.textContent = text;
}

function setOutput(stdout, stderr) {
  stdoutEl.textContent = stdout || "";
  stderrEl.textContent = stderr || "";
}

function setBusy(isBusy) {
  runBtn.disabled = isBusy;
  validateBtn.disabled = isBusy;
  emitBtn.disabled = isBusy;
}

async function postJson(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

async function runAction(kind) {
  const source = sourceEl.value || "";
  if (clearToggle.checked) {
    setOutput("", "");
  }
  setBusy(true);
  setStatus("busy", "Running...");
  try {
    let payload = { source };
    let result;
    if (kind === "run") {
      payload.ffi = ffiToggle.checked;
      result = await postJson("/api/run", payload);
    } else if (kind === "validate") {
      result = await postJson("/api/validate", payload);
    } else if (kind === "emit") {
      payload.kind = emitKind.value;
      result = await postJson("/api/emit", payload);
    } else {
      throw new Error("unknown action");
    }
    setOutput(result.stdout, result.stderr);
    if (result.ok) {
      setStatus("ok", `OK (${result.duration_ms} ms)`);
    } else {
      setStatus("error", `Error (${result.duration_ms} ms)`);
    }
  } catch (err) {
    setStatus("error", "Request failed");
    setOutput("", String(err));
  } finally {
    setBusy(false);
  }
}

runBtn.addEventListener("click", () => runAction("run"));
validateBtn.addEventListener("click", () => runAction("validate"));
emitBtn.addEventListener("click", () => runAction("emit"));

clearBtn.addEventListener("click", () => {
  sourceEl.value = "";
  sourceEl.focus();
});

insertBtn.addEventListener("click", () => {
  const key = snippetSelect.value;
  if (!key) {
    return;
  }
  sourceEl.value = SNIPPETS[key] || "";
  sourceEl.focus();
});

sourceEl.addEventListener("input", () => {
  localStorage.setItem("tks_source", sourceEl.value);
});

const saved = localStorage.getItem("tks_source");
if (saved) {
  sourceEl.value = saved;
}
