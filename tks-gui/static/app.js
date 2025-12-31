const sourceEl = document.getElementById("source");
const stdoutEl = document.getElementById("stdout");
const bytecodeEl = document.getElementById("bytecode");
const stderrEl = document.getElementById("stderr");
const statusEl = document.getElementById("status");
const editorErrorEl = document.getElementById("editor-error");
const runBtn = document.getElementById("run-btn");
const runBcBtn = document.getElementById("run-bc-btn");
const validateBtn = document.getElementById("validate-btn");
const emitBtn = document.getElementById("emit-btn");
const emitKind = document.getElementById("emit-kind");
const ffiToggle = document.getElementById("ffi-toggle");
const clearToggle = document.getElementById("clear-toggle");
const clearBtn = document.getElementById("clear-btn");
const insertBtn = document.getElementById("insert-btn");
const snippetSelect = document.getElementById("snippet-select");
const savedSnippetSelect = document.getElementById("saved-snippet-select");
const loadSnippetBtn = document.getElementById("load-snippet-btn");
const saveSnippetBtn = document.getElementById("save-snippet-btn");
const deleteSnippetBtn = document.getElementById("delete-snippet-btn");
const savedProjectSelect = document.getElementById("saved-project-select");
const loadProjectBtn = document.getElementById("load-project-btn");
const saveProjectBtn = document.getElementById("save-project-btn");
const deleteProjectBtn = document.getElementById("delete-project-btn");

const SNIPPET_STORE_KEY = "tks_snippets";
const PROJECT_STORE_KEY = "tks_projects";

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

function setOutput(stdout, stderr, bytecode) {
  stdoutEl.textContent = stdout || "";
  stderrEl.textContent = stderr || "";
  bytecodeEl.textContent = bytecode || "";
}

function setBusy(isBusy) {
  runBtn.disabled = isBusy;
  runBcBtn.disabled = isBusy;
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
    setOutput("", "", "");
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
      if (emitKind.value === "bc") {
        result.bytecode = result.stdout;
        result.stdout = "";
      }
    } else if (kind === "runBytecode") {
      payload.ffi = ffiToggle.checked;
      result = await postJson("/api/run_bytecode", payload);
    } else {
      throw new Error("unknown action");
    }
    const bytecode = result.bytecode || "";
    setOutput(result.stdout, result.stderr, bytecode);
    if (result.ok) {
      clearEditorError();
    } else {
      showEditorError(result.stderr || result.stdout || "Unknown error");
    }
    if (result.ok) {
      setStatus("ok", `OK (${result.duration_ms} ms)`);
    } else {
      setStatus("error", `Error (${result.duration_ms} ms)`);
    }
  } catch (err) {
    setStatus("error", "Request failed");
    setOutput("", String(err), "");
    showEditorError(String(err));
  } finally {
    setBusy(false);
  }
}

runBtn.addEventListener("click", () => runAction("run"));
runBcBtn.addEventListener("click", () => runAction("runBytecode"));
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

function loadStore(key) {
  const raw = localStorage.getItem(key);
  if (!raw) {
    return [];
  }
  try {
    const items = JSON.parse(raw);
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
}

function saveStore(key, items) {
  localStorage.setItem(key, JSON.stringify(items));
}

function refreshSelect(selectEl, items, placeholder) {
  while (selectEl.options.length > 0) {
    selectEl.remove(0);
  }
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = placeholder;
  selectEl.appendChild(empty);
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.name;
    selectEl.appendChild(option);
  }
}

function saveSnippet() {
  const name = prompt("Snippet name?");
  if (!name) {
    return;
  }
  const items = loadStore(SNIPPET_STORE_KEY);
  const existing = items.find((item) => item.name === name);
  if (existing) {
    existing.source = sourceEl.value || "";
    existing.updated_at = Date.now();
  } else {
    items.push({ name, source: sourceEl.value || "", updated_at: Date.now() });
  }
  saveStore(SNIPPET_STORE_KEY, items);
  refreshSelect(savedSnippetSelect, items, "Saved snippets...");
  savedSnippetSelect.value = name;
}

function loadSnippet() {
  const name = savedSnippetSelect.value;
  if (!name) {
    return;
  }
  const items = loadStore(SNIPPET_STORE_KEY);
  const snippet = items.find((item) => item.name === name);
  if (!snippet) {
    return;
  }
  sourceEl.value = snippet.source || "";
  sourceEl.focus();
}

function deleteSnippet() {
  const name = savedSnippetSelect.value;
  if (!name) {
    return;
  }
  const items = loadStore(SNIPPET_STORE_KEY).filter(
    (item) => item.name !== name
  );
  saveStore(SNIPPET_STORE_KEY, items);
  refreshSelect(savedSnippetSelect, items, "Saved snippets...");
}

function saveProject() {
  const name = prompt("Project name?");
  if (!name) {
    return;
  }
  const items = loadStore(PROJECT_STORE_KEY);
  const project = {
    name,
    source: sourceEl.value || "",
    ffi: ffiToggle.checked,
    emitKind: emitKind.value,
    clearOutput: clearToggle.checked,
    updated_at: Date.now(),
  };
  const existing = items.find((item) => item.name === name);
  if (existing) {
    Object.assign(existing, project);
  } else {
    items.push(project);
  }
  saveStore(PROJECT_STORE_KEY, items);
  refreshSelect(savedProjectSelect, items, "Saved projects...");
  savedProjectSelect.value = name;
}

function loadProject() {
  const name = savedProjectSelect.value;
  if (!name) {
    return;
  }
  const items = loadStore(PROJECT_STORE_KEY);
  const project = items.find((item) => item.name === name);
  if (!project) {
    return;
  }
  sourceEl.value = project.source || "";
  ffiToggle.checked = Boolean(project.ffi);
  clearToggle.checked = project.clearOutput !== false;
  if (project.emitKind) {
    emitKind.value = project.emitKind;
  }
  sourceEl.focus();
}

function deleteProject() {
  const name = savedProjectSelect.value;
  if (!name) {
    return;
  }
  const items = loadStore(PROJECT_STORE_KEY).filter(
    (item) => item.name !== name
  );
  saveStore(PROJECT_STORE_KEY, items);
  refreshSelect(savedProjectSelect, items, "Saved projects...");
}

function highlightError(line, col) {
  const text = sourceEl.value || "";
  const lines = text.split("\n");
  if (line < 1 || line > lines.length) {
    return;
  }
  const lineText = lines[line - 1];
  let offset = 0;
  for (let i = 0; i < line - 1; i += 1) {
    offset += lines[i].length + 1;
  }
  const start = offset;
  const end = offset + lineText.length;
  sourceEl.focus();
  sourceEl.setSelectionRange(start, end);
  const lineHeight = parseInt(
    window.getComputedStyle(sourceEl).lineHeight || "18",
    10
  );
  sourceEl.scrollTop = Math.max(0, lineHeight * (line - 1) - 60);
}

function showEditorError(stderr) {
  if (!stderr) {
    clearEditorError();
    return;
  }
  const firstLine = stderr.split("\n").find((line) => line.trim().length > 0);
  const match = firstLine ? firstLine.match(/:(\d+):(\d+):/) : null;
  if (match) {
    const line = Number.parseInt(match[1], 10);
    const col = Number.parseInt(match[2], 10);
    highlightError(line, col);
    editorErrorEl.textContent = `Line ${line}, Col ${col}: ${firstLine}`;
  } else {
    editorErrorEl.textContent = firstLine || stderr;
  }
  editorErrorEl.classList.remove("hidden");
  editorErrorEl.classList.add("visible");
}

function clearEditorError() {
  editorErrorEl.textContent = "";
  editorErrorEl.classList.remove("visible");
  editorErrorEl.classList.add("hidden");
}

loadSnippetBtn.addEventListener("click", loadSnippet);
saveSnippetBtn.addEventListener("click", saveSnippet);
deleteSnippetBtn.addEventListener("click", deleteSnippet);
loadProjectBtn.addEventListener("click", loadProject);
saveProjectBtn.addEventListener("click", saveProject);
deleteProjectBtn.addEventListener("click", deleteProject);

refreshSelect(
  savedSnippetSelect,
  loadStore(SNIPPET_STORE_KEY),
  "Saved snippets..."
);
refreshSelect(
  savedProjectSelect,
  loadStore(PROJECT_STORE_KEY),
  "Saved projects..."
);
