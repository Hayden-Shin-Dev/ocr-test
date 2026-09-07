const state = {
  sample: null,
  file: null,
  previewUrl: null,
  result: null,
};

const $ = (selector) => document.querySelector(selector);

function setPreview(url, name) {
  const image = $("#previewImage");
  const empty = $("#emptyPreview");
  if (!url) {
    image.hidden = true;
    empty.hidden = false;
    $("#sourceName").textContent = "선택 대기 중";
    return;
  }
  image.src = url;
  image.hidden = false;
  empty.hidden = true;
  $("#sourceName").textContent = name;
}

function refreshRunState() {
  const selected = Boolean(state.sample || state.file);
  $("#runButton").disabled = !selected;
  $("#selectionLabel").textContent = state.file
    ? `업로드: ${state.file.name}`
    : state.sample
      ? `샘플: ${state.sample.document_type} · ${state.sample.name}`
      : "문서를 선택하면 분석할 수 있습니다.";
}

function clearError() {
  $("#errorMessage").hidden = true;
  $("#errorMessage").textContent = "";
}

function showError(message) {
  $("#errorMessage").hidden = false;
  $("#errorMessage").textContent = message;
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error("샘플 목록을 불러오지 못했습니다.");
  const payload = await response.json();
  const select = $("#sampleSelect");
  const groups = new Map();
  for (const item of payload.items) {
    if (!groups.has(item.document_type)) groups.set(item.document_type, []);
    groups.get(item.document_type).push(item);
  }
  for (const [type, items] of groups) {
    const group = document.createElement("optgroup");
    group.label = type;
    for (const item of items) {
      const option = document.createElement("option");
      option.value = item.path;
      option.textContent = item.name;
      group.appendChild(option);
    }
    select.appendChild(group);
  }
  $("#sampleCount").textContent = `${payload.count}개 샘플 사용 가능`;
}

function resetResult() {
  state.result = null;
  $("#fullText").value = "";
  $("#lineList").className = "line-list empty-lines";
  $("#lineList").innerHTML = "<p>추출을 실행하면 라벨과 값이 영역별로 매핑됩니다.</p>";
  $("#textStatus").textContent = "결과 대기 중";
  for (const selector of ["#lineCount", "#confidence", "#imageSize", "#elapsed"]) $(selector).textContent = "—";
  $("#copyButton").disabled = true;
  $("#downloadButton").disabled = true;
}

function renderResult(result) {
  state.result = result;
  $("#fullText").value = result.structured_text || result.text || "";
  const layoutMode = result.layout_mode || "coordinate_fallback";
  const lowConfidence = result.low_confidence_count || 0;
  $("#textStatus").textContent = `${result.fields?.length || 0}개 영역 · ${layoutMode} · 저신뢰 ${lowConfidence}개`;
  const average = result.lines.length
    ? result.lines.reduce((sum, line) => sum + line.confidence, 0) / result.lines.length
    : 0;
  $("#lineCount").textContent = result.lines.length.toLocaleString();
  $("#confidence").textContent = `${(average * 100).toFixed(1)}%`;
  $("#imageSize").textContent = `${result.width} × ${result.height}`;
  $("#elapsed").textContent = `${result.elapsed_ms.toLocaleString()} ms`;
  const list = $("#lineList");
  list.className = "line-list";
  list.innerHTML = "";
  for (const field of result.fields || []) {
    const row = document.createElement("div");
    row.className = "line-row";
    const label = document.createElement("span");
    label.className = "field-label";
    label.textContent = field.label;
    label.title = field.label;
    const value = document.createElement("span");
    value.className = "field-value";
    const displayValue = Array.isArray(field.value) ? field.value.join(" | ") : field.value;
    value.textContent = displayValue || "—";
    value.title = displayValue || "—";
    const score = document.createElement("span");
    score.className = "line-score";
    score.textContent = `${(field.confidence * 100).toFixed(1)}%`;
    row.append(label, value, score);
    list.appendChild(row);
  }
  $("#copyButton").disabled = !(result.structured_text || result.text);
  $("#downloadButton").disabled = !(result.structured_text || result.text);
}

async function runOCR() {
  clearError();
  const button = $("#runButton");
  button.disabled = true;
  button.innerHTML = "분석 중… <span>◌</span>";
  $("#textStatus").textContent = "PaddleOCR 실행 중";
  try {
    let response;
    if (state.file) {
      const formData = new FormData();
      formData.append("file", state.file);
      response = await fetch("/api/ocr", { method: "POST", body: formData });
    } else {
      response = await fetch("/api/ocr/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: state.sample.path }),
      });
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "OCR 처리에 실패했습니다.");
    renderResult(payload);
  } catch (error) {
    $("#textStatus").textContent = "처리 실패";
    showError(error.message);
  } finally {
    button.disabled = !(state.sample || state.file);
    button.innerHTML = "텍스트 추출 <span>→</span>";
  }
}

$("#sampleSelect").addEventListener("change", (event) => {
  const option = event.target.options[event.target.selectedIndex];
  if (!event.target.value) {
    state.sample = null;
    setPreview(null);
  } else {
    state.file = null;
    $("#fileInput").value = "";
    state.sample = { path: event.target.value, document_type: option.parentElement.label, name: option.textContent };
    setPreview(`/api/sample-image?path=${encodeURIComponent(state.sample.path)}`, state.sample.name);
  }
  resetResult();
  refreshRunState();
});

$("#fileInput").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) return;
  state.sample = null;
  $("#sampleSelect").value = "";
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  setPreview(state.previewUrl, file.name);
  resetResult();
  refreshRunState();
});

$("#runButton").addEventListener("click", runOCR);
$("#copyButton").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.result?.structured_text || state.result?.text || "");
  $("#copyButton").textContent = "복사 완료";
  setTimeout(() => { $("#copyButton").textContent = "텍스트 복사"; }, 1400);
});
$("#downloadButton").addEventListener("click", () => {
  const blob = new Blob([state.result?.structured_text || state.result?.text || ""], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.result?.source_name || "ocr-result"}.txt`;
  link.click();
  URL.revokeObjectURL(link.href);
});

loadSamples().catch((error) => showError(error.message));
