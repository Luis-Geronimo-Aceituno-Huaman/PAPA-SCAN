/* PapaScan — lógica del cliente (SPA, sin frameworks, funciona offline) */
"use strict";

const API = "";
const LABELS = { sana: "Sana", tizon_tardio: "Tizón tardío", tizon_temprano: "Tizón temprano" };
const state = { token: localStorage.getItem("papascan_token") || null, user: null, file: null, case: null };

const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");
const loader = (txt) => { $("#loader-text").textContent = txt || "Procesando…"; show($("#loader")); };
const hideLoader = () => hide($("#loader"));

/* ---------------- Fetch con token ---------------- */
async function api(path, { method = "GET", body = null, form = null } = {}) {
  const headers = {};
  if (state.token) headers["Authorization"] = "Bearer " + state.token;
  let payload = null;
  if (form) { payload = form; }
  else if (body) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(API + path, { method, headers, body: payload });
  if (res.status === 401) { logout(); throw new Error("Sesión expirada"); }
  if (!res.ok) {
    let detail = "Error " + res.status;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

/* ---------------- Autenticación ---------------- */
let authMode = "login";
document.querySelectorAll(".login-tabs .tab").forEach((t) =>
  t.addEventListener("click", () => {
    authMode = t.dataset.tab;
    document.querySelectorAll(".login-tabs .tab").forEach((x) => x.classList.toggle("active", x === t));
    $("#auth-submit").textContent = authMode === "login" ? "Ingresar" : "Crear cuenta";
    $("#fullname-field").classList.toggle("hidden", authMode === "login");
    $("#role-field").classList.toggle("hidden", authMode === "login");
  }));

$("#auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#auth-error").textContent = "";
  const username = $("#auth-username").value.trim();
  const password = $("#auth-password").value;
  try {
    let data;
    if (authMode === "register") {
      const role = document.querySelector('input[name="role"]:checked').value;
      data = await api("/api/auth/register", { method: "POST",
        body: { username, password, full_name: $("#auth-fullname").value, role } });
    } else {
      const fd = new FormData();
      fd.append("username", username); fd.append("password", password);
      data = await api("/api/auth/login", { method: "POST", form: fd });
    }
    state.token = data.access_token; state.user = data.user;
    localStorage.setItem("papascan_token", state.token);
    enterApp();
  } catch (err) { $("#auth-error").textContent = err.message; }
});

function logout() {
  state.token = null; state.user = null;
  localStorage.removeItem("papascan_token");
  hide($("#app-view")); show($("#login-view"));
}
$("#logout").addEventListener("click", logout);

async function enterApp() {
  try {
    state.user = await api("/api/auth/me");
    $("#user-label").textContent = (state.user.full_name || state.user.username) +
      " · " + state.user.role;
    hide($("#login-view")); show($("#app-view"));
    navigate("captura");
  } catch (e) { logout(); }
}

/* ---------------- Navegación ---------------- */
function navigate(section) {
  document.querySelectorAll(".section").forEach((s) => s.classList.toggle("hidden", s.id !== section));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.section === section));
  $(".sidebar").classList.remove("open");
  if (section === "historial") loadHistory();
}
document.querySelectorAll(".nav-item").forEach((n) =>
  n.addEventListener("click", () => navigate(n.dataset.section)));
// Botones de retroceso del flujo (Diagnóstico → Captura, Explic./Recom. → Diagnóstico)
document.querySelectorAll(".btn-back").forEach((b) =>
  b.addEventListener("click", () => navigate(b.dataset.goto)));
$("#menu-toggle").addEventListener("click", () => $(".sidebar").classList.toggle("open"));

/* ---------------- Captura ---------------- */
const dropzone = $("#dropzone"), fileInput = $("#file-input");
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault(); dropzone.classList.remove("drag");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(file) {
  if (!file.type.startsWith("image/")) { alert("Selecciona una imagen válida."); return; }
  state.file = file;
  $("#preview-img").src = URL.createObjectURL(file);
  hide(dropzone); show($("#preview-wrap"));
}
$("#btn-clear").addEventListener("click", () => {
  state.file = null; fileInput.value = "";
  show(dropzone); hide($("#preview-wrap"));
});

$("#btn-analyze").addEventListener("click", async () => {
  if (!state.file) return;
  loader("Analizando hoja (CNN + Grad-CAM)…");
  try {
    const fd = new FormData(); fd.append("file", state.file);
    const data = await api("/api/diagnose", { method: "POST", form: fd });
    state.case = data;
    renderDiagnosis(data);
    setCaseChat(data.case_id, []);   // nuevo caso: chat vacío
    navigate("diagnostico");
  } catch (err) { alert("Error al analizar: " + err.message); }
  finally { hideLoader(); }
});

/* ---------------- Diagnóstico ---------------- */
function renderDiagnosis(d) {
  $("#diag-img").src = d.imagen_url;
  // Imágenes de la vista de explicabilidad (listas para cuando se abra esa opción).
  $("#exp-original").src = d.imagen_url;
  $("#exp-masked").src = d.masked_url || d.imagen_url;
  $("#exp-overlay").src = d.overlay_url;
  const q = state.case.calidad; // puede no venir; usar estado_confianza
  $("#diag-name").textContent = d.diagnostico;
  const sci = { tizon_tardio: "Late Blight · Phytophthora infestans",
                tizon_temprano: "Early Blight · Alternaria solani", sana: "Healthy" };
  $("#diag-sci").textContent = sci[d.probabilidades && winnerKey(d)] || "";
  $("#diag-conf-val").textContent = (d.confianza * 100).toFixed(1) + "%";
  const card = $("#result-card");
  card.classList.toggle("ok", winnerKey(d) === "sana");

  // Calidad
  const qb = $("#quality-badge");
  qb.textContent = "Calidad de imagen: " + (d.estado_confianza === "baja_confianza" ? "Revisar" : "Aprobada");
  qb.classList.toggle("fail", d.estado_confianza === "baja_confianza");
  $("#quality-summary").textContent = "Capa Grad-CAM: " + d.capa_usada +
    (d.alerta_sesgo ? " · ⚠ posible foco fuera de la hoja" : "");

  // Barras de probabilidad
  const bars = $("#prob-bars"); bars.innerHTML = "";
  const win = winnerKey(d);
  Object.keys(LABELS).forEach((k) => {
    const pct = (d.probabilidades[k] * 100);
    const row = document.createElement("div");
    row.className = "prob-row" + (k === win ? " win" : "");
    row.innerHTML = `<span class="lbl">${LABELS[k]}</span>
      <div class="track"><div class="fill" style="width:${pct}%"></div></div>
      <span class="pct">${pct.toFixed(1)}%</span>`;
    bars.appendChild(row);
  });

  // Alerta de confianza baja
  const alert = $("#conf-alert");
  if (d.estado_confianza === "baja_confianza" || d.confianza < 0.6) {
    alert.textContent = "⚠ Confianza baja. Consulta a un técnico agrónomo o a SENASA antes de actuar.";
    show(alert);
  } else hide(alert);

  $("#model-meta").textContent = `Modelo: ${d.version_modelo} · severidad estimada: ${d.severidad}`;

  renderRecommendation(d.recomendacion, d);
  // Reset de la explicación (el chat del caso se fija aparte con setCaseChat).
  $("#llm-explanation").innerHTML = "";
  $("#interp-text").textContent = "";
}
function winnerKey(d) {
  return Object.keys(d.probabilidades).reduce((a, b) => d.probabilidades[a] >= d.probabilidades[b] ? a : b);
}

/* ---------------- Opciones desde el diagnóstico ---------------- */
// Ver recomendación (ya renderizada en renderDiagnosis): solo navegar.
$("#btn-recom").addEventListener("click", () => {
  if (state.case) navigate("recomendacion");
});

// Ver explicabilidad: navega y genera la explicación del LLM si aún no existe
// (al abrir un caso del historial ya viene generada, no se vuelve a pedir).
$("#btn-explain").addEventListener("click", async () => {
  if (!state.case) return;
  navigate("explicabilidad");
  if ($("#interp-text").textContent.trim()) return;  // ya generada
  loader("Generando explicación con el asistente local…");
  try {
    const data = await api(`/api/explain/${state.case.case_id}`, { method: "POST" });
    renderExplanation(data, state.case);
  } catch (err) {
    $("#interp-text").textContent = "No se pudo generar la explicación: " + err.message;
  } finally { hideLoader(); }
});

function renderExplanation(data, c) {
  const e = data.explicacion;
  $("#interp-text").textContent = e.que_observo_el_modelo ||
    ("El modelo se concentró en " + (c.zona_gradcam || "la zona resaltada") + ".");
  const blocks = [
    ["Resumen", e.resumen, ""],
    ["Nivel de confianza", null, `<span class="llm-badge ${e.nivel_confianza}">${e.nivel_confianza || "—"}</span>`],
    ["Qué hacer", e.que_hacer, ""],
    ["Siguiente paso", e.siguiente_paso, ""],
  ];
  let html = "";
  blocks.forEach(([t, txt, raw]) => {
    if (!txt && !raw) return;
    html += `<div class="llm-block"><h5>${t}</h5><div>${raw || txt}</div></div>`;
  });
  if (e.alerta) html += `<div class="llm-block alerta"><h5>Alerta</h5><div>${e.alerta}</div></div>`;
  if (!data.explicacion_disponible)
    html += `<p class="llm-fallback">El asistente LLM no estaba disponible; se muestra una explicación de respaldo. El diagnóstico y la recomendación siguen siendo válidos.</p>`;
  $("#llm-explanation").innerHTML = html;
}

/* ---------------- Chat de seguimiento ----------------
   Dos ámbitos: "case" (compartido entre Explicabilidad y Recomendación, ligado al
   caso diagnosticado y persistido en BD) y "standalone" (pestaña Asistente, sin
   diagnóstico, historial solo en memoria del cliente). */
const chats = {
  case: { caseId: null, messages: [] },
  standalone: { caseId: null, messages: [] },
};

function renderChatLogs(scope) {
  const msgs = chats[scope].messages;
  document.querySelectorAll(`.chat-log[data-chat="${scope}"]`).forEach((log) => {
    log.innerHTML = "";
    msgs.forEach((m) => {
      const div = document.createElement("div");
      div.className = "msg " + m.role; div.textContent = m.content;
      log.appendChild(div);
    });
    log.scrollTop = log.scrollHeight;
  });
}
function setCaseChat(caseId, messages) {
  chats.case.caseId = caseId;
  chats.case.messages = messages || [];
  renderChatLogs("case");
}

document.querySelectorAll(".chat-form").forEach((form) =>
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const scope = form.dataset.chat;            // "case" | "standalone"
    const ctx = chats[scope];
    if (scope === "case" && !ctx.caseId) return; // sin diagnóstico aún
    const input = form.querySelector(".chat-input");
    const msg = input.value.trim();
    if (!msg) return;
    ctx.messages.push({ role: "user", content: msg });
    renderChatLogs(scope); input.value = "";
    try {
      const body = { mensaje: msg };
      if (scope === "case") body.case_id = ctx.caseId;
      else body.historial = ctx.messages.slice(0, -1);  // turnos previos (chat libre)
      const data = await api("/api/chat", { method: "POST", body });
      ctx.messages.push({ role: "assistant", content: data.respuesta });
    } catch (err) {
      ctx.messages.push({ role: "assistant", content: "Error: " + err.message });
    }
    renderChatLogs(scope);
  }));

/* ---------------- Recomendación ---------------- */
function renderRecommendation(rec, d) {
  $("#rec-header").innerHTML = `
    <div class="item"><span class="k">Enfermedad</span><span class="v red">${rec.enfermedad}</span></div>
    <div class="item"><span class="k">Severidad estimada</span><span class="sev-badge">${rec.severidad}</span></div>
    <div class="item"><span class="k">Estadio fenológico</span><span class="v">${rec.estadio_fenologico || "—"}</span></div>
    <div class="item"><span class="k">Región</span><span class="v">${rec.region || "—"}</span></div>`;
  const cards = $("#rec-cards"); cards.innerHTML = "";
  rec.controles.forEach((c) => {
    const el = document.createElement("div");
    el.className = "rec-card";
    el.innerHTML = `<div class="head">${c.titulo}</div>
      <div class="body">${c.contenido}</div>
      <div class="src">Fuente: ${c.fuente}</div>`;
    cards.appendChild(el);
  });
  $("#rec-traza").textContent = rec.trazabilidad;
}
$("#btn-export").addEventListener("click", () => window.print());

/* ---------------- Historial ---------------- */
async function loadHistory() {
  const list = $("#history-list"); list.innerHTML = "<p class='muted'>Cargando…</p>";
  try {
    const cases = await api("/api/history");
    if (!cases.length) { list.innerHTML = "<p class='muted'>Aún no hay casos analizados.</p>"; return; }
    list.innerHTML = "";
    cases.forEach((c) => {
      const card = document.createElement("div");
      card.className = "history-card";
      const fecha = new Date(c.created_at).toLocaleString("es-PE");
      card.innerHTML = `<img src="${c.imagen_url}" alt="hoja" />
        <div><div class="hc-name">${c.diagnostico}</div>
        <div class="hc-meta">${(c.confianza*100).toFixed(1)}% · ${c.severidad}</div>
        <div class="hc-meta">${fecha}</div></div>`;
      card.addEventListener("click", () => openCase(c.id));
      list.appendChild(card);
    });
  } catch (err) { list.innerHTML = `<p class='muted'>Error: ${err.message}</p>`; }
}

async function openCase(id) {
  loader("Cargando caso…");
  try {
    const c = await api(`/api/history/${id}`);
    state.case = {
      case_id: c.id, diagnostico: c.diagnostico, confianza: c.confianza,
      estado_confianza: c.estado_confianza, probabilidades: c.probabilidades,
      severidad: c.severidad, imagen_url: c.imagen_url, masked_url: c.masked_url,
      heatmap_url: c.heatmap_url,
      overlay_url: c.overlay_url, capa_usada: "features[-1]", alerta_sesgo: false,
      zona_gradcam: "", recomendacion: c.recomendacion, version_modelo: c.version_modelo,
    };
    renderDiagnosis(state.case);
    // Restaurar explicación y chat guardados
    if (c.explicacion && c.explicacion.resumen)
      renderExplanation({ explicacion: c.explicacion, explicacion_disponible: true }, state.case);
    setCaseChat(c.id, (c.mensajes || []).map((m) => ({ role: m.role, content: m.content })));
    navigate("diagnostico");
  } catch (err) { alert("Error: " + err.message); }
  finally { hideLoader(); }
}

/* ---------------- Arranque ---------------- */
if (state.token) enterApp(); else { hide($("#app-view")); show($("#login-view")); }
