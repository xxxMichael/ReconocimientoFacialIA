import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const JS_PATH = path.resolve("src/reconocimiento/portal_reconocimiento/static/portal_reconocimiento.js");
const JS_CODE = fs.readFileSync(JS_PATH, "utf-8");

function crearEntornoDOM({ fetchImpl, getUserMediaImpl, toBlobImpl }) {
  const listenersDoc = {};
  const storage = new Map();
  let uuidCounter = 0;

  function makeEl(id, extra = {}) {
    const el = {
      id,
      value: "",
      textContent: "",
      checked: false,
      disabled: false,
      style: {},
      className: "",
      dataset: {},
      options: [],
      selectedIndex: 0,
      readyState: 2,
      videoWidth: 640,
      videoHeight: 480,
      srcObject: null,
      _listeners: {},
      classList: {
        _set: new Set(extra.classes || []),
        add(c) { this._set.add(c); },
        remove(c) { this._set.delete(c); },
        contains(c) { return this._set.has(c); },
      },
      addEventListener(ev, fn) {
        this._listeners[ev] = this._listeners[ev] || [];
        this._listeners[ev].push(fn);
      },
      async trigger(ev) {
        const list = this._listeners[ev] || [];
        for (const fn of list) {
          await fn({ target: this });
        }
      },
      getContext() {
        return { drawImage() {} };
      },
      toBlob(cb) {
        if (toBlobImpl) {
          toBlobImpl(cb);
        } else {
          cb(new Blob(["jpeg-bytes"], { type: "image/jpeg" }));
        }
      },
      ...extra,
    };
    return el;
  }

  const selectCand = makeEl("select-candidato", {
    value: "cnn_antialiasing_052",
    selectedIndex: 0,
    options: [
      { value: "cnn_antialiasing_052", dataset: { nombre: "CNN Antialiasing — 0.52", umbral: "52.0" } },
      { value: "cnn_estandar_portal_061", dataset: { nombre: "CNN Estándar Portal Completo — 0.61", umbral: "61.0" } },
    ],
  });
  const selectCam = makeEl("select-camara", { value: "user" });
  const selectCat = makeEl("select-categoria-real", { value: "persona_1" });

  const elements = {
    "chk-consentimiento": makeEl("chk-consentimiento"),
    "select-candidato": selectCand,
    "select-camara": selectCam,
    "btn-iniciar": makeEl("btn-iniciar", { disabled: true }),
    "btn-detener": makeEl("btn-detener", { disabled: true }),
    "btn-registrar-intento": makeEl("btn-registrar-intento", { disabled: true }),
    "btn-descartar-intento": makeEl("btn-descartar-intento", { disabled: true, classes: ["hidden"] }),
    "video-preview": makeEl("video-preview"),
    "video-placeholder": makeEl("video-placeholder"),
    "lbl-candidato-activo": makeEl("lbl-candidato-activo"),
    "lbl-umbral-fijo": makeEl("lbl-umbral-fijo"),
    "badge-detector": makeEl("badge-detector"),
    "lbl-latencia": makeEl("lbl-latencia"),
    "door-box": makeEl("door-box"),
    "door-icon": makeEl("door-icon"),
    "door-state": makeEl("door-state"),
    "door-reason": makeEl("door-reason"),
    "lbl-identidad": makeEl("lbl-identidad"),
    "lbl-score": makeEl("lbl-score"),
    "score-bar": makeEl("score-bar"),
    "select-categoria-real": selectCat,
    "input-participante": makeEl("input-participante"),
    "intento-feedback": makeEl("intento-feedback", { classes: ["hidden"] }),
  };

  const timers = [];
  const fakeSetTimeout = (fn, ms) => {
    const delay = ms >= 5000 ? 80 : Math.min(ms, 15);
    const id = setTimeout(fn, delay);
    timers.push(id);
    return id;
  };

  const sandbox = {
    console,
    Math,
    Date,
    String,
    Number,
    Boolean,
    Array,
    Object,
    Promise,
    Error,
    AbortController,
    FormData,
    Blob,
    performance: { now: () => Date.now() },
    setTimeout: fakeSetTimeout,
    clearTimeout,
    localStorage: {
      getItem: (k) => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, String(v)),
    },
    window: {
      crypto: {
        randomUUID: () => `uuid-test-${++uuidCounter}`,
      },
    },
    navigator: {
      mediaDevices: {
        getUserMedia: getUserMediaImpl,
      },
    },
    document: {
      addEventListener: (ev, fn) => {
        listenersDoc[ev] = fn;
      },
      getElementById: (id) => elements[id] || null,
      createElement: () => makeEl("canvas-tmp"),
    },
    fetch: fetchImpl,
  };

  vm.createContext(sandbox);
  vm.runInContext(JS_CODE, sandbox);
  listenersDoc["DOMContentLoaded"]();

  return {
    elements,
    limpiarTimers: () => timers.forEach((t) => clearTimeout(t)),
  };
}

function crearFakeTrack() {
  const listeners = {};
  return {
    stopped: false,
    stop() { this.stopped = true; },
    addEventListener(ev, fn) {
      listeners[ev] = listeners[ev] || [];
      listeners[ev].push(fn);
    },
    emitEnded() {
      for (const fn of listeners["ended"] || []) fn();
    },
  };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// 1. Reintento idempotente tras pérdida de respuesta (recupera registro original)
// ---------------------------------------------------------------------------
{
  const guardados = new Map();
  let llamadasIntento = 0;
  const track = crearFakeTrack();

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    fetchImpl: async (url, opts) => {
      if (url === "/api/reconocer") {
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            frame_seq: Number(opts.body.get("frame_seq")),
            candidato_id: opts.body.get("candidato_id"),
            estado_detector: "UN_ROSTRO",
            clase_predicha: "persona_1",
            score_pct: 94.0,
            decision_puerta: "ABIERTA",
            motivo: "Acceso concedido",
            latencia_ms: 15,
          }),
        };
      }
      if (url === "/api/intento") {
        llamadasIntento++;
        const id = opts.body.get("id_intento");
        if (!guardados.has(id)) {
          guardados.set(id, {
            id_intento: id,
            estado_detector: "UN_ROSTRO",
            categoria_real: opts.body.get("categoria_real"),
            clase_predicha: "persona_1",
            score: 0.94,
            decision_puerta: "ABIERTA",
          });
        }
        if (llamadasIntento === 1) {
          // Simular pérdida de respuesta tras guardar en servidor
          throw new Error("Conexión interrumpida");
        }
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            duplicado: true,
            id_intento: id,
            registro: guardados.get(id),
          }),
        };
      }
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");

  await env.elements["btn-registrar-intento"].trigger("click");
  assert.equal(guardados.size, 1);
  assert.equal(env.elements["btn-registrar-intento"].textContent, "Reintentar intento");
  assert.equal(env.elements["btn-descartar-intento"].classList.contains("hidden"), false);

  // Reintentar con el mismo id_intento
  await env.elements["btn-registrar-intento"].trigger("click");
  assert.equal(guardados.size, 1, "No debe crear un segundo registro en el servidor");
  assert.equal(llamadasIntento, 2);
  assert.match(env.elements["intento-feedback"].textContent, /recuperado sin duplicar/);
  assert.equal(env.elements["btn-registrar-intento"].textContent, "Registrar intento");
  env.limpiarTimers();
  console.log("OK test 1: Reintento idempotente tras pérdida de respuesta");
}

// ---------------------------------------------------------------------------
// 2. Doble clic en Registrar intento: una sola captura y un solo envío
// ---------------------------------------------------------------------------
{
  let capturas = 0;
  let enviosIntento = 0;
  const track = crearFakeTrack();
  let resolverIntento;

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    toBlobImpl: (cb) => {
      capturas++;
      setTimeout(() => cb(new Blob(["img"], { type: "image/jpeg" })), 15);
    },
    fetchImpl: (url) => {
      if (url === "/api/intento") {
        enviosIntento++;
        return new Promise((res) => {
          resolverIntento = () =>
            res({
              ok: true,
              json: async () => ({
                status: "ok",
                duplicado: false,
                registro: {
                  id_intento: "uuid-test-1",
                  estado_detector: "UN_ROSTRO",
                  categoria_real: "persona_1",
                  clase_predicha: "persona_1",
                  score: 0.9,
                  decision_puerta: "ABIERTA",
                },
              }),
            });
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({ status: "ok", estado_detector: "SIN_ROSTRO", frame_seq: 1 }) });
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");

  const p1 = env.elements["btn-registrar-intento"].trigger("click");
  assert.equal(env.elements["btn-registrar-intento"].disabled, true, "Debe deshabilitarse antes del primer await");
  const p2 = env.elements["btn-registrar-intento"].trigger("click");
  await sleep(35);
  resolverIntento();
  await Promise.all([p1, p2]);

  assert.equal(capturas, 1);
  assert.equal(enviosIntento, 1);
  env.limpiarTimers();
  console.log("OK test 2: Doble clic produce una sola captura y envío");
}

// ---------------------------------------------------------------------------
// 3. Registro mientras hay vista previa en vuelo: nunca se superponen
// ---------------------------------------------------------------------------
{
  let enVueloAhora = 0;
  let maxConcurrentes = 0;
  const track = crearFakeTrack();

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    fetchImpl: (url, opts) => {
      enVueloAhora++;
      maxConcurrentes = Math.max(maxConcurrentes, enVueloAhora);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          enVueloAhora--;
          if (url === "/api/reconocer") {
            resolve({
              ok: true,
              json: async () => ({
                status: "ok",
                frame_seq: Number(opts.body.get("frame_seq")),
                candidato_id: opts.body.get("candidato_id"),
                estado_detector: "UN_ROSTRO",
                clase_predicha: "persona_1",
                score_pct: 91.0,
                decision_puerta: "ABIERTA",
                latencia_ms: 20,
              }),
            });
          } else {
            resolve({
              ok: true,
              json: async () => ({
                status: "ok",
                duplicado: false,
                registro: {
                  id_intento: opts.body.get("id_intento"),
                  estado_detector: "UN_ROSTRO",
                  categoria_real: "persona_1",
                  clase_predicha: "persona_1",
                  score: 0.91,
                  decision_puerta: "ABIERTA",
                },
              }),
            });
          }
        }, 40);

        opts.signal.addEventListener("abort", () => {
          clearTimeout(timer);
          enVueloAhora--;
          const err = new Error("Aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");
  await sleep(30); // Esperar a que la vista previa esté en vuelo
  await env.elements["btn-registrar-intento"].trigger("click");

  assert.equal(maxConcurrentes, 1, "No deben superponerse vista previa y registro");
  env.limpiarTimers();
  console.log("OK test 3: Sin solicitudes superpuestas entre vista previa y registro");
}

// ---------------------------------------------------------------------------
// 4. Cambio de candidato durante petición: descarta respuesta obsoleta
// ---------------------------------------------------------------------------
{
  const track = crearFakeTrack();
  let resolverPreviewViejo;

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    fetchImpl: (url, opts) => {
      if (url === "/api/reconocer" && opts.body.get("candidato_id") === "cnn_antialiasing_052") {
        return new Promise((resolve) => {
          resolverPreviewViejo = () =>
            resolve({
              ok: true,
              json: async () => ({
                status: "ok",
                frame_seq: Number(opts.body.get("frame_seq")),
                candidato_id: "cnn_antialiasing_052",
                estado_detector: "UN_ROSTRO",
                clase_predicha: "persona_1",
                score_pct: 95.0,
                decision_puerta: "ABIERTA",
                latencia_ms: 10,
              }),
            });
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ status: "ok", estado_detector: "SIN_ROSTRO", frame_seq: 99, candidato_id: "cnn_estandar_portal_061" }),
      });
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");
  await sleep(30);

  // Cambiar candidato mientras el request viejo aún no responde
  env.elements["select-candidato"].value = "cnn_estandar_portal_061";
  env.elements["select-candidato"].selectedIndex = 1;
  await env.elements["select-candidato"].trigger("change");

  if (resolverPreviewViejo) resolverPreviewViejo();
  await sleep(15);

  assert.equal(env.elements["door-state"].textContent, "PUERTA CERRADA");
  assert.equal(env.elements["lbl-identidad"].textContent, "—");
  env.limpiarTimers();
  console.log("OK test 4: Cambio de candidato descarta respuesta obsoleta");
}

// ---------------------------------------------------------------------------
// 5. Timeout / error después de apertura: limpia identidad y cierra puerta
// ---------------------------------------------------------------------------
{
  const track = crearFakeTrack();
  let llamada = 0;

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    fetchImpl: async (url, opts) => {
      llamada++;
      if (llamada === 1) {
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            frame_seq: Number(opts.body.get("frame_seq")),
            candidato_id: opts.body.get("candidato_id"),
            estado_detector: "UN_ROSTRO",
            clase_predicha: "persona_1",
            score_pct: 92.0,
            decision_puerta: "ABIERTA",
            latencia_ms: 10,
          }),
        };
      }
      const err = new Error("Timeout");
      err.name = "AbortError";
      throw err;
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");
  await sleep(30);
  assert.equal(env.elements["door-state"].textContent, "PUERTA ABIERTA");
  assert.equal(env.elements["lbl-identidad"].textContent, "persona_1");

  // Desconexión inesperada de la pista de cámara también debe cerrar puerta y limpiar
  track.emitEnded();
  assert.equal(env.elements["door-state"].textContent, "PUERTA CERRADA");
  assert.equal(env.elements["lbl-identidad"].textContent, "—");
  assert.equal(env.elements["lbl-score"].textContent, "—");
  env.limpiarTimers();
  console.log("OK test 5: Timeout/desconexión limpia identidad y cierra puerta");
}

// ---------------------------------------------------------------------------
// 6. Detener cámara mientras getUserMedia está pendiente: no se reactiva
// ---------------------------------------------------------------------------
{
  const track = crearFakeTrack();
  let resolverCamara;

  const env = crearEntornoDOM({
    getUserMediaImpl: () =>
      new Promise((res) => {
        resolverCamara = () => res({ getTracks: () => [track] });
      }),
    fetchImpl: async () => ({ ok: true, json: async () => ({ status: "ok" }) }),
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");

  const pInit = env.elements["btn-iniciar"].trigger("click");
  // Verificar que Detener está habilitado mientras getUserMedia está pendiente
  assert.equal(env.elements["btn-iniciar"].disabled, true);
  assert.equal(env.elements["btn-detener"].disabled, false, "Detener debe estar habilitado mientras getUserMedia espera permiso");
  assert.equal(env.elements["door-reason"].textContent, "Solicitando acceso a la cámara...");

  await env.elements["btn-detener"].trigger("click");
  assert.equal(env.elements["btn-iniciar"].disabled, false);
  assert.equal(env.elements["btn-detener"].disabled, true);
  resolverCamara();
  await pInit;

  assert.equal(track.stopped, true, "Debe detener inmediatamente las pistas si el usuario canceló antes");
  assert.equal(env.elements["video-preview"].srcObject, null);
  assert.equal(env.elements["door-state"].textContent, "PUERTA CERRADA");
  env.limpiarTimers();
  console.log("OK test 6: Detener habilitado durante getUserMedia pendiente y cancelación limpia");
}

// ---------------------------------------------------------------------------
// 7. Timeout cubre toda la respuesta (cabeceras inmediatas y resp.json() atascado)
// ---------------------------------------------------------------------------
{
  const track = crearFakeTrack();
  let llamada = 0;
  let resolverCall3 = null;

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    fetchImpl: async () => {
      llamada++;
      if (llamada === 1) {
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            frame_seq: 1,
            candidato_id: "cnn_antialiasing_052",
            estado_detector: "UN_ROSTRO",
            clase_predicha: "persona_1",
            score_pct: 93.0,
            decision_puerta: "ABIERTA",
            latencia_ms: 10,
          }),
        };
      }
      if (llamada === 2) {
        // Cabeceras inmediatas, pero resp.json() nunca termina
        return {
          ok: true,
          json: () => new Promise(() => {}),
        };
      }
      return new Promise((resolve) => {
        resolverCall3 = () =>
          resolve({
            ok: true,
            json: async () => ({
              status: "ok",
              frame_seq: llamada,
              candidato_id: "cnn_antialiasing_052",
              estado_detector: "UN_ROSTRO",
              clase_predicha: "persona_2",
              score_pct: 88.0,
              decision_puerta: "ABIERTA",
              latencia_ms: 10,
            }),
          });
      });
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");
  await sleep(30);
  assert.equal(env.elements["door-state"].textContent, "PUERTA ABIERTA");

  // Esperar a que la llamada 2 atasque en resp.json() y venza el timeout (~80 ms)
  for (let i = 0; i < 20 && env.elements["badge-detector"].textContent !== "TIMEOUT"; i++) {
    await sleep(15);
  }
  assert.equal(env.elements["door-state"].textContent, "PUERTA CERRADA");
  assert.equal(env.elements["lbl-identidad"].textContent, "—");
  assert.equal(env.elements["badge-detector"].textContent, "TIMEOUT");

  // Esperar a que arranque la llamada 3 (demuestra que el bloqueo se liberó) y resolverla
  for (let i = 0; i < 10 && !resolverCall3; i++) {
    await sleep(15);
  }
  assert.ok(resolverCall3, "Debe haber iniciado la siguiente solicitud tras liberar el bloqueo");
  resolverCall3();
  await sleep(15);
  assert.equal(env.elements["lbl-identidad"].textContent, "persona_2");
  env.limpiarTimers();
  console.log("OK test 7: Timeout cubre resp.json() atascado y libera el bloqueo");
}

// ---------------------------------------------------------------------------
// 8. Error 503 y timeout al registrar intento limpian puerta abierta y permiten reintentar
// ---------------------------------------------------------------------------
{
  const track = crearFakeTrack();
  let intentoCall = 0;
  const idsVistos = [];

  const env = crearEntornoDOM({
    getUserMediaImpl: async () => ({ getTracks: () => [track] }),
    fetchImpl: async (url, opts) => {
      if (url === "/api/reconocer") {
        return {
          ok: true,
          json: async () => ({
            status: "ok",
            frame_seq: Number(opts.body.get("frame_seq")),
            candidato_id: "cnn_antialiasing_052",
            estado_detector: "UN_ROSTRO",
            clase_predicha: "persona_1",
            score_pct: 94.0,
            decision_puerta: "ABIERTA",
            latencia_ms: 10,
          }),
        };
      }
      if (url === "/api/intento") {
        intentoCall++;
        idsVistos.push(opts.body.get("id_intento"));
        if (intentoCall === 1) {
          return {
            ok: false,
            status: 503,
            json: async () => ({ status: "ocupado", mensaje: "Servidor ocupado" }),
          };
        }
        if (intentoCall === 2) {
          return {
            ok: true,
            status: 200,
            json: () => new Promise(() => {}), // cuerpo atascado -> timeout
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: "ok",
            duplicado: false,
            registro: {
              id_intento: opts.body.get("id_intento"),
              estado_detector: "UN_ROSTRO",
              categoria_real: opts.body.get("categoria_real"),
              clase_predicha: "persona_1",
              score: 0.94,
              decision_puerta: "ABIERTA",
            },
          }),
        };
      }
    },
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");
  await sleep(25);
  assert.equal(env.elements["door-state"].textContent, "PUERTA ABIERTA");
  assert.equal(env.elements["lbl-identidad"].textContent, "persona_1");

  // 1er envío: 503 ocupado -> cierra puerta, limpia identidad/score, mantiene intentoPendiente
  await env.elements["btn-registrar-intento"].trigger("click");
  assert.equal(env.elements["door-state"].textContent, "PUERTA CERRADA");
  assert.equal(env.elements["lbl-identidad"].textContent, "—");
  assert.equal(env.elements["lbl-score"].textContent, "—");
  assert.equal(env.elements["badge-detector"].textContent, "OCUPADO");
  assert.equal(env.elements["btn-registrar-intento"].textContent, "Reintentar intento");

  // 2do envío (reintento): timeout en resp.json() -> sigue cerrada y limpia, conserva mismo ID
  await env.elements["btn-registrar-intento"].trigger("click");
  assert.equal(env.elements["door-state"].textContent, "PUERTA CERRADA");
  assert.equal(env.elements["lbl-identidad"].textContent, "—");
  assert.equal(env.elements["badge-detector"].textContent, "TIMEOUT");

  // 3er envío (reintento): éxito con el mismo id_intento
  await env.elements["btn-registrar-intento"].trigger("click");
  assert.equal(new Set(idsVistos).size, 1, "Los tres intentos deben compartir el mismo id_intento");
  assert.equal(env.elements["btn-registrar-intento"].textContent, "Registrar intento");
  env.limpiarTimers();
  console.log("OK test 8: 503 y timeout al registrar limpian puerta abierta y conservan reintento");
}

// ---------------------------------------------------------------------------
// 9. Permiso de cámara rechazado restaura controles para otro intento
// ---------------------------------------------------------------------------
{
  const env = crearEntornoDOM({
    getUserMediaImpl: async () => {
      throw new Error("Permission denied");
    },
    fetchImpl: async () => ({ ok: true, json: async () => ({ status: "ok" }) }),
  });

  env.elements["chk-consentimiento"].checked = true;
  await env.elements["chk-consentimiento"].trigger("change");
  await env.elements["btn-iniciar"].trigger("click");

  assert.equal(env.elements["btn-iniciar"].disabled, false);
  assert.equal(env.elements["btn-detener"].disabled, true);
  assert.equal(env.elements["badge-detector"].textContent, "ERROR_CAMARA");
  env.limpiarTimers();
  console.log("OK test 9: Rechazo de permiso restaura botones correctamente");
}
