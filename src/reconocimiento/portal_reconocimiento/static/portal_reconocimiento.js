document.addEventListener("DOMContentLoaded", () => {
  const TIMEOUT_PREVIEW_MS = 5000;
  const TIMEOUT_INTENTO_MS = 8000;
  const INTERVALO_MIN_MS = 500;

  const chkConsentimiento = document.getElementById("chk-consentimiento");
  const selectCandidato = document.getElementById("select-candidato");
  const selectCamara = document.getElementById("select-camara");
  const btnIniciar = document.getElementById("btn-iniciar");
  const btnDetener = document.getElementById("btn-detener");
  const btnRegistrarIntento = document.getElementById("btn-registrar-intento");
  const btnDescartarIntento = document.getElementById("btn-descartar-intento");

  const videoEl = document.getElementById("video-preview");
  const placeholderEl = document.getElementById("video-placeholder");
  const lblCandidatoActivo = document.getElementById("lbl-candidato-activo");
  const lblUmbralFijo = document.getElementById("lbl-umbral-fijo");
  const badgeDetector = document.getElementById("badge-detector");
  const lblLatencia = document.getElementById("lbl-latencia");

  const doorBox = document.getElementById("door-box");
  const doorIcon = document.getElementById("door-icon");
  const doorState = document.getElementById("door-state");
  const doorReason = document.getElementById("door-reason");
  const lblIdentidad = document.getElementById("lbl-identidad");
  const lblScore = document.getElementById("lbl-score");
  const scoreBar = document.getElementById("score-bar");

  const selectCategoriaReal = document.getElementById("select-categoria-real");
  const inputParticipante = document.getElementById("input-participante");
  const intentoFeedback = document.getElementById("intento-feedback");

  const canvasCaptura = document.createElement("canvas");
  const ctxCaptura = canvasCaptura.getContext("2d");

  let streamActivo = null;
  let camaraActiva = false;
  let deseoCamaraActiva = false;
  let sesionCamaraToken = 0;
  let epochEstado = 0;
  let frameSeq = 0;
  let ultimaLatenciaMs = 200;
  let timerEnvio = null;

  // Control compartido de operación única en vuelo (preview o intento)
  let nextOpId = 0;
  let operacionEnCurso = null; // { id, tipo, abortCtrl, promesa }
  let pausaPreviewPorIntento = false;

  // Estado en memoria del intento pendiente (nunca se persiste en localStorage ni disco)
  let intentoPendiente = null; // { id_intento, blob, candidato_id, categoria_real, participante_id }

  function obtenerParticipanteId() {
    let pid = localStorage.getItem("portal_recon_participante_id");
    if (!pid) {
      const rand = Math.random().toString(36).slice(2, 10);
      pid = `part_${rand}`;
      localStorage.setItem("portal_recon_participante_id", pid);
    }
    return pid;
  }

  inputParticipante.value = obtenerParticipanteId();
  inputParticipante.addEventListener("change", () => {
    const limpio = inputParticipante.value.trim().replace(/[^a-zA-Z0-9_\-]/g, "");
    if (limpio) {
      inputParticipante.value = limpio;
      localStorage.setItem("portal_recon_participante_id", limpio);
    }
  });

  function actualizarInfoCandidato() {
    const opt = selectCandidato.options[selectCandidato.selectedIndex];
    lblCandidatoActivo.textContent = opt ? opt.dataset.nombre : "—";
    lblUmbralFijo.textContent = opt ? `${opt.dataset.umbral}%` : "—";
  }

  function actualizarControlesIntento() {
    const enIntento = operacionEnCurso && operacionEnCurso.tipo === "intento";
    if (intentoPendiente) {
      btnRegistrarIntento.textContent = "Reintentar intento";
      btnRegistrarIntento.disabled = Boolean(enIntento);
      if (btnDescartarIntento) {
        btnDescartarIntento.classList.remove("hidden");
        btnDescartarIntento.disabled = Boolean(enIntento);
      }
    } else {
      btnRegistrarIntento.textContent = "Registrar intento";
      btnRegistrarIntento.disabled = !camaraActiva || Boolean(enIntento);
      if (btnDescartarIntento) {
        btnDescartarIntento.classList.add("hidden");
        btnDescartarIntento.disabled = true;
      }
    }
  }

  function reiniciarResultadoVisual(razon, estadoBadge = "SIN_CAMARA") {
    doorBox.className = "door-box door-closed";
    doorIcon.textContent = "🔒";
    doorState.textContent = "PUERTA CERRADA";
    doorReason.textContent = razon || "Acceso bloqueado.";
    lblIdentidad.textContent = "—";
    lblScore.textContent = "—";
    scoreBar.style.width = "0%";
    badgeDetector.className = "badge badge-neutral";
    badgeDetector.textContent = estadoBadge;
  }

  function actualizarResultadoVisual(data) {
    const det = data.estado_detector || "DESCONOCIDO";
    badgeDetector.textContent = det;
    if (det === "UN_ROSTRO") {
      badgeDetector.className = "badge badge-ok";
    } else if (det === "SIN_ROSTRO") {
      badgeDetector.className = "badge badge-warn";
    } else {
      badgeDetector.className = "badge badge-err";
    }

    lblLatencia.textContent = data.latencia_ms !== undefined ? `Latencia: ${data.latencia_ms} ms` : "Latencia: —";

    if (det !== "UN_ROSTRO") {
      doorBox.className = "door-box door-closed";
      doorIcon.textContent = "🔒";
      doorState.textContent = "PUERTA CERRADA";
      doorReason.textContent = data.motivo || `Estado del detector: ${det}`;
      lblIdentidad.textContent = "—";
      lblScore.textContent = "—";
      scoreBar.style.width = "0%";
      return;
    }

    const abierta = data.decision_puerta === "ABIERTA";
    doorBox.className = abierta ? "door-box door-open" : "door-box door-closed";
    doorIcon.textContent = abierta ? "🚪" : "🔒";
    doorState.textContent = abierta ? "PUERTA ABIERTA" : "PUERTA CERRADA";
    doorReason.textContent = data.motivo || "";

    lblIdentidad.textContent = data.clase_predicha || "—";
    const pct = typeof data.score_pct === "number" ? data.score_pct : 0;
    lblScore.textContent = `${pct.toFixed(1)}%`;
    scoreBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  }

  function aplicarEspejoPreview() {
    const camVal = selectCamara.value;
    if (camVal === "user") {
      videoEl.classList.add("espejo");
    } else {
      videoEl.classList.remove("espejo");
    }
  }

  async function capturarBlobJpeg() {
    if (!camaraActiva || videoEl.readyState < 2 || !videoEl.videoWidth || !videoEl.videoHeight) {
      return null;
    }
    const maxDim = 640;
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const escala = Math.min(1.0, maxDim / Math.max(vw, vh));
    const tw = Math.max(1, Math.round(vw * escala));
    const th = Math.max(1, Math.round(vh * escala));

    canvasCaptura.width = tw;
    canvasCaptura.height = th;
    ctxCaptura.drawImage(videoEl, 0, 0, tw, th);

    return new Promise((resolve) => {
      canvasCaptura.toBlob((blob) => resolve(blob), "image/jpeg", 0.78);
    });
  }

  function programarSiguienteEnvio() {
    if (!camaraActiva || pausaPreviewPorIntento) return;
    clearTimeout(timerEnvio);
    const esperaMs = Math.max(INTERVALO_MIN_MS, Math.round(ultimaLatenciaMs * 1.5));
    timerEnvio = setTimeout(enviarFotogramaPreview, esperaMs);
  }

  async function ejecutarJsonConTimeout(url, options, timeoutMs, abortExt = null) {
    const ctrl = new AbortController();
    let porTimeout = false;
    const timer = setTimeout(() => {
      porTimeout = true;
      ctrl.abort();
    }, timeoutMs);

    const onAbortExt = () => ctrl.abort();
    if (abortExt) {
      if (abortExt.aborted) {
        clearTimeout(timer);
        ctrl.abort();
      } else {
        abortExt.addEventListener("abort", onAbortExt, { once: true });
      }
    }

    const esperarAbort = () =>
      new Promise((_, reject) => {
        if (ctrl.signal.aborted) {
          const err = new Error(porTimeout ? "Tiempo de espera agotado" : "Operación abortada");
          err.name = "AbortError";
          err.esTimeout = porTimeout;
          reject(err);
          return;
        }
        ctrl.signal.addEventListener(
          "abort",
          () => {
            const err = new Error(porTimeout ? "Tiempo de espera agotado" : "Operación abortada");
            err.name = "AbortError";
            err.esTimeout = porTimeout;
            reject(err);
          },
          { once: true }
        );
      });

    try {
      const resp = await fetch(url, { ...options, signal: ctrl.signal });
      const data = await Promise.race([resp.json(), esperarAbort()]);
      return { resp, data };
    } catch (err) {
      if (err && err.name === "AbortError" && err.esTimeout === undefined) {
        err.esTimeout = porTimeout;
      }
      throw err;
    } finally {
      clearTimeout(timer);
      if (abortExt) {
        abortExt.removeEventListener("abort", onAbortExt);
      }
    }
  }

  async function enviarFotogramaPreview() {
    if (!camaraActiva || pausaPreviewPorIntento || operacionEnCurso !== null) {
      programarSiguienteEnvio();
      return;
    }

    // Adquirir el bloqueo compartido ANTES del primer await (incluyendo conversión a JPEG)
    const miOpId = ++nextOpId;
    const abortCtrl = new AbortController();
    const seqLocal = ++frameSeq;
    const epochLocal = epochEstado;
    const tokenCamLocal = sesionCamaraToken;
    const candLocal = selectCandidato.value;

    let resolverPromesa;
    const promesaOp = new Promise((r) => {
      resolverPromesa = r;
    });
    operacionEnCurso = { id: miOpId, tipo: "preview", abortCtrl, promesa: promesaOp };

    try {
      const blob = await capturarBlobJpeg();
      if (!blob || !camaraActiva || epochLocal !== epochEstado || tokenCamLocal !== sesionCamaraToken || abortCtrl.signal.aborted) {
        return;
      }

      const formData = new FormData();
      formData.append("frame", blob, "frame.jpg");
      formData.append("candidato_id", candLocal);
      formData.append("frame_seq", String(seqLocal));

      const t0 = performance.now();
      let resp;
      let data;
      try {
        ({ resp, data } = await ejecutarJsonConTimeout(
          "/api/reconocer",
          { method: "POST", body: formData },
          TIMEOUT_PREVIEW_MS,
          abortCtrl.signal
        ));
      } catch (errFetch) {
        if (!camaraActiva || epochLocal !== epochEstado || tokenCamLocal !== sesionCamaraToken) {
          return;
        }
        if (errFetch && errFetch.name === "AbortError") {
          if (errFetch.esTimeout) {
            reiniciarResultadoVisual("Tiempo de espera agotado al consultar el servidor.", "TIMEOUT");
          }
          return;
        }
        reiniciarResultadoVisual("Error de conexión con el servidor.", "DESCONECTADO");
        return;
      }

      ultimaLatenciaMs = Math.round(performance.now() - t0);

      // Descartar respuestas obsoletas tras detener cámara, cambiar candidato o cambiar de sesión
      if (
        !camaraActiva ||
        epochLocal !== epochEstado ||
        tokenCamLocal !== sesionCamaraToken ||
        candLocal !== selectCandidato.value ||
        seqLocal !== frameSeq
      ) {
        return;
      }

      if (!resp.ok || data.status !== "ok") {
        reiniciarResultadoVisual(data.mensaje || "Servidor ocupado o error al procesar.", "ERROR");
      } else {
        actualizarResultadoVisual(data);
      }
    } catch (err) {
      if (camaraActiva && epochLocal === epochEstado && tokenCamLocal === sesionCamaraToken) {
        reiniciarResultadoVisual("Error al procesar el fotograma.", "ERROR");
      }
    } finally {
      if (operacionEnCurso && operacionEnCurso.id === miOpId) {
        operacionEnCurso = null;
      }
      resolverPromesa();
      programarSiguienteEnvio();
    }
  }

  async function iniciarCamara() {
    if (!chkConsentimiento.checked) return;
    detenerCamara(false);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      reiniciarResultadoVisual("Tu navegador no soporta getUserMedia.", "NO_SOPORTADO");
      return;
    }

    deseoCamaraActiva = true;
    const tokenInicio = ++sesionCamaraToken;
    btnIniciar.disabled = true;
    btnDetener.disabled = false;
    reiniciarResultadoVisual("Solicitando acceso a la cámara...", "SOLICITANDO_CAMARA");

    const camVal = selectCamara.value;
    const videoConstraints = (camVal === "user" || camVal === "environment")
      ? { facingMode: { ideal: camVal }, width: { ideal: 640 }, height: { ideal: 480 } }
      : { deviceId: { exact: camVal }, width: { ideal: 640 }, height: { ideal: 480 } };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });

      // Si el usuario detuvo o canceló mientras getUserMedia estaba pendiente, cerrar el stream sin activar
      if (!deseoCamaraActiva || tokenInicio !== sesionCamaraToken) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      streamActivo = stream;
      stream.getTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          if (camaraActiva && tokenInicio === sesionCamaraToken) {
            detenerCamara(true, "La cámara se desconectó o finalizó inesperadamente.", "DESCONECTADA");
          }
        });
      });

      videoEl.srcObject = streamActivo;
      aplicarEspejoPreview();
      placeholderEl.classList.add("hidden");

      camaraActiva = true;
      epochEstado++;
      btnIniciar.disabled = true;
      btnDetener.disabled = false;
      actualizarControlesIntento();

      reiniciarResultadoVisual("Analizando rostro...", "INICIANDO");
      programarSiguienteEnvio();
    } catch (err) {
      if (tokenInicio === sesionCamaraToken) {
        deseoCamaraActiva = false;
        btnIniciar.disabled = !chkConsentimiento.checked;
        btnDetener.disabled = true;
        reiniciarResultadoVisual(`No se pudo abrir la cámara: ${err.message}`, "ERROR_CAMARA");
      }
    }
  }

  function detenerCamara(actualizarTexto = true, razon = "Cámara detenida.", badge = "SIN_CAMARA") {
    deseoCamaraActiva = false;
    camaraActiva = false;
    sesionCamaraToken++;
    epochEstado++;
    clearTimeout(timerEnvio);

    if (operacionEnCurso && operacionEnCurso.tipo === "preview") {
      operacionEnCurso.abortCtrl.abort();
    }

    if (streamActivo) {
      streamActivo.getTracks().forEach((t) => t.stop());
      streamActivo = null;
    }
    videoEl.srcObject = null;
    placeholderEl.classList.remove("hidden");
    btnIniciar.disabled = !chkConsentimiento.checked;
    btnDetener.disabled = true;
    actualizarControlesIntento();

    if (actualizarTexto) {
      reiniciarResultadoVisual(razon, badge);
      lblLatencia.textContent = "Latencia: —";
    }
  }

  chkConsentimiento.addEventListener("change", () => {
    if (!chkConsentimiento.checked) {
      detenerCamara(true);
    }
    btnIniciar.disabled = !chkConsentimiento.checked || camaraActiva;
  });

  selectCandidato.addEventListener("change", () => {
    epochEstado++;
    if (operacionEnCurso && operacionEnCurso.tipo === "preview") {
      operacionEnCurso.abortCtrl.abort();
    }
    actualizarInfoCandidato();
    if (camaraActiva) {
      reiniciarResultadoVisual("Cambiando de modelo candidato...", "CAMBIANDO");
    }
  });

  selectCamara.addEventListener("change", () => {
    aplicarEspejoPreview();
    if (deseoCamaraActiva || camaraActiva) {
      iniciarCamara();
    }
  });

  btnIniciar.addEventListener("click", iniciarCamara);
  btnDetener.addEventListener("click", () => detenerCamara(true));

  function generarUuid() {
    if (window.crypto && window.crypto.randomUUID) {
      return window.crypto.randomUUID();
    }
    return `int_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  async function manejarRegistroIntento() {
    if (operacionEnCurso && operacionEnCurso.tipo === "intento") {
      return;
    }
    if (!intentoPendiente && !camaraActiva) {
      return;
    }

    // Deshabilitar inmediatamente antes de cualquier await (evita doble clic)
    btnRegistrarIntento.disabled = true;
    if (btnDescartarIntento) btnDescartarIntento.disabled = true;
    pausaPreviewPorIntento = true;
    clearTimeout(timerEnvio);

    // Fijar metadatos del intento ANTES de la captura asíncrona (si es un intento nuevo)
    if (!intentoPendiente) {
      intentoPendiente = {
        id_intento: generarUuid(),
        blob: null,
        candidato_id: selectCandidato.value,
        categoria_real: selectCategoriaReal.value,
        participante_id: inputParticipante.value.trim() || obtenerParticipanteId(),
      };
    }

    // Si hay una vista previa en vuelo, cancelarla de forma controlada y esperar a que libere
    if (operacionEnCurso && operacionEnCurso.tipo === "preview") {
      operacionEnCurso.abortCtrl.abort();
      try {
        await operacionEnCurso.promesa;
      } catch (_) {}
    }

    // Adquirir el bloqueo compartido ANTES del primer await de captura JPEG / fetch
    const miOpId = ++nextOpId;
    const tokenCamLocal = sesionCamaraToken;
    const epochLocal = epochEstado;
    const abortCtrl = new AbortController();
    let resolverPromesa;
    const promesaOp = new Promise((r) => {
      resolverPromesa = r;
    });
    operacionEnCurso = { id: miOpId, tipo: "intento", abortCtrl, promesa: promesaOp };
    actualizarControlesIntento();

    try {
      if (!intentoPendiente.blob) {
        const blob = await capturarBlobJpeg();
        if (!blob) {
          intentoPendiente = null;
          intentoFeedback.textContent = "No hay fotograma disponible para registrar.";
          intentoFeedback.classList.remove("hidden");
          return;
        }
        intentoPendiente.blob = blob;
      }

      const formData = new FormData();
      formData.append("frame", intentoPendiente.blob, "intento.jpg");
      formData.append("candidato_id", intentoPendiente.candidato_id);
      formData.append("categoria_real", intentoPendiente.categoria_real);
      formData.append("participante_id", intentoPendiente.participante_id);
      formData.append("id_intento", intentoPendiente.id_intento);

      const { resp, data } = await ejecutarJsonConTimeout(
        "/api/intento",
        { method: "POST", body: formData },
        TIMEOUT_INTENTO_MS,
        abortCtrl.signal
      );

      if (!resp.ok || data.status !== "ok" || !data.registro) {
        const msgErr = data.mensaje || `Error HTTP ${resp.status}`;
        if (tokenCamLocal === sesionCamaraToken && epochLocal === epochEstado) {
          const badgeErr = resp.status === 503 ? "OCUPADO" : "ERROR";
          reiniciarResultadoVisual(`Registro no confirmado (${msgErr}). Resultado anterior invalidado.`, badgeErr);
        }
        intentoFeedback.textContent =
          `No se confirmó el intento (${msgErr}). ` +
          `Puedes pulsar "Reintentar intento" con el mismo identificador (${intentoPendiente.id_intento.slice(0, 8)}) sin duplicar.`;
        intentoFeedback.classList.remove("hidden");
        return;
      }

      const r = data.registro;
      const notaDup = data.duplicado ? " (recuperado sin duplicar)" : "";
      intentoFeedback.textContent =
        `Intento ${r.id_intento.slice(0, 8)} guardado${notaDup} | Detector: ${r.estado_detector} | ` +
        `Real: ${r.categoria_real} | Pred: ${r.clase_predicha || "—"} (${((r.score || 0) * 100).toFixed(1)}%) | ` +
        `Puerta: ${r.decision_puerta}`;
      intentoFeedback.classList.remove("hidden");
      intentoPendiente = null;
    } catch (err) {
      const esTimeout = Boolean(err && (err.esTimeout || err.name === "AbortError"));
      const detalle = esTimeout ? "Tiempo de espera agotado" : `Error de red (${err.message})`;
      if (tokenCamLocal === sesionCamaraToken && epochLocal === epochEstado) {
        reiniciarResultadoVisual(`${detalle} al registrar el intento. Resultado anterior invalidado.`, esTimeout ? "TIMEOUT" : "ERROR");
      }
      intentoFeedback.textContent =
        `${detalle}. El servidor pudo haber guardado el intento: pulsa "Reintentar intento" para confirmarlo con el mismo ID ` +
        `(${intentoPendiente ? intentoPendiente.id_intento.slice(0, 8) : "—"}) sin duplicarlo.`;
      intentoFeedback.classList.remove("hidden");
    } finally {
      if (operacionEnCurso && operacionEnCurso.id === miOpId) {
        operacionEnCurso = null;
      }
      resolverPromesa();
      pausaPreviewPorIntento = false;
      actualizarControlesIntento();
      programarSiguienteEnvio();
    }
  }

  btnRegistrarIntento.addEventListener("click", manejarRegistroIntento);

  if (btnDescartarIntento) {
    btnDescartarIntento.addEventListener("click", () => {
      if (operacionEnCurso && operacionEnCurso.tipo === "intento") return;
      const idCorto = intentoPendiente ? intentoPendiente.id_intento.slice(0, 8) : "";
      intentoPendiente = null;
      actualizarControlesIntento();
      intentoFeedback.textContent =
        `Intento pendiente (${idCorto}) descartado localmente. Si la solicitud anterior llegó al servidor, ` +
        `el registro podría haberse guardado allí (descartar localmente no elimina un registro del servidor).`;
      intentoFeedback.classList.remove("hidden");
    });
  }

  actualizarInfoCandidato();
  actualizarControlesIntento();
  reiniciarResultadoVisual("Marca el consentimiento e inicia tu cámara.", "SIN_CAMARA");
});
