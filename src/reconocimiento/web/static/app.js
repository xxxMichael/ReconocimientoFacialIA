/**
 * Lógica del cliente web para adquisición facial, simulación de acceso y revisión de capturas.
 */
document.addEventListener("DOMContentLoaded", () => {
  // Elementos DOM - Video y Conexión
  const videoStream = document.getElementById("video-stream");
  const videoOverlay = document.getElementById("video-overlay-msg");
  const overlayText = document.getElementById("overlay-text");
  const badgeConexion = document.getElementById("badge-conexion");
  const badgeDetector = document.getElementById("badge-detector");
  const badgeFps = document.getElementById("badge-fps");
  const lblFrameId = document.getElementById("lbl-frame-id");
  const lblResolucion = document.getElementById("lbl-resolucion");
  const lblTotalRostros = document.getElementById("lbl-total-rostros");

  // Elementos DOM - Pestañas de Navegación Principal
  const tabNavCamara = document.getElementById("tab-nav-camara");
  const tabNavRevision = document.getElementById("tab-nav-revision");
  const mainLayout = document.getElementById("main-layout");
  const revisionLayout = document.getElementById("revision-layout");
  const modeSwitchGroup = document.getElementById("mode-switch-group");

  // Elementos DOM - Alternancia de Vista en Cámara (Entrenamiento vs Producción)
  const btnModoEntrenamiento = document.getElementById("btn-modo-entrenamiento");
  const btnModoProduccion = document.getElementById("btn-modo-produccion");

  // Elementos DOM - Cámara
  const selectCamara = document.getElementById("select-camara");
  const btnSondear = document.getElementById("btn-sondear");
  const btnIniciarCam = document.getElementById("btn-iniciar-cam");
  const btnDetenerCam = document.getElementById("btn-detener-cam");

  // Elementos DOM - Configuración de Sesión
  const radioEnsayo = document.getElementById("radio-ensayo");
  const radioReal = document.getElementById("radio-real");
  const labelEnsayo = document.getElementById("label-ensayo");
  const labelReal = document.getElementById("label-real");
  const modeTag = document.getElementById("mode-tag");
  const selectPersona = document.getElementById("select-persona");
  const selectTipoSujeto = document.getElementById("select-tipo-sujeto");
  const selectSesion = document.getElementById("select-sesion");
  const selectParticion = document.getElementById("select-particion");
  const chkConsentimiento = document.getElementById("chk-consentimiento");
  const poseButtonsContainer = document.getElementById("pose-buttons-container");

  // Elementos DOM - Acciones de Captura
  const btnCapturaValida = document.getElementById("btn-captura-valida");
  const btnCapturaFallo = document.getElementById("btn-captura-fallo");
  const alertBanner = document.getElementById("alert-banner");
  const alertMsg = document.getElementById("alert-msg");

  // Elementos DOM - Simulación de Puerta y Control de Acceso
  const badgePuerta = document.getElementById("badge-puerta");
  const doorGraphic = document.getElementById("door-graphic");
  const doorIcon = document.getElementById("door-icon");
  const doorStateText = document.getElementById("door-state-text");
  const accessMsg = document.getElementById("access-msg");
  const lblPersonaIdentificada = document.getElementById("lbl-persona-identificada");
  const lblConfianzaReconocimiento = document.getElementById("lbl-confianza-reconocimiento");
  const confidenceBar = document.getElementById("confidence-bar");
  const selectModeloRecon = document.getElementById("select-modelo-recon");
  const btnCambiarModelo = document.getElementById("btn-cambiar-modelo");
  const sliderUmbral = document.getElementById("slider-umbral");
  const lblUmbralValor = document.getElementById("lbl-umbral-valor");

  // Elementos DOM - Progreso
  const lblSesionActual = document.getElementById("lbl-sesion-actual");
  const lblPersonaActual = document.getElementById("lbl-persona-actual");
  const lblFallosSesion = document.getElementById("lbl-fallos-sesion");
  const posesProgressContainer = document.getElementById("poses-progress-container");

  // Elementos DOM - Modal de Fotograma Congelado (Requisito 4)
  const modalCongelar = document.getElementById("modal-congelar");
  const btnCongelarCerrarX = document.getElementById("btn-congelar-cerrar-x");
  const btnCongelarDescartar = document.getElementById("btn-congelar-descartar");
  const btnCongelarCorregir = document.getElementById("btn-congelar-corregir");
  const btnCongelarConfirmar = document.getElementById("btn-congelar-confirmar");
  const canvasCongelado = document.getElementById("canvas-congelado");
  const imgCongeladaRecorte = document.getElementById("img-congelada-recorte");
  const lblCongeladoFrame = document.getElementById("lbl-congelado-frame");
  const lblCongeladoEstado = document.getElementById("lbl-congelado-estado");
  const lblCongeladoPose = document.getElementById("lbl-congelado-pose");
  const lblCongeladoPersona = document.getElementById("lbl-congelado-persona");
  const lblCongeladoAvisoDibujo = document.getElementById("lbl-congelado-aviso-dibujo");

  // Elementos DOM - Pestaña Revisar Capturas (Requisitos 1 y 2)
  const revFilterPersona = document.getElementById("rev-filter-persona");
  const revFilterSesion = document.getElementById("rev-filter-sesion");
  const revFilterPose = document.getElementById("rev-filter-pose");
  const revFilterParticion = document.getElementById("rev-filter-particion");
  const revFilterDecision = document.getElementById("rev-filter-decision");
  const revFilterEntorno = document.getElementById("rev-filter-entorno");
  const btnRevRecargar = document.getElementById("btn-rev-recargar");
  const lblRevTotal = document.getElementById("lbl-rev-total");
  const lblRevResumenEstado = document.getElementById("lbl-rev-resumen-estado");
  const revItemsContainer = document.getElementById("rev-items-container");

  const lblInspectorId = document.getElementById("lbl-inspector-id");
  const badgeInspectorDecision = document.getElementById("badge-inspector-decision");
  const badgeInspectorDetector = document.getElementById("badge-inspector-detector");
  const badgeInspectorOrigen = document.getElementById("badge-inspector-origen");
  const btnRevAnterior = document.getElementById("btn-rev-anterior");
  const btnRevSiguiente = document.getElementById("btn-rev-siguiente");
  const bannerRevRecalculada = document.getElementById("banner-rev-recalculada");
  const canvasInspector = document.getElementById("canvas-inspector");
  const imgInspectorRecorte = document.getElementById("img-inspector-recorte");
  const inpBoxX = document.getElementById("inp-box-x");
  const inpBoxY = document.getElementById("inp-box-y");
  const inpBoxW = document.getElementById("inp-box-w");
  const inpBoxH = document.getElementById("inp-box-h");

  const btnDecidirPendiente = document.getElementById("btn-decidir-pendiente");
  const btnDecidirExcluir = document.getElementById("btn-decidir-excluir");
  const btnDecidirCorregir = document.getElementById("btn-decidir-corregir");
  const btnDecidirAprobar = document.getElementById("btn-decidir-aprobar");

  // Estado local
  let currentFrameId = 0;
  let currentFrameEstado = "SIN_CAMARA";
  let cameraActive = false;
  let currentPose = "frontal";
  let clientFpsCount = 0;
  let clientFps = 0.0;
  let lastFpsCalcTime = performance.now();
  let inFlightCapture = false;

  // Estado del fotograma congelado
  let frozenData = null;
  let frozenRawImg = null;
  let frozenCurrentBox = null;
  let frozenDrawingMode = false;
  let frozenIsMouseDown = false;
  let frozenStartX = 0;
  let frozenStartY = 0;

  // Estado de la revisión
  let revisionItems = [];
  let currentRevisionIndex = -1;
  let currentRevisionData = null;
  let inspectorRawImg = null;
  let inspectorCurrentBox = null;
  let inspectorIsMouseDown = false;
  let inspectorStartX = 0;
  let inspectorStartY = 0;

  // =========================================================================
  // 1. INICIALIZACIÓN
  // =========================================================================
  sondearCamaras();
  actualizarModoVisual();
  cargarProgresoSesion();
  iniciarBucleVideo();
  configurarAlternanciaVistas();
  configurarPestanasNavegacion();
  configurarAtajoEspacio();
  configurarEventosRevision();
  configurarEventosCongelado();

  // =========================================================================
  // 2. NAVEGACIÓN ENTRE PESTAÑAS (CAPTURA vs REVISIÓN)
  // =========================================================================
  function configurarPestanasNavegacion() {
    if (!tabNavCamara || !tabNavRevision) return;

    tabNavCamara.addEventListener("click", () => {
      tabNavCamara.classList.add("active");
      tabNavRevision.classList.remove("active");
      mainLayout.classList.remove("hidden");
      if (modeSwitchGroup) modeSwitchGroup.style.display = "flex";
      revisionLayout.classList.add("hidden");
    });

    tabNavRevision.addEventListener("click", () => {
      tabNavRevision.classList.add("active");
      tabNavCamara.classList.remove("active");
      mainLayout.classList.add("hidden");
      if (modeSwitchGroup) modeSwitchGroup.style.display = "none";
      revisionLayout.classList.remove("hidden");
      cargarListaRevision();
    });
  }

  // =========================================================================
  // 3. VISTAS DE CÁMARA (ENTRENAMIENTO vs PRODUCCIÓN)
  // =========================================================================
  function configurarAlternanciaVistas() {
    if (!btnModoEntrenamiento || !btnModoProduccion || !mainLayout) return;

    btnModoEntrenamiento.addEventListener("click", () => setModoVista("entrenamiento"));
    btnModoProduccion.addEventListener("click", () => setModoVista("produccion"));

    const modoGuardado = localStorage.getItem("vista_activa_reconocimiento") || "entrenamiento";
    setModoVista(modoGuardado);
  }

  function setModoVista(modo) {
    if (modo === "produccion") {
      mainLayout.classList.remove("mode-entrenamiento");
      mainLayout.classList.add("mode-produccion");
      btnModoProduccion.classList.add("active");
      btnModoEntrenamiento.classList.remove("active");
      localStorage.setItem("vista_activa_reconocimiento", "produccion");
    } else {
      mainLayout.classList.remove("mode-produccion");
      mainLayout.classList.add("mode-entrenamiento");
      btnModoEntrenamiento.classList.add("active");
      btnModoProduccion.classList.remove("active");
      localStorage.setItem("vista_activa_reconocimiento", "entrenamiento");
    }
  }

  // =========================================================================
  // 4. ATAJO ESPACIO Y CONGELAMIENTO (REQUISITO 4)
  // =========================================================================
  function configurarAtajoEspacio() {
    document.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        // Si el modal de congelado está abierto, confirmar con espacio
        if (modalCongelar && !modalCongelar.classList.contains("hidden")) {
          e.preventDefault();
          if (btnCongelarConfirmar) btnCongelarConfirmar.click();
          return;
        }

        // Si estamos en la pestaña de cámara activa y el botón de captura está habilitado
        if (tabNavCamara && tabNavCamara.classList.contains("active")) {
          if (btnCapturaValida && !btnCapturaValida.disabled && !inFlightCapture) {
            e.preventDefault();
            abrirModalCongelado(currentFrameId);
          }
        }
      }
    });
  }

  if (btnCapturaValida) {
    btnCapturaValida.addEventListener("click", () => {
      abrirModalCongelado(currentFrameId);
    });
  }

  if (btnCapturaFallo) {
    btnCapturaFallo.addEventListener("click", () => {
      ejecutarCapturaDirecta("fallo");
    });
  }

  async function abrirModalCongelado(fid) {
    if (inFlightCapture) return;
    inFlightCapture = true;
    mostrarAlerta(`Congelando fotograma #${fid} para inspección...`, "info");

    try {
      const resp = await fetch("/api/captura/congelar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id_fotograma: fid })
      });

      if (!resp.ok) {
        mostrarAlerta("No se pudo congelar el fotograma (cámara detenida o fotograma caducado).", "warning");
        inFlightCapture = false;
        return;
      }

      frozenData = await resp.json();
      lblCongeladoFrame.textContent = "#" + frozenData.id_fotograma;
      lblCongeladoEstado.textContent = frozenData.estado;
      lblCongeladoPose.textContent = currentPose;
      lblCongeladoPersona.textContent = selectPersona.value;

      frozenCurrentBox = frozenData.recuadro ? [...frozenData.recuadro] : null;
      frozenDrawingMode = false;
      if (lblCongeladoAvisoDibujo) lblCongeladoAvisoDibujo.style.display = "none";

      if (frozenData.recorte_base64) {
        imgCongeladaRecorte.src = "data:image/jpeg;base64," + frozenData.recorte_base64;
        imgCongeladaRecorte.style.display = "inline-block";
      } else {
        imgCongeladaRecorte.src = "";
        imgCongeladaRecorte.style.display = "none";
      }

      // Cargar imagen congelada y pintar en canvas
      frozenRawImg = new Image();
      frozenRawImg.onload = () => {
        modalCongelar.classList.remove("hidden");
        dibujarCanvasCongelado();
      };
      frozenRawImg.onerror = () => {
        modalCongelar.classList.remove("hidden");
        mostrarAlerta("Error al cargar imagen en el visor.", "danger");
      };
      frozenRawImg.src = "data:image/jpeg;base64," + frozenData.image_base64;
    } catch (err) {
      mostrarAlerta("Error al congelar fotograma: " + err.message, "danger");
    } finally {
      inFlightCapture = false;
    }
  }

  function configurarEventosCongelado() {
    if (!modalCongelar) return;

    if (btnCongelarCerrarX) {
      btnCongelarCerrarX.addEventListener("click", cerrarModalCongelado);
    }
    if (btnCongelarDescartar) {
      btnCongelarDescartar.addEventListener("click", cerrarModalCongelado);
    }

    if (btnCongelarCorregir) {
      btnCongelarCorregir.addEventListener("click", () => {
        frozenDrawingMode = true;
        if (lblCongeladoAvisoDibujo) lblCongeladoAvisoDibujo.style.display = "block";
        mostrarAlerta("Arrastra con el ratón sobre la imagen para dibujar el recuadro correcto.", "info");
      });
    }

    if (btnCongelarConfirmar) {
      btnCongelarConfirmar.addEventListener("click", async () => {
        if (!frozenData) return;
        const fid = frozenData.id_fotograma;
        const box = frozenCurrentBox;
        const fueCorregida = (frozenDrawingMode && box !== null);

        cerrarModalCongelado();

        await ejecutarCapturaDirecta("valida", {
          id_fotograma: fid,
          caja: box,
          decision: fueCorregida ? "corregida" : "aprobada",
          origen_caja: fueCorregida ? "manual" : "vivo"
        });
      });
    }

    // Dibujo en canvas congelado
    if (canvasCongelado) {
      canvasCongelado.addEventListener("mousedown", (e) => {
        if (!frozenDrawingMode && frozenCurrentBox !== null) return;
        frozenIsMouseDown = true;
        const pos = getCanvasPos(canvasCongelado, e);
        frozenStartX = pos.x;
        frozenStartY = pos.y;
        frozenCurrentBox = [pos.x, pos.y, 1, 1];
        dibujarCanvasCongelado();
      });

      canvasCongelado.addEventListener("mousemove", (e) => {
        if (!frozenIsMouseDown) return;
        const pos = getCanvasPos(canvasCongelado, e);
        const x = Math.min(frozenStartX, pos.x);
        const y = Math.min(frozenStartY, pos.y);
        const w = Math.max(10, Math.abs(pos.x - frozenStartX));
        const h = Math.max(10, Math.abs(pos.y - frozenStartY));
        frozenCurrentBox = [x, y, w, h];
        dibujarCanvasCongelado();
      });

      canvasCongelado.addEventListener("mouseup", () => {
        if (!frozenIsMouseDown) return;
        frozenIsMouseDown = false;
        actualizarRecorteDinamicoCliente(frozenRawImg, frozenCurrentBox, imgCongeladaRecorte);
      });
    }
  }

  function cerrarModalCongelado() {
    if (modalCongelar) modalCongelar.classList.add("hidden");
    frozenData = null;
    frozenRawImg = null;
    if (imgCongeladaRecorte) {
      imgCongeladaRecorte.src = "";
      imgCongeladaRecorte.style.display = "none";
    }
  }

  function dibujarCanvasCongelado() {
    if (!canvasCongelado || !frozenRawImg) return;
    canvasCongelado.width = frozenRawImg.naturalWidth;
    canvasCongelado.height = frozenRawImg.naturalHeight;
    const ctx = canvasCongelado.getContext("2d");
    ctx.drawImage(frozenRawImg, 0, 0);

    if (frozenCurrentBox && frozenCurrentBox.length === 4) {
      const [x, y, w, h] = frozenCurrentBox;
      ctx.strokeStyle = frozenDrawingMode ? "#388bfd" : "#2ea043";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = frozenDrawingMode ? "rgba(56, 139, 253, 0.15)" : "rgba(46, 160, 67, 0.15)";
      ctx.fillRect(x, y, w, h);

      ctx.fillStyle = frozenDrawingMode ? "#388bfd" : "#2ea043";
      ctx.font = "bold 14px sans-serif";
      const txt = frozenDrawingMode ? "Caja Corregida" : "Rostro Identificado";
      ctx.fillText(txt, x + 4, Math.max(18, y - 6));
    }
  }

  // =========================================================================
  // 5. CAPTURA DIRECTA Y GUARDADO ATÓMICO
  // =========================================================================
  async function ejecutarCapturaDirecta(modo, opcionesExtra = {}) {
    if (inFlightCapture) return;
    inFlightCapture = true;
    actualizarEstadoBotones();

    const fid = opcionesExtra.id_fotograma || currentFrameId;
    const esEnsayo = radioEnsayo.checked;
    const persona = selectPersona.value;
    const tipoSujeto = selectTipoSujeto.value;
    const sesion = selectSesion.value;
    const pose = currentPose;
    const particion = selectParticion.value;
    const consentimiento = true;
    const idempotencyKey = `cap_${persona}_${sesion}_${pose}_${fid}_${Date.now()}`;

    mostrarAlerta(`Guardando captura #${fid} de forma coherente...`, "info");

    try {
      const payload = {
        id_fotograma: fid,
        modo: modo,
        persona: persona,
        tipo_sujeto: tipoSujeto,
        sesion: sesion,
        pose: pose,
        particion: particion,
        consentimiento: consentimiento,
        es_ensayo: esEnsayo,
        idempotency_key: idempotencyKey,
        caja: opcionesExtra.caja || null,
        decision: opcionesExtra.decision || (modo === "valida" ? "aprobada" : "pendiente"),
        origen_caja: opcionesExtra.origen_caja || null
      };

      const resp = await fetch("/api/capturar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await resp.json();

      if (!resp.ok) {
        if (resp.status === 410) {
          mostrarAlerta("Fotograma caducado en búfer. Vuelva a intentar.", "warning");
        } else {
          mostrarAlerta("Error al capturar: " + (data.mensaje || "Rechazado"), "danger");
        }
      } else {
        const tag = (modo === "valida") ? "✓ Captura Válida" : "⚠ Fallo Registrado";
        mostrarAlerta(`${tag} guardada como ${data.identificador} (${data.decision || "aprobada"}).`, "success");
        if (data.progreso_sesion) {
          renderizarProgreso(data.progreso_sesion);
        } else {
          cargarProgresoSesion();
        }
      }
    } catch (err) {
      mostrarAlerta("Fallo de red: " + err.message, "danger");
    } finally {
      inFlightCapture = false;
      actualizarEstadoBotones();
    }
  }

  // =========================================================================
  // 6. PESTAÑA: REVISAR CAPTURAS (REQUISITOS 1, 2 Y 3)
  // =========================================================================
  function configurarEventosRevision() {
    if (btnRevRecargar) {
      btnRevRecargar.addEventListener("click", cargarListaRevision);
    }
    if (revFilterEntorno) {
      revFilterEntorno.addEventListener("change", cargarListaRevision);
    }

    if (btnRevAnterior) {
      btnRevAnterior.addEventListener("click", () => {
        if (currentRevisionIndex > 0) {
          seleccionarCapturaPorIndice(currentRevisionIndex - 1);
        }
      });
    }

    if (btnRevSiguiente) {
      btnRevSiguiente.addEventListener("click", () => {
        if (currentRevisionIndex < revisionItems.length - 1) {
          seleccionarCapturaPorIndice(currentRevisionIndex + 1);
        }
      });
    }

    // Botones de Decisión
    if (btnDecidirAprobar) {
      btnDecidirAprobar.addEventListener("click", () => aplicarDecisionRevision("aprobada"));
    }
    if (btnDecidirCorregir) {
      btnDecidirCorregir.addEventListener("click", () => aplicarDecisionRevision("corregida"));
    }
    if (btnDecidirExcluir) {
      btnDecidirExcluir.addEventListener("click", () => aplicarDecisionRevision("excluida"));
    }
    if (btnDecidirPendiente) {
      btnDecidirPendiente.addEventListener("click", () => aplicarDecisionRevision("pendiente"));
    }

    // Formulario de coordenadas
    [inpBoxX, inpBoxY, inpBoxW, inpBoxH].forEach(inp => {
      if (inp) {
        inp.addEventListener("change", () => {
          if (!inspectorCurrentBox) inspectorCurrentBox = [0, 0, 100, 100];
          inspectorCurrentBox[0] = parseInt(inpBoxX.value) || 0;
          inspectorCurrentBox[1] = parseInt(inpBoxY.value) || 0;
          inspectorCurrentBox[2] = parseInt(inpBoxW.value) || 10;
          inspectorCurrentBox[3] = parseInt(inpBoxH.value) || 10;
          dibujarCanvasInspector();
          actualizarRecorteDinamicoCliente(inspectorRawImg, inspectorCurrentBox, imgInspectorRecorte);
        });
      }
    });

    // Dibujo en canvas inspector
    if (canvasInspector) {
      canvasInspector.addEventListener("mousedown", (e) => {
        inspectorIsMouseDown = true;
        const pos = getCanvasPos(canvasInspector, e);
        inspectorStartX = pos.x;
        inspectorStartY = pos.y;
        inspectorCurrentBox = [pos.x, pos.y, 1, 1];
        actualizarInputsCoordenadas();
        dibujarCanvasInspector();
      });

      canvasInspector.addEventListener("mousemove", (e) => {
        if (!inspectorIsMouseDown) return;
        const pos = getCanvasPos(canvasInspector, e);
        const x = Math.min(inspectorStartX, pos.x);
        const y = Math.min(inspectorStartY, pos.y);
        const w = Math.max(10, Math.abs(pos.x - inspectorStartX));
        const h = Math.max(10, Math.abs(pos.y - inspectorStartY));
        inspectorCurrentBox = [x, y, w, h];
        actualizarInputsCoordenadas();
        dibujarCanvasInspector();
      });

      canvasInspector.addEventListener("mouseup", () => {
        if (!inspectorIsMouseDown) return;
        inspectorIsMouseDown = false;
        actualizarRecorteDinamicoCliente(inspectorRawImg, inspectorCurrentBox, imgInspectorRecorte);
      });
    }
  }

  async function cargarListaRevision() {
    if (!revItemsContainer) return;

    const persona = revFilterPersona?.value || "todas";
    const sesion = revFilterSesion?.value || "todas";
    const pose = revFilterPose?.value || "todas";
    const particion = revFilterParticion?.value || "todas";
    const decision = revFilterDecision?.value || "todas";
    const esEnsayo = (revFilterEntorno?.value === "ensayo");

    revItemsContainer.innerHTML = `<div style="padding: 16px; color: var(--text-muted);">Cargando capturas...</div>`;

    try {
      const url = `/api/revision/capturas?es_ensayo=${esEnsayo}&persona=${persona}&sesion=${sesion}&pose=${pose}&particion=${particion}&decision=${decision}`;
      const resp = await fetch(url);
      const data = await resp.json();

      revisionItems = data.capturas || [];
      if (lblRevTotal) lblRevTotal.textContent = revisionItems.length;

      const pendientesCount = revisionItems.filter(i => i.decision_humana === "pendiente").length;
      if (lblRevResumenEstado) lblRevResumenEstado.textContent = `${pendientesCount} pendientes`;

      if (revisionItems.length === 0) {
        revItemsContainer.innerHTML = `<div style="padding: 20px; color: var(--text-muted); text-align: center;">No se encontraron capturas con los filtros seleccionados.</div>`;
        limpiarInspector();
        return;
      }

      revItemsContainer.innerHTML = "";
      revisionItems.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = "revision-item-card" + (idx === 0 ? " selected" : "");
        card.id = `rev-card-${idx}`;

        const badgeClass = `badge-decision-${item.decision_humana || "pendiente"}`;
        const decLabel = (item.decision_humana || "pendiente").toUpperCase();

        card.innerHTML = `
          <div class="revision-item-top">
            <span>${item.identificador}</span>
            <span class="${badgeClass}">${decLabel}</span>
          </div>
          <div class="revision-item-sub">
            <span>${item.pose}</span> •
            <span>${item.sesion}</span> •
            <span>${item.particion}</span>
          </div>
        `;

        card.addEventListener("click", () => seleccionarCapturaPorIndice(idx));
        revItemsContainer.appendChild(card);
      });

      seleccionarCapturaPorIndice(0);
    } catch (err) {
      revItemsContainer.innerHTML = `<div style="padding: 16px; color: var(--color-danger);">Error: ${err.message}</div>`;
    }
  }

  function seleccionarCapturaPorIndice(idx) {
    if (idx < 0 || idx >= revisionItems.length) return;
    currentRevisionIndex = idx;

    // Actualizar clase selected en sidebar
    const allCards = revItemsContainer.querySelectorAll(".revision-item-card");
    allCards.forEach((c, i) => {
      c.classList.toggle("selected", i === idx);
    });

    if (btnRevAnterior) btnRevAnterior.disabled = (idx === 0);
    if (btnRevSiguiente) btnRevSiguiente.disabled = (idx === revisionItems.length - 1);

    const item = revisionItems[idx];
    cargarDetalleRevision(item.identificador);
  }

  async function cargarDetalleRevision(identificador) {
    const esEnsayo = (revFilterEntorno?.value === "ensayo");
    try {
      const resp = await fetch(`/api/revision/detalle/${identificador}?es_ensayo=${esEnsayo}`);
      if (!resp.ok) return;

      currentRevisionData = await resp.json();
      if (lblInspectorId) lblInspectorId.textContent = currentRevisionData.identificador;

      actualizarBadgesInspector(currentRevisionData);

      // Mostrar/ocultar banner de recalculada
      if (bannerRevRecalculada) {
        bannerRevRecalculada.style.display = currentRevisionData.es_recalculada ? "block" : "none";
      }

      inspectorCurrentBox = currentRevisionData.caja ? [...currentRevisionData.caja] : null;
      actualizarInputsCoordenadas();

      if (currentRevisionData.recorte_base64) {
        imgInspectorRecorte.src = "data:image/jpeg;base64," + currentRevisionData.recorte_base64;
      } else {
        imgInspectorRecorte.src = "";
      }

      inspectorRawImg = new Image();
      inspectorRawImg.onload = () => {
        dibujarCanvasInspector();
      };
      inspectorRawImg.src = "data:image/jpeg;base64," + currentRevisionData.imagen_base64;
    } catch (err) {
      console.error("Error al cargar detalle de revisión:", err);
    }
  }

  function actualizarBadgesInspector(data) {
    if (badgeInspectorDecision) {
      badgeInspectorDecision.className = `badge-decision-${data.decision_humana || "pendiente"}`;
      badgeInspectorDecision.textContent = (data.decision_humana || "pendiente").toUpperCase();
    }
    if (badgeInspectorDetector) {
      badgeInspectorDetector.textContent = data.estado_historico_detector || "UN_ROSTRO";
    }
    if (badgeInspectorOrigen) {
      badgeInspectorOrigen.textContent = data.origen_caja || (data.es_recalculada ? "recalculada" : "vivo");
      badgeInspectorOrigen.className = data.es_recalculada ? "badge badge-warning" : "badge badge-secondary";
    }
  }

  function actualizarInputsCoordenadas() {
    if (inspectorCurrentBox && inspectorCurrentBox.length === 4) {
      inpBoxX.value = inspectorCurrentBox[0];
      inpBoxY.value = inspectorCurrentBox[1];
      inpBoxW.value = inspectorCurrentBox[2];
      inpBoxH.value = inspectorCurrentBox[3];
    } else {
      inpBoxX.value = 0;
      inpBoxY.value = 0;
      inpBoxW.value = 0;
      inpBoxH.value = 0;
    }
  }

  function dibujarCanvasInspector() {
    if (!canvasInspector || !inspectorRawImg) return;
    canvasInspector.width = inspectorRawImg.naturalWidth;
    canvasInspector.height = inspectorRawImg.naturalHeight;
    const ctx = canvasInspector.getContext("2d");
    ctx.drawImage(inspectorRawImg, 0, 0);

    if (inspectorCurrentBox && inspectorCurrentBox.length === 4) {
      const [x, y, w, h] = inspectorCurrentBox;
      const dec = currentRevisionData?.decision_humana || "pendiente";

      let color = "#d29922";
      if (dec === "aprobada") color = "#2ea043";
      else if (dec === "corregida") color = "#388bfd";
      else if (dec === "excluida") color = "#da3633";

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = color + "26"; // 15% opacidad
      ctx.fillRect(x, y, w, h);

      ctx.fillStyle = color;
      ctx.font = "bold 14px sans-serif";
      const origen = currentRevisionData?.es_recalculada ? "recalculada" : (currentRevisionData?.origen_caja || "recuadro");
      ctx.fillText(`${dec.toUpperCase()} (${origen})`, x + 4, Math.max(18, y - 6));
    }
  }

  async function aplicarDecisionRevision(decision) {
    if (!currentRevisionData) return;
    const identificador = currentRevisionData.identificador;
    const esEnsayo = (revFilterEntorno?.value === "ensayo");
    const box = inspectorCurrentBox;

    try {
      const resp = await fetch("/api/revision/decidir", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          identificador: identificador,
          decision: decision,
          caja: box,
          es_ensayo: esEnsayo
        })
      });

      if (!resp.ok) {
        mostrarAlerta("Error al aplicar decisión de revisión.", "danger");
        return;
      }

      const res = await resp.json();
      currentRevisionData.decision_humana = decision;
      currentRevisionData.caja = res.caja;
      currentRevisionData.origen_caja = res.origen_caja;

      actualizarBadgesInspector(currentRevisionData);
      dibujarCanvasInspector();

      // Actualizar tarjeta lateral
      const card = document.getElementById(`rev-card-${currentRevisionIndex}`);
      if (card) {
        const badgeSpan = card.querySelector(`[class*="badge-decision-"]`);
        if (badgeSpan) {
          badgeSpan.className = `badge-decision-${decision}`;
          badgeSpan.textContent = decision.toUpperCase();
        }
      }

      mostrarAlerta(`Captura ${identificador} marcada como ${decision.toUpperCase()}.`, "success");

      // Avanzar automáticamente a la siguiente captura si hay más
      if (currentRevisionIndex < revisionItems.length - 1) {
        setTimeout(() => seleccionarCapturaPorIndice(currentRevisionIndex + 1), 250);
      }
    } catch (err) {
      mostrarAlerta("Error de red: " + err.message, "danger");
    }
  }

  function limpiarInspector() {
    if (lblInspectorId) lblInspectorId.textContent = "Ninguna captura seleccionada";
    if (imgInspectorRecorte) imgInspectorRecorte.src = "";
    if (canvasInspector) {
      const ctx = canvasInspector.getContext("2d");
      ctx.clearRect(0, 0, canvasInspector.width, canvasInspector.height);
    }
  }

  // =========================================================================
  // 7. UTILIDADES GRÁFICAS DEL CLIENTE
  // =========================================================================
  function getCanvasPos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((evt.clientX - rect.left) * scaleX),
      y: Math.round((evt.clientY - rect.top) * scaleY)
    };
  }

  function actualizarRecorteDinamicoCliente(sourceImg, box, targetImgElem) {
    if (!sourceImg || !box || box.length !== 4 || !targetImgElem) return;
    const [x, y, w, h] = box;
    if (w <= 0 || h <= 0) return;

    // Calcular recorte cuadrado 1:1 con margen
    const margen = 0.1;
    const dx = Math.round(w * margen);
    const dy = Math.round(h * margen);
    let x1 = x - dx, y1 = y - dy, x2 = x + w + dx, y2 = y + h + dy;
    let ancho = x2 - x1, alto = y2 - y1;
    if (ancho > alto) {
      const dif = ancho - alto;
      y1 -= Math.floor(dif / 2);
      y2 += Math.ceil(dif / 2);
    } else if (alto > ancho) {
      const dif = alto - ancho;
      x1 -= Math.floor(dif / 2);
      x2 += Math.ceil(dif / 2);
    }

    const side = Math.max(30, x2 - x1);
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 160;
    tempCanvas.height = 160;
    const tCtx = tempCanvas.getContext("2d");

    tCtx.fillStyle = "#000";
    tCtx.fillRect(0, 0, 160, 160);
    try {
      tCtx.drawImage(sourceImg, x1, y1, side, side, 0, 0, 160, 160);
      targetImgElem.src = tempCanvas.toDataURL("image/jpeg", 0.9);
    } catch (e) {
      // Ignorar si desborda límites del buffer del navegador
    }
  }

  // =========================================================================
  // 8. BUCLE DE VÍDEO Y WEBCAM
  // =========================================================================
  async function sondearCamaras() {
    try {
      const resp = await fetch("/api/camaras?max_indices=5");
      const data = await resp.json();
      selectCamara.innerHTML = "";
      if (!data.dispositivos || data.dispositivos.length === 0) {
        selectCamara.innerHTML = `<option value="-1">No se detectaron cámaras</option>`;
        return;
      }
      data.dispositivos.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.indice;
        opt.textContent = `Cámara ${d.indice}: ${d.resolucion} (${d.backend})`;
        if (d.disponible) opt.selected = true;
        selectCamara.appendChild(opt);
      });
    } catch (err) {
      selectCamara.innerHTML = `<option value="1">Cámara 1 (por defecto)</option>`;
    }
  }

  btnSondear?.addEventListener("click", sondearCamaras);

  btnIniciarCam?.addEventListener("click", async () => {
    const idx = parseInt(selectCamara.value, 10);
    try {
      mostrarOverlay("Conectando con DirectShow...");
      const resp = await fetch("/api/camara/iniciar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ indice: idx })
      });
      const data = await resp.json();
      if (resp.ok) {
        mostrarAlerta(`Cámara en índice ${idx} iniciada correctamente.`, "success");
      } else {
        mostrarAlerta(`Error al iniciar cámara: ${data.mensaje}`, "danger");
      }
    } catch (err) {
      mostrarAlerta(`Error de conexión: ${err.message}`, "danger");
    }
  });

  btnDetenerCam?.addEventListener("click", async () => {
    try {
      const resp = await fetch("/api/camara/detener", { method: "POST" });
      const data = await resp.json();
      mostrarAlerta(data.mensaje || "Cámara detenida.", "info");
    } catch (err) {
      mostrarAlerta(`Error al detener: ${err.message}`, "danger");
    }
  });

  function iniciarBucleVideo() {
    async function tick() {
      try {
        const resp = await fetch("/api/frame/latest");
        if (resp.ok) {
          const data = await resp.json();
          if (data.disponible) {
            cameraActive = true;
            ocultarOverlay();
            videoStream.src = "data:image/jpeg;base64," + data.image_base64;

            currentFrameId = data.id_fotograma;
            currentFrameEstado = data.estado;

            lblFrameId.textContent = "#" + data.id_fotograma;
            lblResolucion.textContent = `${data.ancho}x${data.alto}`;
            lblTotalRostros.textContent = data.total_rostros;

            clientFpsCount++;
            const tAhora = performance.now();
            if (tAhora - lastFpsCalcTime >= 1000) {
              clientFps = (clientFpsCount * 1000 / (tAhora - lastFpsCalcTime)).toFixed(1);
              clientFpsCount = 0;
              lastFpsCalcTime = tAhora;
            }

            actualizarBadges({
              activa: true,
              estado_deteccion: data.estado,
              fps_servidor: data.fps,
              fps_cliente: clientFps
            });

            actualizarSimulacionPuerta(data);
          } else {
            cameraActive = false;
            currentFrameEstado = "SIN_SEÑAL";
            mostrarOverlay(data.mensaje || "Esperando fotogramas de la cámara...");
            actualizarBadges({ activa: false, estado_deteccion: "SIN_SEÑAL" });
            actualizarSimulacionPuerta({ estado_puerta: "CERRADA", mensaje_acceso: "Cámara sin señal", persona_reconocida: "", confianza: 0 });
          }
        }
      } catch (err) {
        cameraActive = false;
        actualizarBadges({ activa: false, estado_deteccion: "ERROR_CONEXION" });
        actualizarSimulacionPuerta({ estado_puerta: "CERRADA", mensaje_acceso: "Error de conexión", persona_reconocida: "", confianza: 0 });
      }

      actualizarEstadoBotones();
      setTimeout(tick, 70);
    }
    tick();
  }

  function actualizarBadges({ activa, estado_deteccion, fps_servidor = 0, fps_cliente = 0 }) {
    if (activa) {
      badgeConexion.className = "badge badge-conectado";
      badgeConexion.textContent = "CONECTADO";
      badgeFps.textContent = `${fps_cliente} FPS (SRV ${fps_servidor})`;
    } else {
      badgeConexion.className = "badge badge-desconectado";
      badgeConexion.textContent = "DESCONECTADO";
      badgeFps.textContent = "0.0 FPS";
    }

    if (estado_deteccion === "UN_ROSTRO") {
      badgeDetector.className = "badge badge-un-rostro";
      badgeDetector.textContent = "UN_ROSTRO";
    } else if (estado_deteccion === "VARIOS_ROSTROS") {
      badgeDetector.className = "badge badge-varios-rostros";
      badgeDetector.textContent = "VARIOS_ROSTROS";
    } else {
      badgeDetector.className = "badge badge-sin-rostro";
      badgeDetector.textContent = estado_deteccion || "SIN_ROSTRO";
    }
  }

  function actualizarEstadoBotones() {
    if (!cameraActive || inFlightCapture) {
      btnCapturaValida.disabled = true;
      btnCapturaFallo.disabled = true;
      return;
    }
    btnCapturaValida.disabled = (currentFrameEstado !== "UN_ROSTRO");
    btnCapturaFallo.disabled = !(["SIN_ROSTRO", "VARIOS_ROSTROS"].includes(currentFrameEstado));
  }

  function mostrarOverlay(msg) {
    overlayText.textContent = msg;
    videoOverlay.classList.remove("hidden");
  }

  function ocultarOverlay() {
    videoOverlay.classList.add("hidden");
  }

  function actualizarSimulacionPuerta(data) {
    if (!badgePuerta) return;
    const puertaAbierta = (data.estado_puerta === "ABIERTA");

    if (puertaAbierta) {
      badgePuerta.className = "badge badge-door-open";
      badgePuerta.textContent = "PUERTA ABIERTA";
      doorGraphic.className = "door-graphic door-open";
      doorIcon.textContent = "🚪";
      doorStateText.textContent = "ABIERTA";
    } else {
      badgePuerta.className = "badge badge-door-closed";
      badgePuerta.textContent = "PUERTA CERRADA";
      doorGraphic.className = "door-graphic door-closed";
      doorIcon.textContent = "🔒";
      doorStateText.textContent = "CERRADA";
    }

    accessMsg.textContent = data.mensaje_acceso || "Puerta Cerrada";
    lblPersonaIdentificada.textContent = data.persona_reconocida || "Ninguno";
    const conf = data.confianza || 0;
    lblConfianzaReconocimiento.textContent = conf + "%";
    confidenceBar.style.width = Math.min(100, Math.max(0, conf)) + "%";
  }

  // Modelos y umbrales
  if (btnCambiarModelo) {
    btnCambiarModelo.addEventListener("click", async () => {
      const modeloElegido = selectModeloRecon.value;
      try {
        const resp = await fetch("/api/reconocimiento/configurar", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ modelo: modeloElegido })
        });
        const resJson = await resp.json();
        if (resp.ok) {
          if (sliderUmbral && resJson.umbral !== undefined) {
            sliderUmbral.value = resJson.umbral;
            lblUmbralValor.textContent = resJson.umbral + "%";
          }
          mostrarAlerta(`Modelo cambiado a ${resJson.modelo_activo || modeloElegido} (Umbral: ${resJson.umbral}%).`, "success");
        } else {
          mostrarAlerta("Error al cargar modelo: " + resJson.mensaje, "danger");
        }
      } catch (e) {
        mostrarAlerta("Error de red al cambiar modelo: " + e.message, "danger");
      }
    });
  }

  if (sliderUmbral) {
    sliderUmbral.addEventListener("input", () => {
      lblUmbralValor.textContent = sliderUmbral.value + "%";
    });
    sliderUmbral.addEventListener("change", async () => {
      const val = parseFloat(sliderUmbral.value) / 100.0;
      try {
        const resp = await fetch("/api/reconocimiento/configurar", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ umbral: val })
        });
        const resJson = await resp.json();
        if (resp.ok) {
          mostrarAlerta(`Umbral actualizado a ${resJson.umbral}%.`, "info");
        }
      } catch (e) {
        console.warn("Fallo al actualizar umbral:", e);
      }
    });
  }

  // Entorno (Ensayo vs Real)
  radioEnsayo?.addEventListener("change", () => {
    actualizarModoVisual();
    cargarProgresoSesion();
  });
  radioReal?.addEventListener("change", () => {
    actualizarModoVisual();
    cargarProgresoSesion();
  });

  function actualizarModoVisual() {
    const esEnsayo = radioEnsayo.checked;
    if (esEnsayo) {
      modeTag.textContent = "Ensayo (Piloto)";
      modeTag.className = "mode-tag ensayo";
      labelEnsayo.classList.add("active");
      labelReal.classList.remove("active");
    } else {
      modeTag.textContent = "Sesión Real";
      modeTag.className = "mode-tag real";
      labelReal.classList.add("active");
      labelEnsayo.classList.remove("active");
    }
  }

  selectPersona?.addEventListener("change", cargarProgresoSesion);
  selectSesion?.addEventListener("change", cargarProgresoSesion);

  if (poseButtonsContainer) {
    poseButtonsContainer.addEventListener("click", (e) => {
      const btn = e.target.closest(".btn-pose");
      if (!btn) return;
      poseButtonsContainer.querySelectorAll(".btn-pose").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentPose = btn.dataset.pose;
    });
  }

  async function cargarProgresoSesion() {
    const esEnsayo = radioEnsayo.checked;
    const persona = selectPersona.value;
    const sesion = selectSesion.value;

    lblPersonaActual.textContent = persona;
    lblSesionActual.textContent = sesion;

    try {
      const resp = await fetch(`/api/progreso?es_ensayo=${esEnsayo}&persona=${persona}&sesion=${sesion}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.progreso) {
        renderizarProgreso(data.progreso);
      }
    } catch (err) {
      console.warn("No se pudo cargar progreso:", err);
    }
  }

  function renderizarProgreso(progreso) {
    lblFallosSesion.textContent = progreso.total_fallos;
    posesProgressContainer.innerHTML = "";
    const posesOrdenadas = ["frontal", "izquierda", "derecha", "arriba", "abajo"];

    posesOrdenadas.forEach(p => {
      const info = progreso.poses[p] || { validas: 0, fallos: 0, meta: 4, completada: false };
      const meta = info.meta || info.objetivo || 4;
      const card = document.createElement("div");
      card.className = "pose-card" + (info.completada ? " completed" : "");
      const pct = Math.min(100, Math.round((info.validas / meta) * 100));

      card.innerHTML = `
        <div class="pose-card-header">
          <span class="pose-card-title">${p}</span>
          <span class="pose-card-count">${info.validas} / ${meta}</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${pct}%"></div>
        </div>
        <div class="pose-card-footer">
          <span>${info.completada ? "✓ Completa" : "Pendiente"}</span>
          <span>Fallos: ${info.fallos}</span>
        </div>
      `;
      posesProgressContainer.appendChild(card);
    });
  }

  function mostrarAlerta(msg, tipo = "info") {
    alertBanner.className = `alert-banner alert-${tipo}`;
    alertMsg.textContent = msg;
    alertBanner.classList.remove("hidden");
  }
});
