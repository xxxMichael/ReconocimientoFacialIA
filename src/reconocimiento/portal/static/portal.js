/**
 * Portal de Captura de Voluntarios Externos
 * Controla WebRTC getUserMedia, captura limpia sin recuadros,
 * gestión estricta de pistas de cámara, fallback nativo y matriz de progreso.
 */

(function () {
  'use strict';

  // Constantes de protocolo
  const POSES = ['frontal', 'izquierda', 'derecha', 'arriba', 'abajo'];
  const DISTANCIAS = ['cercana', 'media', 'lejana'];

  // Resolución resiliente del token (dataset body, query string o path)
  function resolverToken() {
    if (document.body && document.body.dataset && document.body.dataset.token) {
      const t = document.body.dataset.token.trim();
      if (t) return t;
    }
    const params = new URLSearchParams(window.location.search);
    if (params.get('token')) {
      return params.get('token').trim();
    }
    const partes = window.location.pathname.split('/');
    for (let i = 0; i < partes.length; i++) {
      if (partes[i] === 'participar' && partes[i + 1]) {
        return partes[i + 1].trim();
      }
    }
    return '';
  }

  // Estado de la aplicación
  const token = resolverToken();
  const identificador = (document.body && document.body.dataset && document.body.dataset.identificador) || '';

  let poseActual = 'frontal';
  let distanciaActual = 'cercana';
  let consentimientoActivo = false;
  let streamCamara = null;
  let facingModeActual = 'user'; // 'user' (frontal) o 'environment' (trasera)
  let blobCapturado = null;
  let poseCapturaPendiente = null;
  let distanciaCapturaPendiente = null;
  let datosProgreso = null;

  // Elementos DOM
  const seccionConsentimiento = document.getElementById('seccion-consentimiento');
  const chkConsentimiento = document.getElementById('chk-consentimiento');
  const btnConfirmarConsentimiento = document.getElementById('btn-confirmar-consentimiento');

  const seccionProtocolo = document.getElementById('seccion-protocolo');
  const seccionCamara = document.getElementById('seccion-camara');
  const seccionProgreso = document.getElementById('seccion-progreso');

  const selectorPoses = document.getElementById('selector-poses');
  const selectorDistancias = document.getElementById('selector-distancias');
  const textoEstadoCombinacion = document.getElementById('texto-estado-combinacion');

  const videoCamara = document.getElementById('video-camara');
  const canvasCaptura = document.getElementById('canvas-captura');
  const imgPrevisualizacion = document.getElementById('img-previsualizacion');
  const badgeCapturaFijada = document.getElementById('badge-captura-fijada');
  const overlayInactiva = document.getElementById('overlay-camara-inactiva');
  const indicadorCargando = document.getElementById('indicador-cargando');
  const textoCargando = document.getElementById('texto-cargando');

  const btnToggleCamara = document.getElementById('btn-toggle-camara');
  const btnCambiarCamara = document.getElementById('btn-cambiar-camara');
  const btnTomarFoto = document.getElementById('btn-tomar-foto');
  const btnRepetirFoto = document.getElementById('btn-repetir-foto');
  const btnEnviarFoto = document.getElementById('btn-enviar-foto');

  const inputArchivoNativo = document.getElementById('input-archivo-nativo');
  const btnActivarNativo = document.getElementById('btn-activar-nativo');

  const cajaNotificaciones = document.getElementById('caja-notificaciones');
  const bannerHttps = document.getElementById('banner-https');
  const bannerCompletado = document.getElementById('banner-completado');

  const lblEnviadas = document.getElementById('lbl-enviadas');
  const lblPendientes = document.getElementById('lbl-pendientes');
  const barraProgreso = document.getElementById('barra-progreso');
  const badgePctProgreso = document.getElementById('badge-pct-progreso');
  const txtContador = document.getElementById('txt-contador');
  const filasMatriz = document.getElementById('filas-matriz-progreso');

  // 1. Verificación inicial de HTTPS
  function verificarHttps() {
    const esLocal = window.location.hostname === 'localhost' ||
                    window.location.hostname === '127.0.0.1' ||
                    window.location.hostname.endsWith('.local');
    const esHttps = window.location.protocol === 'https:';

    if (!esHttps && !esLocal) {
      bannerHttps.classList.remove('oculto');
    }
  }

  // 2. Notificaciones al usuario
  function notificar(mensaje, tipo = 'info') {
    cajaNotificaciones.textContent = mensaje;
    cajaNotificaciones.className = `notificacion ${tipo}`;
    cajaNotificaciones.classList.remove('oculto');

    if (tipo !== 'error') {
      setTimeout(() => {
        cajaNotificaciones.classList.add('oculto');
      }, 5000);
    }
  }

  // 3. Consulta de Estado y Progreso desde el Servidor
  async function cargarEstadoServidor() {
    try {
      const resp = await fetch(`/api/estado?token=${encodeURIComponent(token)}`);
      const data = await resp.json();

      if (!data.exito) {
        notificar(data.error || 'No se pudo verificar el estado de la invitación.', 'error');
        return;
      }

      datosProgreso = data.progreso;
      actualizarInterfazProgreso();

      if (data.consentimiento_registrado) {
        consentimientoActivo = true;
        chkConsentimiento.checked = true;
        chkConsentimiento.disabled = true;
        actualizarEstiloCheckbox();
        btnConfirmarConsentimiento.textContent = '✓ Consentimiento Registrado';
        btnConfirmarConsentimiento.disabled = true;

        habilitarPasoProtocoloYCamara();
      }
    } catch (err) {
      notificar('Error de conexión con el servidor: ' + err.message, 'error');
    }
  }

  // 4. Habilitar secciones tras consentir
  function habilitarPasoProtocoloYCamara() {
    seccionProtocolo.classList.remove('bloque-deshabilitado');
    seccionCamara.classList.remove('bloque-deshabilitado');
  }

  // 5. Gestión del Consentimiento (Optimizado para Interacción Táctil en Celulares)
  const contenedorCheckbox = document.getElementById('contenedor-checkbox-consentimiento') || document.querySelector('.fila-checkbox');
  const btnActivarDesdeOverlay = document.getElementById('btn-activar-desde-overlay');

  if (btnActivarDesdeOverlay) {
    btnActivarDesdeOverlay.addEventListener('click', () => {
      iniciarCamara();
    });
  }

  function actualizarEstiloCheckbox() {
    if (chkConsentimiento && contenedorCheckbox) {
      if (chkConsentimiento.checked) {
        contenedorCheckbox.classList.add('seleccionada');
      } else {
        contenedorCheckbox.classList.remove('seleccionada');
      }
    }
  }

  if (chkConsentimiento) {
    chkConsentimiento.addEventListener('change', actualizarEstiloCheckbox);
    chkConsentimiento.addEventListener('input', actualizarEstiloCheckbox);
    chkConsentimiento.addEventListener('click', actualizarEstiloCheckbox);
  }

  // Permitir pulsar en cualquier parte de la fila para marcar/desmarcar
  if (contenedorCheckbox) {
    contenedorCheckbox.addEventListener('click', (e) => {
      if (e.target !== chkConsentimiento) {
        chkConsentimiento.checked = !chkConsentimiento.checked;
        actualizarEstiloCheckbox();
      }
    });
  }

  btnConfirmarConsentimiento.addEventListener('click', async () => {
    // En móviles, si el usuario pulsa directamente el botón de confirmar,
    // se asume su consentimiento explícito directo sin bloquear por foco
    if (!chkConsentimiento.checked) {
      chkConsentimiento.checked = true;
      actualizarEstiloCheckbox();
    }

    btnConfirmarConsentimiento.disabled = true;
    btnConfirmarConsentimiento.textContent = 'Registrando...';

    const tok = token || resolverToken();
    if (!tok) {
      btnConfirmarConsentimiento.disabled = false;
      btnConfirmarConsentimiento.textContent = 'Confirmar Consentimiento y Continuar';
      notificar('No se detectó el token de invitación en la página. Acceda mediante el enlace completo.', 'error');
      return;
    }

    try {
      const resp = await fetch('/api/consentimiento', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Invitation-Token': tok
        },
        body: JSON.stringify({ token: tok })
      });
      const data = await resp.json();

      if (data.exito) {
        consentimientoActivo = true;
        chkConsentimiento.disabled = true;
        btnConfirmarConsentimiento.textContent = '✓ Consentimiento Confirmado';
        btnConfirmarConsentimiento.disabled = true;
        habilitarPasoProtocoloYCamara();
        notificar('Consentimiento registrado con éxito.', 'exito');

        // Desplazamiento suave hacia la siguiente sección en pantallas móviles
        if (seccionProtocolo) {
          seccionProtocolo.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Intento de inicio automático con protección de excepciones (Safari iOS)
        try {
          iniciarCamara();
        } catch (e) {
          // El usuario podrá activar la cámara pulsando el botón manual
        }
      } else {
        btnConfirmarConsentimiento.disabled = false;
        btnConfirmarConsentimiento.textContent = 'Confirmar Consentimiento y Continuar';
        notificar(data.error || 'Error al registrar consentimiento.', 'error');
      }
    } catch (err) {
      btnConfirmarConsentimiento.disabled = false;
      btnConfirmarConsentimiento.textContent = 'Confirmar Consentimiento y Continuar';
      notificar('Error de red al registrar consentimiento.', 'error');
    }
  });

  // 6. Selección de Pose y Distancia
  selectorPoses.querySelectorAll('.btn-opcion').forEach(btn => {
    btn.addEventListener('click', () => {
      selectorPoses.querySelectorAll('.btn-opcion').forEach(b => b.classList.remove('activa'));
      btn.classList.add('activa');
      poseActual = btn.dataset.pose;
      actualizarEstadoCombinacion();
    });
  });

  selectorDistancias.querySelectorAll('.btn-opcion').forEach(btn => {
    btn.addEventListener('click', () => {
      selectorDistancias.querySelectorAll('.btn-opcion').forEach(b => b.classList.remove('activa'));
      btn.classList.add('activa');
      distanciaActual = btn.dataset.distancia;
      actualizarEstadoCombinacion();
    });
  });

  function actualizarInsigniaFijada() {
    if (blobCapturado && poseCapturaPendiente && distanciaCapturaPendiente && badgeCapturaFijada) {
      const pNom = poseCapturaPendiente.charAt(0).toUpperCase() + poseCapturaPendiente.slice(1);
      const dNom = distanciaCapturaPendiente.charAt(0).toUpperCase() + distanciaCapturaPendiente.slice(1);
      badgeCapturaFijada.textContent = `Fotografía fijada: ${pNom} - ${dNom}`;
      badgeCapturaFijada.classList.remove('oculto');
    } else if (badgeCapturaFijada) {
      badgeCapturaFijada.classList.add('oculto');
    }
  }

  function actualizarEstadoCombinacion() {
    const clave = `${poseActual}_${distanciaActual}`;
    const yaEnviada = datosProgreso && datosProgreso.combinaciones && datosProgreso.combinaciones[clave];

    const poseNombre = poseActual.charAt(0).toUpperCase() + poseActual.slice(1);
    const distNombre = distanciaActual.charAt(0).toUpperCase() + distanciaActual.slice(1);

    if (blobCapturado && poseCapturaPendiente && distanciaCapturaPendiente) {
      const pFijada = poseCapturaPendiente.charAt(0).toUpperCase() + poseCapturaPendiente.slice(1);
      const dFijada = distanciaCapturaPendiente.charAt(0).toUpperCase() + distanciaCapturaPendiente.slice(1);
      textoEstadoCombinacion.innerHTML = `⚠️ Captura pendiente fijada en: <strong>${pFijada} - ${dFijada}</strong> <span style="color:#60a5fa;">(metadatos preservados)</span>. Selector marca: ${poseNombre} - ${distNombre}.`;
    } else {
      if (yaEnviada) {
        textoEstadoCombinacion.innerHTML = `Combinación actual: <strong>${poseNombre} - ${distNombre}</strong> <span style="color:#10b981;">(✓ Ya enviada)</span>`;
      } else {
        textoEstadoCombinacion.innerHTML = `Combinación actual: <strong>${poseNombre} - ${distNombre}</strong> <span style="color:#f59e0b;">(Pendiente)</span>`;
      }
    }

    resaltarCeldaMatriz(poseActual, distanciaActual);
  }

  // 7. Control de la Cámara (WebRTC getUserMedia)
  async function iniciarCamara() {
    detenerCamara();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      notificar('Su navegador no soporta captura de cámara en vivo. Use la opción de cámara nativa abajo.', 'error');
      return;
    }

    indicadorCargando.classList.remove('oculto');
    textoCargando.textContent = 'Accediendo a cámara...';

    const restricciones = {
      audio: false,
      video: {
        facingMode: facingModeActual,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    try {
      streamCamara = await navigator.mediaDevices.getUserMedia(restricciones);
      videoCamara.srcObject = streamCamara;

      // Configurar efecto espejo solo en cámara frontal para naturalidad
      if (facingModeActual === 'user') {
        videoCamara.classList.add('espejo');
      } else {
        videoCamara.classList.remove('espejo');
      }

      await videoCamara.play();

      overlayInactiva.classList.add('oculto');
      indicadorCargando.classList.add('oculto');
      videoCamara.classList.remove('oculto');
      imgPrevisualizacion.classList.add('oculto');

      btnToggleCamara.textContent = '⏹ Detener Cámara';
      btnTomarFoto.disabled = false;
      btnTomarFoto.classList.remove('oculto');
      btnRepetirFoto.classList.add('oculto');
      btnEnviarFoto.classList.add('oculto');
    } catch (err) {
      indicadorCargando.classList.add('oculto');
      overlayInactiva.classList.remove('oculto');
      btnTomarFoto.disabled = true;

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        notificar('Permiso de cámara denegado. Permita el acceso en la barra del navegador o use la cámara nativa.', 'error');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        notificar('No se encontró ninguna cámara disponible en el dispositivo.', 'error');
      } else {
        notificar('No se pudo iniciar la cámara (' + err.message + '). Intente cambiar de cámara o usar el selector nativo.', 'error');
      }
    }
  }

  function detenerCamara() {
    if (streamCamara) {
      streamCamara.getTracks().forEach(track => track.stop());
      streamCamara = null;
    }
    if (videoCamara) {
      videoCamara.srcObject = null;
    }
    btnToggleCamara.textContent = '📹 Iniciar Cámara';
    btnTomarFoto.disabled = true;
    overlayInactiva.classList.remove('oculto');
  }

  btnToggleCamara.addEventListener('click', () => {
    if (streamCamara) {
      detenerCamara();
    } else {
      iniciarCamara();
    }
  });

  btnCambiarCamara.addEventListener('click', () => {
    facingModeActual = (facingModeActual === 'user') ? 'environment' : 'user';
    btnCambiarCamara.textContent = (facingModeActual === 'user') ? '🔄 Cámara Frontal' : '🔄 Cámara Trasera';
    iniciarCamara();
  });

  // Limpiar pistas activas al salir de la página
  window.addEventListener('beforeunload', () => {
    detenerCamara();
  });

  // 8. Captura de Fotograma en el Navegador
  btnTomarFoto.addEventListener('click', () => {
    if (!videoCamara || !videoCamara.videoWidth) {
      notificar('Espere a que el video esté transmitiendo para capturar.', 'error');
      return;
    }

    // ASOCIACIÓN ESTRICTA EN EL INSTANTE DE CAPTURA
    const poseFijada = poseActual;
    const distFijada = distanciaActual;

    const ancho = videoCamara.videoWidth;
    const alto = videoCamara.videoHeight;

    canvasCaptura.width = ancho;
    canvasCaptura.height = alto;
    const ctx = canvasCaptura.getContext('2d');

    // IMPORTANTE: Dibujar el fotograma crudo DIRECTAMENTE sin inversión especular.
    // Aunque el video se previsualice con CSS transform: scaleX(-1) para efecto espejo,
    // el canvas almacena la imagen óptica real sin textos ni recuadros.
    ctx.drawImage(videoCamara, 0, 0, ancho, alto);

    canvasCaptura.toBlob(blob => {
      if (!blob) {
        notificar('Error al procesar la imagen capturada.', 'error');
        return;
      }
      blobCapturado = blob;
      poseCapturaPendiente = poseFijada;
      distanciaCapturaPendiente = distFijada;

      // Mostrar previsualización congelada
      const urlPreview = URL.createObjectURL(blob);
      imgPrevisualizacion.src = urlPreview;
      imgPrevisualizacion.classList.remove('oculto');
      videoCamara.classList.add('oculto');

      actualizarInsigniaFijada();
      actualizarEstadoCombinacion();

      btnTomarFoto.classList.add('oculto');
      btnRepetirFoto.classList.remove('oculto');
      btnEnviarFoto.classList.remove('oculto');
      btnEnviarFoto.disabled = false;
    }, 'image/jpeg', 0.95);
  });

  // 9. Repetir captura
  btnRepetirFoto.addEventListener('click', () => {
    blobCapturado = null;
    poseCapturaPendiente = null;
    distanciaCapturaPendiente = null;
    actualizarInsigniaFijada();

    imgPrevisualizacion.classList.add('oculto');
    videoCamara.classList.remove('oculto');

    btnTomarFoto.classList.remove('oculto');
    btnRepetirFoto.classList.add('oculto');
    btnEnviarFoto.classList.add('oculto');

    actualizarEstadoCombinacion();

    if (!streamCamara) {
      iniciarCamara();
    }
  });

  // 10. Fallback con Selector Nativo de Smartphone
  btnActivarNativo.addEventListener('click', () => {
    inputArchivoNativo.click();
  });

  inputArchivoNativo.addEventListener('change', event => {
    const archivo = event.target.files && event.target.files[0];
    if (!archivo) return;

    if (!archivo.type.startsWith('image/')) {
      notificar('El archivo seleccionado debe ser una imagen.', 'error');
      return;
    }

    // ASOCIACIÓN ESTRICTA EN EL INSTANTE DE SELECCIÓN CON SELECTOR NATIVO
    const poseFijada = poseActual;
    const distFijada = distanciaActual;

    detenerCamara();

    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        canvasCaptura.width = img.naturalWidth;
        canvasCaptura.height = img.naturalHeight;
        const ctx = canvasCaptura.getContext('2d');
        ctx.drawImage(img, 0, 0);

        canvasCaptura.toBlob(blob => {
          blobCapturado = blob;
          poseCapturaPendiente = poseFijada;
          distanciaCapturaPendiente = distFijada;

          imgPrevisualizacion.src = e.target.result;
          imgPrevisualizacion.classList.remove('oculto');
          videoCamara.classList.add('oculto');
          overlayInactiva.classList.add('oculto');

          actualizarInsigniaFijada();
          actualizarEstadoCombinacion();

          btnTomarFoto.classList.add('oculto');
          btnRepetirFoto.classList.remove('oculto');
          btnEnviarFoto.classList.remove('oculto');
          btnEnviarFoto.disabled = false;
        }, 'image/jpeg', 0.95);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(archivo);
  });

  // 11. Envío Seguro al Servidor
  btnEnviarFoto.addEventListener('click', async () => {
    if (!blobCapturado) {
      notificar('No hay ninguna fotografía capturada para enviar.', 'error');
      return;
    }

    if (!consentimientoActivo) {
      notificar('Debe aceptar el consentimiento antes de enviar.', 'error');
      return;
    }

    // Utilizar estrictamente los metadatos asociados en el instante de captura
    const poseAEnviar = poseCapturaPendiente || poseActual;
    const distanciaAEnviar = distanciaCapturaPendiente || distanciaActual;

    btnEnviarFoto.disabled = true;
    btnRepetirFoto.disabled = true;
    indicadorCargando.classList.remove('oculto');
    textoCargando.textContent = `Enviando fotografía (${poseAEnviar} - ${distanciaAEnviar})...`;

    const formData = new FormData();
    formData.append('token', token);
    formData.append('pose', poseAEnviar);
    formData.append('distancia', distanciaAEnviar);
    formData.append('imagen', blobCapturado, `${poseAEnviar}_${distanciaAEnviar}.jpg`);

    try {
      const resp = await fetch('/api/subir', {
        method: 'POST',
        body: formData
      });

      const data = await resp.json();
      indicadorCargando.classList.add('oculto');
      btnRepetirFoto.disabled = false;

      if (data.exito) {
        notificar(`✓ Fotografía (${poseAEnviar} - ${distanciaAEnviar}) guardada exitosamente.`, 'exito');
        datosProgreso = data.progreso;
        actualizarInterfazProgreso();

        // Limpiar captura pendiente tras éxito
        blobCapturado = null;
        poseCapturaPendiente = null;
        distanciaCapturaPendiente = null;
        actualizarInsigniaFijada();

        imgPrevisualizacion.classList.add('oculto');
        videoCamara.classList.remove('oculto');
        btnTomarFoto.classList.remove('oculto');
        btnRepetirFoto.classList.add('oculto');
        btnEnviarFoto.classList.add('oculto');

        // Avanzar automáticamente a la siguiente combinación pendiente
        autoAvanzarSiguientePendiente();

        if (!streamCamara) {
          iniciarCamara();
        }
      } else {
        // En caso de fallo en el servidor, se conservan intactos los metadatos y la captura para reintento
        btnEnviarFoto.disabled = false;
        notificar(data.error || 'Error al guardar la fotografía.', 'error');
      }
    } catch (err) {
      // En caso de fallo de red, se conservan intactos los metadatos y la captura para reintento
      indicadorCargando.classList.add('oculto');
      btnEnviarFoto.disabled = false;
      btnRepetirFoto.disabled = false;
      notificar('Error de red al subir la imagen. Presione "Enviar Fotografía" para reintentar sin perder su captura.', 'error');
    }
  });

  // 12. Actualización de Progreso y Matriz
  function actualizarInterfazProgreso() {
    if (!datosProgreso) return;

    const enviadas = datosProgreso.total_enviadas || 0;
    const requeridas = datosProgreso.total_requeridas || 15;
    const pendientes = datosProgreso.total_pendientes || 0;
    const pct = Math.round((enviadas / requeridas) * 100);

    lblEnviadas.textContent = enviadas;
    lblPendientes.textContent = pendientes;
    txtContador.textContent = `${enviadas} / ${requeridas}`;
    barraProgreso.style.width = `${pct}%`;
    badgePctProgreso.textContent = `${pct}%`;

    renderizarMatriz();
    actualizarEstadoCombinacion();

    if (datosProgreso.completado) {
      bannerCompletado.classList.remove('oculto');
    }
  }

  function renderizarMatriz() {
    if (!filasMatriz) return;
    filasMatriz.innerHTML = '';

    POSES.forEach(p => {
      const tr = document.createElement('tr');
      const tdPose = document.createElement('td');
      tdPose.textContent = p.charAt(0).toUpperCase() + p.slice(1);
      tdPose.style.fontWeight = '600';
      tr.appendChild(tdPose);

      DISTANCIAS.forEach(d => {
        const td = document.createElement('td');
        const clave = `${p}_${d}`;
        const hecha = datosProgreso && datosProgreso.combinaciones && datosProgreso.combinaciones[clave];

        td.className = `celda-comb ${hecha ? 'completada' : 'pendiente'}`;
        td.dataset.pose = p;
        td.dataset.distancia = d;
        td.innerHTML = hecha ? '✓ Lista' : '○ Pend.';

        if (p === poseActual && d === distanciaActual) {
          td.classList.add('activa');
        }

        td.addEventListener('click', () => {
          seleccionarCombinacion(p, d);
        });

        tr.appendChild(td);
      });

      filasMatriz.appendChild(tr);
    });
  }

  function resaltarCeldaMatriz(p, d) {
    if (!filasMatriz) return;
    filasMatriz.querySelectorAll('.celda-comb').forEach(c => {
      if (c.dataset.pose === p && c.dataset.distancia === d) {
        c.classList.add('activa');
      } else {
        c.classList.remove('activa');
      }
    });
  }

  function seleccionarCombinacion(p, d) {
    poseActual = p;
    distanciaActual = d;

    selectorPoses.querySelectorAll('.btn-opcion').forEach(b => {
      b.classList.toggle('activa', b.dataset.pose === p);
    });
    selectorDistancias.querySelectorAll('.btn-opcion').forEach(b => {
      b.classList.toggle('activa', b.dataset.distancia === d);
    });

    actualizarEstadoCombinacion();
  }

  function autoAvanzarSiguientePendiente() {
    if (!datosProgreso || !datosProgreso.combinaciones) return;

    for (let p of POSES) {
      for (let d of DISTANCIAS) {
        if (!datosProgreso.combinaciones[`${p}_${d}`]) {
          seleccionarCombinacion(p, d);
          return;
        }
      }
    }
  }

  // Inicialización
  verificarHttps();
  cargarEstadoServidor();

})();
