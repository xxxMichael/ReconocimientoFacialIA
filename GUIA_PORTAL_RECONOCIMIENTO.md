# Guía rápida — Portal independiente de reconocimiento remoto

Este portal permite que tus compañeros prueben los modelos congelados desde la cámara de su celular o laptop (`getUserMedia`) sin exponer el panel administrativo (`5000`), el portal de recopilación (`5001`) ni las fotografías del proyecto.

- **Modelos disponibles (umbral fijo, no editable):**
  - `CNN Estándar Portal Completo — 0.61` (principal por defecto, `configuracion_candidato_estandar_portal_061.json`)
  - `CNN Antialiasing — 0.52` (comparación, `configuracion_candidato_antialiasing_052.json`)
- **Privacidad:** las imágenes se procesan únicamente en memoria y **nunca** se guardan en disco. Solo se registra el resultado estructurado cuando el participante presiona **Registrar intento** (guardado en `resultados/portal_reconocimiento/intentos.jsonl`).

---

## 1. Iniciar y detener el portal en la laptop

Abre una terminal PowerShell en la raíz del proyecto (`C:\Dev\7-semestre\AI2\ReconocimientoFacialIA`) y ejecuta:

```powershell
python scripts\iniciar_portal_reconocimiento.py --puerto 5002
```

*(Alternativa equivalente por módulo CLI: `python -m reconocimiento portal-reconocimiento --port 5002`)*

- El servicio escucha únicamente en `http://127.0.0.1:5002/` usando **Waitress**.
- Para **detenerlo**, presiona `Ctrl + C` en esa terminal.

---

## 2. Publicar con Cloudflare Tunnel (HTTPS obligatorio para cámaras)

Los navegadores de celular y laptop exigen **HTTPS** para permitir `getUserMedia` en enlaces remotos. Abre una **segunda terminal** mientras el portal sigue corriendo y ejecuta:

```powershell
cloudflared tunnel --url http://127.0.0.1:5002
```

- Cloudflare mostrará una URL pública `https://<subdominio>.trycloudflare.com`. Comparte **esa URL** con tus compañeros.
- **Importante:** la laptop anfitriona debe permanecer **encendida, despierta (sin suspender) y conectada a internet** durante toda la prueba.
- Al terminar la sesión, presiona `Ctrl + C` en ambas terminales para cerrar el túnel y el servidor.

---

## 3. Prueba piloto recomendada (antes de compartir el enlace a todos)

1. Inicia el portal (`5002`) y el túnel `cloudflared`.
2. Abre la URL `https://...trycloudflare.com` en **dos dispositivos simultáneos** (por ejemplo, tu celular con datos móviles y otra laptop/navegador).
3. En el dispositivo 1, marca el consentimiento, selecciona **CNN Antialiasing — 0.52** y enciende la cámara.
4. En el dispositivo 2, selecciona **CNN Estándar Portal Completo — 0.61** y enciende la cámara.
5. Verifica en ambos que:
   - Se muestra la cámara del propio visitante (con vista previa espejo en cámara frontal y sin abrir la cámara de tu laptop).
   - Cambiar de candidato en un dispositivo no altera el candidato ni el umbral del otro.
   - Al tapar la cámara (`SIN_ROSTRO`), la puerta vuelve a `PUERTA CERRADA` al recibirse la siguiente detección (no de forma instantánea); al presionar **Detener Cámara**, se cierra de inmediato.
   - Al presionar **Registrar intento**, se guarda una sola línea en `resultados/portal_reconocimiento/intentos.jsonl` sin guardar ninguna fotografía.
   - Si falla o tarda la red al registrar, aparece **Reintentar intento** (mismo `id_intento`, sin duplicar aunque el servidor ya lo hubiera guardado) y **Descartar intento pendiente** (solo descarta localmente).
