"""
Indicador de LEDs sobre Arduino Uno para el reconocimiento facial.

Cada persona autorizada enciende un LED distinto cuando la puerta se ABRE para ella:
    persona_1 -> D8,  persona_2 -> D9,  persona_3 -> D10.
Con la puerta cerrada, desconocido o sin rostro, todos los LED permanecen apagados.

Protocolo serie (9600 baudios, un byte por orden):
    b"0" apaga todos | b"1" LED persona_1 | b"2" LED persona_2 | b"3" LED persona_3

Seguridad ante fallos:
- Este módulo reenvía la orden vigente cada `reenvio_s` segundos (latido). El sketch apaga todos
  los LED si pasan más de 3 s sin recibir nada, de modo que un cierre brusco de la aplicación
  no deja un LED encendido.
- Ningún error de comunicación se propaga al bucle de cámara: ante un fallo, el indicador se
  desactiva con un único aviso.

Varios procesos (web local en 5000 y portal remoto en 5002) pueden usar el mismo Arduino: solo
uno puede abrir el puerto COM, así que ese proceso levanta un relevo UDP en 127.0.0.1 y los demás
le envían sus órdenes por ahí (ver ServidorRelevo / ConexionRelevo).
"""
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

BAUDIOS = 9600
PUERTO_RELEVO = 50123  # UDP, solo en 127.0.0.1
_ORDENES_VALIDAS = {b"0", b"1", b"2", b"3"}
VID_ARDUINO = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}  # Arduino, clones CH340, FTDI, CP210x

ORDEN_POR_PERSONA: Dict[str, bytes] = {
    "persona_1": b"1",
    "persona_2": b"2",
    "persona_3": b"3",
}
ORDEN_APAGAR = b"0"


class IndicadorLeds:
    """Enciende el LED de la persona autorizada reconocida y mantiene un latido hacia el Arduino."""

    def __init__(
        self,
        conexion: Any,
        mantener_s: float = 1.0,
        reenvio_s: float = 0.5,
        reloj: Callable[[], float] = time.monotonic,
    ):
        """
        conexion: objeto con write(bytes) y close() (por ejemplo serial.Serial).
        mantener_s: tiempo mínimo que el LED sigue encendido tras el último reconocimiento,
                    para evitar parpadeos cuando un solo fotograma falla.
        reenvio_s: intervalo del latido; la orden vigente se reenvía aunque no cambie.
        """
        self._con = conexion
        self._mantener_s = mantener_s
        self._reenvio_s = reenvio_s
        self._reloj = reloj
        self._lock = threading.Lock()
        self._activo = True
        self._orden_actual = ORDEN_APAGAR
        self._ultima_visto = 0.0
        self._ultimo_envio = 0.0
        self._aviso_emitido = False

    @property
    def activo(self) -> bool:
        return self._activo

    def actualizar(self, persona: Optional[str]) -> None:
        """
        Informa del resultado del fotograma actual.
        persona: identificador de la persona autorizada cuya puerta está ABIERTA, o None en
        cualquier otro caso (desconocido, puerta cerrada, sin rostro).
        """
        with self._lock:
            if not self._activo:
                return
            ahora = self._reloj()
            orden_persona = ORDEN_POR_PERSONA.get(str(persona).strip().lower()) if persona else None

            if orden_persona is not None:
                self._ultima_visto = ahora
                destino = orden_persona
            elif self._orden_actual != ORDEN_APAGAR and (ahora - self._ultima_visto) < self._mantener_s:
                destino = self._orden_actual  # mantiene el LED unos instantes
            else:
                destino = ORDEN_APAGAR

            cambio = destino != self._orden_actual
            if cambio or (ahora - self._ultimo_envio) >= self._reenvio_s:
                self._enviar(destino, ahora)

    def apagar(self) -> None:
        """Apaga todos los LED de inmediato."""
        with self._lock:
            if not self._activo:
                return
            self._ultima_visto = 0.0
            self._enviar(ORDEN_APAGAR, self._reloj())

    def cerrar(self) -> None:
        """Apaga los LED y libera el puerto serie."""
        self.apagar()
        with self._lock:
            self._activo = False
            try:
                self._con.close()
            except Exception:
                pass

    def _enviar(self, orden: bytes, ahora: float) -> None:
        try:
            self._con.write(orden)
            try:
                self._con.flush()
            except Exception:
                pass
            self._orden_actual = orden
            self._ultimo_envio = ahora
        except Exception as e:
            self._activo = False
            if not self._aviso_emitido:
                self._aviso_emitido = True
                print(f"[ARDUINO] Comunicación perdida ({e}). Indicador de LED desactivado.")


class ConexionSerieCompartida:
    """Puerto serie protegido con un lock, porque lo usan el indicador propio y el relevo UDP."""

    def __init__(self, serie: Any):
        self._serie = serie
        self._lock = threading.Lock()
        self.relevo: Optional["ServidorRelevo"] = None

    def write(self, datos: bytes) -> None:
        with self._lock:
            self._serie.write(datos)

    def flush(self) -> None:
        with self._lock:
            self._serie.flush()

    def close(self) -> None:
        if self.relevo is not None:
            self.relevo.detener()
        with self._lock:
            self._serie.close()


class ServidorRelevo:
    """
    Recibe órdenes por UDP (127.0.0.1) de otros procesos y las escribe en el Arduino.
    Responde b"A" al saludo b"?" para que los clientes sepan que hay alguien escuchando.
    """

    def __init__(self, destino: Any, puerto: int = PUERTO_RELEVO):
        self._destino = destino
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", puerto))
        self._sock.settimeout(0.5)
        self.puerto = self._sock.getsockname()[1]
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="RelevoArduino", daemon=True)
        self._hilo.start()

    def _bucle(self) -> None:
        while not self._parar.is_set():
            try:
                datos, origen = self._sock.recvfrom(16)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue  # Windows: el cliente UDP ya cerró su socket
            except OSError:
                break
            if datos == b"?":
                try:
                    self._sock.sendto(b"A", origen)
                except OSError:
                    pass
            elif datos in _ORDENES_VALIDAS:
                try:
                    self._destino.write(datos)
                except Exception:
                    pass

    def detener(self) -> None:
        self._parar.set()
        try:
            self._sock.close()
        except OSError:
            pass


class ConexionRelevo:
    """Conexión de un proceso que no puede abrir el COM: envía sus órdenes al proceso que sí lo abrió."""

    def __init__(self, puerto: int = PUERTO_RELEVO):
        self._destino = ("127.0.0.1", puerto)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)

    def verificar(self) -> bool:
        """True si hay un ServidorRelevo escuchando."""
        try:
            self._sock.sendto(b"?", self._destino)
            return self._sock.recv(16) == b"A"
        except OSError:
            return False

    def write(self, datos: bytes) -> None:
        self._sock.sendto(datos, self._destino)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def _detectar_puerto() -> Optional[str]:
    """Devuelve el puerto del Arduino: variable ARDUINO_PUERTO, o detección por fabricante, o único puerto serie."""
    forzado = os.environ.get("ARDUINO_PUERTO", "").strip()
    if forzado.lower() in {"no", "off", "none", "0"}:
        return None
    if forzado:
        return forzado
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    puertos = list(list_ports.comports())
    for p in puertos:
        if p.vid in VID_ARDUINO:
            return p.device
    if len(puertos) == 1:
        return puertos[0].device
    return None


def crear_indicador_auto(baudios: int = BAUDIOS) -> Optional[IndicadorLeds]:
    """
    Intenta conectar con el Arduino. Retorna None (sin lanzar excepción) si no hay pyserial,
    no hay placa conectada o el puerto está ocupado, para que el sistema siga funcionando.
    """
    try:
        import serial
    except ImportError:
        print("[ARDUINO] pyserial no está instalado (pip install pyserial). LED desactivados.")
        return None

    puerto = _detectar_puerto()
    if not puerto:
        print("[ARDUINO] No se detectó ningún Arduino. LED desactivados.")
        return None
    try:
        serie = serial.Serial(puerto, baudios, timeout=1, write_timeout=1)
    except Exception as e:
        # ¿Lo tiene abierto otro proceso de este proyecto (web 5000 o portal 5002)? Usar su relevo.
        relevo = ConexionRelevo()
        if relevo.verificar():
            print(f"[ARDUINO] {puerto} ya lo usa otro proceso del sistema: los LED se comparten por relevo local.")
            return IndicadorLeds(relevo)
        relevo.close()
        print(f"[ARDUINO] No se pudo abrir {puerto}: {e}. LED desactivados "
              "(¿está abierto el Monitor Serie del IDE de Arduino?).")
        return None

    time.sleep(2.0)  # el Uno se reinicia al abrir el puerto; esperar a que arranque el sketch
    conexion = ConexionSerieCompartida(serie)
    try:
        conexion.relevo = ServidorRelevo(conexion)  # permite que otros procesos usen el mismo Arduino
    except OSError:
        conexion.relevo = None
    indicador = IndicadorLeds(conexion)
    indicador.apagar()
    print(f"[ARDUINO] Conectado en {puerto}: persona_1=D8, persona_2=D9, persona_3=D10.")
    return indicador
