"""El flujo de desarrollo no debe consumir sesiones reservadas."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import entrenar_desarrollo as desarrollo


class TestReservaDesarrollo(unittest.TestCase):
    def test_filtra_sesion_cinco_antes_de_validar_o_crear_registros(self):
        filas = []
        for numero in range(1, 6):
            filas.append(dict(
                identificador=f"p1_s{numero}", ruta="no_abrir.jpg", procedencia="camara",
                persona="persona_1", tipo_sujeto="autorizado", sesion=f"sesion_{numero:02d}",
                pose="frontal", particion="entrenamiento" if numero < 4 else "validacion",
                consentimiento="True", hash_contenido="", anotaciones="",
            ))
        # Una fila de prueba deliberadamente inválida no pertenece al desarrollo.
        filas[-1]["particion"] = "no_validar_reserva"
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "manifest.csv"
            with ruta.open("w", newline="", encoding="utf-8") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=filas[0].keys())
                escritor.writeheader()
                escritor.writerows(filas)
            with patch.object(desarrollo, "MANIFIESTO", ruta):
                registros, digest = desarrollo.cargar_registros()
        self.assertEqual(len(registros), 4)
        self.assertNotIn("sesion_05", {r.sesion for r in registros})
        self.assertEqual(len(digest), 64)

    def test_error_detector_cuenta_en_denominador_extremo_a_extremo(self):
        muestras = [
            {"intento": True, "estado": "procesada", "sesion": "sesion_04", "correcta": True},
            {"intento": True, "estado": "sin_rostro", "sesion": "sesion_04", "correcta": None},
        ]
        resultado = desarrollo.metricas_validacion(muestras)
        self.assertEqual(resultado["exactitud_condicional_pct"], 100)
        self.assertEqual(resultado["exactitud_extremo_a_extremo_pct"], 50)
