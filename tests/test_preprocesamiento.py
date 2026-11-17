"""
Pruebas unitarias para el módulo de preprocesamiento, detección, bordes y conectividad.
"""
import unittest
from pathlib import Path
import time
import cv2
import numpy as np
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion
)
from reconocimiento.preprocesamiento.filtros import (
    submuestrear,
    suavizado_gaussiano
)
from reconocimiento.preprocesamiento.mascara import (
    aislar_componente_mayor_8vecinos,
    generar_mascara_binaria,
    verificar_conectividad_8_vs_4
)


class TestPreprocesamiento(unittest.TestCase):

    def test_verificacion_conectividad_8_vs_4(self):
        """
        Comprueba que dos píxeles unidos diagonalmente pertenecen al mismo
        componente con 8-vecinos y a dos componentes distintos con 4-vecinos.
        """
        resultado = verificar_conectividad_8_vs_4()
        # En 4-vecinos: 2 componentes de primer plano
        self.assertEqual(resultado["conectividad_4"]["num_componentes_primer_plano"], 2)
        self.assertFalse(resultado["conectividad_4"]["comparten_componente"])
        # En 8-vecinos: 1 único componente de primer plano
        self.assertEqual(resultado["conectividad_8"]["num_componentes_primer_plano"], 1)
        self.assertTrue(resultado["conectividad_8"]["comparten_componente"])

    def test_manejo_seguro_bordes_detector(self):
        """Verifica que un recuadro que exceda bordes físicos no cause error y use padding seguro."""
        detector = DetectorFacial()
        img_sintetica = np.full((100, 100), 128, dtype=np.uint8)
        # Recuadro que empieza en coordenadas negativas y desborda
        recuadro_desbordante = (-20, -10, 50, 50)
        recorte = detector.recortar_con_margen_seguro(img_sintetica, recuadro_desbordante)
        self.assertIsNotNone(recorte)
        self.assertGreater(recorte.shape[0], 0)
        self.assertGreater(recorte.shape[1], 0)
        # Preserva relación de aspecto cuadrada (1:1)
        self.assertEqual(recorte.shape[0], recorte.shape[1])

    def test_deteccion_sin_rostro(self):
        """Verifica que una imagen uniforme devuelva estado explícito SIN_ROSTRO."""
        detector = DetectorFacial()
        img_plana = np.full((120, 120), 200, dtype=np.uint8)
        res = detector.detectar(img_plana)
        self.assertEqual(res.estado, EstadoDeteccion.SIN_ROSTRO)
        self.assertIsNone(res.recuadro)
        self.assertEqual(res.total_rostros_detectados, 0)

    def test_deteccion_varios_rostros_simulados(self):
        """Verifica que al simular dos recuadros se devuelva VARIOS_ROSTROS y total_rostros_detectados=2."""
        detector = DetectorFacial()
        lienzo = np.full((200, 300), 128, dtype=np.uint8)
        recuadros = [(20, 20, 50, 50), (100, 20, 50, 50)]
        res = detector.detectar(lienzo, recuadros_simulados=recuadros)
        self.assertEqual(res.estado, EstadoDeteccion.VARIOS_ROSTROS)
        self.assertEqual(res.total_rostros_detectados, 2)
        self.assertIsNone(res.recuadro)
        self.assertIsNone(res.rostro_recortado)

    def test_normalizacion_resolucion_e_intensidad(self):
        """Verifica redimensionamiento a 64x64 e intervalos de intensidad."""
        img = np.random.randint(0, 256, (120, 80), dtype=np.uint8)
        img[0, 0] = 0
        img[1, 1] = 255
        norm_res = normalizar_resolucion(img, (64, 64))
        self.assertEqual(norm_res.shape, (64, 64))

        norm_float = normalizar_intensidades(norm_res, "float32_0_1")
        self.assertGreaterEqual(norm_float.min(), 0.0)
        self.assertLessEqual(norm_float.max(), 1.0)
        self.assertEqual(norm_float.dtype, np.float32)

    def test_submuestreo_antialiasing(self):
        """Verifica que el submuestreo reduzca a la mitad exacta las dimensiones."""
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        sub_dir = submuestrear(img, factor=2, aplicar_antialiasing=False)
        sub_aa = submuestrear(img, factor=2, aplicar_antialiasing=True)
        self.assertEqual(sub_dir.shape, (32, 32))
        self.assertEqual(sub_aa.shape, (32, 32))

    def test_regresion_giro_izquierda_persona_1_sesion_02_f06(self):
        """
        Caso de regresión solicitado: persona_1_sesion_02_izquierda_f06.jpg
        Con el detector anterior devolvía SIN_ROSTRO.
        Con el clasificador de perfil reflejado ahora detecta UN_ROSTRO
        en el marco de coordenadas original.
        """
        detector = DetectorFacial()
        ruta = Path("data/propio/imagenes/persona_1/sesion_02/izquierda/persona_1_sesion_02_izquierda_f06.jpg")
        if not ruta.exists():
            self.skipTest(f"Imagen de regresión no encontrada en {ruta}")
        img = cv2.imread(str(ruta))
        res = detector.detectar(img)
        self.assertEqual(res.estado, EstadoDeteccion.UN_ROSTRO)
        self.assertEqual(res.total_rostros_detectados, 1)
        self.assertIsNotNone(res.recuadro)
        x, y, w, h = res.recuadro
        # El recuadro esperado es aproximadamente (222, 106, 280, 280)
        self.assertAlmostEqual(x, 222, delta=25)
        self.assertAlmostEqual(y, 106, delta=25)
        self.assertAlmostEqual(w, 280, delta=25)
        self.assertAlmostEqual(h, 280, delta=25)
        self.assertIsNotNone(res.rostro_recortado)
        self.assertEqual(res.rostro_recortado.shape[0], res.rostro_recortado.shape[1])

    def test_deteccion_ambos_giros_laterales(self):
        """Verifica detección de giro tanto hacia la izquierda como hacia la derecha."""
        detector = DetectorFacial()
        ruta_der = Path("data/propio/imagenes/persona_1/sesion_02/derecha/persona_1_sesion_02_derecha_f01.jpg")
        if not ruta_der.exists():
            ruta_der = Path("data/propio/imagenes/persona_1/sesion_01/derecha/persona_1_sesion_01_derecha_f01.jpg")
        if ruta_der.exists():
            img_der = cv2.imread(str(ruta_der))
            res_der = detector.detectar(img_der)
            self.assertEqual(res_der.estado, EstadoDeteccion.UN_ROSTRO)

    def test_fusion_recuadros_solapados_misma_cara(self):
        """Verifica que dos detecciones solapadas de una misma cara se combinen en un solo recuadro."""
        detector = DetectorFacial()
        # Dos recuadros con alto IoU de la misma cara
        b1 = (100, 100, 200, 200)
        b2 = (110, 105, 195, 205)
        fusiones = detector.fusionar_recuadros([b1, b2])
        self.assertEqual(len(fusiones), 1)
        # Coordenadas promediadas
        self.assertEqual(fusiones[0], (105, 102, 198, 202))

    def test_escena_dos_personas_frontal_y_girada_devuelve_varios_rostros(self):
        """
        Verifica que una escena con dos personas (una frontal y otra girada)
        mantenga sus recuadros separados y devuelva estrictamente VARIOS_ROSTROS.
        """
        detector = DetectorFacial()
        ruta_front = Path("data/propio/imagenes/persona_1/sesion_01/frontal/persona_1_sesion_01_frontal_f03.jpg")
        ruta_turned = Path("data/propio/imagenes/persona_1/sesion_02/izquierda/persona_1_sesion_02_izquierda_f06.jpg")
        if ruta_front.exists() and ruta_turned.exists():
            img_f = cv2.imread(str(ruta_front))
            img_t = cv2.imread(str(ruta_turned))
            # Combinar horizontalmente
            escena_doble = np.hstack([img_f, img_t])
            res = detector.detectar(escena_doble)
            self.assertEqual(res.estado, EstadoDeteccion.VARIOS_ROSTROS)
            self.assertGreaterEqual(res.total_rostros_detectados, 2)
            self.assertIsNone(res.recuadro)

    def test_escenas_vacias_sin_falsos_positivos(self):
        """Verifica que imágenes vacías o de ruido uniforme devuelvan SIN_ROSTRO."""
        detector = DetectorFacial()
        negra = np.zeros((480, 640, 3), dtype=np.uint8)
        res_negra = detector.detectar(negra)
        self.assertEqual(res_negra.estado, EstadoDeteccion.SIN_ROSTRO)
        self.assertEqual(res_negra.total_rostros_detectados, 0)

        blanca = np.full((480, 640, 3), 255, dtype=np.uint8)
        res_blanca = detector.detectar(blanca)
        self.assertEqual(res_blanca.estado, EstadoDeteccion.SIN_ROSTRO)
        self.assertEqual(res_blanca.total_rostros_detectados, 0)

    def test_rendimiento_fps_detector_multiview(self):
        """Mide la latencia de detección y asegura tasa interactiva adecuada."""
        detector = DetectorFacial()
        ruta = Path("data/propio/imagenes/persona_1/sesion_02/izquierda/persona_1_sesion_02_izquierda_f06.jpg")
        if not ruta.exists():
            self.skipTest("Imagen de benchmark no disponible")
        img = cv2.imread(str(ruta))
        # Warmup
        _ = detector.detectar(img)
        # Medir iteraciones
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            _ = detector.detectar(img)
        duracion_promedio = (time.perf_counter() - t0) / iters
        fps = 1.0 / duracion_promedio if duracion_promedio > 0 else 0
        self.assertLess(duracion_promedio, 0.12, f"Latencia excesiva: {duracion_promedio*1000:.2f} ms ({fps:.1f} FPS)")


if __name__ == "__main__":
    unittest.main()
