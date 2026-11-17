"""
Pruebas unitarias para el módulo de adquisición y particionado.
"""
import unittest
from pathlib import Path
from reconocimiento.adquisicion.olivetti import cargar_olivetti
from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    generar_manifiesto_plantilla_300
)


class TestAdquisicion(unittest.TestCase):

    def test_carga_y_particion_olivetti_3_1_1(self):
        """Verifica que Olivetti cargue 400 muestras particionadas exactamente 60/20/20 (3/1/1)."""
        dataset = cargar_olivetti()
        self.assertEqual(dataset.n_clases, 40)
        self.assertEqual(len(dataset.y_train), 240)  # 40 * 6
        self.assertEqual(len(dataset.y_val), 80)     # 40 * 2
        self.assertEqual(len(dataset.y_test), 80)    # 40 * 2
        self.assertEqual(dataset.X_train.shape, (240, 64, 64))
        self.assertTrue(dataset.pre_recortada)

    def test_aislamiento_sesiones_propio(self):
        """Verifica que la plantilla de 300 fotos cumpla estrictamente el aislamiento de sesiones."""
        manifiesto = generar_manifiesto_plantilla_300()
        es_valido, errores = manifiesto.validar_aislamiento_sesiones()
        self.assertTrue(es_valido, f"Errores encontrados: {errores}")
        self.assertEqual(len(errores), 0)

    def test_deteccion_fuga_sesion(self):
        """Verifica que se rechace una sesión que cruza particiones."""
        manifiesto = ManifiestoDataset(Path("data/propio/test_fuga.csv"))
        manifiesto.registros.clear()

        r1 = RegistroCaptura(
            identificador="p1_s1_f01",
            ruta="img1.jpg",
            procedencia="camara",
            persona="p1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True
        )
        r2 = RegistroCaptura(
            identificador="p1_s1_f02",
            ruta="img2.jpg",
            procedencia="camara",
            persona="p1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="prueba",  # Misma persona y sesión en otra partición
            consentimiento=True
        )
        manifiesto.agregar_registro(r1)
        manifiesto.agregar_registro(r2)

        es_valido, errores = manifiesto.validar_aislamiento_sesiones()
        self.assertFalse(es_valido)
        self.assertEqual(len(errores), 1)
        self.assertIn("Fuga de sesión detectada", errores[0])

    def test_exigencia_consentimiento_explicito(self):
        """Verifica que el consentimiento sea False por defecto y se rechace el registro sin él."""
        r_sin = RegistroCaptura(
            identificador="p1_s1_f01",
            ruta="img1.jpg",
            procedencia="camara",
            persona="p1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento"
            # consentimiento omitido -> False por defecto
        )
        self.assertFalse(r_sin.consentimiento)
        with self.assertRaises(ValueError) as ctx:
            r_sin.validar()
        self.assertIn("consentimiento", str(ctx.exception).lower())

        # Con consentimiento explícito no debe fallar
        r_con = RegistroCaptura(
            identificador="p1_s1_f01",
            ruta="img1.jpg",
            procedencia="camara",
            persona="p1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True
        )
        r_con.validar()  # no lanza excepción

    def test_rechazo_archivo_duplicado_entre_particiones_renombrado(self):
        """
        Verifica que se rechace registrar el mismo archivo (mismo hash) entre distintas particiones
        incluso si se le asigna otro identificador, otra sesión u otro nombre de archivo.
        """
        manifiesto = ManifiestoDataset(Path("data/propio/test_dup.csv"))
        manifiesto.registros.clear()

        # Registro 1 en entrenamiento
        r1 = RegistroCaptura(
            identificador="voluntario_A_s01_f01",
            ruta="foto_original.jpg",
            procedencia="camara",
            persona="voluntario_A",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True,
            hash_contenido="abc123hashidentico"
        )
        manifiesto.agregar_registro(r1)

        # Registro 2: mismo hash pero con otro ID, otra persona, otra sesión y en partición de prueba
        r2_duplicado = RegistroCaptura(
            identificador="voluntario_B_s05_f99",
            ruta="foto_copiada_renombrada.jpg",
            procedencia="camara",
            persona="voluntario_B",
            tipo_sujeto="autorizado",
            sesion="sesion_05",
            pose="derecha",
            particion="prueba",  # Partición distinta -> fuga de datos
            consentimiento=True,
            hash_contenido="abc123hashidentico"
        )

        with self.assertRaises(ValueError) as ctx:
            manifiesto.agregar_registro(r2_duplicado)
        self.assertIn("Fuga de imagen detectada", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
