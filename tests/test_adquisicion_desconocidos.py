"""
Pruebas unitarias para la adquisición, registro y aislamiento de voluntarios desconocidos.

Verifica:
1. Rechazo de registro de desconocidos con identificador de personas autorizadas.
2. Rechazo de registro de autorizados con identificador previamente usado como desconocido.
3. Rechazo estricto de mezclar una misma identidad externa entre diferentes grupos
   ('desconocidos_ajuste', 'desconocidos_calibracion', 'desconocidos_prueba').
4. Aislamiento por sujeto: fotos de un mismo voluntario no pueden cruzar de grupo.
5. Consentimiento informado obligatorio para desconocidos.
6. Validación de destinos experimentales y distancias físicas permitidas.
7. Método validar_aislamiento_desconocidos() en ManifiestoDataset.
8. Reporte en resumen_conteo() de grupos incompletos cuando faltan voluntarios.
"""
from pathlib import Path
import tempfile
import unittest

from reconocimiento.adquisicion.propio import (
    DESTINOS_DESCONOCIDOS_VALIDOS,
    DISTANCIAS_VALIDAS,
    PERSONAS_AUTORIZADAS_PROPIO,
    ManifiestoDataset,
    RegistroCaptura
)


class TestAdquisicionDesconocidos(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ruta_manifiesto = Path(self.temp_dir.name) / "manifiesto_test.csv"
        self.manifiesto = ManifiestoDataset(self.ruta_manifiesto)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_desconocido_no_puede_ser_persona_autorizada(self):
        """Impide registrar un desconocido usando el identificador de un autorizado."""
        for aut in PERSONAS_AUTORIZADAS_PROPIO:
            with self.assertRaises(ValueError) as ctx:
                r = RegistroCaptura(
                    identificador=f"{aut}_externo_01",
                    ruta=f"data/test/{aut}.jpg",
                    procedencia="camara_web",
                    persona=aut,
                    tipo_sujeto="desconocido",
                    sesion="sesion_01",
                    pose="frontal",
                    particion="entrenamiento",
                    consentimiento=True,
                    anotaciones="destino=desconocidos_ajuste"
                )
                r.validar()
            self.assertIn("Conflicto de rol", str(ctx.exception))

    def test_desconocido_requiere_consentimiento(self):
        """Verifica que no se puede registrar un desconocido sin consentimiento informado."""
        with self.assertRaises(ValueError) as ctx:
            r = RegistroCaptura(
                identificador="externo_001_f01",
                ruta="data/test/ext1.jpg",
                procedencia="camara_web",
                persona="externo_001",
                tipo_sujeto="desconocido",
                sesion="sesion_01",
                pose="frontal",
                particion="entrenamiento",
                consentimiento=False,
                anotaciones="destino=desconocidos_ajuste"
            )
            r.validar()
        self.assertIn("consentimiento", str(ctx.exception).lower())

    def test_destino_experimental_invalido(self):
        """Rechaza destinos experimentales fuera de los tres grupos definidos."""
        with self.assertRaises(ValueError) as ctx:
            r = RegistroCaptura(
                identificador="externo_001_f01",
                ruta="data/test/ext1.jpg",
                procedencia="camara_web",
                persona="externo_001",
                tipo_sujeto="desconocido",
                sesion="sesion_01",
                pose="frontal",
                particion="entrenamiento",
                consentimiento=True,
                anotaciones="destino=grupo_inventado"
            )
            r.validar()
        self.assertIn("Destino experimental 'grupo_inventado' inválido", str(ctx.exception))

    def test_distancia_invalida(self):
        """Rechaza distancias distintas a cercana, media y lejana."""
        with self.assertRaises(ValueError) as ctx:
            r = RegistroCaptura(
                identificador="externo_001_f01",
                ruta="data/test/ext1.jpg",
                procedencia="camara_web",
                persona="externo_001",
                tipo_sujeto="desconocido",
                sesion="sesion_01",
                pose="frontal",
                particion="entrenamiento",
                consentimiento=True,
                anotaciones="destino=desconocidos_ajuste;distancia=kilometros"
            )
            r.validar()
        self.assertIn("Distancia 'kilometros' inválida", str(ctx.exception))

    def test_mismo_voluntario_no_puede_aparecer_en_multiples_grupos(self):
        """
        Regla crítica: Una identidad externa NO puede aparecer en más de un grupo.
        No basta con separar fotos de la misma persona.
        """
        # Foto 1 de externo_001 en desconocidos_ajuste
        r1 = RegistroCaptura(
            identificador="externo_001_f01",
            ruta="data/test/ext1_f01.jpg",
            procedencia="camara_web",
            persona="externo_001",
            tipo_sujeto="desconocido",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True,
            anotaciones="destino=desconocidos_ajuste;distancia=media"
        )
        self.manifiesto.agregar_registro(r1)

        # Foto 2 de externo_001 intentando registrarse en desconocidos_calibracion
        r2 = RegistroCaptura(
            identificador="externo_001_f02",
            ruta="data/test/ext1_f02.jpg",
            procedencia="camara_web",
            persona="externo_001",
            tipo_sujeto="desconocido",
            sesion="sesion_01",
            pose="izquierda",
            particion="validacion",
            consentimiento=True,
            anotaciones="destino=desconocidos_calibracion;distancia=cercana"
        )
        with self.assertRaises(ValueError) as ctx:
            self.manifiesto.agregar_registro(r2)
        self.assertIn("Fuga de identidad externa", str(ctx.exception))
        self.assertIn("desconocidos_ajuste", str(ctx.exception))

    def test_voluntarios_distintos_en_grupos_distintos_permitidos(self):
        """Verifica que voluntarios disjuntos en grupos separados son aceptados."""
        r_ajuste = RegistroCaptura(
            identificador="ext_ajuste_01",
            ruta="data/test/ext_a.jpg",
            procedencia="camara_web",
            persona="externo_001",
            tipo_sujeto="desconocido",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True,
            anotaciones="destino=desconocidos_ajuste;distancia=media"
        )
        r_calib = RegistroCaptura(
            identificador="ext_calib_01",
            ruta="data/test/ext_c.jpg",
            procedencia="camara_web",
            persona="externo_002",
            tipo_sujeto="desconocido",
            sesion="sesion_01",
            pose="frontal",
            particion="validacion",
            consentimiento=True,
            anotaciones="destino=desconocidos_calibracion;distancia=cercana"
        )
        r_prueba = RegistroCaptura(
            identificador="ext_prueba_01",
            ruta="data/test/ext_p.jpg",
            procedencia="camara_web",
            persona="externo_003",
            tipo_sujeto="desconocido",
            sesion="sesion_01",
            pose="frontal",
            particion="prueba",
            consentimiento=True,
            anotaciones="destino=desconocidos_prueba;distancia=lejana"
        )
        self.manifiesto.agregar_registro(r_ajuste)
        self.manifiesto.agregar_registro(r_calib)
        self.manifiesto.agregar_registro(r_prueba)

        ok, errores = self.manifiesto.validar_aislamiento_desconocidos()
        self.assertTrue(ok)
        self.assertEqual(len(errores), 0)

        conteo = self.manifiesto.resumen_conteo()
        grupos = conteo["desconocidos_por_grupo"]
        self.assertEqual(grupos["desconocidos_ajuste"]["voluntarios"], 1)
        self.assertEqual(grupos["desconocidos_calibracion"]["voluntarios"], 1)
        self.assertEqual(grupos["desconocidos_prueba"]["voluntarios"], 1)

    def test_resumen_conteo_indica_grupos_incompletos(self):
        """Verifica que el resumen señale explícitamente qué grupos no tienen voluntarios registrados."""
        # Manifiesto sin voluntarios
        conteo = self.manifiesto.resumen_conteo()
        grupos = conteo["desconocidos_por_grupo"]
        for g in DESTINOS_DESCONOCIDOS_VALIDOS:
            self.assertFalse(grupos[g]["completo"])
            self.assertEqual(grupos[g]["voluntarios"], 0)


if __name__ == "__main__":
    unittest.main()
