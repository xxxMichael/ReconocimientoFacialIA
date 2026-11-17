"""Detecta anotaciones sin resolver incluso en Python con evaluación diferida."""
import importlib
import inspect
import pkgutil
import unittest
from typing import get_type_hints

import reconocimiento


class CompatibilidadPythonTest(unittest.TestCase):
    def test_web_sin_camara(self):
        from reconocimiento.web.app import crear_app

        cliente = crear_app(indice_inicial=-1).test_client()
        for ruta in ("/", "/static/app.js", "/static/style.css"):
            with self.subTest(ruta=ruta), cliente.get(ruta) as respuesta:
                self.assertEqual(respuesta.status_code, 200)

    def test_anotaciones_de_los_modulos(self):
        for info in pkgutil.walk_packages(reconocimiento.__path__, "reconocimiento."):
            modulo = importlib.import_module(info.name)
            for nombre, objeto in vars(modulo).items():
                if not (inspect.isclass(objeto) or inspect.isfunction(objeto)):
                    continue
                if objeto.__module__ != modulo.__name__:
                    continue
                with self.subTest(modulo=info.name, nombre=nombre):
                    get_type_hints(objeto)
                    if inspect.isclass(objeto):
                        for _, metodo in inspect.getmembers(objeto, inspect.isfunction):
                            if metodo.__module__ == modulo.__name__:
                                get_type_hints(metodo)
