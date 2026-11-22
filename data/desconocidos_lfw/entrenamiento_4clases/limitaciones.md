# Limitaciones del Conjunto de Desconocidos LFW (4 Clases)

1. **Discrepancia de Dominio (Domain Shift):**
   LFW consiste en fotografías de celebridades y figuras públicas en interiores/exteriores con iluminación    heterogénea, sensores profesionales y posprocesamiento fotográfico de prensa, mientras que el sistema de    acceso opera con sensor webcam CMOS de baja resolución y rango dinámico limitado.

2. **Prealineamiento funneled:**
   Las imágenes originales de LFW funneled tienen rotaciones y escalas corregidas algorítmicamente, reduciendo    la variabilidad angular respecto a las poses libres de un sujeto frente a la webcam.

3. **Representatividad Negativa Parcial:**
   Aunque LFW provee 300 identidades para inducir una frontera de decisión para la clase 'desconocido',    no representa adecuadamente negativos bajo el entorno físico del sistema (mismo fondo de oficina/habitación,    misma iluminación o distancia del sensor).

4. **Aislamiento Estricto:**
   Se excluyeron todas las 300 identidades usadas en calibración o reserva de prueba, garantizando que    ningún sujeto de evaluación haya sido observado en este entrenamiento.
